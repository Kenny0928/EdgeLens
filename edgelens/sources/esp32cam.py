"""ESP32-CAM 來源：以 Arduino 官方 CameraWebServer 範例為標準。

- http://<ip>/status               JSON，目前所有設定
- http://<ip>/control?var=&val=    設定一項
- http://<ip>/capture              單張 JPEG
- http://<ip>:81/stream            MJPEG 串流（同時只能有一個連線）

串流連不上（例如被電腦瀏覽器占用）時自動退回定時抓 /capture，並每隔一段時間再試著接回串流。
參數面板依 /status 回報的內容自動產生：只列出 _KNOWN 表裡有、而且這台韌體真的有回報的項目。
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any

from edgelens.core.controls import ControlResult, ControlSpec, ControlType
from edgelens.i18n import tr
from edgelens.sources.base import SourceCapabilities, SourceError, SourceInfo, register_source
from edgelens.sources.mjpeg import HttpJpegSource, HttpStatusError, _Stopped, http_fetch

log = logging.getLogger(__name__)

DEFAULT_IP = "192.168.4.1"
DEFAULT_STREAM_PORT = 81

# esp32-camera 的 framesize_t：值 → (名稱, 寬, 高)。0–13 是 OV2640 以上都有的；
# 14 以後依感光元件（OV3660、OV5640），只在 /status 回報到時才列出。
FRAME_SIZES: dict[int, tuple[str, int, int]] = {
    0: ("96x96", 96, 96), 1: ("QQVGA", 160, 120), 2: ("QCIF", 176, 144), 3: ("HQVGA", 240, 176),
    4: ("240x240", 240, 240), 5: ("QVGA", 320, 240), 6: ("CIF", 400, 296), 7: ("HVGA", 480, 320),
    8: ("VGA", 640, 480), 9: ("SVGA", 800, 600), 10: ("XGA", 1024, 768), 11: ("HD", 1280, 720),
    12: ("SXGA", 1280, 1024), 13: ("UXGA", 1600, 1200), 14: ("FHD", 1920, 1080),
    15: ("P_HD", 720, 1280), 16: ("P_3MP", 864, 1536), 17: ("QXGA", 2048, 1536),
    18: ("QHD", 2560, 1440), 19: ("WQXGA", 2560, 1600), 20: ("P_FHD", 1080, 1920),
    21: ("QSXGA", 2560, 1920),
}


@dataclass(frozen=True)
class _Known:
    key: str
    kind: ControlType
    group: str
    lo: int = 0
    hi: int = 1
    advanced: bool = False
    label_key: str = ""       # 空的就用 control.esp32cam.<key>
    help: bool = False        # 有 control.esp32cam.<key>.help
    options: tuple[int, ...] = ()   # ENUM 的值；文字在 control.esp32cam.<key>.<值>


_B, _I, _E = ControlType.BOOL, ControlType.INT, ControlType.ENUM

# 已知鍵的表：順序就是面板順序。不在表裡的 /status 鍵一律忽略。
_KNOWN: tuple[_Known, ...] = (
    _Known("framesize", _E, "camera", help=True),
    _Known("quality", _I, "camera", 4, 63, help=True),
    _Known("hmirror", _B, "camera", label_key="control.hmirror"),
    _Known("vflip", _B, "camera", label_key="control.vflip"),
    _Known("led_intensity", _I, "camera", 0, 255, help=True),
    _Known("brightness", _I, "image", -2, 2, label_key="control.brightness"),
    _Known("contrast", _I, "image", -2, 2, label_key="control.contrast"),
    _Known("saturation", _I, "image", -2, 2, label_key="control.saturation"),
    _Known("sharpness", _I, "image", -2, 2),
    _Known("denoise", _I, "image", 0, 8, advanced=True),
    _Known("special_effect", _E, "image", options=(0, 1, 2, 3, 4, 5, 6)),
    _Known("awb", _B, "image", label_key="control.auto_white_balance"),
    _Known("wb_mode", _E, "image", options=(0, 1, 2, 3, 4)),
    _Known("aec", _B, "image", label_key="control.auto_exposure"),
    _Known("ae_level", _I, "image", -2, 2, help=True),
    _Known("aec_value", _I, "image", 0, 1200, label_key="control.exposure", help=True),
    _Known("agc", _B, "image", label_key="control.auto_gain"),
    _Known("agc_gain", _I, "image", 0, 30, label_key="control.gain"),
    _Known("gainceiling", _E, "image", options=(0, 1, 2, 3, 4, 5, 6), help=True),
    _Known("awb_gain", _B, "advanced", advanced=True),
    _Known("aec2", _B, "advanced", advanced=True, help=True),
    _Known("bpc", _B, "advanced", advanced=True, help=True),
    _Known("wpc", _B, "advanced", advanced=True, help=True),
    _Known("raw_gma", _B, "advanced", advanced=True, help=True),
    _Known("lenc", _B, "advanced", advanced=True, help=True),
    _Known("dcw", _B, "advanced", advanced=True, help=True),
    _Known("colorbar", _B, "advanced", advanced=True, help=True),
)
KNOWN_KEYS = tuple(k.key for k in _KNOWN)
_GAIN_LABELS = {0: "2x", 1: "4x", 2: "8x", 3: "16x", 4: "32x", 5: "64x", 6: "128x"}


def _as_int(v: Any) -> int | None:
    try:
        if isinstance(v, bool):
            return int(v)
        return int(float(v))
    except (TypeError, ValueError):
        return None


def build_specs(status: dict[str, Any]) -> list[ControlSpec]:
    """依 /status 的內容產生參數宣告。預設值取目前的設定，所以「設成預設值」不會改變相機。"""
    specs: list[ControlSpec] = []
    for k in _KNOWN:
        cur = _as_int(status.get(k.key)) if k.key in status else None
        if cur is None:
            continue
        label = k.label_key or f"control.esp32cam.{k.key}"
        help_key = f"control.esp32cam.{k.key}.help" if k.help else ""
        common: dict[str, Any] = dict(label_key=label, group=k.group, advanced=k.advanced, readback=True,
                                      help_key=help_key)
        if k.kind == ControlType.BOOL:
            specs.append(ControlSpec(k.key, ControlType.BOOL, default=bool(cur), **common))
        elif k.kind == ControlType.INT:
            specs.append(ControlSpec(k.key, ControlType.INT, default=cur, min=min(k.lo, cur), max=max(k.hi, cur),
                                     step=1, **common))
        elif k.key == "framesize":
            ids = [i for i in FRAME_SIZES if i <= 13 or cur >= 14]
            if cur not in ids:
                ids.append(cur)
            opts = tuple((i, _framesize_label(i)) for i in sorted(ids))
            specs.append(ControlSpec(k.key, ControlType.ENUM, default=cur, options=opts, **common))
        else:
            values = list(k.options)
            if cur not in values:
                values.append(cur)
            if k.key == "gainceiling":
                opts = tuple((v, _GAIN_LABELS.get(v, f"#{v}")) for v in values)
            else:
                opts = tuple((v, f"control.esp32cam.{k.key}.{v}" if v in k.options else f"#{v}") for v in values)
            specs.append(ControlSpec(k.key, ControlType.ENUM, default=cur, options=opts, **common))
    return specs


def _framesize_label(i: int) -> str:
    name, w, h = FRAME_SIZES.get(i, (f"#{i}", 0, 0))
    return f"{name} {w}×{h}" if w else name


def parse_ip(ip: str) -> tuple[str, int]:
    """'192.168.4.1' → ('192.168.4.1', 80)；'host:8080' → ('host', 8080)。也容忍 http:// 與結尾斜線。"""
    s = ip.strip()
    for prefix in ("http://", "https://"):
        if s.lower().startswith(prefix):
            s = s[len(prefix):]
    s = s.split("/", 1)[0]
    host, sep, port = s.rpartition(":")
    if sep and port.isdigit() and "]" not in port:
        return host, int(port)
    return s, 80


@register_source
class Esp32CamSource(HttpJpegSource):
    kind = "esp32cam"
    display_name_key = "source.esp32cam.name"
    description_key = "source.esp32cam.desc"
    discover_is_safe = True
    manual_add = True

    STREAM_RETRY_S = 10.0   # 退回 /capture 之後，每隔這麼久再試一次串流
    STATUS_TIMEOUT_S = 2.5

    @classmethod
    def discover(cls, timeout: float = 2.0) -> list[SourceInfo]:
        return []   # 官方範例不廣播 mDNS，只能手動輸入 IP

    @classmethod
    def param_specs(cls) -> list[ControlSpec]:
        return [
            ControlSpec("ip", ControlType.STR, "control.net.ip", default=DEFAULT_IP, live=False, group="source",
                        help_key="control.net.ip.help"),
            ControlSpec("stream_port", ControlType.INT, "control.net.stream_port", default=DEFAULT_STREAM_PORT,
                        min=1, max=65535, live=False, group="source", advanced=True),
        ]

    @classmethod
    def info_from_params(cls, params: dict[str, Any]) -> SourceInfo:
        ip = str(params.get("ip") or DEFAULT_IP).strip()
        host, port = parse_ip(ip)
        shown = host if port == 80 else f"{host}:{port}"
        clean = {"ip": shown, "stream_port": int(params.get("stream_port") or DEFAULT_STREAM_PORT)}
        for k, v in params.items():   # 保留測試用的時間參數
            if k not in clean and k in ("fail_after_s", "stall_s", "open_wait_s", "capture_interval_ms"):
                clean[k] = v
        return SourceInfo(kind=cls.kind, id=f"esp32cam:{shown}", name=f"ESP32-CAM {shown}", params=clean,
                          detail=f"http://{shown}/")

    @classmethod
    def contract_fixture(cls, tmp_path: Any) -> SourceInfo | None:
        from edgelens.sources._fake_http import shared_server
        srv = shared_server("esp32cam")
        return cls.info_from_params({"ip": f"127.0.0.1:{srv.port}", "stream_port": srv.stream_port})

    def __init__(self, info: SourceInfo):
        super().__init__(info)
        p = info.params
        self._host, self._port = parse_ip(str(p.get("ip") or DEFAULT_IP))
        self._stream_port = int(p.get("stream_port") or DEFAULT_STREAM_PORT)
        self._capture_interval = max(0.02, float(p.get("capture_interval_ms", 100)) / 1000.0)
        self._status: dict[str, Any] = {}
        self._specs: list[ControlSpec] = []
        self._mode_now = "stream"
        self._retry_stream_at = 0.0

    # ---- 網址 ----------------------------------------------------------------

    def _base(self) -> str:
        return f"http://{self._host}:{self._port}"

    def _stream_url(self) -> str:
        return f"http://{self._host}:{self._stream_port}/stream"

    def _target(self) -> str:
        return self._host if self._port == 80 else f"{self._host}:{self._port}"

    # ---- 開啟與狀態 ------------------------------------------------------------

    def _fetch_status(self) -> dict[str, Any]:
        _, body = http_fetch(self._base() + "/status", timeout=self.STATUS_TIMEOUT_S, max_bytes=1 << 20)
        data = json.loads(body.decode("utf-8", "replace"))
        if not isinstance(data, dict):
            raise ValueError("status is not an object")
        return data

    def _pre_open(self) -> None:
        target = self._target()
        try:
            status = self._fetch_status()
        except HttpStatusError as e:
            if e.status in (401, 403):
                raise SourceError(tr("source.net.err.auth", target=target), retryable=True,
                                  hint=tr("source.net.hint.auth")) from e
            raise SourceError(tr("source.esp32cam.err.not_camera", target=target), retryable=True,
                              hint=tr("source.esp32cam.hint.sketch")) from e
        except (ValueError, json.JSONDecodeError) as e:
            raise SourceError(tr("source.esp32cam.err.not_camera", target=target), retryable=True,
                              hint=tr("source.esp32cam.hint.sketch")) from e
        except OSError as e:
            raise SourceError(tr("source.esp32cam.err.unreachable", target=target, detail=str(e) or type(e).__name__),
                              retryable=True, hint=tr("source.net.hint.check")) from e
        except Exception as e:  # noqa: BLE001  http.client 的各種協定錯誤
            raise SourceError(tr("source.esp32cam.err.not_camera", target=target), retryable=True,
                              hint=tr("source.esp32cam.hint.sketch")) from e
        if "framesize" not in status and "quality" not in status:
            raise SourceError(tr("source.esp32cam.err.not_camera", target=target), retryable=True,
                              hint=tr("source.esp32cam.hint.sketch"))
        self._status = status
        self._specs = build_specs(status)
        self._mode_now = "stream"
        self._retry_stream_at = 0.0

    # ---- 背景連線：串流，失敗就退回 /capture -----------------------------------------

    def _session(self) -> bool | None:
        if self._mode_now == "capture" and time.monotonic() < self._retry_stream_at:
            self._capture_until(self._retry_stream_at)
            return False   # 預定的換手（再試一次串流），不算斷線
        try:
            self._mode_now = "stream"
            self._stream_loop(self._stream_url(), {})
        except _Stopped:
            raise
        except Exception as e:  # noqa: BLE001
            if self._got_frame:
                raise   # 串流本來是通的，現在斷了：交給重連機制
            log.info("ESP32-CAM 串流連不上（%r），改用 /capture", e)
            self._mode_now = "capture"
            self._retry_stream_at = time.monotonic() + self.STREAM_RETRY_S
            self._capture_until(self._retry_stream_at)
            return False
        return None

    def _capture_until(self, until: float) -> None:
        self._snapshot_loop(self._base() + "/capture", {}, self._capture_interval, until=until, label="capture")

    # ---- 參數 ----------------------------------------------------------------

    def controls(self) -> list[ControlSpec]:
        return list(self._specs) if self._open else []

    def get_control(self, key: str) -> Any:
        spec = next((s for s in self._specs if s.key == key), None)
        if spec is None or key not in self._status:
            raise KeyError(key)
        v = _as_int(self._status[key])
        return bool(v) if spec.type == ControlType.BOOL else v

    def set_control(self, key: str, value: Any) -> ControlResult:
        spec = next((s for s in self._specs if s.key == key), None)
        if spec is None:
            return ControlResult.failed(tr("source.esp32cam.ctl.unsupported", key=key))
        try:
            want = spec.validate(value)
        except ValueError as e:
            return ControlResult.failed(str(e), value)
        sent = int(want)
        url = f"{self._base()}/control?{urllib.parse.urlencode({'var': key, 'val': sent})}"
        try:
            http_fetch(url, timeout=self.STATUS_TIMEOUT_S, max_bytes=1 << 16)
        except HttpStatusError as e:
            return ControlResult.failed(tr("source.esp32cam.ctl.rejected", key=key, value=sent, code=e.status), want)
        except Exception as e:  # noqa: BLE001
            return ControlResult.failed(tr("source.esp32cam.ctl.no_connection", detail=str(e) or type(e).__name__),
                                        want)
        try:
            self._status = self._fetch_status()
        except Exception:  # noqa: BLE001  設定已送出，只是讀不回來
            self._status[key] = sent
            return ControlResult(ok=True, value=want, verified=None,
                                 message=tr("source.esp32cam.ctl.unverified"))
        got = _as_int(self._status.get(key))
        if got is None:
            return ControlResult(ok=True, value=want, verified=None)
        shown = bool(got) if spec.type == ControlType.BOOL else got
        if got == sent:
            return ControlResult(ok=True, value=shown, verified=True)
        return ControlResult(ok=True, value=shown, verified=False,
                             message=tr("source.esp32cam.ctl.mismatch", key=key, want=sent, got=got))

    def capabilities(self) -> SourceCapabilities:
        fs = next((s for s in self._specs if s.key == "framesize"), None)
        resolutions = []
        if fs is not None:
            resolutions = [(FRAME_SIZES[v][1], FRAME_SIZES[v][2]) for v in fs.option_values if v in FRAME_SIZES]
        return SourceCapabilities(provides_jpeg=True, live_controls=True, resolutions=resolutions, max_fps=25.0,
                                  extra={"mode": self._mode_now, "stream_port": self._stream_port})


@register_source
class Esp32CamDemoSource(Esp32CamSource):
    """模擬的 ESP32-CAM：在這台電腦上啟動一個假的 CameraWebServer（`_fake_http`），
    沒有硬體也能練習 ESP32-CAM 的參數面板、改解析度、串流被占用時改抓單張等行為。
    開兩台模擬相機時，第二台會因為「同時只能一個串流」而改抓單張，正好可以示範。"""

    kind = "esp32cam_demo"
    display_name_key = "source.esp32cam_demo.name"
    description_key = "source.esp32cam_demo.desc"
    discover_is_safe = True
    manual_add = False

    @classmethod
    def discover(cls, timeout: float = 2.0) -> list[SourceInfo]:
        # 兩個項目連到同一台模擬板子：都開起來時，第二個會因為「同時只能一個串流」改抓單張
        return [SourceInfo(kind=cls.kind, id=f"esp32cam_demo:{n}", name=f"{tr('source.esp32cam_demo.name')} #{n}",
                           params={"n": n}, detail="127.0.0.1") for n in (1, 2)]

    @classmethod
    def param_specs(cls) -> list[ControlSpec]:
        return []

    @classmethod
    def info_from_params(cls, params: dict[str, Any]) -> SourceInfo:
        return cls.discover()[0]

    @classmethod
    def contract_fixture(cls, tmp_path: Any) -> SourceInfo | None:
        return cls.discover()[0]

    def open(self) -> None:
        if self.is_open:
            return
        from edgelens.sources._fake_http import shared_server

        srv = shared_server("esp32cam")   # 埠號每次啟動都不同，所以在開啟時才決定
        self._host, self._port, self._stream_port = "127.0.0.1", srv.port, srv.stream_port
        super().open()
