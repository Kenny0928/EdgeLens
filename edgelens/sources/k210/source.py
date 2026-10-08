"""K210Source：Maixduino／Maix Bit（MaixPy v1）經 USB 序列埠串流。

流程：開序列埠 → 進入 raw REPL → 送出 `P = {...}` 加上 board/stream_v1.py → 等板子印出
STREAM_START → 切到 1.5 Mbps 開始收封包。關閉時照舊程式實測調出來的時序送 Ctrl+C、
切回 115200、退出 raw REPL，連續開關不需要重插 USB。

單一執行緒使用（見 FrameSource）：read、set_control、run_action 都在同一條執行緒，
所以不需要鎖。等待 ack 或重新啟動串流時收到的畫面只保留最新一張，不會遺失。
"""

from __future__ import annotations

import importlib.resources
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from edgelens.core.controls import ControlResult, ControlSpec, ControlType
from edgelens.core.frame import Frame
from edgelens.core.modelcard import ModelCard
from edgelens.core.results import Classification, Detection, InferenceResult
from edgelens.i18n import tr
from edgelens.sources.base import FrameSource, SourceCapabilities, SourceError, SourceInfo, register_source
from edgelens.sources.k210 import flash as flash_mod
from edgelens.sources.k210 import ports as ports_mod
from edgelens.sources.k210.protocol import MAGIC, Packet, PacketParser, TextChunk
from edgelens.sources.k210.repl import REPL_BAUD, RawRepl, ReplCancelled, ReplError

log = logging.getLogger(__name__)

STREAM_BAUD = 1500000   # 實測 1.5 Mbps 穩定，2 Mbps 收不到資料
START_TIMEOUT = 15      # 等 STREAM_START（含載入模型）的秒數
ACK_TIMEOUT = 2.0       # 等板子確認即時指令的秒數
STALL_SECONDS = 8.0     # 這麼久完全沒收到資料就當作連線中斷
SIZES = {"QQVGA": (160, 120), "QVGA": (320, 240), "VGA": (640, 480)}
CAMERA_KEYS = ("brightness", "contrast", "saturation", "auto_gain", "gain_db", "auto_exposure",
               "exposure_us", "auto_whitebal")
# 板子 ack 用的項目名稱和參數 key 不完全一樣（增益、曝光和自動模式綁在一起）
ACK_NAMES = {"gain_db": {"gain_db", "auto_gain"}, "auto_gain": {"gain_db", "auto_gain"},
             "exposure_us": {"exposure_us", "auto_exposure"}, "auto_exposure": {"exposure_us", "auto_exposure"}}
LIVE_KEYS = CAMERA_KEYS + ("vflip", "hmirror", "threshold", "nms")

_script_cache: str | None = None


def open_serial(port: str, baud: int) -> Any:
    """開啟序列埠。獨立成函式，測試可以換成假序列埠；"sim:" 開頭用模擬板子。"""
    if port.startswith("sim:"):
        from edgelens.sources.k210.sim import FakeK210Board

        board = FakeK210Board()
        board.time_scale = 0.02  # type: ignore[attr-defined]  # 模擬板子不必真的等那麼久
        return board
    import serial  # 延遲匯入：選配套件

    kw: dict[str, Any] = {}
    if os.name == "posix":
        kw["exclusive"] = True  # 被別的程式開著時要明確失敗，而不是兩邊搶資料
    return serial.Serial(port, baud, timeout=0.1, **kw)


def board_script() -> str:
    """讀出板上程式，並去掉整行註解與空行（少傳資料、少占板子的編譯記憶體）。"""
    global _script_cache
    if _script_cache is None:
        text = importlib.resources.files("edgelens.sources.k210").joinpath("board", "stream_v1.py").read_text(
            encoding="utf-8")
        _script_cache = "\n".join(ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")) + "\n"
    return _script_cache


def _specs() -> list[ControlSpec]:
    S, T = ControlSpec, ControlType
    return [
        S("brightness", T.INT, "control.brightness", default=0, min=-2, max=2, readback=True, group="image"),
        S("contrast", T.INT, "control.contrast", default=0, min=-2, max=2, readback=True, group="image"),
        S("saturation", T.INT, "control.saturation", default=0, min=-2, max=2, readback=True, group="image"),
        S("auto_gain", T.BOOL, "control.auto_gain", default=True, readback=True, group="camera"),
        S("gain_db", T.FLOAT, "control.gain", default=10.0, min=0, max=48, step=1, unit="dB", readback=True,
          group="camera"),
        S("auto_exposure", T.BOOL, "control.auto_exposure", default=True, readback=True, group="camera"),
        S("exposure_us", T.INT, "control.exposure", default=10000, min=1000, max=100000, step=1000, unit="µs",
          readback=True, group="camera"),
        S("auto_whitebal", T.BOOL, "control.auto_white_balance", default=True, readback=True, group="camera"),
        S("vflip", T.BOOL, "control.vflip", default=False, readback=True, group="camera"),
        S("hmirror", T.BOOL, "control.hmirror", default=False, readback=True, group="camera"),
        S("threshold", T.FLOAT, "control.k210.threshold", default=0.5, min=0.1, max=0.95, step=0.05, readback=True,
          group="inference"),
        S("nms", T.FLOAT, "control.k210.nms", default=0.3, min=0.1, max=0.9, step=0.05, readback=True,
          group="inference"),
        S("framesize", T.ENUM, "control.k210.framesize", default="QVGA", live=False, group="camera",
          options=tuple((k, f"{w}×{h}") for k, (w, h) in SIZES.items())),
        S("pixformat", T.ENUM, "control.k210.pixformat", default="rgb", live=False, group="camera",
          options=(("rgb", "control.k210.pixformat.rgb"), ("gray", "control.k210.pixformat.gray"))),
        S("send_every", T.INT, "control.k210.send_every", default=1, min=0, max=30, live=False, group="advanced",
          advanced=True),
        S("jpeg_quality", T.INT, "control.jpeg_quality", default=70, min=30, max=95, live=False, group="advanced",
          advanced=True),
        S("load_model", T.ACTION, "control.k210.load_model", group="action", extra={"inputs": ["card"]}),
        S("unload_model", T.ACTION, "control.k210.unload_model", group="action"),
        S("upload_file", T.ACTION, "control.k210.upload_file", group="action",
          extra={"inputs": ["local_path", "remote_path"]}),
        S("set_cpu_freq", T.ACTION, "control.k210.set_cpu_freq", group="action", advanced=True,
          extra={"inputs": ["cpu_mhz", "kpu_mhz"]}),
        S("flash_firmware", T.ACTION, "control.k210.flash_firmware", group="action", advanced=True,
          extra={"inputs": ["firmware_path"]}),
    ]


_SPECS = _specs()
_SPEC_BY_KEY = {s.key: s for s in _SPECS}
_RESTART_CONFIG = {"framesize": "framesize", "send_every": "send_every", "jpeg_quality": "quality"}


def _explain_open_error(exc: BaseException, port: str) -> SourceError:
    """把開序列埠的例外轉成學生看得懂的說明。"""
    msg = str(exc).lower()
    errno = getattr(exc, "errno", None)
    if isinstance(exc, FileNotFoundError) or errno == 2 or "no such file" in msg or "could not find" in msg:
        return SourceError(tr("source.k210.err.port_missing", port=port), retryable=True,
                           hint=tr("source.k210.hint.check_cable"))
    busy = errno in (11, 16) or "busy" in msg or "temporarily unavailable" in msg or "in use" in msg
    denied = errno == 13 or "permission" in msg or "access is denied" in msg
    if denied and sys.platform.startswith("linux"):
        return SourceError(tr("source.k210.err.port_denied", port=port), hint=tr("source.k210.hint.linux_dialout"))
    if busy or denied:  # Windows 上被占用常常顯示「存取被拒」
        return SourceError(tr("source.k210.err.port_busy", port=port), retryable=True,
                           hint=tr("source.k210.hint.close_other_app"))
    return SourceError(tr("source.k210.err.port_open", port=port, error=str(exc)), retryable=True,
                       hint=tr("source.k210.hint.check_cable"))


def _explain_board_error(text: str) -> str:
    """板子回傳的錯誤文字 → 給學生看的說明（保留原文最後幾行方便老師查）。"""
    text = text.strip()
    tail = "\n".join(text.splitlines()[-4:])
    low = text.lower()
    if "找不到模型" in text or "enoent" in low or "errno 2" in low:
        return tr("source.k210.err.model_missing", text=tail)
    if "memoryerror" in low or "memory" in low and "alloc" in low:
        return tr("source.k210.err.out_of_memory", text=tail)
    if "沒有 kpu" in text:
        return tr("source.k210.err.no_kpu")
    return tr("source.k210.err.start_failed", text=tail or "(none)")


@register_source
class K210Source(FrameSource):
    kind = "k210"
    display_name_key = "source.k210.name"
    description_key = "source.k210.desc"
    requires = ("serial",)
    discover_is_safe = True
    manual_add = True

    def __init__(self, info: SourceInfo):
        super().__init__(info)
        self._ser: Any = None
        self._repl: RawRepl | None = None
        self._open = False
        self._streaming = False
        self._port = ""
        self._started_at = 0.0
        self._parser = PacketParser()
        self._values: dict[str, Any] = {s.key: s.default_value() for s in _SPECS if s.type != ControlType.ACTION}
        for k, v in info.params.items():
            spec = _SPEC_BY_KEY.get(k)
            if spec is not None and spec.type != ControlType.ACTION:
                try:
                    self._values[k] = spec.validate(v)
                except ValueError:
                    pass
        self._user_camera: dict[str, Any] = {}   # 使用者改過的攝影機項目，重新啟動串流時帶回去
        self._card: ModelCard | None = None
        self._latest: Packet | None = None
        self._dropped = 0
        self._decode_failed = 0
        self._cmd_id = 0
        self._last_ack: dict[str, Any] = {"id": 0, "ok": [], "err": []}
        self._last_cfg: dict[str, Any] = {}
        self._last_rx = 0.0
        self._board_error = ""
        self.board_info: dict[str, Any] = {}
        self.last_text = ""
        self.last_meta: dict[str, Any] = {}
        self.last_result: InferenceResult | None = None

    # ---- 類別層級 --------------------------------------------------------------

    @classmethod
    def discover(cls, timeout: float = 2.0) -> list[SourceInfo]:
        try:
            found = ports_mod.likely_ports()
        except Exception as e:  # noqa: BLE001 - 列舉失敗就當作沒找到
            log.warning("列舉序列埠失敗：%s", e)
            return []
        return [SourceInfo(kind=cls.kind, id=p.stable_id, name=f"K210 ({p.chip or p.device})",
                           params={"port": p.device, "baud": REPL_BAUD}, detail=p.label) for p in found]

    @classmethod
    def param_specs(cls) -> list[ControlSpec]:
        try:
            choices = [(c.device, c.label, c.likely) for c in ports_mod.list_serial_ports()]
        except Exception:  # noqa: BLE001
            choices = []
        return [
            ControlSpec("port", ControlType.STR, "source.k210.param.port", default="", group="source",
                        extra={"choices": choices}),
            ControlSpec("baud", ControlType.INT, "source.k210.param.baud", default=REPL_BAUD, min=9600,
                        max=3000000, group="source", advanced=True),
        ]

    @classmethod
    def info_from_params(cls, params: dict[str, Any]) -> SourceInfo:
        port = str(params.get("port") or "").strip()
        baud = int(params.get("baud") or REPL_BAUD)
        pid, name = "k210:auto", "K210"
        if port:
            pid, name = f"k210:manual:{port}", f"K210 ({port})"
            try:
                for c in ports_mod.list_serial_ports():
                    if c.device == port:
                        pid = c.stable_id
                        break
            except Exception:  # noqa: BLE001
                pass
        return SourceInfo(kind=cls.kind, id=pid, name=name, params={"port": port, "baud": baud}, detail=port)

    @classmethod
    def contract_fixture(cls, tmp_path: Path) -> SourceInfo | None:
        # 用模擬板子（sim.py）：不需要硬體就能驗證 open／read／controls／close
        return SourceInfo(kind=cls.kind, id="k210:sim", name="K210 (simulated)", params={"port": "sim:"})

    # ---- 開啟與關閉 ------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._open

    def _resolve_port(self) -> str:
        want = str(self.info.params.get("port") or "").strip()
        if want.startswith("sim:"):
            return want
        try:
            cands = ports_mod.list_serial_ports()
        except Exception:  # noqa: BLE001
            cands = []
        if want:
            if any(c.device == want for c in cands):
                return want
            for c in cands:  # 插到別的 USB 孔，路徑變了，但穩定 id 還一樣
                if c.stable_id == self.info.id:
                    return c.device
            return want  # 讓開啟時的錯誤訊息說明找不到
        likely = [c for c in cands if c.likely]
        if len(likely) == 1:
            return likely[0].device
        if not likely:
            raise SourceError(tr("source.k210.err.no_port"), retryable=True, hint=tr("source.k210.hint.check_cable"))
        raise SourceError(tr("source.k210.err.many_ports", ports=", ".join(c.device for c in likely)))

    def open(self) -> None:
        if self._open:
            return
        port = self._resolve_port()
        baud = int(self.info.params.get("baud") or REPL_BAUD)
        try:
            ser = open_serial(port, baud)
        except Exception as e:  # noqa: BLE001 - 轉成給學生看的說明
            raise _explain_open_error(e, port) from e
        self._port = port
        self._ser = ser
        self._repl = RawRepl(ser)
        self._repl.time_scale = float(getattr(ser, "time_scale", 1.0))
        self._parser = PacketParser()
        self._latest = None
        self._dropped = self._decode_failed = 0
        self._cmd_id = 0
        try:
            self._enter_repl()
            self._start_stream()
        except BaseException:
            self._abort()
            raise
        self._open = True

    def _enter_repl(self) -> None:
        assert self._repl is not None
        try:
            self._repl.enter()
        except TimeoutError as e:
            raise SourceError(tr("source.k210.err.no_response"), retryable=True,
                              hint=tr("source.k210.hint.press_reset")) from e
        except OSError as e:
            raise SourceError(tr("source.k210.err.lost"), retryable=True, hint=tr("source.k210.hint.check_cable")) from e

    def _abort(self) -> None:
        """開啟失敗時盡力收尾，不讓序列埠一直被占著。"""
        try:
            if self._repl is not None:
                try:
                    self._halt_stream()
                    self._repl.exit()
                finally:
                    self._repl.close()
        except Exception as e:  # noqa: BLE001
            log.debug("收尾時發生錯誤（已忽略）：%s", e)
        self._repl = None
        self._ser = None
        self._open = False
        self._streaming = False

    def close(self) -> None:
        if self._repl is None:
            self._open = False
            return
        repl = self._repl
        try:
            self._halt_stream()
            repl.exit()
        except Exception as e:  # noqa: BLE001 - 板子被拔掉時也要能關
            log.debug("關閉時發生錯誤（已忽略）：%s", e)
        finally:
            try:
                repl.close()
            except Exception:  # noqa: BLE001
                pass
            self._repl = None
            self._ser = None
            self._open = False
            self._streaming = False
            self._latest = None

    # ---- 啟動與停止串流 --------------------------------------------------------

    def _build_params(self) -> dict[str, Any]:
        v = self._values
        extra_ops = [o for o in self.info.params.get("ops", []) if o in ("stats", "diff", "sharp", "blobs")]
        params: dict[str, Any] = {
            "framesize": v["framesize"], "pixformat": v["pixformat"], "quality": int(v["jpeg_quality"]),
            "send_every": int(v["send_every"]), "lcd": bool(self.info.params.get("lcd", False)),
            "vflip": bool(v["vflip"]), "hmirror": bool(v["hmirror"]), "dual_buff": False,
            "camera": dict(self._user_camera), "baud": STREAM_BAUD, "ops": extra_ops,
        }
        cam = params["camera"]
        if "auto_gain" in cam or "gain_db" in cam:  # 自動／手動要成對送，否則切手動時增益會變 0
            cam.update(auto_gain=bool(v["auto_gain"]), gain_db=float(v["gain_db"]))
        if "auto_exposure" in cam or "exposure_us" in cam:
            cam.update(auto_exposure=bool(v["auto_exposure"]), exposure_us=int(v["exposure_us"]))
        card = self._card
        if card is not None:
            dev = card.device
            task = _card_task(card)
            size = card.input.size or dev.get("input_size") or (320, 240)
            params.update(model=dev["path"], input_size=[int(size[0]), int(size[1])],
                          threshold=float(v["threshold"]), nms=float(v["nms"]))
            if task == "classify":
                params.update(ops=extra_ops + ["classify"], topk=int(dev.get("topk", 5)),
                              softmax=dev.get("softmax", "auto"))
            else:
                params["ops"] = extra_ops + ["yolo"]
                if dev.get("anchors"):
                    params["anchor"] = [float(a) for a in dev["anchors"]]
                params["classes"] = int(len(card.labels) or dev.get("classes", 1))
        return params

    def _start_stream(self) -> None:
        """送出程式、等 STREAM_START、切到串流鮑率。失敗時丟 SourceError（呼叫端負責收尾）。"""
        repl, ser = self._repl, self._ser
        assert repl is not None
        params = self._build_params()
        code = f"P = {params!r}\n" + board_script()
        repl.buf = b""
        try:
            repl.send_code(code)
            tok, pre = repl.read_until_any((b"STREAM_START", b"\x04"), timeout=START_TIMEOUT)
        except TimeoutError as e:
            err = repl.buf.decode("utf-8", "replace")
            self._halt_stream(force=True)
            if err.strip():
                raise SourceError(_explain_board_error(err), retryable=True,
                                  hint=tr("source.k210.hint.press_reset")) from e
            raise SourceError(tr("source.k210.err.start_timeout", seconds=START_TIMEOUT), retryable=True,
                              hint=tr("source.k210.hint.press_reset")) from e
        except OSError as e:
            raise SourceError(tr("source.k210.err.lost"), retryable=True, hint=tr("source.k210.hint.check_cable")) from e
        text = pre.decode("utf-8", "replace")
        if tok == b"\x04":  # 程式還沒開始串流就結束了：後面接著是錯誤訊息
            try:
                text += repl.read_until(b"\x04", timeout=3)[:-1].decode("utf-8", "replace")
            except TimeoutError:
                pass
            raise SourceError(_explain_board_error(text), hint=tr("source.k210.hint.press_reset"))
        ser.baudrate = STREAM_BAUD
        ser.timeout = 0.05
        self._started_at = time.time()
        self._streaming = True
        self._parser.reset()
        self._latest = None
        self._last_rx = time.monotonic()
        self._last_ack = {"id": 0, "ok": [], "err": []}
        self._last_cfg = {}
        self._board_error = ""
        self._parse_board_info(text)
        self.board_info["model_path"] = params.get("model", "")

    def _parse_board_info(self, text: str) -> None:
        info: dict[str, Any] = {"notes": []}
        for line in text.splitlines():
            key, _, val = line.strip().partition(" ")
            try:
                if key in ("CAPS", "CONFIG"):
                    info[key.lower()] = json.loads(val)
                elif key == "FW":
                    info["fw"] = val
                elif key in ("MEM", "MODEL"):
                    parts = val.split()
                    info[key.lower()] = {parts[i]: int(parts[i + 1]) for i in range(0, len(parts) - 1, 2)}
                elif key == "CAMERA":
                    info["camera"] = val
                elif key == "NOTE":
                    info["notes"].append(val)
            except (ValueError, IndexError):
                log.debug("無法解析板子資訊：%r", line)
        self.board_info = info

    def _halt_stream(self, force: bool = False) -> None:
        """停止板上串流並把序列埠切回 115200。時序是實測調出來的，不要改動。"""
        repl, ser = self._repl, self._ser
        if repl is None or ser is None:
            return
        sleep = repl.sleep
        if self._streaming or force:
            # 板子印出 STREAM_START 後約 0.4 秒才切到串流鮑率，太早送 Ctrl+C 會收不到
            wait = self._started_at + 0.6 - time.time()
            if wait > 0:
                sleep(wait)
            # 板子在 1.5 Mbps 下也能接收 Ctrl+C；重送直到不再收到串流封包
            ser.timeout = 0.3
            for _ in range(10):
                ser.write(b"\x03")
                sleep(0.3)
                if MAGIC not in ser.read(65536):
                    break
            sleep(0.5)               # 等 finally 把鮑率切回 115200
        ser.baudrate = REPL_BAUD
        ser.timeout = 0.1
        sleep(0.2)
        ser.reset_input_buffer()
        repl.buf = b""
        self._streaming = False

    def _restart(self) -> None:
        """停止目前的串流並用目前的設定重新啟動。"""
        assert self._repl is not None
        self._halt_stream()
        self._enter_repl()
        self._start_stream()

    # ---- 讀取 ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, int]:
        s = self._parser.stats
        s.update(dropped_frames=self._dropped, decode_failed=self._decode_failed)
        return s

    def _pump(self) -> None:
        """讀一次序列埠並處理收到的封包；只保留最新的一張有影像的封包。"""
        if self._ser is None:
            raise SourceError(tr("source.k210.err.not_open"))
        try:
            data = self._ser.read(65536)
        except (OSError, ValueError) as e:  # 拔線時 pyserial 丟 SerialException（OSError）
            raise SourceError(tr("source.k210.err.lost"), retryable=True, hint=tr("source.k210.hint.check_cable")) from e
        now = time.monotonic()
        if data:
            self._last_rx = now
            for item in self._parser.feed(data):
                if isinstance(item, TextChunk):
                    self._on_text(item.text)
                else:
                    self._on_packet(item)
            return
        silent = now - self._last_rx
        if self._board_error and silent > 1.0:
            raise SourceError(tr("source.k210.err.board_error", text=self._board_error[-300:]),
                              hint=tr("source.k210.hint.press_reset"))
        if silent > STALL_SECONDS:
            raise SourceError(tr("source.k210.err.stalled"), retryable=True, hint=tr("source.k210.hint.press_reset"))

    def _on_text(self, text: str) -> None:
        if not text.strip():
            return
        self.last_text = text
        if "Traceback" in text or "ERROR" in text or "Error" in text:
            log.warning("K210 板子訊息：%s", text.strip())
            self._board_error = (self._board_error + "\n" + text.strip()).strip()
        else:
            log.info("K210 板子訊息：%s", text.strip())

    def _on_packet(self, pkt: Packet) -> None:
        meta = pkt.meta
        ack = meta.get("ack")
        if isinstance(ack, dict) and isinstance(ack.get("id", 0), int) and ack.get("id", 0) >= self._last_ack["id"]:
            self._last_ack = ack
        cfg = meta.get("cfg")
        if isinstance(cfg, dict):
            self._last_cfg = cfg
        if pkt.jpeg:
            if self._latest is not None:
                self._dropped += 1  # 來不及處理的舊畫面
            self._latest = pkt
        else:  # 只傳數據：沒有影像，但統計與裝置推論結果要更新
            self.last_meta = self._public_meta(meta)
            self.last_result = self._make_result(meta, self._seq, None)

    def read(self, timeout: float = 1.0) -> Frame | None:
        if not self._open:
            raise SourceError(tr("source.k210.err.not_open"))
        if not self._streaming:
            raise SourceError(tr("source.k210.err.not_streaming"), retryable=True)
        deadline = time.monotonic() + timeout
        while True:
            self._pump()
            pkt, self._latest = self._latest, None
            if pkt is not None:
                frame = self._to_frame(pkt)
                if frame is not None:
                    return frame
            if time.monotonic() >= deadline:
                return None

    def _to_frame(self, pkt: Packet) -> Frame | None:
        img = cv2.imdecode(np.frombuffer(pkt.jpeg, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim not in (2, 3):
            self._decode_failed += 1
            return None
        seq = self._next_seq()
        meta = self._public_meta(pkt.meta)
        self.last_meta = meta
        result = self._make_result(pkt.meta, seq, (int(img.shape[1]), int(img.shape[0])))
        self.last_result = result
        return Frame(image=np.ascontiguousarray(img), seq=seq, t_cap=time.monotonic(), source_id=self.source_id,
                     jpeg=pkt.jpeg, meta=meta, device_result=result)

    def _public_meta(self, meta: dict[str, Any]) -> dict[str, Any]:
        """給 Frame.meta 的內容：板上資訊 + 電腦端統計（都能轉成 JSON）。"""
        out = {k: v for k, v in meta.items() if k != "ack"}
        out["board_seq"] = out.pop("seq", None)
        out["host"] = self.stats
        return out

    def _make_result(self, meta: dict[str, Any], seq: int, size: tuple[int, int] | None) -> InferenceResult | None:
        cls_list, objs = meta.get("cls"), meta.get("objs")
        if not isinstance(cls_list, list) and not isinstance(objs, list):
            return None
        w, h = size if size else (int(meta.get("w", 0) or 0), int(meta.get("h", 0) or 0))
        card = self._card
        us = meta.get("us") if isinstance(meta.get("us"), dict) else {}
        timings = {"infer": float(us.get("kpu", 0)) / 1000.0, "total": float(meta.get("lat", 0) or 0) / 1000.0}
        name = card.name if card else os.path.basename(str(self.board_info.get("model_path", ""))) or "device"

        def label(i: int) -> str:
            return card.label(i) if card else ""

        try:
            if isinstance(cls_list, list):
                top = [Classification(class_id=int(c[0]), score=float(c[1]), label=label(int(c[0])))
                       for c in cls_list if isinstance(c, (list, tuple)) and len(c) >= 2]
                top.sort(key=lambda c: c.score, reverse=True)
                return InferenceResult(kind="classify", model_name=name, frame_seq=seq, where="device", top_k=top,
                                       timings_ms=timings, image_size=(w, h))
            dets = []
            for o in objs or []:
                if isinstance(o, (list, tuple)) and len(o) >= 6:
                    x, y, bw, bh, conf, cid = o[:6]
                    dets.append(Detection(box=(float(x), float(y), float(x) + float(bw), float(y) + float(bh)),
                                          score=float(conf), class_id=int(cid), label=label(int(cid))))
            return InferenceResult(kind="detect", model_name=name, frame_seq=seq, where="device", detections=dets,
                                   timings_ms=timings, image_size=(w, h))
        except (TypeError, ValueError):
            log.debug("板上推論結果格式不正確：%r", meta)
            return None

    # ---- 參數 ------------------------------------------------------------------

    def controls(self) -> list[ControlSpec]:
        caps_sizes = (self.board_info.get("caps") or {}).get("framesizes")
        specs = []
        for s in _SPECS:
            if s.key == "framesize" and caps_sizes:
                opts = tuple(o for o in s.options if o[0] in caps_sizes) or s.options
                s = ControlSpec(**{**s.__dict__, "options": opts})
            specs.append(s)
        return specs

    def get_control(self, key: str) -> Any:
        if key not in self._values:
            raise KeyError(key)
        return self._values[key]

    def set_control(self, key: str, value: Any) -> ControlResult:
        spec = _SPEC_BY_KEY.get(key)
        if spec is None or spec.type == ControlType.ACTION:
            return ControlResult.failed(tr("source.k210.err.unknown_control", key=key))
        try:
            value = spec.validate(value)
        except ValueError as e:
            return ControlResult.failed(str(e))
        if not self._open:  # 還沒開啟：記下來，開啟時帶入
            self._values[key] = value
            if key in CAMERA_KEYS:
                self._user_camera[key] = value
            return ControlResult(ok=True, value=value, verified=None)
        if spec.live:
            return self.apply_controls({key: value})[key]
        return self._change_restart(key, value)

    def set_controls(self, changes: dict[str, Any]) -> dict[str, ControlResult]:
        """即時參數合併成一條 CMD（apply_controls）；需要重新啟動串流的參數逐一用 set_control。"""
        live = {k: v for k, v in changes.items() if (s := _SPEC_BY_KEY.get(k)) is not None and s.live}
        results = self.apply_controls(live) if live else {}
        for key, value in changes.items():
            if key not in live:
                results[key] = self.set_control(key, value)
        return results

    def apply_controls(self, changes: dict[str, Any]) -> dict[str, ControlResult]:
        """一次調整多個即時參數：合併成一行 CMD（舊 viewer 也是 150 ms 內合併），各自回報是否確認。"""
        results: dict[str, ControlResult] = {}
        cmd: dict[str, Any] = {}
        old: dict[str, Any] = {}
        for k, v in changes.items():
            spec = _SPEC_BY_KEY.get(k)
            if spec is None or not spec.live:
                results[k] = ControlResult.failed(tr("source.k210.err.unknown_control", key=k))
                continue
            try:
                v = spec.validate(v)
            except ValueError as e:
                results[k] = ControlResult.failed(str(e))
                continue
            old[k] = self._values[k]
            self._values[k] = v
            cmd.update(self._cmd_for(k, v))
        if not cmd:
            return results
        self._cmd_id += 1
        cid = self._cmd_id
        line = b"CMD " + json.dumps(dict(cmd, _id=cid), separators=(",", ":")).encode("ascii") + b"\n"
        try:
            assert self._ser is not None
            self._ser.write(line)
        except OSError as e:
            for k in old:
                self._values[k] = old[k]
                results[k] = ControlResult.failed(tr("source.k210.err.lost"), old[k])
            log.warning("送出指令失敗：%s", e)
            return results
        ack = self._wait_ack(cid)
        for k in old:
            results[k] = self._judge(k, old[k], ack)
        return results

    def _cmd_for(self, key: str, value: Any) -> dict[str, Any]:
        """板子的增益、曝光和自動模式綁在一起，所以一起送，避免切到手動時變成 0。"""
        v = self._values
        if key in ("auto_gain", "gain_db"):
            return {"auto_gain": bool(v["auto_gain"]), "gain_db": float(v["gain_db"])}
        if key in ("auto_exposure", "exposure_us"):
            return {"auto_exposure": bool(v["auto_exposure"]), "exposure_us": int(v["exposure_us"])}
        return {key: value}

    def _wait_ack(self, cid: int, timeout: float | None = None) -> dict[str, Any] | None:
        end = time.monotonic() + (ACK_TIMEOUT if timeout is None else timeout)
        while True:
            if self._last_ack.get("id", 0) >= cid:
                return self._last_ack
            if time.monotonic() >= end:
                return None
            self._pump()

    def _judge(self, key: str, old: Any, ack: dict[str, Any] | None) -> ControlResult:
        new = self._values[key]
        if ack is None:
            return ControlResult(ok=True, value=new, verified=None, message=tr("source.k210.err.no_ack"))
        related = ACK_NAMES.get(key, {key})
        bad = [e for e in ack.get("err", []) if str(e).split(":", 1)[0] in related]
        if bad:
            self._values[key] = old  # 板子不接受，還原
            return ControlResult(ok=True, value=old, verified=False,
                                 message=tr("source.k210.err.rejected", detail="; ".join(map(str, bad))))
        if key in CAMERA_KEYS:
            self._user_camera.update(self._cmd_for(key, new))
        echoed = self._last_cfg.get(key)
        same = echoed == new or (isinstance(new, (int, float)) and isinstance(echoed, (int, float))
                                 and not isinstance(new, bool) and abs(float(echoed) - float(new)) < 1e-6)
        confirmed = bool(related & set(ack.get("ok", []))) or same
        return ControlResult(ok=True, value=new, verified=True if confirmed else None)

    def _change_restart(self, key: str, value: Any) -> ControlResult:
        """framesize、pixformat、send_every、jpeg_quality 要重新啟動串流才會生效。"""
        old = self._values[key]
        self._values[key] = value
        try:
            self._restart()
        except SourceError as e:
            self._values[key] = old
            self._recover()
            return ControlResult.failed(e.message, old)
        cfg = self.board_info.get("config") or {}
        if key == "pixformat":
            ok = cfg.get("gray") == (value == "gray")
        else:
            ok = cfg.get(_RESTART_CONFIG[key]) == value
        return ControlResult(ok=True, value=value, verified=True if ok else None)

    def _recover(self) -> bool:
        """重新啟動失敗後，盡量讓串流回到可用狀態。"""
        try:
            self._restart()
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("無法恢復串流：%s", e)
            return False

    # ---- 動作 ------------------------------------------------------------------

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            device_inference=True, resolutions=list(SIZES.values()),
            actions=[s.key for s in _SPECS if s.type == ControlType.ACTION],
            provides_jpeg=True, live_controls=True, max_fps=20.0,
            extra=dict(self.board_info.get("caps") or {}))

    def run_action(self, key: str, **kwargs: Any) -> ControlResult:
        handlers = {"load_model": self._load_model, "unload_model": self._unload_model,
                    "upload_file": self._upload_file, "set_cpu_freq": self._set_cpu_freq,
                    "flash_firmware": self._flash_firmware}
        fn = handlers.get(key)
        if fn is None:
            return ControlResult.failed(tr("source.k210.err.unknown_control", key=key))
        if key != "flash_firmware" and not self._open:
            return ControlResult.failed(tr("source.k210.err.not_open"))
        progress, cancel = kwargs.pop("progress", None), kwargs.pop("cancel", None)
        try:
            return fn(progress=progress, cancel=cancel, **kwargs)
        except ReplCancelled:
            self._recover()
            return ControlResult.failed(tr("source.k210.action.cancelled"))
        except SourceError as e:
            return ControlResult.failed(e.message)
        except ReplError as e:
            self._recover()
            return ControlResult.failed(tr("source.k210.action.board_error", text=str(e)[-300:]))
        except (TimeoutError, OSError) as e:
            return ControlResult.failed(tr("source.k210.err.lost") + f" ({e})")
        except TypeError as e:
            return ControlResult.failed(tr("source.k210.action.bad_args", error=str(e)))

    @staticmethod
    def _report(progress: Any, lo: float, hi: float):
        """把子步驟的 0–1 進度換算到整個動作的 lo–hi 區間。"""
        def fn(frac: float, text: str = "") -> None:
            if progress:
                progress(lo + (hi - lo) * frac, text)
        return fn

    def _enter_for_edit(self) -> None:
        """停止串流並進入 raw REPL，準備上傳檔案或更換模型。"""
        self._halt_stream()
        self._enter_repl()

    def _load_model(self, *, card: ModelCard, progress: Any = None, cancel: Any = None) -> ControlResult:
        dev = card.device
        remote = str(dev.get("path") or "")
        if not remote:
            return ControlResult.failed(tr("source.k210.action.no_device_path"))
        if progress:
            progress(0.0, tr("source.k210.progress.stopping"))
        assert self._repl is not None
        self._enter_for_edit()
        local = card.model_path if card.file else None
        data = local.read_bytes() if local is not None and local.is_file() else None
        size = self._repl.stat_size(remote)
        if data is None and size is None:
            self._start_stream_or_recover()
            return ControlResult.failed(tr("source.k210.action.model_not_found", path=remote))
        if data is not None and size != len(data):
            if remote.startswith("/sd/") and not (self.board_info.get("caps") or {}).get("sd", True):
                self._start_stream_or_recover()
                return ControlResult.failed(tr("source.k210.err.no_sd"))
            self._repl.put(data, remote, self._report(progress, 0.02, 0.9), cancel)
        elif progress:
            progress(0.9, tr("source.k210.progress.up_to_date"))
        prev_card, prev_vals = self._card, dict(self._values)
        self._card = card
        self._values["threshold"] = card.conf_threshold
        self._values["nms"] = card.nms_threshold
        task = _card_task(card)
        size_wh = card.input.size or dev.get("input_size")
        if task == "detect" and size_wh:
            for name, wh in SIZES.items():  # 偵測模型要求畫面大小剛好等於模型輸入
                if tuple(wh) == (int(size_wh[0]), int(size_wh[1])):
                    self._values["framesize"] = name
                    self._values["pixformat"] = "rgb"
        if progress:
            progress(0.92, tr("source.k210.progress.starting"))
        try:
            self._start_stream()
        except SourceError as e:
            self._card, self._values = prev_card, prev_vals
            self._start_stream_or_recover()
            return ControlResult.failed(e.message)
        if progress:
            progress(1.0, "")
        loaded = "model" in self.board_info
        return ControlResult(ok=True, value=remote, verified=True if loaded else None,
                             message=tr("source.k210.action.model_loaded", name=card.name))

    def _start_stream_or_recover(self) -> None:
        try:
            self._start_stream()
        except Exception as e:  # noqa: BLE001
            log.warning("無法重新啟動串流：%s", e)

    def _unload_model(self, *, progress: Any = None, cancel: Any = None) -> ControlResult:
        self._card = None
        self._restart()
        return ControlResult(ok=True, message=tr("source.k210.action.model_unloaded"), verified=True)

    def _upload_file(self, *, local_path: str, remote_path: str, progress: Any = None,
                     cancel: Any = None) -> ControlResult:
        try:
            data = Path(local_path).read_bytes()
        except OSError as e:
            return ControlResult.failed(tr("source.k210.action.read_failed", path=str(local_path), error=str(e)))
        assert self._repl is not None
        self._enter_for_edit()
        try:
            self._repl.put(data, str(remote_path), self._report(progress, 0.0, 0.95), cancel)
        except (ReplCancelled, ReplError):
            self._start_stream_or_recover()
            raise
        self._start_stream()
        if progress:
            progress(1.0, "")
        return ControlResult(ok=True, value=str(remote_path), verified=True,
                             message=tr("source.k210.action.upload_done", remote=str(remote_path)))

    def _set_cpu_freq(self, *, cpu_mhz: int = 400, kpu_mhz: int = 400, progress: Any = None,
                      cancel: Any = None) -> ControlResult:
        cpu, kpu = int(cpu_mhz), int(kpu_mhz)
        if not (100 <= cpu <= 600 and 100 <= kpu <= 600):
            return ControlResult.failed(tr("source.k210.action.bad_freq"))
        assert self._repl is not None
        self._enter_for_edit()
        code = f"from Maix import freq\nfreq.set(cpu={cpu}, pll1=400, kpu_div={kpu_div_for(kpu)})\n"
        try:  # 設定後板子會重開機，來不及回 OK 是正常的，所以不等
            self._repl.send_code(code, wait_ok=False)
        except OSError:
            pass
        if progress:
            progress(0.3, tr("source.k210.progress.rebooting"))
        self._reconnect(progress)
        verified: bool | None = None
        try:
            out = self._repl.exec("from Maix import freq\nprint('FREQ', freq.get())") if self._repl else ""
            digits = [int(t) for t in "".join(c if c.isdigit() else " " for c in out).split()]
            if digits:
                verified = digits[0] == cpu
        except Exception:  # noqa: BLE001 - 讀不到就維持「未確認」
            pass
        self._start_stream()
        if progress:
            progress(1.0, "")
        return ControlResult(ok=True, value={"cpu_mhz": cpu, "kpu_mhz": kpu}, verified=verified,
                             message=tr("source.k210.action.freq_done"))

    def _reconnect(self, progress: Any = None) -> None:
        """板子重開機後重新開序列埠並進入 raw REPL（最多約 25 秒）。"""
        assert self._repl is not None
        scale = self._repl.time_scale
        try:
            self._repl.close()
        except Exception:  # noqa: BLE001
            pass
        last: Exception | None = None
        for attempt in range(8):
            time.sleep(2.0 * scale)
            try:
                self._ser = open_serial(self._port, int(self.info.params.get("baud") or REPL_BAUD))
                self._repl = RawRepl(self._ser)
                self._repl.time_scale = scale
                self._repl.enter(timeout=4)
                return
            except (OSError, TimeoutError) as e:
                last = e
                if self._ser is not None:
                    try:
                        self._ser.close()
                    except Exception:  # noqa: BLE001
                        pass
            if progress:
                progress(0.3 + 0.05 * attempt, tr("source.k210.progress.rebooting"))
        raise SourceError(tr("source.k210.err.reconnect_failed"), hint=tr("source.k210.hint.press_reset")) from last

    def _flash_firmware(self, *, firmware_path: str, board: str = flash_mod.DEFAULT_BOARD,
                        baud: int = flash_mod.FLASH_BAUD, progress: Any = None, cancel: Any = None) -> ControlResult:
        port = self._port or self._resolve_port()
        self.close()  # 燒錄前必須先關閉串流，序列埠同時只能被一個程式開著
        res = flash_mod.flash_firmware(port, str(firmware_path), board=board, baud=baud, progress=progress,
                                       cancel=cancel)
        msg = res.message + ("\n" + tr("source.k210.flash.reopen") if res.ok else "")
        return ControlResult(ok=res.ok, verified=True if res.ok else False, message=msg)


def kpu_div_for(kpu_mhz: int, pll1: int = 400) -> int:
    """MaixPy 的 KPU 頻率 = pll1 / (kpu_div + 1)。需要在實體板子上確認。"""
    return max(0, round(pll1 / kpu_mhz) - 1)


def _card_task(card: ModelCard) -> str:
    """板上模型是偵測還是分類；說明檔沒寫（auto）時，有 anchors 的當偵測。"""
    task = str(card.device.get("task") or card.task or "auto").lower()
    if task in ("detect", "classify"):
        return task
    return "detect" if card.device.get("anchors") else "classify"
