"""電腦相機（USB webcam、筆電內建相機、iPhone 接續互通相機）。

設計重點：
- 列舉用 enumerate.py，**不開相機**；真正開相機才會碰硬體（macOS 這時才跳權限視窗）。
- OpenCV 沒辦法查詢相機有哪些參數、範圍多大，所以 open() 之後對候選屬性逐一「試著改、
  讀回、改回去」，只有真的有反應的才列在 controls()，面板就不會出現一堆沒用的滑桿。
- macOS 的 AVFoundation 幾乎不開放參數，這是正常的：只提供解析度與 FPS。
- 只回傳最新畫面：用 grab()+retrieve()，BUFFERSIZE 設 1，避免畫面越積越舊。
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import time
from typing import Any

import cv2
import numpy as np

from edgelens.core.controls import ControlResult, ControlSpec, ControlType
from edgelens.core.frame import Frame
from edgelens.i18n import tr
from edgelens.sources.base import FrameSource, SourceCapabilities, SourceError, SourceInfo, register_source
from edgelens.sources.enumerate import (
    API_IDS,
    CameraDevice,
    current_platform,
    default_api,
    find_camera,
    linux_device_path,
    list_cameras,
    role_detail,
)

log = logging.getLogger(__name__)

STANDARD_RESOLUTIONS = [(320, 240), (640, 480), (800, 600), (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)]
DEFAULT_RESOLUTION = (640, 480)
DEFAULT_FPS = 30
API_CHOICES = ("auto", "avfoundation", "msmf", "dshow", "v4l2")

WARMUP_SECONDS = 4.0      # 開啟後等第一張畫面的時間（macOS 第一次要等相機啟動）
MAX_GRAB_FAILS = 10       # 連續 grab 失敗幾次視為斷線

# ---- 候選參數表 -------------------------------------------------------------------
# key → (OpenCV 屬性名稱, 翻譯鍵, 是否放進「進階」)
_NUMERIC: dict[str, tuple[str, str, bool]] = {
    "brightness": ("CAP_PROP_BRIGHTNESS", "control.brightness", False),
    "contrast": ("CAP_PROP_CONTRAST", "control.contrast", False),
    "saturation": ("CAP_PROP_SATURATION", "control.saturation", False),
    "hue": ("CAP_PROP_HUE", "control.webcam.hue", True),
    "gain": ("CAP_PROP_GAIN", "control.gain", False),
    "exposure": ("CAP_PROP_EXPOSURE", "control.exposure", False),
    "white_balance": ("CAP_PROP_WB_TEMPERATURE", "control.white_balance", False),
    "focus": ("CAP_PROP_FOCUS", "control.focus", False),
    "sharpness": ("CAP_PROP_SHARPNESS", "control.webcam.sharpness", True),
    "zoom": ("CAP_PROP_ZOOM", "control.webcam.zoom", True),
    "backlight": ("CAP_PROP_BACKLIGHT", "control.webcam.backlight", True),
}
_BOOLEAN: dict[str, tuple[str, str]] = {
    "auto_exposure": ("CAP_PROP_AUTO_EXPOSURE", "control.auto_exposure"),
    "auto_white_balance": ("CAP_PROP_AUTO_WB", "control.auto_white_balance"),
    "autofocus": ("CAP_PROP_AUTOFOCUS", "control.auto_focus"),
}
# 手動參數依賴的「自動」開關：自動開著時改手動值常常沒有作用
_AUTO_OF = {"exposure": "auto_exposure", "white_balance": "auto_white_balance", "focus": "autofocus"}

# 預設範圍（OpenCV 無法查詢範圍，這是各平台常見 UVC 相機的經驗值；讀回值超出時會自動擴大）。
# 來源：Linux uvcvideo 的 V4L2 控制項（brightness/contrast/saturation/gain 多為 0–255，
# exposure_absolute 單位 100µs、常見 3–2047，zoom_absolute 100–500）；Windows（DSHOW/MSMF）的
# VideoProcAmp 屬性（brightness 常見 −64–64，contrast 0–64，saturation 0–128，sharpness 0–14）；
# Windows 的 exposure 是 log2 秒（約 −13～−1，也就是 1/8192 秒到 0.5 秒）。
_RANGES_V4L2: dict[str, tuple[float, float]] = {
    "brightness": (0, 255), "contrast": (0, 255), "saturation": (0, 255), "hue": (-180, 180),
    "gain": (0, 255), "exposure": (3, 2047), "white_balance": (2800, 6500), "focus": (0, 255),
    "sharpness": (0, 255), "zoom": (100, 500), "backlight": (0, 10),
}
_RANGES_WIN: dict[str, tuple[float, float]] = {
    "brightness": (-64, 64), "contrast": (0, 64), "saturation": (0, 128), "hue": (-180, 180),
    "gain": (0, 255), "exposure": (-13, -1), "white_balance": (2800, 6500), "focus": (0, 255),
    "sharpness": (0, 14), "zoom": (100, 500), "backlight": (0, 10),
}
# 「自動」開關的 (開, 關) 值：V4L2 的 auto_exposure 是 3＝自動光圈、1＝手動；
# DSHOW 是 0.75／0.25（OpenCV 文件）；MSMF 與其他是 1／0。
_AUTO_VALUES: dict[str, dict[str, tuple[float, float]]] = {
    "auto_exposure": {"v4l2": (3, 1), "dshow": (0.75, 0.25), "msmf": (1, 0)},
}


def _range_for(key: str, api: str) -> tuple[float, float]:
    table = _RANGES_V4L2 if api in ("v4l2", "avfoundation", "") else _RANGES_WIN
    return table[key]


def _auto_values(key: str, api: str) -> tuple[float, float]:
    return _AUTO_VALUES.get(key, {}).get(api, (1, 0))


# ---- 平台相關檢查（獨立成函式，測試才能替換）--------------------------------------


def _device_exists(path: str) -> bool:
    return os.path.exists(path)


def _device_access(path: str) -> bool:
    return os.access(path, os.R_OK | os.W_OK)


def _res_key(w: int, h: int) -> str:
    return f"{w}x{h}"


def _parse_res(s: Any) -> tuple[int, int]:
    try:
        w, h = str(s).lower().replace("×", "x").split("x")
        return int(w), int(h)
    except (ValueError, TypeError):
        return DEFAULT_RESOLUTION


def _num(v: Any) -> float | None:
    """屬性值轉成 float；NaN 或無法轉換回傳 None。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


@register_source
class WebcamSource(FrameSource):
    kind = "webcam"
    display_name_key = "source.webcam.name"
    description_key = "source.webcam.desc"
    discover_is_safe = False   # 真的列舉會呼叫系統的相機 API
    manual_add = True

    def __init__(self, info: SourceInfo):
        super().__init__(info)
        p = info.params
        self._index = int(p.get("index", 0) or 0)
        self._api_param = str(p.get("api", "auto") or "auto")
        self._name = str(p.get("name", "") or "")
        self._dup = int(p.get("dup", 1) or 1)
        self._want_res = _parse_res(p.get("resolution", _res_key(*DEFAULT_RESOLUTION)))
        self._want_fps = int(p.get("fps", DEFAULT_FPS) or DEFAULT_FPS)
        self._cap: Any = None
        self._cur_index = self._index
        self._cur_api = ""           # 實際使用的後端名稱
        self._first: np.ndarray | None = None   # 開啟時為了確認有畫面而讀的第一張
        self._actual_res = self._want_res
        self._fails = 0
        self._numeric: dict[str, int] = {}      # key → 屬性編號（實測可用的）
        self._boolean: dict[str, int] = {}
        self._controls: list[ControlSpec] = []
        self._last_fps_report = 0.0

    # ---- 類別層級 ------------------------------------------------------------------

    @classmethod
    def discover(cls, timeout: float = 2.0) -> list[SourceInfo]:
        try:
            devices = list_cameras(probe=False)
        except Exception as e:  # noqa: BLE001 - 列舉失敗不可讓「新增相機」整個壞掉
            log.warning("列舉相機失敗：%s", e)
            return []
        return [cls._info_from_device(d) for d in devices]

    @staticmethod
    def _info_from_device(d: CameraDevice) -> SourceInfo:
        parts = [role_detail(d.role)]
        if d.backend:
            parts.append(d.backend)
        if d.vid_pid:
            parts.append(d.vid_pid)
        label = d.name if d.dup <= 1 else f"{d.name} #{d.dup}"
        return SourceInfo(kind="webcam", id=d.id, name=label,
                          params={"index": d.index, "api": d.backend or "auto", "name": d.name, "dup": d.dup},
                          detail=" · ".join(x for x in parts if x))

    @classmethod
    def param_specs(cls) -> list[ControlSpec]:
        return [
            ControlSpec("index", ControlType.INT, "control.webcam.index", default=0, min=0, max=63, group="source"),
            ControlSpec("api", ControlType.ENUM, "control.webcam.api", default="auto", group="source",
                        options=tuple((a, f"source.webcam.api.{a}") for a in API_CHOICES)),
        ]

    @classmethod
    def info_from_params(cls, params: dict[str, Any]) -> SourceInfo:
        index = int(params.get("index", 0) or 0)
        api = str(params.get("api", "auto") or "auto")
        suffix = "" if api == "auto" else f"@{api}"
        return SourceInfo(kind=cls.kind, id=f"webcam:camera{index}{suffix}",
                          name=tr("source.webcam.camera_n", index=index),
                          params={"index": index, "api": api})

    # ---- 開啟與關閉 ----------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    def open(self) -> None:
        if self._cap is not None:
            return
        attempts = self._resolve_attempts()
        failure = "open"
        for idx, api in attempts:
            cap, failure = self._try_open(idx, api, self._want_res, self._want_fps)
            if cap is not None:
                self._cap, self._cur_index, self._cur_api = cap, idx, api
                break
        else:
            raise self._open_error(attempts[0][0] if attempts else self._index, failure)
        self._fails = 0
        self._probe_controls()

    def close(self) -> None:
        cap, self._cap = self._cap, None
        self._first = None
        self._controls = []
        self._numeric = {}
        self._boolean = {}
        if cap is not None:
            try:
                cap.release()
            except Exception as e:  # noqa: BLE001
                log.debug("釋放相機時出錯：%s", e)

    def _resolve_attempts(self) -> list[tuple[int, str]]:
        """決定要嘗試的 (index, 後端) 清單。有名稱時先用名稱重新找 index
        （插拔 USB 後編號會變），找不到而且列舉有結果時，不要亂開別的相機。"""
        api = self._api_param
        if self._name:
            try:
                devices = list_cameras(probe=False)
            except Exception as e:  # noqa: BLE001
                log.warning("列舉相機失敗，改用原本的編號：%s", e)
                devices = []
            dev = find_camera(self._name, self._dup, devices)
            if dev is not None:
                attempts: list[tuple[int, str]] = []
                if api == "auto":
                    attempts.append((dev.index, dev.backend or default_api()))
                    attempts += [(i, a) for a, i in dev.alternatives.items()]
                elif api == dev.backend:
                    attempts.append((dev.index, api))
                elif api in dev.alternatives:
                    attempts.append((dev.alternatives[api], api))
                else:
                    attempts.append((self._index, api))
                if dev.index != self._index:
                    log.info("相機「%s」的編號從 %s 變成 %s", self._name, self._index, dev.index)
                return attempts
            if devices:
                raise SourceError(tr("source.webcam.err.not_found", name=self._name),
                                  retryable=True, hint=tr("source.webcam.hint.replug"))
        if api == "auto":
            api = default_api()
        return [(self._index, api)]

    def _try_open(self, idx: int, api: str, res: tuple[int, int], fps: int) -> tuple[Any, str]:
        """開啟並確認有畫面。回傳 (cap, "")；失敗回傳 (None, "open"|"no_frame")。"""
        cap = None
        try:
            cap = cv2.VideoCapture(idx, API_IDS[api]) if api in API_IDS else cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                return None, "open"
        except Exception as e:  # noqa: BLE001 - 某些驅動在不存在的編號會直接丟例外
            log.info("開啟相機 %s（%s）失敗：%s", idx, api, e)
            if cap is not None:
                cap.release()
            return None, "open"
        self._apply_stream_props(cap, api, res, fps)
        deadline = time.monotonic() + WARMUP_SECONDS
        while True:
            ok, img = cap.read()
            if ok and img is not None:
                self._first = img
                self._actual_res = (int(img.shape[1]), int(img.shape[0]))
                return cap, ""
            if time.monotonic() >= deadline:
                cap.release()
                return None, "no_frame"
            time.sleep(0.05)

    @staticmethod
    def _apply_stream_props(cap: Any, api: str, res: tuple[int, int], fps: int) -> None:
        # AVFoundation 不接受 FOURCC（固定由系統決定格式）；其他 USB 相機高解析度需要 MJPG，
        # 否則會落在未壓縮的 YUY2，頻寬不夠只剩很低的 FPS。
        if api != "avfoundation":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        cap.set(cv2.CAP_PROP_FPS, fps)
        prop = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
        if prop is not None:
            cap.set(prop, 1)   # 不支援時 set 回傳 False，忽略即可

    def _open_error(self, index: int, failure: str) -> SourceError:
        plat = current_platform()
        key = "source.webcam.err.no_frame" if failure == "no_frame" else "source.webcam.err.open"
        message = tr(key, index=index)
        if plat == "darwin":
            return SourceError(message, retryable=True, hint=tr("source.webcam.hint.mac"))
        if plat == "win":
            return SourceError(message, retryable=True, hint=tr("source.webcam.hint.win"))
        if plat == "linux":
            path = linux_device_path(index)
            if not _device_exists(path):
                return SourceError(tr("source.webcam.err.linux_missing", path=path), retryable=True,
                                   hint=tr("source.webcam.hint.linux_missing"))
            if not _device_access(path):
                return SourceError(tr("source.webcam.err.linux_perm", path=path), retryable=True,
                                   hint=tr("source.webcam.hint.linux_perm", path=path))
        return SourceError(message, retryable=True, hint=tr("source.webcam.hint.generic"))

    # ---- 讀取 ----------------------------------------------------------------------

    def read(self, timeout: float = 1.0) -> Frame | None:
        cap = self._cap
        if cap is None:
            raise SourceError(tr("source.webcam.err.not_open"))
        t_cap = time.monotonic()
        img = self._first
        if img is not None:
            self._first = None
        else:
            deadline = t_cap + max(timeout, 0.0)
            while True:
                ok = cap.grab()
                if ok:
                    ok, img = cap.retrieve()
                    if ok and img is not None:
                        self._fails = 0
                        t_cap = time.monotonic()
                        break
                self._fails += 1
                if self._fails >= MAX_GRAB_FAILS:
                    self.close()
                    raise SourceError(tr("source.webcam.err.disconnected"), retryable=True,
                                      hint=tr("source.webcam.hint.replug"))
                if time.monotonic() >= deadline:
                    return None
                time.sleep(0.02)
        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        self._actual_res = (w, h)
        return Frame(image=img, seq=self._next_seq(), t_cap=t_cap, source_id=self.source_id,
                     meta={"width": w, "height": h})

    # ---- 參數探測 ------------------------------------------------------------------

    def _get(self, pid: int) -> float | None:
        try:
            return _num(self._cap.get(pid))
        except Exception:  # noqa: BLE001
            return None

    def _set(self, pid: int, value: float) -> bool:
        try:
            return bool(self._cap.set(pid, float(value)))
        except Exception:  # noqa: BLE001
            return False

    def _try_change(self, pid: int, test: float) -> tuple[bool, float | None]:
        """設成 test，回傳 (set 成功且讀回值有改變, 讀回值)。不負責還原。"""
        before = self._get(pid)
        if before is None or not self._set(pid, test):
            return False, None
        after = self._get(pid)
        return after is not None and abs(after - before) > 1e-6, after

    def _probe_controls(self) -> None:
        """對候選屬性逐一試著改、讀回、改回原值；有反應的才算支援。"""
        api = self._cur_api
        specs: list[ControlSpec] = []
        self._numeric, self._boolean = {}, {}
        if api != "avfoundation":
            booleans: dict[str, ControlSpec] = {}
            for key, (cv_name, label_key) in _BOOLEAN.items():
                pid = getattr(cv2, cv_name, None)
                if pid is None:
                    continue
                spec = self._probe_boolean(key, pid, label_key, api)
                if spec is not None:
                    booleans[key] = spec
                    self._boolean[key] = pid
            for key, (cv_name, label_key, advanced) in _NUMERIC.items():
                pid = getattr(cv2, cv_name, None)
                if pid is None:
                    continue
                spec = self._probe_numeric(key, pid, label_key, advanced, api, key in _AUTO_OF and
                                           _AUTO_OF[key] in booleans)
                if spec is not None:
                    specs.append(spec)
                    self._numeric[key] = pid
            specs += list(booleans.values())
        self._controls = [self._resolution_spec(), self._fps_spec()] + specs
        if api == "dshow":
            self._controls.append(ControlSpec("settings_dialog", ControlType.ACTION, "control.webcam.settings_dialog",
                                              group="action"))

    def _probe_boolean(self, key: str, pid: int, label_key: str, api: str) -> ControlSpec | None:
        v0 = self._get(pid)
        if v0 is None:
            return None
        on, off = _auto_values(key, api)
        test = off if abs(v0 - on) <= abs(v0 - off) else on
        changed, _ = self._try_change(pid, test)
        self._set(pid, v0)   # 不管成功與否都還原
        if not changed:
            return None
        return ControlSpec(key, ControlType.BOOL, label_key, default=abs(v0 - on) <= abs(v0 - off),
                           readback=True, group="camera")

    def _probe_numeric(self, key: str, pid: int, label_key: str, advanced: bool, api: str,
                       has_auto: bool) -> ControlSpec | None:
        v0 = self._get(pid)
        if v0 is None:
            return None
        lo, hi = _range_for(key, api)
        is_float = key == "exposure" and api in ("msmf", "dshow")
        lo, hi = min(lo, v0), max(hi, v0)
        delta = max(1.0, (hi - lo) // 4) if not is_float else 2.0
        test = v0 + delta if v0 + delta <= hi else v0 - delta
        changed, after = self._try_change(pid, test)
        self._set(pid, v0)
        if not changed and has_auto:
            # 自動模式開著時手動值常常不會動：暫時切到手動再試一次，然後恢復
            auto_key = _AUTO_OF[key]
            apid = self._boolean[auto_key]
            a0 = self._get(apid)
            if a0 is not None:
                self._set(apid, _auto_values(auto_key, api)[1])
                changed, after = self._try_change(pid, test)
                self._set(pid, v0)
                self._set(apid, a0)
        if not changed:
            return None
        if after is not None:
            lo, hi = min(lo, after), max(hi, after)
        if is_float:
            return ControlSpec(key, ControlType.FLOAT, label_key, default=v0, min=lo, max=hi, step=1.0,
                               readback=True, group="advanced" if advanced else "camera", advanced=advanced)
        return ControlSpec(key, ControlType.INT, label_key, default=int(round(v0)), min=int(math.floor(lo)),
                           max=int(math.ceil(hi)), step=1, readback=True,
                           group="advanced" if advanced else "camera", advanced=advanced)

    def _resolution_options(self) -> list[tuple[int, int]]:
        opts = list(STANDARD_RESOLUTIONS)
        if self._actual_res not in opts:
            opts.append(self._actual_res)
        return sorted(opts)

    def _resolution_spec(self) -> ControlSpec:
        return ControlSpec("resolution", ControlType.ENUM, "control.resolution", default=_res_key(*self._actual_res),
                           options=tuple((_res_key(w, h), f"{w}×{h}") for w, h in self._resolution_options()),
                           live=False, group="camera")

    def _fps_spec(self) -> ControlSpec:
        cur = _num(self._cap.get(cv2.CAP_PROP_FPS)) if self._cap is not None else None
        default = int(round(cur)) if cur and 1 <= cur <= 240 else self._want_fps
        return ControlSpec("fps", ControlType.INT, "control.fps", default=default, min=1, max=240, unit="fps",
                           live=False, group="camera")

    # ---- 參數讀寫 ------------------------------------------------------------------

    def controls(self) -> list[ControlSpec]:
        return list(self._controls) if self._cap is not None else []

    def _spec(self, key: str) -> ControlSpec | None:
        return next((s for s in self._controls if s.key == key), None)

    def get_control(self, key: str) -> Any:
        if self._cap is None:
            raise KeyError(key)
        if key == "resolution":
            return _res_key(*self._actual_res)
        if key == "fps":
            cur = self._get(cv2.CAP_PROP_FPS)
            return int(round(cur)) if cur and cur > 0 else self._want_fps
        if key in self._numeric:
            v = self._get(self._numeric[key])
            spec = self._spec(key)
            if v is None:
                raise KeyError(key)
            return int(round(v)) if spec is not None and spec.type == ControlType.INT else v
        if key in self._boolean:
            v = self._get(self._boolean[key])
            if v is None:
                raise KeyError(key)
            on, off = _auto_values(key, self._cur_api)
            return abs(v - on) <= abs(v - off)
        raise KeyError(key)

    def set_control(self, key: str, value: Any) -> ControlResult:
        if self._cap is None:
            return ControlResult.failed(tr("source.webcam.err.not_open"), value)
        spec = self._spec(key)
        if spec is None or spec.type == ControlType.ACTION:
            return ControlResult.failed(tr("source.webcam.msg.unsupported", key=key), value)
        try:
            v = spec.validate(value)
        except ValueError as e:
            return ControlResult.failed(str(e), value)
        if key in ("resolution", "fps"):
            return self._reopen_with(key, v)
        if key in self._boolean:
            return self._set_boolean(key, bool(v))
        return self._set_numeric(key, spec, v)

    def _set_boolean(self, key: str, want: bool) -> ControlResult:
        pid = self._boolean[key]
        on, off = _auto_values(key, self._cur_api)
        if not self._set(pid, on if want else off):
            return ControlResult.failed(tr("source.webcam.msg.set_failed"), want)
        read = self._get(pid)
        if read is None:
            return ControlResult(ok=True, value=want, verified=None)
        actual = abs(read - on) <= abs(read - off)
        return ControlResult(ok=True, value=actual, verified=actual == want)

    def _set_numeric(self, key: str, spec: ControlSpec, v: float) -> ControlResult:
        pid = self._numeric[key]
        if not self._set(pid, v):
            return ControlResult.failed(tr("source.webcam.msg.set_failed"), v)
        read = self._get(pid)
        if read is None:
            return ControlResult(ok=True, value=v, verified=None)
        span = (spec.max or 0) - (spec.min or 0)
        verified = abs(read - v) <= max(1.0, 0.01 * span)
        if (spec.max is not None and read > spec.max) or (spec.min is not None and read < spec.min):
            # 實際範圍比預設大：擴大範圍，之後才能調到這個值
            wide = dataclasses.replace(spec, min=min(spec.min, read) if spec.min is not None else read,
                                       max=max(spec.max, read) if spec.max is not None else read)
            self._controls = [wide if s.key == key else s for s in self._controls]
        message = ""
        if not verified:
            auto_key = _AUTO_OF.get(key)
            if auto_key in self._boolean and self.get_control(auto_key):
                message = tr("source.webcam.msg.turn_off_auto")
            else:
                message = tr("source.webcam.msg.not_accepted")
        value = int(round(read)) if spec.type == ControlType.INT else read
        return ControlResult(ok=True, value=value, verified=verified, message=message)

    def _reopen_with(self, key: str, v: Any) -> ControlResult:
        """解析度與 FPS 要重新開啟才會生效；失敗時盡量回到原本的設定。"""
        old_res, old_fps = self._actual_res, self._want_fps
        old_idx, old_api = self._cur_index, self._cur_api
        res, fps = (_parse_res(v), self._want_fps) if key == "resolution" else (self._want_res, int(v))
        # 保留序號遞增，所以不重設 self._seq
        self.close()
        cap, _ = self._try_open(old_idx, old_api, res, fps)
        if cap is None:
            cap, _ = self._try_open(old_idx, old_api, old_res, old_fps)
            if cap is None:
                return ControlResult.failed(tr("source.webcam.msg.reopen_failed"), v)
            self._finish_reopen(cap, old_idx, old_api, old_res, old_fps)
            return ControlResult(ok=False, value=_res_key(*self._actual_res) if key == "resolution" else old_fps,
                                 verified=False, message=tr("source.webcam.msg.reopen_failed"))
        self._finish_reopen(cap, old_idx, old_api, res, fps)
        if key == "resolution":
            return ControlResult(ok=True, value=_res_key(*self._actual_res),
                                 verified=self._actual_res == res,
                                 message="" if self._actual_res == res else tr("source.webcam.msg.res_changed"))
        actual = self.get_control("fps")
        return ControlResult(ok=True, value=actual, verified=abs(actual - fps) <= 1)

    def _finish_reopen(self, cap: Any, idx: int, api: str, res: tuple[int, int], fps: int) -> None:
        self._cap, self._cur_index, self._cur_api = cap, idx, api
        self._want_res, self._want_fps = res, fps
        self._fails = 0
        self._probe_controls()

    def run_action(self, key: str, **kwargs: Any) -> ControlResult:
        if key == "settings_dialog" and self._cap is not None and self._cur_api == "dshow":
            # 開啟相機驅動程式自己的設定視窗（只有 DSHOW 有）
            ok = self._set(cv2.CAP_PROP_SETTINGS, 1)
            return ControlResult(ok=ok, message="" if ok else tr("source.webcam.msg.set_failed"))
        return super().run_action(key, **kwargs)

    def capabilities(self) -> SourceCapabilities:
        extra: dict[str, Any] = {"api": self._cur_api, "index": self._cur_index,
                                 "probed_controls": sorted([*self._numeric, *self._boolean])}
        if self._cur_api == "avfoundation":
            extra["note"] = tr("source.webcam.note.mac_no_controls")
        fps = self._get(cv2.CAP_PROP_FPS) if self._cap is not None else None
        return SourceCapabilities(
            resolutions=self._resolution_options(),
            actions=[c.key for c in self._controls if c.type == ControlType.ACTION],
            live_controls=True,
            max_fps=float(fps) if fps and fps > 0 else float(self._want_fps),
            extra=extra,
        )
