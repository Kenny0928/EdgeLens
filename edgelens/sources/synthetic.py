"""合成測試圖案：沒有相機也能操作與測試。也是寫新來源時的參考範例。"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from edgelens.core.controls import ControlResult, ControlSpec, ControlType
from edgelens.core.frame import Frame
from edgelens.sources.base import FrameSource, SourceCapabilities, SourceError, SourceInfo, register_source

PATTERNS = ("bars", "moving_box", "noise", "checker", "shapes")
RESOLUTIONS = [(160, 120), (320, 240), (640, 480), (1280, 720)]


@register_source
class SyntheticSource(FrameSource):
    kind = "synthetic"
    display_name_key = "source.synthetic.name"
    description_key = "source.synthetic.desc"
    discover_is_safe = True
    manual_add = True

    _SPECS = [
        ControlSpec("pattern", ControlType.ENUM, "control.synthetic.pattern", default="moving_box",
                    options=tuple((p, f"control.synthetic.pattern.{p}") for p in PATTERNS), group="image"),
        ControlSpec("fps", ControlType.INT, "control.fps", default=30, min=1, max=120, unit="fps", group="camera"),
        ControlSpec("resolution", ControlType.ENUM, "control.resolution", default="640x480",
                    options=tuple((f"{w}x{h}", f"{w}×{h}") for w, h in RESOLUTIONS), group="camera"),
        ControlSpec("brightness", ControlType.INT, "control.brightness", default=0, min=-100, max=100,
                    readback=True, group="camera"),
        ControlSpec("noise", ControlType.INT, "control.synthetic.noise", default=0, min=0, max=100, group="image"),
        ControlSpec("gray", ControlType.BOOL, "control.synthetic.gray", default=False, group="image"),
        ControlSpec("encode_jpeg", ControlType.BOOL, "control.synthetic.encode_jpeg", default=False,
                    group="advanced", advanced=True),
        ControlSpec("reset", ControlType.ACTION, "control.synthetic.reset", group="action"),
    ]

    def __init__(self, info: SourceInfo):
        super().__init__(info)
        self._open = False
        self._values: dict[str, Any] = {s.key: s.default_value() for s in self._SPECS if s.type != ControlType.ACTION}
        for k, v in info.params.items():
            spec = next((s for s in self._SPECS if s.key == k), None)
            if spec is not None and spec.type != ControlType.ACTION:
                self._values[k] = spec.validate(v)
        self._t0 = 0.0
        self._next_due = 0.0
        self._rng = np.random.default_rng(0)

    @classmethod
    def discover(cls, timeout: float = 2.0) -> list[SourceInfo]:
        return [cls.info_from_params({})]

    @classmethod
    def param_specs(cls) -> list[ControlSpec]:
        return [s for s in cls._SPECS if s.key in ("pattern", "resolution", "fps")]

    @classmethod
    def info_from_params(cls, params: dict[str, Any]) -> SourceInfo:
        pattern = params.get("pattern", "moving_box")
        return SourceInfo(kind=cls.kind, id=f"synthetic:{pattern}", name=f"Synthetic ({pattern})",
                          params=dict(params), detail="test pattern")

    @classmethod
    def contract_fixture(cls, tmp_path: Path) -> SourceInfo | None:
        return cls.info_from_params({"fps": 120, "resolution": "160x120"})

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> None:
        if self._open:
            return
        self._open = True
        self._t0 = time.monotonic()
        self._next_due = self._t0

    def close(self) -> None:
        self._open = False

    def read(self, timeout: float = 1.0) -> Frame | None:
        if not self._open:
            raise SourceError("合成來源尚未開啟")
        now = time.monotonic()
        wait = self._next_due - now
        if wait > timeout:
            time.sleep(max(timeout, 0))
            return None
        if wait > 0:
            time.sleep(wait)
        period = 1.0 / float(self._values["fps"])
        # 落後太多時直接跳到現在，不補拍（行為和真的相機一樣：只給最新畫面）
        self._next_due = max(self._next_due + period, time.monotonic())
        t_cap = time.monotonic()
        img = self._render(t_cap - self._t0)
        jpeg = None
        if self._values["encode_jpeg"]:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpeg = buf.tobytes() if ok else None
        return Frame(image=img, seq=self._next_seq(), t_cap=t_cap, source_id=self.source_id, jpeg=jpeg,
                     meta={"pattern": self._values["pattern"]})

    def _render(self, t: float) -> np.ndarray:
        w, h = (int(x) for x in str(self._values["resolution"]).split("x"))
        pattern = self._values["pattern"]
        if pattern == "bars":
            colors = np.array([[255, 255, 255], [0, 255, 255], [255, 255, 0], [0, 255, 0],
                               [255, 0, 255], [0, 0, 255], [255, 0, 0], [0, 0, 0]], np.uint8)
            idx = (np.arange(w) * len(colors) // w)
            img = np.repeat(colors[idx][None, :, :], h, axis=0)
        elif pattern == "checker":
            size = max(8, w // 16)
            yy, xx = np.mgrid[0:h, 0:w]
            shift = int(t * 20) % (2 * size)
            board = (((xx + shift) // size + yy // size) % 2).astype(np.uint8) * 255
            img = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
        elif pattern == "noise":
            img = self._rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        elif pattern == "shapes":
            img = np.full((h, w, 3), 40, np.uint8)
            cv2.circle(img, (w // 4, h // 2), h // 6, (0, 0, 255), -1)
            cv2.rectangle(img, (w // 2 - h // 8, h // 2 - h // 8), (w // 2 + h // 8, h // 2 + h // 8), (0, 255, 0), -1)
            pts = np.array([[3 * w // 4, h // 3], [3 * w // 4 - h // 6, 2 * h // 3], [3 * w // 4 + h // 6, 2 * h // 3]])
            cv2.fillPoly(img, [pts.astype(np.int32)], (255, 0, 0))
        else:  # moving_box
            img = np.full((h, w, 3), 90, np.uint8)
            img[:, :, 0] = np.linspace(60, 160, w, dtype=np.uint8)[None, :]
            bw, bh = w // 5, h // 4
            cx = int((w - bw) * (0.5 + 0.45 * math.sin(t * 1.3)))
            cy = int((h - bh) * (0.5 + 0.45 * math.sin(t * 0.9)))
            cv2.rectangle(img, (cx, cy), (cx + bw, cy + bh), (0, 200, 255), -1)
        brightness = int(self._values["brightness"])
        if brightness:
            img = cv2.add(img, np.full_like(img, abs(brightness))) if brightness > 0 else \
                cv2.subtract(img, np.full_like(img, abs(brightness)))
        noise = int(self._values["noise"])
        if noise:
            n = self._rng.normal(0, noise * 0.5, img.shape)
            img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
        if self._values["gray"]:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.ascontiguousarray(img)

    def controls(self) -> list[ControlSpec]:
        return list(self._SPECS)

    def get_control(self, key: str) -> Any:
        if key not in self._values:
            raise KeyError(key)
        return self._values[key]

    def set_control(self, key: str, value: Any) -> ControlResult:
        spec = next((s for s in self._SPECS if s.key == key), None)
        if spec is None or spec.type == ControlType.ACTION:
            return ControlResult.failed(f"不支援參數 {key}")
        try:
            v = spec.validate(value)
        except ValueError as e:
            return ControlResult.failed(str(e), value)
        self._values[key] = v
        return ControlResult(ok=True, value=v, verified=True if spec.readback else None)

    def run_action(self, key: str, **kwargs: Any) -> ControlResult:
        if key == "reset":
            self._t0 = time.monotonic()
            self._seq = 0
            return ControlResult(ok=True)
        return super().run_action(key, **kwargs)

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(resolutions=list(RESOLUTIONS), actions=["reset"],
                                  provides_jpeg=bool(self._values["encode_jpeg"]), max_fps=float(self._values["fps"]))
