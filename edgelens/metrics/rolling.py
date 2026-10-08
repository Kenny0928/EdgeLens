"""效能滑動視窗；FPS 由取樣時間算出，不以延遲倒數代替。"""
from __future__ import annotations

import math
import statistics
import threading
import time
from collections import deque


class RollingStats:
    def __init__(self, window: int = 300):
        if window < 1:
            raise ValueError("統計視窗必須至少為 1，請重新設定。")
        self.window = window
        self._samples: deque[tuple[float, float]] = deque(maxlen=window)
        self._lock = threading.RLock()

    def add(self, value: float, timestamp: float | None = None) -> None:
        with self._lock:
            stamp = time.monotonic() if timestamp is None else float(timestamp)
            value = float(value)
            if not math.isfinite(value) or not math.isfinite(stamp):
                raise ValueError("統計值或時間不是有效數字，請檢查資料。")
            if self._samples and stamp < self._samples[-1][1]:
                raise ValueError("取樣時間倒退，請先重設統計或依時間順序加入。")
            self._samples.append((value, stamp))

    def summary(self) -> dict[str, float | int] | None:
        with self._lock:
            samples = list(self._samples)
        if not samples:
            return None
        vals = sorted(v for v, _ in samples)
        elapsed = samples[-1][1] - samples[0][1]
        return {"last": samples[-1][0], "mean": statistics.fmean(vals),
                "median": statistics.median(vals), "p95": vals[round(.95 * (len(vals) - 1))],
                "min": vals[0], "max": vals[-1], "n": len(vals),
                "fps": (len(vals) - 1) / elapsed if elapsed > 0 else 0.0}

    @property
    def fps(self) -> float:
        return float((self.summary() or {}).get("fps", 0.0))

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
