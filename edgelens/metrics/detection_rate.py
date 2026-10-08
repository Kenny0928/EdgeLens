"""依時間視窗計算命中率；最高分數平均只納入命中的畫面。"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass

from edgelens.core.results import InferenceResult


@dataclass(frozen=True)
class DetectionSummary:
    total: int
    hits: int
    rate: float | None
    consecutive_hits: int
    last_hit_time: float | None
    mean_best_score: float
    max_score: float

    @property
    def percent(self) -> float | None:
        return None if self.rate is None else 100 * self.rate

    def to_dict(self) -> dict:
        return asdict(self)


class DetectionRate:
    """類別可用數字 ID 或 label 查詢；連續命中限定目前視窗，最後命中時間保留。"""

    def __init__(self, window: float = 5.0, *, threshold: float = 0.0):
        if not math.isfinite(window) or window <= 0 or not 0 <= threshold <= 1:
            raise ValueError("視窗秒數需大於零，分數門檻需為 0–1，請重新設定。")
        self.window, self.threshold = window, threshold
        self._hist: deque = deque()
        self._last_hits: dict = {}
        self._last_time = None
        self._lock = threading.RLock()

    def _prune(self, now):
        while self._hist and self._hist[0][0] < now - self.window:
            self._hist.popleft()

    def add(self, result: InferenceResult, now: float | None = None) -> None:
        with self._lock:
            now = time.monotonic() if now is None else float(now)
            if not math.isfinite(now) or (self._last_time is not None and now < self._last_time):
                raise ValueError("偵測時間不正確，請依時間加入或先重設統計。")
            items = result.detections if result.kind == "detect" else result.top_k
            scores = {}
            for item in items:
                if not math.isfinite(item.score) or item.score < self.threshold:
                    continue
                for key in (None, item.class_id, item.label or item.class_id):
                    scores[key] = max(scores.get(key, 0.0), float(item.score))
                    self._last_hits[key] = now
            self._hist.append((now, scores))
            self._last_time = now
            self._prune(now)

    def summary(self, class_id: str | int | None = None, *, now: float | None = None) -> DetectionSummary:
        with self._lock:
            now = time.monotonic() if now is None else float(now)
            if not math.isfinite(now) or (self._last_time is not None and now < self._last_time):
                raise ValueError("查詢時間早於最新畫面，請使用目前時間。")
            self._prune(now)
            best = [s[class_id] for _, s in self._hist if class_id in s]
            streak = 0
            for _, scores in reversed(self._hist):
                if class_id not in scores:
                    break
                streak += 1
            total = len(self._hist)
            return DetectionSummary(total, len(best), len(best) / total if total else None,
                                    streak, self._last_hits.get(class_id),
                                    sum(best) / len(best) if best else 0.0, max(best, default=0.0))

    def rate(self, now: float | None = None, class_id=None) -> tuple[float | None, int, int]:
        summary = self.summary(class_id, now=now)
        return summary.percent, summary.hits, summary.total

    def reset(self) -> None:
        with self._lock:
            self._hist.clear()
            self._last_hits.clear()
            self._last_time = None
