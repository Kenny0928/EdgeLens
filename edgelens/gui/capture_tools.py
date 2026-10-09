"""拍照面板用的非 GUI 邏輯：靜止偵測、編號、品質門檻。可以單獨測試。"""

from __future__ import annotations

import uuid

import cv2
import numpy as np

from edgelens.metrics.quality import QualityThresholds


def new_id() -> str:
    """連拍（burst）與多相機同拍（group）用的短識別碼。"""
    return uuid.uuid4().hex[:12]


def small_gray(image: np.ndarray, size: tuple[int, int] = (64, 48)) -> np.ndarray:
    """縮小成灰階小圖來比較畫面變化，省時間也能濾掉感光雜訊。"""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


class StillDetector:
    """畫面靜止偵測：連續 frames 張的平均差異都低於 threshold 才算靜止並觸發一次。

    觸發後必須先看到畫面動過（差異超過門檻），才會再次觸發，
    避免相機對著同一個畫面一直拍。第一次靜止也會觸發。
    """

    def __init__(self, threshold: float = 2.0, frames: int = 8):
        self.threshold = float(threshold)
        self.frames = max(1, int(frames))
        self.reset()

    def reset(self) -> None:
        self._prev: np.ndarray | None = None
        self._still = 0
        self._armed = True
        self.last_diff = 0.0

    def configure(self, threshold: float, frames: int) -> None:
        self.threshold = float(threshold)
        self.frames = max(1, int(frames))

    @property
    def still_count(self) -> int:
        return self._still

    def update(self, image: np.ndarray) -> bool:
        """餵入一張畫面；回傳 True 表示「現在該拍一張」。"""
        small = small_gray(image)
        prev, self._prev = self._prev, small
        if prev is None:
            return False
        self.last_diff = float(cv2.absdiff(small, prev).mean())
        if self.last_diff >= self.threshold:
            self._still = 0
            self._armed = True
            return False
        self._still += 1
        if self._armed and self._still >= self.frames:
            self._armed = False
            return True
        return False


def thresholds_from_settings(settings) -> QualityThresholds:
    """讀 capture.quality.*；數值不合理（例如手動改壞設定檔）就退回預設值。"""
    d = QualityThresholds()
    keys = ("min_sharpness", "min_brightness", "max_brightness", "min_contrast", "max_overexposed",
            "max_underexposed")
    values = {}
    for k in keys:
        v = settings.get(f"capture.quality.{k}", getattr(d, k))
        try:
            values[k] = float(v)
        except (TypeError, ValueError):
            values[k] = getattr(d, k)
    try:
        return QualityThresholds(**values)
    except ValueError:
        return d
