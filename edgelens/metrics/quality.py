"""用固定尺度比較清晰度，亮度與曝光比例保留原圖的像素分布。"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityReport:
    sharpness: float
    brightness: float
    contrast: float
    overexposed: float
    underexposed: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class QualityThresholds:
    min_sharpness: float = 50.0
    min_brightness: float = 30.0
    max_brightness: float = 225.0
    min_contrast: float = 10.0
    max_overexposed: float = 0.2
    max_underexposed: float = 0.2

    def __post_init__(self):
        values = asdict(self)
        if not all(np.isfinite(v) and v >= 0 for v in values.values()):
            raise ValueError("品質門檻必須是非負數，請重新設定。")
        if (self.min_brightness > self.max_brightness or self.max_brightness > 255
                or self.max_overexposed > 1 or self.max_underexposed > 1):
            raise ValueError("亮度範圍應為 0–255，曝光比例應為 0–1，請重新設定。")


def gray_image(image: np.ndarray) -> np.ndarray:
    """先驗證輸入，避免空畫面造成 OpenCV 難懂的錯誤。"""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("沒有可用的影像，請提供非空的 uint8 畫面。")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("影像格式不符，請使用灰階或 BGR 三色畫面。")


def measure(image: np.ndarray, *, width: int = 320) -> QualityReport:
    gray = gray_image(image)
    if width <= 0:
        raise ValueError("計算寬度必須大於零，請重新設定。")
    h, w = gray.shape
    small = cv2.resize(gray, (width, max(1, round(h * width / w))),
                       interpolation=cv2.INTER_AREA if w >= width else cv2.INTER_LINEAR)
    mean, std = cv2.meanStdDev(gray)
    lap = cv2.Laplacian(small, cv2.CV_32F)
    return QualityReport(float(lap.var()), float(mean[0, 0]), float(std[0, 0]),
                         float(np.count_nonzero(gray >= 250) / gray.size),
                         float(np.count_nonzero(gray <= 5) / gray.size))


def verdict(report: QualityReport, thresholds: QualityThresholds | None = None) -> tuple[bool, list[str]]:
    t = thresholds or QualityThresholds()
    checks = ((report.sharpness < t.min_sharpness, "dataset.quality.blurry"),
              (report.brightness < t.min_brightness, "dataset.quality.dark"),
              (report.brightness > t.max_brightness, "dataset.quality.bright"),
              (report.contrast < t.min_contrast, "dataset.quality.low_contrast"),
              (report.overexposed > t.max_overexposed, "dataset.quality.overexposed"),
              (report.underexposed > t.max_underexposed, "dataset.quality.underexposed"))
    reasons = [key for failed, key in checks if failed]
    return not reasons, reasons
