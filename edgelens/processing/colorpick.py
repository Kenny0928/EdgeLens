"""吸管取色；H 用圓形平均與最短色相弧，避免紅色被平均成青色。

範圍皆含上下界。HSV 的 H 下界大於上界表示跨 0；margin 使用 OpenCV
各通道的數值單位，套用到 BGR、HSV、8-bit LAB 後才轉出板端 LAB。
"""
from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import cv2
import numpy as np

Triple = tuple[int, int, int]
FloatTriple = tuple[float, float, float]


def _lab_board(values):
    return float(values[0]) * 100 / 255, float(values[1]) - 128, float(values[2]) - 128


def _hue_arc(hues, margin=0):
    values = np.unique(np.asarray(hues, dtype=int))
    gaps = np.diff(np.r_[values, values[0] + 180])
    cut = int(np.argmax(gaps))
    start, end = int(values[(cut + 1) % len(values)]), int(values[cut])
    width = (end - start) % 180
    if width + 2 * margin >= 179:
        return 0, 179
    return (start - margin) % 180, (end + margin) % 180


@dataclass(frozen=True)
class ColorSample:
    """平均值及區域內實際最小／最大值；LAB 同時提供兩種表示。"""

    bgr: FloatTriple
    hsv: FloatTriple
    lab: FloatTriple
    maixpy_lab: FloatTriple
    bgr_min: Triple
    bgr_max: Triple
    hsv_min: Triple
    hsv_max: Triple
    lab_min: Triple
    lab_max: Triple
    maixpy_lab_min: FloatTriple
    maixpy_lab_max: FloatTriple
    hues: tuple[int, ...]
    bounds: tuple[int, int, int, int]
    pixel_count: int


@dataclass(frozen=True)
class ColorRanges:
    """OpenCV 範圍；HSV 允許跨 0，其餘通道下界不大於上界。"""

    bgr_lower: Triple
    bgr_upper: Triple
    hsv_lower: Triple
    hsv_upper: Triple
    lab_lower: Triple
    lab_upper: Triple

    @property
    def maixpy_lab_lower(self) -> FloatTriple:
        return _lab_board(self.lab_lower)

    @property
    def maixpy_lab_upper(self) -> FloatTriple:
        return _lab_board(self.lab_upper)


def pick(image_bgr: np.ndarray, x: int, y: int, radius: int = 5) -> ColorSample:
    """取邊長 2r+1 的方形區域，靠邊時截在影像內；座標需落在圖中。"""
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8
            or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.size == 0):
        raise ValueError("取色需要 uint8 彩色影像，請先載入有效的 BGR 畫面。")
    try:
        x, y, radius = operator.index(x), operator.index(y), operator.index(radius)
    except TypeError as exc:
        raise ValueError("取色座標與半徑要是整數，請重新選取像素。") from exc
    h, w = image_bgr.shape[:2]
    if radius < 0 or not (0 <= x < w and 0 <= y < h):
        raise ValueError("取色位置超出畫面或半徑小於 0，請在畫面內重新選取。")
    bounds = (max(0, x - radius), max(0, y - radius), min(w, x + radius + 1), min(h, y + radius + 1))
    x1, y1, x2, y2 = bounds
    region = np.ascontiguousarray(image_bgr[y1:y2, x1:x2])
    spaces = (region, cv2.cvtColor(region, cv2.COLOR_BGR2HSV), cv2.cvtColor(region, cv2.COLOR_BGR2LAB))
    means, lows, highs = [], [], []
    for space in spaces:
        pixels = space.reshape(-1, 3)
        means.append(tuple(float(v) for v in pixels.mean(axis=0)))
        lows.append(tuple(int(v) for v in pixels.min(axis=0)))
        highs.append(tuple(int(v) for v in pixels.max(axis=0)))
    hues = tuple(int(v) for v in np.unique(spaces[1][:, :, 0]))
    angles = spaces[1][:, :, 0].astype(float) * (2 * np.pi / 180)
    sine, cosine = float(np.sin(angles).mean()), float(np.cos(angles).mean())
    # 均勻色相沒有唯一平均方向，採用 0 作為穩定的顯示值。
    hue_mean = (math.atan2(sine, cosine) * 180 / (2 * np.pi)) % 180 if math.hypot(sine, cosine) > 1e-8 else 0.0
    means[1] = (hue_mean, *means[1][1:])
    h_min, h_max = _hue_arc(hues)
    lows[1], highs[1] = (h_min, *lows[1][1:]), (h_max, *highs[1][1:])
    return ColorSample(means[0], means[1], means[2], _lab_board(means[2]),
                       lows[0], highs[0], lows[1], highs[1], lows[2], highs[2],
                       _lab_board(lows[2]), _lab_board(highs[2]), hues, bounds, region.shape[0] * region.shape[1])


def ranges_from_samples(samples, margin: float = 5) -> ColorRanges:
    """合併區域內所有像素的範圍；留邊不會把跨 0 的紅色擴成整圈。"""
    samples = list(samples)
    if not samples or not all(isinstance(s, ColorSample) for s in samples):
        raise ValueError("尚未取得有效色彩樣本，請先用吸管取色。")
    try:
        margin = float(margin)
    except (ValueError, TypeError) as exc:
        raise ValueError("範圍留邊要是非負數字，請重新輸入。") from exc
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("範圍留邊要是有限的非負數字，請重新輸入。")
    margin = math.ceil(margin)
    ranges = []
    for space in ("bgr", "hsv", "lab"):
        lo = np.min([getattr(s, f"{space}_min") for s in samples], axis=0).astype(float)
        hi = np.max([getattr(s, f"{space}_max") for s in samples], axis=0).astype(float)
        lo = np.clip(lo - margin, 0, 255).astype(int).tolist()
        hi = np.clip(hi + margin, 0, 255).astype(int).tolist()
        if space == "hsv":
            lo[0], hi[0] = _hue_arc([h for sample in samples for h in sample.hues], margin)
        ranges.extend((tuple(lo), tuple(hi)))
    return ColorRanges(*ranges)


def to_maixpy_lab_threshold(ranges: ColorRanges) -> tuple[int, int, int, int, int, int]:
    """向外取整，避免板端整數門檻排除原本取到的顏色。"""
    lo, hi = ranges.maixpy_lab_lower, ranges.maixpy_lab_upper
    return (math.floor(lo[0]), math.ceil(hi[0]), math.floor(lo[1]), math.ceil(hi[1]),
            math.floor(lo[2]), math.ceil(hi[2]))


def to_opencv_hsv(ranges: ColorRanges) -> list[tuple[Triple, Triple]]:
    lo, hi = ranges.hsv_lower, ranges.hsv_upper
    if lo[0] <= hi[0]:
        return [(lo, hi)]
    return [(lo, (179, hi[1], hi[2])), ((0, lo[1], lo[2]), hi)]


def to_hsv_mask_params(ranges: ColorRanges) -> dict[str, int]:
    return {f"{channel}_{bound}": int(value)
            for bound, values in (("min", ranges.hsv_lower), ("max", ranges.hsv_upper))
            for channel, value in zip("hsv", values, strict=True)}
