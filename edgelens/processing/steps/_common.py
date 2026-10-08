"""共用參數與座標工具，讓各步驟保持相同的操作方式。"""
from __future__ import annotations

import cv2

from edgelens.core.controls import ControlSpec, ControlType
from edgelens.core.results import Overlay
from edgelens.processing.pipeline import ProcessingStep


def spec(kind, key, default, low=None, high=None, *, options=None, advanced=False, unit=""):
    if options is not None:
        typ = ControlType.ENUM
    elif isinstance(default, bool):
        typ = ControlType.BOOL
    elif isinstance(default, int):
        typ = ControlType.INT
    else:
        typ = ControlType.FLOAT
    return ControlSpec(key, typ, f"step.{kind}.{key}", default=default, min=low, max=high,
                       options=tuple((v, f"step.{kind}.{key}.{v}") for v in options or ()),
                       group="image", advanced=advanced, unit=unit)


class ImageStep(ProcessingStep):
    """名稱鍵可由穩定的 kind 推導，避免複製時漏改。"""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.kind:
            cls.category = cls.kind.split(".")[0]
            cls.display_name_key = f"step.{cls.kind}.name"
            cls.description_key = f"step.{cls.kind}.desc"


def gray(image):
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image


def odd(value, minimum=1):
    return max(minimum, int(value) | 1)


def roi_specs(kind):
    return [spec(kind, k, v, 0 if k in ("x", "y") else 0.1, 100, unit="%")
            for k, v in (("x", 0.0), ("y", 0.0), ("w", 100.0), ("h", 100.0))]


def roi_rect(image, params):
    h, w = image.shape[:2]
    x = min(w - 1, int(w * params["x"] / 100))
    y = min(h - 1, int(h * params["y"] / 100))
    right = min(w, x + max(1, round(w * params["w"] / 100)))
    bottom = min(h, y + max(1, round(h * params["h"] / 100)))
    return x, y, right, bottom


def coordinate_matrix(ctx):
    """目前影像座標 → 原始畫面座標的 3×3 仿射矩陣。"""
    return ctx.matrix.copy()


def transform(ctx, inverse):
    """幾何步驟輸出新影像後呼叫；inverse 把新影像座標對應回這一步的輸入座標。"""
    ctx.push_transform(inverse)


def add_mask(ctx, mask, color="#00FF00"):
    matrix = coordinate_matrix(ctx)
    original = cv2.warpAffine(mask, matrix[:2], ctx.frame.size, flags=cv2.INTER_NEAREST)
    ctx.overlays.append(Overlay("mask", {"mask": original}, color))


def resize_image(image, size, ctx, interpolation=cv2.INTER_AREA):
    w, h = size
    out = cv2.resize(image, (w, h), interpolation=interpolation)
    transform(ctx, [[image.shape[1] / w, 0, 0], [0, image.shape[0] / h, 0], [0, 0, 1]])
    return out


def add_box(ctx, rect, color="#00FF00"):
    x1, y1, x2, y2 = rect
    points = [ctx.to_original(x, y) for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]
    # 旋轉後畫多邊形，避免把傾斜框誤畫成水平框。
    if all(abs(points[i][axis] - points[j][axis]) < 1e-6
           for i, j, axis in ((0, 1, 1), (1, 2, 0), (2, 3, 1), (3, 0, 0))):
        xs, ys = zip(*points, strict=True)
        ctx.overlays.append(Overlay("box", {"box": [min(xs), min(ys), max(xs), max(ys)]}, color))
    else:
        ctx.overlays.append(Overlay("polygon", {"points": points}, color))
