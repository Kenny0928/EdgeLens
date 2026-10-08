"""依亮度把畫面分成黑白，方便尋找形狀。"""
from __future__ import annotations

import cv2

from edgelens.processing.pipeline import register_step
from edgelens.processing.steps._common import ImageStep, gray, odd, spec


@register_step
class Binary(ImageStep):
    kind = "threshold.binary"

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "threshold", 127, 0, 255), spec(cls.kind, "invert", False)]

    def apply(self, image, ctx):
        mode = cv2.THRESH_BINARY_INV if self.params["invert"] else cv2.THRESH_BINARY
        return cv2.threshold(gray(image), self.params["threshold"], 255, mode)[1]


@register_step
class Adaptive(ImageStep):
    kind = "threshold.adaptive"

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "method", "gaussian", options=("mean", "gaussian")),
                spec(cls.kind, "block_size", 11, 3, 101), spec(cls.kind, "c", 2.0, -100, 100),
                spec(cls.kind, "invert", False, advanced=True)]

    def apply(self, image, ctx):
        p = self.params
        method = cv2.ADAPTIVE_THRESH_MEAN_C if p["method"] == "mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        mode = cv2.THRESH_BINARY_INV if p["invert"] else cv2.THRESH_BINARY
        return cv2.adaptiveThreshold(gray(image), 255, method, mode, odd(p["block_size"], 3), p["c"])


@register_step
class Otsu(ImageStep):
    kind = "threshold.otsu"

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "invert", False)]

    def apply(self, image, ctx):
        mode = cv2.THRESH_BINARY_INV if self.params["invert"] else cv2.THRESH_BINARY
        value, out = cv2.threshold(gray(image), 0, 255, mode | cv2.THRESH_OTSU)
        ctx.data[f"{self.kind}.threshold"] = float(value)
        return out
