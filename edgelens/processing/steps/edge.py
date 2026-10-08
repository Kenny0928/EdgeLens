"""把亮度快速變化的位置當成邊緣。"""
from __future__ import annotations

import cv2
import numpy as np

from edgelens.processing.pipeline import register_step
from edgelens.processing.steps._common import ImageStep, add_mask, gray, spec


@register_step
class Canny(ImageStep):
    kind = "edge.canny"

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "low", 50, 0, 255), spec(cls.kind, "high", 150, 0, 255),
                spec(cls.kind, "overlay", False)]

    def apply(self, image, ctx):
        low, high = sorted((self.params["low"], self.params["high"]))
        edges = cv2.Canny(gray(image), low, high)
        if not self.params["overlay"]:
            return edges
        add_mask(ctx, edges)
        return image.copy()


@register_step
class Sobel(ImageStep):
    kind = "edge.sobel"

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "direction", "magnitude", options=("x", "y", "magnitude")),
                spec(cls.kind, "size", 3, options=(1, 3, 5, 7), advanced=True)]

    def apply(self, image, ctx):
        src = gray(image)
        p = self.params
        dx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=p["size"])
        dy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=p["size"])
        value = cv2.magnitude(dx, dy) if p["direction"] == "magnitude" else np.abs(dx if p["direction"] == "x" else dy)
        return np.clip(value, 0, 255).astype(np.uint8)
