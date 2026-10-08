"""平滑雜訊與加強細節。"""
from __future__ import annotations

import cv2

from edgelens.processing.pipeline import register_step
from edgelens.processing.steps._common import ImageStep, odd, spec


@register_step
class GaussianBlur(ImageStep):
    kind = "filter.gaussian_blur"

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "size", 5, 1, 51), spec(cls.kind, "sigma", 0.0, 0, 20, advanced=True)]

    def apply(self, image, ctx):
        n = odd(self.params["size"])
        return cv2.GaussianBlur(image, (n, n), self.params["sigma"])


@register_step
class MedianBlur(ImageStep):
    kind = "filter.median_blur"

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "size", 5, 1, 31)]

    def apply(self, image, ctx):
        return cv2.medianBlur(image, odd(self.params["size"]))


@register_step
class Bilateral(ImageStep):
    kind = "filter.bilateral"

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "diameter", 5, 1, 25), spec(cls.kind, "sigma_color", 50.0, 1, 255),
                spec(cls.kind, "sigma_space", 50.0, 1, 255, advanced=True)]

    def apply(self, image, ctx):
        p = self.params
        return cv2.bilateralFilter(image, p["diameter"], p["sigma_color"], p["sigma_space"])


@register_step
class Sharpen(ImageStep):
    kind = "filter.sharpen"

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "amount", 1.0, 0, 5), spec(cls.kind, "sigma", 1.0, 0.1, 5, advanced=True)]

    def apply(self, image, ctx):
        blur = cv2.GaussianBlur(image, (0, 0), self.params["sigma"])
        amount = self.params["amount"]
        return cv2.addWeighted(image, 1 + amount, blur, -amount, 0)
