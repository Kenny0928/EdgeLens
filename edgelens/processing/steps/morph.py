"""用小型核整理遮罩；各操作共用核形狀與次數。"""
from __future__ import annotations

import cv2

from edgelens.processing.pipeline import register_step
from edgelens.processing.steps._common import ImageStep, odd, spec


class MorphStep(ImageStep):
    operation = cv2.MORPH_ERODE

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "shape", "ellipse", options=("rect", "ellipse", "cross")),
                spec(cls.kind, "size", 3, 1, 51), spec(cls.kind, "iterations", 1, 1, 20, advanced=True)]

    def apply(self, image, ctx):
        shape = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}[self.params["shape"]]
        n = odd(self.params["size"])
        kernel = cv2.getStructuringElement(shape, (n, n))
        return cv2.morphologyEx(image, self.operation, kernel, iterations=self.params["iterations"])


@register_step
class Erode(MorphStep):
    kind = "morph.erode"
    operation = cv2.MORPH_ERODE


@register_step
class Dilate(MorphStep):
    kind = "morph.dilate"
    operation = cv2.MORPH_DILATE


@register_step
class Open(MorphStep):
    kind = "morph.open"
    operation = cv2.MORPH_OPEN


@register_step
class Close(MorphStep):
    kind = "morph.close"
    operation = cv2.MORPH_CLOSE


@register_step
class Gradient(MorphStep):
    kind = "morph.gradient"
    operation = cv2.MORPH_GRADIENT
