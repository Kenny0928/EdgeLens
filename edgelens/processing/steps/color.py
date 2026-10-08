"""顏色篩選與亮度調整；遮罩保留白色代表的區域。"""
from __future__ import annotations

import cv2
import numpy as np

from edgelens.processing.pipeline import register_step
from edgelens.processing.steps._common import ImageStep, bgr, gray, spec


def mask_output(image, mask, params):
    mode = params["output"]
    if mode == "mask":
        return mask
    if mode == "keep":
        return cv2.bitwise_and(image, image, mask=mask)
    out = bgr(image).copy()
    tint = np.empty_like(out)
    tint[:] = (0, 255, 0)
    blended = cv2.addWeighted(out, 1 - params["alpha"], tint, params["alpha"], 0)
    out[mask != 0] = blended[mask != 0]
    return out


def mask_specs(kind, channels):
    specs = []
    for channel, maximum in channels:
        specs.extend([spec(kind, f"{channel}_min", 0, 0, maximum),
                      spec(kind, f"{channel}_max", maximum, 0, maximum)])
    return specs + [spec(kind, "output", "mask", options=("mask", "keep", "overlay")),
                    spec(kind, "alpha", 0.4, 0, 1, advanced=True)]


@register_step
class Grayscale(ImageStep):
    kind = "color.grayscale"

    def apply(self, image, ctx):
        return gray(image).copy()


@register_step
class HSVMask(ImageStep):
    kind = "color.hsv_mask"
    accepts_gray = False

    @classmethod
    def param_specs(cls):
        return mask_specs(cls.kind, (("h", 179), ("s", 255), ("v", 255)))

    def apply(self, image, ctx):
        p = self.params
        hsv = cv2.cvtColor(bgr(image), cv2.COLOR_BGR2HSV)
        lo, hi = tuple(p[f"{k}_min"] for k in "hsv"), tuple(p[f"{k}_max"] for k in "hsv")
        if lo[0] > hi[0]:
            mask = cv2.bitwise_or(cv2.inRange(hsv, lo, (179, hi[1], hi[2])),
                                  cv2.inRange(hsv, (0, lo[1], lo[2]), hi))
        else:
            mask = cv2.inRange(hsv, lo, hi)
        return mask_output(image, mask, p)


@register_step
class LABMask(ImageStep):
    kind = "color.lab_mask"
    accepts_gray = False

    @classmethod
    def param_specs(cls):
        return mask_specs(cls.kind, (("l", 255), ("a", 255), ("b", 255)))

    def apply(self, image, ctx):
        lab = cv2.cvtColor(bgr(image), cv2.COLOR_BGR2LAB)
        lo = tuple(self.params[f"{k}_min"] for k in "lab")
        hi = tuple(self.params[f"{k}_max"] for k in "lab")
        return mask_output(image, cv2.inRange(lab, lo, hi), self.params)


@register_step
class BrightnessContrast(ImageStep):
    kind = "color.brightness_contrast"

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "brightness", 0, -255, 255), spec(cls.kind, "contrast", 1.0, 0, 4)]

    def apply(self, image, ctx):
        # 不使用 convertScaleAbs，負亮度應截成黑色而不是反彈變亮。
        table = np.clip(np.arange(256) * self.params["contrast"] + self.params["brightness"], 0, 255).astype(np.uint8)
        return cv2.LUT(image, table)


@register_step
class Gamma(ImageStep):
    kind = "color.gamma"

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "gamma", 1.0, 0.1, 5)]

    def apply(self, image, ctx):
        table = np.round(255 * (np.arange(256) / 255) ** (1 / self.params["gamma"])).astype(np.uint8)
        return cv2.LUT(image, table)


@register_step
class CLAHE(ImageStep):
    kind = "color.clahe"

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "clip_limit", 2.0, 0.1, 40),
                spec(cls.kind, "grid_size", 8, 1, 32, advanced=True)]

    def apply(self, image, ctx):
        n = self.params["grid_size"]
        clahe = cv2.createCLAHE(clipLimit=self.params["clip_limit"], tileGridSize=(n, n))
        if image.ndim == 2:
            return clahe.apply(image)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


@register_step
class Invert(ImageStep):
    kind = "color.invert"

    def apply(self, image, ctx):
        return cv2.bitwise_not(image)


@register_step
class Channel(ImageStep):
    kind = "color.channel"
    accepts_gray = False

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "channel", "b", options=("b", "g", "r", "h", "s", "v", "lab_l", "lab_a", "lab_b"))]

    def apply(self, image, ctx):
        ch = self.params["channel"]
        src = bgr(image)
        if ch in ("b", "g", "r"):
            return src[:, :, "bgr".index(ch)].copy()
        if ch in ("h", "s", "v"):
            return cv2.cvtColor(src, cv2.COLOR_BGR2HSV)[:, :, "hsv".index(ch)].copy()
        return cv2.cvtColor(src, cv2.COLOR_BGR2LAB)[:, :, ("lab_l", "lab_a", "lab_b").index(ch)].copy()
