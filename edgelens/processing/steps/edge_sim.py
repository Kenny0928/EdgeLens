"""教學用相機退化模擬；profile 是示意值，不是感測器的精確校正。"""
from __future__ import annotations

import cv2
import numpy as np

from edgelens.processing.pipeline import register_step
from edgelens.processing.steps._common import ImageStep, bgr, resize_image, spec

RESOLUTIONS = {"qqvga": (160, 120), "qvga": (320, 240), "vga": (640, 480)}
PROFILES = {
    "esp32_cam": {"width": 320, "height": 240, "rgb565": True, "quality": 35, "noise": 3.0, "blur": 0.6},
    "k210": {"width": 320, "height": 240, "rgb565": True, "quality": 70, "noise": 2.0, "blur": 0.4},
}


def quantize565(image):
    src = bgr(image)
    out = np.empty_like(src)
    for channel, shift in enumerate((3, 2, 3)):
        q = src[:, :, channel] >> shift
        # 重複高位元把黑白端點保留下來。
        out[:, :, channel] = (q << shift) | (q >> (8 - 2 * shift))
    return out


def jpeg_roundtrip(image, quality):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG 壓縮失敗，請換一張影像後重試。")
    out = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE if image.ndim == 2 else cv2.IMREAD_COLOR)
    if out is None:
        raise ValueError("無法讀回 JPEG 影像，請換一張影像後重試。")
    return out


def add_noise(image, strength):
    if strength == 0:
        return image.copy()
    noise = np.random.default_rng().normal(0, strength, image.shape)
    return np.clip(np.rint(image.astype(np.float32) + noise), 0, 255).astype(np.uint8)


@register_step
class Resolution(ImageStep):
    kind = "edge_sim.resolution"
    changes_size = True

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "resolution", "qvga", options=(*RESOLUTIONS, "custom")),
                spec(cls.kind, "width", 320, 1, 4096, advanced=True),
                spec(cls.kind, "height", 240, 1, 4096, advanced=True), spec(cls.kind, "upscale", True)]

    def apply(self, image, ctx):
        p = self.params
        size = RESOLUTIONS.get(p["resolution"], (p["width"], p["height"]))
        if p["upscale"]:
            low = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            return cv2.resize(low, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        return resize_image(image, size, ctx)


@register_step
class RGB565(ImageStep):
    kind = "edge_sim.rgb565"
    accepts_gray = False

    def apply(self, image, ctx):
        return quantize565(image)


@register_step
class JPEG(ImageStep):
    kind = "edge_sim.jpeg"

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "quality", 50, 1, 100)]

    def apply(self, image, ctx):
        return jpeg_roundtrip(image, self.params["quality"])


@register_step
class Noise(ImageStep):
    kind = "edge_sim.noise"

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "strength", 5.0, 0, 100)]

    def apply(self, image, ctx):
        return add_noise(image, self.params["strength"])


@register_step
class Sensor(ImageStep):
    kind = "edge_sim.sensor"
    changes_size = True
    accepts_gray = False

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "profile", "esp32_cam", options=("esp32_cam", "k210", "custom")),
                spec(cls.kind, "upscale", True),
                spec(cls.kind, "width", 320, 1, 4096, advanced=True),
                spec(cls.kind, "height", 240, 1, 4096, advanced=True),
                spec(cls.kind, "rgb565", True, advanced=True), spec(cls.kind, "quality", 50, 1, 100, advanced=True),
                spec(cls.kind, "noise", 3.0, 0, 100, advanced=True), spec(cls.kind, "blur", 0.5, 0, 5, advanced=True)]

    def apply(self, image, ctx):
        p = PROFILES.get(self.params["profile"], self.params)
        out = cv2.resize(bgr(image), (p["width"], p["height"]), interpolation=cv2.INTER_AREA)
        if p["rgb565"]:
            out = quantize565(out)
        out = jpeg_roundtrip(out, p["quality"])
        out = add_noise(out, p["noise"])
        if p["blur"] > 0:
            out = cv2.GaussianBlur(out, (0, 0), p["blur"])
        if self.params["upscale"]:
            return cv2.resize(out, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        # 影像已縮小，只更新一次座標；不用再做一次影像插值。
        from edgelens.processing.steps._common import transform

        transform(ctx, [[image.shape[1] / p["width"], 0, 0],
                        [0, image.shape[0] / p["height"], 0], [0, 0, 1]])
        return out
