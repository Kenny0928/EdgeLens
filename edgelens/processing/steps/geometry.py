"""裁切、旋轉與縮放；後續標記一律換算回原圖。"""
from __future__ import annotations

import math

import cv2
import numpy as np

from edgelens.processing.pipeline import register_step
from edgelens.processing.steps._common import (
    ImageStep,
    add_box,
    resize_image,
    roi_rect,
    roi_specs,
    spec,
    transform,
)


@register_step
class Crop(ImageStep):
    kind = "geometry.crop"
    changes_size = True

    @classmethod
    def param_specs(cls):
        return roi_specs(cls.kind)

    def apply(self, image, ctx):
        x, y, right, bottom = roi_rect(image, self.params)
        out = image[y:bottom, x:right].copy()
        transform(ctx, [[1, 0, x], [0, 1, y], [0, 0, 1]])
        return out


@register_step
class ROI(Crop):
    kind = "geometry.roi"

    @classmethod
    def param_specs(cls):
        return roi_specs(cls.kind) + [spec(cls.kind, "keep_only", False)]

    def apply(self, image, ctx):
        rect = roi_rect(image, self.params)
        add_box(ctx, rect)
        ctx.data[f"{self.kind}.points"] = [ctx.to_original(rect[0], rect[1]),
                                           ctx.to_original(rect[2], rect[3])]
        return super().apply(image, ctx) if self.params["keep_only"] else image.copy()


@register_step
class Rotate(ImageStep):
    kind = "geometry.rotate"
    changes_size = True

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "mode", "0", options=("0", "90", "180", "270", "custom")),
                spec(cls.kind, "angle", 0.0, -360, 360, advanced=True, unit="°"),
                spec(cls.kind, "expand", True, advanced=True)]

    def apply(self, image, ctx):
        h, w = image.shape[:2]
        angle = self.params["angle"] if self.params["mode"] == "custom" else float(self.params["mode"])
        angle %= 360
        if angle in (0, 90, 180, 270) and (self.params["expand"] or angle in (0, 180)):
            k = int(angle / 90)
            inverse = (np.eye(3), [[0, -1, w - 1], [1, 0, 0], [0, 0, 1]],
                       [[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]],
                       [[0, 1, 0], [-1, 0, h - 1], [0, 0, 1]])[k]
            out = np.rot90(image, k).copy()
        else:
            mat = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1)
            nw, nh = w, h
            if self.params["expand"]:
                c, s = abs(mat[0, 0]), abs(mat[0, 1])
                nw, nh = math.ceil(w * c + h * s), math.ceil(w * s + h * c)
                mat[:, 2] += ((nw - w) / 2, (nh - h) / 2)
            out = cv2.warpAffine(image, mat, (nw, nh))
            inverse = np.vstack((cv2.invertAffineTransform(mat), [0, 0, 1]))
        transform(ctx, inverse)
        return out


@register_step
class Flip(ImageStep):
    kind = "geometry.flip"

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "direction", "horizontal", options=("horizontal", "vertical", "both"))]

    def apply(self, image, ctx):
        direction = self.params["direction"]
        sx = -1 if direction in ("horizontal", "both") else 1
        sy = -1 if direction in ("vertical", "both") else 1
        out = cv2.flip(image, {"horizontal": 1, "vertical": 0, "both": -1}[direction])
        transform(ctx, [[sx, 0, image.shape[1] - 1 if sx < 0 else 0],
                        [0, sy, image.shape[0] - 1 if sy < 0 else 0], [0, 0, 1]])
        return out


@register_step
class Resize(ImageStep):
    kind = "geometry.resize"
    changes_size = True

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "mode", "scale", options=("scale", "size")),
                spec(cls.kind, "scale", 1.0, 0.01, 4),
                spec(cls.kind, "width", 320, 1, 4096), spec(cls.kind, "height", 240, 1, 4096),
                spec(cls.kind, "interpolation", "area", options=("nearest", "linear", "area", "cubic"), advanced=True)]

    def apply(self, image, ctx):
        p = self.params
        size = (p["width"], p["height"])
        if p["mode"] == "scale":
            size = (max(1, round(image.shape[1] * p["scale"])), max(1, round(image.shape[0] * p["scale"])))
        methods = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR,
                   "area": cv2.INTER_AREA, "cubic": cv2.INTER_CUBIC}
        return resize_image(image, size, ctx, methods[p["interpolation"]])
