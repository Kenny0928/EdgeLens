"""分析遮罩並輸出數值；標記不直接寫進影像。"""
from __future__ import annotations

import cv2
import numpy as np

from edgelens.core.results import Overlay
from edgelens.processing.pipeline import register_step
from edgelens.processing.steps._common import ImageStep, add_box, gray, roi_rect, roi_specs, spec


def find_contours(image):
    # findContours 本來就把非零值視為前景，不需再建立一張遮罩。
    return cv2.findContours(gray(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


@register_step
class Blobs(ImageStep):
    kind = "analyze.blobs"

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "min_area", 20.0, 0, 1e9), spec(cls.kind, "max_area", 1e9, 0, 1e9),
                spec(cls.kind, "draw_boxes", True), spec(cls.kind, "draw_centers", True)]

    def apply(self, image, ctx):
        found = []
        for contour in find_contours(image):
            area = float(cv2.contourArea(contour))
            if not self.params["min_area"] <= area <= self.params["max_area"]:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            m = cv2.moments(contour)
            center = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else (x + (w - 1) / 2, y + (h - 1) / 2)
            original_center = ctx.to_original(*center)
            found.append((area, original_center))
            if self.params["draw_boxes"]:
                add_box(ctx, (x, y, x + w, y + h))
            if self.params["draw_centers"]:
                ctx.overlays.append(Overlay("point", {"xy": original_center, "radius": 4}))
        found.sort(key=lambda v: (-v[0], v[1]))
        ctx.data[f"{self.kind}.count"] = len(found)
        # 面積以目前遮罩的像素計算，中心點則供 GUI 使用原圖座標。
        ctx.data[f"{self.kind}.max_area"] = found[0][0] if found else 0.0
        ctx.data[f"{self.kind}.centers"] = [center for _, center in found]
        return image.copy()


@register_step
class Contours(ImageStep):
    kind = "analyze.contours"

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "approximate", False), spec(cls.kind, "epsilon", 0.02, 0, 0.2, advanced=True),
                spec(cls.kind, "min_area", 20.0, 0, 1e9, advanced=True)]

    def apply(self, image, ctx):
        count = 0
        for contour in find_contours(image):
            if cv2.contourArea(contour) < self.params["min_area"]:
                continue
            if self.params["approximate"]:
                contour = cv2.approxPolyDP(contour, self.params["epsilon"] * cv2.arcLength(contour, True), True)
            points = [ctx.to_original(float(x), float(y)) for x, y in contour.reshape(-1, 2)]
            if points:
                ctx.overlays.append(Overlay("polygon", {"points": points}))
                count += 1
        ctx.data[f"{self.kind}.count"] = count
        return image.copy()


@register_step
class HistogramStats(ImageStep):
    kind = "analyze.histogram_stats"

    @classmethod
    def param_specs(cls):
        return roi_specs(cls.kind)

    def apply(self, image, ctx):
        x, y, right, bottom = roi_rect(image, self.params)
        region = gray(image)[y:bottom, x:right]
        mean, std = cv2.meanStdDev(region)
        ctx.data[f"{self.kind}.mean"] = float(mean[0, 0])
        ctx.data[f"{self.kind}.std"] = float(std[0, 0])
        return np.copy(image)
