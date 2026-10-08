"""比較連續影格；第一張只建立基準，換尺寸與來源時重新開始。"""
from __future__ import annotations

import cv2
import numpy as np

from edgelens.processing.pipeline import register_step
from edgelens.processing.steps._common import ImageStep, gray, spec


class PreviousFrameStep(ImageStep):
    stateful = True

    def __init__(self, **params):
        super().__init__(**params)
        self.reset()

    def reset(self):
        self._previous = None
        self._signature = None

    def difference(self, image, ctx):
        current = gray(image)
        signature = (image.shape, ctx.frame.source_id)
        if self._signature != signature:
            self.reset()
        ready = self._previous is not None
        difference = cv2.absdiff(current, self._previous) if ready else np.zeros_like(current)
        self._previous = current.copy()
        self._signature = signature
        return difference, float(cv2.mean(difference)[0]), ready


@register_step
class FrameDiff(PreviousFrameStep):
    kind = "motion.frame_diff"

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "show_diff", True)]

    def apply(self, image, ctx):
        difference, score, ready = self.difference(image, ctx)
        ctx.data[f"{self.kind}.score"] = score
        ctx.data[f"{self.kind}.ready"] = ready
        return difference if self.params["show_diff"] else image.copy()


@register_step
class BackgroundSub(ImageStep):
    kind = "motion.background_sub"
    stateful = True

    def __init__(self, **params):
        super().__init__(**params)
        self.reset()

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "method", "mog2", options=("mog2", "knn")),
                spec(cls.kind, "learning_rate", -1.0, -1, 1),
                spec(cls.kind, "history", 200, 1, 2000, advanced=True),
                spec(cls.kind, "detect_shadows", False, advanced=True)]

    def reset(self):
        self._model = None
        self._signature = None

    def apply(self, image, ctx):
        p = self.params
        signature = (image.shape, ctx.frame.source_id, p["method"], p["history"], p["detect_shadows"])
        if self._model is None or signature != self._signature:
            self.reset()
            factory = cv2.createBackgroundSubtractorMOG2 if p["method"] == "mog2" else cv2.createBackgroundSubtractorKNN
            self._model = factory(history=p["history"], detectShadows=p["detect_shadows"])
            self._signature = signature
        mask = self._model.apply(image, learningRate=p["learning_rate"])
        ctx.data[f"{self.kind}.foreground_ratio"] = float(np.count_nonzero(mask == 255) / mask.size)
        return mask


@register_step
class StillDetect(PreviousFrameStep):
    kind = "motion.still_detect"

    @classmethod
    def param_specs(cls):
        return [spec(cls.kind, "threshold", 2.0, 0, 255), spec(cls.kind, "frames", 10, 1, 300)]

    def reset(self):
        super().reset()
        self._count = 0
        self._settings = None

    def apply(self, image, ctx):
        settings = (self.params["threshold"], self.params["frames"])
        if self._settings != settings:
            self.reset()
        _, score, ready = self.difference(image, ctx)
        self._settings = settings
        self._count = self._count + 1 if ready and score < self.params["threshold"] else 0
        ctx.data[f"{self.kind}.score"] = score
        ctx.data[f"{self.kind}.count"] = self._count
        ctx.data[f"{self.kind}.still"] = self._count >= self.params["frames"]
        return image.copy()
