"""黑白分界與反相行為。"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from edgelens.processing.pipeline import StepContext
from edgelens.processing.steps.threshold import Adaptive, Binary, Otsu


def test_binary_boundary_and_inverse(make_frame):
    image = np.array([[0, 126, 127, 128, 255]], np.uint8)
    ctx = StepContext(make_frame(image))
    out = Binary(threshold=127).apply(image, ctx)
    assert out.tolist() == [[0, 0, 0, 255, 255]]
    assert np.array_equal(Binary(threshold=127, invert=True).apply(image, ctx), 255 - out)


@pytest.mark.parametrize("method", ["mean", "gaussian"])
def test_adaptive_local_threshold(method, make_frame):
    image = np.tile(np.arange(30, dtype=np.uint8) * 6, (30, 1))
    image[10:20, 10:20] = 0
    ctx = StepContext(make_frame(image))
    step = Adaptive(method=method, block_size=10, c=2)
    out = step.apply(image, ctx)
    assert set(np.unique(out)) == {0, 255}
    assert out[10, 10] == 0 and out[5, 15] == 255
    step.set_param("invert", True)
    assert np.array_equal(step.apply(image, ctx), 255 - out)


def test_otsu_splits_bimodal_color_image(make_frame):
    image = np.full((20, 30), 40, np.uint8)
    image[:, 15:] = 200
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ctx = StepContext(make_frame(image))
    out = Otsu().apply(image, ctx)
    assert not out[:, :15].any() and np.all(out[:, 15:] == 255)
    assert ctx.data["threshold.otsu.threshold"] == 40
    assert np.array_equal(Otsu(invert=True).apply(image, ctx), 255 - out)
