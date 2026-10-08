"""形態學對白色區域的面積、雜點與孔洞影響。"""
from __future__ import annotations

import numpy as np
import pytest

from edgelens.processing.pipeline import StepContext
from edgelens.processing.steps.morph import Close, Dilate, Erode, Gradient, Open


@pytest.mark.parametrize("shape", ["rect", "ellipse", "cross"])
def test_morphology_shapes(shape, make_frame):
    image = np.zeros((21, 21), np.uint8)
    image[6:15, 6:15] = 255
    ctx = StepContext(make_frame(image))
    eroded = Erode(shape=shape).apply(image, ctx)
    dilated = Dilate(shape=shape).apply(image, ctx)
    assert np.count_nonzero(eroded) < 81 < np.count_nonzero(dilated)
    assert np.count_nonzero(Erode(shape=shape, iterations=2).apply(image, ctx)) < np.count_nonzero(eroded)
    gradient = Gradient(shape=shape).apply(image, ctx)
    assert gradient[10, 10] == 0 and gradient[6, 10] == 255
    assert np.array_equal(gradient, dilated - eroded)


def test_open_removes_speck_close_fills_hole(make_frame):
    image = np.zeros((21, 21), np.uint8)
    image[6:15, 6:15] = 255
    image[2, 2] = 255
    ctx = StepContext(make_frame(image))
    out = Open(shape="rect").apply(image, ctx)
    assert out[2, 2] == 0 and np.count_nonzero(out) == 81
    image[10, 10] = 0
    out = Close(shape="rect").apply(image, ctx)
    assert out[10, 10] == 255
