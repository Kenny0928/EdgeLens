"""邊緣位置、方向與原圖疊色。"""
from __future__ import annotations

import numpy as np

from edgelens.processing.pipeline import StepContext
from edgelens.processing.steps.edge import Canny, Sobel


def test_canny_detects_rectangle_and_overlay(make_frame):
    image = np.zeros((40, 50), np.uint8)
    image[10:30, 15:35] = 200
    ctx = StepContext(make_frame(image))
    mask = Canny(low=150, high=50).apply(image, ctx)
    assert 60 < np.count_nonzero(mask) < 100 and mask[20, 20] == 0
    out = Canny(overlay=True).apply(image, ctx)
    assert np.array_equal(out, image)
    assert ctx.overlays[0].kind == "mask"
    assert np.array_equal(ctx.overlays[0].data["mask"], mask)


def test_sobel_direction_and_magnitude(make_frame):
    image = np.tile(np.arange(20, dtype=np.uint8) * 5, (20, 1))
    ctx = StepContext(make_frame(image))
    dx = Sobel(direction="x").apply(image, ctx)
    dy = Sobel(direction="y").apply(image, ctx)
    both = Sobel(direction="magnitude").apply(image, ctx)
    assert dx[10, 10] > 0 and not dy.any()
    assert np.array_equal(dx, both)
