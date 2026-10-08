"""形狀數量、數值統計與原圖標記。"""
from __future__ import annotations

import numpy as np
import pytest

from edgelens.processing.pipeline import StepContext
from edgelens.processing.steps.analyze import Blobs, Contours, HistogramStats
from edgelens.processing.steps.geometry import Crop, Resize, Rotate


def test_blobs_count_area_centers_and_filter(make_frame):
    image = np.zeros((100, 120), np.uint8)
    image[10:21, 20:31] = 255
    image[40:61, 60:81] = 255
    ctx = StepContext(make_frame(image))
    out = Blobs().apply(image, ctx)
    assert np.array_equal(out, image)
    assert ctx.data["analyze.blobs.count"] == 2
    assert ctx.data["analyze.blobs.max_area"] == 400
    assert ctx.data["analyze.blobs.centers"] == [(70, 50), (25, 15)]
    assert len(ctx.overlays) == 4
    ctx = StepContext(make_frame(image))
    Blobs(min_area=200, max_area=500, draw_boxes=False, draw_centers=False).apply(image, ctx)
    assert ctx.data["analyze.blobs.count"] == 1 and not ctx.overlays
    Blobs(min_area=600).apply(image, ctx)
    assert ctx.data["analyze.blobs.count"] == 0 and ctx.data["analyze.blobs.max_area"] == 0


def test_blobs_after_rotation_crop_resize(make_frame):
    image = np.zeros((80, 100), np.uint8)
    image[20:31, 40:51] = 255
    ctx = StepContext(make_frame(image))
    out = Crop(x=20, y=0, w=60, h=100).apply(image, ctx)
    out = Resize(scale=2, interpolation="nearest").apply(out, ctx)
    out = Rotate(mode="90").apply(out, ctx)
    Blobs().apply(out, ctx)
    assert ctx.data["analyze.blobs.centers"][0] == pytest.approx((45.25, 25.25))
    assert len(ctx.overlays) == 2


@pytest.mark.parametrize("approximate", [False, True])
def test_contours_rectangle(approximate, make_frame):
    image = np.zeros((40, 40), np.uint8)
    image[10:31, 10:31] = 255
    ctx = StepContext(make_frame(image))
    out = Contours(approximate=approximate).apply(image, ctx)
    assert np.array_equal(out, image)
    assert ctx.data["analyze.contours.count"] == 1
    assert set(ctx.overlays[0].data["points"]) == {(10, 10), (10, 30), (30, 10), (30, 30)}


def test_histogram_roi_statistics(make_frame):
    image = np.full((20, 20), 40, np.uint8)
    image[:, 10:] = 120
    ctx = StepContext(make_frame(image))
    out = HistogramStats().apply(image, ctx)
    assert np.array_equal(out, image)
    assert ctx.data["analyze.histogram_stats.mean"] == 80
    assert ctx.data["analyze.histogram_stats.std"] == 40
    HistogramStats(x=50, w=50).apply(image, ctx)
    assert ctx.data["analyze.histogram_stats.mean"] == 120
    assert ctx.data["analyze.histogram_stats.std"] == pytest.approx(0, abs=1e-6)
