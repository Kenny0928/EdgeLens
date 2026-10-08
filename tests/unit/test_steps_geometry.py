"""驗證幾何內容與連續座標換算，避免 overlay 偏移。"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from edgelens.processing.pipeline import StepContext
from edgelens.processing.steps.geometry import ROI, Crop, Flip, Resize, Rotate


def test_crop_and_resize_coordinates(make_frame):
    image = np.arange(100 * 200, dtype=np.uint8).reshape(100, 200)
    ctx = StepContext(make_frame(image))
    cropped = Crop(x=25, y=20, w=50, h=60).apply(image, ctx)
    assert cropped.shape == (60, 100) and ctx.offset == (50, 20)
    assert np.array_equal(cropped, image[20:80, 50:150])
    out = Resize(mode="size", width=50, height=30).apply(cropped, ctx)
    assert out.shape == (30, 50) and ctx.scale == (2, 2)
    assert ctx.to_original(5, 5) == (60, 30)
    Crop(x=10, y=10, w=20, h=20).apply(out, ctx)
    assert ctx.offset == (60, 26)
    assert not np.shares_memory(cropped, image)


def test_crop_clamps_at_border(make_frame):
    image = np.zeros((4, 5), np.uint8)
    ctx = StepContext(make_frame(image))
    assert Crop(x=100, y=100, w=100, h=100).apply(image, ctx).shape == (1, 1)
    assert ctx.offset == (4, 3)


@pytest.mark.parametrize("keep", [False, True])
def test_roi_overlay_and_keep(make_frame, keep):
    image = np.zeros((100, 200, 3), np.uint8)
    ctx = StepContext(make_frame(image))
    out = ROI(x=10, y=20, w=30, h=40, keep_only=keep).apply(image, ctx)
    assert ctx.overlays[0].data["box"] == [20, 20, 80, 60]
    assert out.shape == ((40, 60, 3) if keep else image.shape)
    assert ctx.offset == ((20, 20) if keep else (0, 0))


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_rotate_right_angles_map_each_pixel(angle, make_frame):
    image = np.arange(15, dtype=np.uint8).reshape(3, 5)
    ctx = StepContext(make_frame(image))
    out = Rotate(mode=str(angle)).apply(image, ctx)
    assert np.array_equal(out, np.rot90(image, angle // 90))
    for y, x in np.ndindex(out.shape):
        ox, oy = ctx.to_original(x, y)
        assert out[y, x] == image[round(oy), round(ox)]


@pytest.mark.parametrize("direction,axis", [("horizontal", 1), ("vertical", 0), ("both", (0, 1))])
def test_flip(direction, axis, make_frame):
    image = np.arange(15, dtype=np.uint8).reshape(3, 5)
    ctx = StepContext(make_frame(image))
    out = Flip(direction=direction).apply(image, ctx)
    assert np.array_equal(out, np.flip(image, axis))
    for y, x in np.ndindex(out.shape):
        ox, oy = ctx.to_original(x, y)
        assert out[y, x] == image[round(oy), round(ox)]


def test_custom_rotation_then_crop_resize_roi(make_frame):
    image = np.zeros((40, 60), np.uint8)
    image[10:25, 20:40] = 255
    ctx = StepContext(make_frame(image))
    out = Rotate(mode="custom", angle=30, expand=False).apply(image, ctx)
    assert out.shape == image.shape and not np.array_equal(out, image)
    matrix = cv2.getRotationMatrix2D((29.5, 19.5), 30, 1)
    inv = cv2.invertAffineTransform(matrix)
    out = Crop(x=10, y=25, w=50, h=50).apply(out, ctx)
    out = Resize(scale=2).apply(out, ctx)
    assert ctx.to_original(4, 6) == pytest.approx(inv @ [8, 13, 1])
    ROI().apply(out, ctx)
    assert ctx.overlays[0].kind == "polygon"
    expanded = Rotate(mode="custom", angle=30).apply(image, StepContext(make_frame(image)))
    assert expanded.shape[0] > 40 and expanded.shape[1] > 60


@pytest.mark.parametrize("method", ["nearest", "linear", "area", "cubic"])
def test_resize_interpolations_and_minimum_size(method, make_frame):
    image = np.full((4, 6), 53, np.uint8)
    ctx = StepContext(make_frame(image))
    out = Resize(scale=0.01, interpolation=method).apply(image, ctx)
    assert out.shape == (1, 1) and out[0, 0] == 53
    assert ctx.scale == (6, 4)
