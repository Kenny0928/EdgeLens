"""用已知 OpenCV 顏色驗證取樣、圓形色相與板端門檻。"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from edgelens.processing.colorpick import (
    pick,
    ranges_from_samples,
    to_hsv_mask_params,
    to_maixpy_lab_threshold,
    to_opencv_hsv,
)
from edgelens.processing.pipeline import StepContext
from edgelens.processing.steps.color import HSVMask


def test_red_samples_wrap_and_mask_integration(make_frame):
    hsv = np.array([[[178, 255, 255], [1, 255, 255], [60, 255, 255], [90, 255, 255]]], np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    samples = [pick(image, i, 0, 0) for i in (0, 1)]
    ranges = ranges_from_samples(samples, margin=2)
    assert ranges.hsv_lower[0] == 176 and ranges.hsv_upper[0] == 3
    assert len(to_opencv_hsv(ranges)) == 2
    out = HSVMask(**to_hsv_mask_params(ranges)).apply(image, StepContext(make_frame(image)))
    assert out.tolist() == [[255, 255, 0, 0]]
    masks = [cv2.inRange(hsv, lo, hi) for lo, hi in to_opencv_hsv(ranges)]
    assert np.array_equal(out, cv2.bitwise_or(*masks))
    region = pick(image[:, :2], 0, 0, 5)
    assert region.hsv[0] > 175 or region.hsv[0] < 5
    assert region.hsv_min[0] > region.hsv_max[0]
    assert region.pixel_count == 2


@pytest.mark.parametrize("color", [(0, 0, 0), (255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)])
def test_lab_conversion_known_colors(color):
    image = np.full((3, 3, 3), color, np.uint8)
    sample = pick(image, 1, 1)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    assert sample.bgr == color and sample.lab == tuple(lab)
    assert sample.maixpy_lab == pytest.approx((lab[0] * 100 / 255, lab[1] - 128, lab[2] - 128))
    threshold = to_maixpy_lab_threshold(ranges_from_samples([sample], margin=0))
    assert threshold == (math.floor(sample.maixpy_lab[0]), math.ceil(sample.maixpy_lab[0]),
                         int(lab[1] - 128), int(lab[1] - 128), int(lab[2] - 128), int(lab[2] - 128))
    assert sample.maixpy_lab_min == sample.maixpy_lab_max == sample.maixpy_lab


def test_region_means_ranges_and_clipped_bounds():
    image = np.array([[[0, 20, 40], [20, 40, 60]], [[40, 60, 80], [60, 80, 100]]], np.uint8)
    sample = pick(image, 0, 0, 1)
    assert sample.bounds == (0, 0, 2, 2) and sample.pixel_count == 4
    assert sample.bgr == (30, 50, 70)
    assert sample.bgr_min == (0, 20, 40) and sample.bgr_max == (60, 80, 100)
    ranges = ranges_from_samples([sample], 5)
    assert ranges.bgr_lower == (0, 15, 35) and ranges.bgr_upper == (65, 85, 105)
    assert len(to_opencv_hsv(ranges)) == 1
    full = ranges_from_samples([sample], 255)
    assert full.hsv_lower == (0, 0, 0) and full.hsv_upper == (179, 255, 255)
    assert to_maixpy_lab_threshold(full) == (0, 100, -128, 127, -128, 127)


def test_combined_arc_covers_all_hues():
    hsv = np.zeros((1, 180, 3), np.uint8)
    hsv[0, :, 0] = np.arange(180)
    hsv[0, :, 1:] = 255
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    samples = [pick(image, x, 0, radius=0) for x in range(180)]
    ranges = ranges_from_samples(samples, 0)
    assert ranges.hsv_lower[0] == 0 and ranges.hsv_upper[0] == 179


@pytest.mark.parametrize("x,y,radius", [(-1, 0, 0), (3, 0, 0), (0, 3, 0), (0, 0, -1), (0.5, 0, 0)])
def test_bad_pick_coordinates(x, y, radius):
    with pytest.raises(ValueError):
        pick(np.zeros((3, 3, 3), np.uint8), x, y, radius)


@pytest.mark.parametrize("image", [None, np.zeros((3, 3), np.uint8), np.zeros((3, 3, 3), np.float32), np.zeros((0, 3, 3), np.uint8)])
def test_bad_pick_images(image):
    with pytest.raises(ValueError):
        pick(image, 0, 0)


@pytest.mark.parametrize("margin", [-1, float("nan"), float("inf"), "bad"])
def test_bad_margin(margin):
    sample = pick(np.zeros((3, 3, 3), np.uint8), 0, 0)
    with pytest.raises(ValueError):
        ranges_from_samples([sample], margin)


def test_empty_samples():
    with pytest.raises(ValueError):
        ranges_from_samples([], 5)
