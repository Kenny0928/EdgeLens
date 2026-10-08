"""顏色與遮罩的已知像素對照。"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from edgelens.processing.pipeline import StepContext
from edgelens.processing.steps.color import (
    CLAHE,
    BrightnessContrast,
    Channel,
    Gamma,
    Grayscale,
    HSVMask,
    Invert,
    LABMask,
)


def test_grayscale_and_invert(bgr_image, make_frame):
    ctx = StepContext(make_frame(bgr_image))
    out = Grayscale().apply(bgr_image, ctx)
    assert np.array_equal(out, cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY))
    assert np.array_equal(Invert().apply(out, ctx), 255 - out)
    assert np.array_equal(Grayscale().apply(out, ctx), out)


@pytest.mark.parametrize("mode", ["mask", "keep", "overlay"])
def test_red_wrap_mask_outputs(mode, make_frame):
    hsv = np.array([[[179, 255, 255], [1, 255, 255], [90, 255, 255], [0, 0, 255]]], np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    ctx = StepContext(make_frame(image))
    step = HSVMask(h_min=170, h_max=10, s_min=100, output=mode, alpha=0.5)
    out = step.apply(image, ctx)
    if mode == "mask":
        assert out.tolist() == [[255, 255, 0, 0]]
    elif mode == "keep":
        assert np.array_equal(out[:, :2], image[:, :2]) and not out[:, 2:].any()
    else:
        assert np.array_equal(out[:, 2:], image[:, 2:])
        assert out[0, 0, 1] in (127, 128) and out[0, 0, 2] in (127, 128)
    step.set_param("s_min", 255)
    step.set_param("s_max", 0)
    step.set_param("output", "mask")
    assert not step.apply(image, ctx).any()


def test_regular_hsv_and_lab_masks(make_frame):
    image = np.array([[[0, 0, 255], [0, 255, 0], [255, 0, 0]]], np.uint8)
    ctx = StepContext(make_frame(image))
    assert HSVMask(h_min=50, h_max=70).apply(image, ctx).tolist() == [[0, 255, 0]]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[0, 0]
    params = {f"{c}_{bound}": int(v) for c, v in zip("lab", lab, strict=True) for bound in ("min", "max")}
    assert LABMask(**params).apply(image, ctx).tolist() == [[255, 0, 0]]
    assert np.array_equal(LABMask(**params, output="keep").apply(image, ctx)[0, 0], image[0, 0])
    assert LABMask(**params, output="overlay").apply(image, ctx).shape == image.shape


def test_brightness_saturation_and_gamma(make_frame):
    image = np.array([[0, 20, 100, 200, 255]], np.uint8)
    ctx = StepContext(make_frame(image))
    assert BrightnessContrast(brightness=-50, contrast=2).apply(image, ctx).tolist() == [[0, 0, 150, 255, 255]]
    light = Gamma(gamma=2).apply(image, ctx)
    dark = Gamma(gamma=0.5).apply(image, ctx)
    assert light[0, 2] > 100 > dark[0, 2]
    assert (light[0, 0], light[0, -1]) == (0, 255)
    assert np.array_equal(Gamma().apply(image, ctx), image)


@pytest.mark.parametrize("color", [False, True])
def test_clahe_improves_local_contrast(color, make_frame):
    image = np.tile(np.linspace(100, 120, 64, dtype=np.uint8), (64, 1))
    if color:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    out = CLAHE().apply(image, StepContext(make_frame(image)))
    assert out.shape == image.shape
    assert np.ptp(out) > np.ptp(image)


@pytest.mark.parametrize("channel,space,index", [("b", None, 0), ("g", None, 1), ("r", None, 2),
    ("h", cv2.COLOR_BGR2HSV, 0), ("s", cv2.COLOR_BGR2HSV, 1), ("v", cv2.COLOR_BGR2HSV, 2),
    ("lab_l", cv2.COLOR_BGR2LAB, 0), ("lab_a", cv2.COLOR_BGR2LAB, 1), ("lab_b", cv2.COLOR_BGR2LAB, 2)])
def test_all_channels(channel, space, index, bgr_image, make_frame):
    expected = bgr_image if space is None else cv2.cvtColor(bgr_image, space)
    out = Channel(channel=channel).apply(bgr_image, StepContext(make_frame(bgr_image)))
    assert np.array_equal(out, expected[:, :, index])
