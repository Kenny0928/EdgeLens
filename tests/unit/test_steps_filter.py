"""驗證去雜訊與銳化的效果。"""
from __future__ import annotations

import cv2
import numpy as np

from edgelens.processing.pipeline import StepContext
from edgelens.processing.steps.filter import Bilateral, GaussianBlur, MedianBlur, Sharpen


def test_gaussian_spreads_impulse_and_rounds_kernel(make_frame):
    image = np.zeros((15, 15), np.uint8)
    image[7, 7] = 255
    out = GaussianBlur(size=4).apply(image, StepContext(make_frame(image)))
    assert 0 < out[7, 7] < 255 and out[7, 6] > 0
    assert np.array_equal(out, cv2.GaussianBlur(image, (5, 5), 0))


def test_median_removes_salt_pepper(make_frame):
    image = np.full((15, 15, 3), 100, np.uint8)
    image[7, 7] = 255
    image[5, 5] = 0
    out = MedianBlur(size=3).apply(image, StepContext(make_frame(image)))
    assert np.all(out == 100)


def test_bilateral_reduces_noise_and_preserves_edge(make_frame):
    image = np.full((50, 60), 40, np.uint8)
    image[:, 30:] = 200
    noise = np.random.default_rng(0).integers(-8, 9, image.shape)
    image = (image.astype(int) + noise).astype(np.uint8)
    out = Bilateral(sigma_color=20).apply(image, StepContext(make_frame(image)))
    assert out[:, :25].std() < image[:, :25].std()
    assert out[:, 30].mean() - out[:, 29].mean() > 140


def test_sharpen_increases_edge_contrast(make_frame):
    image = np.full((30, 30), 80, np.uint8)
    image[:, 15:] = 160
    image = cv2.GaussianBlur(image, (5, 5), 1)
    ctx = StepContext(make_frame(image))
    out = Sharpen(amount=2).apply(image, ctx)
    assert int(out[15, 15]) - int(out[15, 14]) > int(image[15, 15]) - int(image[15, 14])
    assert np.array_equal(Sharpen(amount=0).apply(image, ctx), image)
