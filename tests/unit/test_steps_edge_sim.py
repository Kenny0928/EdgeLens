"""模擬畫質退化與尺寸座標；不需要實體相機。"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from edgelens.processing.pipeline import StepContext
from edgelens.processing.steps.edge_sim import JPEG, RGB565, Noise, Resolution, Sensor, jpeg_roundtrip


@pytest.mark.parametrize("name,size", [("qqvga", (120, 160)), ("qvga", (240, 320)), ("vga", (480, 640)), ("custom", (30, 40))])
def test_resolution_dimensions_and_scale(name, size, make_frame):
    image = np.full((480, 640, 3), 120, np.uint8)
    ctx = StepContext(make_frame(image))
    step = Resolution(resolution=name, width=40, height=30, upscale=False)
    out = step.apply(image, ctx)
    assert out.shape[:2] == size
    assert ctx.scale == (640 / size[1], 480 / size[0])
    ctx = StepContext(make_frame(image))
    step.set_param("upscale", True)
    assert step.apply(image, ctx).shape == image.shape
    assert ctx.scale == (1, 1)


def test_rgb565_levels_and_max_error(make_frame):
    image = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)
    out = RGB565().apply(image, StepContext(make_frame(image)))
    assert [len(np.unique(out[:, :, c])) for c in range(3)] == [32, 64, 32]
    error = np.abs(out.astype(int) - image.astype(int)).max(axis=(0, 1))
    assert np.all(error <= (7, 3, 7))
    assert np.all(out[0, 0] == 0) and np.all(out[0, -1] == 255)


def test_jpeg_quality_affects_error(make_frame):
    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, (100, 100), dtype=np.uint8)
    ctx = StepContext(make_frame(image))
    low = JPEG(quality=5).apply(image, ctx)
    high = JPEG(quality=95).apply(image, ctx)
    low_error = np.square(low.astype(float) - image).mean()
    high_error = np.square(high.astype(float) - image).mean()
    assert low.shape == image.shape and low_error > 5 * high_error


def test_jpeg_errors_are_actionable(monkeypatch):
    image = np.zeros((5, 5), np.uint8)
    with monkeypatch.context() as m:
        m.setattr(cv2, "imencode", lambda *args: (False, None))
        with pytest.raises(ValueError, match="重試"):
            jpeg_roundtrip(image, 50)
    with monkeypatch.context() as m:
        m.setattr(cv2, "imdecode", lambda *args: None)
        with pytest.raises(ValueError, match="重試"):
            jpeg_roundtrip(image, 50)


def test_noise_statistics_and_zero(make_frame):
    image = np.full((300, 300), 128, np.uint8)
    ctx = StepContext(make_frame(image))
    assert np.array_equal(Noise(strength=0).apply(image, ctx), image)
    out = Noise(strength=10).apply(image, ctx)
    assert abs(float(out.mean()) - 128) < 0.3
    assert 9.5 < out.std() < 10.5


@pytest.mark.parametrize("profile", ["esp32_cam", "k210"])
def test_sensor_profiles_ignore_custom_parameters(profile, make_frame, monkeypatch):
    monkeypatch.setattr("edgelens.processing.steps.edge_sim.add_noise", lambda image, strength: image.copy())
    image = np.random.default_rng(1).integers(0, 256, (80, 100, 3), dtype=np.uint8)
    ctx = StepContext(make_frame(image))
    baseline = Sensor(profile=profile, upscale=False).apply(image, ctx)
    out = Sensor(profile=profile, upscale=False, width=10, height=10, rgb565=False,
                 quality=1, noise=100, blur=5).apply(image, StepContext(make_frame(image)))
    assert np.array_equal(baseline, out) and out.shape == (240, 320, 3)
    assert ctx.scale == (100 / 320, 80 / 240)


def test_sensor_custom_composition(make_frame):
    image = np.random.default_rng(0).integers(0, 256, (40, 60, 3), dtype=np.uint8)
    ctx = StepContext(make_frame(image))
    out = Sensor(profile="custom", width=20, height=10, rgb565=False, quality=80,
                 noise=0, blur=0, upscale=False).apply(image, ctx)
    expected = jpeg_roundtrip(cv2.resize(image, (20, 10), interpolation=cv2.INTER_AREA), 80)
    assert np.array_equal(out, expected) and ctx.scale == (3, 4)
    up = Sensor(profile="custom", width=20, height=10, noise=0, blur=0).apply(image, ctx)
    assert up.shape == image.shape
