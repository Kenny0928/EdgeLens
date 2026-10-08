"""品質指標的解析度正規化、曝光界線與翻譯鍵。"""
from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from edgelens.i18n import has
from edgelens.metrics.quality import QualityThresholds, measure, verdict


def test_flat_exposure_and_verdict():
    report = measure(np.full((480, 640, 3), 128, np.uint8))
    assert report.sharpness == 0 and report.brightness == 128 and report.contrast == pytest.approx(0, abs=2e-6)
    assert report.overexposed == report.underexposed == 0
    ok, reasons = verdict(report)
    assert not ok and "dataset.quality.blurry" in reasons
    assert all(has(key, lang) for key in reasons for lang in ("zh_TW", "en"))
    image = np.array([[5, 6], [249, 250]], np.uint8)
    r = measure(image)
    assert r.underexposed == r.overexposed == .25
    assert r.brightness == 127.5
    assert measure(np.zeros((10, 10), np.uint8)).underexposed == 1
    assert measure(np.full((10, 10), 255, np.uint8)).overexposed == 1


def test_resolution_normalization():
    low = np.random.default_rng(1).integers(0, 256, (240, 320), dtype=np.uint8)
    high = cv2.resize(low, (640, 480), interpolation=cv2.INTER_NEAREST)
    assert measure(low).sharpness == pytest.approx(measure(high).sharpness)
    assert measure(high).sharpness > measure(cv2.GaussianBlur(high, (15, 15), 4)).sharpness
    assert verdict(measure(low), QualityThresholds(min_sharpness=0, min_contrast=0))[0]


def test_speed_640x480():
    image = np.random.default_rng(7).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    for _ in range(5):
        measure(image)
    times = []
    for _ in range(50):
        start = time.perf_counter()
        measure(image)
        times.append((time.perf_counter() - start) * 1000)
    assert np.percentile(times, 95) < 5


@pytest.mark.parametrize("image", [np.zeros((0, 0), np.uint8), np.zeros((3, 3), float), np.zeros((3, 3, 4), np.uint8)])
def test_invalid_image(image):
    with pytest.raises(ValueError):
        measure(image)


def test_invalid_thresholds():
    with pytest.raises(ValueError):
        QualityThresholds(min_brightness=250, max_brightness=10)
    with pytest.raises(ValueError):
        QualityThresholds(max_overexposed=2)
