"""時間戳 FPS、滑動視窗與多執行緒統計。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from edgelens.metrics.rolling import RollingStats


def test_summary_window_and_fps():
    stats = RollingStats(4)
    assert stats.summary() is None and stats.fps == 0
    for i, value in enumerate((100, 1, 2, 3, 4)):
        stats.add(value, i*.25)
    s = stats.summary()
    assert s == {"last": 4, "mean": 2.5, "median": 2.5, "p95": 4, "min": 1, "max": 4, "n": 4, "fps": 4}
    stats.reset()
    stats.add(1, 0)
    stats.add(2, 0)
    assert stats.fps == 0


def test_threads():
    stats = RollingStats(1000)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(stats.add, range(1000)))
    assert stats.summary()["mean"] == 499.5
    assert stats.summary()["n"] == 1000


def test_invalid():
    with pytest.raises(ValueError):
        RollingStats(0)
    stats = RollingStats()
    with pytest.raises(ValueError):
        stats.add(float("nan"))
    stats.add(1, 10)
    with pytest.raises(ValueError):
        stats.add(2, 9)
