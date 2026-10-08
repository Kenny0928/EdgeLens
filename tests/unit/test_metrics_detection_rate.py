"""滑動命中率以畫面為分母，分數只平均命中畫面。"""
from __future__ import annotations

import pytest

from edgelens.core.results import Detection, InferenceResult
from edgelens.metrics.detection_rate import DetectionRate


def result(*items):
    return InferenceResult("detect", "模型", 1, detections=[Detection((0, 0, 10, 10), score, cid, label)
                                                           for cid, label, score in items])


def test_rate_class_threshold_streak_and_expiry():
    rate = DetectionRate(window=5, threshold=.5)
    assert rate.rate(now=0) == (None, 0, 0)
    rate.add(result((0, "蘋果", .9), (0, "蘋果", .7)), now=1)
    rate.add(result((0, "蘋果", .4)), now=2)
    rate.add(result((1, "香蕉", .6)), now=3)
    rate.add(result((0, "蘋果", .8), (1, "香蕉", .7)), now=4)
    s = rate.summary(now=4)
    assert s.rate == .75 and s.consecutive_hits == 2 and s.last_hit_time == 4
    assert s.mean_best_score == pytest.approx((.9+.6+.8)/3)
    assert rate.summary("蘋果", now=4).rate == .5
    assert rate.summary(1, now=4).consecutive_hits == 2
    assert rate.summary("蘋果", now=4).mean_best_score == pytest.approx(.85)
    assert rate.rate(now=8) == (100, 2, 2)
    s = rate.summary(now=10)
    assert s.total == 0 and s.rate is None and s.consecutive_hits == 0 and s.last_hit_time == 4
    rate.reset()
    assert rate.summary(now=10).last_hit_time is None


def test_invalid():
    with pytest.raises(ValueError):
        DetectionRate(window=0)
    rate = DetectionRate()
    rate.add(result(), now=5)
    with pytest.raises(ValueError):
        rate.add(result(), now=4)
    with pytest.raises(ValueError):
        rate.summary(now=4)
