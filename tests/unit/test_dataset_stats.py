"""平衡圖包含零張的類別，避免把缺資料誤認為平衡。"""
from __future__ import annotations

import math

import pytest

from edgelens.dataset.stats import dataset_stats
from edgelens.dataset.writer import SampleRecord


def test_counts_quality_and_empty_classes():
    records = [SampleRecord(f"a/{i}.jpg", "a", source={"id": "手機"},
                            quality={"sharpness": 10*i, "brightness": 100+i}) for i in range(3)]
    report = dataset_stats(records, ["a", "b"])
    assert report.class_counts == {"a": 3, "b": 0}
    assert math.isinf(report.max_min_ratio) and report.imbalanced
    assert report.source_counts == {"手機": 3}
    assert report.quality["sharpness"]["mean"] == 10
    assert sum(report.quality["brightness"]["histogram"]) == 3
    empty = dataset_stats([], ["a", "b"])
    assert empty.total == 0 and not empty.imbalanced
    assert empty.quality["sharpness"]["mean"] is None


def test_deleted_and_invalid():
    assert dataset_stats([SampleRecord("a/x.jpg", "a", deleted=True)]).total == 0
    with pytest.raises(ValueError):
        dataset_stats([], bins=0)
