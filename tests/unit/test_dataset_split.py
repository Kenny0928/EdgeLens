"""驗證固定種子、比例與傳遞分組關聯。"""
from __future__ import annotations

import warnings
from collections import Counter

import pytest

from edgelens.core.project import Project
from edgelens.dataset.split import load_splits, split_dataset, split_records
from edgelens.dataset.writer import SampleRecord


def samples(n=20):
    return [SampleRecord(f"{cid}/{i}.jpg", cid) for cid in ("a", "b") for i in range(n)]


def test_stratified_seed_and_roundtrip(tmp_path):
    records = samples()
    project = Project(tmp_path / "中文專案", "切分")
    result = split_dataset(project, records=records, seed=7)
    assert result.to_dict() == split_records(reversed(records), seed=7).to_dict()
    assert result.to_dict() == load_splits(project).to_dict()
    assert result.train != split_records(records, seed=8).train
    for part, n in (("train", 14), ("val", 3), ("test", 3)):
        assert Counter(f.split("/")[0] for f in getattr(result, part)) == {"a": n, "b": n}


def test_transitive_groups_cross_class():
    records = samples(10)
    records[0].group_id = records[1].group_id = "組一"
    records[1].burst_id = records[10].burst_id = "連拍一"
    records[10].group_id = records[11].group_id = "組二"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = split_records(records)
    owner = {f: p for p in ("train", "val", "test") for f in getattr(result, p)}
    assert len({owner[records[i].file] for i in (0, 1, 10, 11)}) == 1
    assert len(owner) == len(records)


@pytest.mark.parametrize("n", [1, 2])
def test_tiny(n):
    with pytest.warns(UserWarning):
        result = split_records(samples(n))
    assert result.train and result.warnings
    assert len(result.train + result.val + result.test) == 2*n


def test_empty_and_zero_ratio():
    assert not split_records([]).train
    result = split_records(samples(), (1, 0, 0))
    assert len(result.train) == 40 and not result.val and not result.test


@pytest.mark.parametrize("ratios", [(0, 0, 0), (.5, .5), (-1, 1, 1), (float("nan"), .2, .8)])
def test_invalid_ratios(ratios):
    with pytest.raises(ValueError):
        split_records(samples(), ratios)


def test_duplicate_records_and_corrupt_splits(tmp_path):
    record = samples()[0]
    with pytest.raises(ValueError):
        split_records([record, record])
    result = split_records(samples())
    result.val.append(result.train[0])
    path = result.save(tmp_path / "splits.json")
    with pytest.raises(ValueError):
        load_splits(path)
