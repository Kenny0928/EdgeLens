"""以可預期梯度驗證 64 位元方向與最近視窗。"""
from __future__ import annotations

import numpy as np
import pytest

from edgelens.dataset.dedupe import RecentHashes, dhash, find_near_duplicates, hamming_distance
from edgelens.dataset.writer import SampleRecord


def test_hash_and_distance():
    gradient = np.tile(np.arange(9, dtype=np.uint8), (8, 1))
    assert dhash(gradient) == 2**64-1
    assert dhash(gradient[:, ::-1]) == 0
    assert hamming_distance("ffffffffffffffff", 0) == 64
    assert hamming_distance(7, 3) == 1


def test_pairs_and_recent():
    a, b, c = [SampleRecord(f"a/{i}.jpg", "a", dhash=h) for i, h in enumerate(("0", "1", "ff"))]
    assert find_near_duplicates([a, b, c], 1) == [(a, b, 1)]
    recent = RecentHashes(2)
    assert recent.check(0) is None
    recent.add(a)
    recent.add(b)
    assert recent.check(0) == (a, 0)
    recent.add(c)
    assert recent.check(0) == (b, 1)
    recent.clear()
    assert recent.check(0) is None


def test_invalid():
    with pytest.raises(ValueError):
        RecentHashes(0)
    with pytest.raises(ValueError):
        hamming_distance(-1, 0)
    with pytest.raises(ValueError):
        find_near_duplicates([], 65)
    with pytest.raises(ValueError):
        dhash(np.zeros((0, 0), np.uint8))
