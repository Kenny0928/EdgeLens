"""NMS 與框格式轉換測試。"""

from __future__ import annotations

import time

import numpy as np
import pytest

from edgelens.inference.nms import batched_nms, box_convert, iou, nms


def test_box_convert_all_formats():
    assert box_convert([[10, 20, 30, 40]], "xyxy").tolist() == [[10, 20, 30, 40]]
    assert box_convert([[10, 20, 20, 20]], "xywh").tolist() == [[10, 20, 30, 40]]
    assert box_convert([[20, 30, 20, 20]], "cxcywh").tolist() == [[10, 20, 30, 40]]
    assert box_convert([[20, 10, 40, 30]], "yxyx").tolist() == [[10, 20, 30, 40]]
    assert box_convert(np.zeros((0, 4)), "cxcywh").shape == (0, 4)
    src = np.array([[1, 2, 3, 4]], np.float32)
    box_convert(src, "xyxy")[0, 0] = 99
    assert src[0, 0] == 1  # 不改到輸入
    with pytest.raises(ValueError):
        box_convert(src, "nope")


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30], [0, 0, 0, 0]], np.float32)
    r = iou(a, b)
    assert r.shape == (1, 4)
    assert r[0, 0] == pytest.approx(1.0) and r[0, 1] == pytest.approx(1 / 3) and r[0, 2] == 0 and r[0, 3] == 0


def test_nms_known_answer():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0, 0, 10, 11]], np.float32)
    scores = np.array([0.8, 0.9, 0.7, 0.6], np.float32)
    keep = nms(boxes, scores, 0.5, 100)
    assert keep.tolist() == [1, 2]
    assert nms(boxes, scores, 0.99, 100).tolist() == [1, 0, 3, 2] or len(nms(boxes, scores, 0.99, 100)) >= 3
    assert nms(boxes, scores, 0.5, 1).tolist() == [1]
    assert nms(boxes[:0], scores[:0], 0.5, 10).size == 0
    assert nms(boxes, scores, 0.5, 0).size == 0


def test_batched_nms_per_class_vs_agnostic():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    ids = np.array([0, 0, 1])
    assert sorted(batched_nms(boxes, scores, ids, 0.5, 100).tolist()) == [0, 2]
    assert batched_nms(boxes, scores, ids, 0.5, 100, agnostic=True).tolist() == [0]
    assert batched_nms(boxes[:0], scores[:0], ids[:0], 0.5, 100).size == 0
    # 座標很大（偏移技巧不能讓類別互相影響）
    big = boxes * 1000
    assert sorted(batched_nms(big, scores, ids, 0.5, 100).tolist()) == [0, 2]


def test_nms_performance_8000_boxes():
    rng = np.random.default_rng(0)
    xy = rng.random((8000, 2), dtype=np.float32) * 600
    boxes = np.concatenate([xy, xy + rng.random((8000, 2), dtype=np.float32) * 80 + 5], axis=1)
    scores = rng.random(8000, dtype=np.float32)
    ids = rng.integers(0, 10, 8000)
    batched_nms(boxes, scores, ids, 0.45, 100)  # 暖機
    best = min(_time(lambda: batched_nms(boxes, scores, ids, 0.45, 100)) for _ in range(5))
    assert best < 0.010, f"{best * 1000:.1f} ms"


def _time(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t
