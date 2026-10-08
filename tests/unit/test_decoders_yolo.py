"""YOLOv5／v6／v7／v8／v9／v10／v11／v12／YOLOX 解碼器的已知答案測試。

做法：在輸入影像座標放已知的框，依各家族的格式反向編碼成模型輸出，加上低分數背景雜訊與
一個分數較低的重複框，再解碼 + NMS，每個已知框都要找到（IoU > 0.99、類別正確、沒有多餘的框）。
"""

from __future__ import annotations

import numpy as np
import pytest

from edgelens.inference.base import InferenceError, TensorInfo
from edgelens.inference.decoders.yolox_raw import yolox_grids
from tests.unit.test_decoders_common import (
    TRUTH,
    C,
    assert_truth,
    cxcywh,
    logit,
    make_decoder,
    run_decoder,
)

SIZE = (640, 640)


def _noise_rows(n, d, rng, cls_start):
    a = rng.random((n, d), dtype=np.float32) * 0.1
    a[:, :4] = rng.random((n, 4), dtype=np.float32) * 600 + 10
    return a


def _slots(n, rng):
    return rng.choice(n, size=2 * len(TRUTH), replace=False)


def encode_rows(n, d, head, rng, normalized=False, obj=False):
    """候選框列：cxcywh + (objectness) + 類別分數。重複框分數較低。"""
    rows = _noise_rows(n, d, rng, head)
    slots = _slots(n, rng)
    for k, (box, cid) in enumerate(TRUTH):
        for slot, score, jit in (((slots[2 * k], 0.9, 0.0), (slots[2 * k + 1], 0.6, 3.0))):
            cx, cy, w, h = cxcywh(box)
            rows[slot, :4] = [cx + jit, cy + jit, w, h]
            rows[slot, head:] = rng.random(d - head) * 0.1
            if obj:
                rows[slot, 4] = score
                rows[slot, head + cid] = 0.95
            else:
                rows[slot, head + cid] = score
    if normalized:
        rows[:, :4] /= np.array([640, 640, 640, 640], np.float32)
    return rows


# ---- yolo_t ---------------------------------------------------------------------

@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
def test_yolo_t_known_answer(transposed, normalized):
    rng = np.random.default_rng(1)
    rows = encode_rows(8400, 4 + C, 4, rng, normalized)
    arr = (rows if transposed else rows.T)[None].copy()
    boxes, _, ids, _ = run_decoder("yolo_t", [arr], SIZE)
    assert_truth(boxes, ids)


def test_yolo_t_logits_and_labels():
    rng = np.random.default_rng(2)
    rows = encode_rows(8400, 4 + C, 4, rng)
    rows[:, 4:] = logit(rows[:, 4:])
    boxes, _, ids, _ = run_decoder("yolo_t", [rows.T[None].copy()], SIZE, labels=[f"c{i}" for i in range(C)])
    assert_truth(boxes, ids)


def test_yolo_t_class_mismatch_error():
    arr = np.zeros((1, 4 + C, 100), np.float32)
    with pytest.raises(InferenceError) as e:
        run_decoder("yolo_t", [arr], SIZE, labels=["a", "b"])
    assert str(C) in e.value.message and e.value.hint


def test_yolo_t_match():
    cls = type(make_decoder("yolo_t")[0])
    _, card = make_decoder("yolo_t")
    m = cls.match([TensorInfo("o", (1, 84, 8400))], card)
    assert m.score >= 0.85 and "80" in m.reason and not m.params["transposed"]
    assert cls.match([TensorInfo("o", (1, 8400, 84))], card).params["transposed"] is True
    assert cls.match([TensorInfo("o", (1, 84, None))], card).score > 0.5
    assert cls.match([TensorInfo("o", (None, None, None))], card).score == 0
    _, c80 = make_decoder("yolo_t", labels=[str(i) for i in range(80)])
    assert cls.match([TensorInfo("o", (1, 84, 8400))], c80).score > 0.9
    _, c3 = make_decoder("yolo_t", labels=["a", "b", "c"])
    assert cls.match([TensorInfo("o", (1, 84, 8400))], c3).score == 0


# ---- yolo_flat ------------------------------------------------------------------

def test_yolo_flat_v5_known_answer():
    rng = np.random.default_rng(3)
    rows = encode_rows(25200, 5 + C, 5, rng, obj=True)
    boxes, scores, ids, _ = run_decoder("yolo_flat", [rows[None]], SIZE)
    assert_truth(boxes, ids)
    assert scores.max() == pytest.approx(0.9 * 0.95, abs=1e-4)


def test_yolo_flat_v5_normalized_logits():
    rng = np.random.default_rng(4)
    rows = encode_rows(25200, 5 + C, 5, rng, normalized=True, obj=True)
    rows[:, 4:] = logit(rows[:, 4:])
    boxes, _, ids, _ = run_decoder("yolo_flat", [rows[None]], SIZE)
    assert_truth(boxes, ids)


def test_yolo_flat_v6_no_objectness_and_xyxy():
    rng = np.random.default_rng(5)
    rows = encode_rows(8400, 4 + C, 4, rng)
    boxes, _, ids, _ = run_decoder("yolo_flat", [rows[None]], SIZE, params={"objectness": False})
    assert_truth(boxes, ids)
    xy = rows.copy()
    c = rows[:, :4]
    xy[:, 0], xy[:, 1], xy[:, 2], xy[:, 3] = c[:, 0] - c[:, 2] / 2, c[:, 1] - c[:, 3] / 2, \
        c[:, 0] + c[:, 2] / 2, c[:, 1] + c[:, 3] / 2
    boxes, _, ids, _ = run_decoder("yolo_flat", [xy[None]], SIZE,
                                   params={"objectness": False, "box_format": "xyxy"})
    assert_truth(boxes, ids)


def test_yolo_flat_v6_with_objectness_decoded():
    rng = np.random.default_rng(6)
    rows = encode_rows(8400, 5 + C, 5, rng, obj=True)
    boxes, _, ids, _ = run_decoder("yolo_flat", [rows[None]], SIZE)
    assert_truth(boxes, ids)


def test_yolo_flat_match():
    cls = type(make_decoder("yolo_flat")[0])
    _, card = make_decoder("yolo_flat")
    m = cls.match([TensorInfo("o", (1, 25200, 85))], card)
    assert m.score >= 0.85 and "25200" in m.reason
    assert cls.match([TensorInfo("o", (1, 84, 8400))], card).score == 0
    assert cls.match([TensorInfo("o", (1, 25200, None))], card).score == 0


# ---- yolox_raw ------------------------------------------------------------------

def encode_yolox(rng, strides=(8, 16, 32)):
    grid, strd = yolox_grids(SIZE, strides)
    n = len(grid)
    out = rng.normal(0, 0.5, (n, 5 + C)).astype(np.float32)
    out[:, 4:] = rng.random((n, 1 + C), dtype=np.float32) * 0.1  # objectness、類別（已 sigmoid）
    used = set()
    for box, cid in TRUTH:
        cx, cy, w, h = cxcywh(box)
        for rep, (score, jit) in enumerate(((0.9, 0.0), (0.6, 3.0))):
            s = 8 if max(w, h) < 100 else 16 if max(w, h) < 200 else 32
            s = s if s in strides else strides[-1]
            gx, gy = int(cx // s) + rep, int(cy // s)  # 重複框放在隔壁格，編碼成 > 1 的偏移
            sel = np.nonzero((grid[:, 0] == gx) & (grid[:, 1] == gy) & (strd[:, 0] == s))[0][0]
            assert sel not in used
            used.add(sel)
            out[sel, 0] = (cx + jit) / s - gx
            out[sel, 1] = (cy + jit) / s - gy
            out[sel, 2:4] = np.log([w / s, h / s])
            out[sel, 4] = score
            out[sel, 5:] = rng.random(C) * 0.1
            out[sel, 5 + cid] = 0.95
    return out


@pytest.mark.parametrize("strides", [(8, 16, 32), (8, 16, 32, 64)])
def test_yolox_raw_known_answer(strides):
    out = encode_yolox(np.random.default_rng(7), strides)
    boxes, _, ids, _ = run_decoder("yolox_raw", [out[None]], SIZE)
    assert_truth(boxes, ids)


def test_yolox_raw_wrong_size_error():
    out = encode_yolox(np.random.default_rng(7))
    with pytest.raises(InferenceError) as e:
        run_decoder("yolox_raw", [out[None]], (416, 416))
    assert e.value.hint


def test_yolox_raw_match_uses_samples():
    cls = type(make_decoder("yolox_raw")[0])
    flat = type(make_decoder("yolo_flat")[0])
    _, card = make_decoder("yolox_raw")
    info = [TensorInfo("o", (1, 8400, 85))]
    raw = encode_yolox(np.random.default_rng(7))[None]
    decoded = np.abs(raw) * 100
    assert cls.match(info, card, [raw]).score > flat.match(info, card, [raw]).score
    assert flat.match(info, card, [decoded]).score > cls.match(info, card, [decoded]).score
    a, b = cls.match(info, card).score, flat.match(info, card).score
    assert abs(a - b) < 0.05
    assert cls.match([TensorInfo("o", (1, 25200, 85))], card).score == 0


# ---- end2end --------------------------------------------------------------------

def _e2e(d, k=300, swap=False):
    rng = np.random.default_rng(8)
    a = np.zeros((k, d), np.float32)
    for i, (box, cid) in enumerate(TRUTH):
        if d == 6:
            row = [*box, 0.9 - 0.1 * i, cid]
            if swap:
                row[4], row[5] = cid, row[4]
        else:
            row = [0, *box, cid, 0.9 - 0.1 * i]
        a[i] = row
    a[len(TRUTH):, 5 if d == 6 else 6] = rng.random(k - len(TRUTH)) * 0.1  # 低分數
    return a[None]


@pytest.mark.parametrize("d", [6, 7])
def test_end2end_known_answer(d):
    boxes, _, ids, out = run_decoder("end2end", [_e2e(d)], SIZE)
    assert out.nms_done
    assert_truth(boxes, ids)


def test_end2end_swapped_and_normalized_and_columns():
    boxes, _, ids, _ = run_decoder("end2end", [_e2e(6, swap=True)], SIZE)
    assert_truth(boxes, ids)
    arr = _e2e(6)
    arr[..., :4] /= 640.0
    boxes, _, ids, _ = run_decoder("end2end", [arr], SIZE)
    assert_truth(boxes, ids)
    arr = _e2e(7)[..., 1:]  # 少 batch 欄 → 6 欄 [x1,y1,x2,y2,class,score]
    boxes, _, ids, _ = run_decoder("end2end", [arr.copy()], SIZE,
                                   params={"columns": ["x1", "y1", "x2", "y2", "class", "score"]})
    assert_truth(boxes, ids)
    with pytest.raises(InferenceError):
        run_decoder("end2end", [arr.copy()], SIZE, params={"columns": ["x1", "y1"]})


def test_end2end_match():
    cls = type(make_decoder("end2end")[0])
    _, card = make_decoder("end2end")
    assert cls.match([TensorInfo("o", (1, 300, 6))], card, [_e2e(6)]).score >= 0.9
    assert cls.match([TensorInfo("o", (1, 100, 7))], card).score > 0.8
    assert cls.match([TensorInfo("o", (1, 8400, 6))], card).score == 0
    assert cls.match([TensorInfo("o", (1, None, 6))], card).score > 0
