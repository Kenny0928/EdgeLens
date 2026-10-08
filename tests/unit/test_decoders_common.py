"""解碼器測試的共用工具（已知答案編碼器的共同部分）與 _util 的測試。"""

from __future__ import annotations

import numpy as np

from edgelens.core.modelcard import ModelCard
from edgelens.core.registry import discover
from edgelens.inference.base import DECODERS, DecodeContext, DecodeOutput, TensorInfo
from edgelens.inference.decoders import _util
from edgelens.inference.nms import batched_nms, iou

discover()

C = 8
TRUTH = [((100.0, 120.0, 220.0, 300.0), 2), ((400.0, 50.0, 560.0, 200.0), 0), ((300.0, 400.0, 380.0, 600.0), 5)]


def cxcywh(b):
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def make_decoder(kind, params=None, labels=None):
    card = ModelCard(name="t", file="t.onnx")
    if labels:
        card.labels = list(labels)
    return DECODERS.get(kind)(card, params), card


def run_decoder(kind, arrays, size, params=None, labels=None, conf=0.25, infos=None, nms_iou=0.45):
    """解碼 + NMS，回傳 (boxes, scores, class_ids, DecodeOutput)。"""
    dec, card = make_decoder(kind, params, labels)
    ctx = DecodeContext(card=card, input_size=size, conf_threshold=conf, num_classes=card.num_classes,
                        output_infos=infos or [])
    out: DecodeOutput = dec.decode(arrays, ctx)
    if out.nms_done:
        return out.boxes, out.scores, out.class_ids, out
    keep = batched_nms(out.boxes, out.scores, out.class_ids, nms_iou, 100)
    return out.boxes[keep], out.scores[keep], out.class_ids[keep], out


def assert_truth(boxes, ids, truth=TRUTH):
    assert len(boxes) == len(truth), f"應有 {len(truth)} 個框，實際 {len(boxes)}"
    for xyxy, cid in truth:
        ov = iou(np.array([xyxy], np.float32), boxes)[0]
        j = int(ov.argmax())
        assert ov[j] > 0.99, f"{xyxy} 的最佳 IoU 只有 {ov[j]:.3f}"
        assert int(ids[j]) == cid


def infos_for(*arrays, dtype="float32"):
    return [TensorInfo(f"o{i}", tuple(a.shape), dtype) for i, a in enumerate(arrays)]


# ---- _util 本身 ----------------------------------------------------------------

def test_sigmoid_softmax_stable():
    assert np.allclose(_util.sigmoid(np.array([-1000.0, 0.0, 1000.0])), [0.0, 0.5, 1.0])
    s = _util.softmax(np.array([[1000.0, 1000.0], [0.0, 10.0]]))
    assert np.allclose(s.sum(axis=1), 1.0) and np.allclose(s[0], 0.5)


def test_grid_match_variants():
    card = ModelCard(name="t", file="t.onnx")
    assert _util.grid_match(8400, card)[:2] == (1, (640, 640))
    assert _util.grid_match(25200, card)[0] == 3
    assert _util.grid_match(8401, card) is None
    # P6
    assert _util.grid_match(_util.grid_total((640, 640), (8, 16, 32, 64)), card)[2] == (8, 16, 32, 64)
    assert _util.grid_match(None, card) is None
    card.input.size = (416, 416)
    assert _util.grid_match(_util.grid_total((416, 416), (8, 16, 32)), card)[1] == (416, 416)


def test_decoded_likelihood():
    raw = np.random.default_rng(0).normal(0, 1, (1, 100, 13)).astype(np.float32)
    assert _util.decoded_likelihood(raw) < 0.2
    px = np.abs(raw) * 200
    assert _util.decoded_likelihood(px) > 0.8
    assert _util.decoded_likelihood(None) is None


def test_is_normalized():
    assert _util.is_normalized(np.array([0.1, 1.0]))
    assert not _util.is_normalized(np.array([0.1, 100.0]))
    assert _util.is_normalized(np.array([100.0]), True)
    assert not _util.is_normalized(np.array([0.1]), False)
