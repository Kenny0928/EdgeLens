"""輸出格式自動判斷：規則表的每一列、samples 區分、說明檔指定優先。"""

from __future__ import annotations

import numpy as np
import pytest

from edgelens.core.modelcard import ModelCard
from edgelens.inference.autodetect import FormatGuess, detect_format, raw_summary
from edgelens.inference.base import TensorInfo


def card(**kw):
    c = ModelCard(name="t", file="t.onnx")
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def top(outputs, c=None, samples=None, **kw):
    g = detect_format(outputs, c or card(), samples, **kw)
    assert g and all(isinstance(x, FormatGuess) for x in g)
    assert [x.score for x in g] == sorted((x.score for x in g), reverse=True)
    return g


def T(*shape, name="o"):
    return TensorInfo(name, tuple(shape))


def test_classification():
    g = top([T(1, 1000)])
    assert g[0].decoder_kind == "classify" and g[0].reason


def test_end2end_k_le_300():
    assert top([T(1, 300, 6)])[0].decoder_kind == "end2end"
    assert top([T(1, 100, 7)])[0].decoder_kind == "end2end"
    assert top([T(1, 8400, 6)])[0].decoder_kind != "end2end"


@pytest.mark.parametrize("shape", [(1, 84, 8400), (1, 8400, 84), (1, 5, 8400), (1, 84, 2100)])
def test_yolo_t(shape):
    g = top([T(*shape)])
    assert g[0].decoder_kind == "yolo_t" and g[0].reason
    assert g[0].params["transposed"] == (shape[1] > shape[2])


def test_yolo_t_with_labels_resolves_5_vs_4():
    c80 = card(labels=[str(i) for i in range(80)])
    assert top([T(1, 8400, 84)], c80)[0].decoder_kind == "yolo_t"
    c79 = card(labels=[str(i) for i in range(79)])
    assert top([T(1, 8400, 84)], c79)[0].decoder_kind in ("yolo_flat", "yolox_raw")


def test_yolo_flat_v5_v7():
    for shape in ((1, 25200, 85), (1, 6300, 85)):    # 640、320
        g = top([T(*shape)])
        assert g[0].decoder_kind == "yolo_flat" and g[0].score >= 0.85
    # 輸入大小從 card 推
    g = top([T(1, 10647, 7)], card())
    assert g[0].decoder_kind == "yolo_flat"          # 416: 3 × 3549 = 10647


def test_input_size_and_p6():
    p6 = 3 * (80 * 80 + 40 * 40 + 20 * 20 + 10 * 10)
    assert top([T(1, p6, 85)])[0].decoder_kind == "yolo_flat"
    c = card()
    c.input.size = (416, 416)
    assert top([T(1, 10647, 85)], c)[0].decoder_kind == "yolo_flat"
    # 輸入大小錯誤時不會誤判成格點數
    c.input.size = (320, 320)
    assert top([T(1, 10647, 85)], c)[0].score < 0.85
    assert top([T(1, 10647, 85)], card(), input_size=(416, 416))[0].decoder_kind == "yolo_flat"


def _raw_yolox(rng):
    a = rng.normal(0, 0.8, (1, 8400, 85)).astype(np.float32)
    a[..., 4:] = rng.random((1, 8400, 81), dtype=np.float32)
    return a


def test_v6_vs_yolox_by_samples():
    info = [T(1, 8400, 85)]
    rng = np.random.default_rng(0)
    raw = _raw_yolox(rng)
    decoded = np.abs(raw) * 300
    decoded[..., 4:] = raw[..., 4:]
    g = top(info, samples=[raw])
    assert g[0].decoder_kind == "yolox_raw" and g[0].score - g[1].score > 0.1
    g = top(info, samples=[decoded])
    assert g[0].decoder_kind == "yolo_flat" and g[0].score - next(
        x for x in g if x.decoder_kind == "yolox_raw").score > 0.1
    # 無法判斷：兩個都回傳、分數接近
    g = top(info)
    kinds = {x.decoder_kind: x.score for x in g}
    assert abs(kinds["yolo_flat"] - kinds["yolox_raw"]) < 0.05
    assert kinds["yolox_raw"] == pytest.approx(kinds["yolo_flat"])


def test_yolo_grid_single_and_multi():
    g = top([T(1, 125, 13, 13)])
    assert g[0].decoder_kind == "yolo_grid" and g[0].params["anchors_preset"] == "tiny_yolov2_voc"
    g = top([T(1, 255, 13, 13, name="a"), T(1, 255, 26, 26, name="b")])
    assert g[0].decoder_kind == "yolo_grid" and g[0].params["anchors_preset"] == "yolov3_tiny_coco"
    g = top([T(1, 255, 13, 13), T(1, 255, 26, 26), T(1, 255, 52, 52)])
    assert g[0].decoder_kind == "yolo_grid" and g[0].params["anchors_preset"] == "yolov3_coco"
    g = top([T(1, 3, 13, 13, 85), T(1, 3, 26, 26, 85)])
    assert g[0].decoder_kind == "yolo_grid"


def test_tflite_ssd():
    g = top([T(1, 10, 4), T(1, 10), T(1, 10), T(1)])
    assert g[0].decoder_kind == "tflite_ssd" and g[0].score >= 0.85


def test_dynamic_and_empty_do_not_raise():
    assert detect_format([], card()) == []
    assert isinstance(detect_format([T(None, None, None)], card()), list)
    detect_format([T(None, 84, None)], card())
    detect_format([T(1, None)], card())


def test_card_decoder_wins():
    g = detect_format([T(1, 84, 8400)], card(decoder="yolo_flat"))
    assert len(g) == 1 and g[0].decoder_kind == "yolo_flat" and g[0].score == 1.0 and g[0].reason
    g = detect_format([T(1, 84, 8400)], card(decoder="no_such_decoder"))
    assert g[0].decoder_kind == "yolo_t"      # 指定的不存在 → 退回自動判斷


def test_raw_summary():
    outs = [T(1, 3, name="scores"), T(1, 2, name="b")]
    lines = raw_summary(outs, [np.array([[0.1, 0.5, 0.9]], np.float32), np.zeros((1, 0), np.float32)])
    assert len(lines) == 2
    assert "scores" in lines[0] and "0.1" in lines[0] and "0.9" in lines[0] and "float32" in lines[0]
    assert raw_summary([], [np.ones((2,), np.uint8)])[0]
