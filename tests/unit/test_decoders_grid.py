"""YOLOv2／v3／v4（yolo_grid）與 TFLite SSD、分類解碼器的已知答案測試。"""

from __future__ import annotations

import numpy as np
import pytest

from edgelens.inference.base import InferenceError, TensorInfo
from edgelens.inference.decoders.yolo_grid import PRESETS, default_preset, parse_grid
from tests.unit.test_decoders_common import (
    TRUTH,
    assert_truth,
    cxcywh,
    infos_for,
    logit,
    make_decoder,
    run_decoder,
)

C = 6
SIZE = (416, 416)
TRUTH416 = [(tuple(v * 0.65 for v in b), c) for b, c in TRUTH]


def to_layout(t, layout):
    """canonical (A,H,W,D) → 含 batch 的各種排列。"""
    a, h, w, d = t.shape
    if layout == "NCHW":
        return t.transpose(0, 3, 1, 2).reshape(1, a * d, h, w)
    if layout == "NHWC":
        return t.transpose(1, 2, 0, 3).reshape(1, h, w, a * d)
    if layout == "NAHWC":
        return t[None]
    return t.transpose(1, 2, 0, 3)[None]


def encode_grid(specs, assign, rng, unit, sxy_list=None, softmax=False, activated=False):
    """specs：每個輸出 (stride, anchors_px (A,2))；assign：[(輸出索引, 錨點索引, 框, 類別, obj 機率)]。
    回傳每個輸出的 canonical (A,H,W,5+C) 陣列。"""
    ts = []
    for stride, anc in specs:
        h = w = SIZE[0] // stride
        t = rng.normal(0, 0.3, (len(anc), h, w, 5 + C)).astype(np.float32)
        t[..., 4] = -6.0 + rng.normal(0, 0.3, t[..., 4].shape)
        t[..., 5:] = rng.normal(-2, 0.3, t[..., 5:].shape) if not softmax else rng.normal(0, 0.3, t[..., 5:].shape)
        ts.append(t)
    for o, a, box, cid, p in assign:
        stride, anc = specs[o]
        sxy = sxy_list[o] if sxy_list else 1.0
        cx, cy, bw, bh = cxcywh(box)
        gx, gy = int(cx // stride), int(cy // stride)
        t = ts[o]
        fx, fy = cx / stride - gx, cy / stride - gy
        t[a, gy, gx, 0] = logit((fx + (sxy - 1) / 2) / sxy)
        t[a, gy, gx, 1] = logit((fy + (sxy - 1) / 2) / sxy)
        t[a, gy, gx, 2] = np.log(bw / anc[a, 0])
        t[a, gy, gx, 3] = np.log(bh / anc[a, 1])
        t[a, gy, gx, 4] = logit(p)
        t[a, gy, gx, 5:] = 0.0 if softmax else -6.0
        t[a, gy, gx, 5 + cid] = 10.0 if softmax else logit(0.95)
    if activated:
        for t in ts:
            t[..., [0, 1, 4]] = 1 / (1 + np.exp(-t[..., [0, 1, 4]]))
            if softmax:
                e = np.exp(t[..., 5:] - t[..., 5:].max(-1, keepdims=True))
                t[..., 5:] = e / e.sum(-1, keepdims=True)
            else:
                t[..., 5:] = 1 / (1 + np.exp(-t[..., 5:]))
    return ts


def px_anchors(preset, stride=None):
    p = PRESETS[preset]
    a = np.asarray(p["anchors"], np.float32).reshape(-1, 2)
    return a * stride if p["anchors_unit"] == "grid" else a


def v2_case(layout="NCHW", **kw):
    rng = np.random.default_rng(10)
    anc = px_anchors("tiny_yolov2_voc", 32)
    # 三個框放在不同的格點；第二個錨點上放重複框
    assign = [(0, i + 1, b, c, 0.9) for i, (b, c) in enumerate(TRUTH416)]
    assign += [(0, 0, b, c, 0.6) for b, c in TRUTH416]
    ts = encode_grid([(32, anc)], assign, rng, "grid", softmax=True, **kw)
    return [to_layout(ts[0], layout)]


@pytest.mark.parametrize("layout", ["NCHW", "NHWC", "NAHWC", "NHWAC"])
def test_yolov2_single_output_layouts(layout):
    arrs = v2_case(layout)
    boxes, _, ids, _ = run_decoder("yolo_grid", arrs, SIZE, params={"anchors_preset": "tiny_yolov2_voc"})
    assert_truth(boxes, ids, TRUTH416)


def test_yolov2_explicit_layout_and_guessed_preset_and_activated():
    arrs = v2_case("NCHW")
    boxes, _, ids, _ = run_decoder("yolo_grid", arrs, SIZE, params={"layout": "NCHW"})  # 沒給 anchors → 猜 tiny voc
    assert_truth(boxes, ids, TRUTH416)
    arrs = v2_case("NHWC", activated=True)
    boxes, _, ids, _ = run_decoder("yolo_grid", arrs, SIZE, params={"sigmoid": False})
    assert_truth(boxes, ids, TRUTH416)
    arrs = v2_case("NHWC", activated=True)
    boxes, _, ids, _ = run_decoder("yolo_grid", arrs, SIZE)   # auto 也認得出已 sigmoid
    assert_truth(boxes, ids, TRUTH416)


def test_k210_style_custom_anchors_grid_unit():
    rng = np.random.default_rng(11)
    anchors = [1.889, 2.5245, 2.9465, 3.94056, 3.99987, 5.3658, 5.155437, 6.92275, 6.718375, 9.01025]
    anc = np.asarray(anchors, np.float32).reshape(-1, 2) * 32
    assign = [(0, i, b, c, 0.9) for i, (b, c) in enumerate(TRUTH416)]
    ts = encode_grid([(32, anc)], assign, rng, "grid", softmax=True)
    boxes, _, ids, _ = run_decoder("yolo_grid", [to_layout(ts[0], "NCHW")], SIZE,
                                   params={"anchors": anchors, "anchors_unit": "grid"})
    assert_truth(boxes, ids, TRUTH416)
    boxes, _, ids, _ = run_decoder("yolo_grid", [to_layout(ts[0], "NCHW")], SIZE,
                                   params={"anchors_preset": "k210_facedetect"})
    assert_truth(boxes, ids, TRUTH416)


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 1, 0)])
@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
def test_yolov3_three_outputs_pixel_anchors(order, layout):
    rng = np.random.default_rng(12)
    anchors = px_anchors("yolov3_coco")
    # stride 8/16/32 對應 anchors 0-2、3-5、6-8
    specs = [(8, anchors[0:3]), (16, anchors[3:6]), (32, anchors[6:9])]
    assign = [(i, 1, b, c, 0.9) for i, (b, c) in enumerate(TRUTH416)]
    assign += [(i, 2, b, c, 0.55) for i, (b, c) in enumerate(TRUTH416)]
    ts = encode_grid(specs, assign, rng, "pixel")
    arrs = [to_layout(ts[i], layout) for i in order]
    boxes, _, ids, _ = run_decoder("yolo_grid", arrs, SIZE, params={"anchors_preset": "yolov3_coco"})
    assert_truth(boxes, ids, TRUTH416)


def test_yolov4_scale_x_y_and_tiny():
    rng = np.random.default_rng(13)
    anchors = px_anchors("yolov4_coco")
    sxy = PRESETS["yolov4_coco"]["scale_x_y"]
    specs = [(8, anchors[0:3]), (16, anchors[3:6]), (32, anchors[6:9])]
    assign = [(i, 0, b, c, 0.9) for i, (b, c) in enumerate(TRUTH416)]
    ts = encode_grid(specs, assign, rng, "pixel", sxy_list=sxy)
    arrs = [to_layout(ts[i], "NCHW") for i in (2, 1, 0)]
    boxes, _, ids, _ = run_decoder("yolo_grid", arrs, SIZE, params={"anchors_preset": "yolov4_coco"})
    assert_truth(boxes, ids, TRUTH416)
    # 沒給 scale_x_y 時中心會偏差 → 證明 scale_x_y 真的有作用
    boxes, _, _, _ = run_decoder("yolo_grid", arrs, SIZE, params={"anchors_preset": "yolov4_coco",
                                                                  "scale_x_y": 1.0})
    assert len(boxes) != 3 or not np.allclose(boxes, np.array([b for b, _ in TRUTH416]), atol=0.5)

    tiny = px_anchors("yolov4_tiny_coco")
    specs = [(16, tiny[0:3]), (32, tiny[3:6])]
    assign = [(i % 2, 2, b, c, 0.9) for i, (b, c) in enumerate(TRUTH416)]
    ts = encode_grid(specs, assign, rng, "pixel", sxy_list=[1.05, 1.05])
    arrs = [to_layout(ts[1], "NHWC"), to_layout(ts[0], "NHWC")]
    boxes, _, ids, _ = run_decoder("yolo_grid", arrs, SIZE, params={"anchors_preset": "yolov4_tiny_coco"})
    assert_truth(boxes, ids, TRUTH416)


def test_yolov3_tiny_default_guess_and_custom_masks():
    rng = np.random.default_rng(14)
    tiny = px_anchors("yolov3_tiny_coco")
    specs = [(16, tiny[0:3]), (32, tiny[3:6])]
    assign = [(i % 2, 0, b, c, 0.9) for i, (b, c) in enumerate(TRUTH416)]
    ts = encode_grid(specs, assign, rng, "pixel")
    arrs = [to_layout(t, "NCHW") for t in ts]
    boxes, _, ids, _ = run_decoder("yolo_grid", arrs, SIZE)   # 2 個輸出 → 猜 yolov3_tiny_coco
    assert_truth(boxes, ids, TRUTH416)
    boxes, _, ids, _ = run_decoder("yolo_grid", arrs, SIZE, params={
        "anchors": tiny.reshape(-1).tolist(), "anchors_unit": "pixel", "masks": [[0, 1, 2], [3, 4, 5]]})
    assert_truth(boxes, ids, TRUTH416)


def test_yolo_grid_errors():
    arrs = v2_case("NCHW")
    with pytest.raises(InferenceError):
        run_decoder("yolo_grid", arrs, SIZE, params={"anchors_preset": "nope"})
    with pytest.raises(InferenceError):
        run_decoder("yolo_grid", arrs, SIZE, params={"anchors": [1, 2, 3]})
    with pytest.raises(InferenceError):  # anchors 數量和輸出不符
        run_decoder("yolo_grid", arrs, SIZE, params={"anchors_preset": "yolov3_tiny_coco"})
    with pytest.raises(InferenceError) as e:
        run_decoder("yolo_grid", arrs, SIZE, params={"anchors_preset": "tiny_yolov2_voc"}, labels=["a", "b"])
    assert e.value.hint
    with pytest.raises(InferenceError):
        run_decoder("yolo_grid", [np.zeros((1, 10), np.float32)], SIZE)


def test_yolo_grid_match():
    cls = type(make_decoder("yolo_grid")[0])
    _, card = make_decoder("yolo_grid")
    m = cls.match([TensorInfo("a", (1, 255, 13, 13)), TensorInfo("b", (1, 255, 26, 26))], card)
    assert m.score >= 0.7 and m.params["anchors_preset"] == "yolov3_tiny_coco" and m.reason
    m = cls.match([TensorInfo(f"o{i}", (1, 255, s, s)) for i, s in enumerate((13, 26, 52))], card)
    assert m.params["anchors_preset"] == "yolov3_coco"
    m = cls.match([TensorInfo("a", (1, 125, 13, 13))], card)   # YOLOv2 VOC：5×(5+20)
    assert m.score >= 0.7 and m.params["anchors_preset"] == "tiny_yolov2_voc"
    m = cls.match([TensorInfo("a", (1, 30, 7, 10))], card)   # 1 類別、K210 常見大小
    assert m.score > 0.5 and m.params["anchors_preset"] == "k210_facedetect"
    assert cls.match([TensorInfo("a", (1, 3, 13, 13, 85))], card).score > 0.5
    assert cls.match([TensorInfo("a", (1, 13, 13, 3, 85))], card).score > 0.5
    assert cls.match([TensorInfo("a", (1, 13, 13, 255))], card).score > 0.5
    assert cls.match([TensorInfo("a", (1, 84, 8400))], card).score == 0
    assert cls.match([TensorInfo("a", (1, None, None, None))], card).score == 0
    # 說明檔給了 anchors → 不再建議預設值
    _, card2 = make_decoder("yolo_grid")
    card2.decoder_params = {"anchors_preset": "yolov3_coco"}
    assert "anchors_preset" not in cls.match(
        [TensorInfo(f"o{i}", (1, 255, s, s)) for i, s in enumerate((13, 26, 52))], card2).params


def test_parse_grid_and_presets():
    assert parse_grid((255, 13, 13), 3, 80)[:2] == ("NCHW", 3)
    assert parse_grid((13, 13, 255), 3, 80)[:2] == ("NHWC", 3)
    assert parse_grid((52, 52, 30), 5, None)[0] == "NHWC"   # 格點比通道多也不會判錯
    assert default_preset(1) == "tiny_yolov2_voc" and default_preset(3) == "yolov3_coco"
    for p in PRESETS.values():
        assert len(p["anchors"]) % 2 == 0


# ---- TFLite SSD -------------------------------------------------------------------

def ssd_arrays(n=10, count=3):
    boxes = np.zeros((1, n, 4), np.float32)
    classes = np.zeros((1, n), np.float32)
    scores = np.zeros((1, n), np.float32)
    for i, (b, c) in enumerate(TRUTH[:count]):
        x1, y1, x2, y2 = (v / 640 for v in b)
        boxes[0, i] = [y1, x1, y2, x2]
        classes[0, i] = c
        scores[0, i] = 0.9 - 0.1 * i
    scores[0, count:] = 0.05
    return boxes, classes, scores, np.array([float(count)], np.float32)


def test_tflite_ssd_standard_order():
    b, c, s, n = ssd_arrays()
    boxes, _, ids, out = run_decoder("tflite_ssd", [b, c, s, n], (640, 640))
    assert out.nms_done
    assert_truth(boxes, ids)


def test_tflite_ssd_orders_by_name_and_values_and_offset():
    b, c, s, n = ssd_arrays()
    # TF Hub 風格順序：scores, boxes, count, classes（有名稱）
    arrays = [s, b, n, c]
    infos = [TensorInfo("scores", s.shape), TensorInfo("boxes", b.shape), TensorInfo("count", n.shape),
             TensorInfo("classes", c.shape)]
    boxes, _, ids, _ = run_decoder("tflite_ssd", arrays, (640, 640), infos=infos)
    assert_truth(boxes, ids)
    # 沒有名稱：用數值判斷（類別是整數）
    for order in ([b, s, c, n], [n, c, s, b]):
        boxes, _, ids, _ = run_decoder("tflite_ssd", order, (640, 640))
        assert_truth(boxes, ids)
    boxes, _, ids, _ = run_decoder("tflite_ssd", [b, c, s, n], (640, 640), params={"class_offset": 1})
    assert_truth(boxes, ids - 1)
    # count 比 N 小時，後面的資料不算
    b2, c2, s2, n2 = ssd_arrays(count=2)
    s2[0, 2] = 0.99
    boxes, _, _, _ = run_decoder("tflite_ssd", [b2, c2, s2, n2], (640, 640))
    assert len(boxes) == 2


def test_tflite_ssd_match_and_error():
    cls = type(make_decoder("tflite_ssd")[0])
    _, card = make_decoder("tflite_ssd")
    infos = infos_for(*ssd_arrays())
    assert cls.match(infos, card).score >= 0.85
    assert cls.match(infos[:3], card).score == 0
    assert cls.match([TensorInfo("x", (1, 10, 4))] * 4, card).score == 0
    with pytest.raises(InferenceError):
        run_decoder("tflite_ssd", [np.zeros((1, 5), np.float32)] * 4, (640, 640))


# ---- 分類 -------------------------------------------------------------------------

def test_classify_softmax_and_probabilities():
    logits = np.array([[0.2, 3.0, -1.0, 0.5]], np.float32)
    _, _, _, out = run_decoder("classify", [logits], (224, 224))
    assert out.task == "classify" and out.class_scores.argmax() == 1
    assert out.class_scores.sum() == pytest.approx(1.0, abs=1e-5)
    probs = np.array([[0.1, 0.7, 0.1, 0.1]], np.float32)
    _, _, _, out = run_decoder("classify", [probs], (224, 224))
    assert out.class_scores.tolist() == pytest.approx(probs[0].tolist())
