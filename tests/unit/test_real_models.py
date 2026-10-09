"""用真實模型驗證解碼器（先執行 `python tools/fetch_test_models.py` 下載模型）。

    pytest -m network tests/unit/test_real_models.py

測試圖 dog.jpg（768x576）有三樣東西：狗、腳踏車、卡車（部分模型會說是 car）。
參考框是用 YOLO 原作者的結果目測得來，容許 IoU > 0.5。模型沒下載就 skip。
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from edgelens.core.frame import Frame
from edgelens.core.registry import discover
from edgelens.inference.base import BACKENDS
from edgelens.inference.nms import iou
from edgelens.inference.runner import InferenceRunner, load_model_file

pytestmark = pytest.mark.network
discover()

MODELS = Path(__file__).resolve().parents[1] / "fixtures" / "models"
DOG = np.array([[130, 220, 312, 540]], np.float32)
BICYCLE = np.array([[125, 125, 570, 420]], np.float32)
TRUCK = np.array([[468, 78, 688, 170]], np.float32)

# (資料夾, 說明檔, 解碼器, 要找到的 [(可接受的名稱, 參考框)])
DETECTORS = [
    ("yolov8n", "yolov8n.card.json", "yolo_t", [({"dog"}, DOG), ({"bicycle"}, BICYCLE), ({"truck", "car"}, TRUCK)]),
    ("yolov5n", "yolov5n.card.json", "yolo_flat", [({"dog"}, DOG), ({"truck", "car"}, TRUCK)]),
    ("yolov5s", "yolov5s.card.json", "yolo_flat", [({"dog"}, DOG), ({"bicycle"}, BICYCLE), ({"truck", "car"}, TRUCK)]),
    ("yolov10n", "yolov10n.card.json", "end2end", [({"dog"}, DOG), ({"bicycle"}, BICYCLE), ({"truck", "car"}, TRUCK)]),
    ("yolox_nano", "yolox_nano.card.json", "yolox_raw", [({"dog"}, DOG), ({"bicycle"}, BICYCLE), ({"truck", "car"}, TRUCK)]),
    ("yolox_s", "yolox_s.card.json", "yolox_raw", [({"dog"}, DOG), ({"bicycle"}, BICYCLE), ({"truck", "car"}, TRUCK)]),
    ("tiny_yolov2", "tinyyolov2-8.card.json", "yolo_grid", [({"dog"}, DOG), ({"car", "bus"}, TRUCK)]),
    ("yolov4_tiny_darknet", "yolov4-tiny.card.json", "yolo_flat", [({"dog"}, DOG), ({"truck", "car"}, TRUCK)]),
    ("ssd_mobilenet_tflite", "detect.card.json", "tflite_ssd", [({"bicycle"}, BICYCLE)]),
]
CLASSIFIERS = [
    ("mobilenetv2", "mobilenetv2-12.card.json", {"malamute", "eskimo dog", "siberian husky"}),
    ("mobilenet_tflite", "mobilenet_v1_1.0_224_quant.card.json", None),
]


def _image(name: str) -> np.ndarray:
    p = MODELS / "images" / name
    if not p.is_file():
        pytest.skip("沒有測試圖，先執行 tools/fetch_test_models.py")
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def _card(folder: str, card_name: str):
    p = MODELS / folder / card_name
    if not p.is_file():
        pytest.skip(f"沒有下載 {folder}，先執行 tools/fetch_test_models.py")
    card = load_model_file(p)
    if not card.model_path.is_file():
        pytest.skip(f"{folder} 的模型檔不完整")
    return card


def _backends_for(card) -> list[str]:
    kinds = [c.kind for c in InferenceRunner.backend_choices(card) if c.ok and c.kind in ("onnxrt", "cvdnn", "tflite")]
    # OpenCV DNN 讀不了量化（uint8）的 TFLite，只拿 tflite 後端驗證；cvdnn 的錯誤訊息另外測
    return [k for k in kinds if not (card.extension == ".tflite" and k == "cvdnn")]


def _cases(table):
    for row in table:
        card_path = MODELS / row[0] / row[1]
        kinds = _backends_for(load_model_file(card_path)) if card_path.is_file() else ["auto"]
        for k in kinds or ["auto"]:
            yield pytest.param(*row, k, id=f"{row[0]}-{k}")


@pytest.mark.parametrize(("folder", "card_name", "decoder", "expect", "backend"), list(_cases(DETECTORS)))
def test_detector_on_dog_photo(folder, card_name, decoder, expect, backend):
    card = _card(folder, card_name)
    img = _image("dog.jpg")
    r = InferenceRunner(card, None if backend == "auto" else backend)
    r.load()
    r.warmup(1)
    t0 = time.perf_counter()
    res = r.run(Frame(img, 1, time.monotonic(), "photo"))
    ms = (time.perf_counter() - t0) * 1000
    assert r.info.decoder == decoder, f"自動判斷 {r.info.decoder}，預期 {decoder}"
    assert res.kind == "detect" and res.image_size == (img.shape[1], img.shape[0])
    summary = [(d.label, [int(v) for v in d.box], round(d.score, 2)) for d in res.detections]
    for names, ref in expect:
        best = 0.0
        for d in res.detections:
            if d.label.lower() in names:
                best = max(best, float(iou(np.array([d.box], np.float32), ref)[0, 0]))
        assert best > 0.5, f"{folder}/{backend}：找不到 {names}（最佳 IoU {best:.2f}）；結果 {summary}，{ms:.0f} ms"
    for d in res.detections:      # 所有框都在畫面內且大小合理
        assert 0 <= d.box[0] < d.box[2] <= img.shape[1] and 0 <= d.box[1] < d.box[3] <= img.shape[0]


@pytest.mark.parametrize(("folder", "card_name", "keywords"), CLASSIFIERS)
def test_classifier_on_dog_photo(folder, card_name, keywords):
    card = _card(folder, card_name)
    img = _image("dog.jpg")
    r = InferenceRunner(card)
    r.load()
    res = r.run(Frame(img, 1, time.monotonic(), "photo"))
    assert res.kind == "classify" and len(res.top_k) == 5
    assert res.top_k[0].score > 0.05, [(c.label, round(c.score, 3)) for c in res.top_k]
    assert sum(c.score for c in res.top_k) <= 1.05
    if keywords:
        assert any(c.label.lower() in keywords for c in res.top_k), [(c.label, round(c.score, 3)) for c in res.top_k]


def test_mobilenetv2_same_on_both_onnx_backends():
    card = _card("mobilenetv2", "mobilenetv2-12.card.json")
    img = _image("dog.jpg")
    tops = []
    for kind in ("onnxrt", "cvdnn"):
        if kind not in _backends_for(card):
            pytest.skip(f"沒有 {kind}")
        r = InferenceRunner(card, kind)
        r.load()
        tops.append(r.run(Frame(img, 1, time.monotonic(), "photo")).top_k[0])
    assert tops[0].class_id == tops[1].class_id and abs(tops[0].score - tops[1].score) < 0.02


def test_haar_finds_face_in_messi_photo():
    from edgelens.inference.backends.haar import builtin_cascades

    img = _image("messi5.jpg")
    r = InferenceRunner(load_model_file(builtin_cascades()["frontalface_default"]))
    r.load()
    res = r.run(Frame(img, 1, time.monotonic(), "photo"))
    assert len(res.detections) >= 1
    d = res.detections[0]
    assert d.width > 20 and d.box[3] < img.shape[0] * 0.8      # 臉在畫面上半部


def test_tflite_backend_registered_and_used():
    if not BACKENDS.get("tflite").available()[0]:
        pytest.skip("沒有 TFLite 執行環境")
    card = _card("ssd_mobilenet_tflite", "detect.card.json")
    r = InferenceRunner(card)
    r.load()
    assert r.info.backend == "tflite" and r.info.inputs[0].dtype == "uint8"


def test_cvdnn_quantized_tflite_fails_with_readable_message():
    card = _card("ssd_mobilenet_tflite", "detect.card.json")
    if "cvdnn" not in [c.kind for c in InferenceRunner.backend_choices(card) if c.ok]:
        pytest.skip("這版 OpenCV 不能讀 TFLite")
    with pytest.raises(Exception) as ei:       # 新版 OpenCV 若能讀，就不會丟錯
        InferenceRunner(card, "cvdnn").load()
    assert getattr(ei.value, "hint", "")
