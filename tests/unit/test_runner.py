"""InferenceRunner：後端選擇、門檻、手動指定解碼器、量化、Darknet 合併、錯誤訊息。"""

from __future__ import annotations

import time
from typing import ClassVar

import numpy as np
import pytest

from edgelens.core.frame import Frame
from edgelens.core.modelcard import ModelCard
from edgelens.core.registry import discover
from edgelens.inference.base import Backend, InferenceError, TensorInfo
from edgelens.inference.runner import InferenceRunner, load_model_file
from tests.fixtures.make_onnx import LABELS, generate_all

discover()
W, H = 128, 96


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    return generate_all(tmp_path_factory.mktemp("runner_onnx"))


def _frame():
    return Frame(np.full((H, W, 3), 100, np.uint8), 5, time.monotonic(), "t")


class StubBackend(Backend):
    """回傳固定輸出的假後端，用來測 runner 對量化與 Darknet 輸出的處理。"""

    kind: ClassVar[str] = "stub"
    input_mode = "tensor"

    def __init__(self, card, outs, out_infos, in_info=None, meta=None):
        super().__init__(card)
        self._outs, self._infos = outs, out_infos
        self._in = in_info or TensorInfo("in", (1, 3, 64, 64))
        self._meta = meta or {}

    def load(self):
        pass

    def inputs(self):
        return [self._in]

    def outputs(self):
        return self._infos

    def infer(self, tensor):
        return [o.copy() for o in self._outs]

    def metadata(self):
        return self._meta


def _stub_runner(card, outs, infos, meta=None):
    r = InferenceRunner(card)
    b = StubBackend(card, outs, infos, meta=meta)
    r._backend = b
    r._darknet = bool((meta or {}).get("darknet"))
    r._labels = list(card.labels)
    r.info.loaded = True
    from edgelens.inference.runner import RunnerInfo

    r.info = RunnerInfo(loaded=True, inputs=b.inputs(), outputs=infos, labels=list(card.labels))
    r._prepare_decoder(None, infos)
    return r


def test_backend_choices_lists_unavailable_with_reason(models):
    choices = InferenceRunner.backend_choices(ModelCard.load(models["classify"]))
    kinds = [c.kind for c in choices]
    assert "onnxrt" in kinds and "cvdnn" in kinds
    assert all(c.name for c in choices) and all(c.reason for c in choices if not c.ok)
    assert choices[0].kind == "onnxrt"          # 優先順位最高且可用


def test_auto_picks_highest_priority_and_info(models):
    r = InferenceRunner(ModelCard.load(models["yolo_t"]))
    r.load()
    i = r.info
    assert i.loaded and i.backend == "onnxrt" and i.mode == "tensor" and i.task == "detect"
    assert i.decoder == "yolo_t" and i.decoder_reason and i.guesses and i.load_ms > 0
    assert "64×64" in i.input_desc and i.inputs[0].shape == (1, 3, 64, 64) and i.outputs


def test_unknown_backend_and_extension(tmp_path, models):
    with pytest.raises(InferenceError) as ei:
        InferenceRunner(ModelCard.load(models["classify"]), "nope").load()
    assert ei.value.hint
    p = tmp_path / "x.xyz"
    p.write_bytes(b"1")
    with pytest.raises(InferenceError) as ei:
        InferenceRunner(ModelCard.load(p)).load()
    assert ei.value.hint
    with pytest.raises(InferenceError):          # 後端不吃這種副檔名
        InferenceRunner(ModelCard.load(models["classify"]), "haar").load()


def test_run_before_load_raises(models):
    with pytest.raises(InferenceError):
        InferenceRunner(ModelCard.load(models["classify"])).run(_frame())


def test_thresholds_can_change_between_runs(models):
    card = ModelCard.load(models["yolo_t"])
    card.labels = list(LABELS)
    r = InferenceRunner(card)
    r.load()
    assert len(r.run(_frame()).detections) == 1
    r.conf_threshold = 0.95
    assert r.run(_frame()).detections == []
    r.conf_threshold = 0.25
    r.max_det = 0
    assert r.run(_frame()).detections == []


def test_nms_and_class_agnostic(models):
    """同一位置兩個類別：預設各自 NMS 兩個都留，class_agnostic 只留分數高的。"""
    card = ModelCard(name="n", file="n.onnx", labels=list(LABELS))
    a = np.zeros((1, 252, 8), np.float32)
    a[0, :, 4] = 0.01
    for row, (cls, sc) in ((10, (0, 0.9)), (11, (1, 0.8))):
        a[0, row, :4] = (32, 24, 24, 16)
        a[0, row, 4] = 0.95
        a[0, row, 5 + cls] = sc
    r = _stub_runner(card, [a], [TensorInfo("o", (1, 252, 8))])
    assert len(r.run(_frame()).detections) == 2
    r.class_agnostic = True
    res = r.run(_frame())
    assert len(res.detections) == 1 and res.detections[0].class_id == 0


def _two_class_runner(card=None):
    """同一位置兩個類別（0 分數 0.9、1 分數 0.8）的假模型。"""
    card = card or ModelCard(name="n", file="n.onnx", labels=list(LABELS))
    a = np.zeros((1, 252, 8), np.float32)
    a[0, :, 4] = 0.01
    for row, (cls, sc) in ((10, (0, 0.9)), (11, (1, 0.8))):
        a[0, row, :4] = (32, 24, 24, 16)
        a[0, row, 4] = 0.95
        a[0, row, 5 + cls] = sc
    return _stub_runner(card, [a], [TensorInfo("o", (1, 252, 8))])


def test_class_filter_keeps_only_chosen_classes():
    r = _two_class_runner()
    r.class_filter = frozenset({1})
    assert [d.class_id for d in r.run(_frame()).detections] == [1]
    r.class_filter = frozenset({2})
    assert r.run(_frame()).detections == []
    r.class_filter = None
    assert len(r.run(_frame()).detections) == 2


def test_class_filter_applies_before_nms_and_max_det():
    """沒選的類別不能在不分類別 NMS 時壓掉選了的框，也不能占掉 max_det 的名額。"""
    r = _two_class_runner()
    r.class_agnostic = True
    r.max_det = 1
    r.class_filter = frozenset({1})
    res = r.run(_frame())
    assert [d.class_id for d in res.detections] == [1]


def test_class_filter_from_card():
    card = ModelCard(name="n", file="n.onnx", labels=list(LABELS), class_filter=[0])
    r = _two_class_runner(card)
    assert r.class_filter == frozenset({0})
    assert [d.class_id for d in r.run(_frame()).detections] == [0]


def test_load_model_file_localizes_labels(tmp_path):
    from edgelens import i18n

    (tmp_path / "labels.txt").write_text("cup\nchair\n", encoding="utf-8")
    (tmp_path / "labels.zh_TW.txt").write_text("杯子\n椅子\n", encoding="utf-8")
    (tmp_path / "m.card.json").write_text(
        '{"version": 1, "file": "m.onnx", "labels_file": "labels.txt", '
        '"labels_i18n": {"zh_TW": "labels.zh_TW.txt"}}', encoding="utf-8")
    card = load_model_file(tmp_path / "m.card.json")
    assert card.labels == ["杯子", "椅子"] and card.labels_base == ["cup", "chair"]
    i18n.set_language("en")
    assert load_model_file(tmp_path / "m.card.json").labels == ["cup", "chair"]


def test_set_decoder_manual(models):
    r = InferenceRunner(ModelCard.load(models["yolo_flat"]))
    r.load()
    r.set_decoder("yolo_flat", {"objectness": False})
    assert r.info.decoder == "yolo_flat" and not r.info.ambiguous and r.info.decoder_reason
    r.run(_frame())
    assert r.info.decoder == "yolo_flat"          # 手動指定後不再被自動判斷覆蓋
    with pytest.raises(InferenceError):
        r.set_decoder("no_such")


def test_ambiguous_flag():
    from edgelens.inference.autodetect import FormatGuess
    from edgelens.inference.runner import _is_ambiguous

    assert _is_ambiguous([FormatGuess("a", 0.3), FormatGuess("b", 0.1)])
    assert _is_ambiguous([FormatGuess("a", 0.8), FormatGuess("b", 0.8)])
    assert not _is_ambiguous([FormatGuess("a", 0.9), FormatGuess("b", 0.4)])


def test_quantized_classifier_probabilities_not_softmaxed_twice():
    """量化機率總和 0.94：不能再做一次 softmax（分數會全部變成約 0.001）。"""
    card = ModelCard(name="q", file="q.tflite")
    probs = np.zeros((1, 6), np.uint8)
    probs[0, 3] = 200           # scale 1/256 → 0.78
    probs[0, 1] = 40
    info = TensorInfo("p", (1, 6), "uint8", quant=(1 / 256, 0))
    r = _stub_runner(card, [probs], [info])
    res = r.run(_frame())
    assert res.kind == "classify" and res.top_k[0].class_id == 3
    assert res.top_k[0].score == pytest.approx(200 / 256, abs=1e-3)
    assert len(res.top_k) == 5


def test_darknet_outputs_merged_and_normalized():
    """OpenCV 解碼過的 Darknet 輸出：多個 [N, 5+C] 合併，中心／寬高是 0–1 比例。"""
    card = ModelCard(name="d", file="d.weights", labels=list(LABELS))
    a = np.zeros((12, 8), np.float32)
    b = np.zeros((48, 8), np.float32)
    b[7] = (0.5, 0.5, 0.25, 0.25, 0.9, 0.0, 0.8, 0.0)
    infos = [TensorInfo("l1", (12, 8)), TensorInfo("l2", (48, 8))]
    r = _stub_runner(card, [a, b], infos, meta={"darknet": True})
    card.input.resize = "stretch"
    res = r.run(_frame())
    assert r.info.decoder == "yolo_flat"
    assert len(res.detections) == 1
    d = res.detections[0]
    # 正規化 → 64x64 輸入 → stretch 回 128x96
    np.testing.assert_allclose(d.box, (48, 36, 80, 60), atol=1.0)
    assert d.label == "beta"


def test_raw_not_available_for_image_mode():
    from edgelens.inference.backends.haar import builtin_cascades

    r = InferenceRunner(ModelCard.load(builtin_cascades()["eye"]))
    r.load()
    with pytest.raises(InferenceError):
        r.raw(_frame())
    r.warmup()                                      # 不是 tensor 模式：直接略過


def test_load_model_file_darknet_pairing(tmp_path):
    (tmp_path / "net.weights").write_bytes(b"w")
    (tmp_path / "other.cfg").write_text("[net]\nwidth=416\nheight=416\n")
    card = load_model_file(tmp_path / "net.weights")
    assert card.extra_files["config"].endswith("other.cfg") and card.input.resize == "stretch"
    (tmp_path / "net.cfg").write_text("[net]\n")
    assert load_model_file(tmp_path / "net.cfg").extension == ".weights"
    assert load_model_file(tmp_path / "net.weights").extra_files["config"].endswith("net.cfg")


def test_close_is_idempotent(models):
    r = InferenceRunner(ModelCard.load(models["classify"]))
    r.load()
    r.close()
    r.close()
    assert not r.info.loaded
