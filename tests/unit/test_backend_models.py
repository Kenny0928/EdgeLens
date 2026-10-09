"""onnxrt、cvdnn 後端 + runner：把假 ONNX 模型從 Frame 跑到 InferenceResult。"""

from __future__ import annotations

import time

import numpy as np
import pytest

from edgelens.core.frame import Frame
from edgelens.core.modelcard import ModelCard
from edgelens.core.registry import discover
from edgelens.inference.base import BACKENDS, InferenceError
from edgelens.inference.runner import InferenceRunner
from tests.fixtures.make_onnx import LABELS, NAMES, SPECS, generate_all

discover()
BACKEND_KINDS = [k for k in ("onnxrt", "cvdnn") if BACKENDS.get(k).available()[0]]
W, H = 128, 96          # 測試畫面；letterbox 到 64x64：scale 0.5、上下各補 8


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    return generate_all(tmp_path_factory.mktemp("fake_onnx"))


def _frame(color=True):
    img = np.full((H, W, 3), 100, np.uint8)
    if not color:
        img = img[:, :, 0]
    return Frame(img, 1, time.monotonic(), "test:0")


def _to_orig(box):
    x1, y1, x2, y2 = box
    return (x1 / 0.5, (y1 - 8) / 0.5, x2 / 0.5, (y2 - 8) / 0.5)


def _runner(path, backend, name):
    card = ModelCard.load(path)
    if name != "ssd":
        card.labels = list(LABELS)
    r = InferenceRunner(card, backend)
    r.load()
    return r


@pytest.mark.parametrize("backend", BACKEND_KINDS)
@pytest.mark.parametrize("name", NAMES)
def test_fake_model_end_to_end(models, backend, name):
    exp = SPECS[name][2]
    r = _runner(models[name], backend, name)
    res = r.run(_frame())
    assert r.info.backend == backend and r.info.decoder == exp.decoder
    assert set(res.timings_ms) >= {"preprocess", "infer", "postprocess", "total"}
    assert res.image_size == (W, H) and res.where == "host"
    if exp.kind == "classify":
        assert res.kind == "classify"
        assert res.top_k[0].class_id == exp.class_id and res.top_k[0].label == LABELS[exp.class_id]
        assert res.top_k[0].score >= exp.min_score
        assert [c.score for c in res.top_k] == sorted((c.score for c in res.top_k), reverse=True)
        return
    assert len(res.detections) == 1
    d = res.detections[0]
    assert d.class_id == exp.class_id and d.score >= exp.min_score
    assert d.label == (str(exp.class_id) if name == "ssd" else LABELS[exp.class_id])
    np.testing.assert_allclose(d.box, _to_orig(exp.box), atol=1.5)
    r.close()


@pytest.mark.parametrize("backend", BACKEND_KINDS)
def test_gray_frame_and_warmup(models, backend):
    r = _runner(models["yolo_flat"], backend, "yolo_flat")
    r.warmup(2)
    res = r.run(_frame(color=False))
    assert len(res.detections) == 1


@pytest.mark.parametrize("backend", BACKEND_KINDS)
def test_raw_outputs(models, backend):
    r = _runner(models["yolo_t"], backend, "yolo_t")
    raw = r.raw(_frame())
    assert len(raw) == 1 and raw[0][1].shape == (1, 7, 84)


def test_onnx_metadata_names(tmp_path):
    """ultralytics 把 names 寫成 Python dict 字串：要安全解析成清單。"""
    import onnx

    from tests.fixtures.make_onnx import build_model

    if "onnxrt" not in BACKEND_KINDS:
        pytest.skip("沒有 onnxruntime")
    m = onnx.load_from_string(build_model("yolo_t"))
    for k, v in (("names", "{0: 'x', 1: 'y', 2: 'z'}"), ("stride", "32")):
        e = m.metadata_props.add()
        e.key, e.value = k, v
    p = tmp_path / "meta.onnx"
    p.write_bytes(m.SerializeToString())
    r = InferenceRunner(ModelCard.load(p), "onnxrt")
    r.load()
    assert r.info.labels == ["x", "y", "z"]
    assert r.run(_frame()).detections[0].label == "y"
    assert "ONNX Runtime" in r.info.backend_desc and "CPUExecutionProvider" in r.info.backend_desc


def test_parse_names_safe():
    from edgelens.inference.backends._common import parse_names

    assert parse_names("{0: 'a', 1: 'b'}") == ["a", "b"]
    assert parse_names("['a', 'b']") == ["a", "b"]
    assert parse_names("__import__('os').system('echo hi')") == []
    assert parse_names("{0: 'a', 2: 'b'}") == []
    assert parse_names("not python") == []


@pytest.mark.parametrize("backend", BACKEND_KINDS)
def test_broken_and_missing_model(tmp_path, backend):
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not an onnx file")
    with pytest.raises(InferenceError) as ei:
        InferenceRunner(ModelCard.load(bad), backend).load()
    assert ei.value.message and ei.value.hint
    with pytest.raises(InferenceError):
        InferenceRunner(ModelCard.load(tmp_path / "nothing.onnx"), backend).load()


def test_label_count_mismatch_gives_student_message(models):
    card = ModelCard.load(models["yolo_flat"])
    card.labels = ["only", "two"]
    r = InferenceRunner(card, BACKEND_KINDS[0])
    r.load()
    with pytest.raises(InferenceError) as ei:
        r.run(_frame())
    assert ei.value.message and ei.value.hint


def test_cvdnn_input_size_from_card(models):
    if "cvdnn" not in BACKEND_KINDS:
        pytest.skip("無 cvdnn")
    card = ModelCard.load(models["classify"])
    card.input.size = (64, 64)
    r = InferenceRunner(card, "cvdnn")
    r.load()
    assert r.info.inputs[0].shape == (1, 3, 64, 64)
    assert r.info.outputs[0].shape == (1, 5)
