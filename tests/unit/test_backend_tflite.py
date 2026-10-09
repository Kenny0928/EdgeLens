"""TFLite 後端：可用性與錯誤訊息（真實模型由 test_real_models 驗證）。"""

from __future__ import annotations

import pytest

from edgelens.core.modelcard import ModelCard
from edgelens.core.registry import discover
from edgelens.inference.backends import tflite as tfl
from edgelens.inference.base import BACKENDS, InferenceError
from edgelens.inference.runner import InferenceRunner

discover()


def test_available_reports_reason(monkeypatch):
    monkeypatch.setattr(tfl, "_find_interpreter_module", lambda: None)
    ok, reason = tfl.TfliteBackend.available()
    assert not ok and reason
    assert "ONNX" in tfl.tr("backend.tflite.missing_hint")


def test_missing_runtime_load_error_has_hint(monkeypatch, tmp_path):
    monkeypatch.setattr(tfl, "_find_interpreter_module", lambda: None)
    p = tmp_path / "m.tflite"
    p.write_bytes(b"x")
    with pytest.raises(InferenceError) as ei:
        tfl.TfliteBackend(ModelCard.load(p)).load()
    assert ei.value.hint


def test_missing_runtime_not_chosen_and_listed(monkeypatch, tmp_path):
    monkeypatch.setattr(tfl, "_find_interpreter_module", lambda: None)
    p = tmp_path / "m.tflite"
    p.write_bytes(b"x")
    choices = InferenceRunner.backend_choices(ModelCard.load(p))
    t = next(c for c in choices if c.kind == "tflite")
    assert not t.ok and t.reason
    with pytest.raises(InferenceError):
        InferenceRunner(ModelCard.load(p), "tflite").load()


@pytest.mark.skipif(not BACKENDS.get("tflite").available()[0], reason="沒有 TFLite 執行環境")
def test_garbage_model_gives_student_message(tmp_path):
    p = tmp_path / "bad.tflite"
    p.write_bytes(b"this is not a tflite file at all")
    with pytest.raises(InferenceError) as ei:
        InferenceRunner(ModelCard.load(p), "tflite").load()
    assert ei.value.message and ei.value.hint


def test_labels_from_embedded_zip():
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("labels.txt", "cat\ndog\n")
    assert tfl._labels_from_metadata(b"FAKEMODEL" + buf.getvalue()) == ["cat", "dog"]
    assert tfl._labels_from_metadata(b"nothing") == []
