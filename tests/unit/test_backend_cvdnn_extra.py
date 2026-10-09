"""cvdnn 的 Darknet 設定檔解析與缺設定檔的訊息。"""

from __future__ import annotations

import pytest

from edgelens.core.modelcard import ModelCard
from edgelens.core.registry import discover
from edgelens.inference.backends.cvdnn import OpenCvDnnBackend, cfg_input_size
from edgelens.inference.base import InferenceError
from edgelens.inference.runner import InferenceRunner

discover()


def test_cfg_input_size():
    assert cfg_input_size("[net]\nbatch=1\nwidth=416\nheight=320\n[convolutional]\nwidth=9") == (416, 320)
    assert cfg_input_size("nothing") is None


def test_weights_without_cfg_cannot_load(tmp_path):
    p = tmp_path / "a.weights"
    p.write_bytes(b"x")
    card = ModelCard.load(p)
    assert not OpenCvDnnBackend.can_load(card)
    choices = InferenceRunner.backend_choices(card)
    assert choices and not choices[0].ok and choices[0].reason
    with pytest.raises(InferenceError) as ei:
        InferenceRunner(card).load()
    assert ei.value.hint


def test_garbage_darknet_gives_student_message(tmp_path):
    (tmp_path / "a.weights").write_bytes(b"garbage")
    (tmp_path / "a.cfg").write_text("[net]\nwidth=32\nheight=32\n")
    with pytest.raises(InferenceError) as ei:
        InferenceRunner(ModelCard.load(tmp_path / "a.weights")).load()
    assert ei.value.message and ei.value.hint
