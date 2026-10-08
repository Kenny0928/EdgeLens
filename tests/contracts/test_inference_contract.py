"""每個註冊的推論後端與解碼器都必須通過的檢查。"""

from __future__ import annotations

import numpy as np
import pytest

from edgelens.core.modelcard import ModelCard
from edgelens.core.registry import discover
from edgelens.i18n import LANGUAGES, has
from edgelens.inference.base import (
    BACKENDS,
    DECODERS,
    Backend,
    DecodeContext,
    DecodeOutput,
    Decoder,
    MatchResult,
    TensorInfo,
)

discover()
BACKEND_KINDS = BACKENDS.kinds()
DECODER_KINDS = DECODERS.kinds()

SAMPLE_OUTPUTS = [
    [TensorInfo("out", (1, 10))],
    [TensorInfo("out", (1, 84, 8400))],
    [TensorInfo("out", (1, 25200, 85))],
    [TensorInfo("out", (1, 300, 6))],
    [TensorInfo("a", (1, 255, 13, 13)), TensorInfo("b", (1, 255, 26, 26))],
    [TensorInfo("a", (1, 10, 4)), TensorInfo("b", (1, 10)), TensorInfo("c", (1, 10)), TensorInfo("d", (1,))],
    [TensorInfo("dyn", (None, None, None))],
    [],
]


@pytest.mark.parametrize("kind", BACKEND_KINDS)
def test_backend_attributes(kind):
    cls = BACKENDS.get(kind)
    assert issubclass(cls, Backend) and cls.kind == kind
    assert cls.input_mode in ("tensor", "image", "device")
    assert all(e.startswith(".") and e == e.lower() for e in cls.extensions)
    for lang in LANGUAGES:
        assert has(cls.display_name_key, lang), f"{kind}: 缺少 {lang} 翻譯 {cls.display_name_key!r}"
    ok, reason = cls.available()
    assert isinstance(ok, bool) and isinstance(reason, str)
    if not ok:
        assert reason, "不能用時要說明原因"


@pytest.mark.parametrize("kind", DECODER_KINDS)
def test_decoder_attributes(kind):
    cls = DECODERS.get(kind)
    assert issubclass(cls, Decoder) and cls.kind == kind
    assert cls.task in ("detect", "classify")
    for lang in LANGUAGES:
        assert has(cls.display_name_key, lang), f"{kind}: 缺少 {lang} 翻譯 {cls.display_name_key!r}"


@pytest.mark.parametrize("kind", DECODER_KINDS)
def test_decoder_match_never_raises(kind, tmp_path):
    cls = DECODERS.get(kind)
    card = ModelCard(name="x", file=str(tmp_path / "x.onnx"))
    for outputs in SAMPLE_OUTPUTS:
        res = cls.match(outputs, card)
        assert isinstance(res, MatchResult)
        assert 0.0 <= res.score <= 1.0
        if res.score > 0:
            assert res.reason, f"{kind} 判斷符合時要說明原因"


@pytest.mark.parametrize("kind", DECODER_KINDS)
def test_decoder_handles_matching_random_output(kind, tmp_path):
    """對自己判斷符合的形狀，用隨機數值解碼不能出錯，且輸出格式正確。"""
    cls = DECODERS.get(kind)
    card = ModelCard(name="x", file=str(tmp_path / "x.onnx"))
    rng = np.random.default_rng(0)
    tried = 0
    for outputs in SAMPLE_OUTPUTS:
        if not outputs or any(d is None for o in outputs for d in o.shape):
            continue
        m = cls.match(outputs, card)
        if m.score < 0.5:
            continue
        tried += 1
        arrays = [rng.random(o.shape, dtype=np.float32) for o in outputs]
        dec = cls(card, m.params)
        out = dec.decode(arrays, DecodeContext(card=card, input_size=(416, 416), conf_threshold=0.5,
                                               output_infos=outputs))
        assert isinstance(out, DecodeOutput) and out.task == cls.task
        if out.task == "detect":
            n = len(out.scores)
            assert out.boxes.shape == (n, 4) and out.class_ids.shape == (n,)
            assert (out.scores >= 0.5).all()
        else:
            assert out.class_scores.ndim == 1 and len(out.class_scores) > 0
    if tried == 0:
        pytest.skip("這組範例形狀都不符合此解碼器；由解碼器自己的單元測試驗證")
