"""每個註冊的處理步驟都必須通過的檢查。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from edgelens.core.controls import ControlSpec, ControlType
from edgelens.core.registry import discover
from edgelens.i18n import LANGUAGES, has
from edgelens.processing.pipeline import CATEGORIES, STEPS, Pipeline, ProcessingStep, StepContext

discover()
KINDS = STEPS.kinds()


@pytest.mark.parametrize("kind", KINDS)
def test_class_attributes(kind):
    cls = STEPS.get(kind)
    assert issubclass(cls, ProcessingStep)
    assert cls.kind == kind
    assert cls.category in CATEGORIES, f"{kind}: category {cls.category!r} 不在 {CATEGORIES}"
    assert kind.split(".")[0] == cls.category or cls.category == "custom", \
        f"{kind}: kind 要以分類開頭，例如 {cls.category}.xxx"
    for lang in LANGUAGES:
        assert has(cls.display_name_key, lang), f"{kind}: 缺少 {lang} 翻譯 {cls.display_name_key!r}"
    specs = cls.param_specs()
    keys = [s.key for s in specs]
    assert len(keys) == len(set(keys))
    for s in specs:
        assert isinstance(s, ControlSpec)
        if s.type != ControlType.ACTION:
            s.default_value()  # 預設值必須合法
        for lang in LANGUAGES:
            assert has(s.label_key, lang) or s.label, f"{kind}.{s.key}: 缺少 {lang} 翻譯 {s.label_key!r}"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("color", ["bgr", "gray"])
def test_apply_defaults(kind, color, bgr_image, gray_image, make_frame):
    cls = STEPS.get(kind)
    img = bgr_image if color == "bgr" else gray_image
    frame = make_frame(img)
    step = cls()
    before = img.copy()
    pipe = Pipeline([step])
    for _ in range(3):  # stateful 步驟要能連續處理
        res = pipe.process(frame)
    assert not res.errors, f"{kind} 出錯：{res.errors}"
    assert np.array_equal(img, before), f"{kind} 修改了輸入影像"
    out = res.image
    assert isinstance(out, np.ndarray) and out.dtype == np.uint8 and out.ndim in (2, 3)
    if out.ndim == 3:
        assert out.shape[2] == 3
    if not cls.changes_size:
        assert out.shape[:2] == img.shape[:2], f"{kind} 改變了影像大小但 changes_size=False"


@pytest.mark.parametrize("kind", KINDS)
def test_serialization_roundtrip(kind):
    cls = STEPS.get(kind)
    step = cls()
    d = step.to_dict()
    json.dumps(d)
    again = ProcessingStep.from_dict(json.loads(json.dumps(d)))
    assert type(again) is cls and again.params == step.params and again.enabled == step.enabled


@pytest.mark.parametrize("kind", KINDS)
def test_reset_and_direct_apply(kind, bgr_image, make_frame):
    cls = STEPS.get(kind)
    step = cls()
    src = bgr_image if cls.accepts_gray or bgr_image.ndim == 3 else bgr_image
    ctx = StepContext(frame=make_frame(src))
    step.apply(src, ctx)
    step.reset()
    step.apply(src, ctx)
