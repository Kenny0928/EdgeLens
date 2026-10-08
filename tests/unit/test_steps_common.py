"""每個步驟的翻譯、錯誤參數與極小畫面都要可預期。"""
from __future__ import annotations

import json

import numpy as np
import pytest

from edgelens.core.controls import ControlType
from edgelens.core.paths import resource_path
from edgelens.i18n import has
from edgelens.processing.library import _register_steps
from edgelens.processing.pipeline import STEPS, Pipeline, StepContext
from edgelens.processing.steps._common import odd, roi_rect, transform

_register_steps()
KINDS = [k for k in STEPS.kinds() if not k.startswith("custom.")]


@pytest.mark.parametrize("kind", KINDS)
def test_all_step_translations_and_invalid_params(kind):
    cls = STEPS.get(kind)
    for lang in ("zh_TW", "en"):
        assert has(cls.display_name_key, lang) and has(cls.description_key, lang)
        for s in cls.param_specs():
            assert s.group == "image" and has(s.label_key, lang)
            for _, label in s.options:
                assert has(label, lang)
    step = cls()
    with pytest.raises(KeyError):
        step.set_param("missing", 0)
    for s in cls.param_specs():
        if s.type in (ControlType.INT, ControlType.FLOAT, ControlType.ENUM):
            with pytest.raises(ValueError):
                step.set_param(s.key, "not a valid value")
        if s.type in (ControlType.INT, ControlType.FLOAT):
            with pytest.raises(ValueError):
                step.set_param(s.key, float("nan"))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", [(1, 1), (1, 1, 3)])
def test_tiny_images_and_changing_shapes(kind, shape, make_frame):
    pipeline = Pipeline([STEPS.get(kind)()])
    for size in (shape, (3, 5) + shape[2:], shape):
        src = np.full(size, 120, np.uint8)
        out = pipeline.process(make_frame(src))
        assert not out.errors
        assert out.image.dtype == np.uint8 and out.image.size > 0
        assert np.all(src == 120)


def test_step_locale_files_match_and_no_duplicates():
    tables = []
    for lang in ("zh_TW", "en"):
        pairs = json.loads(resource_path("i18n", "locales", lang, "steps.json").read_text(), object_pairs_hook=list)
        assert len(dict(pairs)) == len(pairs)
        tables.append(dict(pairs))
    assert tables[0].keys() == tables[1].keys()


def test_common_roi_and_transform(make_frame):
    image = np.zeros((4, 6), np.uint8)
    assert odd(2) == 3 and odd(0, 3) == 3
    assert roi_rect(image, dict(x=100, y=100, w=1, h=1)) == (5, 3, 6, 4)
    ctx = StepContext(make_frame(image), matrix=np.array([[2.0, 0, 3], [0, 3.0, 4], [0, 0, 1]]))
    transform(ctx, [[1, 0, 2], [0, 1, 1], [0, 0, 1]])
    assert ctx.to_original(1, 1) == (9, 10)


def test_mask_overlay_uses_original_coordinates(make_frame):
    from edgelens.processing.steps._common import add_mask
    from edgelens.processing.steps.geometry import Crop, Rotate

    image = np.zeros((10, 20), np.uint8)
    ctx = StepContext(make_frame(image))
    out = Crop(x=50, w=50).apply(image, ctx)
    out = Rotate(mode="90").apply(out, ctx)
    out[2, 3] = 255
    assert ctx.scale == (1, 1)
    add_mask(ctx, out)
    mask = ctx.overlays[0].data["mask"]
    assert mask.shape == image.shape and np.count_nonzero(mask) == 1
    x, y = ctx.to_original(3, 2)
    assert mask[round(y), round(x)] == 255
