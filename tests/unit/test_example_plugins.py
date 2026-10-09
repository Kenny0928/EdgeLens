"""載入教學外掛並重用內建步驟契約；無翻譯檔的名稱改驗備援行為。"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from edgelens import i18n
from edgelens.core import registry
from edgelens.core.controls import ControlSpec, ControlType
from edgelens.core.paths import atomic_write_text
from edgelens.core.project import Workspace
from edgelens.processing.pipeline import CATEGORIES, STEPS, Pipeline, ProcessingStep, StepContext
from tests.contracts import test_steps_contract as contract

EXAMPLES = Path(__file__).resolve().parents[2] / "docs" / "examples" / "plugins"
KINDS = ("custom.invert_colors", "custom.count_bright_pixels", "custom.pixelate")


@pytest.fixture
def plugins(tmp_path, monkeypatch):
    # load_plugins 會寫入全域註冊表與 sys.modules；測完還原，避免影響其他工作包。
    monkeypatch.setattr(STEPS, "_items", dict(STEPS.items()))
    monkeypatch.setattr(STEPS, "_origin", dict(STEPS._origin))
    monkeypatch.setattr(registry, "import_problems", [])
    previous = {k: v for k, v in sys.modules.items() if k.startswith("edgelens_plugins.")}
    folder = Workspace(tmp_path / "workspace").plugins_dir
    for path in EXAMPLES.glob("*.py"):
        atomic_write_text(folder / path.name, path.read_text(encoding="utf-8"))
    try:
        loaded = registry.load_plugins(folder)
        assert set(loaded) == {f"edgelens_plugins.{k.split('.')[1]}" for k in KINDS}
        assert not registry.import_problems
        yield folder
    finally:
        for name in list(sys.modules):
            if name.startswith("edgelens_plugins."):
                sys.modules.pop(name)
        sys.modules.update(previous)


@pytest.mark.parametrize("kind", KINDS)
def test_plugin_attributes_without_translations(plugins, kind, monkeypatch):
    cls = STEPS.get(kind)
    assert issubclass(cls, ProcessingStep) and cls.kind == kind
    assert cls.category in CATEGORIES and cls.category == "custom"
    assert kind.startswith("custom.")
    assert STEPS.origin(kind).startswith("edgelens_plugins.")
    # 真正移除翻譯表，證明這些外掛不需要內建或自己的翻譯鍵。
    monkeypatch.setattr(i18n, "_tables", {"zh_TW": {}, "en": {}})
    assert cls.display_name_key == ""
    # 沒有翻譯檔時，選單顯示類別屬性 display_name（不是看不懂的 kind）
    assert cls.display_name and i18n.component_name(cls) == cls.display_name
    assert cls.description and i18n.component_description(cls) == cls.description
    specs = cls.param_specs()
    assert len({s.key for s in specs}) == len(specs)
    for spec in specs:
        assert isinstance(spec, ControlSpec)
        assert spec.label and not spec.label_key
        assert i18n.tr_or(spec.label_key, spec.label or spec.key) == spec.label
        if spec.type != ControlType.ACTION:
            spec.default_value()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("color", ["bgr", "gray"])
def test_plugin_apply_contract(plugins, kind, color, bgr_image, gray_image, make_frame):
    contract.test_apply_defaults(kind, color, bgr_image, gray_image, make_frame)


@pytest.mark.parametrize("kind", KINDS)
def test_plugin_serialization_and_reset_contract(plugins, kind, bgr_image, make_frame):
    contract.test_serialization_roundtrip(kind)
    contract.test_reset_and_direct_apply(kind, bgr_image, make_frame)
    step = STEPS.get(kind)(enabled=False)
    restored = Pipeline.from_dict(json.loads(json.dumps(Pipeline([step]).to_dict())))
    assert restored.steps[0].enabled is False
    assert np.array_equal(restored.process(make_frame(bgr_image)).image, bgr_image)


def test_invert_values(plugins, make_frame):
    image = np.array([[0, 127, 255]], dtype=np.uint8)
    output = STEPS.get(KINDS[0])().apply(image, StepContext(make_frame(image)))
    np.testing.assert_array_equal(output, [[255, 128, 0]])
    np.testing.assert_array_equal(image, [[0, 127, 255]])


@pytest.mark.parametrize("color", [False, True])
def test_bright_ratio_and_overlay(plugins, make_frame, color):
    image = np.array([[0, 199], [200, 255]], dtype=np.uint8)
    if color:
        image = np.repeat(image[:, :, None], 3, axis=2)
    before = image.copy()
    ctx = StepContext(make_frame(image))
    ctx.push_transform([[2, 0, 10], [0, 3, 5]])
    output = STEPS.get(KINDS[1])().apply(image, ctx)
    assert ctx.data == {"custom.count_bright_pixels.ratio": 0.5}
    json.dumps(ctx.data)
    overlay, = ctx.overlays
    assert overlay.kind == "text"
    assert overlay.data == {"xy": [26.0, 65.0], "text": ">=200: 50.0%"}
    np.testing.assert_array_equal(output, before)
    np.testing.assert_array_equal(image, before)


def test_pixelate_small_image_and_parameters(plugins, make_frame):
    step = STEPS.get(KINDS[2])()
    assert step.set_param("block_size", 0) == 1
    assert step.set_param("block_size", 999) == 64
    with pytest.raises(ValueError):
        step.set_param("block_size", "不是數字")
    with pytest.raises(KeyError):
        step.set_param("unknown", 1)
    image = np.array([[0, 0], [200, 200]], dtype=np.uint8)
    output = step.apply(image, StepContext(make_frame(image)))
    np.testing.assert_array_equal(output, np.full((2, 2), 100, dtype=np.uint8))
    restored = ProcessingStep.from_dict(json.loads(json.dumps(step.to_dict())))
    assert restored.params == {"block_size": 64}
    step.set_param("block_size", 1)
    np.testing.assert_array_equal(step.apply(image, StepContext(make_frame(image))), image)


def test_broken_plugin_is_reported_and_other_plugins_load(plugins):
    atomic_write_text(plugins / "broken.py", "raise RuntimeError('教學外掛故意失敗')\n")
    atomic_write_text(plugins / "_ignored.py", "raise RuntimeError('不應被執行')\n")
    loaded = registry.load_plugins(plugins)
    assert len(loaded) == 3
    problem, = registry.import_problems
    assert Path(problem.module).name == "broken.py"
    assert "教學外掛故意失敗" in problem.error
    assert "RuntimeError" in problem.detail
    assert "edgelens_plugins.broken" not in sys.modules


def test_examples_use_python310_syntax():
    # 環境執行的是 3.13；另外用 3.10 文法檢查，不宣稱代替實際 3.10 執行。
    for path in EXAMPLES.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))
