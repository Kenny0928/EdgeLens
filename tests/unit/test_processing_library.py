"""內建教學流程、中文存檔與每張處理時間。"""
from __future__ import annotations

import json
import statistics
import time

import cv2
import numpy as np
import pytest

from edgelens.processing.library import builtin_presets, list_presets, load_preset, save_user_preset
from edgelens.processing.pipeline import Pipeline


@pytest.mark.parametrize("name", list(builtin_presets()))
def test_builtin_load_process_and_roundtrip(name, make_frame):
    data = builtin_presets()[name]
    assert name.endswith(".json")
    assert set(data["title"]) == {"zh_TW", "en"} and set(data["description"]) == {"zh_TW", "en"}
    pipeline = load_preset(name)
    image = np.zeros((480, 640, 3), np.uint8)
    image[100:250, 150:300] = (0, 0, 255)
    for _ in range(3):
        result = pipeline.process(make_frame(image))
        assert not result.errors
    serialized = pipeline.to_dict()
    assert Pipeline.from_dict(json.loads(json.dumps(serialized))).to_dict() == serialized
    assert load_preset(name.removesuffix(".json")).to_dict() == serialized


def test_save_list_load_and_titles(tmp_path):
    pipeline = load_preset("red_objects")
    folder = tmp_path / "中文專案" / "pipelines"
    path = save_user_preset(pipeline, folder, "我的流程", {"zh_TW": "紅色實驗", "en": "Red experiment"})
    assert path.parent == folder and path.name == "我的流程.json"
    assert load_preset(path).to_dict() == pipeline.to_dict()
    assert not list(folder.glob(".tmp_*"))
    infos = list_presets(folder)
    assert len(infos) >= 9
    info = next(p for p in infos if not p.builtin)
    assert info.title("zh_TW") == "紅色實驗" and info.title("en") == "Red experiment"
    assert info.title("unknown") == "Red experiment" and info.description("en") == ""
    assert all(p.description("zh_TW") and p.description("en") for p in infos if p.builtin)
    path = save_user_preset(pipeline, folder, "nested/escape", "班級流程")
    assert path.parent == folder
    assert json.loads(path.read_text())["title"] == {"zh_TW": "班級流程", "en": "班級流程"}


def test_invalid_presets_skip_or_actionable_error(tmp_path):
    for i, value in enumerate(("{bad", "[]", '{"steps": {}}', '{"steps": [1]}', '{"steps": [], "title": 2}')):
        path = tmp_path / f"bad{i}.json"
        path.write_text(value)
        with pytest.raises(ValueError, match="流程 JSON"):
            load_preset(path)
    assert all(info.builtin for info in list_presets(tmp_path))
    path = tmp_path / "unknown.json"
    path.write_text('{"steps": [{"kind": "custom.missing"}]}')
    with pytest.raises(ValueError, match="步驟名稱"):
        load_preset(path)
    with pytest.raises(ValueError):
        load_preset(tmp_path / "missing.json")
    with pytest.raises(ValueError):
        save_user_preset(Pipeline(), tmp_path, "", "title")
    with pytest.raises(ValueError):
        save_user_preset(Pipeline(), tmp_path, "okay", {"en": 7})


def test_builtin_returns_independent_data():
    data = builtin_presets()
    assert len(data) >= 8
    key = next(iter(data))
    data[key]["steps"].clear()
    assert builtin_presets()[key]["steps"]


@pytest.mark.parametrize("name", ["grayscale_blur", "red_objects"])
def test_teaching_preset_under_ten_ms(name, make_frame):
    image = np.full((480, 640, 3), 50, np.uint8)
    cv2.circle(image, (180, 200), 60, (0, 0, 255), -1)
    cv2.rectangle(image, (350, 140), (500, 290), (0, 0, 255), -1)
    frame = make_frame(image)
    pipeline = load_preset(name)
    for _ in range(3):
        pipeline.process(frame)
    timings = []
    for _ in range(20):
        start = time.perf_counter()
        result = pipeline.process(frame)
        timings.append((time.perf_counter() - start) * 1000)
        assert not result.errors
    median = statistics.median(timings)
    print(f"{name}: median of 20 = {median:.3f} ms")
    assert median < 10
    if name == "red_objects":
        assert result.data["analyze.blobs.count"] == 2


def test_uppercase_extension_is_listed(tmp_path):
    path = save_user_preset(Pipeline(), tmp_path, "example.JSON", "Example")
    assert path.name == "example.json"
    assert path in [p.path for p in list_presets(tmp_path)]
