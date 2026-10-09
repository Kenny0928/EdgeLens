"""WP0 核心型別的單元測試。"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from edgelens.core.controls import ControlSpec, ControlType, validate_params
from edgelens.core.frame import Frame
from edgelens.core.modelcard import ModelCard
from edgelens.core.paths import atomic_write_text, cloud_sync_provider, safe_filename, slugify, unique_path
from edgelens.core.project import ClassSpec, Project, Workspace, project_templates
from edgelens.core.registry import Registry, load_plugins
from edgelens.core.results import Classification, Detection, InferenceResult
from edgelens.core.settings import Settings
from edgelens.inference.base import PreprocessMeta
from edgelens.processing.pipeline import STEPS, Pipeline

# ---- Frame -----------------------------------------------------------------

def test_frame_validation():
    img = np.zeros((4, 5, 3), np.uint8)
    f = Frame(image=img, seq=1, t_cap=time.monotonic(), source_id="s")
    assert f.size == (5, 4) and not f.is_gray
    g = Frame(image=np.zeros((4, 5, 1), np.uint8), seq=1, t_cap=0, source_id="s")
    assert g.is_gray and g.image.shape == (4, 5)
    with pytest.raises(TypeError):
        Frame(image=np.zeros((4, 5), np.float32), seq=1, t_cap=0, source_id="s")
    with pytest.raises(ValueError):
        Frame(image=np.zeros((4, 5, 4), np.uint8), seq=1, t_cap=0, source_id="s")


def test_frame_replace_drops_jpeg():
    f = Frame(image=np.zeros((2, 2, 3), np.uint8), seq=1, t_cap=0, source_id="s", jpeg=b"x")
    g = f.replace(image=np.ones((2, 2), np.uint8))
    assert g.jpeg is None and f.jpeg == b"x"
    assert f.replace(seq=2).jpeg == b"x"


# ---- ControlSpec -------------------------------------------------------------

def test_control_validate():
    s = ControlSpec("b", ControlType.INT, min=-2, max=2, default=0)
    assert s.validate("5") == 2 and s.validate(-9.4) == -2 and s.validate(1.6) == 2
    with pytest.raises(ValueError):
        s.validate("abc")
    e = ControlSpec("m", ControlType.ENUM, options=((1, "a"), (2, "b")))
    assert e.validate("2") == 2 and e.default_value() == 1
    with pytest.raises(ValueError):
        e.validate(3)
    b = ControlSpec("x", ControlType.BOOL)
    assert b.validate("off") is False and b.validate("1") is True
    c = ControlSpec("c", ControlType.COLOR, default="#ff0000")
    assert c.default_value() == "#FF0000"
    with pytest.raises(ValueError):
        ControlSpec("e", ControlType.ENUM)


def test_validate_params_ignores_unknown():
    specs = [ControlSpec("a", ControlType.INT, default=3, min=0, max=10)]
    assert validate_params(specs, {"a": 20, "zzz": 1}) == {"a": 10}
    with pytest.raises(ValueError):
        validate_params(specs, {"zzz": 1}, strict=True)


# ---- Results ---------------------------------------------------------------

def test_inference_result_roundtrip():
    r = InferenceResult(kind="detect", model_name="m", frame_seq=3,
                        detections=[Detection((1, 2, 3, 4), 0.9, 0, "a"), Detection((0, 0, 1, 1), 0.3, 1)],
                        timings_ms={"infer": 1.23456}, image_size=(64, 48))
    assert r.best_score == pytest.approx(0.9)
    d = json.loads(json.dumps(r.to_dict()))
    again = InferenceResult.from_dict(d)
    assert again.detections[0].box == (1, 2, 3, 4) and again.image_size == (64, 48)
    c = InferenceResult(kind="classify", model_name="m", frame_seq=1, top_k=[Classification(2, 0.7, "x")])
    assert c.best.class_id == 2 and c.has_hit


# ---- Paths -----------------------------------------------------------------

def test_slugify_and_filenames():
    assert slugify("Flat Up!") == "flat_up"
    assert slugify("聖筊") == "item"
    assert slugify("聖筊", fallback="class_1") == "class_1"
    assert slugify("3 cats") == "c_3_cats"
    assert safe_filename('a<b>:c?.jpg') == "a_b__c_.jpg"
    assert safe_filename("CON.txt") == "_CON.txt"
    assert safe_filename("聖筊.jpg") == "聖筊.jpg"


def test_atomic_write_chinese_path(tmp_path):
    p = atomic_write_text(tmp_path / "中文資料夾" / "設定.json", "{}")
    assert p.read_text(encoding="utf-8") == "{}"
    assert not [x for x in p.parent.iterdir() if x.name.startswith(".tmp_")]
    assert unique_path(p).name == "設定_1.json"


def test_cloud_sync_detection(tmp_path):
    assert cloud_sync_provider("/Users/x/Library/Mobile Documents/com~apple~CloudDocs/a") == "iCloud"
    assert cloud_sync_provider("C:/Users/x/OneDrive/Documents/EdgeLens") == "OneDrive"
    assert cloud_sync_provider(tmp_path) is None


# ---- Settings --------------------------------------------------------------

def test_settings_roundtrip_and_corrupt_file(tmp_path):
    path = tmp_path / "s.json"
    s = Settings(path)
    assert s.get("ui.language") == "zh_TW"
    s.set("ui.language", "en")
    s.set_source_settings("webcam:0", {"brightness": 3})
    s.push_recent("models", "a.onnx")
    s.save()
    s2 = Settings(path)
    assert s2.get("ui.language") == "en"
    assert s2.source_settings("webcam:0") == {"brightness": 3}
    assert s2.get("recent.models") == ["a.onnx"]
    path.write_text("{broken", encoding="utf-8")
    s3 = Settings(path)
    assert s3.get("ui.language") == "zh_TW"
    assert path.with_suffix(".broken.json").exists()


# ---- Project ---------------------------------------------------------------

def test_project_templates_valid():
    tpls = project_templates()
    assert {"blank", "rock_paper_scissors", "jiaobei"} <= set(tpls)
    for name, tpl in tpls.items():
        for c in tpl.get("classes", []):
            ClassSpec.from_dict(c)
        hotkeys = [c.get("hotkey") for c in tpl.get("classes", []) if c.get("hotkey")]
        assert len(hotkeys) == len(set(hotkeys)), f"{name} 快捷鍵重複"


def test_workspace_create_and_reload(tmp_path):
    ws = Workspace(tmp_path / "ws")
    p = ws.create_project("剪刀石頭布 A 班", template="rock_paper_scissors")
    assert p.root.parent == ws.projects_dir
    assert (p.root / "dataset").is_dir() and (p.root / "project.json").is_file()
    assert p.class_ids() == ["rock", "paper", "scissors", "background"]
    c = p.add_class("新類別")
    assert c.id == "class_5" and c.hotkey == "5"
    p.save()
    again = ws.open_project(p.id)
    assert again.class_by_id("class_5").display("zh_TW") == "新類別"
    assert [x.id for x in ws.projects()] == [p.id]
    p2 = ws.create_project("剪刀石頭布 A 班")
    assert p2.root != p.root
    with pytest.raises(ValueError):
        ClassSpec(id="Bad Id")


def test_project_rejects_unknown_subdir(tmp_path):
    p = Project(root=tmp_path / "p", name="p")
    with pytest.raises(ValueError):
        p.dir("nope")


# ---- ModelCard -------------------------------------------------------------

def test_modelcard_auto_and_labels(tmp_path):
    model = tmp_path / "net.onnx"
    model.write_bytes(b"fake")
    (tmp_path / "labels.txt").write_text("cat\ndog\n\n# comment\n", encoding="utf-8")
    card = ModelCard.load(model)
    assert card.labels == ["cat", "dog"] and card.decoder == "auto" and card.extension == ".onnx"
    card.decoder = "classify"
    card.input.size = (224, 224)
    path = card.save()
    assert path.name == "net.card.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["file"] == "net.onnx" and data["labels_file"] == "labels.txt"
    again = ModelCard.load(model)
    assert again.decoder == "classify" and again.input.size == (224, 224) and again.labels == ["cat", "dog"]
    assert again.model_path.resolve() == model.resolve()


def test_modelcard_darknet_and_numbered_labels(tmp_path):
    (tmp_path / "y.weights").write_bytes(b"w")
    (tmp_path / "y.cfg").write_text("[net]", encoding="utf-8")
    (tmp_path / "y.names").write_text("0 person\n1 car\n", encoding="utf-8")
    card = ModelCard.load(tmp_path / "y.weights")
    assert card.extra_path("config").name == "y.cfg"
    assert card.labels == ["person", "car"]


def test_preprocess_meta_roundtrip():
    meta = PreprocessMeta(orig_size=(640, 480), input_size=(416, 416), scale=(0.65, 0.65), pad=(0, 52))
    boxes = np.array([[10, 20, 100, 200]], np.float32)
    assert np.allclose(meta.to_original(meta.to_input(boxes)), boxes, atol=1e-3)


# ---- Registry / plugins ----------------------------------------------------

def test_registry_rules():
    class Base:
        pass

    reg: Registry[Base] = Registry("t", Base)

    @reg.register
    class A(Base):
        kind = "a.one"

    assert reg.get("a.one") is A and "a.one" in reg
    with pytest.raises(TypeError):
        reg.register(int)  # type: ignore[arg-type]

    class B(Base):
        kind = "Bad Kind"

    with pytest.raises(ValueError):
        reg.register(B)

    class C(Base):
        kind = "a.one"

    with pytest.raises(ValueError):
        reg.register(C)
    with pytest.raises(KeyError):
        reg.get("zzz")


def test_plugin_step_loads(tmp_path, bgr_image, make_frame):
    plugin = tmp_path / "my_invert.py"
    plugin.write_text(
        "from edgelens.processing.pipeline import ProcessingStep, register_step\n"
        "@register_step\n"
        "class Invert(ProcessingStep):\n"
        "    kind = 'custom.test_invert'\n"
        "    category = 'custom'\n"
        "    display_name_key = 'step.custom.test_invert'\n"
        "    def apply(self, image, ctx):\n"
        "        return 255 - image\n", encoding="utf-8")
    (tmp_path / "broken.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    try:
        loaded = load_plugins(tmp_path)
        assert loaded == ["edgelens_plugins.my_invert"]
        pipe = Pipeline.from_dict({"version": 1, "steps": [{"kind": "custom.test_invert"}]})
        out = pipe.process(make_frame(bgr_image)).image
        assert np.array_equal(out, 255 - bgr_image)
    finally:
        STEPS.unregister("custom.test_invert")


def test_pipeline_survives_failing_step(bgr_image, make_frame):
    from edgelens.processing.pipeline import ProcessingStep

    class Boom(ProcessingStep):
        kind = "custom.boom"

        def apply(self, image, ctx):
            raise RuntimeError("x")

    res = Pipeline([Boom()]).process(make_frame(bgr_image))
    assert res.errors and res.errors[0][0] == "custom.boom"
    assert np.array_equal(res.image, bgr_image)
    with pytest.raises(ValueError):
        Pipeline.from_dict({"version": 99, "steps": []})


def test_info_for_storage_strips_secrets():
    from edgelens.core.registry import discover
    from edgelens.sources.base import SourceInfo, info_for_storage

    discover()
    info = SourceInfo(kind="mjpeg", id="mjpeg:http://cam/video", name="cam",
                      params={"url": "http://cam/video", "username": "u", "password": "p@ss"})
    d = info_for_storage(info)
    assert "password" not in d["params"] and d["params"]["username"] == "u"
    assert info.params["password"] == "p@ss"  # 原物件不受影響


def test_info_for_storage_strips_url_passwords():
    from edgelens.sources.base import SourceInfo, info_for_storage, strip_url_password

    assert strip_url_password("rtsp://admin:1234@192.168.1.5:554/live") == "rtsp://admin@192.168.1.5:554/live"
    assert strip_url_password("rtsp://192.168.1.5/live") == "rtsp://192.168.1.5/live"
    assert strip_url_password("C:/影片/a.mp4") == "C:/影片/a.mp4"
    info = SourceInfo(kind="rtsp", id="rtsp:x", name="x", params={"url": "rtsp://u:secret@cam/1", "transport": "tcp"})
    assert "secret" not in str(info_for_storage(info))


def test_default_workspace_avoids_cloud_synced_documents(tmp_path, monkeypatch):
    from edgelens.core import paths

    monkeypatch.delenv("EDGELENS_WORKSPACE", raising=False)
    monkeypatch.setattr(paths, "documents_dir", lambda: tmp_path / "OneDrive" / "Documents")
    monkeypatch.setattr(paths.Path, "home", classmethod(lambda cls: tmp_path))
    assert paths.default_workspace() == tmp_path / "EdgeLens"
    monkeypatch.setattr(paths, "documents_dir", lambda: tmp_path / "Docs")
    assert paths.default_workspace() == tmp_path / "Docs" / "EdgeLens"
