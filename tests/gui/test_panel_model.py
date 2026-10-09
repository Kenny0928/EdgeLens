"""模型面板：載入、後端選擇、門檻、解碼器、原始輸出、另存說明檔、A/B、K210 板上推論。"""

from __future__ import annotations

import json
import types

import pytest

from edgelens.gui.app import create_app
from edgelens.inference.backends.haar import builtin_cascades
from edgelens.sources.base import SourceInfo
from edgelens.sources.synthetic import SyntheticSource
from tests.fixtures.make_onnx import LABELS, generate_all

pytestmark = pytest.mark.gui


def synth(**params) -> SourceInfo:
    return SyntheticSource.info_from_params({"pattern": "bars", "fps": 60, "resolution": "160x120", **params})


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    folder = tmp_path_factory.mktemp("pmodels")
    paths = generate_all(folder)
    for p in paths.values():
        p.with_name(p.stem + ".labels.txt").write_text("\n".join(LABELS), encoding="utf-8")
    return paths


@pytest.fixture
def env(qtbot, tmp_path, models):
    args = types.SimpleNamespace(workspace=str(tmp_path / "ws"), lang="zh_TW", safe_mode=True)
    app, ctx, window = create_app(args)
    qtbot.addWidget(window)
    ctx.set_project(ctx.workspace.create_project("RPS", "rock_paper_scissors"))
    handle = window.add_source(synth())
    qtbot.waitUntil(lambda: handle.last_packet is not None, timeout=5000)
    panel = window.panel("model")
    yield ctx, window, handle, panel
    window.close()


def loaded(qtbot, panel, timeout=8000):
    qtbot.waitUntil(lambda: panel._session is not None and panel._session.last_result is not None, timeout=timeout)
    return panel._session


def test_panel_registered(env):
    ctx, window, handle, panel = env
    assert panel.order == 30 and window.panel("compare").order == 60
    assert window.tabs.tabText(window.tabs.indexOf(panel)) == "模型"


def test_no_camera_shows_hint(qtbot, tmp_path):
    args = types.SimpleNamespace(workspace=str(tmp_path / "ws"), lang="zh_TW", safe_mode=True)
    app, ctx, window = create_app(args)
    qtbot.addWidget(window)
    panel = window.panel("model")
    assert panel.empty_label.isVisibleTo(panel) and not panel.content.isVisibleTo(panel)
    window.close()


def test_load_model_shows_info_and_live(qtbot, env, models):
    ctx, window, handle, panel = env
    assert panel.load_path(models["yolo_t"])
    assert "載入" in panel.status_label.text()
    s = loaded(qtbot, panel)
    v = panel._info_values
    assert "偵測" in v["task"].text()
    assert "64" in v["input"].text() and v["outputs"].text() != "-"
    assert "3" in v["classes"].text() and "alpha" in v["classes"].text()
    assert "YOLOv8" in v["decoder"].text() and s.info.decoder_reason in v["decoder"].text()
    assert "ms" in v["load"].text()
    panel.update_live()
    assert "beta" in panel.live_list.text() and "推論" in panel.live_stats.text()
    assert str(ctx.settings.get("recent.models")[0]).endswith("fake_yolo_t.onnx")
    assert panel.recent_menu.actions()[0].text() == "fake_yolo_t.onnx"


def test_classify_shows_bars(qtbot, env, models):
    ctx, window, handle, panel = env
    panel.load_path(models["classify"])
    loaded(qtbot, panel)
    panel.update_live()
    assert panel._bars[0][1].isVisibleTo(panel) and panel._bars[0][0].text() == "gamma"
    assert not panel.thresholds.row("conf").widget.isEnabled()
    assert panel.threshold_note.text()


def test_backend_choices_disabled_with_reason(qtbot, env, models, monkeypatch):
    from edgelens.inference.backends.tflite import TfliteBackend

    monkeypatch.setattr(TfliteBackend, "available", classmethod(lambda cls: (False, "沒有安裝 TFLite，可改用 ONNX")))
    ctx, window, handle, panel = env
    fake = models["yolo_t"].with_name("x.tflite")
    fake.write_bytes(b"junk")
    from edgelens.inference.runner import load_model_file

    panel._fill_backends(load_model_file(fake))
    model = panel.backend_combo.model()
    texts = [model.item(i).text() for i in range(model.rowCount())]
    tf = next(i for i, t in enumerate(texts) if "沒有安裝 TFLite" in t)
    assert not model.item(tf).isEnabled()


def test_backend_switch_reloads(qtbot, env, models):
    ctx, window, handle, panel = env
    panel.load_path(models["yolo_t"])
    first = loaded(qtbot, panel)
    kinds = [panel.backend_combo.itemData(i, 0x0100) for i in range(panel.backend_combo.count())]
    assert "onnxrt" in kinds and "cvdnn" in kinds
    other = "cvdnn" if first.info.backend == "onnxrt" else "onnxrt"
    panel.backend_combo.setCurrentIndex(kinds.index(other))
    panel._on_backend_activated(kinds.index(other))
    s2 = loaded(qtbot, panel)
    assert s2 is not first and s2.backend == other and len(ctx.inference.sessions_for(handle)) == 1


def test_threshold_slider_changes_session(qtbot, env, models):
    ctx, window, handle, panel = env
    panel.load_path(models["yolo_t"])
    s = loaded(qtbot, panel)
    row = panel.thresholds.row("conf")
    row.slider.setValue(row.slider.maximum())      # 真的拖動滑桿
    assert s.thresholds["conf"] > 0.95
    qtbot.waitUntil(lambda: s.last_result is not None and not s.last_result.detections, timeout=5000)
    panel._on_threshold("max_det", 7)
    panel._on_threshold("nms", 0.3)
    panel._on_threshold("class_agnostic", True)
    assert s.thresholds["max_det"] == 7 and s.thresholds["nms"] == 0.3 and s.thresholds["class_agnostic"]


def test_haar_conf_disabled_and_neighbors(qtbot, env):
    ctx, window, handle, panel = env
    path = builtin_cascades()["frontalface_default"]
    panel.load_path(path, builtin=True)
    s = loaded(qtbot, panel)
    assert s.builtin and s.info.mode == "image"
    assert not panel.thresholds.row("conf").widget.isEnabled()
    assert "min_neighbors" in panel.thresholds.keys()
    assert "靈敏度" in panel.threshold_note.text()
    panel._on_threshold("min_neighbors", 9)
    assert s.card.decoder_params["min_neighbors"] == 9
    assert "frontalface" not in ctx.settings.get("recent.models", [])
    assert panel.builtin_menu.actions()


def test_input_choice(qtbot, env, models):
    ctx, window, handle, panel = env
    panel.load_path(models["yolo_t"])
    s = loaded(qtbot, panel)
    assert not s.use_processed
    panel.input_combo.setCurrentIndex(1)
    assert s.use_processed
    assert "原始畫面" in panel.input_note.text()


def test_ambiguous_decoder_choice(qtbot, env, models):
    ctx, window, handle, panel = env
    panel.load_path(models["yolo_t"])
    s = loaded(qtbot, panel)
    # 手動讓面板顯示有疑義的情況，確認候選清單與套用
    s.info.ambiguous = True
    panel._refresh_info()
    assert panel.ambiguous_box.isVisibleTo(panel)
    assert panel.decoder_combo.count() >= 7        # 候選 + 全部解碼器
    idx = next(i for i in range(panel.decoder_combo.count()) if panel.decoder_combo.itemData(i)[0] == "yolo_t")
    panel.decoder_combo.setCurrentIndex(idx)
    panel.apply_decoder()
    qtbot.waitUntil(lambda: not s.info.ambiguous, timeout=3000)


def test_ambiguous_detected_for_yolox(qtbot, env, models):
    ctx, window, handle, panel = env
    panel.load_path(models["yolox_raw"])
    s = loaded(qtbot, panel)
    assert s.info.guesses
    assert panel.ambiguous_box.isVisibleTo(panel) == s.info.ambiguous


def test_load_error_shows_message_and_hint(qtbot, env, tmp_path):
    ctx, window, handle, panel = env
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"nope")
    panel.load_path(bad)
    s = panel._session
    qtbot.waitUntil(lambda: s.state == "failed", timeout=8000)
    qtbot.waitUntil(lambda: s.last_error["message"] in panel.status_label.text(), timeout=2000)
    text = panel.status_label.text()
    assert s.last_error["hint"] in text and "\n" in text


def test_bad_card_file(qtbot, env, tmp_path):
    ctx, window, handle, panel = env
    card = tmp_path / "broken.card.json"
    card.write_text("{not json", encoding="utf-8")
    assert not panel.load_path(card)
    assert "broken.card.json" in panel.status_label.text()


def test_raw_dialog(qtbot, env, models):
    ctx, window, handle, panel = env
    panel.load_path(models["yolo_t"])
    s = loaded(qtbot, panel)
    dlg = panel.open_raw_dialog()
    qtbot.addWidget(dlg)
    qtbot.waitUntil(lambda: dlg.picker.count() == 1, timeout=5000)
    assert "最小" in dlg.summary.toPlainText() or "min" in dlg.summary.toPlainText().lower()
    assert dlg.table.rowCount() == 7 and dlg.table.columnCount() == 24
    dlg.refresh()
    assert s.raw_outputs
    dlg.close()


def test_raw_dialog_haar_unsupported(qtbot, env):
    ctx, window, handle, panel = env
    panel.load_path(builtin_cascades()["frontalface_default"], builtin=True)
    loaded(qtbot, panel)
    dlg = panel.open_raw_dialog()
    qtbot.addWidget(dlg)
    assert not dlg.refresh_button.isEnabled() and "沒有原始輸出" in dlg.note.text()


def test_save_card_then_reload_without_guessing(qtbot, env, models, tmp_path):
    import shutil

    ctx, window, handle, panel = env
    copy = tmp_path / "copy_yolo.onnx"
    shutil.copy(models["yolo_t"], copy)
    panel.load_path(copy)
    s = loaded(qtbot, panel)
    panel._on_threshold("conf", 0.6)
    path = panel.save_card()
    assert path == tmp_path / "copy_yolo.card.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["decoder"] == "yolo_t" and data["thresholds"]["conf"] == 0.6 and data["backend"] == s.info.backend
    panel.load_path(copy)            # 旁邊有 .card.json 了
    s2 = loaded(qtbot, panel)
    assert s2.card.decoder == "yolo_t" and [g.decoder_kind for g in s2.info.guesses] == ["yolo_t"]
    assert panel.thresholds.values()["conf"] == pytest.approx(0.6)


def test_save_card_for_builtin_goes_to_project(qtbot, env):
    ctx, window, handle, panel = env
    panel.load_path(builtin_cascades()["frontalface_default"], builtin=True)
    loaded(qtbot, panel)
    path = panel.save_card()
    assert path is not None and ctx.project.models_dir in path.parents


def test_ab_second_model_and_switch(qtbot, env, models):
    ctx, window, handle, panel = env
    panel.load_path(models["yolo_t"])
    a = loaded(qtbot, panel)
    panel.ab_check.setChecked(True)
    panel.load_path(models["classify"])
    qtbot.waitUntil(lambda: len(ctx.inference.sessions_for(handle)) == 2, timeout=3000)
    b = panel._session
    assert b is not a and panel.session_combo.count() == 2 and b.style_index == 1
    qtbot.waitUntil(lambda: b.last_result is not None and a.last_result is not None, timeout=8000)
    panel.session_combo.setCurrentIndex(0)
    assert panel._session is a
    # 第三個模型取代第二個
    panel.load_path(models["yolo_flat"])
    qtbot.waitUntil(lambda: len(ctx.inference.sessions_for(handle)) == 2 and panel._session.card.name.endswith("flat"),
                    timeout=3000)
    assert a in ctx.inference.sessions_for(handle) and b not in ctx.inference.sessions_for(handle)


def test_replace_without_ab(qtbot, env, models):
    ctx, window, handle, panel = env
    panel.load_path(models["yolo_t"])
    a = loaded(qtbot, panel)
    panel.load_path(models["classify"])
    qtbot.waitUntil(lambda: a not in ctx.inference.sessions_for(handle), timeout=3000)
    assert len(ctx.inference.sessions_for(handle)) == 1


def test_stop_current(qtbot, env, models):
    ctx, window, handle, panel = env
    panel.load_path(models["yolo_t"])
    s = loaded(qtbot, panel)
    view = window.video_view_for(handle)
    panel.stop_current()
    assert ctx.inference.sessions_for(handle) == [] and s.state == "stopped"
    assert view._results == [] and panel._session is None
    assert "infer" not in window._hud_extra.get(handle, {})


def test_language_switch_keeps_working(qtbot, env, models):
    ctx, window, handle, panel = env
    panel.load_path(models["yolo_t"])
    loaded(qtbot, panel)
    ctx.set_language("en")
    qtbot.wait(50)
    assert panel.load_button.text() == "Load model..."
    ctx.set_language("zh_TW")


def test_kmodel_needs_k210(qtbot, env, tmp_path):
    ctx, window, handle, panel = env
    k = tmp_path / "m.kmodel"
    k.write_bytes(b"KMODEL" * 10)
    assert not panel.load_path(k)
    assert "只能在 K210" in panel.status_label.text()
    assert ctx.inference.sessions_for(handle) == []


# ---- K210 板上推論（模擬板）--------------------------------------------------------------

def test_kmodel_on_k210_uses_load_model_action(qtbot, tmp_path, monkeypatch):
    from edgelens.sources.k210 import source as S
    from edgelens.sources.k210.sim import FakeK210Board
    from tests.unit.test_k210_source import ANCHORS

    boards = []

    def factory(port, baud):
        b = FakeK210Board()
        b.time_scale = 0.0
        boards.append(b)
        return b

    monkeypatch.setattr(S, "open_serial", factory)
    args = types.SimpleNamespace(workspace=str(tmp_path / "ws"), lang="zh_TW", safe_mode=True)
    app, ctx, window = create_app(args)
    qtbot.addWidget(window)
    try:
        handle = window.add_source(SourceInfo(kind="k210", id="k210:sim", name="K210 sim", params={"port": "sim:"}))
        qtbot.waitUntil(lambda: handle.last_packet is not None, timeout=8000)
        panel = window.panel("model")
        kmodel = tmp_path / "dev.kmodel"
        kmodel.write_bytes(b"KMODEL" * 700)
        (tmp_path / "dev.card.json").write_text(json.dumps({
            "file": "dev.kmodel", "name": "dev", "labels": ["cat", "dog"], "task": "detect",
            "input": {"size": [320, 240]}, "thresholds": {"conf": 0.4, "nms": 0.25},
            "device": {"path": "/flash/dev.kmodel", "task": "detect", "anchors": ANCHORS}}), encoding="utf-8")
        actions = []
        orig = handle.run_action
        handle.run_action = lambda key, **kw: (actions.append(key), orig(key, **kw))[1]
        assert panel.load_path(kmodel)
        qtbot.waitUntil(lambda: bool(ctx.inference.sessions_for(handle)), timeout=15000)
        assert actions[0] == "load_model"
        s = ctx.inference.primary(handle)
        assert s.backend == "device"
        assert bytes(boards[-1].files["/flash/dev.kmodel"]) == kmodel.read_bytes()
        boards[-1].objs = [[10, 20, 30, 40, 0.9, 1]]
        qtbot.waitUntil(lambda: s.last_result is not None and bool(s.last_result.detections), timeout=10000)
        r = s.last_result
        assert r.where == "device" and r.detections[0].label == "dog"
        assert not panel.device_progress.isVisibleTo(panel)
        # 面板的門檻只剩信心
        assert panel.thresholds.keys() == ["conf"]
        panel.stop_current()
        qtbot.waitUntil(lambda: "unload_model" in actions, timeout=3000)
    finally:
        window.close()


def test_live_results_do_not_make_layout_jump(qtbot, env, models):
    """即時結果偵測到的物體數一直變（0、1、3、8 個），下面的按鈕位置不能跟著跳動。"""
    from edgelens.core.results import Detection, InferenceResult

    ctx, window, handle, panel = env
    window.resize(1400, 1000)
    window.show()
    window.tabs.setCurrentWidget(panel)
    assert panel.load_path(models["yolo_t"])
    s = loaded(qtbot, panel)
    s.stop()   # 停掉真的推論，改送自己造的結果
    base = s.last_result

    def show(n: int) -> int:
        dets = [Detection((0, 0, 10, 10), 0.9 - i * 0.05, i % 3, f"物體{i}") for i in range(n)]
        s.last_result = InferenceResult(kind="detect", model_name="m", frame_seq=base.frame_seq, detections=dets,
                                        timings_ms=dict(base.timings_ms))
        panel.update_live()
        panel.layout().activate()
        qtbot.wait(10)
        return panel.raw_button.mapTo(panel, panel.raw_button.rect().topLeft()).y()

    ys = {show(n) for n in (1, 0, 3, 8, 2, 0)}
    assert len(ys) == 1, f"按鈕位置跟著偵測數量跳動：{ys}"
