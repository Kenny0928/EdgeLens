"""拍照面板：快捷鍵、連拍、定時、靜止自動拍、品質、相近提醒、全部一起拍（synthetic 來源）。"""

from __future__ import annotations

import time
import types

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage

from edgelens.dataset.writer import shared_writer
from edgelens.gui.app import create_app
from edgelens.sources.base import SourceInfo
from edgelens.sources.synthetic import SyntheticSource

pytestmark = pytest.mark.gui


def synth(pattern: str = "bars", **params) -> SourceInfo:
    return SyntheticSource.info_from_params({"pattern": pattern, "fps": 60, "resolution": "160x120", **params})


@pytest.fixture
def env(qtbot, tmp_path):
    args = types.SimpleNamespace(workspace=str(tmp_path / "ws"), lang="zh_TW", safe_mode=True)
    app, ctx, window = create_app(args)
    qtbot.addWidget(window)
    project = ctx.workspace.create_project("RPS", "rock_paper_scissors")
    ctx.set_project(project)
    handle = window.add_source(synth())
    qtbot.waitUntil(lambda: handle.last_packet is not None, timeout=3000)
    panel = window.panel("capture")
    yield ctx, window, handle, panel
    window.close()


def records(ctx, class_id=None):
    return shared_writer(ctx.project).list_samples(class_id)


def wait_saved(qtbot, panel, n):
    qtbot.waitUntil(lambda: panel.pending == 0 and panel.saved_total >= n, timeout=5000)


def test_panel_registered_with_class_buttons(qtbot, env):
    ctx, window, handle, panel = env
    assert panel.order == 40
    assert set(panel._buttons) == {c.id for c in ctx.project.classes}
    assert panel.current_class() is not None
    first = ctx.project.classes[0]
    assert first.hotkey in panel._buttons[first.id].text()


def test_space_shortcut_saves_with_metadata(qtbot, env):
    ctx, window, handle, panel = env
    window.show()
    qtbot.keyClick(window, Qt.Key.Key_Space)
    if panel.saved_total == 0 and panel.pending == 0:  # offscreen 沒有焦點時改直接觸發快捷鍵
        panel._shortcuts[0].activated.emit()
    wait_saved(qtbot, panel, 1)
    recs = records(ctx)
    assert len(recs) == 1
    r = recs[0]
    assert r.class_id == panel._class_id and r.source["id"] == handle.info.id
    assert r.path.is_file() and r.width == 160 and r.height == 120
    assert r.controls.get("pattern") == "bars"
    assert "已存" in panel.last_saved_text and "第 1 張" in panel.last_saved_text
    assert panel.count_of(r.class_id) == 1
    assert "1 張" in panel._buttons[r.class_id].text()


def test_digit_shortcut_switches_class(qtbot, env):
    ctx, window, handle, panel = env
    second = ctx.project.classes[1]
    panel._on_digit(second.hotkey)
    assert panel._class_id == second.id
    panel.capture_now()
    wait_saved(qtbot, panel, 1)
    assert len(records(ctx, second.id)) == 1
    # 焦點在文字框時不處理
    panel.burst_interval.setFocus()
    window.show()
    panel.burst_interval.setFocus()
    if panel._typing():
        panel._on_digit(ctx.project.classes[0].hotkey)
        assert panel._class_id == second.id


def test_processed_image_saved_only_when_pipeline_and_enabled(qtbot, env):
    ctx, window, handle, panel = env
    panel.capture_now()
    wait_saved(qtbot, panel, 1)
    assert records(ctx)[0].processed_file is None  # 沒有處理流程就不存副本
    handle.set_pipeline({"version": 1, "steps": [{"kind": "color.grayscale", "enabled": True, "params": {}}]})
    qtbot.waitUntil(lambda: handle.last_packet.result.timings_ms != [], timeout=3000)
    panel.capture_now()
    wait_saved(qtbot, panel, 2)
    with_proc = [r for r in records(ctx) if r.processed_file]
    assert len(with_proc) == 1 and with_proc[0].pipeline["steps"][0]["kind"] == "color.grayscale"
    panel.save_processed.setChecked(False)
    panel.capture_now()
    wait_saved(qtbot, panel, 3)
    assert len([r for r in records(ctx) if r.processed_file]) == 1


def test_dataset_changed_emitted(qtbot, env):
    ctx, window, handle, panel = env
    with qtbot.waitSignal(ctx.datasetChanged, timeout=3000):
        panel.capture_now()


def test_burst_same_burst_id_and_cancel(qtbot, env):
    ctx, window, handle, panel = env
    panel.burst_count.setValue(4)
    panel.burst_interval.setValue(20)
    assert panel.start_burst()
    wait_saved(qtbot, panel, 4)
    recs = records(ctx)
    assert len(recs) == 4
    ids = {r.burst_id for r in recs}
    assert len(ids) == 1 and None not in ids
    # 取消
    panel.burst_count.setValue(30)
    panel.burst_interval.setValue(200)
    assert panel.start_burst()
    wait_saved(qtbot, panel, 5)
    panel.cancel_burst()
    time.sleep(0.5)
    qtbot.waitUntil(lambda: panel.pending == 0, timeout=3000)
    assert 5 <= len(records(ctx)) < 34


def test_timer_capture_with_countdown(qtbot, env):
    ctx, window, handle, panel = env
    panel._cd_tick_ms = 40
    panel.timer_seconds.setValue(2)
    assert panel.start_countdown()
    ov = panel._overlay(handle)
    assert ov.text == "2"
    assert records(ctx) == []
    wait_saved(qtbot, panel, 1)
    assert ov.text == ""
    assert len(records(ctx)) == 1
    # 取消倒數不拍
    panel.start_countdown()
    panel.cancel_countdown()
    time.sleep(0.2)
    assert len(records(ctx)) == 1


def test_still_auto_capture_waits_for_motion(qtbot, env):
    ctx, window, handle, panel = env
    handle.set_control("pattern", "shapes")
    handle.flush_controls()
    panel.still_frames.setValue(4)
    qtbot.wait(300)
    panel.still_check.setChecked(True)
    wait_saved(qtbot, panel, 1)
    time.sleep(0.5)   # 畫面一直不動：不可以再拍
    assert panel.saved_total == 1
    handle.set_control("pattern", "bars")   # 動一下（畫面大改變），再靜止
    handle.flush_controls()
    wait_saved(qtbot, panel, 2)
    time.sleep(0.4)
    assert panel.saved_total == 2


def test_still_not_triggered_while_moving(qtbot, env):
    ctx, window, handle, panel = env
    handle.set_control("pattern", "noise")   # 每張都完全不同
    handle.flush_controls()
    qtbot.wait(300)
    panel.still_check.setChecked(True)
    qtbot.wait(700)
    assert panel.saved_total == 0 and panel.pending == 0


def test_quality_blocks_capture(qtbot, env):
    ctx, window, handle, panel = env
    panel.q_sharp.setValue(5000.0)           # 任何畫面都達不到 → 視為模糊
    panel._last_quality = 0.0
    panel.on_frame(handle.last_packet)
    assert "⚠" in panel.quality_verdict.text() and "清楚" in panel.quality_verdict.text()
    panel.capture_now()                      # 預設不禁止：還是會拍
    wait_saved(qtbot, panel, 1)
    panel.block_check.setChecked(True)
    assert panel.capture_now() is False
    time.sleep(0.2)
    assert len(records(ctx)) == 1
    panel.block_check.setChecked(False)
    assert panel.capture_now() is True
    wait_saved(qtbot, panel, 2)


def test_quality_ok_display(qtbot, env):
    ctx, window, handle, panel = env
    panel._last_quality = 0.0
    panel.on_frame(handle.last_packet)
    assert "清晰度" in panel.quality_label.text()


def test_similar_photo_hint(qtbot, env):
    ctx, window, handle, panel = env
    panel.capture_now()
    wait_saved(qtbot, panel, 1)
    assert panel.similar_label.text() == ""   # 不提醒時留白（不隱藏，版面才不會跳動）
    panel.capture_now()
    assert "很像" in panel.similar_label.text()
    wait_saved(qtbot, panel, 2)
    assert len(records(ctx)) == 2          # 只提醒，不阻止
    handle.set_control("pattern", "noise")
    handle.flush_controls()
    qtbot.wait(300)
    panel.capture_now()
    assert panel.similar_label.text() == ""


def test_capture_all_cameras_same_group(qtbot, env):
    ctx, window, handle, panel = env
    h2 = window.add_source(synth("shapes"))
    qtbot.waitUntil(lambda: h2.last_packet is not None, timeout=3000)
    panel._update_state()
    assert panel.all_button.isEnabled()
    assert panel.capture_all() == 2
    wait_saved(qtbot, panel, 2)
    recs = records(ctx)
    assert len(recs) == 2
    assert recs[0].group_id and recs[0].group_id == recs[1].group_id
    assert {r.source["id"] for r in recs} == {handle.info.id, h2.info.id}


def test_no_project_shows_hint(qtbot, env):
    ctx, window, handle, panel = env
    ctx.set_project(None)
    assert not panel.project_hint.isHidden() and not panel.project_button.isHidden()
    assert panel.class_box.isHidden()
    assert panel.capture_now() is False
    assert not panel.shoot_button.isEnabled()


def test_balance_hint_for_weakest_class(qtbot, env):
    ctx, window, handle, panel = env
    for _ in range(3):
        panel.capture_now()
    wait_saved(qtbot, panel, 3)
    assert not panel.balance_hint.isHidden() and "比較少" in panel.balance_hint.text()
    assert panel.balance.height() >= len(ctx.project.classes) * 20


def test_reference_overlay_set_and_clear(qtbot, env, tmp_path):
    ctx, window, handle, panel = env
    img = QImage(32, 24, QImage.Format.Format_RGB32)
    img.fill(0xFF00AA00)
    src = tmp_path / "ref.png"
    img.save(str(src))
    view = window.video_view_for(handle)
    assert panel.set_reference_file(src)
    assert (ctx.project.dir("references") / "ref.png").is_file()
    assert view._reference is not None and abs(view._reference_opacity - 0.4) < 0.01
    panel.ref_opacity.setValue(70)
    assert abs(view._reference_opacity - 0.7) < 0.01
    panel.ref_show.setChecked(False)
    assert view._reference is None
    panel.ref_show.setChecked(True)
    assert view._reference is not None
    view.grab()  # 繪製不能出錯
    panel.clear_reference()
    assert view._reference is None
    assert panel.set_reference_file(tmp_path / "missing.png") is False


def test_settings_persist(qtbot, env):
    ctx, window, handle, panel = env
    panel.burst_count.setValue(9)
    panel.block_check.setChecked(True)
    panel.q_sharp.setValue(123.0)
    assert ctx.settings.get("capture.burst_count") == 9
    assert ctx.settings.get("capture.block_bad_quality") is True
    assert ctx.settings.get("capture.quality.min_sharpness") == 123.0


def test_flash_overlay_after_save(qtbot, env):
    ctx, window, handle, panel = env
    panel.capture_now()
    qtbot.waitUntil(lambda: panel.saved_total == 1, timeout=3000)
    ov = panel._overlay(handle)
    ov.flash()
    assert ov.flash_level == 1.0
    qtbot.waitUntil(lambda: ov.flash_level == 0.0, timeout=2000)


def test_stilldetector_logic():
    from edgelens.gui.capture_tools import StillDetector
    d = StillDetector(threshold=2.0, frames=3)
    still = np.full((48, 64, 3), 100, np.uint8)
    other = np.full((48, 64, 3), 200, np.uint8)
    fired = [d.update(still) for _ in range(6)]
    assert fired.count(True) == 1
    assert d.update(other) is False
    fired = [d.update(other) for _ in range(5)]
    assert fired.count(True) == 1
