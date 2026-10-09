"""影像處理面板：步驟編輯、參數、預設組合、吸管取色（synthetic 來源，不碰實體相機）。"""

from __future__ import annotations

import types

import pytest
from PySide6.QtGui import QGuiApplication

from edgelens.gui.app import create_app
from edgelens.sources.base import SourceInfo
from edgelens.sources.synthetic import SyntheticSource

pytestmark = pytest.mark.gui


def synth(pattern: str = "shapes", **params) -> SourceInfo:
    return SyntheticSource.info_from_params({"pattern": pattern, "fps": 60, "resolution": "320x240", **params})


@pytest.fixture
def env(qtbot, tmp_path):
    args = types.SimpleNamespace(workspace=str(tmp_path / "ws"), lang="zh_TW", safe_mode=True)
    app, ctx, window = create_app(args)
    qtbot.addWidget(window)
    handle = window.add_source(synth())
    qtbot.waitUntil(lambda: handle.last_packet is not None, timeout=3000)
    panel = window.panel("pipeline")
    yield ctx, window, handle, panel
    window.close()


def kinds(handle):
    return [s["kind"] for s in handle.pipeline_dict.get("steps", [])]


def wait_timing(qtbot, handle, kind, present=True):
    def ok():
        pkt = handle.last_packet
        return pkt is not None and (kind in [k for k, _ in pkt.result.timings_ms]) == present
    qtbot.waitUntil(ok, timeout=3000)


def test_panel_registered_and_empty_hint(qtbot, tmp_path):
    args = types.SimpleNamespace(workspace=str(tmp_path / "ws"), lang="zh_TW", safe_mode=True)
    _, ctx, window = create_app(args)
    qtbot.addWidget(window)
    panel = window.panel("pipeline")
    assert panel is not None and panel.order == 20
    assert not panel.empty_label.isHidden() and panel.main_box.isHidden()
    window.close()


def test_add_remove_move_disable(qtbot, env):
    ctx, window, handle, panel = env
    assert not panel.main_box.isHidden()
    panel.add_step("color.grayscale")
    assert kinds(handle)[0] == "color.grayscale"
    wait_timing(qtbot, handle, "color.grayscale")
    panel.add_step("color.hsv_mask")
    assert kinds(handle) == ["color.grayscale", "color.hsv_mask"]
    panel.select(1)
    panel.move_step(-1)
    assert kinds(handle) == ["color.hsv_mask", "color.grayscale"]
    panel.duplicate_step()
    assert kinds(handle) == ["color.hsv_mask", "color.hsv_mask", "color.grayscale"]
    panel.set_enabled(0, False)
    assert handle.pipeline_dict["steps"][0]["enabled"] is False
    panel.select(1)
    panel.remove_step()
    assert kinds(handle) == ["color.hsv_mask", "color.grayscale"]
    wait_timing(qtbot, handle, "color.grayscale")
    # worker 真的套用：停用的遮罩不在耗時清單裡
    qtbot.waitUntil(lambda: [k for k, _ in handle.last_packet.result.timings_ms] == ["color.grayscale"], timeout=3000)
    panel.clear_steps()
    assert kinds(handle) == []
    qtbot.waitUntil(lambda: handle.last_packet.result.timings_ms == [], timeout=3000)
    assert ctx.settings.source_settings(handle.info.id)["pipeline"]["steps"] == []


def test_param_change_reaches_worker(qtbot, env):
    ctx, window, handle, panel = env
    idx = panel.add_step("color.hsv_mask")
    panel.set_param(idx, "h_min", 33)
    assert handle.pipeline_dict["steps"][idx]["params"]["h_min"] == 33
    qtbot.waitUntil(lambda: handle.last_packet is not None, timeout=2000)
    # 實際用 ControlPanel 的元件修改也要生效
    panel.controls.row("s_min").spin.setValue(77)
    assert handle.pipeline_dict["steps"][idx]["params"]["s_min"] == 77


def test_timings_and_errors_shown(qtbot, env):
    ctx, window, handle, panel = env
    panel.add_step("color.grayscale")
    wait_timing(qtbot, handle, "color.grayscale")
    panel._last_timing = 0.0
    panel.on_frame(handle.last_packet)
    assert " ms" in panel.list.item(0).text()
    assert "ms" in panel.time_label.text()


def test_slow_warning(qtbot, env):
    ctx, window, handle, panel = env
    panel.add_step("color.grayscale")
    wait_timing(qtbot, handle, "color.grayscale")
    pkt = handle.last_packet
    import copy
    slow = copy.copy(pkt)
    slow.result = copy.copy(pkt.result)
    slow.result.total_ms = 80.0
    panel._update_timing_labels(slow)
    assert "80" in panel.slow_label.text()
    panel._update_timing_labels(pkt)
    assert panel.slow_label.text() == ""          # 不慢時留白（不隱藏，版面才不會跳動）


def test_presets_apply_and_save(qtbot, env):
    ctx, window, handle, panel = env
    names = [p.name for p in panel._presets]
    assert "red_objects.json" in names
    info = next(p for p in panel._presets if p.name == "red_objects.json")
    assert panel.apply_preset(info)
    assert kinds(handle)[0] == "color.hsv_mask"
    qtbot.waitUntil(lambda: "color.hsv_mask" in [k for k, _ in handle.last_packet.result.timings_ms], timeout=3000)
    path = panel.save_preset("我的紅色")
    assert path is not None and path.parent == ctx.project.pipelines_dir
    assert any((not p.builtin) and p.path == path for p in panel._presets)
    panel.clear_steps()
    user = next(p for p in panel._presets if p.path == path)
    panel.apply_preset(user)
    assert kinds(handle)[0] == "color.hsv_mask"


def test_show_processed_sync(qtbot, env):
    ctx, window, handle, panel = env
    panel.show_processed.setChecked(False)
    assert window._show_processed is False and window.act_raw.isChecked()
    window.set_show_processed(True)
    assert panel.show_processed.isChecked()


def test_pick_color_applies_hsv_mask_and_clipboard(qtbot, env):
    ctx, window, handle, panel = env
    view = window.video_view_for(handle)
    panel.pick_button.setChecked(True)
    assert view.mode == "pick"
    # synthetic shapes：左邊的圓是紅色 (BGR 0,0,255)，位置約 (w/4, h/2)
    view.pointPicked.emit(80.0, 120.0)
    view.pointPicked.emit(82.0, 118.0)
    assert len(panel._samples) == 2
    assert "H" in panel.sample_info.text()
    assert panel.apply_to_hsv_mask()
    assert "color.hsv_mask" in kinds(handle)
    p = handle.pipeline_dict["steps"][kinds(handle).index("color.hsv_mask")]["params"]
    # 紅色跨 0：h_min > h_max，且飽和度與明度下限合理
    assert p["h_min"] > p["h_max"]
    assert p["v_max"] >= 250
    assert 0 <= p["s_min"] < p["s_max"] <= 255
    qtbot.waitUntil(lambda: "color.hsv_mask" in [k for k, _ in handle.last_packet.result.timings_ms], timeout=3000)
    # 再套用一次：更新同一個步驟，不新增
    view.pointPicked.emit(80.0, 120.0)
    panel.apply_to_hsv_mask()
    assert kinds(handle).count("color.hsv_mask") == 1

    lab = panel.copy_maixpy_lab()
    assert QGuiApplication.clipboard().text() == lab
    nums = [int(x) for x in lab.strip("()").split(",")]
    assert len(nums) == 6 and nums[0] <= nums[1] and nums[2] <= nums[3] and nums[4] <= nums[5]
    hsv = panel.copy_opencv_hsv()
    assert QGuiApplication.clipboard().text() == hsv
    assert "np.array(" in hsv and "inRange" in hsv and "lower1" in hsv  # 紅色兩段

    panel.pick_button.setChecked(False)
    assert view.mode == "none"
    panel.clear_samples()
    assert not panel.apply_hsv_button.isEnabled()


def test_pick_mode_reset_on_camera_change(qtbot, env):
    ctx, window, handle, panel = env
    view = window.video_view_for(handle)
    panel.pick_button.setChecked(True)
    h2 = window.add_source(synth("bars"))
    assert view.mode == "none"
    assert not panel.pick_button.isChecked()
    assert panel._handle is h2


def test_add_menu_groups_and_plugin_tag(qtbot, env):
    ctx, window, handle, panel = env
    titles = [a.text() for a in panel.add_menu.actions()]
    assert "顏色" in titles
    assert panel.is_plugin("color.grayscale") is False
