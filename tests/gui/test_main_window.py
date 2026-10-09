"""主視窗整合測試：用 synthetic 來源，不碰實體相機。"""

from __future__ import annotations

import types

import pytest
from PySide6.QtCore import QThread

from edgelens import i18n
from edgelens.gui.app import create_app
from edgelens.sources.base import SourceInfo
from edgelens.sources.synthetic import SyntheticSource

pytestmark = pytest.mark.gui


def synth(pattern: str = "moving_box", **params) -> SourceInfo:
    return SyntheticSource.info_from_params({"pattern": pattern, "fps": 60, "resolution": "320x240", **params})


@pytest.fixture
def gui(qtbot, tmp_path):
    args = types.SimpleNamespace(workspace=str(tmp_path / "ws"), lang="zh_TW", safe_mode=True)
    app, ctx, window = create_app(args)
    qtbot.addWidget(window)
    yield ctx, window
    window.close()


def wait_frame(qtbot, handle, timeout=3000):
    qtbot.waitUntil(lambda: handle.last_packet is not None, timeout=timeout)


def test_view_receives_frames_and_hud(qtbot, gui):
    ctx, window = gui
    handle = window.add_source(synth())
    wait_frame(qtbot, handle, 2000)
    view = window.video_view_for(handle)
    qtbot.waitUntil(lambda: view.image is not None and len(view._hud) >= 2, timeout=2000)
    assert any("FPS" in line for line in view._hud)
    assert ctx.active_source is handle


def test_two_sources_grid_and_active_switch(qtbot, gui):
    ctx, window = gui
    h1 = window.add_source(synth("moving_box"))
    h2 = window.add_source(synth("bars"))
    window.set_layout_mode("grid")
    window.show()
    wait_frame(qtbot, h1)
    wait_frame(qtbot, h2)
    qtbot.waitUntil(lambda: all(window.video_view_for(h).image is not None for h in (h1, h2)), timeout=2000)
    assert all(window.video_view_for(h).isVisible() for h in (h1, h2))
    camera = window.panel("camera")
    window.set_active_handle(h1)
    qtbot.waitUntil(lambda: camera.controls.values().get("pattern") == "moving_box", timeout=2000)
    window.set_active_handle(h2)
    qtbot.waitUntil(lambda: camera.controls.values().get("pattern") == "bars", timeout=2000)


def test_brightness_verified_and_saved(qtbot, gui):
    ctx, window = gui
    handle = window.add_source(synth())
    wait_frame(qtbot, handle)
    camera = window.panel("camera")
    qtbot.waitUntil(lambda: "brightness" in camera.controls.keys(), timeout=2000)
    with qtbot.waitSignal(handle.controlResult, timeout=3000, check_params_cb=lambda k, r: k == "brightness") as sig:
        camera.controls.row("brightness").spin.setValue(25)
    assert sig.args[1].verified is True
    assert ctx.settings.source_settings(handle.info.id)["controls"]["brightness"] == 25
    assert camera.controls._rows["brightness"].status.text() == "✓"


def test_language_switch_updates_everything(qtbot, gui):
    ctx, window = gui
    handle = window.add_source(synth())
    wait_frame(qtbot, handle)
    camera = window.panel("camera")
    qtbot.waitUntil(lambda: "brightness" in camera.controls.keys(), timeout=2000)
    menu_zh = window._menus["file"].title()
    tab_zh = window.tabs.tabText(0)
    label_zh = camera.controls.row("brightness").label.text()
    ctx.set_language("en")
    qtbot.waitUntil(lambda: window._menus["file"].title() != menu_zh, timeout=1000)
    assert window._menus["file"].title() == "File"
    assert window.tabs.tabText(0) == "Camera" != tab_zh
    assert camera.controls.row("brightness").label.text() == "Brightness" != label_zh
    assert window.act_add_camera.text() == "Add camera"


def test_project_and_status(qtbot, gui):
    ctx, window = gui
    assert ctx.project is not None
    assert ctx.project.name in window.windowTitle()
    assert ctx.project.name in window.status_project.text()


def test_shortcut_and_menu_api(qtbot, gui):
    from PySide6.QtGui import QAction

    ctx, window = gui
    hit = []
    sc = window.register_shortcut("Ctrl+Shift+K", lambda: hit.append(1))
    assert sc is not None
    act = QAction("x", window)
    window.add_menu_action("help", act)
    assert act in window._menus["help"].actions()
    window.add_menu_action("brand_new_menu", QAction("y", window))
    assert "brand_new_menu" in window._menus


def test_close_stops_all_threads(qtbot, gui):
    ctx, window = gui
    handles = [window.add_source(synth("moving_box")), window.add_source(synth("bars"))]
    for h in handles:
        wait_frame(qtbot, h)
    threads = [h.thread for h in handles]
    assert all(isinstance(t, QThread) and t.isRunning() for t in threads)
    window.close()
    assert all(t.isFinished() for t in threads)


def test_retry_after_error(qtbot, gui):
    """來源開啟失敗時顯示錯誤橫幅，retry 後可恢復。"""
    from edgelens.sources.base import SOURCES, FrameSource, SourceError, register_source

    class Flaky(FrameSource):
        kind = "test_flaky_gui"
        display_name_key = "source.synthetic.name"
        attempts = 0

        @property
        def is_open(self):
            return False

        def open(self):
            type(self).attempts += 1
            raise SourceError("測試：打不開", retryable=True, hint="再試一次")

        def read(self, timeout=1.0):
            return None

        def close(self):
            pass

    register_source(Flaky)
    try:
        ctx, window = gui
        h = window.add_source(SourceInfo(kind="test_flaky_gui", id="flaky:1", name="Flaky"))
        view = window.video_view_for(h)
        qtbot.waitUntil(lambda: h.state == "error", timeout=2000)
        qtbot.waitUntil(view.error_visible, timeout=1000)
        assert view._banner.button.isVisibleTo(view._banner)
        h.retry()
        qtbot.waitUntil(lambda: Flaky.attempts >= 2, timeout=2000)
    finally:
        SOURCES.unregister("test_flaky_gui")


def test_language_is_restored_by_fixture(gui):
    assert i18n.current_language() == "zh_TW"


def test_hud_says_when_file_playback_ended(qtbot, gui, tmp_path):
    """影片／圖片資料夾播完（不循環）時畫面會停住，HUD 要說明「已播完」，學生才不會以為當機。"""
    import cv2
    import numpy as np

    folder = tmp_path / "圖片"
    folder.mkdir()
    for i in range(2):
        ok, buf = cv2.imencode(".png", np.full((24, 32, 3), 60 * i, np.uint8))
        (folder / f"{i}.png").write_bytes(buf.tobytes())
    ctx, window = gui
    info = SourceInfo(kind="file", id=f"file:{folder}", name="圖片", params={"path": str(folder), "loop": False,
                                                                                   "fps": 30})
    handle = window.add_source(info)
    view = window.video_view_for(handle)
    qtbot.waitUntil(lambda: any("已播完" in line for line in view._hud), timeout=4000)
