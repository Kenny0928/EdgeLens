"""K210 面板：用模擬板（sim:）測試板子資訊、即時數據、上傳、頻率；燒錄用假函式。"""

from __future__ import annotations

import types

import pytest

from edgelens.gui.app import create_app
from edgelens.sources.base import SourceInfo
from edgelens.sources.k210 import flash as flash_mod
from edgelens.sources.k210 import source as S
from edgelens.sources.k210.sim import FakeK210Board
from edgelens.sources.synthetic import SyntheticSource

pytestmark = pytest.mark.gui


@pytest.fixture
def boards(monkeypatch):
    made = []

    def factory(port, baud):
        b = FakeK210Board()
        b.time_scale = 0.0
        made.append(b)
        return b

    monkeypatch.setattr(S, "open_serial", factory)
    return made


@pytest.fixture
def gui(qtbot, tmp_path):
    args = types.SimpleNamespace(workspace=str(tmp_path / "ws"), lang="zh_TW", safe_mode=True)
    app, ctx, window = create_app(args)
    qtbot.addWidget(window)
    window.show()
    yield ctx, window, window.panel("k210")
    window.close()


def add_k210(qtbot, window):
    h = window.add_source(SourceInfo(kind="k210", id="k210:sim", name="K210 sim", params={"port": "sim:"}))
    qtbot.waitUntil(lambda: h.last_packet is not None, timeout=5000)
    return h


def test_non_k210_shows_hint(qtbot, gui):
    ctx, window, panel = gui
    assert panel.order == 80 and panel.empty_label.isVisibleTo(panel)
    window.add_source(SyntheticSource.info_from_params({"fps": 30}))
    qtbot.wait(300)
    assert panel.empty_label.isVisibleTo(panel) and not panel.content.isVisibleTo(panel)


def test_info_and_live(qtbot, gui, boards):
    ctx, window, panel = gui
    add_k210(qtbot, window)
    assert panel.content.isVisibleTo(panel)
    qtbot.waitUntil(lambda: "simulated" in panel._info["fw"].text(), timeout=3000)
    assert "GC0328" in panel._info["sensor"].text()
    qtbot.waitUntil(lambda: panel._live["fps"].text() == "20.0", timeout=3000)
    assert "MB" in panel._live["mem"].text()
    assert panel.us_table.rowCount() == 3 and "ms" in panel.us_table.item(0, 1).text()
    bench = window.panel("bench")
    qtbot.waitUntil(lambda: bench.board_table.isVisibleTo(bench), timeout=3000)


def test_upload_flow(qtbot, gui, boards, tmp_path, monkeypatch):
    ctx, window, panel = gui
    h = add_k210(qtbot, window)
    f = tmp_path / "data.bin"
    f.write_bytes(b"0123456789" * 300)
    panel.upload_path.setText(str(f))
    panel.upload_name.setText("data.bin")
    monkeypatch.setattr(type(panel), "_confirm", lambda self, *a, **k: False)
    assert panel.upload() is False and not panel._busy_key
    monkeypatch.setattr(type(panel), "_confirm", lambda self, *a, **k: True)
    with qtbot.waitSignal(h.actionFinished, timeout=8000):
        assert panel.upload()
        assert not panel.upload_button.isEnabled()
    qtbot.waitUntil(lambda: panel.last_result is not None, timeout=2000)
    assert panel.last_result.ok and "/flash/data.bin" in panel.result_label.text()
    assert bytes(boards[-1].files["/flash/data.bin"]) == f.read_bytes()
    assert panel.upload_button.isEnabled()
    panel.upload_path.setText(str(tmp_path / "nope"))
    assert panel.upload() is False


def test_set_freq(qtbot, gui, boards, monkeypatch):
    ctx, window, panel = gui
    h = add_k210(qtbot, window)
    texts = []
    monkeypatch.setattr(type(panel), "_confirm", lambda self, t, text, strong=False: texts.append(text) or True)
    panel.cpu_spin.setValue(500)
    with qtbot.waitSignal(h.actionFinished, timeout=15000):
        panel.set_freq()
    assert "超頻" in texts[0]
    assert panel.last_result.ok
    assert "cpu=500" in boards[0].freq_calls[0]


def test_flash_uses_fake(qtbot, gui, boards, tmp_path, monkeypatch):
    ctx, window, panel = gui
    h = add_k210(qtbot, window)
    calls = {}

    def fake(port, fw, **kw):
        calls.update(port=port, fw=fw, board=kw.get("board"))
        return flash_mod.FlashResult(True, "done")

    monkeypatch.setattr(flash_mod, "flash_firmware", fake)
    monkeypatch.setattr(type(panel), "_confirm", lambda self, t, text, strong=False: strong)
    assert panel.flash() is False   # 沒選檔案
    fw = tmp_path / "fw.bin"
    fw.write_bytes(b"x" * 64)
    panel.flash_path.setText(str(fw))
    panel.board_combo.setCurrentText("bit")
    with qtbot.waitSignal(h.actionFinished, timeout=8000):
        assert panel.flash()
    assert calls == {"port": "sim:", "fw": str(fw), "board": "bit"}
    assert panel.last_result.ok
