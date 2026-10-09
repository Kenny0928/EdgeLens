"""診斷對話框：顯示、複製、匯出 ZIP、去敏。"""

from __future__ import annotations

import json
import types
import zipfile

import pytest

from edgelens import diagnostics_report as dr
from edgelens.gui import diagnostics as D
from edgelens.gui.app import create_app

pytestmark = pytest.mark.gui


@pytest.fixture
def gui(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "camera_list", lambda: [])
    monkeypatch.setattr(dr, "serial_ports", lambda: [])
    args = types.SimpleNamespace(workspace=str(tmp_path / "ws"), lang="zh_TW", safe_mode=True)
    app, ctx, window = create_app(args)
    qtbot.addWidget(window)
    yield ctx, window
    window.close()


def test_menu_and_dialog_with_secret(qtbot, gui, tmp_path):
    ctx, window = gui
    ctx.settings.set("session.sources", [{"kind": "mjpeg", "id": "x", "name": "n",
                                          "params": {"url": "http://a/b", "password": "pw-secret-9"}}])
    helps = [a for a in window.menuBar().actions() if a.menu() is not None and any(
        "診斷" in x.text() for x in a.menu().actions())]
    assert helps
    dlg = D.show_diagnostics(window, ctx)
    qtbot.waitUntil(lambda: dlg.report is not None, timeout=20000)
    assert "pw-secret-9" not in dlg.text() and "pw-secret-9" not in dlg.settings_text.toPlainText()
    assert dlg.checks.rowCount() >= 5 and dlg.zip_button.isEnabled()
    dlg.copy_to_clipboard()
    from PySide6.QtGui import QGuiApplication
    assert "EdgeLens" in QGuiApplication.clipboard().text()
    out = dlg.export_to(tmp_path / "d" / "diag.zip")
    assert out is not None and "diag.zip" in dlg.status.text()
    with zipfile.ZipFile(out) as z:
        assert {"report.json", "report.txt", "settings.json"} <= set(z.namelist())
        assert "pw-secret-9" not in "".join(z.read(n).decode() for n in z.namelist())
        assert json.loads(z.read("report.json"))["settings"]["session"]["sources"][0]["params"]["password"] == "***"
    dlg.close()


def test_export_failure_is_reported(qtbot, gui, tmp_path):
    ctx, window = gui
    dlg = D.DiagnosticsDialog(window, ctx, collector=lambda: {"edgelens_version": "x", "selftest": [], "log_tail": []})
    qtbot.waitUntil(lambda: dlg.report is not None, timeout=5000)
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert dlg.export_to(blocker / "sub" / "d.zip") is None
    assert "無法匯出" in dlg.status.text()
    dlg.close()
