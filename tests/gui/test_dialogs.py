from __future__ import annotations

import types

import pytest

from edgelens.core.project import Workspace
from edgelens.core.settings import Settings
from edgelens.gui.context import AppContext
from edgelens.gui.project_dialog import ProjectDialog
from edgelens.gui.source_dialog import SourceDialog
from edgelens.sources.base import SOURCES, SourceInfo

pytestmark = pytest.mark.gui


@pytest.fixture
def ctx(tmp_path):
    from edgelens.core.registry import discover

    discover()
    c = AppContext(Settings(tmp_path / "s.json"), Workspace(tmp_path / "ws"))
    yield c
    c.dispose()


def test_source_dialog_lists_synthetic_and_manual_add(qtbot, ctx):
    dlg = SourceDialog(ctx)
    qtbot.addWidget(dlg)
    kinds = [dlg.kind_list.item(i).data(0x0100) for i in range(dlg.kind_list.count())]
    assert "synthetic" in kinds
    dlg.select_kind("synthetic")
    dlg.show()
    assert dlg.manual_box.isVisible()
    dlg.manual_panel.set_values({"pattern": "bars", "fps": 10})
    dlg.manual_button.click()
    info = dlg.selected_info()
    assert isinstance(info, SourceInfo)
    assert info.kind == "synthetic" and info.id == "synthetic:bars"
    assert info.params["fps"] == 10


def test_source_dialog_discover_safe_source(qtbot, ctx):
    dlg = SourceDialog(ctx)
    qtbot.addWidget(dlg)
    safe = [k for k, c in SOURCES.items() if c.discover_is_safe and c.available()[0]]
    assert "synthetic" in safe
    dlg.select_kind("synthetic")
    dlg.start_discover()
    assert dlg.is_searching()
    qtbot.waitUntil(lambda: dlg.result_list.count() > 0, timeout=3000)
    qtbot.waitUntil(lambda: not dlg.is_searching(), timeout=3000)
    dlg.result_list.setCurrentRow(0)
    dlg.result_list.item(0).setSelected(True)
    dlg.buttons.accepted.emit()
    assert dlg.selected_info().kind == "synthetic"


def test_source_dialog_recent_listed_first(qtbot, ctx):
    info = SourceInfo(kind="synthetic", id="synthetic:noise", name="Noise one", params={"pattern": "noise"})
    ctx.settings.set("recent.sources", [info.to_dict()])
    dlg = SourceDialog(ctx)
    qtbot.addWidget(dlg)
    assert dlg.recent_list.count() == 1
    dlg.recent_list.setCurrentRow(0)
    dlg.recent_list.item(0).setSelected(True)
    dlg._accept_selected()
    assert dlg.selected_info().id == "synthetic:noise"


def test_project_dialog_create_from_template_and_add_class(qtbot, ctx):
    dlg = ProjectDialog(ctx)
    qtbot.addWidget(dlg)
    assert dlg.template_combo.findData("jiaobei") >= 0
    project = dlg.create_project("測試專案", "jiaobei")
    assert [c.id for c in project.classes][:2] == ["holy", "laughing"]
    dlg.tabs.setCurrentIndex(ProjectDialog.TAB_CLASSES)
    ed = dlg.class_editor
    n = ed.table.rowCount()
    assert n == 4
    row = ed.add_class(zh="新類別", en="New thing")
    assert ed.table.item(row, 0).text() == "new_thing"
    ed.confirm_delete = lambda name: True
    ed.table.selectRow(3)
    ed.remove_selected()
    dlg.accept()
    saved = ctx.workspace.open_project(project.id)
    ids = [c.id for c in saved.classes]
    assert "new_thing" in ids and "empty" not in ids
    assert dlg.selected_project().id == project.id


def test_class_editor_validation(qtbot):
    from edgelens.gui.project_dialog import ClassEditor

    ed = ClassEditor()
    qtbot.addWidget(ed)
    ed.add_class(zh="甲", en="a")
    ed.add_class(zh="乙", en="a")  # ID 自動避開重複
    ids = [ed.table.item(r, 0).text() for r in range(2)]
    assert ids == ["a", "a_2"]
    ed.table.item(1, 0).setText("a")
    with pytest.raises(ValueError):
        ed.collect()
    ed.table.item(1, 0).setText("Bad ID")
    with pytest.raises(ValueError):
        ed.collect()
    ed.table.item(1, 0).setText("b")
    specs = ed.collect()
    assert [s.id for s in specs] == ["a", "b"]
    assert specs[0].hotkey == "1" and specs[1].hotkey == "2"


def test_workspace_dialog_cloud_warning(qtbot, tmp_path):
    from edgelens.gui.workspace_dialog import WorkspaceDialog

    dlg = WorkspaceDialog(None, tmp_path / "x", first_run=True)
    qtbot.addWidget(dlg)
    assert dlg.warning.text() == ""
    dlg.path_edit.setText(str(tmp_path / "Dropbox" / "EdgeLens"))
    assert "Dropbox" in dlg.warning.text()
    assert dlg.chosen_path().name == "EdgeLens"
    _ = types
