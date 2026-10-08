from __future__ import annotations

import pytest

from edgelens.core.controls import ControlSpec, ControlType
from edgelens.gui.control_panel import ControlPanel

pytestmark = pytest.mark.gui

SPECS = [
    ControlSpec("i", ControlType.INT, label="Int", default=5, min=0, max=10, unit="px"),
    ControlSpec("f", ControlType.FLOAT, label="Float", default=0.5, min=0, max=1, step=0.1),
    ControlSpec("b", ControlType.BOOL, label="Bool", default=False),
    ControlSpec("e", ControlType.ENUM, label="Enum", default="a", options=(("a", "control.synthetic.pattern.bars"), ("b", "Bee"))),
    ControlSpec("s", ControlType.STR, label="Str"),
    ControlSpec("c", ControlType.COLOR, label="Color", default="#FF0000"),
    ControlSpec("go", ControlType.ACTION, label="Go", group="action"),
    ControlSpec("adv", ControlType.INT, label="Adv", default=1, min=0, max=5, group="advanced", advanced=True),
]


@pytest.fixture
def panel(qtbot):
    p = ControlPanel()
    qtbot.addWidget(p)
    p.set_specs(SPECS, {"i": 5})
    p.show()
    return p


def collect(panel):
    got = []
    panel.valueChanged.connect(lambda k, v: got.append((k, v)))
    return got


def test_each_type_emits_correct_type(panel):
    got = collect(panel)
    panel.row("i").spin.setValue(7)
    panel.row("f").spin.setValue(0.8)
    panel.row("b").widget.setChecked(True)
    panel.row("e").widget.setCurrentIndex(1)
    panel.row("s").widget.setText("hello")
    panel.row("s").widget.editingFinished.emit()
    panel.row("c").user_set_color("#00FF00")
    d = dict(got)
    assert d["i"] == 7 and isinstance(d["i"], int)
    assert d["f"] == pytest.approx(0.8) and isinstance(d["f"], float)
    assert d["b"] is True
    assert d["e"] == "b"
    assert d["s"] == "hello"
    assert d["c"] == "#00FF00"


def test_slider_emits_continuously_and_syncs_spin(panel):
    got = collect(panel)
    sl = panel.row("i").slider
    sl.setValue(3)
    sl.setValue(4)
    assert [v for k, v in got if k == "i"] == [3, 4]
    assert panel.row("i").spin.value() == 4
    assert panel.row("i").spin.suffix() == " px"


def test_set_values_does_not_emit(panel):
    got = collect(panel)
    panel.set_values({"i": 9, "b": True, "e": "b", "s": "x", "c": "#112233", "f": 0.3})
    assert got == []
    v = panel.values()
    assert v["i"] == 9 and v["b"] is True and v["e"] == "b" and v["s"] == "x" and v["c"] == "#112233"
    assert v["f"] == pytest.approx(0.3)
    assert panel.row("i").slider.value() == 9


def test_action_signal(panel, qtbot):
    with qtbot.waitSignal(panel.actionTriggered) as sig:
        panel.row("go").widget.click()
    assert sig.args == ["go"]


def test_advanced_collapsed_and_toggle(panel):
    assert panel.has_advanced()
    assert not panel.advanced_visible()
    panel.set_advanced_expanded(True)
    assert panel.advanced_visible()
    panel.retranslate()  # 重建後仍保持展開
    assert panel.advanced_visible()
    assert panel.values()["adv"] == 1


def test_status_indicator(panel):
    panel.set_status("i", True)
    assert panel.row("i").status.text() == "✓"
    panel.set_status("i", False, "no")
    assert panel.row("i").status.text() == "✗"
    panel.set_status("i", None)
    assert panel.row("i").status.text() == "?"


def test_enum_option_text_translated(panel):
    from edgelens import i18n

    assert panel.row("e").widget.itemText(0) == i18n.tr("control.synthetic.pattern.bars")
    assert panel.row("e").widget.itemText(1) == "Bee"


def test_str_browse_button_and_secret(qtbot, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QLineEdit, QPushButton

    from edgelens.core.controls import ControlSpec, ControlType
    from edgelens.gui.control_panel import ControlPanel

    panel = ControlPanel()
    qtbot.addWidget(panel)
    specs = [ControlSpec("path", ControlType.STR, label="path", extra={"browse": "folder"}),
             ControlSpec("password", ControlType.STR, label="pw", extra={"secret": True})]
    panel.set_specs(specs, {"path": "", "password": ""})
    got = []
    panel.valueChanged.connect(lambda k, v: got.append((k, v)))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: "/tmp/影像"))
    row = panel.row("path")
    row.widget.findChild(QPushButton).click()
    assert got == [("path", "/tmp/影像")] and panel.values()["path"] == "/tmp/影像"
    panel.set_values({"path": "/x"})
    assert panel.values()["path"] == "/x"
    assert panel.row("password").editor.echoMode() == QLineEdit.EchoMode.Password
