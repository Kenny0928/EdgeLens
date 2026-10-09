"""StableLabel：即時更新的文字不能讓版面跳動。"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from edgelens.gui.stable import StableLabel

pytestmark = pytest.mark.gui


def _button_y(qtbot, label: StableLabel, text: str, container: QWidget, button: QPushButton) -> int:
    label.setText(text)
    container.layout().activate()
    qtbot.wait(5)
    return button.y()


def test_content_changes_do_not_move_widgets_below(qtbot):
    w = QWidget()
    lay = QVBoxLayout(w)
    label = StableLabel(3)
    button = QPushButton("下面的按鈕")
    lay.addWidget(label)
    lay.addWidget(button)
    lay.addStretch(1)
    qtbot.addWidget(w)
    w.resize(260, 400)
    w.show()
    ys = {_button_y(qtbot, label, t, w, button)
          for t in ("", "一行", "第一行\n第二行", "第一行\n第二行\n第三行", "-", "")}
    assert len(ys) == 1, f"按鈕位置跟著內容跳動：{ys}"


def test_grows_but_never_shrinks_until_reset(qtbot):
    label = StableLabel(1)
    qtbot.addWidget(label)
    label.resize(200, 20)
    label.show()
    base = label.minimumHeight()
    label.setText("1\n2\n3\n4")
    grown = label.minimumHeight()
    assert grown > base
    label.setText("短")
    assert label.minimumHeight() == grown       # 不縮回，避免來回跳
    label.reset_height()
    assert label.minimumHeight() == base
