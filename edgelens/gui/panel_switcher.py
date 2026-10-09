"""右側面板切換：左邊一欄面板名稱（直向清單），右邊顯示選到的面板。

面板有 8 個以上時，橫向分頁會擠到右邊的箭頭後面，學生找不到「比較」「K210」等面板；
直向清單每個名稱都看得到。提供和 QTabWidget 相同的常用方法（addTab、tabText、setTabText、
indexOf、count、widget、currentWidget、setCurrentWidget、setCurrentIndex、currentIndex、
currentChanged），主視窗與測試不用改寫。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QListWidget, QListWidgetItem, QStackedWidget, QWidget

ITEM_PADDING_PX = 10


class PanelSwitcher(QWidget):
    currentChanged = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.list = QListWidget()
        self.list.setFrameShape(QFrame.Shape.NoFrame)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setStyleSheet(f"QListWidget::item {{ padding: {ITEM_PADDING_PX}px 8px; }}")
        self.stack = QStackedWidget()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.list)
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.VLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        lay.addWidget(divider)
        lay.addWidget(self.stack, 1)
        self.list.currentRowChanged.connect(self._on_row)

    # ---- QTabWidget 相容的方法 ----------------------------------------------------
    def addTab(self, widget: QWidget, title: str) -> int:  # noqa: N802 - 與 QTabWidget 同名
        self.stack.addWidget(widget)
        self.list.addItem(QListWidgetItem(title))
        self._fit_width()
        if self.list.currentRow() < 0:
            self.list.setCurrentRow(0)
        return self.count() - 1

    def count(self) -> int:
        return self.stack.count()

    def tabText(self, index: int) -> str:  # noqa: N802
        item = self.list.item(index)
        return item.text() if item is not None else ""

    def setTabText(self, index: int, title: str) -> None:  # noqa: N802
        item = self.list.item(index)
        if item is not None:
            item.setText(title)
            self._fit_width()

    def indexOf(self, widget: QWidget) -> int:  # noqa: N802
        return self.stack.indexOf(widget)

    def widget(self, index: int) -> QWidget | None:
        return self.stack.widget(index)

    def currentIndex(self) -> int:  # noqa: N802
        return self.stack.currentIndex()

    def currentWidget(self) -> QWidget | None:  # noqa: N802
        return self.stack.currentWidget()

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802
        self.list.setCurrentRow(index)

    def setCurrentWidget(self, widget: QWidget) -> None:  # noqa: N802
        index = self.indexOf(widget)
        if index >= 0:
            self.setCurrentIndex(index)

    # ---- 內部 ---------------------------------------------------------------------
    def _on_row(self, row: int) -> None:
        if 0 <= row < self.stack.count():
            self.stack.setCurrentIndex(row)
            self.currentChanged.emit(row)

    def _fit_width(self) -> None:
        """清單寬度剛好放得下最長的名稱（換語言時會重算）。"""
        fm = self.list.fontMetrics()
        widest = max((fm.horizontalAdvance(self.list.item(i).text()) for i in range(self.list.count())), default=60)
        self.list.setFixedWidth(widest + 2 * 8 + 12)
