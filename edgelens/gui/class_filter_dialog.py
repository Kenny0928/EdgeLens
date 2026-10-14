"""類別篩選對話框：從模型的類別裡勾選要顯示的（例如 365 類只看「人」和「杯子」）。

搜尋同時比對顯示名稱與原始名稱（繁中介面也能打英文找），「全選」「全不選」只作用在目前搜尋到的項目。
全部勾選等於不篩選，selected() 回傳 None。
"""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from edgelens.i18n import tr


class ClassFilterDialog(QDialog):
    def __init__(self, labels: list[str], alt_labels: list[str] | None = None,
                 selected: Iterable[int] | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(tr("gui.model.filter.title"))
        self.resize(420, 560)
        chosen = set(selected) if selected else set(range(len(labels)))
        alt = alt_labels if alt_labels and len(alt_labels) == len(labels) else []

        lay = QVBoxLayout(self)
        note = QLabel(tr("gui.model.filter.note"))
        note.setWordWrap(True)
        lay.addWidget(note)
        self.search = QLineEdit()
        self.search.setPlaceholderText(tr("gui.model.filter.search"))
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._apply_search)
        lay.addWidget(self.search)

        self.list = QListWidget()
        for i, name in enumerate(labels):
            text = f"{i}  {name}" + (f"（{alt[i]}）" if alt and alt[i] != name else "")
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if i in chosen else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, i)
            item.setData(Qt.ItemDataRole.UserRole + 1, f"{name} {alt[i] if alt else ''}".lower())
            self.list.addItem(item)
        self.list.itemChanged.connect(lambda _i: self._update_count())
        lay.addWidget(self.list, 1)

        row = QHBoxLayout()
        self.all_button = QPushButton(tr("gui.model.filter.check_all"))
        self.all_button.clicked.connect(lambda: self._set_visible_checked(True))
        self.none_button = QPushButton(tr("gui.model.filter.check_none"))
        self.none_button.clicked.connect(lambda: self._set_visible_checked(False))
        self.count_label = QLabel()
        row.addWidget(self.all_button)
        row.addWidget(self.none_button)
        row.addStretch(1)
        row.addWidget(self.count_label)
        lay.addLayout(row)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr("common.apply"))
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr("common.cancel"))
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self._update_count()

    def _items(self) -> list[QListWidgetItem]:
        return [self.list.item(i) for i in range(self.list.count())]

    def _apply_search(self, text: str) -> None:
        needle = text.strip().lower()
        for item in self._items():
            item.setHidden(bool(needle) and needle not in item.data(Qt.ItemDataRole.UserRole + 1))

    def _set_visible_checked(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.list.blockSignals(True)
        for item in self._items():
            if not item.isHidden():
                item.setCheckState(state)
        self.list.blockSignals(False)
        self._update_count()

    def _checked(self) -> list[int]:
        return [int(it.data(Qt.ItemDataRole.UserRole)) for it in self._items()
                if it.checkState() == Qt.CheckState.Checked]

    def _update_count(self) -> None:
        n = len(self._checked())
        self.count_label.setText(tr("gui.model.filter.count", count=n, total=self.list.count()))
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(n > 0)

    def selected(self) -> list[int] | None:
        """勾選的類別編號；全部勾選時回傳 None（不篩選）。"""
        ids = self._checked()
        return None if len(ids) == self.list.count() else ids
