"""專案對話框：選擇專案、新增專案（可用範本）、編輯類別。"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from edgelens import i18n
from edgelens.core.paths import is_valid_id, slugify
from edgelens.core.project import DEFAULT_COLORS, ClassSpec, Project, project_templates
from edgelens.gui.context import AppContext
from edgelens.i18n import tr, tr_or

COL_ID, COL_ZH, COL_EN, COL_COLOR, COL_KEY = range(5)


def template_title(key: str, tpl: dict[str, Any], lang: str) -> str:
    title = tpl.get("title")
    fallback = title.get(lang) or title.get("zh_TW") or key if isinstance(title, dict) else (title or key)
    return tr_or(f"project.template.{key}", fallback)


class ClassEditor(QWidget):
    """類別表格：顯示名稱（中／英）、ID、顏色、快捷鍵；可新增、刪除、排序。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.project: Project | None = None
        self._busy = False
        self._existing: set[str] = set()
        self.confirm_delete = self._ask_confirm_delete
        lay = QVBoxLayout(self)
        self.note = QLabel(tr("gui.project.class_note"))
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([tr("gui.project.col.id"), tr("gui.project.col.zh"),
                                              tr("gui.project.col.en"), tr("gui.project.col.color"),
                                              tr("gui.project.col.hotkey")])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.cellChanged.connect(self._cell_changed)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.add_button = QPushButton(tr("common.add"))
        self.del_button = QPushButton(tr("common.delete"))
        self.up_button = QPushButton("↑")
        self.down_button = QPushButton("↓")
        self.add_button.clicked.connect(lambda: self.add_class())
        self.del_button.clicked.connect(self.remove_selected)
        self.up_button.clicked.connect(lambda: self.move_selected(-1))
        self.down_button.clicked.connect(lambda: self.move_selected(1))
        for b in (self.add_button, self.del_button, self.up_button, self.down_button):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

    # ---- 載入／讀出 -------------------------------------------------------------
    def set_project(self, project: Project | None) -> None:
        self.project = project
        self._existing = {c.id for c in project.classes} if project else set()
        rows = []
        if project:
            for c in project.classes:
                rows.append({"id": c.id, "zh": c.names.get("zh_TW", ""), "en": c.names.get("en", ""),
                             "color": c.color, "hotkey": c.hotkey, "description": c.description})
        self._fill(rows)
        self.setEnabled(project is not None)

    def _fill(self, rows: list[dict[str, Any]]) -> None:
        self._busy = True
        try:
            self.table.setRowCount(0)
            for r in rows:
                self._append_row(r)
        finally:
            self._busy = False

    def _append_row(self, r: dict[str, Any]) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        id_item = QTableWidgetItem(r["id"])
        existing = r["id"] in self._existing
        if existing:
            id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            id_item.setToolTip(tr("gui.project.id_locked"))
        id_item.setData(Qt.ItemDataRole.UserRole, {"auto": not existing and r.get("auto", False),
                                                   "description": r.get("description", "")})
        self.table.setItem(row, COL_ID, id_item)
        self.table.setItem(row, COL_ZH, QTableWidgetItem(r.get("zh", "")))
        self.table.setItem(row, COL_EN, QTableWidgetItem(r.get("en", "")))
        btn = QPushButton()
        self._style_color(btn, r["color"])
        btn.clicked.connect(lambda _=False, b=btn: self._pick_color(b))
        self.table.setCellWidget(row, COL_COLOR, btn)
        combo = QComboBox()
        combo.addItem(tr("gui.project.hotkey_none"), "")
        for k in range(1, 10):
            combo.addItem(str(k), str(k))
        idx = combo.findData(r.get("hotkey", ""))
        combo.setCurrentIndex(max(idx, 0))
        self.table.setCellWidget(row, COL_KEY, combo)

    @staticmethod
    def _style_color(btn: QPushButton, color: str) -> None:
        btn.setProperty("color", color)
        c = QColor(color)
        btn.setText(color)
        btn.setStyleSheet(f"background:{color};color:{'#fff' if c.lightness() < 128 else '#000'};")

    def _pick_color(self, btn: QPushButton) -> None:
        c = QColorDialog.getColor(QColor(btn.property("color")), self)
        if c.isValid():
            self._style_color(btn, c.name().upper())

    def set_row_color(self, row: int, color: str) -> None:
        self._style_color(self.table.cellWidget(row, COL_COLOR), color)

    def _read_rows(self) -> list[dict[str, Any]]:
        out = []
        for r in range(self.table.rowCount()):
            id_item = self.table.item(r, COL_ID)
            meta = id_item.data(Qt.ItemDataRole.UserRole) or {}
            out.append({
                "id": id_item.text().strip(), "zh": self.table.item(r, COL_ZH).text().strip(),
                "en": self.table.item(r, COL_EN).text().strip(),
                "color": self.table.cellWidget(r, COL_COLOR).property("color"),
                "hotkey": self.table.cellWidget(r, COL_KEY).currentData() or "",
                "auto": meta.get("auto", False), "description": meta.get("description", ""),
            })
        return out

    # ---- 操作 -------------------------------------------------------------------
    def add_class(self, zh: str = "", en: str = "") -> int:
        rows = self._read_rows()
        used_keys = {r["hotkey"] for r in rows}
        n = len(rows)
        row = {"id": "", "zh": zh, "en": en, "color": DEFAULT_COLORS[n % len(DEFAULT_COLORS)],
               "hotkey": next((str(k) for k in range(1, 10) if str(k) not in used_keys), ""), "auto": True}
        row["id"] = self._unique_id(zh, en, [r["id"] for r in rows])
        self._busy = True
        try:
            self._append_row(row)
        finally:
            self._busy = False
        self.table.selectRow(self.table.rowCount() - 1)
        return self.table.rowCount() - 1

    @staticmethod
    def _unique_id(zh: str, en: str, used: list[str]) -> str:
        base = slugify(en or zh, fallback=f"class_{len(used) + 1}")
        cid, i = base, 2
        while cid in used:
            cid = f"{base}_{i}"
            i += 1
        return cid

    def _cell_changed(self, row: int, col: int) -> None:
        if self._busy or col not in (COL_ID, COL_ZH, COL_EN):
            return
        id_item = self.table.item(row, COL_ID)
        meta = dict(id_item.data(Qt.ItemDataRole.UserRole) or {})
        self._busy = True
        try:
            if col == COL_ID:
                meta["auto"] = False  # 使用者自己改過 ID，之後不再自動產生
                id_item.setData(Qt.ItemDataRole.UserRole, meta)
            elif meta.get("auto"):
                used = [self.table.item(r, COL_ID).text() for r in range(self.table.rowCount()) if r != row]
                id_item.setText(self._unique_id(self.table.item(row, COL_ZH).text(),
                                                self.table.item(row, COL_EN).text(), used))
        finally:
            self._busy = False

    def _ask_confirm_delete(self, names: str) -> bool:
        return QMessageBox.question(
            self, tr("gui.project.delete_title"), tr("gui.project.delete_text", name=names),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes

    def remove_selected(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        name = self.table.item(row, COL_ZH).text() or self.table.item(row, COL_EN).text() \
            or self.table.item(row, COL_ID).text()
        if self.confirm_delete(name):
            self.table.removeRow(row)

    def move_selected(self, delta: int) -> None:
        row = self.table.currentRow()
        new = row + delta
        if row < 0 or not 0 <= new < self.table.rowCount():
            return
        rows = self._read_rows()
        rows[row], rows[new] = rows[new], rows[row]
        self._fill(rows)
        self.table.selectRow(new)

    def collect(self) -> list[ClassSpec]:
        """讀出目前表格；資料不合格時丟 ValueError（訊息給使用者看）。"""
        specs: list[ClassSpec] = []
        seen: set[str] = set()
        keys: set[str] = set()
        for i, r in enumerate(self._read_rows(), 1):
            if not r["zh"] and not r["en"]:
                raise ValueError(tr("gui.project.err_no_name", row=i))
            if not is_valid_id(r["id"]):
                raise ValueError(tr("gui.project.err_bad_id", id=r["id"] or "(空)"))
            if r["id"] in seen:
                raise ValueError(tr("gui.project.err_dup_id", id=r["id"]))
            if r["hotkey"] and r["hotkey"] in keys:
                raise ValueError(tr("gui.project.err_dup_key", key=r["hotkey"]))
            seen.add(r["id"])
            if r["hotkey"]:
                keys.add(r["hotkey"])
            names = {k: v for k, v in (("zh_TW", r["zh"]), ("en", r["en"])) if v}
            specs.append(ClassSpec(id=r["id"], names=names, color=r["color"], hotkey=r["hotkey"],
                                   description=r["description"]))
        return specs

    def apply(self) -> None:
        """把表格內容寫回專案並存檔。"""
        if self.project is None:
            return
        self.project.classes = self.collect()
        self.project.save()
        self._existing = {c.id for c in self.project.classes}

    def retranslate(self) -> None:
        self.note.setText(tr("gui.project.class_note"))


class ProjectDialog(QDialog):
    TAB_CHOOSE, TAB_NEW, TAB_CLASSES = range(3)

    def __init__(self, ctx: AppContext, parent: QWidget | None = None, tab: int = 0):
        super().__init__(parent)
        self.ctx = ctx
        self._project: Project | None = ctx.project
        self._classes_dirty_loaded = False
        self.setWindowTitle(tr("gui.project.title"))
        self.resize(720, 520)
        root = QVBoxLayout(self)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        # 選擇
        choose = QWidget()
        cl = QVBoxLayout(choose)
        self.project_list = QListWidget()
        self.project_list.currentItemChanged.connect(self._list_selected)
        self.project_list.itemDoubleClicked.connect(lambda _i: self.accept())
        cl.addWidget(self.project_list)
        self.tabs.addTab(choose, tr("gui.project.tab.choose"))

        # 新增
        new = QWidget()
        nl = QFormLayout(new)
        self.name_edit = QLineEdit()
        self.template_combo = QComboBox()
        lang = i18n.current_language()
        for key, tpl in project_templates().items():
            self.template_combo.addItem(template_title(key, tpl, lang), key)
        idx = self.template_combo.findData("blank")
        if idx >= 0:
            self.template_combo.setCurrentIndex(idx)
        self.create_button = QPushButton(tr("gui.project.create"))
        self.create_button.clicked.connect(self._create_clicked)
        nl.addRow(tr("gui.project.name"), self.name_edit)
        nl.addRow(tr("gui.project.template"), self.template_combo)
        nl.addRow(self.create_button)
        self.tabs.addTab(new, tr("gui.project.tab.new"))

        # 類別
        self.class_editor = ClassEditor()
        self.tabs.addTab(self.class_editor, tr("gui.project.tab.classes"))
        self.tabs.currentChanged.connect(self._tab_changed)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr("common.ok"))
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr("common.cancel"))
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

        self._refresh_list()
        self.tabs.setCurrentIndex(tab)
        if tab == self.TAB_CLASSES:
            self._tab_changed(tab)

    def _refresh_list(self, select_id: str = "") -> None:
        self.project_list.blockSignals(True)
        self.project_list.clear()
        want = select_id or (self._project.id if self._project else "")
        sel = None
        for p in self.ctx.workspace.projects():
            item = QListWidgetItem(f"{p.name}   ({p.id})")
            item.setData(Qt.ItemDataRole.UserRole, p)
            self.project_list.addItem(item)
            if p.id == want:
                sel = item
        self.project_list.blockSignals(False)
        if sel is not None:
            self.project_list.setCurrentItem(sel)
            self._project = sel.data(Qt.ItemDataRole.UserRole)

    def _list_selected(self, item: QListWidgetItem | None, _prev=None) -> None:
        if item is not None:
            self._project = item.data(Qt.ItemDataRole.UserRole)
            if self.tabs.currentIndex() == self.TAB_CLASSES:
                self.class_editor.set_project(self._project)

    def _tab_changed(self, index: int) -> None:
        if index == self.TAB_CLASSES:
            self.class_editor.set_project(self._project)

    def create_project(self, name: str, template: str | None = None) -> Project:
        """建立專案並選取它（對話框按鈕與測試共用）。"""
        project = self.ctx.workspace.create_project(name.strip(), template or None, lang=i18n.current_language())
        self._project = project
        self._refresh_list(project.id)
        return project

    def _create_clicked(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.information(self, tr("gui.project.title"), tr("gui.project.err_empty_name"))
            return
        self.create_project(name, self.template_combo.currentData())
        self.name_edit.clear()
        self.tabs.setCurrentIndex(self.TAB_CLASSES)

    def selected_project(self) -> Project | None:
        return self._project

    def accept(self) -> None:
        if self.class_editor.project is not None and self.class_editor.project is self._project:
            try:
                self.class_editor.apply()
            except ValueError as e:
                QMessageBox.warning(self, tr("gui.project.title"), str(e))
                self.tabs.setCurrentIndex(self.TAB_CLASSES)
                return
        super().accept()
