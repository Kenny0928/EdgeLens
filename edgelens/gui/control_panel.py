"""ControlPanel：由 ControlSpec 清單自動產生的參數面板。

相機參數、處理步驟參數、模型門檻都用它，所以不含任何相機專屬的邏輯。
- INT／FLOAT：滑桿 + 數值框（拖曳時連續送值；數值框輸入完成才送）
- BOOL：勾選框、ENUM：下拉選單、STR：文字框、COLOR：色塊按鈕、ACTION：按鈕
- 依 spec.group 分組；advanced=True 的收在可展開的「進階」區。
"""

from __future__ import annotations

import math
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from edgelens.core.controls import ControlSpec, ControlType, defaults_of
from edgelens.gui.theme import status_color
from edgelens.i18n import tr, tr_or

MAX_SLIDER_STEPS = 10000


def spec_label(spec: ControlSpec) -> str:
    return tr_or(spec.label_key, spec.label or spec.key)


class _Row:
    """一個參數的編輯元件與取值／設值方法。"""

    def __init__(self, panel: ControlPanel, spec: ControlSpec):
        self.panel = panel
        self.spec = spec
        self.widget: QWidget = QWidget()
        self.editor: QWidget = self.widget
        self.status = QLabel()
        self.status.setFixedWidth(22)
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label = QLabel(spec_label(spec))
        self.label.setWordWrap(True)
        self._build()
        tip = tr_or(spec.help_key, "") if spec.help_key else ""
        for w in (self.label, self.widget):
            w.setToolTip(tip or spec.key)

    # -- 建立 --
    def _emit(self, value: Any) -> None:
        if not self.panel._updating:
            self.panel.valueChanged.emit(self.spec.key, value)

    def _build(self) -> None:
        spec, t = self.spec, self.spec.type
        if t in (ControlType.INT, ControlType.FLOAT):
            self._build_number()
        elif t == ControlType.BOOL:
            self.editor = self.widget = QCheckBox()
            self.widget.toggled.connect(lambda v: self._emit(bool(v)))
        elif t == ControlType.ENUM:
            box = QComboBox()
            for value, text in spec.options:
                box.addItem(tr_or(text, text), value)
            box.currentIndexChanged.connect(lambda i: self._emit(box.itemData(i)))
            self.editor = self.widget = box
        elif t == ControlType.STR:
            edit = QLineEdit()
            self._last_text = ""
            edit.editingFinished.connect(lambda: self._text_done(edit))
            if spec.extra.get("secret"):
                edit.setEchoMode(QLineEdit.EchoMode.Password)
            self.editor = self.widget = edit
            browse = spec.extra.get("browse")
            if browse:
                # extra["browse"]："file"、"folder" 或 "file_or_folder"：旁邊加「瀏覽…」按鈕
                box = QWidget()
                row = QHBoxLayout(box)
                row.setContentsMargins(0, 0, 0, 0)
                row.addWidget(edit, 1)
                btn = QPushButton(tr("common.browse"))
                btn.clicked.connect(lambda: self._browse(edit, str(browse), btn))
                row.addWidget(btn)
                self.widget = box
        elif t == ControlType.COLOR:
            btn = QPushButton()
            btn.setMinimumWidth(80)
            btn.clicked.connect(self._pick_color)
            self._color = "#000000"
            self.editor = self.widget = btn
        else:  # ACTION
            btn = QPushButton(spec_label(spec))
            btn.clicked.connect(lambda: self.panel.actionTriggered.emit(spec.key))
            self.editor = self.widget = btn

    def _browse(self, edit: QLineEdit, mode: str, anchor: QWidget) -> None:
        start = edit.text() or ""
        if mode == "file_or_folder":
            menu = QMenu(anchor)
            act_file = menu.addAction(tr("gui.control.browse_file"))
            act_dir = menu.addAction(tr("gui.control.browse_folder"))
            chosen = menu.exec(anchor.mapToGlobal(anchor.rect().bottomLeft()))
            if chosen not in (act_file, act_dir):
                return
            mode = "file" if chosen is act_file else "folder"
        if mode == "folder":
            path = QFileDialog.getExistingDirectory(self.panel, tr("gui.control.browse_folder"), start)
        else:
            path, _ = QFileDialog.getOpenFileName(self.panel, tr("gui.control.browse_file"), start)
        if path:
            edit.setText(path)
            self._text_done(edit)

    def _text_done(self, edit: QLineEdit) -> None:
        if edit.text() != self._last_text:
            self._last_text = edit.text()
            self._emit(edit.text())

    def _build_number(self) -> None:
        spec = self.spec
        is_int = spec.type == ControlType.INT
        lo = spec.min if spec.min is not None else (-1_000_000 if is_int else -1e6)
        hi = spec.max if spec.max is not None else (1_000_000 if is_int else 1e6)
        ranged = spec.min is not None and spec.max is not None
        step = spec.step or (1 if is_int else (max((hi - lo) / 100.0, 0.001) if ranged else 0.1))
        self._step = float(step)
        self._lo = float(lo)
        if is_int:
            spin: QSpinBox | QDoubleSpinBox = QSpinBox()
            spin.setRange(int(lo), int(hi))
            spin.setSingleStep(max(1, int(round(step))))
        else:
            spin = QDoubleSpinBox()
            spin.setDecimals(1 if step >= 1 else min(6, max(1, math.ceil(-math.log10(step)))))
            spin.setRange(float(lo), float(hi))
            spin.setSingleStep(float(step))
        spin.setKeyboardTracking(False)  # 輸入完成（Enter／離開）才送值
        if spec.unit:
            spin.setSuffix(" " + spec.unit)
        self.spin = spin
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        self.slider: QSlider | None = None
        if spec.min is not None and spec.max is not None and spec.max > spec.min:
            n = min(MAX_SLIDER_STEPS, max(1, int(round((hi - lo) / step))))
            self._n = n
            self._per = (hi - lo) / n
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(0, n)
            sl.valueChanged.connect(self._slider_changed)
            self.slider = sl
            lay.addWidget(sl, 1)
        lay.addWidget(spin)
        spin.valueChanged.connect(self._spin_changed)
        self.widget = box
        self.editor = spin

    def _slider_to_value(self, pos: int) -> float:
        return self._lo + pos * self._per

    def _value_to_slider(self, v: float) -> int:
        return int(round((v - self._lo) / self._per)) if self._per else 0

    def _cast(self, v: float) -> Any:
        return int(round(v)) if self.spec.type == ControlType.INT else float(v)

    def _slider_changed(self, pos: int) -> None:
        v = self._cast(self._slider_to_value(pos))
        if self.panel._updating:
            return
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self._emit(v)

    def _spin_changed(self, v: float) -> None:
        if self.slider is not None:
            self.slider.blockSignals(True)
            self.slider.setValue(self._value_to_slider(float(v)))
            self.slider.blockSignals(False)
        self._emit(self._cast(v))

    def _pick_color(self) -> None:
        c = QColorDialog.getColor(QColor(self._color), self.widget)
        if c.isValid():
            self.user_set_color(c.name().upper())

    def user_set_color(self, value: str) -> None:
        self.set_value(value)
        self._emit(self._color)

    # -- 取值／設值（不 emit）--
    def set_value(self, value: Any) -> None:
        t = self.spec.type
        if value is None or t == ControlType.ACTION:
            return
        panel_flag = self.panel._updating
        self.panel._updating = True
        try:
            if t in (ControlType.INT, ControlType.FLOAT):
                self.spin.setValue(float(value) if t == ControlType.FLOAT else int(round(float(value))))
                if self.slider is not None:
                    self.slider.setValue(self._value_to_slider(float(value)))
            elif t == ControlType.BOOL:
                self.widget.setChecked(bool(value))  # type: ignore[attr-defined]
            elif t == ControlType.ENUM:
                box: QComboBox = self.widget  # type: ignore[assignment]
                for i in range(box.count()):
                    if box.itemData(i) == value or str(box.itemData(i)) == str(value):
                        box.setCurrentIndex(i)
                        break
            elif t == ControlType.STR:
                self.editor.setText(str(value))  # type: ignore[attr-defined]
                self._last_text = str(value)
            elif t == ControlType.COLOR:
                self._color = str(value).upper()
                c = QColor(self._color)
                dark = c.lightness() < 128
                self.widget.setText(self._color)  # type: ignore[attr-defined]
                self.widget.setStyleSheet(f"background:{self._color};color:{'#fff' if dark else '#000'};")
        finally:
            self.panel._updating = panel_flag

    def value(self) -> Any:
        t = self.spec.type
        if t in (ControlType.INT, ControlType.FLOAT):
            return self._cast(self.spin.value())
        if t == ControlType.BOOL:
            return self.widget.isChecked()  # type: ignore[attr-defined]
        if t == ControlType.ENUM:
            return self.widget.currentData()  # type: ignore[attr-defined]
        if t == ControlType.STR:
            return self.editor.text()  # type: ignore[attr-defined]
        if t == ControlType.COLOR:
            return self._color
        return None

    def set_status(self, verified: bool | None, message: str, shown: bool) -> None:
        if not shown:
            self.status.setText("")
            self.status.setToolTip("")
            return
        if verified is True:
            text, kind, tip = "✓", "ok", message or tr("gui.control.status.verified")
        elif verified is False:
            text, kind, tip = "✗", "bad", message or tr("gui.control.status.rejected")
        else:
            text, kind, tip = "?", "muted", message or tr("gui.control.status.unverified")
        self.status.setText(text)
        self.status.setToolTip(tip)
        self.status.setStyleSheet(f"color:{status_color(kind).name()};font-weight:bold;")


class ControlPanel(QWidget):
    valueChanged = Signal(str, object)
    actionTriggered = Signal(str)
    advancedToggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None, *, show_status: bool = True, advanced_expanded: bool = False):
        super().__init__(parent)
        self._specs: list[ControlSpec] = []
        self._rows: dict[str, _Row] = {}
        self._statuses: dict[str, tuple[bool | None, str]] = {}
        self._updating = False
        self._show_status = show_status
        self._advanced_expanded = advanced_expanded
        self._adv_container: QWidget | None = None
        self._adv_button: QToolButton | None = None
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)

    # ---- 建立 -------------------------------------------------------------------
    def set_specs(self, specs: list[ControlSpec], values: dict[str, Any] | None = None) -> None:
        self._specs = list(specs)
        self._clear()
        normal = [s for s in self._specs if not s.advanced]
        adv = [s for s in self._specs if s.advanced]
        self._build_groups(self._root, normal, in_advanced=False)
        if adv:
            btn = QToolButton()
            btn.setCheckable(True)
            btn.setChecked(self._advanced_expanded)
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            btn.setArrowType(Qt.ArrowType.DownArrow if self._advanced_expanded else Qt.ArrowType.RightArrow)
            btn.setText(tr("common.advanced"))
            btn.setAutoRaise(True)
            container = QWidget()
            lay = QVBoxLayout(container)
            lay.setContentsMargins(0, 0, 0, 0)
            self._build_groups(lay, adv, in_advanced=True)
            btn.toggled.connect(self.set_advanced_expanded)
            self._adv_button, self._adv_container = btn, container
            self._root.addWidget(btn)
            self._root.addWidget(container)
            container.setVisible(self._advanced_expanded)  # 加入版面之後才設，否則會變成獨立視窗
        self._root.addStretch(1)
        self.set_values({**defaults_of(self._specs), **(values or {})})
        for key, (ok, msg) in self._statuses.items():
            if key in self._rows:
                self._rows[key].set_status(ok, msg, self._show_status)

    def _clear(self) -> None:
        self._rows.clear()
        self._adv_container = self._adv_button = None
        while self._root.count():
            item = self._root.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _build_groups(self, layout: QVBoxLayout, specs: list[ControlSpec], in_advanced: bool) -> None:
        order: list[str] = []
        for s in specs:
            if s.group not in order:
                order.append(s.group)
        for g in order:
            members = [s for s in specs if s.group == g]
            if in_advanced and g == "advanced":
                holder: QWidget = QWidget()
            else:
                holder = QGroupBox(tr_or(f"gui.group.{g}", g))
            form = QFormLayout(holder)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            for spec in members:
                row = _Row(self, spec)
                self._rows[spec.key] = row
                if spec.type == ControlType.ACTION:
                    form.addRow(row.widget)
                else:
                    cell = QWidget()
                    h = QHBoxLayout(cell)
                    h.setContentsMargins(0, 0, 0, 0)
                    h.addWidget(row.widget, 1)
                    if self._show_status:
                        h.addWidget(row.status)
                    form.addRow(row.label, cell)
            layout.addWidget(holder)

    # ---- 值與狀態 ---------------------------------------------------------------
    def set_values(self, values: dict[str, Any]) -> None:
        """更新顯示的值，不會 emit valueChanged。"""
        self._updating = True
        try:
            for k, v in values.items():
                row = self._rows.get(k)
                if row is not None:
                    row.set_value(v)
        finally:
            self._updating = False

    def values(self) -> dict[str, Any]:
        return {k: r.value() for k, r in self._rows.items() if r.spec.type != ControlType.ACTION}

    def set_status(self, key: str, verified: bool | None, message: str = "") -> None:
        self._statuses[key] = (verified, message)
        row = self._rows.get(key)
        if row is not None:
            row.set_status(verified, message, self._show_status)

    def clear_status(self) -> None:
        self._statuses.clear()
        for r in self._rows.values():
            r.set_status(None, "", False)

    def set_enabled_key(self, key: str, enabled: bool) -> None:
        row = self._rows.get(key)
        if row is not None:
            row.widget.setEnabled(enabled)

    def set_all_enabled(self, enabled: bool, *, actions_only: bool = False) -> None:
        for r in self._rows.values():
            if not actions_only or r.spec.type == ControlType.ACTION:
                r.widget.setEnabled(enabled)

    def row(self, key: str) -> _Row | None:
        return self._rows.get(key)

    def keys(self) -> list[str]:
        return list(self._rows)

    # ---- 進階區 -----------------------------------------------------------------
    @property
    def advanced_expanded(self) -> bool:
        return self._advanced_expanded

    def has_advanced(self) -> bool:
        return self._adv_container is not None

    def set_advanced_expanded(self, expanded: bool) -> None:
        changed = expanded != self._advanced_expanded
        self._advanced_expanded = expanded
        if self._adv_container is not None:
            self._adv_container.setVisible(expanded)
        if self._adv_button is not None:
            self._adv_button.blockSignals(True)
            self._adv_button.setChecked(expanded)
            self._adv_button.blockSignals(False)
            self._adv_button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        if changed:
            self.advancedToggled.emit(expanded)

    def advanced_visible(self) -> bool:
        return self._adv_container is not None and not self._adv_container.isHidden()

    # ---- 語言 -------------------------------------------------------------------
    def retranslate(self) -> None:
        """換語言：用目前的值重建（標籤、群組名稱、下拉選項都會重新翻譯）。"""
        self.set_specs(self._specs, self.values())
