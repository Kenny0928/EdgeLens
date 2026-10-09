"""影像處理面板：編輯作用中相機的處理流程，並用吸管取色產生 HSV 遮罩。

為什麼每次修改都送一份新的 dict 給 worker：Pipeline 只能在 worker 執行緒使用，
GUI 絕不直接改它；換掉整條流程時 stateful 步驟會 reset，這是預期行為。
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from edgelens.core.controls import specs_by_key
from edgelens.core.registry import discover
from edgelens.gui.control_panel import ControlPanel
from edgelens.gui.panels.base import Panel
from edgelens.gui.workers import FramePacket, SourceHandle
from edgelens.i18n import current_language, tr, tr_or
from edgelens.processing.colorpick import (
    ColorSample,
    pick,
    ranges_from_samples,
    to_hsv_mask_params,
    to_maixpy_lab_threshold,
    to_opencv_hsv,
)
from edgelens.processing.library import PresetInfo, list_presets, load_preset, save_user_preset
from edgelens.processing.pipeline import CATEGORIES, PIPELINE_VERSION, STEPS, Pipeline

log = logging.getLogger(__name__)

SLOW_MS = 30.0           # 超過這個總處理時間就提醒
TIMING_INTERVAL = 0.25   # 耗時顯示的更新間隔（秒）
HSV_KIND = "color.hsv_mask"


def maixpy_lab_text(ranges) -> str:
    """K210（MaixPy v1）find_blobs 用的 LAB 門檻，可直接貼進程式。"""
    return str(tuple(to_maixpy_lab_threshold(ranges)))


def opencv_hsv_text(ranges) -> str:
    """OpenCV 的 HSV 範圍程式碼；紅色跨 0 時是兩段。"""
    parts = to_opencv_hsv(ranges)
    lines = []
    for i, (lo, hi) in enumerate(parts, 1):
        suffix = "" if len(parts) == 1 else str(i)
        lines.append(f"lower{suffix} = np.array({list(lo)})")
        lines.append(f"upper{suffix} = np.array({list(hi)})")
    if len(parts) == 1:
        lines.append("mask = cv2.inRange(hsv, lower, upper)")
    else:
        lines.append("mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))")
    return "\n".join(lines)


class PipelinePanel(Panel):
    panel_id = "pipeline"
    title_key = "gui.panel.pipeline.title"
    order = 20

    def __init__(self, ctx, main):
        super().__init__(ctx, main)
        discover()
        self._handle: SourceHandle | None = None
        self._steps: list[dict[str, Any]] = []
        self._updating = False
        self._last_timing = 0.0
        self._presets: list[PresetInfo] = []
        self._samples: list[ColorSample] = []
        self._pick_view = None
        self._connected_views: set[int] = set()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        lay = QVBoxLayout(body)

        self.empty_label = QLabel()
        self.empty_label.setWordWrap(True)
        lay.addWidget(self.empty_label)

        self.main_box = QWidget()
        ml = QVBoxLayout(self.main_box)
        ml.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.main_box)

        self.show_processed = QCheckBox()
        self.show_processed.setChecked(True)
        self.show_processed.toggled.connect(self._on_show_processed)
        ml.addWidget(self.show_processed)
        try:
            main.act_processed.toggled.connect(self._sync_show_processed)
        except AttributeError:
            pass

        row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.preset_combo.setMinimumContentsLength(10)
        self.preset_apply = QPushButton()
        self.preset_apply.clicked.connect(self._apply_selected_preset)
        self.preset_save = QPushButton()
        self.preset_save.clicked.connect(self._ask_save_preset)
        self.preset_label = QLabel()
        ml.addWidget(self.preset_label)
        ml.addWidget(self.preset_combo)
        row.addWidget(self.preset_apply)
        row.addWidget(self.preset_save)
        ml.addLayout(row)

        self.steps_label = QLabel()
        ml.addWidget(self.steps_label)
        self.list = QListWidget()
        self.list.setMinimumHeight(110)
        self.list.setMaximumHeight(170)
        self.list.currentRowChanged.connect(self._on_row_changed)
        self.list.itemChanged.connect(self._on_item_changed)
        ml.addWidget(self.list)

        tools = QHBoxLayout()
        self.add_button = QToolButton()
        self.add_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.add_menu = QMenu(self.add_button)
        self.add_button.setMenu(self.add_menu)
        self.up_button = QPushButton("▲")
        self.down_button = QPushButton("▼")
        self.dup_button = QPushButton()
        self.del_button = QPushButton()
        self.clear_button = QPushButton()
        self.up_button.clicked.connect(lambda: self.move_step(-1))
        self.down_button.clicked.connect(lambda: self.move_step(1))
        self.dup_button.clicked.connect(self.duplicate_step)
        self.del_button.clicked.connect(self.remove_step)
        self.clear_button.clicked.connect(self.clear_steps)
        tools2 = QHBoxLayout()
        for w in (self.add_button, self.up_button, self.down_button):
            tools.addWidget(w)
        tools.addStretch(1)
        for w in (self.dup_button, self.del_button, self.clear_button):
            tools2.addWidget(w)
        self.up_button.setFixedWidth(34)
        self.down_button.setFixedWidth(34)
        ml.addLayout(tools)
        ml.addLayout(tools2)

        self.time_label = QLabel()
        self.time_label.setWordWrap(True)
        self.slow_label = QLabel()
        self.slow_label.setWordWrap(True)
        self.slow_label.setStyleSheet("color:#C77700;font-weight:bold;")
        self.slow_label.hide()
        ml.addWidget(self.time_label)
        ml.addWidget(self.slow_label)

        self.params_box = QGroupBox()
        pl = QVBoxLayout(self.params_box)
        self.params_hint = QLabel()
        self.params_hint.setWordWrap(True)
        self.controls = ControlPanel(show_status=False,
                                     advanced_expanded=bool(ctx.settings.get("ui.show_advanced", False)))
        self.controls.valueChanged.connect(self._on_param_changed)
        pl.addWidget(self.params_hint)
        pl.addWidget(self.controls)
        ml.addWidget(self.params_box)

        self._build_pick_box(ml)
        ml.addStretch(1)
        self.retranslate()
        self.on_project_changed(ctx.project)
        self.on_active_source_changed(ctx.active_source)

    # ---- 吸管取色的介面 ---------------------------------------------------------
    def _build_pick_box(self, parent: QVBoxLayout) -> None:
        self.pick_box = QGroupBox()
        v = QVBoxLayout(self.pick_box)
        self.pick_hint = QLabel()
        self.pick_hint.setWordWrap(True)
        v.addWidget(self.pick_hint)
        row = QHBoxLayout()
        self.pick_button = QPushButton()
        self.pick_button.setCheckable(True)
        self.pick_button.toggled.connect(self._on_pick_toggled)
        self.pick_clear = QPushButton()
        self.pick_clear.clicked.connect(self.clear_samples)
        row.addWidget(self.pick_button)
        row.addWidget(self.pick_clear)
        v.addLayout(row)
        opts = QHBoxLayout()
        self.radius_label = QLabel()
        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(0, 20)
        self.radius_spin.setValue(int(ctx_get(self.ctx, "pipeline.pick_radius", 5)))
        self.margin_label = QLabel()
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 60)
        self.margin_spin.setValue(int(ctx_get(self.ctx, "pipeline.pick_margin", 5)))
        self.radius_spin.valueChanged.connect(lambda x: self.ctx.settings.set("pipeline.pick_radius", int(x)))
        self.margin_spin.valueChanged.connect(self._on_margin_changed)
        for w in (self.radius_label, self.radius_spin, self.margin_label, self.margin_spin):
            opts.addWidget(w)
        opts.addStretch(1)
        v.addLayout(opts)
        sw = QHBoxLayout()
        self.swatch = QLabel()
        self.swatch.setFixedSize(48, 48)
        self.swatch.setFrameShape(QFrame.Shape.Box)
        self.swatch.setStyleSheet("background:#808080;")
        self.sample_info = QLabel()
        self.sample_info.setWordWrap(True)
        self.sample_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        sw.addWidget(self.swatch)
        sw.addWidget(self.sample_info, 1)
        v.addLayout(sw)
        self.apply_hsv_button = QPushButton()
        self.apply_hsv_button.clicked.connect(self.apply_to_hsv_mask)
        self.copy_lab_button = QPushButton()
        self.copy_lab_button.clicked.connect(self.copy_maixpy_lab)
        self.copy_hsv_button = QPushButton()
        self.copy_hsv_button.clicked.connect(self.copy_opencv_hsv)
        for b in (self.apply_hsv_button, self.copy_lab_button, self.copy_hsv_button):
            v.addWidget(b)
        parent.addWidget(self.pick_box)
        self._update_sample_ui()

    # ---- 切換相機與專案 ---------------------------------------------------------
    def on_active_source_changed(self, handle: SourceHandle | None) -> None:
        self._stop_pick()
        self._handle = handle
        self._samples = []
        self._load_steps()
        self._rebuild_list()
        self._update_sample_ui()
        self._refresh_visibility()

    def on_project_changed(self, project: Any) -> None:
        self._refresh_presets()

    def _refresh_visibility(self) -> None:
        has = self._handle is not None
        self.empty_label.setVisible(not has)
        self.main_box.setVisible(has)

    def _load_steps(self) -> None:
        d = self._handle.pipeline_dict if self._handle is not None else {}
        self._steps = []
        for s in (d or {}).get("steps", []):
            if isinstance(s, dict) and isinstance(s.get("kind"), str):
                self._steps.append({"kind": s["kind"], "enabled": bool(s.get("enabled", True)),
                                    "params": dict(s.get("params", {}))})

    # ---- 把目前清單送給 worker --------------------------------------------------
    def current_dict(self) -> dict[str, Any]:
        return {"version": PIPELINE_VERSION, "steps": copy.deepcopy(self._steps)}

    def _commit(self) -> None:
        if self._handle is None:
            return
        self._handle.set_pipeline(self.current_dict())
        self.ctx.save_settings_later()

    # ---- 步驟清單 ---------------------------------------------------------------
    @staticmethod
    def step_name(kind: str) -> str:
        cls = STEPS.find(kind)
        return tr_or(cls.display_name_key, kind) if cls is not None and cls.display_name_key else kind

    @staticmethod
    def is_plugin(kind: str) -> bool:
        return STEPS.origin(kind).startswith("edgelens_plugins.")

    def _item_text(self, index: int, timing: float | None, error: str | None) -> str:
        step = self._steps[index]
        text = f"{index + 1}. {self.step_name(step['kind'])}"
        if self.is_plugin(step["kind"]):
            text += f" [{tr('gui.pipeline.plugin')}]"
        if step["enabled"] and timing is not None:
            text += f"   {timing:.1f} ms"
        if error:
            text += "   ⚠"
        return text

    def _rebuild_list(self, select: int | None = None) -> None:
        if select is None:
            select = self.list.currentRow()
        self._updating = True
        self.list.clear()
        for i, step in enumerate(self._steps):
            item = QListWidgetItem(self._item_text(i, None, None))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if step["enabled"] else Qt.CheckState.Unchecked)
            cls = STEPS.find(step["kind"])
            item.setToolTip(tr_or(cls.description_key, "") if cls is not None else tr("gui.pipeline.unknown_step"))
            self.list.addItem(item)
        self._updating = False
        if self._steps:
            row = min(max(select, 0), len(self._steps) - 1)
            self.list.setCurrentRow(row)
        self._on_row_changed(self.list.currentRow())
        self._update_buttons()
        self._update_timing_labels(None)

    def _update_buttons(self) -> None:
        row = self.list.currentRow()
        n = len(self._steps)
        self.up_button.setEnabled(row > 0)
        self.down_button.setEnabled(0 <= row < n - 1)
        self.dup_button.setEnabled(row >= 0)
        self.del_button.setEnabled(row >= 0)
        self.clear_button.setEnabled(n > 0)

    def selected_index(self) -> int:
        return self.list.currentRow()

    def select(self, index: int) -> None:
        self.list.setCurrentRow(index)

    def add_step(self, kind: str, params: dict[str, Any] | None = None) -> int:
        """新增一個步驟到清單最後面，回傳它的位置。參數用步驟自己的預設值補齊。"""
        cls = STEPS.get(kind)
        step = cls(enabled=True, **(params or {}))
        self._steps.append(step.to_dict())
        self._rebuild_list(select=len(self._steps) - 1)
        self._commit()
        return len(self._steps) - 1

    def remove_step(self) -> None:
        row = self.list.currentRow()
        if 0 <= row < len(self._steps):
            del self._steps[row]
            self._rebuild_list(select=row)
            self._commit()

    def move_step(self, delta: int) -> None:
        row = self.list.currentRow()
        new = row + delta
        if 0 <= row < len(self._steps) and 0 <= new < len(self._steps):
            self._steps[row], self._steps[new] = self._steps[new], self._steps[row]
            self._rebuild_list(select=new)
            self._commit()

    def duplicate_step(self) -> None:
        row = self.list.currentRow()
        if 0 <= row < len(self._steps):
            self._steps.insert(row + 1, copy.deepcopy(self._steps[row]))
            self._rebuild_list(select=row + 1)
            self._commit()

    def clear_steps(self) -> None:
        self._steps = []
        self._rebuild_list()
        self._commit()

    def set_enabled(self, index: int, enabled: bool) -> None:
        if 0 <= index < len(self._steps):
            item = self.list.item(index)
            item.setCheckState(Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked)

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._updating:
            return
        row = self.list.row(item)
        if 0 <= row < len(self._steps):
            self._steps[row]["enabled"] = item.checkState() == Qt.CheckState.Checked
            self._commit()

    def _on_row_changed(self, row: int) -> None:
        self._update_buttons()
        if not (0 <= row < len(self._steps)):
            self.controls.set_specs([], {})
            self.params_hint.setText(tr("gui.pipeline.pick_step"))
            self.params_hint.show()
            return
        step = self._steps[row]
        cls = STEPS.find(step["kind"])
        if cls is None:
            self.controls.set_specs([], {})
            self.params_hint.setText(tr("gui.pipeline.unknown_step"))
            self.params_hint.show()
            return
        specs = cls.param_specs()
        self.params_hint.setVisible(not specs)
        self.params_hint.setText(tr("gui.pipeline.no_params"))
        self.controls.set_specs(specs, step["params"])

    def _on_param_changed(self, key: str, value: Any) -> None:
        row = self.list.currentRow()
        if not (0 <= row < len(self._steps)):
            return
        cls = STEPS.find(self._steps[row]["kind"])
        spec = specs_by_key(cls.param_specs()).get(key) if cls is not None else None
        if spec is None:
            return
        try:
            value = spec.validate(value)
        except ValueError as e:
            self.ctx.show_message(str(e))
            return
        self._steps[row]["params"][key] = value
        self._commit()

    def set_param(self, index: int, key: str, value: Any) -> None:
        """程式化修改參數（測試與其他面板用），行為和使用者拖滑桿相同。"""
        self.select(index)
        row = self.controls.row(key)
        if row is not None:
            self.controls.set_values({key: value})
        self._on_param_changed(key, value)

    # ---- 新增步驟選單 -----------------------------------------------------------
    def _rebuild_add_menu(self) -> None:
        self.add_menu.clear()
        groups: dict[str, list[tuple[str, type]]] = {}
        for kind, cls in STEPS.items():
            groups.setdefault(cls.category, []).append((kind, cls))
        order = list(CATEGORIES) + sorted(c for c in groups if c not in CATEGORIES)
        for cat in order:
            entries = groups.get(cat)
            if not entries:
                continue
            sub = self.add_menu.addMenu(tr_or(f"gui.pipeline.category.{cat}", cat))
            for kind, cls in sorted(entries, key=lambda e: e[0]):
                name = tr_or(cls.display_name_key, kind) if cls.display_name_key else kind
                if self.is_plugin(kind):
                    name += f" [{tr('gui.pipeline.plugin')}]"
                act = sub.addAction(name)
                act.setToolTip(tr_or(cls.description_key, "") if cls.description_key else "")
                act.setStatusTip(act.toolTip())
                act.hovered.connect(lambda a=act: self._show_action_tip(a))
                act.triggered.connect(lambda _=False, k=kind: self.add_step(k))

    @staticmethod
    def _show_action_tip(action) -> None:
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QToolTip
        if action.toolTip():
            QToolTip.showText(QCursor.pos(), action.toolTip())

    # ---- 預設組合 ---------------------------------------------------------------
    def _refresh_presets(self) -> None:
        folder = None
        project = self.ctx.project
        if project is not None:
            try:
                folder = project.pipelines_dir
            except (OSError, ValueError):
                folder = None
        try:
            self._presets = list_presets(folder)
        except OSError:
            self._presets = []
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        lang = current_language()
        for info in self._presets:
            tag = tr("gui.pipeline.preset.builtin") if info.builtin else tr("gui.pipeline.preset.project")
            self.preset_combo.addItem(f"{info.title(lang)} ({tag})", info)
            self.preset_combo.setItemData(self.preset_combo.count() - 1, info.description(lang),
                                          Qt.ItemDataRole.ToolTipRole)
        self.preset_combo.blockSignals(False)

    def apply_preset(self, info: PresetInfo) -> bool:
        try:
            pipeline = load_preset(info.path)
        except ValueError as e:
            self.ctx.show_message(str(e))
            return False
        data = pipeline.to_dict()
        self._steps = [{"kind": s["kind"], "enabled": bool(s.get("enabled", True)), "params": dict(s["params"])}
                       for s in data["steps"]]
        self._rebuild_list(select=0)
        self._commit()
        self.ctx.show_message(tr("gui.pipeline.preset.applied", name=info.title(current_language())))
        return True

    def _apply_selected_preset(self) -> None:
        info = self.preset_combo.currentData()
        if info is not None:
            self.apply_preset(info)

    def save_preset(self, title: str):
        """把目前流程存成專案的預設組合；回傳檔案路徑，失敗回傳 None 並顯示原因。"""
        project = self.ctx.project
        if project is None:
            self.ctx.show_message(tr("gui.pipeline.preset.need_project"))
            return None
        title = title.strip()
        if not title:
            return None
        try:
            path = save_user_preset(Pipeline.from_dict(self.current_dict()), project.pipelines_dir, title, title)
        except (ValueError, OSError, KeyError) as e:
            self.ctx.show_message(str(e))
            return None
        self._refresh_presets()
        idx = next((i for i, p in enumerate(self._presets) if p.path == path), -1)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)
        self.ctx.show_message(tr("gui.pipeline.preset.saved", name=title))
        return path

    def _ask_save_preset(self) -> None:
        if self.ctx.project is None:
            self.ctx.show_message(tr("gui.pipeline.preset.need_project"))
            return
        text, ok = QInputDialog.getText(self, tr("gui.pipeline.preset.save"), tr("gui.pipeline.preset.name_prompt"))
        if ok:
            self.save_preset(text)

    # ---- 顯示原始／處理後 -------------------------------------------------------
    def _on_show_processed(self, on: bool) -> None:
        self.main.set_show_processed(bool(on))

    def _sync_show_processed(self, _checked: bool = False) -> None:
        on = bool(self.main.act_processed.isChecked())
        self.show_processed.blockSignals(True)
        self.show_processed.setChecked(on)
        self.show_processed.blockSignals(False)

    # ---- 耗時與錯誤 -------------------------------------------------------------
    def on_frame(self, packet: FramePacket) -> None:
        now = time.monotonic()
        if now - self._last_timing < TIMING_INTERVAL:
            return
        self._last_timing = now
        self._update_timing_labels(packet)

    def _update_timing_labels(self, packet: FramePacket | None) -> None:
        if packet is None:
            self.time_label.setText(tr("gui.pipeline.total_none"))
            self.slow_label.hide()
            return
        res = packet.result
        enabled = [i for i, s in enumerate(self._steps) if s["enabled"]]
        timings = list(res.timings_ms)
        errors = {kind: msg for kind, msg in res.errors}
        in_sync = len(timings) == len(enabled) and all(
            timings[n][0] == self._steps[i]["kind"] for n, i in enumerate(enabled))
        self._updating = True
        for i in range(len(self._steps)):
            item = self.list.item(i)
            if item is None:
                continue
            ms = None
            if in_sync and i in enabled:
                ms = timings[enabled.index(i)][1]
            err = errors.get(self._steps[i]["kind"]) if in_sync else None
            text = self._item_text(i, ms, err)
            if item.text() != text:
                item.setText(text)
            tip = item.toolTip().split("\n\n⚠ ")[0]
            item.setToolTip(f"{tip}\n\n⚠ {err}" if err else tip)
            item.setForeground(QColor("#D13438") if err else self.list.palette().text().color())
        self._updating = False
        total = res.total_ms
        max_fps = 1000.0 / total if total > 0.01 else 0.0
        self.time_label.setText(tr("gui.pipeline.total", ms=f"{total:.1f}", fps=f"{max_fps:.0f}"))
        self.slow_label.setVisible(total > SLOW_MS)
        self.slow_label.setText(tr("gui.pipeline.slow", ms=f"{total:.0f}"))

    # ---- 吸管取色 ---------------------------------------------------------------
    def _on_pick_toggled(self, on: bool) -> None:
        h = self._handle
        view = self.main.video_view_for(h) if h is not None else None
        if on:
            if view is None:
                self.pick_button.blockSignals(True)
                self.pick_button.setChecked(False)
                self.pick_button.blockSignals(False)
                return
            if id(view) not in self._connected_views:
                view.pointPicked.connect(self._on_point_picked)
                self._connected_views.add(id(view))
            view.set_mode("pick")
            self._pick_view = view
        else:
            self._stop_pick(uncheck=False)

    def _stop_pick(self, uncheck: bool = True) -> None:
        view, self._pick_view = self._pick_view, None
        if view is not None:
            try:
                view.set_mode("none")
            except RuntimeError:
                pass
        if uncheck and self.pick_button.isChecked():
            self.pick_button.blockSignals(True)
            self.pick_button.setChecked(False)
            self.pick_button.blockSignals(False)

    def hideEvent(self, event) -> None:  # noqa: N802
        self._stop_pick()
        super().hideEvent(event)

    def _on_point_picked(self, x: float, y: float) -> None:
        h = self._handle
        if h is None or h.last_packet is None:
            return
        img = h.last_packet.frame.image
        if img.ndim == 2:
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        try:
            sample = pick(np.ascontiguousarray(img), int(x), int(y), self.radius_spin.value())
        except ValueError as e:
            self.ctx.show_message(str(e))
            return
        self._samples.append(sample)
        self._update_sample_ui()

    def add_sample_at(self, x: float, y: float) -> None:
        self._on_point_picked(x, y)

    def clear_samples(self) -> None:
        self._samples = []
        self._update_sample_ui()

    def _on_margin_changed(self, value: int) -> None:
        self.ctx.settings.set("pipeline.pick_margin", int(value))
        self._update_sample_ui()

    def ranges(self):
        if not self._samples:
            return None
        return ranges_from_samples(self._samples, self.margin_spin.value())

    def _update_sample_ui(self) -> None:
        has = bool(self._samples)
        for b in (self.apply_hsv_button, self.copy_lab_button, self.copy_hsv_button, self.pick_clear):
            b.setEnabled(has)
        if not has:
            self.swatch.setStyleSheet("background:#808080;")
            self.sample_info.setText(tr("gui.pipeline.pick.none"))
            return
        r = self.ranges()
        b, g, rr = (int(round(float(np.mean([s.bgr[i] for s in self._samples])))) for i in range(3))
        self.swatch.setStyleSheet(f"background:rgb({rr},{g},{b});")
        lo, hi = r.hsv_lower, r.hsv_upper
        lab = to_maixpy_lab_threshold(r)
        self.sample_info.setText(tr("gui.pipeline.pick.info", n=len(self._samples),
                                    hlo=lo[0], hhi=hi[0], slo=lo[1], shi=hi[1], vlo=lo[2], vhi=hi[2],
                                    lab=str(tuple(lab))))

    def apply_to_hsv_mask(self) -> bool:
        """把取到的範圍寫進 HSV 遮罩：選取的是遮罩就更新它，否則更新第一個遮罩，都沒有就新增。"""
        r = self.ranges()
        if r is None:
            return False
        params = to_hsv_mask_params(r)
        row = self.list.currentRow()
        target = None
        if 0 <= row < len(self._steps) and self._steps[row]["kind"] == HSV_KIND:
            target = row
        else:
            target = next((i for i, s in enumerate(self._steps) if s["kind"] == HSV_KIND), None)
        if target is None:
            self.add_step(HSV_KIND, params)
        else:
            self._steps[target]["params"].update(params)
            self._rebuild_list(select=target)
            self._commit()
        self.ctx.show_message(tr("gui.pipeline.pick.applied"))
        return True

    def copy_maixpy_lab(self) -> str | None:
        r = self.ranges()
        if r is None:
            return None
        text = maixpy_lab_text(r)
        QGuiApplication.clipboard().setText(text)
        self.ctx.show_message(tr("gui.pipeline.pick.copied", text=text))
        return text

    def copy_opencv_hsv(self) -> str | None:
        r = self.ranges()
        if r is None:
            return None
        text = opencv_hsv_text(r)
        QGuiApplication.clipboard().setText(text)
        self.ctx.show_message(tr("gui.pipeline.pick.copied_hsv"))
        return text

    # ---- 語言 -------------------------------------------------------------------
    def retranslate(self) -> None:
        self.empty_label.setText(tr("gui.pipeline.empty"))
        self.show_processed.setText(tr("gui.pipeline.show_processed"))
        self.preset_label.setText(tr("gui.pipeline.preset.label"))
        self.preset_apply.setText(tr("common.apply"))
        self.preset_save.setText(tr("gui.pipeline.preset.save"))
        self.steps_label.setText(tr("gui.pipeline.steps"))
        self.add_button.setText(tr("gui.pipeline.add") + " ▾")
        self.dup_button.setText(tr("gui.pipeline.duplicate"))
        self.del_button.setText(tr("common.delete"))
        self.clear_button.setText(tr("gui.pipeline.clear"))
        self.up_button.setToolTip(tr("gui.pipeline.move_up"))
        self.down_button.setToolTip(tr("gui.pipeline.move_down"))
        self.params_box.setTitle(tr("gui.pipeline.params"))
        self.pick_box.setTitle(tr("gui.pipeline.pick.title"))
        self.pick_hint.setText(tr("gui.pipeline.pick.hint"))
        self.pick_button.setText(tr("gui.pipeline.pick.button"))
        self.pick_clear.setText(tr("gui.pipeline.pick.clear"))
        self.radius_label.setText(tr("gui.pipeline.pick.radius"))
        self.margin_label.setText(tr("gui.pipeline.pick.margin"))
        self.apply_hsv_button.setText(tr("gui.pipeline.pick.apply_hsv"))
        self.copy_lab_button.setText(tr("gui.pipeline.pick.copy_lab"))
        self.copy_hsv_button.setText(tr("gui.pipeline.pick.copy_hsv"))
        self.controls.retranslate()
        self._rebuild_add_menu()
        self._refresh_presets()
        self._update_sample_ui()
        if hasattr(self, "list"):
            row = self.list.currentRow()
            if self._steps:
                self._rebuild_list(select=row)

    def shutdown(self) -> None:
        self._stop_pick()


def ctx_get(ctx, key: str, default):
    value = ctx.settings.get(key, default)
    return default if value is None else value
