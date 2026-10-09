"""模型面板：載入模型、選後端、調門檻、看即時結果，學生最常用的推論介面。

載入與推論都在背景執行緒（gui/inference.py 的 InferenceSession），這裡只負責顯示與操作。
同一台相機可以載入第二個模型做 A/B 比較；畫面上第二個模型用虛線和另一組顏色。
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from edgelens.core.controls import ControlSpec, ControlType
from edgelens.core.registry import discover
from edgelens.core.results import InferenceResult
from edgelens.gui.control_panel import ControlPanel
from edgelens.gui.inference import MAX_SESSIONS_PER_CAMERA, InferenceSession
from edgelens.gui.panels.base import Panel
from edgelens.gui.raw_output_dialog import RawOutputDialog
from edgelens.gui.workers import SourceHandle
from edgelens.i18n import tr, tr_or
from edgelens.inference.backends.haar import builtin_cascades, cascade_display_name
from edgelens.inference.base import DECODERS
from edgelens.inference.runner import InferenceRunner, load_model_file

log = logging.getLogger(__name__)

FILE_FILTER = "*.onnx *.tflite *.weights *.cfg *.xml *.kmodel *.card.json"
LIVE_INTERVAL = 0.25         # 即時資訊最多每 0.25 秒更新一次（畫面不要一直重畫文字）
TOP_SHOWN = 5
DEFAULT_DEVICE_DIR = "/flash/"


def threshold_specs(mode: str, task: str) -> list[ControlSpec]:
    """依推論模式（tensor／image／device）與任務（detect／classify）決定要顯示哪些門檻。"""
    F, N, B = ControlType.FLOAT, ControlType.INT, ControlType.BOOL
    conf = ControlSpec("conf", F, "gui.model.th.conf", default=0.25, min=0.01, max=0.99, step=0.01,
                       group="inference", help_key="gui.model.th.conf.help")
    nms = ControlSpec("nms", F, "gui.model.th.nms", default=0.45, min=0.05, max=0.95, step=0.01,
                      group="inference", help_key="gui.model.th.nms.help")
    max_det = ControlSpec("max_det", N, "gui.model.th.max_det", default=100, min=1, max=300, step=1,
                          group="inference", help_key="gui.model.th.max_det.help")
    agnostic = ControlSpec("class_agnostic", B, "gui.model.th.agnostic", default=False,
                           group="inference", help_key="gui.model.th.agnostic.help")
    if mode == "image":
        neighbors = ControlSpec("min_neighbors", N, "gui.model.th.neighbors", default=5, min=1, max=20, step=1,
                                group="inference", help_key="gui.model.th.neighbors.help")
        return [conf, max_det, neighbors]
    if mode == "device":
        return [conf]
    return [conf, nms, max_det, agnostic]


class ModelPanel(Panel):
    panel_id = "model"
    title_key = "gui.panel.model.title"
    order = 30

    def __init__(self, ctx, main):
        super().__init__(ctx, main)
        discover()
        self._handle: SourceHandle | None = None
        self._session: InferenceSession | None = None
        self._pending_card = None
        self._pending_builtin = False
        self._hooked: set[int] = set()
        self._spec_key: tuple = ()
        self._last_live = 0.0
        self._raw_dialog: RawOutputDialog | None = None
        self.last_saved_card: Path | None = None
        self._build()
        self.manager = ctx.inference
        self.manager.sessionsChanged.connect(self._on_sessions_changed)
        self.retranslate()
        self.on_active_source_changed(ctx.active_source)

    # ---- 畫面 -------------------------------------------------------------------
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        lay = QVBoxLayout(body)

        self.empty_label = QLabel()
        self.empty_label.setWordWrap(True)
        lay.addWidget(self.empty_label)

        self.content = QWidget()
        cl = QVBoxLayout(self.content)
        cl.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.content)

        # 載入
        row = QHBoxLayout()
        self.load_button = QPushButton()
        self.load_button.clicked.connect(self.choose_model)
        self.builtin_button = QToolButton()
        self.builtin_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.builtin_menu = QMenu(self.builtin_button)
        self.builtin_button.setMenu(self.builtin_menu)
        self.recent_button = QToolButton()
        self.recent_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.recent_menu = QMenu(self.recent_button)
        self.recent_button.setMenu(self.recent_menu)
        row.addWidget(self.load_button, 1)
        row.addWidget(self.builtin_button)
        row.addWidget(self.recent_button)
        cl.addLayout(row)

        form = QFormLayout()
        self.backend_label = QLabel()
        self.backend_combo = QComboBox()
        self.backend_combo.setModel(QStandardItemModel(self.backend_combo))
        self.backend_combo.activated.connect(self._on_backend_activated)
        form.addRow(self.backend_label, self.backend_combo)
        cl.addLayout(form)
        self.ab_check = QCheckBox()
        cl.addWidget(self.ab_check)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        cl.addWidget(self.status_label)
        self.device_progress = QProgressBar()
        self.device_progress.setRange(0, 100)
        self.device_progress.hide()
        cl.addWidget(self.device_progress)

        # 目前的模型
        srow = QHBoxLayout()
        self.session_combo = QComboBox()
        self.session_combo.currentIndexChanged.connect(self._on_session_picked)
        self.stop_button = QPushButton()
        self.stop_button.clicked.connect(self.stop_current)
        srow.addWidget(self.session_combo, 1)
        srow.addWidget(self.stop_button)
        cl.addLayout(srow)

        # 模型資訊
        self.info_box = QGroupBox()
        iform = QFormLayout(self.info_box)
        self._info_names: dict[str, QLabel] = {}
        self._info_values: dict[str, QLabel] = {}
        for key in ("task", "backend", "input", "outputs", "classes", "load", "decoder"):
            name, value = QLabel(), QLabel("-")
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._info_names[key], self._info_values[key] = name, value
            iform.addRow(name, value)
        cl.addWidget(self.info_box)

        self.ambiguous_box = QGroupBox()
        self.ambiguous_box.setStyleSheet("QGroupBox { border: 2px solid #E69F00; margin-top: 8px; }"
                                         "QGroupBox::title { color: #C77700; font-weight: bold; }")
        av = QVBoxLayout(self.ambiguous_box)
        self.ambiguous_text = QLabel()
        self.ambiguous_text.setWordWrap(True)
        self.decoder_combo = QComboBox()
        self.decoder_apply = QPushButton()
        self.decoder_apply.clicked.connect(self.apply_decoder)
        av.addWidget(self.ambiguous_text)
        av.addWidget(self.decoder_combo)
        av.addWidget(self.decoder_apply)
        cl.addWidget(self.ambiguous_box)

        # 門檻
        self.thresholds = ControlPanel(show_status=False)
        self.thresholds.valueChanged.connect(self._on_threshold)
        cl.addWidget(self.thresholds)
        self.threshold_note = QLabel()
        self.threshold_note.setWordWrap(True)
        cl.addWidget(self.threshold_note)

        # 推論輸入
        irow = QHBoxLayout()
        self.input_label = QLabel()
        self.input_combo = QComboBox()
        self.input_combo.currentIndexChanged.connect(self._on_input_changed)
        irow.addWidget(self.input_label)
        irow.addWidget(self.input_combo, 1)
        cl.addLayout(irow)
        self.input_note = QLabel()
        self.input_note.setWordWrap(True)
        cl.addWidget(self.input_note)

        # 即時資訊
        self.live_box = QGroupBox()
        lv = QVBoxLayout(self.live_box)
        self.live_stats = QLabel("-")
        self.live_stats.setWordWrap(True)
        self.live_list = QLabel()
        self.live_list.setWordWrap(True)
        lv.addWidget(self.live_stats)
        lv.addWidget(self.live_list)
        self._bars: list[tuple[QLabel, QProgressBar]] = []
        for _ in range(TOP_SHOWN):
            r = QHBoxLayout()
            name, bar = QLabel(), QProgressBar()
            bar.setRange(0, 100)
            bar.setFormat("%v%")
            name.setMinimumWidth(80)
            r.addWidget(name)
            r.addWidget(bar, 1)
            lv.addLayout(r)
            name.hide()
            bar.hide()
            self._bars.append((name, bar))
        cl.addWidget(self.live_box)

        brow = QHBoxLayout()
        self.raw_button = QPushButton()
        self.raw_button.clicked.connect(self.open_raw_dialog)
        self.save_button = QPushButton()
        self.save_button.clicked.connect(self.save_card)
        brow.addWidget(self.raw_button)
        brow.addWidget(self.save_button)
        cl.addLayout(brow)
        lay.addStretch(1)
        self._set_session_widgets(None)

    def retranslate(self) -> None:
        self.empty_label.setText(tr("gui.model.empty"))
        self.load_button.setText(tr("gui.model.load"))
        self.builtin_button.setText(tr("gui.model.builtin"))
        self.recent_button.setText(tr("gui.model.recent"))
        self.backend_label.setText(tr("gui.model.backend"))
        self.ab_check.setText(tr("gui.model.ab"))
        self.ab_check.setToolTip(tr("gui.model.ab.tip"))
        self.stop_button.setText(tr("gui.model.stop"))
        self.info_box.setTitle(tr("gui.model.info"))
        for key, lab in self._info_names.items():
            lab.setText(tr(f"gui.model.info.{key}"))
        self.ambiguous_box.setTitle(tr("gui.model.ambiguous.title"))
        self.ambiguous_text.setText(tr("gui.model.ambiguous.text"))
        self.decoder_apply.setText(tr("common.apply"))
        self.input_label.setText(tr("gui.model.input"))
        cur = self.input_combo.currentIndex()
        self.input_combo.blockSignals(True)
        self.input_combo.clear()
        self.input_combo.addItem(tr("gui.model.input.raw"), False)
        self.input_combo.addItem(tr("gui.model.input.processed"), True)
        self.input_combo.setCurrentIndex(max(0, cur))
        self.input_combo.blockSignals(False)
        self.input_note.setText(tr("gui.model.input.note"))
        self.live_box.setTitle(tr("gui.model.live"))
        self.raw_button.setText(tr("gui.model.raw"))
        self.save_button.setText(tr("gui.model.save_card"))
        self.thresholds.retranslate()
        self._fill_menus()
        self._fill_session_combo()
        self._refresh_info()

    def _fill_menus(self) -> None:
        self.builtin_menu.clear()
        for name, path in builtin_cascades().items():
            act = self.builtin_menu.addAction(cascade_display_name(name))
            act.triggered.connect(lambda _=False, p=path: self.load_path(p, builtin=True))
        self.recent_menu.clear()
        recent = [str(p) for p in (self.ctx.settings.get("recent.models", []) or [])]
        for p in recent:
            act = self.recent_menu.addAction(Path(p).name)
            act.setToolTip(p)
            act.triggered.connect(lambda _=False, p=p: self.load_path(p))
        if not recent:
            self.recent_menu.addAction(tr("gui.model.recent.none")).setEnabled(False)

    # ---- 相機與 session -----------------------------------------------------------
    def on_active_source_changed(self, handle: SourceHandle | None) -> None:
        self._handle = handle
        self._pending_card = None
        self.empty_label.setVisible(handle is None)
        self.content.setVisible(handle is not None)
        self.status_label.setText("")
        self._fill_session_combo()
        self._select(self._current_from_combo())

    def on_project_changed(self, project: Any) -> None:
        self._fill_menus()

    def _on_sessions_changed(self) -> None:
        for s in self.manager.sessions():
            if s.session_id not in self._hooked:
                self._hooked.add(s.session_id)
                s.loaded.connect(lambda info, s=s: self._on_loaded(s, info))
                s.failed.connect(lambda err, s=s: self._on_failed(s, err))
                s.infoChanged.connect(lambda s=s: self._on_info_changed(s))
                s.resultReady.connect(lambda r, s=s: self._on_result(s, r))
        keep = self._session
        self._fill_session_combo()
        if keep is not None and keep in self.manager.sessions():
            self._select(keep)
        else:
            self._select(self._current_from_combo())

    def _handle_sessions(self) -> list[InferenceSession]:
        return self.manager.sessions_for(self._handle) if self._handle is not None else []

    def _fill_session_combo(self) -> None:
        if not hasattr(self, "manager"):
            return
        sessions = self._handle_sessions()
        cur = self._session
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        for i, s in enumerate(sessions):
            tag = "AB"[i] if i < 2 else str(i + 1)
            self.session_combo.addItem(f"{tag}: {s.label}", s)
        if cur in sessions:
            self.session_combo.setCurrentIndex(sessions.index(cur))
        self.session_combo.blockSignals(False)
        self.session_combo.setVisible(bool(sessions))

    def _current_from_combo(self) -> InferenceSession | None:
        i = self.session_combo.currentIndex()
        return self.session_combo.itemData(i) if i >= 0 else None

    def _on_session_picked(self, _index: int) -> None:
        self._select(self._current_from_combo())

    def _select(self, session: InferenceSession | None) -> None:
        self._session = session
        self._spec_key = ()
        if session is not None:
            self.input_combo.blockSignals(True)
            self.input_combo.setCurrentIndex(1 if session.use_processed else 0)
            self.input_combo.blockSignals(False)
            self._fill_backends(session.card)
            if session.state == "failed" and session.last_error:
                self._show_error(session.last_error)
            elif session.state == "loading":
                self._set_status(tr("gui.model.loading", name=session.card.name))
        self._set_session_widgets(session)
        self._refresh_info()

    def _set_session_widgets(self, session: InferenceSession | None) -> None:
        has = session is not None
        for w in (self.stop_button, self.info_box, self.thresholds, self.live_box, self.raw_button, self.save_button,
                  self.input_combo, self.input_label, self.input_note):
            w.setVisible(has)
        self.threshold_note.setVisible(has and bool(self.threshold_note.text()))
        if not has:
            self.ambiguous_box.hide()

    # ---- 載入模型 -----------------------------------------------------------------
    def choose_model(self) -> None:
        start = str(self.ctx.project.models_dir) if self.ctx.project is not None else ""
        path, _ = QFileDialog.getOpenFileName(self, tr("gui.model.load"), start,
                                              tr("gui.model.filter") + f" ({FILE_FILTER})")
        if path:
            self.load_path(path)

    def load_path(self, path: str | Path, *, builtin: bool = False) -> bool:
        """讀模型檔（或 .card.json）→ 選後端 → 開始在背景載入。回傳是否已開始載入。"""
        if self._handle is None:
            self._set_status(tr("gui.model.empty"), error=True)
            return False
        try:
            card = load_model_file(path)
        except Exception as e:  # noqa: BLE001 - 說明檔壞掉、版本太新等，要轉成學生看得懂的訊息
            self._set_status(tr("gui.model.err.card", path=Path(path).name, error=str(e)[:200]) + "\n"
                             + tr("gui.model.err.card_hint"), error=True)
            return False
        if not builtin:
            self.ctx.settings.push_recent("models", str(Path(path).resolve()))
            self.ctx.save_settings_later()
            self._fill_menus()
        self._pending_card, self._pending_builtin = card, builtin
        if card.extension == ".kmodel":
            return self._start_device(card)
        self._fill_backends(card)
        ok = [c for c in InferenceRunner.backend_choices(card) if c.ok]
        return self._start(card, ok[0].kind if ok else None)

    def _fill_backends(self, card) -> None:
        model = self.backend_combo.model()
        model.clear()
        choices = InferenceRunner.backend_choices(card)
        current = self._session.backend if self._session is not None else None
        pick = -1
        for i, c in enumerate(choices):
            text = c.name if c.ok else f"{c.name} - {tr('gui.model.backend.unavailable')}：{c.reason}"
            item = QStandardItem(text)
            item.setData(c.kind, Qt.ItemDataRole.UserRole)
            item.setToolTip(c.reason or c.name)
            if not c.ok:
                item.setEnabled(False)
            model.appendRow(item)
            if c.ok and pick < 0 and (current is None or current == c.kind):
                pick = i
        if pick < 0 and choices:
            pick = next((i for i, c in enumerate(choices) if c.ok), 0)
        self.backend_combo.setCurrentIndex(pick)

    def _on_backend_activated(self, index: int) -> None:
        """學生在下拉選單換後端：用同一個模型換後端重新載入。"""
        card = self._session.card if self._session is not None else self._pending_card
        kind = self.backend_combo.itemData(index, Qt.ItemDataRole.UserRole)
        if card is not None and kind:
            self._start(copy.deepcopy(card), kind, builtin=self._session.builtin if self._session else False,
                        replace_selected=True)

    def _start(self, card, backend: str | None, *, builtin: bool | None = None,
               replace_selected: bool = False) -> bool:
        handle = self._handle
        if handle is None:
            return False
        builtin = self._pending_builtin if builtin is None else builtin
        existing = self._handle_sessions()
        add_second = self.ab_check.isChecked() and bool(existing) and not replace_selected
        if replace_selected and self._session in existing:
            self.manager.remove(self._session)
        elif add_second:
            while len(existing) >= MAX_SESSIONS_PER_CAMERA:
                self.manager.remove(existing.pop())
        else:
            for s in existing:
                self.manager.remove(s)
        session = self.manager.add(handle, card, backend=backend,
                                   use_processed=bool(self.input_combo.currentData()))
        session.builtin = builtin
        self._set_status(tr("gui.model.loading", name=card.name))
        self._fill_session_combo()
        self.session_combo.setCurrentIndex(self._handle_sessions().index(session))
        self._select(session)
        return True

    def _start_device(self, card) -> bool:
        """kmodel 只能在 K210 板上執行：先把模型傳給板子，成功後才建立 session。"""
        handle = self._handle
        if handle is None or handle.info.kind != "k210":
            self._set_status(tr("gui.model.kmodel_only_k210"), error=True)
            return False
        if not card.device.get("path"):
            card.device["path"] = DEFAULT_DEVICE_DIR + card.model_path.name
        for s in self._handle_sessions():
            self.manager.remove(s, unload=False)
        self.device_progress.setValue(0)
        self.device_progress.show()
        self._set_status(tr("gui.model.device_loading", name=card.name))

        def progress(ratio: float, text: str) -> None:
            self.device_progress.setValue(int(max(0.0, min(1.0, ratio)) * 100))
            if text:
                self._set_status(text)

        def done(session, res) -> None:
            self.device_progress.hide()
            if session is None:
                self._set_status((res.message or tr("gui.model.device_failed")), error=True)
                return
            session.builtin = False
            self._set_status(res.message or tr("gui.model.device_done", name=card.name))
            self._fill_session_combo()
            self.session_combo.setCurrentIndex(self._handle_sessions().index(session))
            self._select(session)

        self.manager.add_device(handle, card, on_progress=progress, on_done=done)
        return True

    # ---- session 事件 -------------------------------------------------------------
    def _on_loaded(self, session: InferenceSession, info) -> None:
        if session is self._session:
            self._set_status(tr("gui.model.loaded", name=session.card.name, ms=f"{info.load_ms:.0f}"))
            self._refresh_info()

    def _on_failed(self, session: InferenceSession, err: dict) -> None:
        if session is self._session:
            self._show_error(err)

    def _on_info_changed(self, session: InferenceSession) -> None:
        if session is self._session:
            self._refresh_info()

    def _show_error(self, err: dict) -> None:
        text = err.get("message", "")
        if err.get("hint"):
            text += "\n" + err["hint"]
        self._set_status(text, error=True)

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet("color:#C0392B;" if error else "")

    # ---- 資訊 ---------------------------------------------------------------------
    def _refresh_info(self) -> None:
        s = self._session
        if s is None:
            for lab in self._info_values.values():
                lab.setText("-")
            self.ambiguous_box.hide()
            return
        info = s.info
        v = self._info_values
        loaded = info.loaded or s.state == "running"
        v["task"].setText(tr(f"gui.model.task.{info.task}") if loaded and info.task in ("detect", "classify") else "-")
        v["backend"].setText(info.backend_desc or "-")
        v["input"].setText(info.input_desc or "-")
        v["outputs"].setText("\n".join(o.describe() for o in info.outputs) or "-")
        labels = info.labels or s.card.labels
        if labels:
            shown = tr("gui.list_sep").join(labels[:6]) + ("…" if len(labels) > 6 else "")
            v["classes"].setText(tr("gui.model.classes", count=len(labels), names=shown))
        else:
            v["classes"].setText(tr("gui.model.classes.none"))
        v["load"].setText(f"{info.load_ms:.0f} ms" if loaded else "-")
        if info.decoder:
            name = self._decoder_name(info.decoder)
            v["decoder"].setText(f"{name}\n{info.decoder_reason}" if info.decoder_reason else name)
        else:
            v["decoder"].setText(info.decoder_reason or "-")
        self._refresh_ambiguous()
        self._refresh_thresholds()

    @staticmethod
    def _decoder_name(kind: str) -> str:
        cls = DECODERS.find(kind)
        return tr_or(cls.display_name_key, kind) if cls is not None else kind

    def _refresh_ambiguous(self) -> None:
        s = self._session
        info = s.info if s is not None else None
        show = bool(info and info.mode == "tensor" and info.ambiguous and info.guesses)
        self.ambiguous_box.setVisible(show)
        if not show:
            return
        self.decoder_combo.blockSignals(True)
        self.decoder_combo.clear()
        seen = set()
        for g in info.guesses:
            seen.add(g.decoder_kind)
            self.decoder_combo.addItem(f"{self._decoder_name(g.decoder_kind)}（{g.score:.0%}）", (g.decoder_kind, g.params))
            self.decoder_combo.setItemData(self.decoder_combo.count() - 1, g.reason, Qt.ItemDataRole.ToolTipRole)
        for kind in DECODERS.kinds():
            if kind not in seen:
                self.decoder_combo.addItem(self._decoder_name(kind), (kind, {}))
        self.decoder_combo.blockSignals(False)

    def apply_decoder(self) -> None:
        """學生手動指定輸出格式（自動判斷有疑義時）。"""
        s = self._session
        data = self.decoder_combo.currentData()
        if s is None or not data:
            return
        kind, params = data
        s.chosen_params = dict(params)
        s.set_decoder(kind, dict(params))
        self._set_status(tr("gui.model.decoder_set", name=self._decoder_name(kind)))

    # ---- 門檻 ---------------------------------------------------------------------
    def _refresh_thresholds(self) -> None:
        s = self._session
        if s is None:
            return
        info = s.info
        mode, task = info.mode, info.task
        key = (s.session_id, mode, task, s.state == "failed")
        values = s.thresholds
        values["min_neighbors"] = int(s.card.decoder_params.get("min_neighbors") or 5)
        if key != self._spec_key:
            self._spec_key = key
            self.thresholds.set_specs(threshold_specs(mode, task), values)
            note = ""
            if mode == "image":
                note = tr("gui.model.th.note.haar")
                self.thresholds.set_enabled_key("conf", False)
            elif mode == "device":
                note = tr("gui.model.th.note.device")
            elif task == "classify" and info.loaded:
                note = tr("gui.model.th.note.classify")
                for k in ("conf", "nms", "max_det", "class_agnostic"):
                    self.thresholds.set_enabled_key(k, False)
            self.threshold_note.setText(note)
            self.threshold_note.setVisible(bool(note))
        else:
            self.thresholds.set_values(values)

    def _on_threshold(self, key: str, value: Any) -> None:
        s = self._session
        if s is None:
            return
        if key == "conf":
            s.set_thresholds(conf=float(value))
        elif key == "nms":
            s.set_thresholds(nms=float(value))
        elif key == "max_det":
            s.set_thresholds(max_det=int(value))
        elif key == "class_agnostic":
            s.set_thresholds(class_agnostic=bool(value))
        elif key == "min_neighbors":
            s.set_param("min_neighbors", int(value))

    def _on_input_changed(self, _index: int) -> None:
        if self._session is not None:
            self._session.use_processed = bool(self.input_combo.currentData())

    # ---- 即時資訊 -----------------------------------------------------------------
    def _on_result(self, session: InferenceSession, result: InferenceResult) -> None:
        if session is not self._session:
            return
        now = time.monotonic()
        if now - self._last_live < LIVE_INTERVAL:
            return
        self._last_live = now
        self.update_live()

    def update_live(self) -> None:
        s = self._session
        if s is None or s.last_result is None:
            return
        r, st = s.last_result, s.stats
        t = r.timings_ms
        self.live_stats.setText(tr("gui.model.live.stats", fps=f"{st['infer_fps']:.1f}",
                                   pre=f"{t.get('preprocess', 0.0):.1f}", infer=f"{t.get('infer', 0.0):.1f}",
                                   post=f"{t.get('postprocess', 0.0):.1f}", total=f"{t.get('total', 0.0):.1f}",
                                   age=int(st["age_frames"])))
        if r.extra.get("device_missing"):
            self.live_list.setText(tr("gui.model.live.device_missing"))
            self._show_bars([])
        elif r.kind == "classify":
            self.live_list.setText("")
            self._show_bars(r.top_k[:TOP_SHOWN])
        else:
            self._show_bars([])
            if r.detections:
                lines = [f"{d.label or d.class_id}  {d.score:.2f}" for d in r.detections[:8]]
                more = len(r.detections) - 8
                if more > 0:
                    lines.append(tr("gui.model.live.more", count=more))
                self.live_list.setText("\n".join(lines))
            else:
                self.live_list.setText(tr("gui.model.live.none"))

    def _show_bars(self, items: list) -> None:
        for i, (name, bar) in enumerate(self._bars):
            if i < len(items):
                c = items[i]
                name.setText(c.label or str(c.class_id))
                bar.setValue(int(round(c.score * 100)))
                name.show()
                bar.show()
            else:
                name.hide()
                bar.hide()

    # ---- 其他動作 -----------------------------------------------------------------
    def stop_current(self) -> None:
        s = self._session
        if s is None:
            return
        self.manager.remove(s)
        self._set_status(tr("gui.model.stopped"))

    def open_raw_dialog(self) -> RawOutputDialog | None:
        s = self._session
        if s is None:
            return None
        if self._raw_dialog is not None:
            self._raw_dialog.close()
        self._raw_dialog = RawOutputDialog(s, self)
        self._raw_dialog.show()
        return self._raw_dialog

    def build_card(self, session: InferenceSession):
        """把目前的解碼器、參數、門檻、類別名稱寫進一份新的 card（下次載入就不用再判斷）。"""
        card = copy.deepcopy(session.card)
        info = session.info
        th = session.thresholds
        card.conf_threshold, card.nms_threshold = float(th["conf"]), float(th["nms"])
        card.max_det, card.class_agnostic_nms = int(th["max_det"]), bool(th["class_agnostic"])
        if info.backend:
            card.backend = info.backend
        if info.task in ("detect", "classify"):
            card.task = info.task
        if info.labels and not card.labels:
            card.labels = list(info.labels)
        if info.mode == "tensor" and info.decoder:
            params = next((g.params for g in info.guesses if g.decoder_kind == info.decoder), {})
            card.decoder = info.decoder
            card.decoder_params = {**card.decoder_params, **params, **getattr(session, "chosen_params", {})}
        return card

    def save_card(self) -> Path | None:
        s = self._session
        if s is None or s.state == "failed":
            return None
        card = self.build_card(s)
        target = None if s.builtin else (card.card_path or card.default_card_path())
        try:
            if target is None:
                raise OSError("builtin")
            path = card.save(target)
        except OSError:
            if self.ctx.project is None:
                self._set_status(tr("gui.model.save_card.need_project"), error=True)
                return None
            path = card.save(self.ctx.project.models_dir / (card.model_path.stem + ".card.json"))
        self.last_saved_card = path
        self._set_status(tr("gui.model.save_card.done", path=str(path)))
        self.ctx.show_message(tr("gui.model.save_card.done", path=str(path)))
        return path

    def shutdown(self) -> None:
        if self._raw_dialog is not None:
            self._raw_dialog.close()
