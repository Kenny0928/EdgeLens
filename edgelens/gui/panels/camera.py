"""相機面板：作用中相機的資訊、參數（自動產生）、動作按鈕與「重設為預設值」。"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from edgelens.core.controls import ControlResult, ControlType, defaults_of
from edgelens.gui.control_panel import ControlPanel
from edgelens.gui.panels.base import Panel
from edgelens.gui.workers import FramePacket, SourceHandle
from edgelens.i18n import tr, tr_or
from edgelens.sources.base import SOURCES


class CameraPanel(Panel):
    panel_id = "camera"
    title_key = "gui.panel.camera.title"
    order = 10

    def __init__(self, ctx, main):
        super().__init__(ctx, main)
        self._handle: SourceHandle | None = None
        self._last_info_update = 0.0

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

        self.info_box = QGroupBox()
        form = QFormLayout(self.info_box)
        self._info_labels: dict[str, QLabel] = {}
        self._info_names: dict[str, QLabel] = {}
        for key in ("name", "kind", "resolution", "fps", "latency", "capabilities", "state"):
            value = QLabel("-")
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            name = QLabel()
            self._info_labels[key] = value
            self._info_names[key] = name
            form.addRow(name, value)
        lay.addWidget(self.info_box)

        self.progress_box = QWidget()
        pl = QHBoxLayout(self.progress_box)
        pl.setContentsMargins(0, 0, 0, 0)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.cancel_button = QPushButton()
        self.cancel_button.clicked.connect(self._cancel_action)
        pl.addWidget(self.progress, 1)
        pl.addWidget(self.cancel_button)
        self.progress_text = QLabel()
        self.progress_text.setWordWrap(True)
        lay.addWidget(self.progress_box)
        lay.addWidget(self.progress_text)
        self.progress_box.hide()
        self.progress_text.hide()

        self.controls = ControlPanel(advanced_expanded=bool(ctx.settings.get("ui.show_advanced", False)))
        self.controls.valueChanged.connect(self._on_value_changed)
        self.controls.actionTriggered.connect(self._on_action)
        self.controls.advancedToggled.connect(self._on_advanced_toggled)
        lay.addWidget(self.controls)

        self.reset_button = QPushButton()
        self.reset_button.clicked.connect(self._reset_defaults)
        lay.addWidget(self.reset_button)
        lay.addStretch(1)
        self.retranslate()
        self.on_active_source_changed(ctx.active_source)

    # ---- 切換相機 ---------------------------------------------------------------
    def on_active_source_changed(self, handle: SourceHandle | None) -> None:
        old = self._handle
        if old is not None:
            for sig, slot in ((old.opened, self._on_opened), (old.controlResult, self._on_control_result),
                              (old.actionProgress, self._on_progress), (old.actionFinished, self._on_finished),
                              (old.stateChanged, self._on_state)):
                try:
                    sig.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
        self._handle = handle
        self.progress_box.hide()
        self.progress_text.hide()
        if handle is not None:
            handle.opened.connect(self._on_opened)
            handle.controlResult.connect(self._on_control_result)
            handle.actionProgress.connect(self._on_progress)
            handle.actionFinished.connect(self._on_finished)
            handle.stateChanged.connect(self._on_state)
        self._refresh()

    def _refresh(self) -> None:
        h = self._handle
        has = h is not None
        self.empty_label.setVisible(not has)
        self.info_box.setVisible(has)
        self.controls.setVisible(has)
        self.reset_button.setVisible(has)
        if not has:
            self.controls.set_specs([], {})
            return
        self.controls.set_specs(h.controls, h.values)
        self.controls.clear_status()
        self._update_info(force=True)

    # ---- 資訊 -------------------------------------------------------------------
    def _update_info(self, force: bool = False) -> None:
        h = self._handle
        if h is None:
            return
        now = time.monotonic()
        if not force and now - self._last_info_update < 0.5:
            return
        self._last_info_update = now
        cls = SOURCES.find(h.info.kind)
        kind_name = tr_or(cls.display_name_key, h.info.kind) if cls is not None else h.info.kind
        pkt = h.last_packet
        lab = self._info_labels
        lab["name"].setText(h.info.name)
        lab["kind"].setText(kind_name)
        lab["state"].setText(tr(f"gui.state.{h.state}"))
        if pkt is not None:
            lab["resolution"].setText(f"{pkt.frame.width}×{pkt.frame.height}")
            lab["fps"].setText(f"{pkt.stats.get('fps', 0.0):.1f}")
            lab["latency"].setText(f"{pkt.stats.get('latency_ms', 0.0):.0f} ms")
        else:
            for k in ("resolution", "fps", "latency"):
                lab[k].setText("-")
        caps = h.capabilities
        parts = []
        if caps.device_inference:
            parts.append(tr("gui.panel.camera.cap.device_inference"))
        if caps.provides_jpeg:
            parts.append("JPEG")
        if caps.live_controls:
            parts.append(tr("gui.panel.camera.cap.live_controls"))
        if caps.max_fps:
            parts.append(f"≤ {caps.max_fps:g} fps")
        lab["capabilities"].setText(tr("gui.list_sep").join(parts) if parts else "-")

    def on_frame(self, packet: FramePacket) -> None:
        self._update_info()

    # ---- 事件 -------------------------------------------------------------------
    def _on_opened(self, _info: dict) -> None:
        self._refresh()

    def _on_state(self, _state: str) -> None:
        self._update_info(force=True)

    def _on_value_changed(self, key: str, value) -> None:
        if self._handle is not None:
            self._handle.set_control(key, value)
            self.controls.set_status(key, None, tr("gui.control.status.sending"))
            self.ctx.save_settings_later()

    def _on_control_result(self, key: str, res: ControlResult) -> None:
        if res.ok:
            self.controls.set_status(key, res.verified, res.message)
            if res.verified is False and res.value is not None:
                self.controls.set_values({key: res.value})  # 硬體不接受：顯示它實際的值
        else:
            self.controls.set_status(key, False, res.message)
            if res.value is not None:
                self.controls.set_values({key: res.value})
            self.ctx.show_message(res.message or tr("gui.control.status.rejected"))

    def _on_advanced_toggled(self, expanded: bool) -> None:
        self.ctx.settings.set("ui.show_advanced", expanded)
        self.ctx.save_settings_later()

    def _on_action(self, key: str) -> None:
        if self._handle is None:
            return
        self.progress.setValue(0)
        self.progress_box.show()
        self.progress_text.setText(tr("gui.panel.camera.action_running"))
        self.progress_text.show()
        self.controls.set_all_enabled(False, actions_only=True)
        self._handle.run_action(key)

    def _on_progress(self, key: str, ratio: float, text: str) -> None:
        self.progress.setValue(int(max(0.0, min(1.0, ratio)) * 100))
        if text:
            self.progress_text.setText(text)

    def _on_finished(self, key: str, res: ControlResult) -> None:
        self.progress_box.hide()
        self.controls.set_all_enabled(True, actions_only=True)
        if res.ok:
            self.progress_text.setText(res.message or tr("gui.panel.camera.action_done"))
        else:
            self.progress_text.setText(res.message or tr("gui.panel.camera.action_failed"))
        self.progress_text.show()

    def _cancel_action(self) -> None:
        if self._handle is not None:
            self._handle.cancel_action()

    def _reset_defaults(self) -> None:
        h = self._handle
        if h is None:
            return
        defaults = defaults_of([s for s in h.controls if s.type != ControlType.ACTION])
        self.controls.set_values(defaults)
        for key, value in defaults.items():
            h.set_control(key, value)
        h.flush_controls()
        self.ctx.show_message(tr("gui.panel.camera.reset_done"))

    # ---- 語言 -------------------------------------------------------------------
    def retranslate(self) -> None:
        self.empty_label.setText(tr("gui.panel.camera.empty"))
        self.info_box.setTitle(tr("gui.panel.camera.info"))
        for key, name in self._info_names.items():
            name.setText(tr(f"gui.panel.camera.info.{key}"))
        self.cancel_button.setText(tr("common.cancel"))
        self.reset_button.setText(tr("gui.panel.camera.reset"))
        self.controls.retranslate()
        self._update_info(force=True)
