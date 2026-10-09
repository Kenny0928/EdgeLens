"""K210 開發板面板：板子資訊、板上即時數據，以及上傳檔案、設定頻率、燒錄韌體。

只有作用中相機是 K210 時才有內容。板子資訊來自 handle.capabilities.extra（開啟時 K210Source
放進去的 CAPS），即時數據來自每張畫面的 frame.meta（板上耗時、fps、記憶體），不會直接碰
worker 執行緒裡的 K210Source。動作都透過 handle.run_action 在相機執行緒執行。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from edgelens.core.controls import ControlResult
from edgelens.gui.panels.base import Panel
from edgelens.gui.workers import FramePacket, SourceHandle
from edgelens.i18n import tr
from edgelens.sources.k210 import flash as flash_mod

ACTIONS = ("upload_file", "set_cpu_freq", "flash_firmware")
DEFAULT_MHZ = 400


def _mb(n: Any) -> str:
    try:
        return f"{float(n) / 1_000_000:.2f} MB"
    except (TypeError, ValueError):
        return "-"


class K210Panel(Panel):
    panel_id = "k210"
    title_key = "k210p.title"
    order = 80

    def __init__(self, ctx, main):
        super().__init__(ctx, main)
        self._handle: SourceHandle | None = None
        self._last_update = 0.0
        self._busy_key = ""
        self.last_result: ControlResult | None = None

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
        self._build_info(cl)
        self._build_live(cl)
        self._build_actions(cl)
        lay.addStretch(1)
        self.retranslate()
        self.on_active_source_changed(ctx.active_source)

    # ---- 建立介面 ---------------------------------------------------------------
    def _build_info(self, lay: QVBoxLayout) -> None:
        self.info_box = QGroupBox()
        form = QFormLayout(self.info_box)
        self._info_names: dict[str, QLabel] = {}
        self._info: dict[str, QLabel] = {}
        for key in ("fw", "sensor", "memory", "sd", "kpu", "sizes"):
            name, val = QLabel(), QLabel("-")
            val.setWordWrap(True)
            self._info_names[key], self._info[key] = name, val
            form.addRow(name, val)
        lay.addWidget(self.info_box)

    def _build_live(self, lay: QVBoxLayout) -> None:
        self.live_box = QGroupBox()
        v = QVBoxLayout(self.live_box)
        form = QFormLayout()
        self._live_names: dict[str, QLabel] = {}
        self._live: dict[str, QLabel] = {}
        for key in ("fps", "mem", "heap", "lost", "bad"):
            name, val = QLabel(), QLabel("-")
            self._live_names[key], self._live[key] = name, val
            form.addRow(name, val)
        v.addLayout(form)
        self.us_table = QTableWidget(0, 2)
        self.us_table.horizontalHeader().hide()
        self.us_table.verticalHeader().hide()
        self.us_table.horizontalHeader().setStretchLastSection(True)
        self.us_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.us_table.setMaximumHeight(130)
        v.addWidget(self.us_table)
        lay.addWidget(self.live_box)

    def _build_actions(self, lay: QVBoxLayout) -> None:
        # 進度與結果（所有動作共用）
        self.progress_box = QWidget()
        pl = QHBoxLayout(self.progress_box)
        pl.setContentsMargins(0, 0, 0, 0)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.cancel_button = QPushButton()
        self.cancel_button.clicked.connect(self.cancel)
        pl.addWidget(self.progress, 1)
        pl.addWidget(self.cancel_button)
        self.progress_box.hide()
        self.result_label = QLabel()
        self.result_label.setWordWrap(True)

        # 上傳檔案
        self.upload_box = QGroupBox()
        v = QVBoxLayout(self.upload_box)
        row = QHBoxLayout()
        self.upload_path = QLineEdit()
        self.upload_browse = QPushButton()
        self.upload_browse.clicked.connect(self._browse_upload)
        row.addWidget(self.upload_path, 1)
        row.addWidget(self.upload_browse)
        v.addLayout(row)
        row = QHBoxLayout()
        self.upload_dir = QComboBox()
        self.upload_dir.addItems(["/flash/", "/sd/"])
        self.upload_name = QLineEdit()
        row.addWidget(self.upload_dir)
        row.addWidget(self.upload_name, 1)
        self.upload_button = QPushButton()
        self.upload_button.clicked.connect(self.upload)
        row.addWidget(self.upload_button)
        v.addLayout(row)
        lay.addWidget(self.upload_box)

        # 頻率
        self.freq_box = QGroupBox()
        v = QVBoxLayout(self.freq_box)
        row = QHBoxLayout()
        self.cpu_label, self.kpu_label = QLabel(), QLabel()
        self.cpu_spin, self.kpu_spin = QSpinBox(), QSpinBox()
        for s in (self.cpu_spin, self.kpu_spin):
            s.setRange(100, 600)
            s.setSingleStep(25)
            s.setValue(DEFAULT_MHZ)
            s.setSuffix(" MHz")
        self.freq_button = QPushButton()
        self.freq_button.clicked.connect(self.set_freq)
        for w in (self.cpu_label, self.cpu_spin, self.kpu_label, self.kpu_spin, self.freq_button):
            row.addWidget(w)
        v.addLayout(row)
        self.freq_note = QLabel()
        self.freq_note.setWordWrap(True)
        v.addWidget(self.freq_note)
        lay.addWidget(self.freq_box)

        # 燒錄韌體
        self.flash_box = QGroupBox()
        v = QVBoxLayout(self.flash_box)
        self.flash_warn = QLabel()
        self.flash_warn.setWordWrap(True)
        self.flash_warn.setStyleSheet("color: #b00020; font-weight: bold;")
        v.addWidget(self.flash_warn)
        row = QHBoxLayout()
        self.flash_path = QLineEdit()
        self.flash_browse = QPushButton()
        self.flash_browse.clicked.connect(self._browse_flash)
        row.addWidget(self.flash_path, 1)
        row.addWidget(self.flash_browse)
        v.addLayout(row)
        row = QHBoxLayout()
        self.board_label = QLabel()
        self.board_combo = QComboBox()
        self.board_combo.addItems(list(flash_mod.BOARDS))
        self.board_combo.setCurrentText(flash_mod.DEFAULT_BOARD)
        self.flash_button = QPushButton()
        self.flash_button.clicked.connect(self.flash)
        row.addWidget(self.board_label)
        row.addWidget(self.board_combo, 1)
        row.addWidget(self.flash_button)
        v.addLayout(row)
        lay.addWidget(self.flash_box)

        self.model_note = QLabel()
        self.model_note.setWordWrap(True)
        lay.addWidget(self.progress_box)
        lay.addWidget(self.result_label)
        lay.addWidget(self.model_note)
        self._action_widgets = (self.upload_button, self.freq_button, self.flash_button)

    # ---- 切換相機 ---------------------------------------------------------------
    def on_active_source_changed(self, handle: SourceHandle | None) -> None:
        old = self._handle
        if old is not None:
            for sig, slot in ((old.opened, self._on_opened), (old.actionProgress, self._on_progress),
                              (old.actionFinished, self._on_finished)):
                try:
                    sig.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
        self._handle = handle if handle is not None and handle.info.kind == "k210" else None
        self._busy_key = ""
        self.progress_box.hide()
        self.result_label.clear()
        if self._handle is not None:
            self._handle.opened.connect(self._on_opened)
            self._handle.actionProgress.connect(self._on_progress)
            self._handle.actionFinished.connect(self._on_finished)
        self.content.setVisible(self._handle is not None)
        self.empty_label.setVisible(self._handle is None)
        self._refresh_info()
        self._refresh_live(force=True)
        self._set_busy(False)

    def _on_opened(self, _info: dict) -> None:
        self._refresh_info()

    # ---- 顯示 -------------------------------------------------------------------
    def _caps(self) -> dict[str, Any]:
        h = self._handle
        return dict(h.capabilities.extra) if h is not None else {}

    def _refresh_info(self) -> None:
        caps = self._caps()
        lab = self._info
        if not caps:
            for v in lab.values():
                v.setText("-")
            return
        lab["fw"].setText(str(caps.get("fw", "-")))
        sensor = caps.get("sensor")
        lab["sensor"].setText(f"{sensor} (id {caps.get('sensor_id')})" if sensor else "-")
        lab["memory"].setText(tr("k210p.info.memory_value", mem=_mb(caps.get("mem_free")),
                                 heap=_mb(caps.get("heap_free"))))
        lab["sd"].setText(tr("common.yes") if caps.get("sd") else tr("k210p.info.no_sd"))
        lab["kpu"].setText(tr("common.yes") if caps.get("kpu") else tr("common.no"))
        lab["sizes"].setText(", ".join(str(s) for s in caps.get("framesizes", [])) or "-")

    def on_frame(self, packet: FramePacket) -> None:
        self._refresh_live(packet)

    def _refresh_live(self, packet: FramePacket | None = None, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_update < 0.4:
            return
        self._last_update = now
        h = self._handle
        if h is not None and self._info["fw"].text() == "-":
            self._refresh_info()   # 相機開啟時面板可能還沒接上 opened，補一次
        if packet is None and h is not None:
            packet = h.last_packet
        meta = packet.frame.meta if (packet is not None and h is not None) else {}
        v = self._live
        if not meta:
            for lab in v.values():
                lab.setText("-")
            self.us_table.setRowCount(0)
            return
        fps = meta.get("fps")
        v["fps"].setText(f"{float(fps):.1f}" if isinstance(fps, (int, float)) else "-")
        v["mem"].setText(_mb(meta.get("mem")))
        v["heap"].setText(_mb(meta.get("heap")))
        host = meta.get("host") or {}
        v["lost"].setText(str(host.get("lost_seq", 0)))
        v["bad"].setText(str(int(host.get("bad_frames", 0)) + int(host.get("decode_failed", 0))))
        us = meta.get("us") if isinstance(meta.get("us"), dict) else {}
        self.us_table.setRowCount(len(us))
        for i, (k, val) in enumerate(us.items()):
            self.us_table.setItem(i, 0, QTableWidgetItem(tr("k210p.live.step", step=k)))
            ms = float(val) / 1000.0 if isinstance(val, (int, float)) else 0.0
            self.us_table.setItem(i, 1, QTableWidgetItem(f"{ms:.2f} ms"))
        self.us_table.resizeColumnToContents(0)

    # ---- 動作 -------------------------------------------------------------------
    def _confirm(self, title: str, text: str, strong: bool = False) -> bool:
        """確認對話框，預設按鈕是「否」。測試會用 monkeypatch 取代。"""
        icon = QMessageBox.Icon.Warning if strong else QMessageBox.Icon.Question
        box = QMessageBox(icon, title, text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, self)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        return box.exec() == QMessageBox.StandardButton.Yes

    def _browse_upload(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("k210p.upload.pick"))
        if path:
            self.upload_path.setText(path)
            if not self.upload_name.text().strip():
                self.upload_name.setText(Path(path).name)

    def _browse_flash(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("k210p.flash.pick"), "", "Firmware (*.bin)")
        if path:
            self.flash_path.setText(path)

    def _fail(self, text: str) -> bool:
        self.result_label.setText(text)
        return False

    def upload(self) -> bool:
        local = self.upload_path.text().strip()
        if not local or not Path(local).is_file():
            return self._fail(tr("k210p.err.no_file"))
        name = self.upload_name.text().strip() or Path(local).name
        if "/" in name or "\\" in name:
            return self._fail(tr("k210p.err.bad_name"))
        remote = self.upload_dir.currentText() + name
        if not self._confirm(tr("k210p.upload.title"), tr("k210p.upload.confirm", local=local, remote=remote)):
            return False
        return self._start("upload_file", local_path=local, remote_path=remote)

    def set_freq(self) -> bool:
        cpu, kpu = self.cpu_spin.value(), self.kpu_spin.value()
        text = tr("k210p.freq.confirm", cpu=cpu, kpu=kpu)
        if cpu > DEFAULT_MHZ or kpu > DEFAULT_MHZ:
            text += "\n\n" + tr("k210p.freq.overclock")
        if not self._confirm(tr("k210p.freq.title"), text, strong=cpu > DEFAULT_MHZ or kpu > DEFAULT_MHZ):
            return False
        return self._start("set_cpu_freq", cpu_mhz=cpu, kpu_mhz=kpu)

    def flash(self) -> bool:
        path = self.flash_path.text().strip()
        if not path or not path.lower().endswith(".bin") or not Path(path).is_file():
            return self._fail(tr("k210p.err.no_bin"))
        board = self.board_combo.currentText()
        if not self._confirm(tr("k210p.flash.title"), tr("k210p.flash.confirm", path=path, board=board), strong=True):
            return False
        return self._start("flash_firmware", firmware_path=path, board=board)

    def _start(self, key: str, **kwargs: Any) -> bool:
        h = self._handle
        if h is None or self._busy_key:
            return False
        self._busy_key = key
        self.progress.setValue(0)
        self.progress_box.show()
        self.result_label.setText(tr("k210p.running"))
        self._set_busy(True)
        h.run_action(key, **kwargs)
        return True

    def cancel(self) -> None:
        if self._handle is not None and self._busy_key:
            self._handle.cancel_action()
            self.result_label.setText(tr("k210p.cancelling"))

    def _set_busy(self, busy: bool) -> None:
        for w in self._action_widgets:
            w.setEnabled(not busy)

    def _on_progress(self, key: str, ratio: float, text: str) -> None:
        if key not in ACTIONS:
            return
        self.progress.setValue(int(max(0.0, min(1.0, ratio)) * 100))
        if text:
            self.result_label.setText(text)

    def _on_finished(self, key: str, res: ControlResult) -> None:
        if key not in ACTIONS:
            return
        self._busy_key = ""
        self.progress_box.hide()
        self._set_busy(False)
        self.last_result = res
        msg = res.message or (tr("k210p.done") if res.ok else tr("k210p.failed"))
        self.result_label.setText(("✓ " if res.ok else "✗ ") + msg)
        self.ctx.show_message(msg.splitlines()[0] if msg else "")

    # ---- 語言與結束 -------------------------------------------------------------
    def retranslate(self) -> None:
        self.empty_label.setText(tr("k210p.empty"))
        self.info_box.setTitle(tr("k210p.info.title"))
        for k, lab in self._info_names.items():
            lab.setText(tr(f"k210p.info.{k}"))
        self.live_box.setTitle(tr("k210p.live.title"))
        for k, lab in self._live_names.items():
            lab.setText(tr(f"k210p.live.{k}"))
        self.upload_box.setTitle(tr("k210p.upload.title"))
        self.upload_path.setPlaceholderText(tr("k210p.upload.pick"))
        self.upload_name.setPlaceholderText(tr("k210p.upload.name"))
        self.upload_browse.setText(tr("common.browse"))
        self.upload_button.setText(tr("k210p.upload.button"))
        self.freq_box.setTitle(tr("k210p.freq.title"))
        self.cpu_label.setText("CPU")
        self.kpu_label.setText("KPU")
        self.freq_button.setText(tr("common.apply"))
        self.freq_note.setText(tr("k210p.freq.note"))
        self.flash_box.setTitle(tr("k210p.flash.title"))
        self.flash_warn.setText(tr("k210p.flash.warning"))
        self.flash_path.setPlaceholderText(tr("k210p.flash.pick"))
        self.flash_browse.setText(tr("common.browse"))
        self.board_label.setText(tr("k210p.flash.board"))
        self.flash_button.setText(tr("k210p.flash.button"))
        self.cancel_button.setText(tr("common.cancel"))
        self.model_note.setText(tr("k210p.model_note"))

    def shutdown(self) -> None:
        pass
