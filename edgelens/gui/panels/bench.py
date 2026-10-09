"""效能／錄影面板：即時統計、記錄數據、錄影、長時間測試、參數掃描。

所有「要持續收畫面」的功能（錄影、記錄、長時間測試、掃描）都在開始時接上那台相機的
frameReady，不依賴「作用中相機」，所以測試途中切換分頁或相機也不會漏畫面。
參數掃描用 QTimer 狀態機，不會卡住介面；取消或結束時一定把參數還原。
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from edgelens.core.controls import ControlSpec, ControlType
from edgelens.core.paths import safe_filename
from edgelens.gui.panels.base import Panel
from edgelens.gui.recorder import Recorder, RecordingResult, flatten_board, flatten_data, qimage_to_bgr
from edgelens.gui.stable import StableLabel
from edgelens.gui.workers import FramePacket, SourceHandle
from edgelens.i18n import tr, tr_or
from edgelens.metrics.csvlog import CsvLogger, write_summary_csv
from edgelens.metrics.rolling import RollingStats

log = logging.getLogger(__name__)

SWEEP_TYPES = (ControlType.INT, ControlType.FLOAT, ControlType.ENUM, ControlType.BOOL)


# ---------------------------------------------------------------- 純邏輯（不依賴 GUI）

def percentile(sorted_vals: list[float], q: float) -> float:
    """和 RollingStats 相同的取法（最近的排名），空清單回傳 0。"""
    if not sorted_vals:
        return 0.0
    return sorted_vals[round(q * (len(sorted_vals) - 1))]


class Collector:
    """收集一段時間的 FPS、延遲、處理耗時，以及 result.data 的數值，最後算出摘要。"""

    def __init__(self) -> None:
        self.series: dict[str, list[float]] = defaultdict(list)
        self.frames = 0
        self.t0 = time.monotonic()

    def add_packet(self, packet: FramePacket) -> None:
        s = packet.stats or {}
        self.series["fps"].append(float(s.get("fps", 0.0)))
        self.series["latency_ms"].append(float(s.get("latency_ms", 0.0)))
        self.series["process_ms"].append(float(s.get("process_ms", 0.0)))
        for k, v in flatten_data(packet.result.data).items():
            self.series[k].append(float(v))
        for k, v in flatten_board(packet.frame.meta or {}).items():
            self.series[k].append(float(v))
        self.frames += 1

    def mean(self, name: str) -> float:
        vals = self.series.get(name)
        return statistics.fmean(vals) if vals else 0.0

    def summary_rows(self) -> list[dict[str, Any]]:
        rows = []
        for name, vals in self.series.items():
            s = sorted(vals)
            rows.append({"metric": name, "mean": round(statistics.fmean(s), 3),
                         "p50": round(statistics.median(s), 3), "p95": round(percentile(s, 0.95), 3),
                         "max": round(s[-1], 3), "min": round(s[0], 3), "n": len(s)})
        return rows


def sweep_values(spec: ControlSpec, start: float = 0.0, stop: float = 0.0, count: int = 3,
                 options: list[Any] | None = None) -> list[Any]:
    """依參數型別產生要掃描的值：數值用等間隔（整數去重複），ENUM 用勾選的選項，BOOL 兩個都測。"""
    if spec.type == ControlType.BOOL:
        return [False, True]
    if spec.type == ControlType.ENUM:
        chosen = list(options) if options is not None else spec.option_values
        return [v for v in spec.option_values if v in chosen]
    if spec.type not in (ControlType.INT, ControlType.FLOAT):
        raise ValueError(tr("bench.sweep.bad_type"))
    count = max(1, int(count))
    raw = [start] if count == 1 else [start + (stop - start) * i / (count - 1) for i in range(count)]
    out: list[Any] = []
    for v in raw:
        v = spec.validate(v)
        if v not in out:
            out.append(v)
    return out


def format_bytes(n: int) -> str:
    return f"{n / 1_000_000:.1f} MB" if n >= 1_000_000 else f"{n / 1000:.0f} KB"


def format_seconds(s: float) -> str:
    s = int(s)
    return f"{s // 60:02d}:{s % 60:02d}"


class _FrameTap(QObject):
    """接上某台相機的 frameReady，之後可以乾淨地中斷。"""

    def __init__(self, handle: SourceHandle, callback, parent: QObject | None = None):
        super().__init__(parent)
        self.handle = handle
        self._cb = callback
        handle.frameReady.connect(self._slot)

    def _slot(self, packet: FramePacket) -> None:
        self._cb(packet)

    def close(self) -> None:
        try:
            self.handle.frameReady.disconnect(self._slot)
        except (RuntimeError, TypeError):
            pass


# ---------------------------------------------------------------- 面板

class BenchPanel(Panel):
    panel_id = "bench"
    title_key = "bench.title"
    order = 70

    def __init__(self, ctx, main):
        super().__init__(ctx, main)
        self._handle: SourceHandle | None = None
        self._window = 300
        self._new_stats()
        self._last_packet: FramePacket | None = None
        self._dropped = 0
        # 各種進行中的工作
        self._rec: Recorder | None = None
        self._rec_tap: _FrameTap | None = None
        self._rec_handle: SourceHandle | None = None
        self._log: Recorder | None = None
        self._log_tap: _FrameTap | None = None
        self._long: dict[str, Any] | None = None
        self._sweep: dict[str, Any] | None = None
        self.last_long_csv: Path | None = None
        self.last_sweep_csv: Path | None = None
        self.last_recording: RecordingResult | None = None
        self.last_log_result: RecordingResult | None = None

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
        self._build_stats(lay)
        self._build_record(lay)
        self._build_long(lay)
        self._build_sweep(lay)
        self._build_files(lay)
        lay.addStretch(1)

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        manager = getattr(ctx, "inference", None)   # 推論結果記進錄影／數據 CSV
        if manager is not None:
            manager.resultReady.connect(self._on_inference_result)
        self.retranslate()
        self.on_active_source_changed(ctx.active_source)

    # ---- 建立介面 ---------------------------------------------------------------
    def _build_stats(self, lay: QVBoxLayout) -> None:
        self.stats_box = QGroupBox()
        v = QVBoxLayout(self.stats_box)
        form = QFormLayout()
        self._stat_names: dict[str, QLabel] = {}
        self._stat_values: dict[str, QLabel] = {}
        for key in ("fps", "latency", "process", "dropped", "samples"):
            name, val = QLabel(), QLabel("-")
            val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._stat_names[key], self._stat_values[key] = name, val
            form.addRow(name, val)
        v.addLayout(form)
        row = QHBoxLayout()
        self.window_label = QLabel()
        self.window_spin = QSpinBox()
        self.window_spin.setRange(10, 5000)
        self.window_spin.setValue(self._window)
        self.window_spin.valueChanged.connect(self._on_window_changed)
        self.reset_stats_button = QPushButton()
        self.reset_stats_button.clicked.connect(self._new_stats)
        row.addWidget(self.window_label)
        row.addWidget(self.window_spin)
        row.addWidget(self.reset_stats_button)
        row.addStretch(1)
        v.addLayout(row)
        self.board_title = QLabel()
        self.board_table = QTableWidget(0, 2)
        self.board_table.horizontalHeader().hide()
        self.board_table.verticalHeader().hide()
        self.board_table.horizontalHeader().setStretchLastSection(True)
        self.board_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.board_table.setMaximumHeight(170)
        v.addWidget(self.board_title)
        v.addWidget(self.board_table)
        self.board_title.hide()
        self.board_table.hide()
        lay.addWidget(self.stats_box)

    def _build_record(self, lay: QVBoxLayout) -> None:
        self.rec_box = QGroupBox()
        v = QVBoxLayout(self.rec_box)
        row = QHBoxLayout()
        self.log_button = QPushButton()
        self.log_button.setCheckable(False)
        self.log_button.clicked.connect(self.toggle_log)
        row.addWidget(self.log_button)
        self.log_label = QLabel()
        row.addWidget(self.log_label, 1)
        v.addLayout(row)
        row = QHBoxLayout()
        self.mode_combo = QComboBox()
        for mode in ("raw", "processed", "annotated"):
            self.mode_combo.addItem("", mode)
        self.mode_combo.setCurrentIndex(1)
        self.record_button = QPushButton()
        self.record_button.clicked.connect(self.toggle_record)
        row.addWidget(self.mode_combo)
        row.addWidget(self.record_button)
        v.addLayout(row)
        self.record_label = StableLabel(2)   # 錄影中每 0.1 秒更新時間與列數
        v.addWidget(self.record_label)
        lay.addWidget(self.rec_box)

    def _build_long(self, lay: QVBoxLayout) -> None:
        self.long_box = QGroupBox()
        v = QVBoxLayout(self.long_box)
        row = QHBoxLayout()
        self.long_label = QLabel()
        self.long_seconds = QSpinBox()
        self.long_seconds.setRange(1, 24 * 3600)
        self.long_seconds.setValue(60)
        self.long_button = QPushButton()
        self.long_button.clicked.connect(self.toggle_long)
        row.addWidget(self.long_label)
        row.addWidget(self.long_seconds)
        row.addWidget(self.long_button)
        row.addStretch(1)
        v.addLayout(row)
        self.long_progress = QProgressBar()
        self.long_progress.setRange(0, 100)
        self.long_progress.hide()
        v.addWidget(self.long_progress)
        self.long_status = StableLabel(2)
        v.addWidget(self.long_status)
        self.long_table = self._result_table()
        v.addWidget(self.long_table)
        lay.addWidget(self.long_box)

    def _build_sweep(self, lay: QVBoxLayout) -> None:
        self.sweep_box = QGroupBox()
        v = QVBoxLayout(self.sweep_box)
        form = QFormLayout()
        self.sweep_param_label, self.sweep_from_label = QLabel(), QLabel()
        self.sweep_to_label, self.sweep_count_label, self.sweep_dwell_label = QLabel(), QLabel(), QLabel()
        self.sweep_param = QComboBox()
        self.sweep_param.currentIndexChanged.connect(self._on_sweep_param_changed)
        self.sweep_from = QDoubleSpinBox()
        self.sweep_to = QDoubleSpinBox()
        for w in (self.sweep_from, self.sweep_to):
            w.setRange(-1e9, 1e9)
            w.setDecimals(3)
        self.sweep_count = QSpinBox()
        self.sweep_count.setRange(1, 50)
        self.sweep_count.setValue(5)
        self.sweep_dwell = QDoubleSpinBox()
        self.sweep_dwell.setRange(0.2, 600)
        self.sweep_dwell.setValue(3.0)
        self.sweep_dwell.setSingleStep(0.5)
        self.sweep_options = QListWidget()
        self.sweep_options.setMaximumHeight(110)
        form.addRow(self.sweep_param_label, self.sweep_param)
        form.addRow(self.sweep_from_label, self.sweep_from)
        form.addRow(self.sweep_to_label, self.sweep_to)
        form.addRow(self.sweep_count_label, self.sweep_count)
        form.addRow(self.sweep_options)
        form.addRow(self.sweep_dwell_label, self.sweep_dwell)
        v.addLayout(form)
        row = QHBoxLayout()
        self.sweep_button = QPushButton()
        self.sweep_button.clicked.connect(self.toggle_sweep)
        row.addWidget(self.sweep_button)
        row.addStretch(1)
        v.addLayout(row)
        self.sweep_progress = QProgressBar()
        self.sweep_progress.setRange(0, 100)
        self.sweep_progress.hide()
        v.addWidget(self.sweep_progress)
        self.sweep_status = StableLabel(2)
        v.addWidget(self.sweep_status)
        self.sweep_table = self._result_table()
        v.addWidget(self.sweep_table)
        lay.addWidget(self.sweep_box)

    def _build_files(self, lay: QVBoxLayout) -> None:
        self.files_box = QGroupBox()
        v = QHBoxLayout(self.files_box)
        self.open_logs_button = QPushButton()
        self.open_logs_button.clicked.connect(lambda: self._open_folder(self.logs_dir()))
        self.open_rec_button = QPushButton()
        self.open_rec_button.clicked.connect(lambda: self._open_folder(self.recordings_dir()))
        v.addWidget(self.open_logs_button)
        v.addWidget(self.open_rec_button)
        lay.addWidget(self.files_box)

    @staticmethod
    def _result_table() -> QTableWidget:
        t = QTableWidget(0, 0)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        t.verticalHeader().hide()
        t.setMaximumHeight(200)
        t.hide()
        return t

    # ---- 資料夾 -----------------------------------------------------------------
    def _base_dir(self, sub: str) -> Path:
        project = self.ctx.project
        if project is not None:
            return project.dir(sub)
        from edgelens.core.paths import ensure_dir

        return ensure_dir(Path(self.ctx.workspace.root) / sub)

    def logs_dir(self) -> Path:
        return self._base_dir("logs")

    def recordings_dir(self) -> Path:
        return self._base_dir("recordings")

    def _open_folder(self, path: Path) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    # ---- 作用中相機 -------------------------------------------------------------
    def on_active_source_changed(self, handle: SourceHandle | None) -> None:
        old = self._handle
        if old is not None:
            try:
                old.opened.disconnect(self._on_opened)
            except (RuntimeError, TypeError):
                pass
        self._handle = handle
        if handle is not None:
            handle.opened.connect(self._on_opened)   # 相機打開後才知道有哪些參數
        self._new_stats()
        self._last_packet = None
        self._fill_sweep_params()
        self._update_enabled()

    def _on_opened(self, _info: dict) -> None:
        if self._sweep is None:
            self._fill_sweep_params()

    def _new_stats(self) -> None:
        w = self._window
        self.fps_stats, self.lat_stats, self.proc_stats = RollingStats(w), RollingStats(w), RollingStats(w)
        self._dropped = 0

    def _on_window_changed(self, value: int) -> None:
        self._window = int(value)
        self._new_stats()

    def on_frame(self, packet: FramePacket) -> None:
        s = packet.stats or {}
        now = time.monotonic()
        try:
            self.fps_stats.add(float(s.get("fps", 0.0)), now)
            self.lat_stats.add(float(s.get("latency_ms", 0.0)), now)
            self.proc_stats.add(float(s.get("process_ms", 0.0)), now)
        except ValueError:
            return
        self._dropped = int(s.get("dropped", 0))
        self._last_packet = packet

    def _update_enabled(self) -> None:
        has = self._handle is not None
        self.empty_label.setVisible(not has)
        self.log_button.setEnabled(has or self._log is not None)
        self.record_button.setEnabled(has or self._rec is not None)
        self.mode_combo.setEnabled(self._rec is None)
        self.long_button.setEnabled(has or self._long is not None)
        self.long_seconds.setEnabled(self._long is None)
        self.sweep_button.setEnabled((has and self.sweep_param.count() > 0) or self._sweep is not None)
        for w in (self.sweep_param, self.sweep_from, self.sweep_to, self.sweep_count, self.sweep_dwell,
                  self.sweep_options):
            w.setEnabled(self._sweep is None)

    # ---- 計時器：更新數字、推進長時間測試與掃描 ----------------------------------
    def _tick(self) -> None:
        self._refresh_stats()
        self._refresh_recording()
        if self._long is not None:
            self._tick_long()
        if self._sweep is not None:
            self._tick_sweep()

    def _refresh_stats(self) -> None:
        v = self._stat_values
        f, lat, proc = self.fps_stats.summary(), self.lat_stats.summary(), self.proc_stats.summary()
        if self._handle is None or f is None or lat is None or proc is None:
            for lab in v.values():
                lab.setText("-")
            self._fill_board(None)
            return
        v["fps"].setText(f"{f['mean']:.1f}")
        v["latency"].setText(tr("bench.stat.mean_p95", mean=f"{lat['mean']:.1f}", p95=f"{lat['p95']:.1f}"))
        v["process"].setText(tr("bench.stat.mean_p95", mean=f"{proc['mean']:.2f}", p95=f"{proc['p95']:.2f}"))
        v["dropped"].setText(str(self._dropped))
        v["samples"].setText(str(f["n"]))
        self._fill_board(self._last_packet)

    def _fill_board(self, packet: FramePacket | None) -> None:
        rows: list[tuple[str, str]] = []
        meta = packet.frame.meta if packet is not None else {}
        board = flatten_board(meta or {})
        if board:
            labels = {"board.fps": tr("bench.board.fps"), "board.lat_ms": tr("bench.board.lat"),
                      "board.mem": tr("bench.board.mem"), "board.heap": tr("bench.board.heap")}
            for k, val in board.items():
                if k.startswith("board.us."):
                    step = k[len("board.us."):-3]
                    rows.append((tr("bench.board.step", step=step), f"{val:.2f} ms"))
                elif k in ("board.mem", "board.heap"):
                    rows.append((labels[k], format_bytes(int(val))))
                elif k == "board.lat_ms":
                    rows.append((labels[k], f"{val:.1f} ms"))
                else:
                    rows.append((labels.get(k, k), f"{val:.1f}"))
        show = bool(rows)
        self.board_title.setVisible(show)
        self.board_table.setVisible(show)
        if not show:
            self.board_table.setRowCount(0)
            return
        self.board_table.setRowCount(len(rows))
        for i, (a, b) in enumerate(rows):
            self.board_table.setItem(i, 0, QTableWidgetItem(a))
            self.board_table.setItem(i, 1, QTableWidgetItem(b))
        self.board_table.resizeColumnToContents(0)
        self.board_table.setMaximumHeight(min(170, 28 * len(rows) + 6))

    # ---- 記錄數據 ---------------------------------------------------------------
    def toggle_log(self) -> None:
        if self._log is not None:
            self.stop_log()
        else:
            self.start_log()

    def start_log(self) -> bool:
        h = self._handle
        if h is None or self._log is not None:
            return False
        try:
            self._log = Recorder(self.logs_dir(), safe_filename(h.info.name), mode="data", suffix="_stats")
            self._log.start()
        except OSError as e:
            self._log = None
            self.log_label.setText(tr("bench.err.cannot_write", error=str(e)))
            return False
        self._log_tap = _FrameTap(h, self._on_log_frame, self)
        self.retranslate()
        return True

    def stop_log(self) -> RecordingResult | None:
        if self._log is None:
            return None
        if self._log_tap is not None:
            self._log_tap.close()
            self._log_tap = None
        res = self._log.stop()
        self._log = None
        self.last_log_result = res
        self.log_label.setText(tr("bench.log.saved", path=str(res.csv_path), rows=res.frames))
        self.ctx.show_message(tr("bench.log.saved", path=str(res.csv_path), rows=res.frames))
        self.retranslate()
        return res

    def _on_log_frame(self, packet: FramePacket) -> None:
        if self._log is not None:
            self._log.add_packet(packet)

    # ---- 錄影 -------------------------------------------------------------------
    def toggle_record(self) -> None:
        if self._rec is not None:
            self.stop_record()
        else:
            self.start_record()

    def start_record(self) -> bool:
        h = self._handle
        if h is None or self._rec is not None:
            return False
        mode = self.mode_combo.currentData()
        fps = self.fps_stats.summary()
        rate = round(fps["mean"]) if fps and fps["mean"] >= 1 else 30
        try:
            self._rec = Recorder(self.recordings_dir(), safe_filename(h.info.name), mode=mode, fps=rate)
            self._rec.start()
        except OSError as e:
            self._rec = None
            self.record_label.setText(tr("bench.err.cannot_write", error=str(e)))
            return False
        self._rec_handle = h
        self._rec_tap = _FrameTap(h, self._on_record_frame, self)
        self.retranslate()
        return True

    def stop_record(self) -> RecordingResult | None:
        if self._rec is None:
            return None
        if self._rec_tap is not None:
            self._rec_tap.close()
            self._rec_tap = None
        res = self._rec.stop()
        self._rec = None
        self._rec_handle = None
        self.last_recording = res
        if res.video_path is not None:
            text = tr("bench.rec.saved", path=str(res.video_path), frames=res.frames, dropped=res.dropped,
                      size=format_bytes(res.size_bytes))
        else:
            text = tr("bench.rec.failed", error=res.error or tr("common.unknown"))
        if res.video_path is not None and res.error:
            text += "\n" + res.error
        self.record_label.setText(text)
        self.ctx.show_message(text.splitlines()[0])
        self.retranslate()
        return res

    def _on_record_frame(self, packet: FramePacket) -> None:
        rec = self._rec
        if rec is None:
            return
        image = None
        if rec.mode == "annotated" and self._rec_handle is not None:
            view = self.main.video_view_for(self._rec_handle)
            if view is not None:
                image = qimage_to_bgr(view.render_annotated())   # render_annotated 只能在 GUI 執行緒
        rec.add_packet(packet, image)

    def _on_inference_result(self, handle: Any, _session: Any, result: Any) -> None:
        self.add_inference(handle, result.frame_seq, result)

    def add_inference(self, handle: SourceHandle | None, seq: int, result: Any) -> None:
        """推論面板每算完一張就呼叫，結果會記進錄影／數據 CSV（handle 是該推論所屬的相機）。"""
        for rec, h in ((self._rec, self._rec_handle), (self._log, self._log_tap.handle if self._log_tap else None)):
            if rec is not None and (handle is None or h is handle):
                rec.add_inference(seq, result)

    def _refresh_recording(self) -> None:
        if self._rec is not None:
            r = self._rec
            self.record_label.setText(tr("bench.rec.running", time=format_seconds(r.elapsed),
                                         size=format_bytes(r.size_bytes), frames=r.frames, dropped=r.dropped))
        if self._log is not None:
            r = self._log
            self.log_label.setText(tr("bench.log.running", time=format_seconds(r.elapsed), rows=r.frames))

    # ---- 長時間測試 -------------------------------------------------------------
    def toggle_long(self) -> None:
        if self._long is not None:
            self.stop_long(cancelled=True)
        else:
            self.start_long()

    def start_long(self, seconds: float | None = None) -> bool:
        h = self._handle
        if h is None or self._long is not None:
            return False
        secs = float(seconds if seconds is not None else self.long_seconds.value())
        col = Collector()
        tap = _FrameTap(h, col.add_packet, self)
        self._long = {"handle": h, "collector": col, "tap": tap, "t0": time.monotonic(), "seconds": secs}
        self.long_progress.setValue(0)
        self.long_progress.show()
        self.long_table.hide()
        self.long_status.setText(tr("bench.long.running"))
        self.retranslate()
        return True

    def _tick_long(self) -> None:
        job = self._long
        if job is None:
            return
        elapsed = time.monotonic() - job["t0"]
        self.long_progress.setValue(int(min(1.0, elapsed / job["seconds"]) * 100))
        self.long_status.setText(tr("bench.long.progress", done=format_seconds(elapsed),
                                    total=format_seconds(job["seconds"]), frames=job["collector"].frames))
        if elapsed >= job["seconds"]:
            self.stop_long(cancelled=False)

    def stop_long(self, cancelled: bool) -> None:
        job, self._long = self._long, None
        if job is None:
            return
        job["tap"].close()
        self.long_progress.hide()
        col: Collector = job["collector"]
        rows = col.summary_rows()
        if cancelled:
            self.long_status.setText(tr("bench.long.cancelled"))
        elif not rows:
            self.long_status.setText(tr("bench.long.no_frames"))
        else:
            name = safe_filename(job["handle"].info.name)
            try:
                path = write_summary_csv(self.logs_dir() / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}_longtest.csv",
                                         rows, ["metric", "mean", "p50", "p95", "max", "min", "n"])
                self.last_long_csv = path
                self.long_status.setText(tr("bench.long.done", frames=col.frames, path=str(path)))
            except OSError as e:
                self.long_status.setText(tr("bench.err.cannot_write", error=str(e)))
        if rows and not cancelled:
            self._fill_table(self.long_table, rows, ["metric", "mean", "p50", "p95", "max", "min", "n"])
        self.retranslate()

    # ---- 參數掃描 ---------------------------------------------------------------
    def _fill_sweep_params(self) -> None:
        self.sweep_param.blockSignals(True)
        self.sweep_param.clear()
        h = self._handle
        if h is not None:
            for spec in h.controls:
                if spec.type in SWEEP_TYPES and not spec.extra.get("secret"):
                    self.sweep_param.addItem(tr_or(spec.label_key, spec.label or spec.key), spec.key)
        self.sweep_param.blockSignals(False)
        self._on_sweep_param_changed()

    def _current_spec(self) -> ControlSpec | None:
        h, key = self._handle, self.sweep_param.currentData()
        if h is None or key is None:
            return None
        return next((s for s in h.controls if s.key == key), None)

    def _on_sweep_param_changed(self, _i: int = 0) -> None:
        spec = self._current_spec()
        numeric = spec is not None and spec.type in (ControlType.INT, ControlType.FLOAT)
        for w in (self.sweep_from, self.sweep_to, self.sweep_count, self.sweep_from_label, self.sweep_to_label,
                  self.sweep_count_label):
            w.setVisible(numeric)
        self.sweep_options.setVisible(spec is not None and spec.type == ControlType.ENUM)
        self.sweep_options.clear()
        if spec is not None and spec.type == ControlType.ENUM:
            for value, label in spec.options:
                item = QListWidgetItem(tr_or(label, str(value)))
                item.setData(Qt.ItemDataRole.UserRole, value)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                self.sweep_options.addItem(item)
        if numeric and spec is not None:
            lo = float(spec.min) if spec.min is not None else 0.0
            hi = float(spec.max) if spec.max is not None else lo + 100.0
            self.sweep_from.setValue(lo)
            self.sweep_to.setValue(hi)
            self.sweep_count.setValue(5)
        self._update_enabled()

    def _checked_options(self) -> list[Any]:
        out = []
        for i in range(self.sweep_options.count()):
            item = self.sweep_options.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(item.data(Qt.ItemDataRole.UserRole))
        return out

    def toggle_sweep(self) -> None:
        if self._sweep is not None:
            self.stop_sweep(cancelled=True)
        else:
            self.start_sweep()

    def start_sweep(self, values: list[Any] | None = None, key: str | None = None,
                    dwell: float | None = None) -> bool:
        """開始掃描。參數（values、key、dwell）可由程式指定，沒給就用畫面上的設定。"""
        h = self._handle
        if h is None or self._sweep is not None:
            return False
        if key is not None:
            idx = self.sweep_param.findData(key)
            if idx >= 0:
                self.sweep_param.setCurrentIndex(idx)
        spec = self._current_spec()
        if spec is None:
            return False
        if values is None:
            values = sweep_values(spec, self.sweep_from.value(), self.sweep_to.value(), self.sweep_count.value(),
                                  self._checked_options())
        if not values:
            self.sweep_status.setText(tr("bench.sweep.no_values"))
            return False
        dwell = float(dwell if dwell is not None else self.sweep_dwell.value())
        name = safe_filename(h.info.name)
        path = self.logs_dir() / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}_sweep_{safe_filename(spec.key)}.csv"
        try:
            logger = CsvLogger(path, ["param", "value", "frames", "fps_mean", "latency_ms_mean", "latency_ms_p95",
                                      "process_ms_mean"], flush_interval=0)
        except OSError as e:
            self.sweep_status.setText(tr("bench.err.cannot_write", error=str(e)))
            return False
        orig = h.values.get(spec.key, spec.default_value())
        job: dict[str, Any] = {
            "handle": h, "spec": spec, "values": list(values), "idx": -1, "orig": orig, "dwell": dwell,
            "settle": min(0.5, dwell * 0.3), "collector": None, "t_value": 0.0, "rows": [], "logger": logger,
            "path": path, "tap": _FrameTap(h, self._on_sweep_frame, self)}
        self._sweep = job
        self.last_sweep_csv = path
        self.sweep_table.hide()
        self.sweep_progress.setValue(0)
        self.sweep_progress.show()
        self._sweep_next()
        self.retranslate()
        return True

    def _sweep_next(self) -> None:
        job = self._sweep
        if job is None:
            return
        job["idx"] += 1
        if job["idx"] >= len(job["values"]):
            self.stop_sweep(cancelled=False)
            return
        value = job["values"][job["idx"]]
        h: SourceHandle = job["handle"]
        h.set_control(job["spec"].key, value)
        h.flush_controls()
        job["collector"] = Collector()
        job["t_value"] = time.monotonic()
        self.sweep_status.setText(tr("bench.sweep.running", key=job["spec"].key, value=str(value),
                                     i=job["idx"] + 1, n=len(job["values"])))
        self.sweep_progress.setValue(int(100 * job["idx"] / len(job["values"])))

    def _on_sweep_frame(self, packet: FramePacket) -> None:
        job = self._sweep
        if job is None or job["collector"] is None:
            return
        if time.monotonic() - job["t_value"] >= job["settle"]:   # 剛改完參數的前幾張不算
            job["collector"].add_packet(packet)

    def _tick_sweep(self) -> None:
        job = self._sweep
        if job is None or job["collector"] is None:
            return
        if time.monotonic() - job["t_value"] < job["dwell"]:
            return
        col: Collector = job["collector"]
        lat = sorted(col.series.get("latency_ms", []))
        row: dict[str, Any] = {
            "param": job["spec"].key, "value": job["values"][job["idx"]], "frames": col.frames,
            "fps_mean": round(col.mean("fps"), 3), "latency_ms_mean": round(col.mean("latency_ms"), 3),
            "latency_ms_p95": round(percentile(lat, 0.95), 3), "process_ms_mean": round(col.mean("process_ms"), 3)}
        for name in col.series:
            if name.startswith("data.") or name.startswith("board."):
                row[name + "_mean"] = round(col.mean(name), 3)
        job["rows"].append(row)
        job["logger"].add(row)
        self._sweep_next()

    def stop_sweep(self, cancelled: bool) -> None:
        job, self._sweep = self._sweep, None
        if job is None:
            return
        job["tap"].close()
        h: SourceHandle = job["handle"]
        h.set_control(job["spec"].key, job["orig"])   # 一定還原原本的值
        h.flush_controls()
        job["logger"].close()
        self.sweep_progress.hide()
        rows = job["rows"]
        fields: list[str] = []
        for r in rows:
            fields += [k for k in r if k not in fields]
        if rows:
            self._fill_table(self.sweep_table, rows, fields)
        key = job["spec"].key
        if cancelled:
            self.sweep_status.setText(tr("bench.sweep.cancelled", key=key, value=str(job["orig"])))
        else:
            self.sweep_status.setText(tr("bench.sweep.done", key=key, value=str(job["orig"]), path=str(job["path"])))
        self.ctx.save_settings_later()
        self.retranslate()

    # ---- 共用 -------------------------------------------------------------------
    @staticmethod
    def _fill_table(table: QTableWidget, rows: list[dict[str, Any]], fields: list[str]) -> None:
        table.setColumnCount(len(fields))
        table.setHorizontalHeaderLabels(fields)
        table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, f in enumerate(fields):
                v = r.get(f, "")
                table.setItem(i, j, QTableWidgetItem(f"{v:g}" if isinstance(v, float) else str(v)))
        table.resizeColumnsToContents()
        table.setMaximumHeight(min(200, 30 * len(rows) + 34))
        table.show()

    # ---- 語言與結束 -------------------------------------------------------------
    def retranslate(self) -> None:
        self.empty_label.setText(tr("bench.empty"))
        self.stats_box.setTitle(tr("bench.stats.title"))
        for key, lab in self._stat_names.items():
            lab.setText(tr(f"bench.stat.{key}"))
        self.window_label.setText(tr("bench.stat.window"))
        self.reset_stats_button.setText(tr("common.reset"))
        self.board_title.setText(tr("bench.board.title"))
        self.rec_box.setTitle(tr("bench.rec.title"))
        self.log_button.setText(tr("bench.log.stop") if self._log is not None else tr("bench.log.start"))
        self.record_button.setText(tr("bench.rec.stop") if self._rec is not None else tr("bench.rec.start"))
        for i, mode in enumerate(("raw", "processed", "annotated")):
            self.mode_combo.setItemText(i, tr(f"bench.rec.mode.{mode}"))
        self.long_box.setTitle(tr("bench.long.title"))
        self.long_label.setText(tr("bench.long.seconds"))
        self.long_button.setText(tr("common.cancel") if self._long is not None else tr("bench.long.start"))
        self.sweep_box.setTitle(tr("bench.sweep.title"))
        self.sweep_param_label.setText(tr("bench.sweep.param"))
        self.sweep_from_label.setText(tr("bench.sweep.from"))
        self.sweep_to_label.setText(tr("bench.sweep.to"))
        self.sweep_count_label.setText(tr("bench.sweep.count"))
        self.sweep_dwell_label.setText(tr("bench.sweep.dwell"))
        self.sweep_button.setText(tr("common.cancel") if self._sweep is not None else tr("bench.sweep.start"))
        self.files_box.setTitle(tr("bench.files.title"))
        self.open_logs_button.setText(tr("bench.files.open_logs"))
        self.open_rec_button.setText(tr("bench.files.open_rec"))
        self._update_enabled()

    def shutdown(self) -> None:
        """關閉前停掉所有進行中的工作並還原參數，錄影檔案一定要正確關閉。"""
        self._timer.stop()
        manager = getattr(self.ctx, "inference", None)
        if manager is not None:
            try:
                manager.resultReady.disconnect(self._on_inference_result)
            except (RuntimeError, TypeError):
                pass
        if self._sweep is not None:
            self.stop_sweep(cancelled=True)
        if self._long is not None:
            self.stop_long(cancelled=True)
        if self._rec is not None:
            self.stop_record()
        if self._log is not None:
            self.stop_log()
