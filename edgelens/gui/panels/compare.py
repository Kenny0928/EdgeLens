"""比較面板：把每台相機、每個模型的數字放在同一張表，方便比較哪個組合比較好。

每列是「一台相機 × 一個模型」，沒有模型的相機也會有一列（只看相機本身的 FPS 與畫質）。
表格每 0.5 秒更新；清晰度與亮度每台相機每秒才算一次（要縮圖再算，不必每張都算）。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from edgelens.core.results import InferenceResult
from edgelens.gui.panels.base import Panel
from edgelens.i18n import tr
from edgelens.metrics.csvlog import write_summary_csv
from edgelens.metrics.detection_rate import DetectionRate
from edgelens.metrics.quality import measure

log = logging.getLogger(__name__)

COLUMNS = ("camera", "model", "cam_fps", "latency", "infer_fps", "infer_ms", "sharpness", "brightness",
           "best", "rate")
METRICS = COLUMNS[2:]   # 表格的列；camera、model 放在欄標題
RATE_WINDOW = 10.0       # 偵測率用最近 10 秒計算
QUALITY_INTERVAL = 1.0
REFRESH_MS = 500


class ComparePanel(Panel):
    panel_id = "compare"
    title_key = "gui.panel.compare.title"
    order = 60

    def __init__(self, ctx, main):
        super().__init__(ctx, main)
        self._rates: dict[int, DetectionRate] = {}
        self._quality: dict[Any, tuple[float, Any]] = {}
        self.last_csv: Path | None = None

        lay = QVBoxLayout(self)
        self.help_label = QLabel()
        self.help_label.setWordWrap(True)
        lay.addWidget(self.help_label)
        row = QHBoxLayout()
        self.grid_button = QPushButton()
        self.grid_button.clicked.connect(lambda: self.main.set_layout_mode("grid"))
        self.reset_button = QPushButton()
        self.reset_button.clicked.connect(self.reset_stats)
        self.export_button = QPushButton()
        self.export_button.clicked.connect(self.export_csv)
        for b in (self.grid_button, self.reset_button, self.export_button):
            row.addWidget(b)
        lay.addLayout(row)
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        # 指標放在列、每組（相機 × 模型）一欄：最多四台相機，側邊面板不用左右捲動
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectColumns)
        self.table.setRowCount(len(METRICS))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        lay.addWidget(self.table, 1)
        self.fair_label = QLabel()
        self.fair_label.setWordWrap(True)
        lay.addWidget(self.fair_label)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        lay.addWidget(self.status_label)

        manager = ctx.inference
        manager.resultReady.connect(self._on_result)
        manager.sessionsChanged.connect(self._on_sessions_changed)
        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.retranslate()
        self.refresh()

    # ---- 資料 -------------------------------------------------------------------
    def _on_result(self, handle, session, result: InferenceResult) -> None:
        rate = self._rates.get(session.session_id)
        if rate is None:
            rate = self._rates[session.session_id] = DetectionRate(RATE_WINDOW)
        rate.add(result)

    def _on_sessions_changed(self) -> None:
        alive = {s.session_id for s in self.ctx.inference.sessions()}
        for sid in [k for k in self._rates if k not in alive]:
            del self._rates[sid]

    def _quality_of(self, handle) -> Any:
        pkt = handle.last_packet
        if pkt is None:
            return None
        now = time.monotonic()
        cached = self._quality.get(handle)
        if cached is not None and now - cached[0] < QUALITY_INTERVAL:
            return cached[1]
        try:
            report = measure(pkt.frame.image)
        except Exception:  # noqa: BLE001 - 畫面格式怪異時只是這欄空白
            report = None
        self._quality[handle] = (now, report)
        return report

    def rows_data(self) -> list[dict[str, Any]]:
        """每列一個（相機 × 模型）；數值欄位是數字或 None（還沒有資料）。"""
        rows = []
        handles = self.main.handles()
        self._quality = {h: v for h, v in self._quality.items() if h in handles}   # 已關閉的相機不留著
        for handle in handles:
            pkt = handle.last_packet
            st = pkt.stats if pkt is not None else {}
            q = self._quality_of(handle)
            sessions = self.ctx.inference.sessions_for(handle) or [None]
            for s in sessions:
                row: dict[str, Any] = {
                    "camera": handle.info.name,
                    "model": s.card.name if s is not None else "",
                    "cam_fps": st.get("fps") if pkt is not None else None,
                    "latency": st.get("latency_ms") if pkt is not None else None,
                    "infer_fps": None, "infer_ms": None,
                    "sharpness": q.sharpness if q is not None else None,
                    "brightness": q.brightness if q is not None else None,
                    "best": "", "rate": None,
                    "_session": s,
                }
                if s is not None and s.last_result is not None:
                    stats = s.stats
                    row["infer_fps"], row["infer_ms"] = stats["infer_fps"], stats["infer_ms"]
                    best = s.last_result.best
                    if best is not None:
                        row["best"] = f"{best.score:.2f} ({best.label or best.class_id})"
                    rate = self._rates.get(s.session_id)
                    if rate is not None:
                        percent = rate.rate()[0]
                        row["rate"] = percent
                rows.append(row)
        return rows

    @staticmethod
    def _fmt(key: str, value: Any) -> str:
        if value is None or value == "":
            return "-"
        if isinstance(value, str):
            return value
        if key == "rate":
            return f"{value:.0f}%"
        if key in ("latency", "infer_ms"):
            return f"{value:.0f} ms" if key == "latency" else f"{value:.1f} ms"
        return f"{value:.1f}"

    def refresh(self) -> None:
        entries = self.rows_data()
        self.table.setColumnCount(len(entries))
        for c, entry in enumerate(entries):
            header = self.table.horizontalHeaderItem(c)
            if header is None:
                header = QTableWidgetItem()
                self.table.setHorizontalHeaderItem(c, header)
            header.setText(f"{entry['camera']}\n{entry['model']}")
            for r, key in enumerate(METRICS):
                item = self.table.item(r, c)
                if item is None:
                    item = QTableWidgetItem()
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    self.table.setItem(r, c, item)
                item.setText(self._fmt(key, entry[key]))

    # ---- 動作 -------------------------------------------------------------------
    def reset_stats(self) -> None:
        for r in self._rates.values():
            r.reset()
        for s in self.ctx.inference.sessions():
            s.reset_stats()
        self._quality.clear()
        self.refresh()
        self.status_label.setText(tr("gui.compare.reset_done"))

    def export_csv(self, path: str | Path | None = None) -> Path | None:
        """匯出比較表（UTF-8 BOM，Excel 直接開得了中文）；沒指定路徑就存到專案的 logs/。"""
        rows = self.rows_data()
        if not rows:
            self.status_label.setText(tr("gui.compare.no_rows"))
            return None
        if not isinstance(path, (str, Path)):    # clicked 訊號會傳 bool
            if self.ctx.project is None:
                self.status_label.setText(tr("gui.compare.need_project"))
                return None
            path = self.ctx.project.dir("logs") / time.strftime("compare_%Y%m%d_%H%M%S.csv")
        names = [tr(f"gui.compare.col.{k}") for k in COLUMNS]
        out = []
        for row in rows:
            rec = {}
            for k, name in zip(COLUMNS, names, strict=True):
                v = row[k]
                rec[name] = round(v, 2) if isinstance(v, float) else ("" if v is None else v)
            out.append(rec)
        try:
            saved = write_summary_csv(path, out, names)
        except OSError as e:
            self.status_label.setText(tr("gui.compare.export_failed", error=str(e)[:200]))
            return None
        self.last_csv = saved
        self.status_label.setText(tr("gui.compare.exported", path=str(saved)))
        self.ctx.show_message(tr("gui.compare.exported", path=str(saved)))
        return saved

    # ---- 其他 -------------------------------------------------------------------
    def on_project_changed(self, project: Any) -> None:
        self.status_label.setText("")

    def retranslate(self) -> None:
        self.help_label.setText(tr("gui.compare.help"))
        self.grid_button.setText(tr("gui.compare.grid"))
        self.reset_button.setText(tr("gui.compare.reset"))
        self.export_button.setText(tr("gui.compare.export"))
        self.table.setVerticalHeaderLabels([tr(f"gui.compare.col.{k}") for k in METRICS])
        self.fair_label.setText(tr("gui.compare.fair"))

    def shutdown(self) -> None:
        self._timer.stop()
