"""診斷資訊對話框：說明選單 →「診斷資訊…」。

學生遇到問題時按「匯出成 ZIP」，把檔案傳給老師。收集工作（含自我檢查、列舉相機）放在背景
執行緒，對話框先顯示「收集中…」，不會讓介面卡住。收集本身見 edgelens/diagnostics_report.py。
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QAction, QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
)

from edgelens import diagnostics_report as dr
from edgelens.diagnostics_report import export_zip  # noqa: F401 - 對外也從這裡提供
from edgelens.i18n import tr

log = logging.getLogger(__name__)


class _Collector(QObject):
    done = Signal(object)

    def __init__(self, fn: Callable[[], dict[str, Any]], parent: QObject | None = None):
        super().__init__(parent)
        self._fn = fn

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self._run, name="DiagCollect", daemon=True)
        t.start()
        return t

    def _run(self) -> None:
        try:
            report = self._fn()
        except Exception as e:  # noqa: BLE001
            log.exception("收集診斷資訊失敗")
            report = {"edgelens_version": "?", "error": f"{type(e).__name__}: {e}", "selftest": [], "log_tail": []}
        try:
            self.done.emit(report)
        except RuntimeError:   # 對話框已關閉
            pass


class DiagnosticsDialog(QDialog):
    def __init__(self, parent=None, ctx=None, *, collector: Callable[[], dict[str, Any]] | None = None):
        super().__init__(parent)
        self.ctx = ctx
        self.report: dict[str, Any] | None = None
        self.last_export: Path | None = None
        self._collect_fn = collector or self._default_collect
        self.resize(780, 560)

        lay = QVBoxLayout(self)
        self.status = QLabel()
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.tabs = QTabWidget()
        self.full_text = self._text_edit()
        self.checks = QTableWidget(0, 3)
        self.checks.verticalHeader().hide()
        self.checks.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.checks.horizontalHeader().setStretchLastSection(True)
        self.settings_text = self._text_edit()
        self.log_text = self._text_edit()
        for w in (self.full_text, self.checks, self.settings_text, self.log_text):
            self.tabs.addTab(w, "")
        lay.addWidget(self.tabs, 1)
        row = QHBoxLayout()
        self.refresh_button = QPushButton()
        self.copy_button = QPushButton()
        self.zip_button = QPushButton()
        self.close_button = QPushButton()
        for b in (self.refresh_button, self.copy_button, self.zip_button):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(self.close_button)
        lay.addLayout(row)
        self.refresh_button.clicked.connect(self.refresh)
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        self.zip_button.clicked.connect(self.export_dialog)
        self.close_button.clicked.connect(self.close)
        self._thread: threading.Thread | None = None
        self._collector: _Collector | None = None
        self.retranslate()
        self.refresh()

    @staticmethod
    def _text_edit() -> QPlainTextEdit:
        e = QPlainTextEdit()
        e.setReadOnly(True)
        e.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        return e

    def _default_collect(self) -> dict[str, Any]:
        settings = self.ctx.settings if self.ctx is not None else None
        workspace = self.ctx.workspace if self.ctx is not None else None
        return dr.collect(settings, workspace)

    def retranslate(self) -> None:
        self.setWindowTitle(tr("diag.title"))
        for i, key in enumerate(("diag.tab.report", "diag.tab.selftest", "diag.tab.settings", "diag.tab.log")):
            self.tabs.setTabText(i, tr(key))
        self.checks.setHorizontalHeaderLabels([tr("diag.col.name"), tr("diag.col.result"), tr("diag.col.detail")])
        self.refresh_button.setText(tr("diag.refresh"))
        self.copy_button.setText(tr("diag.copy"))
        self.zip_button.setText(tr("diag.export"))
        self.close_button.setText(tr("common.close"))

    # ---- 收集 -------------------------------------------------------------------
    def refresh(self) -> None:
        self.status.setText(tr("diag.collecting"))
        for b in (self.copy_button, self.zip_button, self.refresh_button):
            b.setEnabled(False)
        self._collector = _Collector(self._collect_fn, self)
        self._collector.done.connect(self._on_report)
        self._thread = self._collector.start()

    def wait(self, timeout: float = 30.0) -> None:
        """等收集的執行緒結束（測試用）。"""
        if self._thread is not None:
            self._thread.join(timeout)

    def _on_report(self, report: dict[str, Any]) -> None:
        self.report = report
        self.status.setText(tr("diag.ready"))
        for b in (self.copy_button, self.zip_button, self.refresh_button):
            b.setEnabled(True)
        self.full_text.setPlainText(dr.to_text(report))
        self.settings_text.setPlainText(json.dumps(report.get("settings"), ensure_ascii=False, indent=2))
        self.log_text.setPlainText("\n".join(report.get("log_tail") or []))
        checks = report.get("selftest") or []
        self.checks.setRowCount(len(checks))
        for i, c in enumerate(checks):
            ok = c.get("ok")
            result = tr("common.ok") if ok else (tr("common.error") if c.get("required", True) else "--")
            for j, text in enumerate((str(c.get("name", "")), result, str(c.get("detail", "")))):
                self.checks.setItem(i, j, QTableWidgetItem(text))
        self.checks.resizeColumnToContents(0)

    # ---- 動作 -------------------------------------------------------------------
    def text(self) -> str:
        return dr.to_text(self.report) if self.report else ""

    def copy_to_clipboard(self) -> None:
        QGuiApplication.clipboard().setText(self.text())
        self.status.setText(tr("diag.copied"))

    def export_dialog(self) -> None:
        default = dr.default_export_path()
        path, _ = QFileDialog.getSaveFileName(self, tr("diag.export"), str(default), "ZIP (*.zip)")
        if path:
            self.export_to(path)

    def export_to(self, path: str | Path) -> Path | None:
        """匯出 ZIP；失敗時在對話框顯示原因（不丟例外）。"""
        try:
            out = export_zip(path, self.report)
        except OSError as e:
            self.status.setText(tr("diag.export_failed", error=str(e)))
            return None
        self.last_export = out
        self.status.setText(tr("diag.exported", path=str(out)))
        return out


def add_menu_action(main, ctx) -> QAction:
    """在「說明」選單加上「診斷資訊…」。呼叫者要在 retranslate 時更新 action 文字。"""
    action = QAction(tr("diag.action"), main)
    action.triggered.connect(lambda: show_diagnostics(main, ctx))
    main.add_menu_action("help", action)
    return action


def show_diagnostics(main, ctx) -> DiagnosticsDialog:
    dlg = DiagnosticsDialog(main, ctx)
    dlg.setModal(False)
    main._diagnostics_dialog = dlg   # 保留參照，避免被回收
    dlg.show()
    return dlg
