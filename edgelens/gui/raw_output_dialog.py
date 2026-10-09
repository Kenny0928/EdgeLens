"""原始輸出檢視器：看模型最後吐出來的數字，幫學生自己寫模型說明檔（card）。

不是模態對話框：學生可以一邊看數字、一邊調整畫面或門檻。資料由 InferenceSession.request_raw()
在下一張畫面時取得（rawReady），取得前顯示「等待中」。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from edgelens.i18n import tr
from edgelens.inference.autodetect import raw_summary
from edgelens.inference.base import TensorInfo

MAX_ROWS = 100
MAX_COLS = 24


def table_view_of(arr: np.ndarray) -> np.ndarray:
    """任意維度的陣列 → 2 維（把前面的維度攤平成「列」，最後一軸是「欄」），並只留前幾列幾欄。"""
    a = np.asarray(arr)
    if a.ndim == 0:
        a = a.reshape(1, 1)
    elif a.ndim == 1:
        a = a.reshape(1, -1)
    else:
        a = a.reshape(-1, a.shape[-1])
    return a[:MAX_ROWS, :MAX_COLS]


class RawOutputDialog(QDialog):
    def __init__(self, session, parent: QWidget | None = None):
        super().__init__(parent)
        self.session = session
        self._outputs: list[tuple[TensorInfo, np.ndarray]] = []
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.resize(760, 560)
        lay = QVBoxLayout(self)
        self.intro = QLabel()
        self.intro.setWordWrap(True)
        lay.addWidget(self.intro)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(130)
        lay.addWidget(self.summary)
        row = QHBoxLayout()
        self.pick_label = QLabel()
        self.picker = QComboBox()
        self.picker.currentIndexChanged.connect(self._show_table)
        self.refresh_button = QPushButton()
        self.refresh_button.clicked.connect(self.refresh)
        row.addWidget(self.pick_label)
        row.addWidget(self.picker, 1)
        row.addWidget(self.refresh_button)
        lay.addLayout(row)
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self.table, 1)
        self.note = QLabel()
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        session.rawReady.connect(self.set_outputs)
        self.retranslate()
        if session.info.mode not in ("tensor", ""):
            self.note.setText(tr("gui.rawout.unsupported"))
            self.refresh_button.setEnabled(False)
        elif session.raw_outputs:
            self.set_outputs(session.raw_outputs)
        else:
            self.refresh()

    def retranslate(self) -> None:
        self.setWindowTitle(tr("gui.rawout.title"))
        self.intro.setText(tr("gui.rawout.intro"))
        self.pick_label.setText(tr("gui.rawout.pick"))
        self.refresh_button.setText(tr("gui.rawout.refresh"))

    def refresh(self) -> None:
        """重新抓一次：下一張畫面進來時，session 會順便取原始輸出。"""
        self.note.setText(tr("gui.rawout.waiting"))
        self.session.request_raw()

    def set_outputs(self, outputs: list) -> None:
        self._outputs = list(outputs)
        if not self._outputs:
            self.note.setText(tr("gui.rawout.empty"))
            self.summary.setPlainText("")
            self.picker.clear()
            self.table.clear()
            return
        infos = [i for i, _ in self._outputs]
        self.summary.setPlainText("\n".join(raw_summary(infos, [a for _, a in self._outputs])))
        self.picker.blockSignals(True)
        keep = self.picker.currentIndex()
        self.picker.clear()
        for i, (info, _a) in enumerate(self._outputs):
            self.picker.addItem(f"{i}: {info.name}", i)
        self.picker.setCurrentIndex(keep if 0 <= keep < len(self._outputs) else 0)
        self.picker.blockSignals(False)
        self.note.setText(tr("gui.rawout.shown", rows=MAX_ROWS, cols=MAX_COLS))
        self._show_table()

    def _show_table(self) -> None:
        i = self.picker.currentIndex()
        if not (0 <= i < len(self._outputs)):
            return
        view = table_view_of(self._outputs[i][1])
        self.table.clear()
        self.table.setRowCount(view.shape[0])
        self.table.setColumnCount(view.shape[1])
        for r in range(view.shape[0]):
            for c in range(view.shape[1]):
                item = QTableWidgetItem(f"{float(view[r, c]):.4g}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, item)

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            self.session.rawReady.disconnect(self.set_outputs)
        except (RuntimeError, TypeError):
            pass
        super().closeEvent(event)
