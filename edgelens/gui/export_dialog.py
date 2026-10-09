"""切分與匯出對話框：先決定 train／val／test 比例，再打包成 ZIP 給教學平台。"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from edgelens.dataset.export import ExportCancelled, export_zip
from edgelens.dataset.split import PARTS, SplitResult, load_splits, split_dataset
from edgelens.gui.thumbs import BgTask, reveal_in_folder, shared_writer
from edgelens.i18n import current_language, tr

log = logging.getLogger(__name__)

MAX_EDGES = (0, 224, 320, 640)   # 0 表示不縮小


class ExportDialog(QDialog):
    def __init__(self, ctx, project, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.project = project
        self.split_result: SplitResult | None = None
        self.last_zip: Path | None = None
        self._task: BgTask | None = None
        self.busy = False
        self._fmt_split = False
        self.resize(640, 760)
        self._build()
        self._load_previous()
        self.retranslate()

    # ---- UI --------------------------------------------------------------------
    def _build(self) -> None:
        lay = QVBoxLayout(self)
        self.split_box = QGroupBox()
        sl = QVBoxLayout(self.split_box)
        form = QFormLayout()
        row = QHBoxLayout()
        self.spin_train, self.spin_val, self.spin_test = (QSpinBox() for _ in range(3))
        for spin, v in ((self.spin_train, 70), (self.spin_val, 15), (self.spin_test, 15)):
            spin.setRange(0, 100)
            spin.setValue(v)
            spin.setSuffix(" %")
            spin.valueChanged.connect(self._check_ratio)
            row.addWidget(spin)
        self.ratio_label = QLabel()
        form.addRow(self.ratio_label, self._wrap(row))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        self.seed_label = QLabel()
        form.addRow(self.seed_label, self.seed_spin)
        sl.addLayout(form)
        self.ratio_warn = QLabel()
        self.ratio_warn.setStyleSheet("color: #b00020;")
        sl.addWidget(self.ratio_warn)
        self.split_button = QPushButton()
        self.split_button.clicked.connect(self.do_split)
        sl.addWidget(self.split_button)
        self.table = QTableWidget(0, 4)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMinimumHeight(120)
        sl.addWidget(self.table)
        self.split_info = QLabel()
        self.split_info.setWordWrap(True)
        sl.addWidget(self.split_info)
        lay.addWidget(self.split_box)

        self.export_box = QGroupBox()
        el = QVBoxLayout(self.export_box)
        self.radio_folders = QRadioButton()
        self.radio_split = QRadioButton()
        self.radio_folders.setChecked(True)
        self.desc_folders, self.desc_split = QLabel(), QLabel()
        for w in (self.desc_folders, self.desc_split):
            w.setWordWrap(True)
            w.setContentsMargins(22, 0, 0, 6)
        for w in (self.radio_folders, self.desc_folders, self.radio_split, self.desc_split):
            el.addWidget(w)
        self.chk_names = QCheckBox()
        self.chk_processed = QCheckBox()
        self.chk_labels = QCheckBox()
        self.chk_meta = QCheckBox()
        self.chk_labels.setChecked(True)
        self.chk_meta.setChecked(True)
        for w in (self.chk_names, self.chk_processed, self.chk_labels, self.chk_meta):
            el.addWidget(w)
        self.edge_combo = QComboBox()
        self.edge_label = QLabel()
        er = QHBoxLayout()
        er.addWidget(self.edge_label)
        er.addWidget(self.edge_combo, 1)
        el.addLayout(er)
        lay.addWidget(self.export_box)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.hide()
        lay.addWidget(self.progress)
        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        self.result_label.setOpenExternalLinks(False)
        lay.addWidget(self.result_label)
        self.next_label = QLabel()
        self.next_label.setWordWrap(True)
        lay.addWidget(self.next_label)
        buttons = QHBoxLayout()
        self.export_button = QPushButton()
        self.cancel_button = QPushButton()
        self.reveal_button = QPushButton()
        self.close_button = QPushButton()
        self.cancel_button.setEnabled(False)
        self.reveal_button.hide()
        for b in (self.export_button, self.cancel_button, self.reveal_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.close_button)
        lay.addLayout(buttons)
        self.export_button.clicked.connect(self.start_export)
        self.cancel_button.clicked.connect(self.cancel_export)
        self.reveal_button.clicked.connect(self._reveal)
        self.close_button.clicked.connect(self.close)

    @staticmethod
    def _wrap(layout) -> QWidget:
        w = QWidget()
        layout.setContentsMargins(0, 0, 0, 0)
        w.setLayout(layout)
        return w

    def retranslate(self) -> None:
        self.setWindowTitle(tr("gui.dataset.export.title"))
        self.split_box.setTitle(tr("gui.dataset.export.split_group"))
        self.export_box.setTitle(tr("gui.dataset.export.export_group"))
        self.ratio_label.setText(tr("gui.dataset.export.ratio"))
        self.seed_label.setText(tr("gui.dataset.export.seed"))
        self.split_button.setText(tr("gui.dataset.export.do_split"))
        self.table.setHorizontalHeaderLabels([tr("gui.dataset.export.col_class"), "train", "val", "test"])
        self.radio_folders.setText(tr("gui.dataset.export.fmt_folders"))
        self.radio_split.setText(tr("gui.dataset.export.fmt_split"))
        self.desc_folders.setText(tr("gui.dataset.export.fmt_folders_desc"))
        self.desc_split.setText(tr("gui.dataset.export.fmt_split_desc"))
        self.chk_names.setText(tr("gui.dataset.export.use_names"))
        self.chk_processed.setText(tr("gui.dataset.export.include_processed"))
        self.chk_labels.setText(tr("gui.dataset.export.include_labels"))
        self.chk_meta.setText(tr("gui.dataset.export.include_metadata"))
        self.edge_label.setText(tr("gui.dataset.export.max_edge"))
        idx = self.edge_combo.currentIndex()
        self.edge_combo.clear()
        for e in MAX_EDGES:
            self.edge_combo.addItem(tr("gui.dataset.export.edge_none") if e == 0 else f"{e} px", e)
        self.edge_combo.setCurrentIndex(max(0, idx))
        self.export_button.setText(tr("gui.dataset.export.start"))
        self.cancel_button.setText(tr("common.cancel"))
        self.reveal_button.setText(tr("gui.dataset.reveal_zip"))
        self.close_button.setText(tr("common.close"))
        self._check_ratio()

    # ---- 切分 ------------------------------------------------------------------
    def ratios(self) -> tuple[float, float, float]:
        return (self.spin_train.value() / 100, self.spin_val.value() / 100, self.spin_test.value() / 100)

    def _check_ratio(self) -> None:
        total = self.spin_train.value() + self.spin_val.value() + self.spin_test.value()
        ok = total == 100
        self.ratio_warn.setText("" if ok else tr("gui.dataset.export.ratio_bad", total=total))
        self.split_button.setEnabled(ok and not self.busy)

    def _load_previous(self) -> None:
        if not (self.project.dataset_dir / "splits.json").exists():
            return
        try:
            result = load_splits(self.project)
        except (OSError, ValueError, KeyError) as exc:
            log.info("略過讀不了的 splits.json：%s", exc)
            return
        self.split_result = result
        for spin, v in zip((self.spin_train, self.spin_val, self.spin_test), result.ratios, strict=True):
            spin.setValue(round(v * 100))
        self.seed_spin.setValue(int(result.seed))
        self._show_split(result, previous=True)

    def do_split(self) -> SplitResult | None:
        try:
            result = split_dataset(self.project, self.ratios(), seed=self.seed_spin.value())
        except (OSError, ValueError) as exc:
            self.split_info.setText(str(exc))
            return None
        self.split_result = result
        self._show_split(result)
        return result

    def _show_split(self, result: SplitResult, previous: bool = False) -> None:
        lang = current_language()
        by_file = {r.file: r.class_id for r in shared_writer(self.project).list_samples()}
        counts = {c.id: [0, 0, 0] for c in self.project.classes}
        for i, part in enumerate(PARTS):
            for f in getattr(result, part):
                cid = by_file.get(f)
                if cid is not None:
                    counts.setdefault(cid, [0, 0, 0])[i] += 1
        self.table.setRowCount(len(counts))
        for row, (cid, vals) in enumerate(counts.items()):
            spec = self.project.class_by_id(cid)
            self.table.setItem(row, 0, QTableWidgetItem(spec.display(lang) if spec else cid))
            for col, v in enumerate(vals, 1):
                self.table.setItem(row, col, QTableWidgetItem(str(v)))
        lines = []
        if previous:
            lines.append(tr("gui.dataset.export.previous", seed=result.seed))
        lines.extend(result.warnings)
        self.split_info.setText("\n".join(lines) if lines else tr("gui.dataset.export.split_ok"))

    # ---- 匯出 ------------------------------------------------------------------
    def start_export(self) -> None:
        if self.busy:
            return
        split = self.radio_split.isChecked()
        opts = dict(
            use_display_names=self.chk_names.isChecked(), include_processed=self.chk_processed.isChecked(),
            include_labels=self.chk_labels.isChecked(), include_metadata=self.chk_meta.isChecked(),
            max_edge=(self.edge_combo.currentData() or None), lang=current_language())
        ratios, seed = self.ratios(), self.seed_spin.value()
        project, current = self.project, self.split_result
        if split and sum(ratios) < 0.999:
            return
        self.busy = True
        self.last_zip = None
        self.result_label.setText("")
        self.next_label.setText("")
        self.reveal_button.hide()
        self.progress.setValue(0)
        self.progress.show()
        self.export_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.split_button.setEnabled(False)

        def work(task):
            splits = None
            if split:
                files = {r.file for r in shared_writer(project).list_samples()}
                have = set(current.train + current.val + current.test) if current else None
                splits = current if have == files else split_dataset(project, ratios, seed=seed)
            return export_zip(project, "split_folders" if split else "folders", splits=splits,
                              progress=task.emit_progress, cancel=task.cancel_event, **opts)

        task = BgTask(work, self)
        self._task = task
        self._fmt_split = split
        task.progress.connect(self._on_progress)
        task.finished.connect(self._on_done)
        task.start()

    def cancel_export(self) -> None:
        if self._task is not None:
            self._task.cancel_event.set()
            self.cancel_button.setEnabled(False)

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setValue(int(done * 100 / total) if total else 100)

    def _on_done(self, result, error) -> None:
        self.busy = False
        self._task = None
        self.progress.hide()
        self.export_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._check_ratio()
        if isinstance(error, ExportCancelled):
            self.result_label.setText(tr("gui.dataset.export.cancelled"))
            return
        if error is not None:
            self.result_label.setText(tr("gui.dataset.export.failed", reason=str(error)))
            return
        self.last_zip = Path(result)
        if self._fmt_split:   # 匯出時可能重新切分，更新表格
            try:
                self.split_result = load_splits(self.project)
                self._show_split(self.split_result)
            except (OSError, ValueError, KeyError):
                pass
        self.result_label.setText(tr("gui.dataset.export.done", path=str(self.last_zip)))
        self.next_label.setText(tr("gui.dataset.export.next_split" if self._fmt_split
                                   else "gui.dataset.export.next_folders"))
        self.reveal_button.show()

    def _reveal(self) -> None:
        if self.last_zip is not None:
            reveal_in_folder(self.last_zip)

    def closeEvent(self, e):
        if self.busy:
            if QMessageBox.question(self, tr("gui.dataset.export.title"),
                                    tr("gui.dataset.export.busy_close")) != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.cancel_export()
            if self._task is not None:
                self._task.wait(10000)
        super().closeEvent(e)
