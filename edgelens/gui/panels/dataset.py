"""資料集面板（側邊精簡版）：每類張數、不平衡提示、最近拍的照片與常用按鈕。

AppContext 目前沒有 datasetChanged signal，所以面板顯示時與每 2 秒檢查一次
metadata.jsonl 的修改時間與大小，有變才重新讀取。
"""

from __future__ import annotations

import logging
import time

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QListView,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from edgelens.dataset.stats import dataset_stats
from edgelens.dataset.writer import TRASH_KEEP_DAYS
from edgelens.gui.dataset_browser import DatasetBrowser, SampleModel, color_pixmap
from edgelens.gui.export_dialog import ExportDialog
from edgelens.gui.panels.base import Panel
from edgelens.gui.stable import StableLabel
from edgelens.gui.thumbs import (
    BgTask,
    PixmapCache,
    ThumbLoader,
    metadata_signature,
    open_folder,
    shared_writer,
)
from edgelens.i18n import current_language, tr

log = logging.getLogger(__name__)

RECENT = 9
RECENT_SIZE = 72


class DatasetPanel(Panel):
    panel_id = "dataset"
    title_key = "gui.panel.dataset.title"
    order = 50

    def __init__(self, ctx, main):
        super().__init__(ctx, main)
        self.project = None
        self.browser: DatasetBrowser | None = None
        self.export_dialog: ExportDialog | None = None
        self.loader: ThumbLoader | None = None
        self.recent_model: SampleModel | None = None
        self._sig = (-1, -1)
        self._records = []
        self._rebuild_task: BgTask | None = None
        self.count_labels: dict[str, QLabel] = {}
        self._stale = False
        self._purge_task: BgTask | None = None
        # 拍照面板存檔後會發出 datasetChanged，立刻更新張數；輪詢（_poll）保留作保險。
        # 面板看不到時只記「過期」，等顯示時再讀：讀整份 metadata 會隨照片變多而卡住拍照。
        ctx.datasetChanged.connect(self._on_dataset_changed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        lay = QVBoxLayout(body)

        self.empty_label = QLabel()
        self.empty_label.setWordWrap(True)
        lay.addWidget(self.empty_label)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.grid_host)
        self.total_label = QLabel()
        self.last_label = QLabel()
        self.hint_label = StableLabel(2)   # 拍照後提示可能出現／消失：保留高度，版面不跳動
        self.hint_label.setStyleSheet("color: #b36b00;")
        for w in (self.total_label, self.last_label, self.hint_label):
            lay.addWidget(w)

        self.browser_button = QPushButton()
        self.folder_button = QPushButton()
        self.export_button = QPushButton()
        self.refresh_button = QPushButton()
        for b in (self.browser_button, self.folder_button, self.export_button, self.refresh_button):
            lay.addWidget(b)
        self.browser_button.clicked.connect(self.open_browser)
        self.folder_button.clicked.connect(self.open_folder)
        self.export_button.clicked.connect(self.open_export)
        self.refresh_button.clicked.connect(self.rebuild)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        lay.addWidget(self.status_label)

        self.recent_title = QLabel()
        lay.addWidget(self.recent_title)
        self.recent_view = QListView()
        self.recent_view.setViewMode(QListView.ViewMode.IconMode)
        self.recent_view.setResizeMode(QListView.ResizeMode.Adjust)
        self.recent_view.setMovement(QListView.Movement.Static)
        self.recent_view.setIconSize(QSize(RECENT_SIZE, RECENT_SIZE))
        self.recent_view.setGridSize(QSize(RECENT_SIZE + 8, RECENT_SIZE + 8))
        self.recent_view.setEditTriggers(QListView.EditTrigger.NoEditTriggers)
        self.recent_view.setFixedHeight(3 * (RECENT_SIZE + 8) + 6)
        self.recent_view.doubleClicked.connect(self._open_recent)
        lay.addWidget(self.recent_view)
        lay.addStretch(1)

        self._timer = QTimer(self)
        self._timer.setInterval(2000)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        self.retranslate()
        self.on_project_changed(ctx.project)

    # ---- 專案與重新整理 ---------------------------------------------------------------
    def on_project_changed(self, project) -> None:
        if self.loader is not None:
            self.loader.stop()
            self.loader = None
        self.project = project
        self.recent_model = None
        self.recent_view.setModel(None)
        if project is not None:
            self.loader = ThumbLoader(project, size=RECENT_SIZE * 2, parent=self)
            self.recent_model = SampleModel(project, self.loader, PixmapCache(64), self)
            self.recent_view.setModel(self.recent_model)
        self.status_label.setText("")
        self.refresh(force=True)
        self._purge_old_trash()

    def _purge_old_trash(self) -> None:
        """開啟專案時在背景清掉超過 TRASH_KEEP_DAYS 天的垃圾桶照片，不阻塞介面。"""
        if self.project is None:
            return
        writer = shared_writer(self.project)
        task = BgTask(lambda _t: writer.purge_trash(TRASH_KEEP_DAYS), self)
        self._purge_task = task

        def done(removed, error, t=task):
            if self._purge_task is t:
                self._purge_task = None
            if error is not None:
                log.warning("自動清理垃圾桶失敗：%s", error)
                return
            log.info("自動清理垃圾桶：移除 %s 張超過 %d 天的照片", removed, TRASH_KEEP_DAYS)
            if removed:
                self._on_dataset_changed()

        task.finished.connect(done)
        task.start()

    def _on_dataset_changed(self) -> None:
        if self.isVisible():
            self.refresh(force=True)
        else:
            self._stale = True

    def showEvent(self, e):
        super().showEvent(e)
        self.refresh(force=self._stale)

    def _poll(self) -> None:
        if self.isVisible():
            self.refresh()

    def refresh(self, force: bool = False) -> None:
        """metadata.jsonl 沒變就什麼都不做，所以可以很常呼叫。"""
        if self.project is None:
            self._records = []
            self._render()
            return
        sig = metadata_signature(self.project)
        if not force and sig == self._sig:
            return
        self._stale = False
        self._sig = sig
        try:
            self._records = shared_writer(self.project).list_samples()
        except (OSError, ValueError) as exc:
            self.status_label.setText(str(exc))
            self._records = []
        self._render()

    def _render(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.count_labels = {}
        has = self.project is not None
        self.empty_label.setVisible(not has)
        for b in (self.browser_button, self.folder_button, self.export_button, self.refresh_button):
            b.setEnabled(has)
        if not has:
            self.total_label.setText("")
            self.last_label.setText("")
            self.hint_label.setText("")
            return
        lang = current_language()
        classes = self.project.classes
        stats = dataset_stats(self._records, [c.id for c in classes])
        for row, spec in enumerate(classes):
            chip = QLabel()
            chip.setPixmap(color_pixmap(spec.color))
            name = QLabel(spec.display(lang))
            count = QLabel(str(stats.class_counts.get(spec.id, 0)))
            count.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.count_labels[spec.id] = count
            self.grid.addWidget(chip, row, 0)
            self.grid.addWidget(name, row, 1)
            self.grid.addWidget(count, row, 2)
        self.grid.setColumnStretch(1, 1)
        self.total_label.setText(tr("gui.dataset.panel.total", count=stats.total))
        epochs = [float(r.time.get("epoch", 0) or 0) for r in self._records]
        last = max(epochs, default=0)
        self.last_label.setText(tr("gui.dataset.panel.last", time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last)))
                                if last else tr("gui.dataset.panel.last_none"))
        hint = ""
        if stats.total and stats.imbalanced:
            counts = stats.class_counts
            hi = max(counts, key=counts.get)
            lo = min(counts, key=counts.get)
            hint = tr("gui.dataset.panel.imbalanced", most=self._name(hi), most_n=counts[hi],
                      least=self._name(lo), least_n=counts[lo])
        elif not stats.total:
            hint = tr("gui.dataset.panel.empty")
        self.hint_label.setText(hint)
        if self.recent_model is not None:
            recent = sorted(self._records, key=lambda r: float(r.time.get("epoch", 0) or 0), reverse=True)[:RECENT]
            self.recent_model.set_records(recent)

    def _name(self, class_id: str) -> str:
        spec = self.project.class_by_id(class_id)
        return spec.display(current_language()) if spec else class_id

    def rebuild(self) -> None:
        """把學生自己丟進資料夾的照片納入紀錄；在背景執行，照片多時才不會卡住。"""
        if self.project is None or self._rebuild_task is not None:
            return
        writer = shared_writer(self.project)
        before = len(self._records)
        self.refresh_button.setEnabled(False)
        self.status_label.setText(tr("gui.dataset.panel.rebuilding"))
        task = BgTask(lambda _t: writer.rebuild_metadata(), self)
        self._rebuild_task = task

        def done(records, error, t=task):
            self._rebuild_task = None
            self.refresh_button.setEnabled(self.project is not None)
            if error is not None:
                self.status_label.setText(tr("gui.dataset.panel.rebuild_failed", reason=str(error)))
            else:
                self.status_label.setText(tr("gui.dataset.panel.rebuilt", added=max(0, len(records) - before)))
            self.refresh(force=True)

        task.finished.connect(done)
        task.start()

    # ---- 按鈕 -----------------------------------------------------------------------
    def open_browser(self, file: str | None = None) -> DatasetBrowser | None:
        if self.project is None:
            return None
        if self.browser is None:
            self.browser = DatasetBrowser(self.ctx, self.project, self.window())
            self.browser.datasetModified.connect(self._on_dataset_changed)
        if isinstance(file, str):
            self.browser.open_sample(file)
        else:
            self.browser.show()
            self.browser.raise_()
            self.browser.activateWindow()
        return self.browser

    def open_folder(self) -> None:
        if self.project is not None:
            open_folder(self.project.dataset_dir)

    def open_export(self) -> ExportDialog | None:
        if self.project is None:
            return None
        self.export_dialog = ExportDialog(self.ctx, self.project, self.window())
        self.export_dialog.show()
        return self.export_dialog

    def _open_recent(self, index) -> None:
        rec = index.data(Qt.ItemDataRole.UserRole)
        if rec is not None:
            self.open_browser(rec.file)

    # ---- 其他 -----------------------------------------------------------------------
    def retranslate(self) -> None:
        self.empty_label.setText(tr("gui.dataset.panel.no_project"))
        self.browser_button.setText(tr("gui.dataset.panel.open_browser"))
        self.folder_button.setText(tr("gui.dataset.panel.open_folder"))
        self.export_button.setText(tr("gui.dataset.panel.export"))
        self.refresh_button.setText(tr("gui.dataset.panel.refresh"))
        self.refresh_button.setToolTip(tr("gui.dataset.panel.refresh_tip"))
        self.recent_title.setText(tr("gui.dataset.panel.recent"))
        if self.browser is not None:
            self.browser.retranslate()
        if self.export_dialog is not None:
            self.export_dialog.retranslate()
        if hasattr(self, "_records"):
            self._render()

    def shutdown(self) -> None:
        self._timer.stop()
        if self.loader is not None:
            self.loader.stop()
        if self.browser is not None:
            self.browser.shutdown()
            self.browser.close()
        if self.export_dialog is not None:
            self.export_dialog.close()

