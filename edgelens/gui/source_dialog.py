"""新增相機對話框：選來源種類 → 尋找（背景執行緒）或手動填參數 → 回傳 SourceInfo。"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from edgelens import zoo
from edgelens.gui.context import AppContext
from edgelens.gui.control_panel import ControlPanel
from edgelens.gui.zoo_download import download_item
from edgelens.i18n import component_description, component_name, tr
from edgelens.sources.base import SOURCES, FrameSource, SourceInfo

log = logging.getLogger(__name__)

DISCOVER_TIMEOUT = 3.0
_orphans: list[QThread] = []  # 還在跑 discover 時對話框就關了：保留參照避免執行緒被銷毀


class _DiscoverThread(QThread):
    found = Signal(str, list, str)  # kind、SourceInfo 清單、錯誤訊息

    def __init__(self, cls: type[FrameSource], parent=None):
        super().__init__(parent)
        self._cls = cls

    def run(self) -> None:
        try:
            infos = self._cls.discover(timeout=DISCOVER_TIMEOUT)
            self.found.emit(self._cls.kind, list(infos), "")
        except Exception as e:  # noqa: BLE001 - discover 失敗只顯示訊息
            log.exception("尋找相機失敗")
            self.found.emit(self._cls.kind, [], str(e))


class SourceDialog(QDialog):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None):
        super().__init__(parent)
        self.ctx = ctx
        self._info: SourceInfo | None = None
        self._thread: _DiscoverThread | None = None
        self._kind = ""
        self.setWindowTitle(tr("gui.source.title"))
        self.resize(820, 560)

        root = QVBoxLayout(self)
        # 最近使用
        self.recent_box = QGroupBox(tr("gui.source.recent"))
        rl = QVBoxLayout(self.recent_box)
        self.recent_list = QListWidget()
        self.recent_list.setMaximumHeight(90)
        self.recent_list.itemDoubleClicked.connect(self._recent_double)
        self.recent_list.itemClicked.connect(self._recent_clicked)
        rl.addWidget(self.recent_list)
        root.addWidget(self.recent_box)
        self._recent_infos: list[SourceInfo] = []
        self._load_recent()

        split = QSplitter()
        root.addWidget(split, 1)
        self.kind_list = QListWidget()
        self.kind_list.currentItemChanged.connect(self._kind_changed)
        split.addWidget(self.kind_list)

        right = QWidget()
        rlay = QVBoxLayout(right)
        self.desc = QLabel()
        self.desc.setWordWrap(True)
        rlay.addWidget(self.desc)
        row = QHBoxLayout()
        self.find_button = QPushButton(tr("gui.source.find"))
        self.find_button.clicked.connect(self.start_discover)
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setMaximumHeight(14)
        self.busy.hide()
        self.samples_button = QPushButton()
        self.samples_button.clicked.connect(self.download_samples)
        self.samples_button.hide()
        row.addWidget(self.find_button)
        row.addWidget(self.samples_button)
        row.addWidget(self.busy, 1)
        rlay.addLayout(row)
        self.result_list = QListWidget()
        self.result_list.itemDoubleClicked.connect(lambda _i: self._accept_selected())
        rlay.addWidget(self.result_list, 1)
        self.result_note = QLabel()
        self.result_note.setWordWrap(True)
        rlay.addWidget(self.result_note)
        self.manual_box = QGroupBox(tr("gui.source.manual"))
        ml = QVBoxLayout(self.manual_box)
        self.manual_panel = ControlPanel(show_status=False)
        ml.addWidget(self.manual_panel)
        self.manual_button = QPushButton(tr("gui.source.manual_add"))
        self.manual_button.clicked.connect(self._manual_add)
        ml.addWidget(self.manual_button)
        rlay.addWidget(self.manual_box)
        split.addWidget(right)
        split.setSizes([260, 560])

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr("gui.source.open"))
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr("common.cancel"))
        self.buttons.accepted.connect(self._accept_selected)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

        for kind, cls in SOURCES.items():
            ok, reason = cls.available()
            name = component_name(cls)
            item = QListWidgetItem(name if ok else f"{name}  ({tr('gui.source.unavailable')})")
            item.setData(Qt.ItemDataRole.UserRole, kind)
            if not ok:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setToolTip(reason)
            else:
                item.setToolTip(component_description(cls))
            self.kind_list.addItem(item)
        for i in range(self.kind_list.count()):
            if self.kind_list.item(i).flags() & Qt.ItemFlag.ItemIsEnabled:
                self.kind_list.setCurrentRow(i)
                break

    # ---- 最近使用 ---------------------------------------------------------------
    def _load_recent(self) -> None:
        for d in self.ctx.settings.get("recent.sources", []) or []:
            try:
                info = SourceInfo.from_dict(d)
            except (KeyError, TypeError):
                continue
            if info.kind not in SOURCES:
                continue
            self._recent_infos.append(info)
            self.recent_list.addItem(f"{info.name}  [{component_name(SOURCES.get(info.kind))}]")
        self.recent_box.setVisible(bool(self._recent_infos))

    def _recent_clicked(self, _item) -> None:
        self.result_list.clearSelection()

    def _recent_double(self, _item) -> None:
        self._info = self._recent_infos[self.recent_list.currentRow()]
        self.accept()

    # ---- 種類 -------------------------------------------------------------------
    def select_kind(self, kind: str) -> None:
        for i in range(self.kind_list.count()):
            if self.kind_list.item(i).data(Qt.ItemDataRole.UserRole) == kind:
                self.kind_list.setCurrentRow(i)
                return
        raise KeyError(kind)

    def _kind_changed(self, item: QListWidgetItem | None, _prev=None) -> None:
        self.result_list.clear()
        self.result_note.setText("")
        if item is None:
            return
        kind = item.data(Qt.ItemDataRole.UserRole)
        cls = SOURCES.get(kind)
        self._kind = kind
        self.desc.setText(component_description(cls))
        self.find_button.setEnabled(self._thread is None)
        specs = cls.param_specs() if cls.manual_add else []
        self.manual_box.setVisible(cls.manual_add)
        self.manual_panel.set_specs(specs, {})
        self.manual_box.setVisible(bool(cls.manual_add))
        self._refresh_samples_button()

    # ---- 示範照片 -----------------------------------------------------------------
    def _missing_samples(self) -> list[zoo.ZooItem]:
        return [it for it in zoo.items("samples") if it.sha256 and not zoo.is_installed(it)]

    def _refresh_samples_button(self) -> None:
        missing = self._missing_samples() if self._kind == "file" else []
        self.samples_button.setVisible(bool(missing))
        if missing:
            self.samples_button.setText(tr("gui.source.samples.download", name=missing[0].title,
                                           size=missing[0].size_text))

    def download_samples(self) -> None:
        """沒有相機也能馬上試物件偵測：下載示範照片，裝好後直接列在尋找結果裡。"""
        missing = self._missing_samples()
        if missing and download_item(self, missing[0]):
            self._refresh_samples_button()
            self.start_discover()

    # ---- 尋找 -------------------------------------------------------------------
    def start_discover(self) -> None:
        if self._thread is not None or not self._kind:
            return
        cls = SOURCES.get(self._kind)
        self.result_list.clear()
        self.result_note.setText(tr("gui.source.searching"))
        self.busy.show()
        self.find_button.setEnabled(False)
        th = _DiscoverThread(cls)
        th.found.connect(self._on_found)
        th.finished.connect(self._thread_done)
        self._thread = th
        th.start()

    def _on_found(self, kind: str, infos: list, error: str) -> None:
        if kind != self._kind:
            return  # 使用者已經切到別種來源
        self.result_list.clear()
        for info in infos:
            text = info.name + (f"  —  {info.detail}" if info.detail else "")
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, info)
            self.result_list.addItem(item)
        if error:
            self.result_note.setText(tr("gui.source.find_failed", error=error))
        elif not infos:
            self.result_note.setText(tr("gui.source.none_found"))
        else:
            self.result_note.setText(tr("gui.source.found", count=len(infos)))
            self.result_list.setCurrentRow(0)

    def _thread_done(self) -> None:
        th, self._thread = self._thread, None
        self.busy.hide()
        self.find_button.setEnabled(True)
        if th is not None:
            th.deleteLater()

    def is_searching(self) -> bool:
        return self._thread is not None

    # ---- 結果 -------------------------------------------------------------------
    def _manual_add(self) -> None:
        cls = SOURCES.get(self._kind)
        try:
            self._info = cls.info_from_params(self.manual_panel.values())
        except Exception as e:  # noqa: BLE001 - 參數不合格時說明原因
            QMessageBox.warning(self, tr("gui.source.title"), tr("gui.source.manual_failed", error=str(e)))
            return
        self.accept()

    def _accept_selected(self) -> None:
        item = self.result_list.currentItem()
        recent = self.recent_list.currentItem()
        if item is not None and item.isSelected():
            self._info = item.data(Qt.ItemDataRole.UserRole)
            self.accept()
        elif recent is not None and recent.isSelected():
            self._info = self._recent_infos[self.recent_list.currentRow()]
            self.accept()
        elif self.manual_box.isVisible():
            self._manual_add()
        else:
            self.result_note.setText(tr("gui.source.pick_first"))

    def selected_info(self) -> SourceInfo | None:
        return self._info

    def done(self, result: int) -> None:
        th = self._thread
        if th is not None and th.isRunning():
            if not th.wait(int((DISCOVER_TIMEOUT + 1) * 1000)):
                _orphans.append(th)
                th.setParent(None)
        super().done(result)
