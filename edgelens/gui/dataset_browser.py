"""資料集瀏覽視窗：縮圖格、篩選、改類別、刪除與復原、找相似照片。

非模態：老師可以一邊看資料集、一邊回到主視窗拍照。照片很多（數千張）時也要順，
所以縮圖只在 data() 被呼叫（也就是看得到）的時候才排進背景佇列。
"""

from __future__ import annotations

import logging
from collections import defaultdict

from PySide6.QtCore import QAbstractListModel, QMimeData, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from edgelens.dataset.dedupe import find_near_duplicates
from edgelens.dataset.writer import SampleRecord
from edgelens.gui.thumbs import (
    THUMB_SIZE,
    BgTask,
    PixmapCache,
    ThumbLoader,
    bgr_to_qimage,
    decode_file,
    metadata_signature,
    reveal_in_folder,
    shared_writer,
    thumb_key,
)
from edgelens.i18n import current_language, tr
from edgelens.metrics.quality import QualityThresholds

log = logging.getLogger(__name__)

MIME = "application/x-edgelens-samples"
ALL = "__all__"
DELETED = "__deleted__"
CELL = QSize(THUMB_SIZE + 12, THUMB_SIZE + 16)


def color_pixmap(color: str, size: int = 12) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(QColor(color))
    return pm


def record_camera(r: SampleRecord) -> str:
    return str(r.source.get("id") or r.source.get("name") or "?")


def record_time(r: SampleRecord) -> float:
    try:
        return float(r.time.get("epoch", 0.0))
    except (TypeError, ValueError):
        return 0.0


def record_sharpness(r: SampleRecord) -> float:
    try:
        return float(r.quality.get("sharpness", 0.0))
    except (TypeError, ValueError):
        return 0.0


def quality_flags(r: SampleRecord, th: QualityThresholds | None = None) -> set[str]:
    """回傳 {'blurry','dark','bright'} 的子集合；沒有品質數值就不亂判斷。"""
    th = th or QualityThresholds()
    flags = set()
    q = r.quality or {}
    try:
        if "sharpness" in q and float(q["sharpness"]) < th.min_sharpness:
            flags.add("blurry")
        if "brightness" in q:
            b = float(q["brightness"])
            if b < th.min_brightness:
                flags.add("dark")
            elif b > th.max_brightness:
                flags.add("bright")
    except (TypeError, ValueError):
        pass
    return flags


def describe_record(r: SampleRecord, project) -> str:
    """給右側面板的 metadata 文字（純文字，每行一項）。"""
    lang = current_language()
    lines = []
    spec = project.class_by_id(r.class_id) if project else None
    lines.append(f"{tr('gui.dataset.meta.class')}：{spec.display(lang) if spec else r.class_id}")
    lines.append(f"{tr('gui.dataset.meta.file')}：{r.file}")
    iso = str(r.time.get("iso", ""))
    if iso:
        lines.append(f"{tr('gui.dataset.meta.time')}：{iso[:19].replace('T', ' ')}")
    cam = r.source.get("name") or r.source.get("id")
    if cam:
        lines.append(f"{tr('gui.dataset.meta.camera')}：{cam}")
    if r.width and r.height:
        lines.append(f"{tr('gui.dataset.meta.size')}：{r.width}×{r.height}")
    q = r.quality or {}
    if "sharpness" in q or "brightness" in q:
        parts = []
        if "sharpness" in q:
            parts.append(f"{tr('gui.dataset.meta.sharpness')} {float(q['sharpness']):.0f}")
        if "brightness" in q:
            parts.append(f"{tr('gui.dataset.meta.brightness')} {float(q['brightness']):.0f}")
        flags = [tr(f"gui.dataset.quality.{f}") for f in sorted(quality_flags(r))]
        text = "、".join(parts) + (f"（{'、'.join(flags)}）" if flags else "")
        lines.append(f"{tr('gui.dataset.meta.quality')}：{text}")
    steps = [s.get("kind", "?") for s in (r.pipeline or {}).get("steps", [])
             if isinstance(s, dict) and s.get("enabled", True)]
    lines.append(f"{tr('gui.dataset.meta.pipeline')}：{' → '.join(steps) if steps else tr('gui.dataset.meta.none')}")
    res = r.result or {}
    summary = ""
    if res.get("top_k"):
        best = max(res["top_k"], key=lambda c: c.get("score", 0))
        summary = f"{best.get('label') or best.get('class_id')} {float(best.get('score', 0)):.2f}"
    elif res.get("detections"):
        summary = tr("gui.dataset.meta.detections", count=len(res["detections"]))
    if summary:
        lines.append(f"{tr('gui.dataset.meta.result')}：{summary}")
    if r.group_id:
        lines.append(f"{tr('gui.dataset.meta.group')}：{r.group_id}")
    if r.burst_id:
        lines.append(f"{tr('gui.dataset.meta.burst')}：{r.burst_id}")
    if r.note:
        lines.append(f"{tr('gui.dataset.meta.note')}：{r.note}")
    return "\n".join(lines)


# ---------------------------------------------------------------- model / delegate

class SampleModel(QAbstractListModel):
    def __init__(self, project, loader: ThumbLoader, cache: PixmapCache | None = None, parent=None):
        super().__init__(parent)
        self.project = project
        self.loader = loader
        self.cache = cache or PixmapCache()
        self.records: list[SampleRecord] = []
        self._keys: list[str] = []
        self._rows: dict[str, list[int]] = defaultdict(list)
        self._placeholder = QPixmap(THUMB_SIZE, THUMB_SIZE)
        self._placeholder.fill(QColor("#d8d8d8"))
        loader.thumbReady.connect(self._on_ready)

    def set_records(self, records: list[SampleRecord]) -> None:
        self.beginResetModel()
        self.records = list(records)
        self._keys = [thumb_key(r, self.loader.size) for r in self.records]
        self._rows = defaultdict(list)
        for i, k in enumerate(self._keys):
            self._rows[k].append(i)
        self.endResetModel()

    def color_of(self, class_id: str) -> str:
        spec = self.project.class_by_id(class_id) if self.project else None
        return spec.color if spec else "#999999"

    def rowCount(self, parent=None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.records)

    def pixmap(self, row: int) -> QPixmap | None:
        return self.cache.get(self._keys[row])

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.records):
            return None
        r = self.records[index.row()]
        if role == Qt.ItemDataRole.DecorationRole:
            pm = self.cache.get(self._keys[index.row()])
            if pm is None:
                self.loader.request(self._keys[index.row()], r.path)   # 看得到才載入
                return QIcon(self._placeholder)
            return QIcon(pm)
        if role == Qt.ItemDataRole.ToolTipRole:
            return describe_record(r, self.project)
        if role == Qt.ItemDataRole.UserRole:
            return r
        return None

    def flags(self, index):
        f = super().flags(index)
        return f | Qt.ItemFlag.ItemIsDragEnabled if index.isValid() else f

    def mimeTypes(self):
        return [MIME]

    def mimeData(self, indexes):
        data = QMimeData()
        files = [self.records[i.row()].file for i in indexes if i.isValid()]
        data.setData(MIME, "\n".join(files).encode("utf-8"))
        return data

    def _on_ready(self, key: str, image) -> None:
        self.cache.put(key, QPixmap.fromImage(image))
        for row in self._rows.get(key, ()):
            idx = self.index(row)
            self.dataChanged.emit(idx, idx, [Qt.ItemDataRole.DecorationRole])


class ThumbDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        return CELL

    def paint(self, painter: QPainter, option, index):
        model: SampleModel = index.model()
        r = model.records[index.row()]
        rect = option.rect.adjusted(3, 3, -3, -3)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        painter.save()
        if selected:
            painter.fillRect(option.rect, option.palette.highlight())
        index.data(Qt.ItemDataRole.DecorationRole)   # 畫到才要求載入縮圖（只有看得到的格子會畫）
        pm = model.pixmap(index.row()) or model._placeholder
        scaled = pm.scaled(THUMB_SIZE, THUMB_SIZE, Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation) if max(pm.width(), pm.height()) > THUMB_SIZE else pm
        x = rect.x() + (rect.width() - scaled.width()) // 2
        y = rect.y() + (THUMB_SIZE - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.fillRect(QRect(rect.x(), rect.bottom() - 5, rect.width(), 5), QColor(model.color_of(r.class_id)))
        painter.restore()


class ClassList(QListWidget):
    """左側類別清單；可以把縮圖拖進來改類別。"""

    samplesDropped = Signal(str, list)   # class_id, files

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(MIME):
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        item = self.itemAt(e.position().toPoint())
        if e.mimeData().hasFormat(MIME) and item is not None and item.data(Qt.ItemDataRole.UserRole) not in (ALL, DELETED):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e):
        item = self.itemAt(e.position().toPoint())
        if item is None or not e.mimeData().hasFormat(MIME):
            return
        cid = item.data(Qt.ItemDataRole.UserRole)
        if cid in (ALL, DELETED):
            return
        files = bytes(e.mimeData().data(MIME)).decode("utf-8").split("\n")
        e.acceptProposedAction()
        self.samplesDropped.emit(cid, [f for f in files if f])


class ScaledImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self._pm: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, pm: QPixmap | None) -> None:
        self._pm = pm
        self._refit()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refit()

    def _refit(self) -> None:
        if self._pm is None or self._pm.isNull():
            self.setPixmap(QPixmap())
            return
        self.setPixmap(self._pm.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation))


# ---------------------------------------------------------------- browser

class DatasetBrowser(QDialog):
    datasetModified = Signal()

    def __init__(self, ctx, project, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.project = None
        self.writer = None
        self.loader: ThumbLoader | None = None
        self.model: SampleModel | None = None
        self.all_records: list[SampleRecord] = []
        self.class_filter = ALL
        self.camera_filter: str | None = None
        self.quality_filter = "all"
        self.burst_filter = "all"
        self.sort_mode = "time_desc"
        self.similar_mode = False
        self._similar: list[list[str]] | None = None
        self._similar_task: BgTask | None = None
        self._sig = (0, 0)
        self.busy = False
        self._tasks: list[BgTask] = []
        self._shortcuts: list[QShortcut] = []
        self.resize(1180, 720)
        self._build()
        self._timer = QTimer(self)
        self._timer.setInterval(2000)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        ctx.projectChanged.connect(self.set_project)
        # 拍照面板存檔後會發出 datasetChanged；輪詢（_poll）保留作保險
        ctx.datasetChanged.connect(lambda: None if self.busy or not self.isVisible() else self.reload())
        self.set_project(project)
        self.retranslate()

    # ---- UI --------------------------------------------------------------------
    def _build(self) -> None:
        outer = QHBoxLayout(self)
        split = QSplitter()
        outer.addWidget(split)
        self.class_list = ClassList()
        self.class_list.setMinimumWidth(150)
        self.class_list.currentItemChanged.connect(self._on_class_item)
        self.class_list.samplesDropped.connect(self._on_drop)
        split.addWidget(self.class_list)

        mid = QWidget()
        ml = QVBoxLayout(mid)
        ml.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.quality_combo = QComboBox()
        self.burst_combo = QComboBox()
        self.sort_combo = QComboBox()
        self.similar_button = QPushButton()
        self.similar_button.setCheckable(True)
        self.similar_distance = QSpinBox()
        self.similar_distance.setRange(0, 16)
        self.similar_distance.setValue(5)
        self.lbl_camera, self.lbl_quality, self.lbl_burst, self.lbl_sort = (QLabel() for _ in range(4))
        for lab, w in ((self.lbl_camera, self.camera_combo), (self.lbl_quality, self.quality_combo),
                       (self.lbl_burst, self.burst_combo), (self.lbl_sort, self.sort_combo)):
            bar.addWidget(lab)
            bar.addWidget(w)
        bar.addWidget(self.similar_button)
        bar.addWidget(self.similar_distance)
        bar.addStretch(1)
        ml.addLayout(bar)
        self.camera_combo.currentIndexChanged.connect(self._on_filter)
        self.quality_combo.currentIndexChanged.connect(self._on_filter)
        self.burst_combo.currentIndexChanged.connect(self._on_filter)
        self.sort_combo.currentIndexChanged.connect(self._on_filter)
        self.similar_button.toggled.connect(self._on_similar_toggled)
        self.similar_distance.valueChanged.connect(self._on_similar_distance)

        self.view = QListView()
        self.view.setViewMode(QListView.ViewMode.IconMode)
        self.view.setResizeMode(QListView.ResizeMode.Adjust)
        self.view.setMovement(QListView.Movement.Static)
        self.view.setUniformItemSizes(True)
        self.view.setGridSize(CELL)
        self.view.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.view.setDragEnabled(True)
        self.view.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._context_menu)
        self.view.setItemDelegate(ThumbDelegate(self.view))
        ml.addWidget(self.view, 1)

        row = QHBoxLayout()
        self.count_label = QLabel()
        row.addWidget(self.count_label, 1)
        self.move_button = QToolButton()
        self.move_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.move_menu = QMenu(self.move_button)
        self.move_button.setMenu(self.move_menu)
        self.delete_button = QPushButton()
        self.restore_button = QPushButton()
        self.reveal_button = QPushButton()
        for w in (self.move_button, self.delete_button, self.restore_button, self.reveal_button):
            row.addWidget(w)
        ml.addLayout(row)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        ml.addWidget(self.status_label)
        self.delete_button.clicked.connect(self.delete_selected)
        self.restore_button.clicked.connect(self.restore_selected)
        self.reveal_button.clicked.connect(self.reveal_selected)
        split.addWidget(mid)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.preview = ScaledImageLabel()
        rl.addWidget(self.preview, 3)
        self.processed_button = QPushButton()
        self.processed_button.setCheckable(True)
        self.processed_button.toggled.connect(lambda _c: self._show_current())
        rl.addWidget(self.processed_button)
        self.meta = QTextEdit()
        self.meta.setReadOnly(True)
        rl.addWidget(self.meta, 2)
        right.setMinimumWidth(280)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([170, 700, 320])

        for key in ("Delete", "Backspace"):
            self._add_shortcut(key, self.delete_selected)

    def _add_shortcut(self, key: str, fn) -> None:
        sc = QShortcut(QKeySequence(key), self)
        sc.setContext(Qt.ShortcutContext.WindowShortcut)
        sc.activated.connect(fn)
        self._shortcuts.append(sc)

    def retranslate(self) -> None:
        self.setWindowTitle(tr("gui.dataset.browser.title"))
        self.lbl_camera.setText(tr("gui.dataset.filter.camera"))
        self.lbl_quality.setText(tr("gui.dataset.filter.quality"))
        self.lbl_burst.setText(tr("gui.dataset.filter.burst"))
        self.lbl_sort.setText(tr("gui.dataset.sort.label"))
        self.similar_button.setText(tr("gui.dataset.similar"))
        self.similar_button.setToolTip(tr("gui.dataset.similar.tip"))
        self.similar_distance.setToolTip(tr("gui.dataset.similar.distance"))
        self.move_button.setText(tr("gui.dataset.move"))
        self.delete_button.setText(tr("common.delete"))
        self.restore_button.setText(tr("gui.dataset.restore"))
        self.reveal_button.setText(tr("gui.dataset.reveal"))
        self.processed_button.setText(tr("gui.dataset.show_processed"))
        keep = {c: c.currentData() for c in (self.quality_combo, self.sort_combo)}
        self.quality_combo.blockSignals(True)
        self.sort_combo.blockSignals(True)
        self.quality_combo.clear()
        for k in ("all", "blurry", "dark", "bright"):
            self.quality_combo.addItem(tr(f"gui.dataset.quality.{k}"), k)
        self.sort_combo.clear()
        for k in ("time_desc", "time_asc", "sharp_desc", "sharp_asc"):
            self.sort_combo.addItem(tr(f"gui.dataset.sort.{k}"), k)
        for c, v in keep.items():
            if v is not None:
                c.setCurrentIndex(max(0, c.findData(v)))
        self.quality_combo.blockSignals(False)
        self.sort_combo.blockSignals(False)
        self._refresh_lists()
        self._update_buttons()

    # ---- 專案與載入 ---------------------------------------------------------------
    def set_project(self, project) -> None:
        if self.loader is not None:
            self.loader.stop()
        self.project = project
        self.all_records = []
        self._similar = None
        self.class_filter = ALL
        self.busy = False
        if project is None:
            self.writer = None
            self.loader = None
            self.model = None
            self.view.setModel(None)
            self._refresh_lists()
            return
        self.writer = shared_writer(project)
        self.loader = ThumbLoader(project, parent=self)
        self.model = SampleModel(project, self.loader, parent=self)
        self.view.setModel(self.model)
        self.view.selectionModel().selectionChanged.connect(self._on_selection)
        self._rebuild_shortcuts()
        self.reload()

    def _rebuild_shortcuts(self) -> None:
        for sc in getattr(self, "_class_shortcuts", []):
            sc.setEnabled(False)
            sc.deleteLater()
        self._class_shortcuts = []
        for spec in (self.project.classes if self.project else []):
            if spec.hotkey and spec.hotkey.isdigit():
                sc = QShortcut(QKeySequence(spec.hotkey), self)
                sc.setContext(Qt.ShortcutContext.WindowShortcut)
                sc.activated.connect(lambda cid=spec.id: self.relabel_selected(cid))
                self._class_shortcuts.append(sc)

    def reload(self) -> None:
        if self.writer is None:
            return
        selected = {r.file for r in self.selected_records()}
        try:
            self.all_records = self.writer.list_samples(include_deleted=True)
        except (OSError, ValueError) as exc:
            self.status_label.setText(str(exc))
            return
        self._sig = metadata_signature(self.project)
        self._similar = None
        self._refresh_lists()
        self._apply(select=selected)

    def _poll(self) -> None:
        if self.project is None or not self.isVisible() or self.busy:
            return
        if metadata_signature(self.project) != self._sig:
            self.reload()

    def showEvent(self, e):
        super().showEvent(e)
        if self.project is not None and metadata_signature(self.project) != self._sig:
            self.reload()

    def closeEvent(self, e):
        self._timer.stop()
        super().closeEvent(e)

    def shutdown(self) -> None:
        self._timer.stop()
        if self.loader is not None:
            self.loader.stop()

    # ---- 左側清單與篩選器 ----------------------------------------------------------
    def _refresh_lists(self) -> None:
        lang = current_language()
        live = [r for r in self.all_records if not r.deleted]
        counts: dict[str, int] = defaultdict(int)
        for r in live:
            counts[r.class_id] += 1
        trash = len(self.all_records) - len(live)
        self.class_list.blockSignals(True)
        self.class_list.clear()
        items = [(ALL, tr("gui.dataset.all", count=len(live)), None)]
        for spec in (self.project.classes if self.project else []):
            items.append((spec.id, f"{spec.display(lang)} ({counts.get(spec.id, 0)})", spec.color))
        items.append((DELETED, tr("gui.dataset.deleted", count=trash), None))
        current = None
        for key, text, color in items:
            it = QListWidgetItem(text)
            it.setData(Qt.ItemDataRole.UserRole, key)
            if color:
                it.setIcon(QIcon(color_pixmap(color)))
            self.class_list.addItem(it)
            if key == self.class_filter:
                current = it
        self.class_list.setCurrentItem(current or self.class_list.item(0))
        self.class_list.blockSignals(False)
        self._rebuild_move_menu(lang)
        # 篩選器選項
        self.camera_combo.blockSignals(True)
        self.burst_combo.blockSignals(True)
        cam_prev, burst_prev = self.camera_combo.currentData(), self.burst_combo.currentData()
        self.camera_combo.clear()
        self.camera_combo.addItem(tr("gui.dataset.filter.all"), None)
        names = {}
        for r in live:
            names.setdefault(record_camera(r), str(r.source.get("name") or record_camera(r)))
        for cid, name in sorted(names.items()):
            self.camera_combo.addItem(name, cid)
        self.camera_combo.setCurrentIndex(max(0, self.camera_combo.findData(cam_prev)))
        self.burst_combo.clear()
        self.burst_combo.addItem(tr("gui.dataset.filter.all"), "all")
        self.burst_combo.addItem(tr("gui.dataset.filter.burst_any"), "any")
        bursts: dict[str, int] = defaultdict(int)
        for r in live:
            if r.burst_id:
                bursts[r.burst_id] += 1
        for bid, n in sorted(bursts.items())[:300]:
            self.burst_combo.addItem(tr("gui.dataset.filter.burst_item", id=bid[:12], count=n), bid)
        self.burst_combo.setCurrentIndex(max(0, self.burst_combo.findData(burst_prev)))
        self.camera_combo.blockSignals(False)
        self.burst_combo.blockSignals(False)
        self.camera_filter = self.camera_combo.currentData()
        self.burst_filter = self.burst_combo.currentData() or "all"

    def _rebuild_move_menu(self, lang: str) -> None:
        self.move_menu.clear()
        for spec in (self.project.classes if self.project else []):
            label = spec.display(lang) + (f"  [{spec.hotkey}]" if spec.hotkey else "")
            act = self.move_menu.addAction(QIcon(color_pixmap(spec.color)), label)
            act.triggered.connect(lambda _c=False, cid=spec.id: self.relabel_selected(cid))

    def _on_class_item(self, item, _prev=None) -> None:
        if item is not None:
            self.set_class_filter(item.data(Qt.ItemDataRole.UserRole))

    def set_class_filter(self, key: str) -> None:
        self.class_filter = key
        for i in range(self.class_list.count()):
            if self.class_list.item(i).data(Qt.ItemDataRole.UserRole) == key:
                self.class_list.blockSignals(True)
                self.class_list.setCurrentRow(i)
                self.class_list.blockSignals(False)
        self._apply()

    def _on_filter(self) -> None:
        self.camera_filter = self.camera_combo.currentData()
        self.quality_filter = self.quality_combo.currentData() or "all"
        self.burst_filter = self.burst_combo.currentData() or "all"
        self.sort_mode = self.sort_combo.currentData() or "time_desc"
        self._apply()

    def set_camera_filter(self, camera: str | None) -> None:
        self.camera_combo.setCurrentIndex(max(0, self.camera_combo.findData(camera)))

    def set_quality_filter(self, key: str) -> None:
        self.quality_combo.setCurrentIndex(max(0, self.quality_combo.findData(key)))

    def set_burst_filter(self, key: str) -> None:
        self.burst_combo.setCurrentIndex(max(0, self.burst_combo.findData(key)))

    def set_sort(self, key: str) -> None:
        self.sort_combo.setCurrentIndex(max(0, self.sort_combo.findData(key)))

    # ---- 套用篩選與排序 -------------------------------------------------------------
    def visible_records(self) -> list[SampleRecord]:
        deleted = self.class_filter == DELETED
        out = []
        for r in self.all_records:
            if r.deleted != deleted:
                continue
            if self.class_filter not in (ALL, DELETED) and r.class_id != self.class_filter:
                continue
            if self.camera_filter is not None and record_camera(r) != self.camera_filter:
                continue
            if self.quality_filter != "all" and self.quality_filter not in quality_flags(r):
                continue
            if self.burst_filter == "any" and not r.burst_id:
                continue
            if self.burst_filter not in ("all", "any") and r.burst_id != self.burst_filter:
                continue
            out.append(r)
        key = {"time_desc": (record_time, True), "time_asc": (record_time, False),
               "sharp_desc": (record_sharpness, True), "sharp_asc": (record_sharpness, False)}[self.sort_mode]
        out.sort(key=lambda r: (key[0](r), r.file), reverse=key[1])
        return out

    def _apply(self, select: set[str] | None = None) -> None:
        if self.model is None:
            self.count_label.setText("")
            return
        records = self.visible_records()
        if self.similar_mode and self.class_filter != DELETED:
            if self._similar is None:
                self._start_similar(records)
            else:
                order = {r.file: r for r in records}
                grouped = []
                for group in self._similar:
                    grouped.extend(order[f] for f in group if f in order)
                records = grouped
                self.status_label.setText(tr("gui.dataset.similar.found", groups=len(self._similar),
                                             count=len(records)))
        self.model.set_records(records)
        if select:
            self.select_files(select)
        self.count_label.setText(tr("gui.dataset.count", shown=len(records), total=sum(
            1 for r in self.all_records if not r.deleted)))
        self._update_buttons()
        self._show_current()

    # ---- 找相似照片 -----------------------------------------------------------------
    def _on_similar_toggled(self, on: bool) -> None:
        self.similar_mode = on
        self._similar = None
        if not on:
            self.status_label.setText("")
        self._apply()

    def _on_similar_distance(self, _v: int) -> None:
        if self.similar_mode:
            self._similar = None
            self._apply()

    def find_similar(self, on: bool = True) -> None:
        self.similar_button.setChecked(on)

    def _start_similar(self, records: list[SampleRecord]) -> None:
        distance = self.similar_distance.value()
        self.status_label.setText(tr("gui.dataset.similar.working"))
        files = [r.file for r in records]

        def work(_task):
            pairs = find_near_duplicates(records, distance)
            parent = {f: f for f in files}

            def root(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for a, b, _d in pairs:
                parent[root(a.file)] = root(b.file)
            groups: dict[str, list[str]] = defaultdict(list)
            for f in files:
                groups[root(f)].append(f)
            return [g for g in groups.values() if len(g) > 1]

        task = BgTask(work, self)
        self._similar_task = task
        task.finished.connect(lambda res, err, t=task: self._similar_done(t, res, err))
        task.start()

    def _similar_done(self, task, result, error) -> None:
        if task is not self._similar_task:
            return
        self._similar_task = None
        if error is not None:
            self.status_label.setText(str(error))
            return
        self._similar = result
        self._apply()

    # ---- 選取與預覽 -----------------------------------------------------------------
    def selected_records(self) -> list[SampleRecord]:
        if self.model is None or self.view.selectionModel() is None:
            return []
        rows = sorted(i.row() for i in self.view.selectionModel().selectedIndexes())
        return [self.model.records[r] for r in rows if r < len(self.model.records)]

    def select_files(self, files) -> None:
        if self.model is None:
            return
        files = set(files)
        sel = self.view.selectionModel()
        sel.clear()
        first = None
        for row, r in enumerate(self.model.records):
            if r.file in files:
                idx = self.model.index(row)
                sel.select(idx, sel.SelectionFlag.Select)
                first = first or idx
        if first is not None:
            sel.setCurrentIndex(first, sel.SelectionFlag.NoUpdate)
            self.view.scrollTo(first)

    def open_sample(self, file: str) -> None:
        """從別處（例如側邊面板的最近縮圖）跳到某一張。"""
        rec = next((r for r in self.all_records if r.file == file), None)
        if rec is None:
            self.reload()
            rec = next((r for r in self.all_records if r.file == file), None)
        if rec is not None:
            self.set_class_filter(DELETED if rec.deleted else ALL)
            self.select_files([file])
        self.show()
        self.raise_()
        self.activateWindow()

    def _on_selection(self, *_a) -> None:
        self._update_buttons()
        self._show_current()

    def _show_current(self) -> None:
        recs = self.selected_records()
        if not recs:
            self.preview.set_image(None)
            self.meta.setPlainText(tr("gui.dataset.nothing_selected"))
            self.processed_button.setEnabled(False)
            return
        r = recs[-1]
        self.processed_button.setEnabled(bool(r.processed_file))
        want = self.processed_button.isChecked() and r.processed_file
        path = (self.project.dataset_dir / (r.processed_file if want else r.file)) if self.project else None
        img = decode_file(path) if path is not None else None
        self.preview.set_image(QPixmap.fromImage(bgr_to_qimage(img)) if img is not None else None)
        text = describe_record(r, self.project)
        if len(recs) > 1:
            text = tr("gui.dataset.multi_selected", count=len(recs)) + "\n\n" + text
        self.meta.setPlainText(text)

    def _update_buttons(self) -> None:
        sel = bool(self.selected_records()) and not self.busy
        in_trash = self.class_filter == DELETED
        self.move_button.setEnabled(sel and not in_trash)
        self.delete_button.setEnabled(sel and not in_trash)
        self.restore_button.setVisible(in_trash)
        self.restore_button.setEnabled(sel and in_trash)
        self.delete_button.setVisible(not in_trash)
        self.move_button.setVisible(not in_trash)
        self.reveal_button.setEnabled(bool(self.selected_records()))

    def _context_menu(self, pos) -> None:
        if not self.selected_records():
            return
        menu = QMenu(self)
        if self.class_filter == DELETED:
            menu.addAction(tr("gui.dataset.restore"), self.restore_selected)
        else:
            sub = menu.addMenu(tr("gui.dataset.move"))
            for act in self.move_menu.actions():
                sub.addAction(act)
            menu.addAction(tr("common.delete"), self.delete_selected)
        menu.addAction(tr("gui.dataset.reveal"), self.reveal_selected)
        menu.exec(self.view.viewport().mapToGlobal(pos))

    # ---- 動作（背景執行，避免數百張時卡住） --------------------------------------------
    def _run_op(self, func_name: str, files: list[str], message_key: str, **fmt) -> None:
        if self.busy or self.writer is None or not files:
            return
        self.busy = True
        self._update_buttons()
        writer = self.writer
        arg = fmt.pop("arg", None)

        def work(_task):
            failed = []
            for f in files:
                try:
                    getattr(writer, func_name)(*((f, arg) if arg is not None else (f,)))
                except (OSError, ValueError, KeyError) as exc:
                    failed.append(str(exc))
            return failed

        task = BgTask(work, self)
        self._tasks.append(task)

        def done(failed, error, t=task):
            self.busy = False
            if t in self._tasks:
                self._tasks.remove(t)
            if error is not None:
                failed = [str(error)]
            msg = tr(message_key, count=len(files) - len(failed), **fmt)
            if failed:
                msg += "\n" + tr("gui.dataset.op_failed", count=len(failed), reason=failed[0])
            self.status_label.setText(msg)
            self.reload()
            self.datasetModified.emit()

        task.finished.connect(done)
        task.start()

    def relabel_selected(self, class_id: str) -> None:
        if self.class_filter == DELETED or self.project is None:
            return
        spec = self.project.class_by_id(class_id)
        files = [r.file for r in self.selected_records() if r.class_id != class_id]
        if spec is not None:
            self._run_op("relabel", files, "gui.dataset.relabeled", arg=class_id,
                         name=spec.display(current_language()))

    def _on_drop(self, class_id: str, files: list) -> None:
        if self.class_filter == DELETED:
            return
        self.select_files(files)
        self.relabel_selected(class_id)

    def delete_selected(self) -> None:
        if self.class_filter == DELETED:
            return
        self._run_op("delete", [r.file for r in self.selected_records()], "gui.dataset.deleted_done")

    def restore_selected(self) -> None:
        if self.class_filter != DELETED:
            return
        self._run_op("restore", [r.file for r in self.selected_records()], "gui.dataset.restored")

    def reveal_selected(self) -> None:
        recs = self.selected_records()
        if recs:
            reveal_in_folder(recs[-1].path)
