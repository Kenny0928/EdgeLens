"""拍照面板：蒐集訓練樣本的主要介面。

設計重點：
- 拍照（空白鍵）一定要立刻有回應，所以存檔放在背景執行緒，完成後才更新張數與訊息。
- 快捷鍵註冊在主視窗（面板不是目前分頁也能用），焦點在輸入框時不處理，才不會打字時誤拍。
- 品質檢查與靜止偵測都只在 on_frame 裡做很輕的運算，不拖慢畫面。
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QObject, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from edgelens.core.paths import atomic_write_bytes, safe_filename
from edgelens.dataset.dedupe import RecentHashes
from edgelens.dataset.stats import dataset_stats
from edgelens.dataset.writer import DatasetWriter, shared_writer
from edgelens.gui.capture_tools import StillDetector, new_id, thresholds_from_settings
from edgelens.gui.panels.base import Panel
from edgelens.gui.stable import StableLabel
from edgelens.gui.workers import FramePacket, SourceHandle
from edgelens.i18n import current_language, tr
from edgelens.metrics.quality import measure, verdict

log = logging.getLogger(__name__)

QUALITY_INTERVAL = 0.5   # 品質檢查的間隔（秒），不要每張畫面都算
REFERENCE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ---------------------------------------------------------------- 小元件

class _ViewOverlay(QWidget):
    """蓋在畫面上的透明層：拍照閃白框、大的倒數數字。不攔截滑鼠。"""

    def __init__(self, view: QWidget):
        super().__init__(view)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self._view = view
        self._flash = 0.0
        self._text = ""
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._fade)
        view.installEventFilter(self)
        self.setGeometry(view.rect())
        self.show()
        self.raise_()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if obj is self._view and event.type() == QEvent.Type.Resize:
            self.setGeometry(self._view.rect())
        return False

    def flash(self) -> None:
        self._flash = 1.0
        self.raise_()
        self._timer.start()
        self.update()

    def set_text(self, text: str) -> None:
        self._text = text
        self.raise_()
        self.update()

    @property
    def flash_level(self) -> float:
        return self._flash

    @property
    def text(self) -> str:
        return self._text

    def _fade(self) -> None:
        self._flash = max(0.0, self._flash - 0.1)
        if self._flash <= 0.0:
            self._timer.stop()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        if self._flash <= 0.0 and not self._text:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._flash > 0.0:
            pen = QPen(QColor(255, 255, 255, int(255 * self._flash)), 16)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(self.rect().adjusted(8, 8, -8, -8))
            p.fillRect(self.rect(), QColor(255, 255, 255, int(70 * self._flash)))
        if self._text:
            size = max(40, min(self.width(), self.height()) // 2)
            font = QFont(p.font())
            font.setPixelSize(size)
            font.setBold(True)
            p.setFont(font)
            r = QRectF(self.rect())
            p.setPen(QColor(0, 0, 0, 160))
            p.drawText(r.translated(3, 3), Qt.AlignmentFlag.AlignCenter, self._text)
            p.setPen(QColor("#FFFFFF"))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, self._text)
        p.end()


class BalanceChart(QWidget):
    """每個類別張數的小長條圖（QPainter 自繪）；最少的類別用粗框標出來。"""

    ROW_H = 22

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._items: list[tuple[str, str, int]] = []
        self._weak: int | None = None
        self.setMinimumHeight(self.ROW_H)

    def set_data(self, items: list[tuple[str, str, int]], weak_index: int | None) -> None:
        self._items = list(items)
        self._weak = weak_index
        self.setFixedHeight(max(1, len(items)) * self.ROW_H + 4)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self._items:
            return
        fm = p.fontMetrics()
        label_w = min(110, max(fm.horizontalAdvance(n) for n, _, _ in self._items) + 8)
        count_w = fm.horizontalAdvance(str(max(c for _, _, c in self._items))) + 10
        bar_x = label_w
        bar_w = max(10, self.width() - label_w - count_w - 4)
        top = max(c for _, _, c in self._items) or 1
        for i, (name, color, count) in enumerate(self._items):
            y = 2 + i * self.ROW_H
            p.setPen(self.palette().text().color())
            p.drawText(QRectF(0, y, label_w - 4, self.ROW_H), Qt.AlignmentFlag.AlignVCenter, fm.elidedText(
                name, Qt.TextElideMode.ElideRight, label_w - 6))
            w = bar_w * count / top
            rect = QRectF(bar_x, y + 3, max(2.0, w) if count else 2.0, self.ROW_H - 6)
            p.fillRect(rect, QColor(color))
            if i == self._weak:
                p.setPen(QPen(QColor("#D13438"), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(rect.adjusted(-1, -1, 1, 1))
            p.setPen(self.palette().text().color())
            p.drawText(QRectF(bar_x + (w if count else 2) + 4, y, count_w + 20, self.ROW_H),
                       Qt.AlignmentFlag.AlignVCenter, str(count))
        p.end()


@dataclass
class _Job:
    writer: DatasetWriter
    handle: SourceHandle
    class_id: str
    frame: Any
    image: Any
    pipeline: dict
    source: Any
    controls: dict
    result: Any
    group_id: str | None
    burst_id: str | None
    record: Any = None
    error: str = ""
    extra: dict = field(default_factory=dict)


class _Saver(QObject):
    """背景存檔完成後，用 signal 回到 GUI 執行緒。"""

    done = Signal(object)


# ---------------------------------------------------------------- 面板

class CapturePanel(Panel):
    panel_id = "capture"
    title_key = "gui.panel.capture.title"
    order = 40

    def __init__(self, ctx, main):
        super().__init__(ctx, main)
        self._handle: SourceHandle | None = None
        self._class_id = ""
        self._recent: dict[str, RecentHashes] = {}
        self._counts: dict[str, int] = {}
        self._buttons: dict[str, QPushButton] = {}
        self._overlays: dict[SourceHandle, _ViewOverlay] = {}
        self._pending = 0
        self._pending_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="edgelens-capture")
        self._saver = _Saver()
        self._saver.done.connect(self._on_saved)
        self._last_quality = 0.0
        self._still = StillDetector(self._s("still_threshold", 2.0), self._s("still_frames", 8))
        self._burst_left = 0
        self._burst_total = 0
        self._burst_id: str | None = None
        self._burst_timer = QTimer(self)
        self._burst_timer.setSingleShot(True)
        self._burst_timer.timeout.connect(self._burst_tick)
        self._cd_left = 0
        self._cd_tick_ms = 1000
        self._cd_timer = QTimer(self)
        self._cd_timer.timeout.connect(self._cd_tick)
        self._ref_image: QImage | None = None
        self.last_quality_ok: bool | None = None
        self.last_saved_text = ""
        self.saved_total = 0

        self._build_ui()
        self._shortcuts = [main.register_shortcut("Space", self._on_space)]
        for k in range(1, 10):
            self._shortcuts.append(main.register_shortcut(str(k), lambda key=str(k): self._on_digit(key)))
        ctx.sourcesChanged.connect(self._apply_reference)
        self.retranslate()
        self.on_project_changed(ctx.project)
        self.on_active_source_changed(ctx.active_source)

    # ---- 設定 -------------------------------------------------------------------
    def _s(self, key: str, default: Any) -> Any:
        v = self.ctx.settings.get(f"capture.{key}", default)
        return default if v is None else v

    def _set(self, key: str, value: Any) -> None:
        self.ctx.settings.set(f"capture.{key}", value)
        self.ctx.save_settings_later()

    # ---- 建立介面 ---------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        lay = QVBoxLayout(body)

        # 沒有專案／類別／相機的提示
        self.project_hint = QLabel()
        self.project_hint.setWordWrap(True)
        self.project_button = QPushButton()
        self.project_button.clicked.connect(self._open_project_settings)
        self.camera_hint = QLabel()
        self.camera_hint.setWordWrap(True)
        for w in (self.project_hint, self.project_button, self.camera_hint):
            lay.addWidget(w)

        # 類別按鈕
        self.class_box = QWidget()
        self.class_grid = QGridLayout(self.class_box)
        self.class_grid.setContentsMargins(0, 0, 0, 0)
        self.class_group = QButtonGroup(self)
        self.class_group.setExclusive(True)
        lay.addWidget(self.class_box)

        # 拍照按鈕
        self.shoot_button = QPushButton()
        self.shoot_button.setMinimumHeight(54)
        f = self.shoot_button.font()
        f.setPointSize(f.pointSize() + 3)
        f.setBold(True)
        self.shoot_button.setFont(f)
        self.shoot_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.shoot_button.clicked.connect(lambda: self.capture_now())
        lay.addWidget(self.shoot_button)
        self.all_button = QPushButton()
        self.all_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.all_button.clicked.connect(lambda: self.capture_all())
        lay.addWidget(self.all_button)
        self.saved_label = StableLabel(1)
        lay.addWidget(self.saved_label)

        # 品質
        self.quality_box = QGroupBox()
        ql = QVBoxLayout(self.quality_box)
        # 品質每秒更新幾次、相近照片提醒拍照時才出現：高度都固定，版面不跳動
        self.quality_label = StableLabel(1, "-")
        self.quality_verdict = StableLabel(2)
        self.block_check = QCheckBox()
        self.block_check.setChecked(bool(self._s("block_bad_quality", False)))
        self.block_check.toggled.connect(lambda v: self._set("block_bad_quality", bool(v)))
        self.similar_label = StableLabel(2)
        self.similar_label.setStyleSheet("color:#C77700;font-weight:bold;")
        for w in (self.quality_label, self.quality_verdict, self.block_check, self.similar_label):
            ql.addWidget(w)
        lay.addWidget(self.quality_box)

        # 連拍／定時／靜止
        self.modes_box = QGroupBox()
        form = QFormLayout(self.modes_box)
        self.burst_count = QSpinBox()
        self.burst_count.setRange(2, 100)
        self.burst_count.setValue(int(self._s("burst_count", 5)))
        self.burst_interval = QSpinBox()
        self.burst_interval.setRange(20, 10000)
        self.burst_interval.setSingleStep(50)
        self.burst_interval.setSuffix(" ms")
        self.burst_interval.setValue(int(self._s("burst_interval_ms", 300)))
        self.burst_count.valueChanged.connect(lambda v: self._set("burst_count", int(v)))
        self.burst_interval.valueChanged.connect(lambda v: self._set("burst_interval_ms", int(v)))
        self.burst_button = QPushButton()
        self.burst_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.burst_button.clicked.connect(self._toggle_burst)
        self.burst_count_label = QLabel()
        self.burst_interval_label = QLabel()
        form.addRow(self.burst_count_label, self.burst_count)
        form.addRow(self.burst_interval_label, self.burst_interval)
        form.addRow(self.burst_button)
        self.timer_seconds = QSpinBox()
        self.timer_seconds.setRange(1, 60)
        self.timer_seconds.setSuffix(" s")
        self.timer_seconds.setValue(int(self._s("timer_seconds", 3)))
        self.timer_seconds.valueChanged.connect(lambda v: self._set("timer_seconds", int(v)))
        self.timer_button = QPushButton()
        self.timer_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.timer_button.clicked.connect(self._toggle_timer)
        self.timer_label = QLabel()
        form.addRow(self.timer_label, self.timer_seconds)
        form.addRow(self.timer_button)
        self.still_check = QCheckBox()
        self.still_check.toggled.connect(self._on_still_toggled)
        self.still_threshold = QDoubleSpinBox()
        self.still_threshold.setRange(0.1, 50.0)
        self.still_threshold.setSingleStep(0.5)
        self.still_threshold.setValue(float(self._s("still_threshold", 2.0)))
        self.still_frames = QSpinBox()
        self.still_frames.setRange(2, 200)
        self.still_frames.setValue(int(self._s("still_frames", 8)))
        self.still_threshold.valueChanged.connect(self._on_still_params)
        self.still_frames.valueChanged.connect(self._on_still_params)
        self.still_state = QLabel()
        self.still_threshold_label = QLabel()
        self.still_frames_label = QLabel()
        form.addRow(self.still_check)
        form.addRow(self.still_threshold_label, self.still_threshold)
        form.addRow(self.still_frames_label, self.still_frames)
        form.addRow(self.still_state)
        self.mode_status = StableLabel(2)
        form.addRow(self.mode_status)
        lay.addWidget(self.modes_box)

        # 構圖參考
        self.ref_box = QGroupBox()
        rl = QVBoxLayout(self.ref_box)
        row = QHBoxLayout()
        self.ref_pick = QPushButton()
        self.ref_pick.clicked.connect(self._pick_reference)
        self.ref_remove = QPushButton()
        self.ref_remove.clicked.connect(self.clear_reference)
        row.addWidget(self.ref_pick)
        row.addWidget(self.ref_remove)
        rl.addLayout(row)
        self.ref_show = QCheckBox()
        self.ref_show.toggled.connect(self._on_ref_toggled)
        self.ref_name = QLabel()
        self.ref_name.setWordWrap(True)
        self.ref_opacity = QSlider(Qt.Orientation.Horizontal)
        self.ref_opacity.setRange(5, 95)
        self.ref_opacity.setValue(int(float(self._s("reference_opacity", 0.4)) * 100))
        self.ref_opacity.valueChanged.connect(self._on_ref_opacity)
        self.ref_opacity_label = QLabel()
        for w in (self.ref_name, self.ref_show, self.ref_opacity_label, self.ref_opacity):
            rl.addWidget(w)
        lay.addWidget(self.ref_box)

        # 類別平衡
        self.balance_box = QGroupBox()
        bl = QVBoxLayout(self.balance_box)
        self.balance = BalanceChart()
        self.balance_hint = StableLabel(2)
        self.balance_hint.setStyleSheet("color:#C77700;font-weight:bold;")
        bl.addWidget(self.balance)
        bl.addWidget(self.balance_hint)
        lay.addWidget(self.balance_box)

        # 進階
        self.adv_toggle = QToolButton()
        self.adv_toggle.setCheckable(True)
        self.adv_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.adv_toggle.toggled.connect(self._on_adv_toggled)
        lay.addWidget(self.adv_toggle)
        self.adv_box = QWidget()
        af = QFormLayout(self.adv_box)
        self.save_processed = QCheckBox()
        self.save_processed.setChecked(bool(self._s("save_processed", True)))
        self.save_processed.toggled.connect(lambda v: self._set("save_processed", bool(v)))
        self.q_sharp = self._spin("min_sharpness", 0, 5000, 50.0, 1)
        self.q_min_b = self._spin("min_brightness", 0, 255, 30.0, 0)
        self.q_max_b = self._spin("max_brightness", 0, 255, 225.0, 0)
        self.q_over = self._spin("max_overexposed", 0, 1, 0.2, 2)
        self.q_over.setSingleStep(0.05)
        self.dup_distance = QSpinBox()
        self.dup_distance.setRange(0, 32)
        self.dup_distance.setValue(int(self._s("similar_distance", 5)))
        self.dup_distance.valueChanged.connect(lambda v: self._set("similar_distance", int(v)))
        self._adv_labels = {k: QLabel() for k in ("sharp", "min_b", "max_b", "over", "dup")}
        af.addRow(self.save_processed)
        af.addRow(self._adv_labels["sharp"], self.q_sharp)
        af.addRow(self._adv_labels["min_b"], self.q_min_b)
        af.addRow(self._adv_labels["max_b"], self.q_max_b)
        af.addRow(self._adv_labels["over"], self.q_over)
        af.addRow(self._adv_labels["dup"], self.dup_distance)
        self.adv_box.hide()
        lay.addWidget(self.adv_box)
        lay.addStretch(1)

    def _spin(self, key: str, lo: float, hi: float, default: float, decimals: int) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setDecimals(decimals)
        s.setRange(lo, hi)
        s.setValue(float(self._s(f"quality.{key}", default)))
        s.valueChanged.connect(lambda v, k=key: self._set(f"quality.{k}", float(v)))
        return s

    # ---- 專案與類別 -------------------------------------------------------------
    def on_project_changed(self, project: Any) -> None:
        self._recent = {}
        self._counts = {}
        self._cancel_modes()
        self._rebuild_class_buttons()
        self.reload_counts()
        self._load_reference()
        self._update_state()

    def current_class(self):
        p = self.ctx.project
        return p.class_by_id(self._class_id) if p is not None and self._class_id else None

    def _rebuild_class_buttons(self) -> None:
        for b in list(self._buttons.values()):
            self.class_group.removeButton(b)
            self.class_grid.removeWidget(b)
            b.deleteLater()
        self._buttons = {}
        p = self.ctx.project
        classes = p.classes if p is not None else []
        wanted = self._s("class_id", "")
        if self._class_id not in [c.id for c in classes]:
            self._class_id = wanted if wanted in [c.id for c in classes] else (classes[0].id if classes else "")
        for i, c in enumerate(classes):
            b = QPushButton()
            b.setCheckable(True)
            b.setMinimumHeight(58)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda _=False, cid=c.id: self.set_class(cid))
            self.class_group.addButton(b)
            self.class_grid.addWidget(b, i // 2, i % 2)
            self._buttons[c.id] = b
        self._refresh_class_buttons()

    @staticmethod
    def _text_on(color: QColor) -> str:
        lum = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
        return "#000000" if lum > 140 else "#FFFFFF"

    def _refresh_class_buttons(self) -> None:
        p = self.ctx.project
        if p is None:
            return
        lang = current_language()
        for c in p.classes:
            b = self._buttons.get(c.id)
            if b is None:
                continue
            col = QColor(c.color)
            if not col.isValid():
                col = QColor("#999999")
            key = f"{c.hotkey}  " if c.hotkey else ""
            b.setText(f"{key}{c.display(lang)}\n" + tr("gui.capture.count", n=self._counts.get(c.id, 0)))
            selected = c.id == self._class_id
            b.setChecked(selected)
            border = "4px solid #FFFFFF" if selected else "2px solid rgba(0,0,0,60)"
            outline = "outline:2px solid #111;" if selected else ""
            b.setStyleSheet(f"QPushButton{{background:{col.name()};color:{self._text_on(col)};font-weight:bold;"
                            f"border:{border};border-radius:8px;padding:4px;{outline}}}")
            b.setToolTip(c.description or c.display(lang))

    def set_class(self, class_id: str) -> None:
        p = self.ctx.project
        if p is None or p.class_by_id(class_id) is None:
            return
        self._class_id = class_id
        self._set("class_id", class_id)
        self._refresh_class_buttons()
        self._update_state()

    def _on_digit(self, key: str) -> None:
        if self._typing():
            return
        p = self.ctx.project
        if p is None:
            return
        for c in p.classes:
            if c.hotkey == key:
                self.set_class(c.id)
                return

    def _on_space(self) -> None:
        if self._typing():
            return
        self.capture_now()

    @staticmethod
    def _typing() -> bool:
        """焦點在可輸入文字的元件時，不處理快捷鍵。"""
        from PySide6.QtWidgets import QApplication
        w = QApplication.focusWidget()
        if isinstance(w, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
            return True
        return isinstance(w, QComboBox) and w.isEditable()

    def _open_project_settings(self) -> None:
        act = getattr(self.main, "act_project_settings", None)
        if act is not None:
            act.trigger()

    # ---- 資料集 -----------------------------------------------------------------
    def writer(self) -> DatasetWriter | None:
        """整個程式每個專案只有一個 writer（與資料集瀏覽器共用），避免互相蓋掉 metadata。
        是否存處理後影像由拍照時是否傳入 image 決定。"""
        p = self.ctx.project
        return shared_writer(p) if p is not None else None

    def reload_counts(self) -> None:
        """從資料集 metadata 讀出每類張數（專案切換時做一次，之後存檔完成再累加）。"""
        p = self.ctx.project
        self._counts = {c.id: 0 for c in (p.classes if p else [])}
        w = self.writer()
        if w is not None and p is not None:
            try:
                stats = dataset_stats(w.list_samples(), p.class_ids())
                self._counts.update({k: v for k, v in stats.class_counts.items() if k in self._counts})
            except (OSError, ValueError) as e:
                log.warning("讀取資料集張數失敗：%s", e)
        self._refresh_class_buttons()
        self._refresh_balance()

    def _refresh_balance(self) -> None:
        p = self.ctx.project
        classes = p.classes if p is not None else []
        lang = current_language()
        items = [(c.display(lang), c.color, self._counts.get(c.id, 0)) for c in classes]
        weak = None
        hint = ""
        counts = [n for _, _, n in items]
        if len(items) >= 2 and max(counts) > 0:
            lo = min(counts)
            if lo * 2 < max(counts):
                weak = counts.index(lo)
                hint = tr("gui.capture.balance_low", name=items[weak][0])
        self.balance.set_data(items, weak)
        self.balance_hint.setText(hint)
        self.balance_box.setVisible(bool(items))

    def count_of(self, class_id: str) -> int:
        return self._counts.get(class_id, 0)

    @property
    def pending(self) -> int:
        return self._pending

    # ---- 狀態顯示 ---------------------------------------------------------------
    def _ready(self) -> str:
        """回傳不能拍照的原因鍵；空字串表示可以拍。"""
        p = self.ctx.project
        if p is None or not p.classes:
            return "gui.capture.need_project"
        if self.current_class() is None:
            return "gui.capture.need_class"
        h = self._handle
        if h is None or h.last_packet is None:
            return "gui.capture.need_camera"
        return ""

    def _update_state(self) -> None:
        p = self.ctx.project
        no_class = p is None or not p.classes
        self.project_hint.setVisible(no_class)
        self.project_button.setVisible(no_class)
        self.class_box.setVisible(not no_class)
        self.camera_hint.setVisible(not no_class and self._handle is None)
        ready = bool(self._handle is not None and not no_class and self.current_class() is not None)
        busy = self._busy()
        self.shoot_button.setEnabled(ready)
        multi = len(self.main.handles()) > 1
        self.all_button.setEnabled(ready and multi)
        self.all_button.setVisible(multi)
        self.burst_button.setEnabled(ready or self._burst_left > 0)
        self.timer_button.setEnabled(ready or self._cd_left > 0)
        self.modes_box.setEnabled(not no_class)
        cls = self.current_class()
        if cls is not None:
            self.shoot_button.setText(tr("gui.capture.shoot", name=cls.display(current_language())))
        else:
            self.shoot_button.setText(tr("gui.capture.shoot_none"))
        self.burst_button.setText(tr("gui.capture.burst_cancel", k=self._burst_total - self._burst_left,
                                     n=self._burst_total) if self._burst_left > 0 else tr("gui.capture.burst_start"))
        self.timer_button.setText(tr("gui.capture.timer_cancel", n=self._cd_left) if self._cd_left > 0
                                  else tr("gui.capture.timer_start"))
        if busy:
            self.burst_button.setEnabled(self._burst_left > 0)
            self.timer_button.setEnabled(self._cd_left > 0)

    def _busy(self) -> bool:
        return self._burst_left > 0 or self._cd_left > 0

    # ---- 切換相機 ---------------------------------------------------------------
    def on_active_source_changed(self, handle: SourceHandle | None) -> None:
        self._cancel_modes()
        self._handle = handle
        self._still.reset()
        self.quality_label.setText("-")
        self.quality_verdict.setText("")
        self.last_quality_ok = None
        self._apply_reference()
        self._update_state()

    # ---- 每張畫面 ---------------------------------------------------------------
    def on_frame(self, packet: FramePacket) -> None:
        now = time.monotonic()
        if self.still_check.isChecked() and not self._busy():
            if self._still.update(packet.frame.image):
                self.capture_now(auto=True)
        if now - self._last_quality >= QUALITY_INTERVAL:
            self._last_quality = now
            self._update_quality(packet)
            if not self.shoot_button.isEnabled() and self._handle is not None:
                self._update_state()
        if self.still_check.isChecked():
            self.still_state.setText(tr("gui.capture.still_state", k=min(self._still.still_count, self._still.frames),
                                        n=self._still.frames, d=f"{self._still.last_diff:.1f}"))

    def _update_quality(self, packet: FramePacket) -> None:
        try:
            rep = measure(packet.frame.image)
        except ValueError:
            return
        ok, reasons = verdict(rep, thresholds_from_settings(self.ctx.settings))
        self.last_quality_ok = ok
        self.quality_label.setText(tr("gui.capture.quality", sharp=f"{rep.sharpness:.0f}",
                                      bright=f"{rep.brightness:.0f}", over=f"{rep.overexposed * 100:.0f}"))
        if ok:
            self.quality_verdict.setText("✓ " + tr("gui.capture.quality_ok"))
            self.quality_verdict.setStyleSheet("color:#2E8B57;")
        else:
            self.quality_verdict.setText("⚠ " + tr("gui.list_sep").join(tr(r) for r in reasons))
            self.quality_verdict.setStyleSheet("color:#C77700;font-weight:bold;")

    # ---- 拍照 -------------------------------------------------------------------
    def capture_now(self, auto: bool = False) -> bool:
        """用目前的類別為作用中相機拍一張。回傳是否已送出存檔。"""
        why = self._ready()
        if why:
            if not auto:
                self.ctx.show_message(tr(why))
            return False
        return self._capture(self._handle, self._class_id, warn_dup=True)

    def capture_all(self) -> int:
        """所有相機各拍一張，用同一個 group_id；回傳成功送出的張數。"""
        if self.current_class() is None:
            self.ctx.show_message(tr(self._ready() or "gui.capture.need_class"))
            return 0
        gid = new_id()
        n = 0
        for h in self.main.handles():
            if h.last_packet is not None and self._capture(h, self._class_id, group_id=gid, warn_dup=False):
                n += 1
        return n

    def _last_inference(self, handle: SourceHandle):
        fn = getattr(self.main, "last_inference", None)
        if not callable(fn):
            return None
        try:
            res = fn(handle)
        except Exception:  # noqa: BLE001 - 推論結果拿不到不能影響拍照
            return None
        if isinstance(res, (list, tuple)):
            res = res[0] if res else None
        return res

    def _capture(self, handle: SourceHandle, class_id: str, *, group_id: str | None = None,
                 burst_id: str | None = None, warn_dup: bool = True) -> bool:
        packet = handle.last_packet
        writer = self.writer()
        if packet is None or writer is None:
            return False
        frame = packet.frame
        thresholds = thresholds_from_settings(self.ctx.settings)
        if self.block_check.isChecked():
            try:
                ok, reasons = verdict(measure(frame.image), thresholds)
            except ValueError:
                ok, reasons = True, []
            if not ok:
                self.ctx.show_message(tr("gui.capture.blocked", reason=tr("gui.list_sep").join(
                    tr(r) for r in reasons)))
                return False
        if warn_dup:
            self._check_similar(frame.image, class_id)
        image = None
        if self.save_processed.isChecked() and packet.result.image is not frame.image:
            image = packet.result.image
        job = _Job(writer=writer, handle=handle, class_id=class_id, frame=frame, image=image,
                   pipeline=dict(handle.pipeline_dict), source=handle.info, controls=dict(handle.values),
                   result=self._last_inference(handle), group_id=group_id, burst_id=burst_id)
        with self._pending_lock:
            self._pending += 1
        self._pool.submit(self._run_job, job)
        return True

    def _run_job(self, job: _Job) -> None:
        try:
            job.record = job.writer.save(job.frame, job.class_id, image=job.image, result=job.result,
                                         pipeline=job.pipeline, source=job.source, controls=job.controls,
                                         group_id=job.group_id, burst_id=job.burst_id)
        except Exception as e:  # noqa: BLE001 - 存檔失敗要告訴使用者，不能讓執行緒默默結束
            log.exception("存檔失敗")
            job.error = str(e)
        try:
            self._saver.done.emit(job)
        except RuntimeError:
            pass  # 面板已被關閉

    def _on_saved(self, job: _Job) -> None:
        with self._pending_lock:
            self._pending = max(0, self._pending - 1)
        if job.record is None:
            self.last_saved_text = tr("gui.capture.save_failed", error=job.error)
            self.saved_label.setText(self.last_saved_text)
            self.ctx.show_message(self.last_saved_text, 8000)
            return
        cid = job.class_id
        self._counts[cid] = self._counts.get(cid, 0) + 1
        self.saved_total += 1
        self._recent.setdefault(cid, RecentHashes(50)).add(job.record)
        p = self.ctx.project
        spec = p.class_by_id(cid) if p is not None else None
        name = spec.display(current_language()) if spec is not None else cid
        self.last_saved_text = tr("gui.capture.saved", name=name, n=self._counts[cid])
        self.saved_label.setText("✓ " + self.last_saved_text)
        self.ctx.show_message(self.last_saved_text, 3000)
        ov = self._overlay(job.handle)
        if ov is not None:
            ov.flash()
        self._refresh_class_buttons()
        self._refresh_balance()
        self.ctx.datasetChanged.emit()

    def _check_similar(self, image: np.ndarray, class_id: str) -> bool:
        rh = self._recent.get(class_id)
        hit = rh.check(image) if rh is not None else None
        similar = hit is not None and hit[1] <= self.dup_distance.value()
        self.similar_label.setText(tr("gui.capture.similar") if similar else "")
        return similar

    # ---- 畫面疊加 ---------------------------------------------------------------
    def _overlay(self, handle: SourceHandle | None) -> _ViewOverlay | None:
        if handle is None:
            return None
        view = self.main.video_view_for(handle)
        if view is None:
            return None
        ov = self._overlays.get(handle)
        try:
            if ov is not None and ov.parent() is view:
                return ov
        except RuntimeError:
            pass
        ov = _ViewOverlay(view)
        self._overlays[handle] = ov
        return ov

    # ---- 連拍 -------------------------------------------------------------------
    def _toggle_burst(self) -> None:
        if self._burst_left > 0:
            self.cancel_burst()
        else:
            self.start_burst()

    def start_burst(self) -> bool:
        why = self._ready()
        if why:
            self.ctx.show_message(tr(why))
            return False
        if self._busy():
            return False
        self._burst_total = self._burst_left = int(self.burst_count.value())
        self._burst_id = new_id()
        self._burst_tick()
        return True

    def _burst_tick(self) -> None:
        if self._burst_left <= 0 or self._handle is None:
            return
        self._burst_left -= 1
        self._capture(self._handle, self._class_id, burst_id=self._burst_id, warn_dup=False)
        if self._burst_left > 0:
            self._burst_timer.start(int(self.burst_interval.value()))
            self.mode_status.setText(tr("gui.capture.burst_progress", k=self._burst_total - self._burst_left,
                                        n=self._burst_total))
        else:
            self.mode_status.setText(tr("gui.capture.burst_done", n=self._burst_total))
        self._update_state()

    def cancel_burst(self) -> None:
        if self._burst_left > 0:
            self._burst_timer.stop()
            done = self._burst_total - self._burst_left
            self._burst_left = 0
            self.mode_status.setText(tr("gui.capture.burst_cancelled", k=done))
            self._update_state()

    # ---- 定時拍 -----------------------------------------------------------------
    def _toggle_timer(self) -> None:
        if self._cd_left > 0:
            self.cancel_countdown()
        else:
            self.start_countdown()

    def start_countdown(self) -> bool:
        why = self._ready()
        if why:
            self.ctx.show_message(tr(why))
            return False
        if self._busy():
            return False
        self._cd_left = int(self.timer_seconds.value())
        self._show_countdown()
        self._cd_timer.start(self._cd_tick_ms)
        self._update_state()
        return True

    def _show_countdown(self) -> None:
        ov = self._overlay(self._handle)
        if ov is not None:
            ov.set_text(str(self._cd_left) if self._cd_left > 0 else "")
        if self._handle is not None:
            self.main.set_hud_line(self._handle, "capture",
                                   tr("gui.capture.countdown", n=self._cd_left) if self._cd_left > 0 else None)

    def _cd_tick(self) -> None:
        self._cd_left -= 1
        if self._cd_left > 0:
            self._show_countdown()
            self._update_state()
            return
        self._cd_timer.stop()
        self._cd_left = 0
        self._show_countdown()
        self.capture_now()
        self._update_state()

    def cancel_countdown(self) -> None:
        if self._cd_left > 0 or self._cd_timer.isActive():
            self._cd_timer.stop()
            self._cd_left = 0
            self._show_countdown()
            self._update_state()

    def _cancel_modes(self) -> None:
        self.cancel_burst()
        self.cancel_countdown()

    # ---- 靜止自動拍 -------------------------------------------------------------
    def _on_still_toggled(self, on: bool) -> None:
        self._still.reset()
        self.still_state.setText("" if not on else tr("gui.capture.still_wait"))

    def _on_still_params(self) -> None:
        self._still.configure(self.still_threshold.value(), self.still_frames.value())
        self._set("still_threshold", float(self.still_threshold.value()))
        self._set("still_frames", int(self.still_frames.value()))

    # ---- 構圖參考 ---------------------------------------------------------------
    def _ref_key(self) -> str:
        p = self.ctx.project
        return f"reference.{p.id}" if p is not None else "reference._"

    def _ref_path(self) -> Path | None:
        p = self.ctx.project
        name = self._s(self._ref_key(), "")
        if p is None or not name:
            return None
        path = p.dir("references") / str(name)
        return path if path.is_file() else None

    def _load_reference(self) -> None:
        path = self._ref_path()
        self._ref_image = None
        if path is not None:
            try:
                img = QImage.fromData(path.read_bytes())
                self._ref_image = img if not img.isNull() else None
            except OSError:
                self._ref_image = None
        self.ref_name.setText(path.name if self._ref_image is not None else tr("gui.capture.ref_none"))
        self.ref_show.blockSignals(True)
        self.ref_show.setChecked(self._ref_image is not None and bool(self._s("reference_on", True)))
        self.ref_show.blockSignals(False)
        self.ref_show.setEnabled(self._ref_image is not None)
        self.ref_remove.setEnabled(self._ref_image is not None)
        self._apply_reference()

    def _apply_reference(self) -> None:
        img = self._ref_image
        show = img is not None and self.ref_show.isChecked()
        opacity = self.ref_opacity.value() / 100.0
        for h in self.main.handles():
            view = self.main.video_view_for(h)
            if view is not None:
                view.set_reference_image(img if show else None, opacity)

    def set_reference_file(self, path: str | Path) -> bool:
        """把一張照片複製到專案 references/ 並當作構圖參考。"""
        p = self.ctx.project
        if p is None:
            self.ctx.show_message(tr("gui.capture.need_project"))
            return False
        src = Path(path)
        try:
            data = src.read_bytes()
            arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                raise ValueError("decode")
        except (OSError, ValueError):
            self.ctx.show_message(tr("gui.capture.ref_bad", name=src.name))
            return False
        folder = p.dir("references")
        name = safe_filename(src.stem) + (src.suffix.lower() if src.suffix.lower() in REFERENCE_EXTS else ".jpg")
        dest = folder / name
        i = 1
        while dest.exists() and dest.read_bytes() != data:
            i += 1
            dest = folder / f"{Path(name).stem}_{i}{Path(name).suffix}"
        if not dest.exists():
            atomic_write_bytes(dest, data)
        self._set(self._ref_key(), dest.name)
        self._set("reference_on", True)
        self._load_reference()
        return True

    def _pick_reference(self) -> None:
        p = self.ctx.project
        if p is None:
            self.ctx.show_message(tr("gui.capture.need_project"))
            return
        start = str(p.dataset_dir)
        path, _ = QFileDialog.getOpenFileName(self, tr("gui.capture.ref_pick"), start,
                                              tr("gui.capture.ref_filter") + " (*.jpg *.jpeg *.png *.bmp *.webp)")
        if path:
            self.set_reference_file(path)

    def clear_reference(self) -> None:
        self._set(self._ref_key(), "")
        self._load_reference()

    def _on_ref_toggled(self, on: bool) -> None:
        self._set("reference_on", bool(on))
        self._apply_reference()

    def _on_ref_opacity(self, value: int) -> None:
        self._set("reference_opacity", value / 100.0)
        self._apply_reference()

    # ---- 進階 -------------------------------------------------------------------
    def _on_adv_toggled(self, on: bool) -> None:
        self.adv_box.setVisible(on)
        self.adv_toggle.setText(("▾ " if on else "▸ ") + tr("common.advanced"))

    # ---- 語言 -------------------------------------------------------------------
    def retranslate(self) -> None:
        self.project_hint.setText(tr("gui.capture.no_classes"))
        self.project_button.setText(tr("gui.capture.open_project"))
        self.camera_hint.setText(tr("gui.capture.need_camera"))
        self.all_button.setText(tr("gui.capture.shoot_all"))
        self.quality_box.setTitle(tr("gui.capture.quality_title"))
        self.block_check.setText(tr("gui.capture.block_bad"))
        self.modes_box.setTitle(tr("gui.capture.modes"))
        self.burst_count_label.setText(tr("gui.capture.burst_count"))
        self.burst_interval_label.setText(tr("gui.capture.burst_interval"))
        self.timer_label.setText(tr("gui.capture.timer_seconds"))
        self.still_check.setText(tr("gui.capture.still"))
        self.still_check.setToolTip(tr("gui.capture.still_tip"))
        self.still_threshold_label.setText(tr("gui.capture.still_threshold"))
        self.still_frames_label.setText(tr("gui.capture.still_frames"))
        self.ref_box.setTitle(tr("gui.capture.ref_title"))
        self.ref_pick.setText(tr("gui.capture.ref_pick"))
        self.ref_remove.setText(tr("gui.capture.ref_remove"))
        self.ref_show.setText(tr("gui.capture.ref_show"))
        self.ref_opacity_label.setText(tr("gui.capture.ref_opacity"))
        self.balance_box.setTitle(tr("gui.capture.balance"))
        self.save_processed.setText(tr("gui.capture.save_processed"))
        for k, lab in self._adv_labels.items():
            lab.setText(tr(f"gui.capture.adv.{k}"))
        self._on_adv_toggled(self.adv_toggle.isChecked())
        if self._ref_image is None:
            self.ref_name.setText(tr("gui.capture.ref_none"))
        self._refresh_class_buttons()
        self._refresh_balance()
        self._update_state()

    def shutdown(self) -> None:
        self._cancel_modes()
        self._burst_timer.stop()
        self._cd_timer.stop()
        self._pool.shutdown(wait=True)
        # 等待中的完成通知不再處理，避免面板關閉後還更新介面
        try:
            self._saver.done.disconnect(self._on_saved)
        except (RuntimeError, TypeError):
            pass
        for view_handle in list(self._overlays):
            self._overlays.pop(view_handle, None)
