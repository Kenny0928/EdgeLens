"""VideoView：等比例顯示畫面，疊上處理步驟的圖形、推論結果、ROI 與 HUD。

所有座標都是「原始畫面像素座標」，換算到螢幕由 view 負責。繪製邏輯集中在
`_paint_annotations`，paintEvent（螢幕）與 render_annotated（原解析度存檔）共用。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from edgelens.core.project import DEFAULT_COLORS, ClassSpec
from edgelens.core.results import InferenceResult, Overlay
from edgelens.gui.theme import accent_color, hud_font
from edgelens.i18n import tr

BACKGROUND = QColor("#16181B")


def _color(value: str, alpha: int = 255) -> QColor:
    c = QColor(value)
    if not c.isValid():
        c = QColor("#00FF00")
    c.setAlpha(alpha)
    return c


class _ErrorBanner(QFrame):
    retryRequested = Signal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setObjectName("errorBanner")
        self.setStyleSheet("#errorBanner{background:rgba(30,30,30,230);border:2px solid #D13438;border-radius:8px;}"
                           "QLabel{color:#FFFFFF;}")
        lay = QVBoxLayout(self)
        self.title = QLabel()
        self.title.setWordWrap(True)
        f = self.title.font()
        f.setBold(True)
        self.title.setFont(f)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.button = QPushButton()
        self.button.clicked.connect(self.retryRequested)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.button)
        lay.addWidget(self.title)
        lay.addWidget(self.hint)
        lay.addLayout(row)
        self.hide()

    def show_error(self, message: str, hint: str, retryable: bool) -> None:
        self.title.setText(message)
        self.hint.setText(hint)
        self.hint.setVisible(bool(hint))
        self.button.setText(tr("common.retry"))
        self.button.setVisible(retryable)
        self.show()
        self.raise_()


class VideoView(QWidget):
    pointPicked = Signal(float, float)
    roiSelected = Signal(int, int, int, int)
    retryRequested = Signal()
    activated = Signal()  # 使用者點了這個畫面（多畫面時用來切換作用中相機）

    MODES = ("none", "pick", "roi")

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(160, 120)
        self.setMouseTracking(False)
        self._image: QImage | None = None
        self._overlays: list[Overlay] = []
        self._mask_images: dict[int, QImage] = {}
        self._results: list[InferenceResult] = []
        self._class_colors: list[str] = list(DEFAULT_COLORS)
        self._class_names: list[str] = []
        self._roi: tuple[int, int, int, int] | None = None
        self._hud: list[str] = []
        self._show_hud = True
        self._placeholder = tr("gui.video.no_image")
        self._active = False
        self._mode = "none"
        self._drag_start: QPointF | None = None
        self._drag_now: QPointF | None = None
        self._reference: QImage | None = None
        self._reference_opacity = 0.4
        self._banner = _ErrorBanner(self)
        self._banner.retryRequested.connect(self.retryRequested)

    # ---- 設定內容 ---------------------------------------------------------------
    def set_image(self, image: QImage | None) -> None:
        self._image = image
        self.update()

    def set_overlays(self, overlays: list[Overlay] | None) -> None:
        self._overlays = list(overlays or [])
        self._mask_images = {}
        for i, ov in enumerate(self._overlays):
            if ov.kind == "mask":
                mask = ov.data.get("mask")
                if isinstance(mask, np.ndarray) and mask.ndim == 2:
                    self._mask_images[i] = self._mask_to_qimage(mask, ov.color)
        self.update()

    def set_inference(self, results: InferenceResult | list[InferenceResult] | None) -> None:
        if results is None:
            self._results = []
        elif isinstance(results, InferenceResult):
            self._results = [results]
        else:
            self._results = list(results)
        self.update()

    def set_class_colors(self, colors: list[str]) -> None:
        self._class_colors = list(colors) or list(DEFAULT_COLORS)
        self.update()

    def set_classes(self, classes: list[ClassSpec], lang: str = "zh_TW") -> None:
        """依專案類別設定顏色與顯示名稱（class_id 即類別在清單中的位置）。"""
        self._class_colors = [c.color for c in classes] or list(DEFAULT_COLORS)
        self._class_names = [c.display(lang) for c in classes]
        self.update()

    def set_roi(self, roi: tuple[int, int, int, int] | None) -> None:
        self._roi = roi
        self.update()

    def set_hud(self, lines: list[str]) -> None:
        self._hud = list(lines)
        self.update()

    def set_reference_image(self, image: QImage | None, opacity: float = 0.4) -> None:
        """構圖參考圖：以半透明疊在畫面上（只畫在螢幕上，不會存進標註圖）；None 表示關閉。"""
        self._reference = image if image is not None and not image.isNull() else None
        self._reference_opacity = max(0.0, min(1.0, float(opacity)))
        self.update()

    def set_hud_visible(self, visible: bool) -> None:
        self._show_hud = visible
        self.update()

    def set_active(self, active: bool) -> None:
        self._active = active
        self.update()

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self.update()

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(mode)
        self._mode = mode
        self._drag_start = self._drag_now = None
        self.setCursor(Qt.CursorShape.CrossCursor if mode != "none" else Qt.CursorShape.ArrowCursor)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def image(self) -> QImage | None:
        return self._image

    def show_error(self, message: str, hint: str = "", retryable: bool = False) -> None:
        self._banner.show_error(message, hint, retryable)
        self._layout_banner()

    def clear_error(self) -> None:
        self._banner.hide()

    def error_visible(self) -> bool:
        return not self._banner.isHidden()

    def retranslate(self) -> None:
        self._banner.button.setText(tr("common.retry"))
        self.update()

    # ---- 座標換算 ---------------------------------------------------------------
    def _image_rect(self) -> QRectF:
        """影像在 widget 內的顯示範圍（等比例縮放、置中）。"""
        if self._image is None or self._image.isNull():
            return QRectF()
        iw, ih = self._image.width(), self._image.height()
        w, h = self.width(), self.height()
        s = min(w / iw, h / ih)
        dw, dh = iw * s, ih * s
        return QRectF((w - dw) / 2, (h - dh) / 2, dw, dh)

    def widget_to_image(self, p: QPointF) -> tuple[float, float] | None:
        r = self._image_rect()
        if r.isEmpty() or self._image is None:
            return None
        s = r.width() / self._image.width()
        return (p.x() - r.x()) / s, (p.y() - r.y()) / s

    def image_to_widget(self, x: float, y: float) -> QPointF:
        r = self._image_rect()
        s = r.width() / self._image.width() if self._image is not None and not r.isEmpty() else 1.0
        return QPointF(r.x() + x * s, r.y() + y * s)

    # ---- 繪製 -------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), BACKGROUND)
        if self._image is None or self._image.isNull():
            p.setPen(QColor("#9AA0A6"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self._placeholder)
        else:
            r = self._image_rect()
            p.drawImage(r, self._image)
            if self._reference is not None:
                p.setOpacity(self._reference_opacity)
                p.drawImage(r, self._reference)
                p.setOpacity(1.0)
            s = r.width() / self._image.width()
            self._paint_annotations(p, s, r.x(), r.y(), font_px=max(12.0, self.font().pointSizeF() * 1.4))
            self._paint_drag(p)
        if self._active:
            pen = QPen(accent_color(), 4)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(self.rect().adjusted(2, 2, -2, -2))
        p.end()

    def _paint_drag(self, p: QPainter) -> None:
        if self._mode == "roi" and self._drag_start is not None and self._drag_now is not None:
            p.setPen(QPen(QColor("#00DC00"), 2, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(self._drag_start, self._drag_now).normalized())

    def _class_color(self, class_id: int) -> QColor:
        colors = self._class_colors or list(DEFAULT_COLORS)
        return _color(colors[class_id % len(colors)])

    def _paint_annotations(self, p: QPainter, s: float, ox: float, oy: float, font_px: float) -> None:
        """在「影像座標 × s + (ox, oy)」的位置畫所有疊加。font_px 是文字高度（像素）。"""

        def pt(x: float, y: float) -> QPointF:
            return QPointF(ox + x * s, oy + y * s)

        font = QFont(p.font())
        font.setPixelSize(int(font_px))
        font.setBold(True)
        p.setFont(font)
        fm = p.fontMetrics()

        def label(x: float, y: float, text: str, color: QColor) -> None:
            if not text:
                return
            w = fm.horizontalAdvance(text) + 8
            h = fm.height() + 2
            top = max(0.0, y - h)
            p.fillRect(QRectF(x, top, w, h), color)
            lum = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
            p.setPen(QColor("#000000") if lum > 140 else QColor("#FFFFFF"))
            p.drawText(QRectF(x + 4, top, w, h), Qt.AlignmentFlag.AlignVCenter, text)

        # 處理步驟的 overlays
        for i, ov in enumerate(self._overlays):
            col = _color(ov.color)
            pen = QPen(col, max(1, ov.thickness))
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            d = ov.data
            try:
                if ov.kind == "box":
                    x1, y1, x2, y2 = d["box"]
                    a, b = pt(x1, y1), pt(x2, y2)
                    p.drawRect(QRectF(a, b))
                    label(a.x(), a.y(), ov.label, col)
                elif ov.kind in ("polygon", "polyline"):
                    pts = [pt(x, y) for x, y in d["points"]]
                    if len(pts) >= 2:
                        if ov.kind == "polygon":
                            p.drawPolygon(QPolygonF(pts))
                        else:
                            p.drawPolyline(QPolygonF(pts))
                        label(pts[0].x(), pts[0].y(), ov.label, col)
                elif ov.kind == "point":
                    c = pt(*d["xy"])
                    r = float(d.get("radius", 4))
                    p.setBrush(col)
                    p.drawEllipse(c, r, r)
                    label(c.x() + r, c.y(), ov.label, col)
                elif ov.kind == "text":
                    c = pt(*d["xy"])
                    p.setPen(col)
                    p.drawText(c, str(d.get("text", ov.label)))
                elif ov.kind == "mask" and i in self._mask_images and self._image is not None:
                    target = QRectF(ox, oy, self._image.width() * s, self._image.height() * s)
                    p.drawImage(target, self._mask_images[i])
            except (KeyError, TypeError, ValueError):
                continue  # 外掛步驟給了不合格式的資料時略過，不要讓畫面壞掉

        # ROI
        if self._roi is not None:
            x, y, w, h = self._roi
            p.setPen(QPen(QColor("#00DC00"), 2, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(pt(x, y), pt(x + w, y + h)))

        # 推論結果
        banner_y = None
        for res in self._results:
            dashed = res.where == "device"
            if res.kind == "detect":
                for det in res.detections:
                    col = self._class_color(det.class_id)
                    pen = QPen(col, 3, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
                    p.setPen(pen)
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    x1, y1, x2, y2 = det.box
                    a, b = pt(x1, y1), pt(x2, y2)
                    p.drawRect(QRectF(a, b))
                    name = det.label or self._name_of(det.class_id)
                    label(a.x(), a.y(), f"{name} {det.score:.2f}", col)
            elif res.kind == "classify" and res.top_k:
                best = res.top_k[0] if len(res.top_k) == 1 else max(res.top_k, key=lambda c: c.score)
                col = self._class_color(best.class_id)
                name = best.label or self._name_of(best.class_id)
                tag = "[device] " if dashed else ""
                if banner_y is None:
                    banner_y = oy + (self._image.height() * s if self._image is not None else 0) - 6
                label(ox + 6, banner_y, f"{tag}{name} {best.score:.2f}", col)  # 標籤底邊對齊 banner_y
                banner_y -= fm.height() + 6

        # HUD
        if self._show_hud and self._hud:
            hf = hud_font()
            hf.setPixelSize(int(font_px))
            p.setFont(hf)
            hm = p.fontMetrics()
            lh = hm.height() + 2
            w = max(hm.horizontalAdvance(t) for t in self._hud) + 16
            p.fillRect(QRectF(ox + 6, oy + 6, w, lh * len(self._hud) + 8), QColor(0, 0, 0, 150))
            p.setPen(QColor("#FFFFFF"))
            for k, line in enumerate(self._hud):
                p.drawText(QPointF(ox + 14, oy + 6 + 4 + lh * (k + 1) - hm.descent()), line)

    def _name_of(self, class_id: int) -> str:
        if 0 <= class_id < len(self._class_names):
            return self._class_names[class_id]
        return str(class_id)

    @staticmethod
    def _mask_to_qimage(mask: np.ndarray, color: str) -> QImage:
        c = _color(color)
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), np.uint8)
        on = mask > 0
        rgba[on] = (c.red(), c.green(), c.blue(), 110)
        return QImage(rgba.data, w, h, rgba.strides[0], QImage.Format.Format_RGBA8888).copy()

    def render_annotated(self) -> QImage:
        """原解析度、含疊加與 HUD 的圖（存「標註圖」用）。沒有畫面時回傳空 QImage。"""
        if self._image is None or self._image.isNull():
            return QImage()
        out = self._image.convertToFormat(QImage.Format.Format_RGB32)
        p = QPainter(out)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._paint_annotations(p, 1.0, 0.0, 0.0, font_px=max(12.0, out.height() / 32.0))
        p.end()
        return out

    # ---- 互動 -------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.activated.emit()
        if event.button() == Qt.MouseButton.LeftButton and self._mode != "none":
            self._drag_start = event.position()
            self._drag_now = event.position()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_start is not None:
            self._drag_now = event.position()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_start is None or event.button() != Qt.MouseButton.LeftButton:
            return
        start, end = self._drag_start, event.position()
        self._drag_start = self._drag_now = None
        self.update()
        a, b = self.widget_to_image(start), self.widget_to_image(end)
        if a is None or b is None or self._image is None:
            return
        iw, ih = self._image.width(), self._image.height()
        if self._mode == "pick":
            x, y = b
            if 0 <= x < iw and 0 <= y < ih:
                self.pointPicked.emit(float(x), float(y))
        elif self._mode == "roi":
            x1, x2 = sorted((min(max(a[0], 0), iw), min(max(b[0], 0), iw)))
            y1, y2 = sorted((min(max(a[1], 0), ih), min(max(b[1], 0), ih)))
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                self.roiSelected.emit(int(x1), int(y1), int(x2 - x1), int(y2 - y1))

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._layout_banner()
        super().resizeEvent(event)

    def _layout_banner(self) -> None:
        w = min(self.width() - 20, 460)
        self._banner.setFixedWidth(max(w, 100))
        self._banner.adjustSize()
        self._banner.move((self.width() - self._banner.width()) // 2,
                          max(10, (self.height() - self._banner.height()) // 2))


__all__ = ["VideoView"]
