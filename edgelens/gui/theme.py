"""字型與顏色。跟隨系統的淺色／深色模式，不寫死字型名稱。"""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication

# 中文備援字型：系統字型找不到字形時依序嘗試
CJK_FALLBACKS = ("Microsoft JhengHei UI", "PingFang TC", "Noto Sans CJK TC")


def apply_theme(app: QApplication) -> None:
    """設定應用程式字型（系統預設字型 + 中文備援）。深色模式由 Qt 依系統設定自動處理。"""
    font = QFont(app.font())
    families = [font.family(), *CJK_FALLBACKS]
    font.setFamilies(families)
    app.setFont(font)


def hud_font(point_size: float = 0) -> QFont:
    """HUD 用的等寬系統字型（不指定名稱，由系統選）。"""
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    font.setFamilies([font.family(), *CJK_FALLBACKS])
    if point_size:
        font.setPointSizeF(point_size)
    return font


def is_dark() -> bool:
    app = QGuiApplication.instance()
    if app is None:
        return False
    return QGuiApplication.palette().color(QPalette.ColorRole.Window).lightness() < 128


def status_color(kind: str) -> QColor:
    """狀態顏色：ok（綠）、warn（橘）、bad（紅）、muted（灰），淺色與深色模式各有對比足夠的色調。"""
    dark = is_dark()
    table = {
        "ok": ("#2E9E5B", "#5BD28A"),
        "warn": ("#C77700", "#FFB84D"),
        "bad": ("#D13438", "#FF6B6F"),
        "muted": ("#707070", "#A0A0A0"),
    }
    light_c, dark_c = table.get(kind, table["muted"])
    return QColor(dark_c if dark else light_c)


def accent_color() -> QColor:
    return QGuiApplication.palette().color(QPalette.ColorRole.Highlight)
