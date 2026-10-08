"""Panel：右側面板的基底類別。

寫新面板：在 `edgelens/gui/panels/` 新增一個模組，定義 Panel 子類別並設定 panel_id、
title_key、order，主視窗啟動時會自動找到並放進右側分頁，不用修改 main_window.py。

    class MyPanel(Panel):
        panel_id = "my_panel"
        title_key = "gui.panel.my.title"
        order = 50

        def __init__(self, ctx, main):
            super().__init__(ctx, main)
            ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from PySide6.QtWidgets import QWidget

from edgelens.i18n import tr

if TYPE_CHECKING:
    from edgelens.gui.context import AppContext
    from edgelens.gui.main_window import MainWindow
    from edgelens.gui.workers import FramePacket, SourceHandle


class Panel(QWidget):
    panel_id: ClassVar[str] = ""
    title_key: ClassVar[str] = ""
    order: ClassVar[int] = 100   # 數字小的排在前面
    icon: ClassVar[str] = ""     # 選用：圖示名稱或檔案路徑

    def __init__(self, ctx: AppContext, main: MainWindow):
        super().__init__()
        self.ctx = ctx
        self.main = main

    @property
    def title(self) -> str:
        return tr(self.title_key) if self.title_key else self.panel_id

    def on_active_source_changed(self, handle: SourceHandle | None) -> None:
        """作用中相機改變（handle 可能是 None）。"""

    def on_project_changed(self, project: Any) -> None:
        """專案改變（project 可能是 None）。"""

    def on_frame(self, packet: FramePacket) -> None:
        """作用中相機有新畫面時呼叫（在 GUI 執行緒）。要很快，不要在這裡做耗時工作。"""

    def retranslate(self) -> None:
        """語言切換時呼叫，重新設定所有文字。"""

    def shutdown(self) -> None:
        """關閉程式前呼叫，停止自己開的執行緒與計時器。"""
