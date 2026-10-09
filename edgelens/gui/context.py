"""AppContext：整個介面共用的狀態（設定、工作區、專案、相機清單、語言）。

面板與對話框都透過 ctx 取得狀態並接 signal，不直接依賴主視窗的內部結構。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QTimer, Signal

from edgelens import i18n
from edgelens.core.project import Project, Workspace
from edgelens.core.settings import Settings

if TYPE_CHECKING:
    from edgelens.gui.workers import SourceHandle

log = logging.getLogger(__name__)


class AppContext(QObject):
    projectChanged = Signal(object)        # Project | None
    languageChanged = Signal(str)
    sourcesChanged = Signal()
    activeSourceChanged = Signal(object)   # SourceHandle | None
    statusMessage = Signal(str, int)       # 文字、顯示毫秒數（0 表示一直顯示）
    datasetChanged = Signal()              # 資料集有變動（拍照存檔、改類別、刪除…）；面板收到就重新讀取
    _languageBridge = Signal(str)

    def __init__(self, settings: Settings, workspace: Workspace, project: Project | None = None,
                 parent: QObject | None = None):
        super().__init__(parent)
        self.settings = settings
        self.workspace = workspace
        self.project = project
        self.sources: list[SourceHandle] = []
        self.active_source: SourceHandle | None = None
        self.inference = None   # InferenceManager（主視窗建立；面板用它管理背景推論）
        # i18n 的通知可能來自任何執行緒；經過自己的 signal 轉成 GUI 執行緒（排隊）觸發
        self._languageBridge.connect(self.languageChanged)
        self._remove_listener = i18n.on_language_changed(self._on_i18n_changed)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(1000)
        self._save_timer.timeout.connect(self.save_settings)

    def _on_i18n_changed(self, lang: str) -> None:
        try:
            self._languageBridge.emit(lang)
        except RuntimeError:  # C++ 物件已被刪除
            pass

    def dispose(self) -> None:
        self._remove_listener()

    # ---- 專案與語言 -------------------------------------------------------------
    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.settings.set("workspace.last_project", project.id if project else "")
        self.projectChanged.emit(project)
        self.save_settings_later()

    def set_language(self, lang: str) -> None:
        self.settings.set("ui.language", lang)
        i18n.set_language(lang)
        self.save_settings_later()

    # ---- 相機 -------------------------------------------------------------------
    def add_source(self, handle: SourceHandle) -> None:
        if handle not in self.sources:
            self.sources.append(handle)
            self.sourcesChanged.emit()

    def remove_source(self, handle: SourceHandle) -> None:
        if handle in self.sources:
            self.sources.remove(handle)
            self.sourcesChanged.emit()
        if self.active_source is handle:
            self.set_active_source(self.sources[0] if self.sources else None)

    def set_active_source(self, handle: SourceHandle | None) -> None:
        if handle is self.active_source:
            return
        self.active_source = handle
        self.activeSourceChanged.emit(handle)

    # ---- 設定與訊息 -------------------------------------------------------------
    def save_settings(self) -> None:
        try:
            self.settings.save()
        except OSError as e:
            log.warning("儲存設定失敗：%s", e)

    def save_settings_later(self) -> None:
        """合併短時間內的多次變更再寫檔（避免拖曳滑桿時狂寫檔案）。"""
        self._save_timer.start()

    def show_message(self, text: str, ms: int = 4000) -> None:
        self.statusMessage.emit(text, ms)
