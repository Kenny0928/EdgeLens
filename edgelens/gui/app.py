"""圖形介面進入點：run(args) 啟動；create_app() 給測試用（不執行事件迴圈、不跳對話框）。"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from typing import Any

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from edgelens import APP_NAME, i18n
from edgelens.core.hangwatch import start_hang_watch
from edgelens.core.housekeeping import run_startup_housekeeping
from edgelens.core.log import current_log_file
from edgelens.core.paths import default_workspace
from edgelens.core.project import Project, Workspace
from edgelens.core.registry import discover
from edgelens.core.settings import Settings
from edgelens.gui.context import AppContext
from edgelens.gui.main_window import MainWindow
from edgelens.gui.plugin_trust import load_workspace_plugins
from edgelens.gui.theme import apply_theme
from edgelens.i18n import tr

log = logging.getLogger(__name__)

_in_excepthook = False


def _install_excepthook() -> None:
    """未捕捉的例外：寫入日誌並顯示對話框（含日誌檔位置），不要直接閃退。"""

    def hook(exc_type, exc, tb) -> None:
        global _in_excepthook
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        log.error("未捕捉的例外：\n%s", text)
        if _in_excepthook or QApplication.instance() is None:
            sys.__excepthook__(exc_type, exc, tb)
            return
        _in_excepthook = True
        try:
            logf = current_log_file()
            box = QMessageBox(QMessageBox.Icon.Critical, tr("gui.crash.title"),
                              tr("gui.crash.text", error=f"{exc_type.__name__}: {exc}",
                                 log=str(logf) if logf else tr("common.unknown")))
            box.setDetailedText(text)
            box.exec()
        finally:
            _in_excepthook = False

    sys.excepthook = hook


def _arg(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default) if args is not None else default


def _pick_workspace(args: Any, settings: Settings, interactive: bool) -> Workspace | None:
    """決定工作區：命令列 > 環境變數 > 設定 > （互動時）第一次啟動對話框 > 預設位置。"""
    chosen = _arg(args, "workspace")
    if chosen:
        return Workspace(chosen)
    env = os.environ.get("EDGELENS_WORKSPACE")
    if env:
        return Workspace(env)
    saved = settings.get("workspace.path", "")
    if saved:
        return Workspace(saved)
    if interactive:
        from edgelens.gui.workspace_dialog import WorkspaceDialog

        dlg = WorkspaceDialog(None, default_workspace(), first_run=True)
        if not dlg.exec():
            return None
        return Workspace(dlg.chosen_path())
    return Workspace(default_workspace())


def _initial_project(workspace: Workspace, settings: Settings) -> Project:
    last = settings.get("workspace.last_project", "")
    if last:
        try:
            return workspace.open_project(last)
        except (OSError, ValueError, KeyError):
            pass
    projects = workspace.projects()
    if projects:
        return projects[0]
    # 第一次使用：建立一個空白專案，學生可以直接開始，不用先學「專案」是什麼
    return workspace.create_project(tr("gui.project.default_name"), "blank", lang=i18n.current_language())


def create_app(args: Any = None, *, interactive: bool = False) -> tuple[QApplication, AppContext, MainWindow]:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    apply_theme(app)

    settings = Settings()
    lang = _arg(args, "lang") or settings.get("ui.language", i18n.DEFAULT_LANGUAGE)
    if lang not in i18n.LANGUAGES:
        lang = i18n.DEFAULT_LANGUAGE
    i18n.set_language(lang)
    settings.set("ui.language", lang)

    workspace = _pick_workspace(args, settings, interactive)
    if workspace is None:
        raise SystemExit(0)
    settings.set("workspace.path", str(workspace.root))
    safe_mode = bool(_arg(args, "safe_mode", False))
    discover()
    if not safe_mode and settings.get("plugins.enabled", True):
        # 外掛要在主視窗建立前載入（它們可能註冊處理步驟），所以確認視窗此時沒有父視窗
        load_workspace_plugins(settings, workspace.plugins_dir, interactive=interactive)

    project = _initial_project(workspace, settings)
    ctx = AppContext(settings, workspace, project)
    window = MainWindow(ctx, interactive=interactive, safe_mode=safe_mode)
    return app, ctx, window


def run(args: Any = None) -> int:
    app, ctx, window = create_app(args, interactive=True)
    _install_excepthook()
    # 只在真正啟動時清理：create_app() 給測試用，而暫存清理會掃系統的 temp 資料夾
    run_startup_housekeeping()
    # 卡住時把各執行緒的位置寫進日誌（學生只能強制關閉時，這是唯一的線索）
    log_file = current_log_file()
    watch = start_hang_watch(log_file.parent if log_file is not None else None)
    if watch is not None:
        beat = QTimer(window)
        beat.timeout.connect(watch.beat)
        beat.start(1000)
    window.show()
    code = app.exec()
    ctx.save_settings()
    return int(code)
