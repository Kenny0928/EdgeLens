"""圖形介面進入點：run(args) 啟動；create_app() 給測試用（不執行事件迴圈、不跳對話框）。"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from typing import Any

from PySide6.QtCore import QTimer
from PySide6.QtGui import QGuiApplication
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


def _activate_macos_app() -> None:
    """[[NSApplication sharedApplication] activateIgnoringOtherApps:YES]，用 ctypes 呼叫，不需要 pyobjc。

    從終端機（run_macos.command）跑的 python 不是 .app，macOS 14 起不會自動被帶到前景，
    視窗會開在別的 App 後面，學生以為「視窗沒有跳出來」。"""
    import ctypes
    import ctypes.util

    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc") or "/usr/lib/libobjc.A.dylib")
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    send = objc.objc_msgSend  # arm64 不能用 variadic 呼叫：每次都要先設好確切的參數型別
    send.restype = ctypes.c_void_p
    send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    ns_app = send(objc.objc_getClass(b"NSApplication"), objc.sel_registerName(b"sharedApplication"))
    if not ns_app:
        return
    send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
    send(ns_app, objc.sel_registerName(b"activateIgnoringOtherApps:"), True)


def _bring_to_front(window: Any = None) -> None:
    """把視窗與整個程式帶到最前面。失敗只記錄，絕不擋啟動。"""
    try:
        if window is not None:
            window.raise_()
            window.activateWindow()
        if sys.platform == "darwin" and QGuiApplication.platformName() == "cocoa":
            _activate_macos_app()
    except Exception as e:  # noqa: BLE001
        log.debug("無法把視窗帶到前景：%s", e)


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
    if interactive:
        _bring_to_front()  # 第一次啟動的工作區對話框、外掛確認視窗會比主視窗先跳出來，也要看得到

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
    _bring_to_front(window)
    code = app.exec()
    ctx.save_settings()
    return int(code)
