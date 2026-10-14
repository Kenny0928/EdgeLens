"""主視窗：選單、工具列、相機畫面（1 或最多 4 格）、右側面板、狀態列。

給面板作者的公開方法：handles()、active_handle()、video_view_for(handle)、add_menu_action()、
register_shortcut()、set_hud_line()、set_inference()、panel()。面板由 discover_panels() 自動載入。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QByteArray, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedLayout,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

import edgelens
from edgelens import i18n
from edgelens.core.paths import cloud_sync_provider
from edgelens.core.project import Project, Workspace
from edgelens.core.results import InferenceResult
from edgelens.gui.context import AppContext
from edgelens.gui.inference import InferenceManager, make_color_resolver
from edgelens.gui.panel_switcher import PanelSwitcher
from edgelens.gui.panels import discover_panels
from edgelens.gui.panels.base import Panel
from edgelens.gui.plugin_trust import load_workspace_plugins
from edgelens.gui.project_dialog import ProjectDialog
from edgelens.gui.source_dialog import SourceDialog
from edgelens.gui.video_view import VideoView
from edgelens.gui.workers import FramePacket, SourceHandle
from edgelens.gui.workspace_dialog import WorkspaceDialog
from edgelens.i18n import tr
from edgelens.sources.base import SourceInfo, info_for_storage

log = logging.getLogger(__name__)

MAX_SOURCES = 4
STOP_TIMEOUT = 2.0


class VideoGrid(QWidget):
    """中央畫面區：沒有相機時顯示說明與大按鈕；有相機時顯示 1 格或 2×2。"""

    addRequested = Signal()
    activated = Signal(object)  # SourceHandle

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._views: dict[SourceHandle, VideoView] = {}
        self._mode = "single"
        self._active: SourceHandle | None = None
        self._stack = QStackedLayout(self)

        empty = QWidget()
        el = QVBoxLayout(empty)
        el.addStretch(1)
        self.empty_title = QLabel()
        self.empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = self.empty_title.font()
        f.setPointSizeF(f.pointSizeF() * 1.6)
        f.setBold(True)
        self.empty_title.setFont(f)
        self.empty_text = QLabel()
        self.empty_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_text.setWordWrap(True)
        self.empty_button = QPushButton()
        bf = self.empty_button.font()
        bf.setPointSizeF(bf.pointSizeF() * 1.4)
        self.empty_button.setFont(bf)
        self.empty_button.setMinimumHeight(56)
        self.empty_button.setMinimumWidth(220)
        self.empty_button.clicked.connect(self.addRequested)
        el.addWidget(self.empty_title)
        el.addWidget(self.empty_text)
        row = QVBoxLayout()
        row.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        row.addWidget(self.empty_button, 0, Qt.AlignmentFlag.AlignHCenter)
        el.addLayout(row)
        el.addStretch(1)
        self._stack.addWidget(empty)

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(2, 2, 2, 2)
        self._grid.setSpacing(4)
        self._stack.addWidget(self._grid_host)
        self.retranslate()

    def retranslate(self) -> None:
        self.empty_title.setText(tr("gui.empty.title"))
        self.empty_text.setText(tr("gui.empty.text"))
        self.empty_button.setText(tr("gui.action.add_camera"))

    # ---- 畫面 -------------------------------------------------------------------
    def add_view(self, handle: SourceHandle) -> VideoView:
        view = VideoView()
        view.activated.connect(lambda h=handle: self.activated.emit(h))
        self._views[handle] = view
        self._relayout()
        return view

    def remove_view(self, handle: SourceHandle) -> None:
        view = self._views.pop(handle, None)
        if view is not None:
            self._grid.removeWidget(view)
            view.setParent(None)
            view.deleteLater()
        if self._active is handle:
            self._active = None
        self._relayout()

    def view_for(self, handle: SourceHandle) -> VideoView | None:
        return self._views.get(handle)

    def views(self) -> list[VideoView]:
        return list(self._views.values())

    def set_active(self, handle: SourceHandle | None) -> None:
        self._active = handle
        for h, v in self._views.items():
            v.set_active(h is handle and len(self._views) > 1)
        self._relayout()

    def set_layout_mode(self, mode: str) -> None:
        self._mode = mode
        self._relayout()

    @property
    def layout_mode(self) -> str:
        return self._mode

    def _relayout(self) -> None:
        for v in self._views.values():
            self._grid.removeWidget(v)
            v.hide()
        rows = 1 if len(self._views) <= 2 else 2  # 兩台以內並排一列，畫面才不會只佔上半部
        for i in range(2):
            self._grid.setRowStretch(i, 1 if i < rows else 0)
            self._grid.setColumnStretch(i, 1)
        if not self._views:
            self._stack.setCurrentIndex(0)
            return
        self._stack.setCurrentIndex(1)
        if self._mode == "single":
            handle = self._active if self._active in self._views else next(iter(self._views))
            self._grid.addWidget(self._views[handle], 0, 0, 2, 2)  # 跨滿 2×2 的格子
            self._views[handle].show()
        else:
            for i, v in enumerate(self._views.values()):
                self._grid.addWidget(v, i // 2, i % 2)
                v.show()


class MainWindow(QMainWindow):
    def __init__(self, ctx: AppContext, *, interactive: bool = False, safe_mode: bool = False):
        super().__init__()
        self.ctx = ctx
        self.interactive = interactive
        self.safe_mode = safe_mode
        self._hud_extra: dict[SourceHandle, dict[str, str]] = {}
        self._tr_items: list[tuple[Any, str, str]] = []
        self._menus: dict[str, QMenu] = {}
        self._show_processed = True
        self._show_hud = True
        self._last_status = 0.0
        self._panels: list[Panel] = []
        self._closing = False

        self.grid = VideoGrid()
        self.grid.addRequested.connect(self.add_camera_dialog)
        self.grid.activated.connect(self.set_active_handle)
        self.tabs = PanelSwitcher()   # 右側面板：直向清單＋面板內容（面板多時橫向分頁會擠不下）
        self.tabs.setMinimumWidth(440)
        splitter = QSplitter()
        splitter.addWidget(self.grid)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([820, 480])
        self.setCentralWidget(splitter)

        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_statusbar()

        ctx.projectChanged.connect(self._on_project_changed)
        ctx.languageChanged.connect(self._on_language_changed)
        ctx.activeSourceChanged.connect(self._on_active_changed)
        ctx.statusMessage.connect(lambda text, ms: self.statusBar().showMessage(text, ms))

        self.inference = InferenceManager(self)      # 背景推論；面板在建立時就會用到，所以要在載入面板之前
        ctx.inference = self.inference
        self._color_resolver = make_color_resolver(ctx)
        self.inference.resultReady.connect(self._on_inference_result)
        self.inference.sessionsChanged.connect(self._on_sessions_changed)

        self._load_panels()
        self.resize(1280, 760)
        self._restore_geometry()
        self.retranslate()
        self._on_project_changed(ctx.project)
        if interactive and not safe_mode:
            QTimer.singleShot(300, self._offer_reopen)

    # ---- 公開 API（給面板）------------------------------------------------------
    def handles(self) -> list[SourceHandle]:
        return list(self.ctx.sources)

    def active_handle(self) -> SourceHandle | None:
        return self.ctx.active_source

    def video_view_for(self, handle: SourceHandle) -> VideoView | None:
        return self.grid.view_for(handle)

    def panel(self, panel_id: str) -> Panel | None:
        return next((p for p in self._panels if p.panel_id == panel_id), None)

    def panels(self) -> list[Panel]:
        return list(self._panels)

    def add_menu_action(self, menu_id: str, action: QAction) -> QAction:
        """把動作加進選單。menu_id：file、camera、view、language、help；其他 ID 會建立新的頂層選單。"""
        menu = self._menus.get(menu_id)
        if menu is None:
            menu = self.menuBar().addMenu(menu_id)
            self._menus[menu_id] = menu
        menu.addAction(action)
        return action

    def add_menu_separator(self, menu_id: str) -> None:
        menu = self._menus.get(menu_id)
        if menu is not None:
            menu.addSeparator()

    def register_shortcut(self, key_sequence: str | QKeySequence, callback) -> QShortcut:
        """註冊整個視窗有效的快捷鍵（例如空白鍵拍照、1–9 切換類別）。"""
        sc = QShortcut(QKeySequence(key_sequence), self)
        sc.setContext(Qt.ShortcutContext.WindowShortcut)
        sc.activated.connect(callback)
        return sc

    def set_hud_line(self, handle: SourceHandle, key: str, text: str | None) -> None:
        """面板在 HUD 加一行文字（例如推論耗時）；text 為 None 或空字串則移除。"""
        lines = self._hud_extra.setdefault(handle, {})
        if text:
            lines[key] = text
        else:
            lines.pop(key, None)
        self._refresh_hud(handle)

    def set_inference(self, handle: SourceHandle, results: InferenceResult | list[InferenceResult] | None) -> None:
        view = self.grid.view_for(handle)
        if view is not None:
            view.set_inference(results)

    def last_inference(self, handle: SourceHandle) -> InferenceResult | None:
        """該相機主要模型最近一次的推論結果（拍照存 metadata 用）；沒有模型時為 None。"""
        return self.inference.last_result(handle)

    def show_message(self, text: str, ms: int = 4000) -> None:
        self.statusBar().showMessage(text, ms)

    # ---- 動作與選單 -------------------------------------------------------------
    def _action(self, key: str, slot=None, *, checkable: bool = False, shortcut: str | None = None) -> QAction:
        act = QAction(self)
        act.setCheckable(checkable)
        if slot is not None:
            act.triggered.connect(slot)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        self._tr_items.append((act, key, "text"))
        return act

    def _menu(self, menu_id: str, key: str) -> QMenu:
        menu = self.menuBar().addMenu("")
        self._menus[menu_id] = menu
        self._tr_items.append((menu, key, "title"))
        return menu

    def _build_actions(self) -> None:
        A = self._action
        self.act_new_project = A("gui.action.new_project", lambda: self.project_dialog(ProjectDialog.TAB_NEW),
                                 shortcut="Ctrl+N")
        self.act_open_project = A("gui.action.open_project", lambda: self.project_dialog(ProjectDialog.TAB_CHOOSE),
                                  shortcut="Ctrl+O")
        self.act_project_settings = A("gui.action.project_settings",
                                      lambda: self.project_dialog(ProjectDialog.TAB_CLASSES))
        self.act_workspace = A("gui.action.switch_workspace", self.switch_workspace)
        self.act_reveal = A("gui.action.reveal_project", self.reveal_project)
        self.act_quit = A("gui.action.quit", self.close, shortcut="Ctrl+Q")
        self.act_add_camera = A("gui.action.add_camera", self.add_camera_dialog, shortcut="Ctrl+Shift+N")
        self.act_close_camera = A("gui.action.close_camera", self.close_active_camera)
        self.act_retry = A("gui.action.retry_camera", self.retry_active_camera)
        self.act_single = A("gui.action.view_single", lambda: self.set_layout_mode("single"), checkable=True)
        self.act_grid = A("gui.action.view_grid", lambda: self.set_layout_mode("grid"), checkable=True)
        self.act_single.setChecked(True)
        lg = QActionGroup(self)
        lg.addAction(self.act_single)
        lg.addAction(self.act_grid)
        self.act_processed = A("gui.action.show_processed", lambda: self.set_show_processed(True), checkable=True)
        self.act_raw = A("gui.action.show_raw", lambda: self.set_show_processed(False), checkable=True)
        self.act_processed.setChecked(True)
        pg = QActionGroup(self)
        pg.addAction(self.act_processed)
        pg.addAction(self.act_raw)
        self.act_hud = A("gui.action.show_hud", self.set_show_hud, checkable=True)
        self.act_hud.setChecked(True)
        self.act_fullscreen = A("gui.action.fullscreen", self._toggle_fullscreen, checkable=True, shortcut="F11")
        self.act_zh = A("gui.lang.zh_TW", lambda: self.ctx.set_language("zh_TW"), checkable=True)
        self.act_en = A("gui.lang.en", lambda: self.ctx.set_language("en"), checkable=True)
        self.act_zh.setText(i18n.LANGUAGES["zh_TW"])
        self.act_en.setText(i18n.LANGUAGES["en"])
        self._tr_items = [t for t in self._tr_items if t[0] not in (self.act_zh, self.act_en)]  # 語言名稱不翻譯
        langg = QActionGroup(self)
        langg.addAction(self.act_zh)
        langg.addAction(self.act_en)
        self.act_about = A("gui.action.about", self.show_about)
        self.act_devdoc = A("gui.action.dev_doc", self.show_dev_doc)
        self.act_diag = A("diag.action", self.show_diagnostics)

    def _build_menus(self) -> None:
        m = self._menu("file", "gui.menu.file")
        for a in (self.act_new_project, self.act_open_project, self.act_project_settings):
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.act_workspace)
        m.addAction(self.act_reveal)
        m.addSeparator()
        m.addAction(self.act_quit)
        m = self._menu("camera", "gui.menu.camera")
        for a in (self.act_add_camera, self.act_close_camera, self.act_retry):
            m.addAction(a)
        m = self._menu("view", "gui.menu.view")
        for a in (self.act_single, self.act_grid):
            m.addAction(a)
        m.addSeparator()
        for a in (self.act_processed, self.act_raw, self.act_hud):
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.act_fullscreen)
        m = self._menu("language", "gui.menu.language")
        m.addAction(self.act_zh)
        m.addAction(self.act_en)
        m = self._menu("help", "gui.menu.help")
        m.addAction(self.act_about)
        m.addAction(self.act_devdoc)
        m.addAction(self.act_diag)
        m.addSeparator()

    def _build_toolbar(self) -> None:
        tb = QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)
        self.toolbar = tb
        self.project_label = QLabel()
        tb.addWidget(self.project_label)
        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(180)
        self.project_combo.activated.connect(self._project_combo_activated)
        tb.addWidget(self.project_combo)
        tb.addAction(self.act_project_settings)
        tb.addSeparator()
        tb.addAction(self.act_add_camera)
        tb.addSeparator()
        tb.addAction(self.act_single)
        tb.addAction(self.act_grid)
        tb.addSeparator()
        tb.addAction(self.act_processed)
        tb.addAction(self.act_raw)

    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        self.status_camera = QLabel()
        self.status_project = QLabel()
        self.status_workspace = QLabel()
        self.status_cloud = QLabel()
        for w in (self.status_camera, self.status_project, self.status_workspace, self.status_cloud):
            sb.addPermanentWidget(w)

    # ---- 面板 -------------------------------------------------------------------
    def _load_panels(self) -> None:
        for cls in discover_panels():
            try:
                panel = cls(self.ctx, self)
            except Exception:  # noqa: BLE001 - 一個面板壞掉不影響其他
                log.exception("建立面板 %s 失敗", cls.panel_id)
                continue
            self._panels.append(panel)
            self.tabs.addTab(panel, panel.title)
            self.ctx.projectChanged.connect(panel.on_project_changed)
            self.ctx.activeSourceChanged.connect(panel.on_active_source_changed)
        for panel in self._panels:
            self._safe(panel.on_project_changed, self.ctx.project)

    @staticmethod
    def _safe(fn, *args) -> None:
        try:
            fn(*args)
        except Exception:  # noqa: BLE001
            log.exception("面板回呼失敗：%s", getattr(fn, "__qualname__", fn))

    # ---- 相機 -------------------------------------------------------------------
    def add_camera_dialog(self) -> SourceHandle | None:
        if len(self.ctx.sources) >= MAX_SOURCES:
            QMessageBox.information(self, tr("gui.action.add_camera"), tr("gui.camera.max", count=MAX_SOURCES))
            return None
        dlg = SourceDialog(self.ctx, self)
        if dlg.exec() and dlg.selected_info() is not None:
            return self.add_source(dlg.selected_info())
        return None

    def add_source(self, info: SourceInfo) -> SourceHandle | None:
        """開啟一台相機並設為作用中；已經開著同一台時只切換過去。"""
        for h in self.ctx.sources:
            if h.info.id == info.id:
                self.set_active_handle(h)
                return h
        if len(self.ctx.sources) >= MAX_SOURCES:
            self.show_message(tr("gui.camera.max", count=MAX_SOURCES))
            return None
        handle = SourceHandle(info, self.ctx.settings, parent=self)
        view = self.grid.add_view(handle)
        view.set_hud_visible(self._show_hud)
        self._apply_classes(view)
        view.retryRequested.connect(handle.retry)
        handle.frameReady.connect(lambda packet, h=handle: self._on_frame(h, packet))
        handle.failed.connect(lambda err, h=handle: self._on_source_failed(h, err))
        handle.opened.connect(lambda _i, h=handle: self._on_source_opened(h))
        handle.stateChanged.connect(lambda _s, h=handle: self._on_state_changed(h))
        self.ctx.add_source(handle)
        self.set_active_handle(handle)
        view.set_placeholder(tr("gui.video.opening"))
        handle.start()
        self._remember_source(info)
        return handle

    def set_active_handle(self, handle: SourceHandle | None) -> None:
        self.ctx.set_active_source(handle)

    def close_source(self, handle: SourceHandle) -> None:
        self.inference.remove_for_handle(handle)
        handle.stop(STOP_TIMEOUT)
        self._hud_extra.pop(handle, None)
        self.grid.remove_view(handle)
        self.ctx.remove_source(handle)
        self.grid.set_active(self.ctx.active_source)
        self._save_session()
        handle.deleteLater()
        self._update_status(force=True)

    def close_active_camera(self) -> None:
        h = self.active_handle()
        if h is not None:
            self.close_source(h)

    def retry_active_camera(self) -> None:
        h = self.active_handle()
        if h is not None:
            h.retry()

    def _on_source_opened(self, handle: SourceHandle) -> None:
        view = self.grid.view_for(handle)
        if view is not None:
            view.clear_error()
            view.set_placeholder(tr("gui.video.waiting"))

    def _on_source_failed(self, handle: SourceHandle, err: dict) -> None:
        view = self.grid.view_for(handle)
        if view is not None:
            view.show_error(err.get("message", ""), err.get("hint", ""), bool(err.get("retryable")))
        self._update_actions()

    def _on_state_changed(self, handle: SourceHandle) -> None:
        if handle is self.active_handle():
            self._update_actions()
            self._update_status(force=True)

    def _on_active_changed(self, handle: SourceHandle | None) -> None:
        self.grid.set_active(handle)
        self._update_actions()
        self._update_status(force=True)

    def _update_actions(self) -> None:
        h = self.active_handle()
        self.act_close_camera.setEnabled(h is not None)
        self.act_retry.setEnabled(h is not None and h.state == "error")
        self.act_add_camera.setEnabled(len(self.ctx.sources) < MAX_SOURCES)

    # ---- 推論 -------------------------------------------------------------------
    def _on_inference_result(self, handle: SourceHandle, session: Any, result: InferenceResult) -> None:
        self._show_inference(handle)

    def _on_sessions_changed(self) -> None:
        for h in self.ctx.sources:
            self._show_inference(h)

    def _show_inference(self, handle: SourceHandle) -> None:
        """把該相機所有模型的最新結果畫到畫面上，並在 HUD 加一行推論耗時。"""
        view = self.grid.view_for(handle)
        if view is None:
            return
        sessions = [s for s in self.inference.sessions_for(handle) if s.last_result is not None]
        view.set_color_resolver(self._color_resolver)
        self.set_inference(handle, [s.last_result for s in sessions] or None)
        parts = []
        for s in sessions:
            st = s.stats
            parts.append(tr("gui.model.hud", ms=f"{s.last_result.timings_ms.get('total', 0.0):.0f}",
                            age=int(st["age_frames"])))
        self.set_hud_line(handle, "infer", tr("gui.list_sep").join(parts) if parts else None)

    # ---- 畫面 -------------------------------------------------------------------
    def _on_frame(self, handle: SourceHandle, packet: FramePacket) -> None:
        view = self.grid.view_for(handle)
        if view is None:
            return
        if self._show_processed:
            view.set_image(packet.qimage)
            view.set_overlays(packet.result.overlays)
        else:
            view.set_image(packet.qimage_raw)
            view.set_overlays([])
        self._refresh_hud(handle)
        if handle is self.active_handle():
            for panel in self._panels:
                self._safe(panel.on_frame, packet)
            self._update_status()

    def _refresh_hud(self, handle: SourceHandle) -> None:
        view = self.grid.view_for(handle)
        pkt = handle.last_packet
        if view is None:
            return
        lines = [handle.info.name]
        if pkt is not None:
            lines.append(tr("gui.hud.fps_latency", fps=f"{pkt.stats.get('fps', 0.0):.1f}",
                            ms=f"{pkt.stats.get('latency_ms', 0.0):.0f}"))
            lines.append(f"{pkt.frame.width}×{pkt.frame.height}")
            if pkt.frame.meta.get("ended"):
                # 影片／圖片資料夾播完且不循環：畫面會停住，要說清楚不是當機
                lines.append(tr("gui.hud.ended"))
        lines.extend(self._hud_extra.get(handle, {}).values())
        view.set_hud(lines)

    def _apply_classes(self, view: VideoView) -> None:
        project = self.ctx.project
        if project is not None and project.classes:
            view.set_classes(project.classes, i18n.current_language())

    def set_layout_mode(self, mode: str) -> None:
        self.grid.set_layout_mode(mode)
        (self.act_grid if mode == "grid" else self.act_single).setChecked(True)

    def set_show_processed(self, processed: bool) -> None:
        self._show_processed = processed
        (self.act_processed if processed else self.act_raw).setChecked(True)
        for h in self.ctx.sources:
            if h.last_packet is not None:
                self._on_frame(h, h.last_packet)

    def set_show_hud(self, visible: bool) -> None:
        self._show_hud = bool(visible)
        self.act_hud.setChecked(self._show_hud)
        for v in self.grid.views():
            v.set_hud_visible(self._show_hud)

    def _toggle_fullscreen(self, on: bool) -> None:
        self.showFullScreen() if on else self.showNormal()

    # ---- 專案與工作區 -----------------------------------------------------------
    def project_dialog(self, tab: int = 0) -> None:
        dlg = ProjectDialog(self.ctx, self, tab)
        if dlg.exec():
            project = dlg.selected_project()
            if project is not None:
                self.ctx.set_project(project)
                self.ctx.save_settings()

    def _project_combo_activated(self, index: int) -> None:
        pid = self.project_combo.itemData(index)
        if pid and (self.ctx.project is None or pid != self.ctx.project.id):
            try:
                self.ctx.set_project(self.ctx.workspace.open_project(pid))
            except (OSError, ValueError, KeyError) as e:
                QMessageBox.warning(self, tr("gui.project.title"), str(e))

    def _fill_project_combo(self) -> None:
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        for p in self.ctx.workspace.projects():
            self.project_combo.addItem(p.name, p.id)
        cur = self.ctx.project
        if cur is not None:
            i = self.project_combo.findData(cur.id)
            if i < 0:
                self.project_combo.addItem(cur.name, cur.id)
                i = self.project_combo.count() - 1
            self.project_combo.setCurrentIndex(i)
        self.project_combo.blockSignals(False)

    def _on_project_changed(self, project: Project | None) -> None:
        self._fill_project_combo()
        for v in self.grid.views():
            self._apply_classes(v)
        self._update_title()
        self._update_status(force=True)

    def switch_workspace(self) -> None:
        dlg = WorkspaceDialog(self, self.ctx.workspace.root)
        if not dlg.exec():
            return
        path = dlg.chosen_path()
        if path == self.ctx.workspace.root:
            return
        self.ctx.workspace = Workspace(path)
        self.ctx.settings.set("workspace.path", str(path))
        self.ctx.settings.set("workspace.last_project", "")
        # 與啟動時相同：安全模式與「外掛停用」都要尊重，並走信任確認
        if not self.safe_mode and self.ctx.settings.get("plugins.enabled", True):
            load_workspace_plugins(
                self.ctx.settings, self.ctx.workspace.plugins_dir, interactive=self.interactive, parent=self
            )
        projects = self.ctx.workspace.projects()
        project = projects[0] if projects else self.ctx.workspace.create_project(tr("gui.project.default_name"),
                                                                                "blank", lang=i18n.current_language())
        self.ctx.set_project(project)
        self.ctx.save_settings()

    def reveal_project(self) -> None:
        p = self.ctx.project
        folder = p.root if p is not None else self.ctx.workspace.root
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    # ---- 說明 -------------------------------------------------------------------
    def show_about(self) -> None:
        QMessageBox.about(self, tr("gui.action.about"),
                          f"<b>{tr('app.name')}</b><br>{tr('app.tagline')}<br>{tr('app.version', version=edgelens.__version__)}")

    def dev_doc_path(self) -> Path:
        return Path(edgelens.__file__).resolve().parent.parent / "docs" / "DEVELOPMENT.md"

    def show_diagnostics(self) -> None:
        """說明 →「診斷資訊…」。放在主視窗而不是某個面板：面板載入失敗時，正是最需要診斷的時候。"""
        try:
            from edgelens.gui.diagnostics import show_diagnostics
        except Exception as e:  # noqa: BLE001
            log.exception("無法開啟診斷資訊")
            QMessageBox.warning(self, tr("common.error"), str(e))
            return
        show_diagnostics(self, self.ctx)

    def show_dev_doc(self) -> None:
        path = self.dev_doc_path()
        box = QMessageBox(self)
        box.setWindowTitle(tr("gui.action.dev_doc"))
        box.setText(tr("gui.help.dev_doc", path=str(path)))
        if path.is_file():
            open_btn = box.addButton(tr("common.open"), QMessageBox.ButtonRole.AcceptRole)
            box.addButton(tr("common.close"), QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is open_btn:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        else:
            box.exec()

    # ---- 語言與文字 -------------------------------------------------------------
    def _on_language_changed(self, lang: str) -> None:
        self.retranslate()

    def retranslate(self) -> None:
        for obj, key, kind in self._tr_items:
            text = tr(key)
            if kind == "title":
                obj.setTitle(text)
            else:
                obj.setText(text)
        self.act_zh.setChecked(i18n.current_language() == "zh_TW")
        self.act_en.setChecked(i18n.current_language() == "en")
        self.project_label.setText(tr("gui.toolbar.project") + " ")
        self.project_combo.setToolTip(tr("gui.toolbar.project_tip"))
        self.grid.retranslate()
        for i, panel in enumerate(self._panels):
            self.tabs.setTabText(i, panel.title)
            self._safe(panel.retranslate)
        for h, view in ((h, self.grid.view_for(h)) for h in self.ctx.sources):
            if view is not None:
                view.retranslate()
                self._apply_classes(view)
                self._refresh_hud(h)
        self._update_title()
        self._update_status(force=True)

    def _update_title(self) -> None:
        p = self.ctx.project
        self.setWindowTitle(f"{tr('app.name')} - {p.name}" if p else tr("app.name"))

    def _update_status(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_status < 0.25:
            return
        self._last_status = now
        h = self.active_handle()
        if h is None:
            self.status_camera.setText(tr("gui.status.no_camera"))
        elif h.state != "running":
            self.status_camera.setText(f"{h.info.name}: {tr(f'gui.state.{h.state}')}")
        else:
            st = h.last_packet.stats if h.last_packet else {}
            self.status_camera.setText(tr("gui.status.camera", name=h.info.name, fps=f"{st.get('fps', 0.0):.1f}",
                                          ms=f"{st.get('latency_ms', 0.0):.0f}"))
        p = self.ctx.project
        self.status_project.setText(tr("gui.status.project", name=p.name if p else tr("common.unknown")))
        root = self.ctx.workspace.root
        self.status_workspace.setText(tr("gui.status.workspace", path=str(root)))
        self.status_workspace.setToolTip(str(root))
        provider = cloud_sync_provider(root)
        if provider:
            self.status_cloud.setText("⚠ " + provider)
            self.status_cloud.setToolTip(tr("gui.workspace.cloud_warning", provider=provider))
            self.status_cloud.setStyleSheet("color:#C77700;font-weight:bold;")
        else:
            self.status_cloud.setText("")
            self.status_cloud.setToolTip("")

    # ---- 啟動／結束 -------------------------------------------------------------
    def _restore_geometry(self) -> None:
        data = self.ctx.settings.get("ui.geometry", "")
        if data:
            try:
                self.restoreGeometry(QByteArray.fromBase64(str(data).encode("ascii")))
            except Exception:  # noqa: BLE001
                log.warning("還原視窗位置失敗")

    def _remember_source(self, info: SourceInfo) -> None:
        s = self.ctx.settings
        recent = [d for d in (s.get("recent.sources", []) or []) if isinstance(d, dict) and d.get("id") != info.id]
        recent.insert(0, info_for_storage(info))
        s.set("recent.sources", recent[:10])
        self._save_session()

    def _save_session(self) -> None:
        if self._closing:
            return
        self.ctx.settings.set("session.sources", [info_for_storage(h.info) for h in self.ctx.sources])
        self.ctx.save_settings_later()

    def _offer_reopen(self) -> None:
        saved = [d for d in (self.ctx.settings.get("session.sources", []) or []) if isinstance(d, dict)]
        infos = []
        for d in saved:
            try:
                infos.append(SourceInfo.from_dict(d))
            except (KeyError, TypeError):
                continue
        if not infos or self.ctx.sources:
            return
        names = "\n".join("• " + i.name for i in infos)
        ans = QMessageBox.question(self, tr("gui.reopen.title"), tr("gui.reopen.text", names=names))
        if ans == QMessageBox.StandardButton.Yes:
            for info in infos[:MAX_SOURCES]:
                self.add_source(info)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._closing:
            event.accept()
            return
        s = self.ctx.settings
        s.set("session.sources", [info_for_storage(h.info) for h in self.ctx.sources])
        self._closing = True
        s.set("ui.geometry", bytes(self.saveGeometry().toBase64()).decode("ascii"))
        for panel in self._panels:
            self._safe(panel.shutdown)
        self.inference.shutdown(STOP_TIMEOUT)
        handles = list(self.ctx.sources)
        for h in handles:
            h.request_stop()  # 先通知全部，再逐一等待，總時間才不會變成相加
        deadline = time.monotonic() + STOP_TIMEOUT
        for h in handles:
            h.stop(max(0.1, deadline - time.monotonic()))
        self.ctx.save_settings()
        self.ctx.dispose()
        super().closeEvent(event)
