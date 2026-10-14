"""模型庫下載與匯入的 GUI：進度對話框（可取消），下載在背景執行緒，畫面不會卡住。

    if download_item(parent, zoo.find("objects365-yolo26n")):
        ...                                  # 已安裝，可以用 zoo.card_path(item)
    item = import_pack(parent)               # 選 zip 匯入離線包，回傳裝好的項目或 None
"""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog, QWidget

from edgelens import zoo
from edgelens.i18n import tr

log = logging.getLogger(__name__)

_orphans: list[QThread] = []   # 對話框關掉但執行緒還在收尾：保留參照避免被銷毀


class _InstallThread(QThread):
    progress = Signal(int, int)          # 已完成、總量（KB，避免超過 int 範圍）
    finished_with = Signal(str, str)     # 錯誤訊息、提示；成功時都是空字串

    def __init__(self, work, parent=None):
        super().__init__(parent)
        self._work = work
        self.cancel = threading.Event()

    def run(self) -> None:
        last = [-1]

        def report(done: int, total: int) -> None:
            kb_done, kb_total = done // 1024, total // 1024
            pct = kb_done * 100 // kb_total if kb_total else -1
            if pct != last[0]:             # 每個百分點才通知一次，不要塞爆事件佇列
                last[0] = pct
                self.progress.emit(kb_done, kb_total)

        try:
            self._work(report, self.cancel)
            self.finished_with.emit("", "")
        except zoo.ZooError as e:
            self.finished_with.emit(e.message, e.hint)
        except Exception as e:  # noqa: BLE001 - 未預期的錯誤也要讓對話框關掉並顯示
            log.exception("模型庫安裝失敗")
            self.finished_with.emit(tr("zoo.err.unexpected", error=str(e)[:200]), "")


def _run_with_dialog(parent: QWidget | None, title: str, label: str, work) -> tuple[str, str]:
    """在背景執行 work(progress, cancel)，顯示進度對話框直到結束。回傳 (錯誤訊息, 提示)。"""
    dlg = QProgressDialog(label, tr("common.cancel"), 0, 0, parent)
    dlg.setWindowTitle(title)
    dlg.setWindowModality(Qt.WindowModality.WindowModal)
    dlg.setMinimumDuration(0)
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    result = ["", ""]
    th = _InstallThread(work)

    def on_progress(done: int, total: int) -> None:
        if total > 0:
            dlg.setMaximum(total)
            dlg.setValue(min(done, total))
            dlg.setLabelText(f"{label}\n{done / 1024:.1f} / {total / 1024:.1f} MB")

    def on_finished(msg: str, hint: str) -> None:
        result[0], result[1] = msg, hint
        dlg.reset()
        dlg.done(0)

    th.progress.connect(on_progress)
    th.finished_with.connect(on_finished)
    dlg.canceled.connect(th.cancel.set)
    th.start()
    dlg.exec()
    if th.isRunning():               # 按了取消：通知執行緒，讓它自己收尾
        th.cancel.set()
        _orphans.append(th)
        th.finished.connect(lambda: _orphans.remove(th) if th in _orphans else None)
    return result[0], result[1]


def _show_error(parent: QWidget | None, msg: str, hint: str) -> None:
    QMessageBox.warning(parent, tr("zoo.dialog.failed"), msg + (f"\n\n{hint}" if hint else ""))


def download_item(parent: QWidget | None, item: zoo.ZooItem) -> bool:
    """下載並安裝一項；已經裝好就直接回傳 True。取消或失敗回傳 False（失敗會跳出說明）。"""
    if zoo.is_installed(item):
        return True
    msg, hint = _run_with_dialog(parent, tr("zoo.dialog.title"),
                                 tr("zoo.dialog.downloading", name=item.title, size=item.size_text),
                                 lambda progress, cancel: zoo.install(item, progress, cancel))
    if msg and msg != tr("zoo.err.cancelled"):
        _show_error(parent, msg, hint)
    return zoo.is_installed(item)


def import_pack(parent: QWidget | None) -> zoo.ZooItem | None:
    """選一個離線包 zip 匯入。成功回傳裝好的項目。"""
    path, _ = QFileDialog.getOpenFileName(parent, tr("zoo.dialog.import"), "", "ZIP (*.zip)")
    if not path:
        return None
    found: list[zoo.ZooItem] = []
    msg, hint = _run_with_dialog(parent, tr("zoo.dialog.import"), tr("zoo.dialog.importing"),
                                 lambda _progress, _cancel: found.append(zoo.install_zip(path)))
    if msg:
        _show_error(parent, msg, hint)
        return None
    item = found[0] if found else None
    if item is not None:
        QMessageBox.information(parent, tr("zoo.dialog.import"), tr("zoo.dialog.imported", name=item.title))
    return item
