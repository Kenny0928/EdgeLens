"""外掛信任檢查：外掛是完整權限的 Python 程式，所以載入前要先讓使用者確認。

信任記在設定的 plugins.trusted：{外掛資料夾的完整路徑: {檔名: sha256}}。以檔案內容為準，
檔案一改（或換成別人的檔）就要重新確認；同一份內容只需要確認一次。
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtWidgets import QMessageBox, QWidget

from edgelens.core.registry import load_plugins, plugin_hashes
from edgelens.core.settings import Settings
from edgelens.i18n import tr

log = logging.getLogger(__name__)

TRUST_KEY = "plugins.trusted"
# 最多記幾個資料夾；超過就丟掉最久以前核准的，免得 settings.json 無限變大
MAX_TRUSTED_FOLDERS = 20


def _folder_key(folder: str | Path) -> str:
    try:
        return str(Path(folder).resolve())
    except OSError:
        return str(folder)


def trusted_hashes(settings: Settings, folder: str | Path) -> dict[str, str]:
    store = settings.get(TRUST_KEY, {})
    entry = store.get(_folder_key(folder)) if isinstance(store, dict) else None
    return {str(k): str(v) for k, v in entry.items()} if isinstance(entry, dict) else {}


def record_trust(settings: Settings, folder: str | Path, hashes: dict[str, str]) -> None:
    """記下這個資料夾目前的檔案內容。dict 保留插入順序，所以重新核准的資料夾會排到最後（最新）。"""
    store = settings.get(TRUST_KEY, {})
    if not isinstance(store, dict):
        store = {}
    key = _folder_key(folder)
    store.pop(key, None)
    store[key] = dict(hashes)
    while len(store) > MAX_TRUSTED_FOLDERS:
        store.pop(next(iter(store)))
    settings.set(TRUST_KEY, store)


def classify(settings: Settings, folder: str | Path) -> tuple[dict[str, str], list[str], list[tuple[str, bool]]]:
    """回傳 (目前的雜湊, 仍受信任且沒改的檔名, [(需要確認的檔名, 是否為已修改)])。"""
    current = plugin_hashes(folder)
    known = trusted_hashes(settings, folder)
    ok = [n for n, h in current.items() if known.get(n) == h]
    pending = [(n, n in known) for n in current if n not in ok]
    return current, ok, pending


def ask_trust(parent: QWidget | None, pending: list[tuple[str, bool]]) -> bool:
    """跳出確認視窗，回傳 True 代表使用者同意載入。測試會把這個函式換掉。"""
    lines = [
        f"• {name}（{tr('gui.plugins.trust.changed') if changed else tr('gui.plugins.trust.new')}）"
        for name, changed in pending
    ]
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(tr("gui.plugins.trust.title"))
    box.setText(tr("gui.plugins.trust.text"))
    box.setInformativeText("\n".join(lines))
    load_btn = box.addButton(tr("gui.plugins.trust.load"), QMessageBox.ButtonRole.AcceptRole)
    skip_btn = box.addButton(tr("gui.plugins.trust.skip"), QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(skip_btn)  # 預設選「不載入」，亂按 Enter 比較安全
    box.exec()
    return box.clickedButton() is load_btn


def load_plugins_trusted(settings: Settings, folder: str | Path, parent: QWidget | None = None) -> list[str]:
    """依信任狀態載入外掛：全部可信就直接載入，否則詢問；拒絕時只載入仍可信的檔案。"""
    current, ok, pending = classify(settings, folder)
    if not pending:
        return load_plugins(folder, only=ok)
    if ask_trust(parent, pending):
        record_trust(settings, folder, current)
        return load_plugins(folder, only=list(current))
    log.info("使用者未核准外掛：%s", [n for n, _ in pending])
    return load_plugins(folder, only=ok)


def load_workspace_plugins(
    settings: Settings, folder: str | Path, *, interactive: bool, parent: QWidget | None = None
) -> list[str]:
    """啟動與切換工作區共用的入口。非互動模式（測試、自動化）沿用舊行為：全部載入。"""
    if not interactive:
        return load_plugins(folder)
    return load_plugins_trusted(settings, folder, parent)
