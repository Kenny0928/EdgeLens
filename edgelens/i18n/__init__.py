"""多語系：繁體中文（zh_TW）與英文（en）。

翻譯檔在 locales/<語言>/*.json，同一語言的所有檔案會合併；每個工作包各自放一個檔案
（例如 locales/zh_TW/sources.json），避免多人同時改同一個檔案。鍵用點號分層：

    "source.webcam.name": "電腦相機"

程式裡一律用 tr("鍵", 參數=值)；文字中的 {參數} 會被代換。找不到翻譯時依序退回英文、
再退回鍵本身（並記錄警告），所以缺翻譯不會讓程式當掉。tests/unit/test_i18n.py 會檢查
兩種語言的鍵是否一致、程式用到的鍵是否都存在。
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

LANGUAGES = {"zh_TW": "繁體中文", "en": "English"}
DEFAULT_LANGUAGE = "zh_TW"
FALLBACK_LANGUAGE = "en"
LOCALES_DIR = Path(__file__).resolve().parent / "locales"

_lock = threading.RLock()
_tables: dict[str, dict[str, str]] = {}
_current = DEFAULT_LANGUAGE
_listeners: list[Callable[[str], None]] = []
_missing_logged: set[str] = set()


def load_language(lang: str) -> dict[str, str]:
    """讀取並合併 locales/<lang>/*.json。同一個鍵出現在兩個檔案時丟 ValueError。"""
    table: dict[str, str] = {}
    origin: dict[str, str] = {}
    folder = LOCALES_DIR / lang
    if not folder.is_dir():
        return table
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} 最外層必須是物件")
        for k, v in data.items():
            if k.startswith("_"):  # "_comment" 之類的註解
                continue
            if k in table:
                raise ValueError(f"翻譯鍵 {k!r} 同時出現在 {origin[k]} 和 {path.name}")
            if not isinstance(v, str):
                raise ValueError(f"{path.name} 的 {k!r} 必須是字串")
            table[k] = v
            origin[k] = path.name
    return table


def _table(lang: str) -> dict[str, str]:
    with _lock:
        if lang not in _tables:
            _tables[lang] = load_language(lang)
        return _tables[lang]


def reload() -> None:
    """重新讀取翻譯檔（開發時改了 JSON 可以呼叫）。"""
    with _lock:
        _tables.clear()
        _missing_logged.clear()


def set_language(lang: str) -> None:
    global _current
    if lang not in LANGUAGES:
        raise ValueError(f"不支援的語言：{lang}（可用：{', '.join(LANGUAGES)}）")
    with _lock:
        changed = lang != _current
        _current = lang
        listeners = list(_listeners)
    if changed:
        for cb in listeners:
            try:
                cb(lang)
            except Exception:  # noqa: BLE001 - 單一元件出錯不影響其他元件換語言
                log.exception("語言切換通知失敗")


def current_language() -> str:
    return _current


def on_language_changed(callback: Callable[[str], None]) -> Callable[[], None]:
    """註冊語言切換時要呼叫的函式，回傳取消註冊的函式。"""
    with _lock:
        _listeners.append(callback)

    def remove() -> None:
        with _lock:
            if callback in _listeners:
                _listeners.remove(callback)

    return remove


def has(key: str, lang: str | None = None) -> bool:
    return key in _table(lang or _current)


def tr(key: str, /, **kwargs: Any) -> str:
    """翻譯 key。kwargs 代換文字中的 {名稱}；代換失敗時回傳未代換的文字。"""
    text = _table(_current).get(key)
    if text is None:
        text = _table(FALLBACK_LANGUAGE).get(key)
        if key not in _missing_logged:
            _missing_logged.add(key)
            log.warning("缺少翻譯：%s（%s）", key, _current)
        if text is None:
            return key
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return text
    return text


def tr_or(key: str, fallback: str, /, **kwargs: Any) -> str:
    """有翻譯就翻譯，沒有就用 fallback（給外掛或使用者自訂的名稱用，不記警告）。"""
    if key and (has(key) or has(key, FALLBACK_LANGUAGE)):
        return tr(key, **kwargs)
    return fallback.format(**kwargs) if kwargs else fallback
