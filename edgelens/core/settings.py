"""程式設定：一個 JSON 檔（app_config_dir()/settings.json）。

用點號分隔的鍵讀寫，例如 settings.get("ui.language")。每台相機的設定放在
"sources.<source_id>" 底下，換相機時各自保留。讀檔失敗（檔案壞掉）時備份舊檔並用預設值。
"""

from __future__ import annotations

import copy
import json
import logging
import threading
from pathlib import Path
from typing import Any

from edgelens.core.paths import app_config_dir, atomic_write_text

log = logging.getLogger(__name__)

SETTINGS_VERSION = 1

DEFAULTS: dict[str, Any] = {
    "version": SETTINGS_VERSION,
    "ui": {"language": "zh_TW", "theme": "system", "show_advanced": False},
    "workspace": {"path": "", "last_project": ""},
    "plugins": {"enabled": True, "trusted": {}},
    "sources": {},
    "recent": {"models": [], "pipelines": [], "sources": []},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Settings:
    """執行緒安全的設定物件。改完要呼叫 save()（GUI 在關閉時與重要變更後呼叫）。"""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else app_config_dir() / "settings.json"
        self._lock = threading.RLock()
        self._data: dict[str, Any] = copy.deepcopy(DEFAULTS)
        self.load()

    def load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._data = copy.deepcopy(DEFAULTS)
                return
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError("設定檔最外層不是物件")
            except (OSError, ValueError) as e:
                backup = self.path.with_suffix(".broken.json")
                log.warning("設定檔讀取失敗（%s），已備份為 %s 並改用預設值", e, backup.name)
                try:
                    self.path.replace(backup)
                except OSError:
                    pass
                self._data = copy.deepcopy(DEFAULTS)
                return
            self._data = _deep_merge(DEFAULTS, raw)

    def save(self) -> None:
        with self._lock:
            text = json.dumps(self._data, ensure_ascii=False, indent=2)
        atomic_write_text(self.path, text)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            node: Any = self._data
            for part in key.split("."):
                if not isinstance(node, dict) or part not in node:
                    return default
                node = node[part]
            return copy.deepcopy(node)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            parts = key.split(".")
            node = self._data
            for part in parts[:-1]:
                nxt = node.get(part)
                if not isinstance(nxt, dict):
                    nxt = {}
                    node[part] = nxt
                node = nxt
            node[parts[-1]] = copy.deepcopy(value)

    def source_settings(self, source_id: str) -> dict[str, Any]:
        """某台相機上次的設定（參數值、處理流程等）；沒有就回傳空 dict。"""
        with self._lock:
            return copy.deepcopy(self._data.setdefault("sources", {}).get(source_id, {}))

    def set_source_settings(self, source_id: str, values: dict[str, Any]) -> None:
        with self._lock:
            self._data.setdefault("sources", {})[source_id] = copy.deepcopy(values)

    def push_recent(self, kind: str, item: str, limit: int = 10) -> None:
        with self._lock:
            lst = [x for x in self._data.setdefault("recent", {}).get(kind, []) if x != item]
            lst.insert(0, item)
            self._data["recent"][kind] = lst[:limit]

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)
