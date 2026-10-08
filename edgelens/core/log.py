"""日誌：寫到 app_cache_dir()/logs/edgelens.log（最多 5 個檔案輪替），診斷匯出時一起打包。"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from edgelens.core.paths import ensure_dir, logs_dir

FORMAT = "%(asctime)s %(levelname)-7s %(threadName)s %(name)s: %(message)s"
_configured = False


def setup_logging(debug: bool = False, log_dir: Path | None = None) -> Path | None:
    """設定日誌；回傳日誌檔路徑（無法寫入時只輸出到終端機並回傳 None）。可重複呼叫。"""
    global _configured
    root = logging.getLogger()
    level = logging.DEBUG if debug else logging.INFO
    root.setLevel(level)
    if _configured:
        return current_log_file()
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if debug else logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(console)
    path: Path | None = None
    try:
        folder = ensure_dir(log_dir or logs_dir())
        path = folder / "edgelens.log"
        fh = logging.handlers.RotatingFileHandler(path, maxBytes=2_000_000, backupCount=4, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(fh)
    except OSError as e:
        root.warning("無法寫入日誌檔：%s", e)
        path = None
    _configured = True
    return path


def current_log_file() -> Path | None:
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler):
            return Path(h.baseFilename)
    return None
