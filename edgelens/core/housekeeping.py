"""開機清理：縮圖快取修剪、當機後遺留的暫存資料夾。

這裡的每個動作都刻意保守：只動自己建立的東西、不跟隨符號連結、任何錯誤都吞掉並記錄，
因為清理失敗絕不能影響程式啟動。
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

from edgelens.core.paths import app_cache_dir

log = logging.getLogger(__name__)

REC_TEMP_PREFIX = "edgelens_rec_"   # 錄影用（recorder.ascii_temp_dir）
VIDEO_TEMP_PREFIX = "edgelens_"     # Windows 影片播放複製用（file.openable_video_path）

# tempfile.mkdtemp 會在前綴後接 8 個 [a-z0-9_]；用完整比對，避免誤刪使用者自己的 edgelens_* 資料夾
_TEMP_NAME = re.compile(r"^edgelens_(?:rec_)?[a-z0-9_]{8}$")


def temp_base_candidates() -> list[Path]:
    """暫存資料夾可能所在的位置（順序即錄影時的嘗試順序）。

    錄影與影片播放都用這份清單建立暫存資料夾，清理也用同一份，兩邊才不會不一致。"""
    candidates = [Path(tempfile.gettempdir())]
    if sys.platform.startswith("win"):
        candidates += [Path(os.environ.get("PUBLIC", "C:/Users/Public")) / "EdgeLensTmp",
                       Path(os.environ.get("SystemDrive", "C:") + "/") / "EdgeLensTmp"]
    else:
        candidates.append(Path("/tmp"))
    return candidates


# ---------------------------------------------------------------- 縮圖快取

def prune_thumb_cache(max_bytes: int = 300_000_000, target_bytes: int = 240_000_000) -> int:
    """縮圖磁碟快取超過 max_bytes 時，從最久沒用（mtime 最舊）的開始刪到低於 target_bytes。

    ThumbLoader 命中快取時會更新 mtime，所以 mtime 近似「最後使用時間」。
    只處理 app_cache_dir()/thumbs 底下的檔案；空的專案資料夾順便移除。回傳刪除的檔案數。"""
    root = app_cache_dir() / "thumbs"
    if not root.is_dir() or root.is_symlink():
        return 0
    files: list[tuple[float, int, Path]] = []
    total = 0
    try:
        projects = [d for d in root.iterdir() if d.is_dir() and not d.is_symlink()]
    except OSError:
        return 0
    for pd in projects:
        try:
            entries = list(os.scandir(pd))
        except OSError:
            continue
        for e in entries:
            try:
                if e.is_symlink() or not e.is_file(follow_symlinks=False):
                    continue
                st = e.stat(follow_symlinks=False)
            except OSError:
                continue
            files.append((st.st_mtime, st.st_size, Path(e.path)))
            total += st.st_size
    removed = 0
    if total > max_bytes:
        files.sort(key=lambda t: t[0])
        for _mtime, size, p in files:
            if total <= target_bytes:
                break
            try:
                p.unlink()
            except OSError:
                continue
            total -= size
            removed += 1
    for pd in projects:
        try:
            pd.rmdir()   # 只有空資料夾會成功
        except OSError:
            pass
    return removed


# ---------------------------------------------------------------- 暫存資料夾

def _only_ours(d: Path, name: str) -> bool:
    """edgelens_xxxxxxxx（影片複本）只放一個 video.*；內容不符就不碰。"""
    if name.startswith(REC_TEMP_PREFIX):
        return True
    try:
        return all(p.name.startswith("video.") and p.is_file() and not p.is_symlink() for p in d.iterdir())
    except OSError:
        return False


def sweep_stale_temp(max_age_days: float = 7) -> int:
    """刪除當機後遺留、超過 max_age_days 天沒動的 edgelens_* 暫存資料夾，回傳刪除數。

    只看候選基底資料夾的「直接子項」，名稱必須完全符合 mkdtemp 的格式，且不是符號連結。"""
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    seen: set[str] = set()
    for base in temp_base_candidates():
        try:
            key = os.path.realpath(base)
            if key in seen or not base.is_dir():
                continue
            seen.add(key)
            entries = list(os.scandir(base))
        except OSError:
            continue
        for e in entries:
            try:
                if not _TEMP_NAME.match(e.name) or e.is_symlink() or not e.is_dir(follow_symlinks=False):
                    continue
                if e.stat(follow_symlinks=False).st_mtime >= cutoff:
                    continue
                if not _only_ours(Path(e.path), e.name):
                    continue
                shutil.rmtree(e.path)   # rmtree 遇到符號連結只刪連結本身，不會跟進
                removed += 1
            except OSError as exc:
                log.debug("清理暫存資料夾失敗 %s：%s", e.path, exc)
    return removed


# ---------------------------------------------------------------- 啟動入口

def _run() -> None:
    for fn in (prune_thumb_cache, sweep_stale_temp):
        try:
            n = fn()
            if n:
                log.info("開機清理 %s：移除 %d 項", fn.__name__, n)
        except Exception:   # noqa: BLE001 - 清理失敗不可影響啟動
            log.exception("開機清理 %s 失敗", fn.__name__)


def run_startup_housekeeping() -> threading.Thread | None:
    """在背景 daemon 執行緒做開機清理；永不丟例外。回傳執行緒（測試用）。"""
    try:
        th = threading.Thread(target=_run, name="edgelens-housekeeping", daemon=True)
        th.start()
        return th
    except Exception:   # noqa: BLE001
        log.exception("無法啟動開機清理")
        return None
