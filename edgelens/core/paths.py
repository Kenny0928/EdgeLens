"""跨平台路徑與檔案寫入工具。

- 程式快取（venv、日誌）：app_cache_dir()
- 程式設定（settings.json）：app_config_dir()
- 使用者資料（專案、資料集、模型）：工作區，預設為「文件／EdgeLens」

所有存檔都經過 atomic_write_*：先寫暫存檔再改名，斷電或當機時不會留下寫一半的檔案。
寫檔一律用 Python 的 open（支援中文路徑），不要用 cv2.imwrite（Windows 中文路徑會失敗）。
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import unicodedata
from importlib import resources
from pathlib import Path

APP_DIR_NAME = "EdgeLens"


def platform_key() -> str:
    """'windows'、'macos' 或 'linux'。"""
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def app_cache_dir() -> Path:
    """可刪除的資料：日誌、縮圖快取、bootstrap 的 venv。可用 EDGELENS_CACHE 覆寫。"""
    env = os.environ.get("EDGELENS_CACHE")
    if env:
        return Path(env)
    p = platform_key()
    if p == "windows":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / APP_DIR_NAME / "Cache"
    if p == "macos":
        return Path.home() / "Library" / "Caches" / APP_DIR_NAME
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / APP_DIR_NAME


def app_config_dir() -> Path:
    """程式設定（不含使用者資料）。可用 EDGELENS_CONFIG 覆寫（測試用）。"""
    env = os.environ.get("EDGELENS_CONFIG")
    if env:
        return Path(env)
    p = platform_key()
    if p == "windows":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        return base / APP_DIR_NAME
    if p == "macos":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / APP_DIR_NAME


def logs_dir() -> Path:
    return app_cache_dir() / "logs"


def documents_dir() -> Path:
    """使用者的「文件」資料夾。Windows 會跟著 OneDrive 重新導向，Linux 依 XDG 設定。"""
    p = platform_key()
    if p == "windows":
        try:
            import ctypes
            from ctypes import wintypes

            # FOLDERID_Documents
            guid = ctypes.c_ubyte * 16
            fid = guid(*bytes.fromhex("D0 9A D3 FD 8F 23 AF 46 AD B4 6C 85 48 03 69 C7".replace(" ", "")))
            out = ctypes.c_wchar_p()
            shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
            shell32.SHGetKnownFolderPath.argtypes = [ctypes.c_void_p, wintypes.DWORD, wintypes.HANDLE,
                                                     ctypes.POINTER(ctypes.c_wchar_p)]
            if shell32.SHGetKnownFolderPath(ctypes.byref(fid), 0, None, ctypes.byref(out)) == 0 and out.value:
                path = Path(out.value)
                ctypes.windll.ole32.CoTaskMemFree(out)  # type: ignore[attr-defined]
                return path
        except Exception:  # noqa: BLE001 - 失敗就用預設位置
            pass
    elif p == "linux":
        cfg = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "user-dirs.dirs"
        try:
            for line in cfg.read_text(encoding="utf-8").splitlines():
                if line.startswith("XDG_DOCUMENTS_DIR="):
                    value = line.split("=", 1)[1].strip().strip('"').replace("$HOME", str(Path.home()))
                    if value and Path(value) != Path.home():
                        return Path(value)
        except OSError:
            pass
    return Path.home() / "Documents"


def default_workspace() -> Path:
    """預設工作區。可用 EDGELENS_WORKSPACE 覆寫。"""
    env = os.environ.get("EDGELENS_WORKSPACE")
    if env:
        return Path(env)
    return documents_dir() / APP_DIR_NAME


def cloud_sync_provider(path: str | Path) -> str | None:
    """path 是否在雲端同步資料夾內；是的話回傳服務名稱（iCloud、OneDrive、Dropbox、Google Drive）。

    雲端同步可能把資料集檔案移到雲端只留佔位檔，或在 iCloud 上把 . 開頭資料夾的檔案設成隱藏，
    所以工作區放在這些地方時 GUI 會提醒。
    """
    raw = Path(path).expanduser()
    try:
        p = raw.resolve() if raw.is_absolute() else raw
    except OSError:
        p = raw
    low = (str(raw) + "|" + str(p)).lower().replace("\\", "/")
    if "mobile documents" in low or "com~apple~clouddocs" in low:
        return "iCloud"
    if "onedrive" in low:
        return "OneDrive"
    if "dropbox" in low:
        return "Dropbox"
    if "google drive" in low or "googledrive" in low:
        return "Google Drive"
    if platform_key() == "macos" and p.is_absolute():
        home = Path.home()
        # 開啟「桌面與文件檔案夾」同步時，~/Documents 和 ~/Desktop 也由 iCloud 管理
        icloud_docs = home / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Documents"
        if icloud_docs.exists():
            for top in (home / "Documents", home / "Desktop"):
                if p == top or top in p.parents:
                    return "iCloud"
    return None


def resource_path(*parts: str) -> Path:
    """套件內附檔案（例如 i18n、預設組合、K210 板上腳本）的實際路徑。

    以 importlib.resources 取得，搬移資料夾或打包後都能找到；EdgeLens 以一般資料夾
    方式安裝，所以可以直接回傳 Path。
    """
    root = resources.files("edgelens")
    target = root.joinpath(*parts)
    return Path(str(target))


_SLUG_RE = re.compile(r"[^a-z0-9_]+")


def slugify(text: str, fallback: str = "item") -> str:
    """把任意文字轉成只含小寫英數字與底線的 ID（類別資料夾名稱用）。

    中文等非拉丁字元會被去掉；全部去掉時回傳 fallback。
    """
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = _SLUG_RE.sub("_", norm.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if not slug:
        return fallback
    if slug[0].isdigit():
        slug = f"c_{slug}"
    return slug[:48]


def is_valid_id(text: str) -> bool:
    return bool(text) and slugify(text) == text


_BAD_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WIN_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_filename(name: str, fallback: str = "untitled") -> str:
    """去掉 Windows／macOS／Linux 檔名不允許的字元；保留中文。"""
    s = _BAD_FILENAME.sub("_", name).strip().rstrip(".")
    if not s:
        s = fallback
    if s.split(".")[0].upper() in _WIN_RESERVED:
        s = "_" + s
    return s[:120]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """先寫到同資料夾的暫存檔，再用 os.replace 換上（同一個磁碟內是原子操作）。"""
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=path.suffix, dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def unique_path(path: str | Path) -> Path:
    """path 已存在時在檔名後面加 _1、_2…，回傳第一個不存在的路徑。"""
    path = Path(path)
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    i = 1
    while True:
        cand = path.with_name(f"{stem}_{i}{suffix}")
        if not cand.exists():
            return cand
        i += 1
