"""可下載的模型與示範照片（模型庫）：學生不用自己找模型、轉檔、寫說明檔。

每一項是一個 zip，放在 EdgeLens 的 GitHub Release（由 tools/export_objects365.py 製作）。
下載後驗證 sha256，解壓到程式快取的 zoo/<id>/；沒有網路的教室可以把同一個 zip 用「匯入離線模型包」
裝進來，一樣驗證 sha256。zip 裡有 .card.json，所以載入後不用再自動判斷格式。

    item = find("objects365-yolo26n")
    install(item, progress=lambda done, total: ..., cancel=threading.Event())
    card_path(item)          # 交給 load_model_file
    install_zip(Path("objects365-yolo26n.zip"))   # 離線包：依 sha256 認出是哪一項

只用標準函式庫；download／install 會阻塞，GUI 要在背景執行緒呼叫。
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from edgelens.core.paths import app_cache_dir
from edgelens.i18n import tr

log = logging.getLogger(__name__)

RELEASE_URL = "https://github.com/Kenny0928/EdgeLens/releases/download/models-v1/"
MARKER = ".edgelens-zoo.json"
TIMEOUT = 30.0
CHUNK = 1 << 16
UA = {"User-Agent": "EdgeLens-zoo/1.0"}

Progress = Callable[[int, int], None]     # (已下載位元組, 總位元組；不知道時為 0)


@dataclass(frozen=True)
class ZooItem:
    id: str
    kind: str              # "model"：解壓後有 <id>.card.json；"samples"：一個圖片資料夾
    file: str              # Release 上的檔名
    size: int              # zip 大小（位元組）
    sha256: str
    license: str

    @property
    def url(self) -> str:
        return RELEASE_URL + self.file

    @property
    def title(self) -> str:
        return tr(f"zoo.item.{self.id}")

    @property
    def size_text(self) -> str:
        return f"{self.size / 1e6:.0f} MB" if self.size >= 1e6 else f"{self.size / 1e3:.0f} KB"


CATALOG: tuple[ZooItem, ...] = (
    ZooItem("objects365-yolo26n", "model", "objects365-yolo26n.zip", 8936161,
            "d28322f4c244f738d2f3f43364d48d8027c631c49eeaef827aa88fbde6d04224", "AGPL-3.0（Ultralytics）"),
    ZooItem("objects365-yolo26s", "model", "objects365-yolo26s.zip", 33296190,
            "3c61394c2dd7b944ae6b9e5e31ef16302b8408d0b4ae43bf366ce55c74f19c21", "AGPL-3.0（Ultralytics）"),
    ZooItem("objects365-samples", "samples", "objects365-samples.zip", 2610648,
            "d2e0fd88fd07fd87756285d38ac92623fd0b070f0e4d19ba6a9c6d1850637b02", "CC BY 4.0（Objects365）"),
)


class ZooError(Exception):
    """下載或安裝失敗；message／hint 可以直接給學生看。"""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint


class Cancelled(ZooError):
    def __init__(self) -> None:
        super().__init__(tr("zoo.err.cancelled"))


def find(item_id: str) -> ZooItem | None:
    return next((it for it in CATALOG if it.id == item_id), None)


def items(kind: str | None = None) -> list[ZooItem]:
    return [it for it in CATALOG if kind is None or it.kind == kind]


# ---- 位置與狀態 -----------------------------------------------------------------------

def zoo_dir() -> Path:
    return app_cache_dir() / "zoo"


def item_dir(item: ZooItem) -> Path:
    return zoo_dir() / item.id


def card_path(item: ZooItem) -> Path:
    return item_dir(item) / f"{item.id}.card.json"


def is_installed(item: ZooItem) -> bool:
    """已安裝且是目前清單上的版本（清單換新版時，舊的會被當成沒裝，重新下載）。"""
    try:
        data = json.loads((item_dir(item) / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return data.get("sha256") == item.sha256 and bool(item.sha256)


def installed_samples() -> list[tuple[ZooItem, Path]]:
    """已安裝的示範照片資料夾（給「圖片／影片檔」來源的尋找清單）。"""
    return [(it, item_dir(it)) for it in items("samples") if is_installed(it)]


def remove(item: ZooItem) -> None:
    shutil.rmtree(item_dir(item), ignore_errors=True)


# ---- 下載與安裝 -----------------------------------------------------------------------

def install(item: ZooItem, progress: Progress | None = None, cancel: threading.Event | None = None) -> Path:
    """下載 → 驗證 → 解壓。回傳安裝資料夾。已經裝好就直接回傳。"""
    if is_installed(item):
        return item_dir(item)
    if not item.sha256:
        raise ZooError(tr("zoo.err.not_published", name=item.title), tr("zoo.err.not_published_hint"))
    downloads = zoo_dir() / "_downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    _check_space(downloads, item.size * 3)
    zip_path = downloads / item.file
    try:
        _download(item.url, zip_path, item.size, progress, cancel)
        _verify(zip_path, item)
        return _extract(zip_path, item)
    finally:
        zip_path.unlink(missing_ok=True)


def install_zip(path: str | Path) -> ZooItem:
    """匯入離線包（老師在有網路的電腦下載的同一個 zip）。依 sha256 認出是哪一項。"""
    path = Path(path)
    try:
        digest = _sha256_of(path)
    except OSError as e:
        raise ZooError(tr("zoo.err.read_failed", name=path.name, error=str(e)[:200])) from e
    item = next((it for it in CATALOG if it.sha256 and it.sha256 == digest), None)
    if item is None:
        raise ZooError(tr("zoo.err.unknown_pack", name=path.name), tr("zoo.err.unknown_pack_hint"))
    _check_space(zoo_dir(), item.size * 2)
    _extract(path, item)
    return item


def _check_space(folder: Path, need: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    try:
        free = shutil.disk_usage(folder).free
    except OSError:
        return
    if free < need:
        raise ZooError(tr("zoo.err.no_space", need=f"{need / 1e6:.0f}", free=f"{free / 1e6:.0f}"),
                       tr("zoo.err.no_space_hint"))


def _download(url: str, dest: Path, expected: int, progress: Progress | None,
              cancel: threading.Event | None) -> None:
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r, part.open("wb") as f:
            total = int(r.headers.get("Content-Length") or 0) or expected
            done = 0
            if progress:
                progress(0, total)
            while chunk := r.read(CHUNK):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    except ZooError:
        part.unlink(missing_ok=True)
        raise
    except urllib.error.HTTPError as e:
        part.unlink(missing_ok=True)
        raise ZooError(tr("zoo.err.http", code=e.code), tr("zoo.err.http_hint")) from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        part.unlink(missing_ok=True)
        raise ZooError(tr("zoo.err.network", error=str(getattr(e, "reason", e))[:200]),
                       tr("zoo.err.network_hint")) from e
    except OSError as e:
        part.unlink(missing_ok=True)
        raise ZooError(tr("zoo.err.write_failed", error=str(e)[:200]), tr("zoo.err.no_space_hint")) from e
    part.replace(dest)


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path: Path, item: ZooItem) -> None:
    if _sha256_of(path) != item.sha256:
        raise ZooError(tr("zoo.err.checksum", name=item.title), tr("zoo.err.checksum_hint"))


def _extract(zip_path: Path, item: ZooItem) -> Path:
    """解壓到 <id>.part 再改名，中途失敗不會留下裝一半的資料夾。zip 裡的路徑不能跳出資料夾。"""
    target = item_dir(item)
    tmp = target.with_name(target.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        with zipfile.ZipFile(zip_path) as z:
            for info in z.infolist():
                rel = PurePosixPath(info.filename)
                if rel.is_absolute() or ".." in rel.parts or ":" in info.filename:
                    raise ZooError(tr("zoo.err.bad_zip", name=zip_path.name))
                if info.is_dir():
                    continue
                out = tmp.joinpath(*rel.parts)
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, out.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        if item.kind == "model" and not (tmp / f"{item.id}.card.json").is_file():
            raise ZooError(tr("zoo.err.bad_zip", name=zip_path.name))
        (tmp / MARKER).write_text(json.dumps({"id": item.id, "sha256": item.sha256}), encoding="utf-8")
        shutil.rmtree(target, ignore_errors=True)
        tmp.replace(target)
    except zipfile.BadZipFile as e:
        raise ZooError(tr("zoo.err.bad_zip", name=zip_path.name)) from e
    except OSError as e:
        raise ZooError(tr("zoo.err.write_failed", error=str(e)[:200]), tr("zoo.err.no_space_hint")) from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log.info("已安裝 %s 到 %s", item.id, target)
    return target
