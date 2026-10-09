"""縮圖快取、背景載入與資料集面板共用的小工具。

為什麼自己做：資料集可能有數千張照片，縮圖必須在背景執行緒產生，並存到磁碟快取，
第二次開啟才不用重新解碼。快取鍵優先用照片內容的 sha1（改類別搬檔後仍然有效）。
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import subprocess
import sys
import threading
import weakref
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QImage, QPixmap

from edgelens.core.paths import app_cache_dir
from edgelens.dataset.writer import DatasetWriter, SampleRecord
from edgelens.dataset.writer import shared_writer as _shared_writer

log = logging.getLogger(__name__)

THUMB_SIZE = 160
WORKERS = 2


def bgr_to_qimage(image: np.ndarray) -> QImage:
    """BGR／灰階 ndarray 轉 QImage（複製資料，離開函式後仍安全）。"""
    if image.ndim == 2:
        h, w = image.shape
        return QImage(np.ascontiguousarray(image).data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    rgb = np.ascontiguousarray(image[:, :, ::-1])
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


def decode_file(path: Path, reduced: int = 1) -> np.ndarray | None:
    """用 imdecode 讀檔（中文路徑也可以）；reduced=2/4/8 時 JPEG 會在解碼時直接縮小，比較快。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if not data.size:
            return None
        flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4,
                8: cv2.IMREAD_REDUCED_COLOR_8}.get(reduced, cv2.IMREAD_COLOR)
        return cv2.imdecode(data, flag)
    except OSError:
        return None


def thumb_key(record: SampleRecord, size: int = THUMB_SIZE) -> str:
    """快取鍵。有內容 sha1 就用它；沒有（學生自己丟的照片）改用路徑＋修改時間。"""
    if record.sha1:
        base = f"{record.sha1}:{size}"
    else:
        try:
            st = record.path.stat()
            base = f"{record.file}:{st.st_mtime_ns}:{st.st_size}:{size}"
        except (OSError, ValueError):
            base = f"{record.file}:{size}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def thumbs_dir(project) -> Path:
    return app_cache_dir() / "thumbs" / project.id


def make_thumb(path: Path, size: int) -> np.ndarray | None:
    """產生縮圖；JPEG 先用縮小解碼加速，再精確縮到最大邊 size。"""
    img = decode_file(path, 4)
    if img is None:
        img = decode_file(path, 1)
    if img is None:
        return None
    if img.shape[0] < size and img.shape[1] < size:
        img = decode_file(path, 1)
        if img is None:
            return None
    h, w = img.shape[:2]
    scale = size / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return img


IDLE_EXIT_S = 2.0
_live_loaders: weakref.WeakSet = weakref.WeakSet()


@atexit.register
def _stop_all_loaders() -> None:
    """程式結束前停掉還沒停的縮圖執行緒（見 ThumbLoader 的說明）。"""
    for loader in list(_live_loaders):
        try:
            loader.stop()
        except RuntimeError:   # Qt 物件已經被刪除
            pass


class ThumbLoader(QObject):
    """背景產生縮圖。後送先做（最近捲到的先載入），完成後用 signal 回到 GUI 執行緒。"""

    thumbReady = Signal(str, QImage)   # key, 縮圖

    def __init__(self, project, size: int = THUMB_SIZE, parent: QObject | None = None):
        super().__init__(parent)
        self.size = size
        self.folder = thumbs_dir(project)
        self.generated = 0        # 重新產生（沒命中磁碟快取）的張數，測試用
        self.disk_hits = 0
        self._stack: list[tuple[str, Path]] = []
        self._pending: set[str] = set()
        self._cond = threading.Condition()
        self._stopped = False
        # 執行緒有工作才啟動、閒置一會兒就自己結束：沒有被 stop() 的 loader（例如視窗沒正常關）
        # 也不會留下執行緒。程式結束時還在跑的執行緒若正在 OpenCV／Qt 的 C++ 程式裡，
        # 會在直譯器收尾時當掉（libc++abi: recursive_mutex lock failed）。
        self._threads: list[threading.Thread] = []
        _live_loaders.add(self)

    def request(self, key: str, path: Path) -> None:
        with self._cond:
            if key in self._pending or self._stopped:
                return
            self._pending.add(key)
            self._stack.append((key, path))
            self._cond.notify()
            self._threads = [t for t in self._threads if t.is_alive()]
            if len(self._threads) < WORKERS:
                t = threading.Thread(target=self._run, daemon=True, name=f"thumb-{len(self._threads)}")
                self._threads.append(t)
                t.start()

    def pending(self) -> int:
        with self._cond:
            return len(self._pending)

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._stack.clear()
            self._cond.notify_all()
            threads = list(self._threads)
        for t in threads:
            if t is not threading.current_thread():
                t.join(timeout=2)
        _live_loaders.discard(self)

    def _run(self) -> None:
        while True:
            with self._cond:
                if not self._stack and not self._stopped:
                    self._cond.wait(timeout=IDLE_EXIT_S)
                if self._stopped:
                    return
                if not self._stack:   # 閒置太久：結束這條執行緒，下次有工作再開
                    me = threading.current_thread()
                    self._threads = [t for t in self._threads if t is not me]
                    return
                key, path = self._stack.pop()
            image = self._load(key, path)
            with self._cond:
                self._pending.discard(key)
            if image is not None and not self._stopped:
                try:
                    self.thumbReady.emit(key, image)
                except RuntimeError:   # C++ 物件已刪除
                    return

    def _load(self, key: str, path: Path) -> QImage | None:
        cache_file = self.folder / f"{key}.jpg"
        img = decode_file(cache_file) if cache_file.is_file() else None
        if img is not None:
            self.disk_hits += 1
        else:
            img = make_thumb(path, self.size)
            if img is None:
                return None
            self.generated += 1
            try:
                self.folder.mkdir(parents=True, exist_ok=True)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    tmp = cache_file.with_suffix(".tmp")
                    tmp.write_bytes(buf.tobytes())
                    tmp.replace(cache_file)
            except OSError as exc:
                log.debug("縮圖快取寫入失敗：%s", exc)
        return bgr_to_qimage(img)


class PixmapCache:
    """記憶體中的縮圖（LRU），避免捲動時反覆從磁碟讀。"""

    def __init__(self, capacity: int = 4000):
        self.capacity = capacity
        self._items: OrderedDict[str, QPixmap] = OrderedDict()

    def get(self, key: str) -> QPixmap | None:
        pm = self._items.get(key)
        if pm is not None:
            self._items.move_to_end(key)
        return pm

    def put(self, key: str, pm: QPixmap) -> None:
        self._items[key] = pm
        self._items.move_to_end(key)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)


class BgTask(QObject):
    """在背景執行緒跑一個函式，結果用 signal 回 GUI。progress 可由函式內呼叫 emit_progress。"""

    finished = Signal(object, object)      # result, error（其中一個是 None）
    progress = Signal(int, int)

    def __init__(self, fn, parent: QObject | None = None):
        super().__init__(parent)
        self._fn = fn
        self.cancel_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            result, error = self._fn(self), None
        except BaseException as exc:   # noqa: BLE001 - 交給 GUI 顯示
            result, error = None, exc
        try:
            self.finished.emit(result, error)
        except RuntimeError:
            pass

    def emit_progress(self, done: int, total: int) -> None:
        try:
            self.progress.emit(done, total)
        except RuntimeError:
            pass

    def wait(self, ms: int = 5000) -> None:
        self._thread.join(ms / 1000)


# ---- 共享 writer、變動偵測、開啟檔案總管 ----------------------------------------------


def shared_writer(project) -> DatasetWriter:
    """同一個專案共用一個 DatasetWriter（見 edgelens.dataset.writer.shared_writer）。"""
    return _shared_writer(project)


def metadata_signature(project) -> tuple[int, int]:
    """metadata.jsonl 的 (修改時間 ns, 大小)；檔案不存在回傳 (0, 0)。輕量，可以每 2 秒檢查。"""
    try:
        st = (project.root / "dataset" / "metadata.jsonl").stat()
        return st.st_mtime_ns, st.st_size
    except OSError:
        return 0, 0


def open_folder(path: Path) -> None:
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def reveal_in_folder(path: Path) -> None:
    """在檔案總管中選取檔案；不支援時退而開啟所在資料夾。"""
    path = Path(path)
    try:
        if path.is_file():
            if sys.platform == "win32":
                subprocess.Popen(["explorer", f"/select,{path}"])  # noqa: S603,S607
                return
            if sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(path)])  # noqa: S603,S607
                return
    except OSError as exc:
        log.debug("無法選取檔案：%s", exc)
    open_folder(path if path.is_dir() else path.parent)
