"""檔案來源：影片檔、單張圖片、圖片資料夾（沒有相機也能練習）。

依路徑自動判斷種類。依 FPS 控制回傳速度（不會一次把影片讀完）；播完且不循環時 read()
回傳 None（不是錯誤），最後一張畫面的 meta["ended"] 為 True，之後 ``ended`` 屬性也是 True。
實作上多讀一張（lookahead），才能在「最後一張」就知道已經到結尾。
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from edgelens.core.controls import ControlResult, ControlSpec, ControlType
from edgelens.core.frame import Frame
from edgelens.core.housekeeping import VIDEO_TEMP_PREFIX
from edgelens.i18n import tr
from edgelens.sources.base import FrameSource, SourceCapabilities, SourceError, SourceInfo, register_source

log = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv", ".flv", ".mpg", ".mpeg"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
JPEG_EXTS = {".jpg", ".jpeg"}
DEFAULT_IMAGE_FPS = 5.0
SAMPLES_FPS = 0.5           # 示範照片：每張停 2 秒，來得及看清楚框


def natural_key(name: str) -> list[Any]:
    """自然排序：img2 排在 img10 前面。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def classify_path(path: str | Path) -> str:
    """"video"／"image"／"folder"／""（不存在或不支援）。"""
    p = Path(path)
    if p.is_dir():
        return "folder"
    ext = p.suffix.lower()
    if p.is_file():
        if ext in VIDEO_EXTS:
            return "video"
        if ext in IMAGE_EXTS:
            return "image"
    return ""


def list_images(folder: Path) -> list[Path]:
    """資料夾裡的圖片，依檔名（自然排序）；略過隱藏檔（例如 macOS 的 ._xxx.jpg）。"""
    files = [p for p in folder.iterdir()
             if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda p: natural_key(p.name))


def _exif_orientation(data: bytes) -> int | None:
    """讀 JPEG 的 EXIF 方向（1–8）。沒有 EXIF 回傳 None。"""
    try:
        if data[:2] != b"\xff\xd8":
            return None
        pos = 2
        while pos + 4 <= len(data) and data[pos] == 0xFF:
            marker = data[pos + 1]
            seglen = int.from_bytes(data[pos + 2:pos + 4], "big")
            if marker == 0xE1 and data[pos + 4:pos + 10] == b"Exif\x00\x00":
                tiff = pos + 10
                endian = "little" if data[tiff:tiff + 2] == b"II" else "big"
                ifd = tiff + int.from_bytes(data[tiff + 4:tiff + 8], endian)
                count = int.from_bytes(data[ifd:ifd + 2], endian)
                for i in range(count):
                    e = ifd + 2 + 12 * i
                    if int.from_bytes(data[e:e + 2], endian) == 0x0112:
                        return int.from_bytes(data[e + 8:e + 10], endian)
                return None
            if marker in (0xDA, 0xD9):
                return None
            pos += 2 + seglen
    except (IndexError, ValueError):
        return None
    return None


def read_image(path: Path) -> tuple[np.ndarray, bytes | None] | None:
    """讀一張圖。用 np.fromfile + imdecode，Windows 中文路徑才不會壞。
    jpg 檔回傳原始位元組（存檔時直接寫入、不重新壓縮）；但 EXIF 有旋轉資訊時，
    OpenCV 解碼出來的圖已經轉正，原始位元組就不能代表這張圖，所以不給。"""
    try:
        data = np.fromfile(str(path), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError as e:
        log.warning("讀不到圖片 %s：%s", path, e)
        return None
    if img is None:
        log.warning("不是有效的圖片：%s", path)
        return None
    jpeg = None
    if path.suffix.lower() in JPEG_EXTS and _exif_orientation(data.tobytes()) in (None, 1):
        jpeg = data.tobytes()
    return img, jpeg


# ---- 影片路徑（Windows 非 ASCII）-----------------------------------------------------


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _short_path(path: str) -> str | None:
    """Windows 的 8.3 短路徑；取不到（沒有啟用 8.3 或非 Windows）回傳 None。"""
    try:
        import ctypes

        buf = ctypes.create_unicode_buffer(32768)
        n = ctypes.windll.kernel32.GetShortPathNameW(path, buf, len(buf))  # type: ignore[attr-defined]
        if n and buf.value:
            return buf.value
    except (AttributeError, OSError):
        pass
    return None


def openable_video_path(path: str) -> tuple[str, str | None]:
    """回傳 (給 cv2.VideoCapture 的路徑, 要清理的暫存資料夾或 None)。

    Windows 上 OpenCV 的 VideoCapture 用「ANSI 編碼」開檔，路徑含中文等非 ASCII 字元時
    會開不起來（沒有錯誤訊息，只是 isOpened() 為 False）。所以先試 8.3 短路徑
    （全是 ASCII）；取不到再把影片複製到暫存資料夾（路徑改成 ASCII）再開，關閉時刪除。
    其他平台直接用原路徑。"""
    if not _is_windows() or path.isascii():
        return path, None
    short = _short_path(path)
    if short and short.isascii():
        return short, None
    tmpdir = tempfile.mkdtemp(prefix=VIDEO_TEMP_PREFIX)
    dst = os.path.join(tmpdir, "video" + Path(path).suffix.lower())
    shutil.copyfile(path, dst)
    return dst, tmpdir


# ---- 讀取器 ------------------------------------------------------------------------


class _Item:
    __slots__ = ("image", "jpeg", "name", "index")

    def __init__(self, image: np.ndarray, jpeg: bytes | None, name: str, index: int):
        self.image, self.jpeg, self.name, self.index = image, jpeg, name, index


class _ImageReader:
    """依序讀圖片（單張或資料夾），一次只解碼一張，不佔很多記憶體。"""

    def __init__(self, files: list[Path]):
        self.files = files
        self.total = len(files)
        self.pos = 0
        self.has_jpeg = any(f.suffix.lower() in JPEG_EXTS for f in files)

    def read(self) -> _Item | None:
        while self.pos < len(self.files):
            i = self.pos
            self.pos += 1
            got = read_image(self.files[i])
            if got is not None:   # 壞掉的檔案跳過
                return _Item(got[0], got[1], self.files[i].name, i + 1)
        return None

    def rewind(self) -> None:
        self.pos = 0

    def release(self) -> None:
        pass


class _VideoReader:
    def __init__(self, cap: Any, name: str, total: int | None, fps: float, tmpdir: str | None):
        self.cap, self.name, self.total, self.fps, self.tmpdir = cap, name, total, fps, tmpdir
        self.pos = 0
        self.has_jpeg = False

    def read(self) -> _Item | None:
        ok, img = self.cap.read()
        if not ok or img is None:
            return None
        self.pos += 1
        return _Item(img, None, self.name, self.pos)

    def rewind(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.pos = 0

    def release(self) -> None:
        self.cap.release()
        if self.tmpdir:
            shutil.rmtree(self.tmpdir, ignore_errors=True)


@register_source
class FileSource(FrameSource):
    kind = "file"
    display_name_key = "source.file.name"
    description_key = "source.file.desc"
    discover_is_safe = True
    manual_add = True

    def __init__(self, info: SourceInfo):
        super().__init__(info)
        p = info.params
        self._path = str(p.get("path", "") or "")
        self._reader: _ImageReader | _VideoReader | None = None
        self._kind = ""
        self._natural_fps = DEFAULT_IMAGE_FPS
        self._values: dict[str, Any] = {
            "loop": bool(p.get("loop", True)),
            "fps": float(p["fps"]) if p.get("fps") else 0.0,   # 0＝尚未決定，open() 時依來源設定
            "paused": bool(p.get("paused", False)),
        }
        self._open = False
        self._peek: _Item | None = None
        self._has_peek = False
        self._next_due = 0.0
        self._step = False
        self._ended = False
        self._delivered = 0

    # ---- 類別層級 ------------------------------------------------------------------

    @classmethod
    def discover(cls, timeout: float = 2.0) -> list[SourceInfo]:
        """已下載的示範照片（模型庫，例如 Objects365）；其他檔案請用手動加入選路徑。"""
        from edgelens import zoo

        found = []
        for item, folder in zoo.installed_samples():
            info = cls.info_from_params({"path": str(folder), "fps": SAMPLES_FPS, "loop": True})
            info.name = item.title
            found.append(info)
        return found

    @classmethod
    def param_specs(cls) -> list[ControlSpec]:
        return [ControlSpec("path", ControlType.STR, "control.file.path", default="", group="source",
                            extra={"browse": "file_or_folder"})]

    @classmethod
    def info_from_params(cls, params: dict[str, Any]) -> SourceInfo:
        raw = str(params.get("path", "") or "").strip()
        path = os.path.abspath(os.path.expanduser(raw)) if raw else ""
        clean = dict(params)
        clean["path"] = path
        name = Path(path).name or Path(path).anchor or tr("source.file.name")
        return SourceInfo(kind=cls.kind, id=f"file:{path}", name=name, params=clean, detail=path)

    @classmethod
    def contract_fixture(cls, tmp_path: Path) -> SourceInfo | None:
        path = tmp_path / "clip.avi"
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (160, 120))
        for i in range(20):
            img = np.full((120, 160, 3), 40, np.uint8)
            cv2.rectangle(img, (i * 5, 30), (i * 5 + 30, 80), (0, 200, 255), -1)
            w.write(img)
        w.release()
        return cls.info_from_params({"path": str(path)})

    # ---- 開啟與關閉 ----------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def ended(self) -> bool:
        """不循環、已經播完。"""
        return self._ended

    def open(self) -> None:
        if self._open:
            return
        if not self._path:
            raise SourceError(tr("source.file.err.no_path"))
        kind = classify_path(self._path)
        if not kind:
            if not os.path.exists(self._path):
                raise SourceError(tr("source.file.err.not_found", path=self._path))
            raise SourceError(tr("source.file.err.unsupported", name=Path(self._path).name))
        if kind == "video":
            self._reader, self._natural_fps = self._open_video(self._path)
        else:
            path = Path(self._path)
            files = list_images(path) if kind == "folder" else [path]
            if not files:
                raise SourceError(tr("source.file.err.empty_folder", path=self._path))
            self._reader = _ImageReader(files)
            self._natural_fps = DEFAULT_IMAGE_FPS
        self._kind = kind
        if self._values["fps"] <= 0:
            self._values["fps"] = self._natural_fps
        self._open = True
        self._reset_playback()

    def _open_video(self, path: str) -> tuple[_VideoReader, float]:
        real, tmpdir = openable_video_path(path)
        cap = cv2.VideoCapture(real)
        if not cap.isOpened():
            cap.release()
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
            raise SourceError(tr("source.file.err.video_open", path=Path(path).name),
                              hint=tr("source.file.hint.video_open"))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if fps and 1.0 <= fps <= 240.0 else 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return _VideoReader(cap, Path(path).name, total if total > 0 else None, fps, tmpdir), fps

    def close(self) -> None:
        reader, self._reader = self._reader, None
        self._open = False
        self._peek, self._has_peek = None, False
        if reader is not None:
            reader.release()

    def _reset_playback(self) -> None:
        if self._reader is not None:
            self._reader.rewind()
        self._peek, self._has_peek = None, False
        self._ended = False
        self._delivered = 0
        self._next_due = time.monotonic()

    # ---- 讀取 ----------------------------------------------------------------------

    def _pull(self) -> _Item | None:
        if self._has_peek:
            item, self._peek, self._has_peek = self._peek, None, False
            return item
        return self._reader.read()  # type: ignore[union-attr]

    def read(self, timeout: float = 1.0) -> Frame | None:
        if not self._open or self._reader is None:
            raise SourceError(tr("source.file.err.not_open"))
        if self._ended and self._values["loop"]:
            self._reset_playback()   # 播完後才打開「循環」：從頭再播
        if self._ended:
            time.sleep(min(max(timeout, 0.0), 0.05))
            return None
        stepping = False
        if self._values["paused"]:
            if not self._step:
                time.sleep(min(max(timeout, 0.0), 0.05))
                return None
            self._step = False
            stepping = True
        if not stepping:
            wait = self._next_due - time.monotonic()
            if wait > timeout:
                time.sleep(max(timeout, 0.0))
                return None
            if wait > 0:
                time.sleep(wait)
        period = 1.0 / max(float(self._values["fps"]), 0.1)
        # 落後太多時直接跳到現在，不補播（和即時相機一樣）
        self._next_due = max(self._next_due + period, time.monotonic()) if not stepping \
            else time.monotonic() + period

        item = self._pull()
        if item is None and self._values["loop"] and self._delivered:
            self._reader.rewind()
            item = self._reader.read()
        if item is None:
            self._ended = True
            return None
        # 多讀下一張，才知道這張是不是最後一張
        nxt = self._reader.read()
        if nxt is None and self._values["loop"]:
            self._reader.rewind()
            nxt = self._reader.read()
        self._peek, self._has_peek = nxt, True
        last = nxt is None and not self._values["loop"]
        if last:
            self._ended = True
        self._delivered += 1
        meta = {"file": item.name, "index": item.index, "total": self._reader.total, "kind": self._kind,
                "ended": last}
        return Frame(image=np.ascontiguousarray(item.image), seq=self._next_seq(), t_cap=time.monotonic(),
                     source_id=self.source_id, jpeg=item.jpeg, meta=meta)

    # ---- 參數 ----------------------------------------------------------------------

    def _specs(self) -> list[ControlSpec]:
        return [
            ControlSpec("loop", ControlType.BOOL, "control.file.loop", default=True, group="camera"),
            ControlSpec("fps", ControlType.FLOAT, "control.fps", default=float(self._values["fps"] or self._natural_fps),
                        min=0.5, max=240.0, unit="fps", group="camera"),
            ControlSpec("paused", ControlType.BOOL, "control.file.paused", default=False, group="camera"),
            ControlSpec("restart", ControlType.ACTION, "control.file.restart", group="action"),
            ControlSpec("step", ControlType.ACTION, "control.file.step", group="action"),
        ]

    def controls(self) -> list[ControlSpec]:
        return self._specs() if self._open else []

    def get_control(self, key: str) -> Any:
        if key not in self._values:
            raise KeyError(key)
        return self._values[key]

    def set_control(self, key: str, value: Any) -> ControlResult:
        spec = next((s for s in self._specs() if s.key == key), None)
        if spec is None or spec.type == ControlType.ACTION:
            return ControlResult.failed(tr("source.file.msg.unsupported", key=key))
        try:
            v = spec.validate(value)
        except ValueError as e:
            return ControlResult.failed(str(e), value)
        self._values[key] = v
        return ControlResult(ok=True, value=v, verified=True)

    def run_action(self, key: str, **kwargs: Any) -> ControlResult:
        if not self._open:
            return ControlResult.failed(tr("source.file.err.not_open"))
        if key == "restart":
            self._reset_playback()
            return ControlResult(ok=True)
        if key == "step":
            if not self._values["paused"]:
                return ControlResult.failed(tr("source.file.msg.step_need_pause"))
            self._step = True
            return ControlResult(ok=True)
        return super().run_action(key, **kwargs)

    def capabilities(self) -> SourceCapabilities:
        reader = self._reader
        return SourceCapabilities(
            actions=["restart", "step"],
            provides_jpeg=bool(reader and reader.has_jpeg),
            live_controls=True,
            max_fps=float(self._values["fps"] or self._natural_fps),
            extra={"kind": self._kind, "total": getattr(reader, "total", None), "path": self._path},
        )
