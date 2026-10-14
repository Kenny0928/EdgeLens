"""資料集照片與可復原的 metadata；同一 writer 的操作由鎖序列化。"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np

from edgelens import __version__
from edgelens.core.frame import Frame
from edgelens.core.paths import atomic_write_bytes, atomic_write_text
from edgelens.core.project import Project
from edgelens.dataset.dedupe import dhash
from edgelens.metrics.quality import measure

log = logging.getLogger(__name__)

# 刪除的照片先進 _trash，超過這個天數才真的從磁碟移除（避免學生以為刪了、卻跟著專案資料夾被收走）。
TRASH_KEEP_DAYS = 30


def check_relative(file: str) -> None:
    """只做字串層級的檢查（不碰檔案系統、不建 Path 物件），解析 metadata 時每筆紀錄都會呼叫，必須很便宜。

    真正存取檔案前仍要走 dataset_path，才會擋住指到資料集外的符號連結。"""
    if not isinstance(file, str) or not file or "\\" in file or ":" in file:
        raise ValueError("照片路徑不正確，請使用資料集內的相對路徑。")
    if file[0] == "/" or any(part in (".", "..") for part in file.split("/")):
        raise ValueError("照片路徑超出資料集，請重新選取照片。")


def dataset_path(root: Path, file: str) -> Path:
    """拒絕越界與符號連結，防止搬檔或匯出時誤碰專案外資料。

    會 resolve 檔案系統，所以只用在真的要存取檔案的地方；解析紀錄請用 check_relative。"""
    check_relative(file)
    target = root.joinpath(*PurePosixPath(file).parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("照片連結超出資料集，請將照片複製到專案內。")
    return target


def read_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ValueError(f"無法讀取照片 {Path(path).name}，請確認檔案完整。")
    return image


def encode_jpeg(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("照片無法轉成 JPEG，請重新擷取畫面。")
    return buf.tobytes()


def _dict(value: Any) -> dict:
    return {} if value is None else dict(value.to_dict() if hasattr(value, "to_dict") else value)


_RECORD_FIELDS: frozenset[str] | None = None


def _record_fields(cls) -> frozenset[str]:
    """欄位名稱只算一次；每筆紀錄都呼叫 fields() 在幾千張時很可觀。"""
    global _RECORD_FIELDS
    if _RECORD_FIELDS is None:
        _RECORD_FIELDS = frozenset(f.name for f in fields(cls)) - {"dataset_dir", "extra"}
    return _RECORD_FIELDS


@dataclass
class SampleRecord:
    file: str
    class_id: str
    time: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)
    controls: dict = field(default_factory=dict)
    pipeline: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    width: int = 0
    height: int = 0
    sha1: str = ""
    dhash: str = ""
    group_id: str | None = None
    burst_id: str | None = None
    note: str = ""
    edgelens_version: str = __version__
    processed_file: str | None = None
    deleted: bool = False
    original_file: str | None = None
    deleted_at: float | None = None   # 刪除時間（epoch）；舊紀錄沒有，改看 _trash 目錄的修改時間
    dataset_dir: Path | None = field(default=None, repr=False, compare=False)
    extra: dict = field(default_factory=dict, repr=False)

    @property
    def path(self) -> Path:
        if self.dataset_dir is None:
            raise ValueError("照片沒有資料集位置，請用 DatasetWriter 讀取或指定 dataset_dir。")
        return dataset_path(self.dataset_dir, self.file)

    def to_dict(self) -> dict:
        return {**self.extra, **{f.name: getattr(self, f.name) for f in fields(self)
                               if f.name not in ("dataset_dir", "extra")}}

    @classmethod
    def from_dict(cls, data: dict, dataset_dir: Path | None = None) -> SampleRecord:
        known = _record_fields(cls)
        if not isinstance(data, dict) or not isinstance(data.get("class_id"), str):
            raise ValueError("照片紀錄缺少類別，請重建資料集紀錄。")
        # 只做字串檢查：每筆都 resolve 會讓幾千張的資料集每次拍照都卡住 GUI。
        # 存取檔案時（record.path、搬移、匯出）仍會經過 dataset_path 的完整檢查。
        check_relative(data.get("file"))
        for key in ("time", "source", "controls", "pipeline", "quality", "result"):
            if key in data and not isinstance(data[key], dict):
                raise ValueError("照片紀錄欄位格式不正確，請重建資料集紀錄。")
        for key in ("processed_file", "original_file"):
            if data.get(key) is not None:
                check_relative(data[key])
        for key in ("group_id", "burst_id"):
            if data.get(key) is not None and not isinstance(data[key], str):
                raise ValueError("照片分組格式不正確，請使用文字識別碼。")
        fixed, extra = {}, {}
        for k, v in data.items():
            (fixed if k in known else extra)[k] = v
        return cls(**fixed, dataset_dir=dataset_dir, extra=extra)


class DatasetWriter:
    """save_processed 可關閉副本；預設讀 project.extra['dataset']['save_processed']。

    拍照時 metadata 只附加一行；改類別、刪除、復原、重建時才整個重寫（原子替換）。
    不支援多程序同時寫同一資料集，GUI 應共享一個 writer。
    """

    def __init__(self, project: Project, *, save_processed: bool | None = None):
        self.project = project
        self.dataset_dir = project.dataset_dir
        self.metadata_path = self.dataset_dir / "metadata.jsonl"
        self.save_processed = (project.extra.get("dataset", {}).get("save_processed", True)
                               if save_processed is None else save_processed)
        self._lock = threading.RLock()
        # 解析結果快取（file -> 紀錄）與對應的 metadata 檔案簽章。拍照時資料集面板每張都要讀清單，
        # 沒有快取就得每次重新解析整個 jsonl，照片越多越慢。簽章不同（外部改動）就整份重讀。
        self._cache: dict[str, SampleRecord] | None = None
        self._cache_sig: tuple[int, int, int] | None = None

    def _sig(self) -> tuple[int, int, int] | None:
        """(修改時間 ns, 大小, inode)。原子改寫一定換 inode，所以隨身碟（FAT／exFAT 時間精度粗）上
        同一時刻改寫成同樣大小也認得出來。"""
        try:
            st = self.metadata_path.stat()
        except OSError:
            return None
        return st.st_mtime_ns, st.st_size, st.st_ino

    def _invalidate(self) -> None:
        self._cache = None
        self._cache_sig = None

    def _class(self, class_id: str) -> None:
        if self.project.class_by_id(class_id) is None:
            raise ValueError(f"找不到類別 {class_id}，請先在專案新增這個類別。")

    def _load(self) -> list[SampleRecord]:
        """回傳紀錄的淺拷貝：呼叫端（_move、relabel 等）會改屬性，不能動到快取本身。"""
        sig = self._sig()
        if sig is None:
            self._cache, self._cache_sig = {}, None
            return []
        if self._cache is not None and sig == self._cache_sig:
            return [copy.copy(r) for r in self._cache.values()]
        out = {}
        for number, line in enumerate(self.metadata_path.read_bytes().splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = SampleRecord.from_dict(json.loads(line), self.dataset_dir)
                out[record.file] = record
            except (ValueError, TypeError, KeyError, UnicodeError) as exc:
                log.warning("跳過 metadata 第 %d 行：%s", number, exc)
        # 簽章在讀檔前取得：讀的途中若被改，下次簽章不同就會重讀，寧可多讀不可讀舊
        self._cache, self._cache_sig = out, sig
        return [copy.copy(r) for r in out.values()]

    def _append(self, line: bytes) -> None:
        """在 metadata.jsonl 結尾附加一行。

        拍照（含連拍）時每張都會呼叫，所以只附加不重寫整個檔案，否則照片越多越慢。
        當機時最多留下半行，_load 會跳過壞掉的行；這裡先補上缺少的換行，避免新紀錄
        接在半行後面一起壞掉。"""
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "ab+") as f:
            if f.tell() > 0:
                f.seek(-1, 2)
                if f.read(1) != b"\n":
                    line = b"\n" + line
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def _write(self, records: list[SampleRecord]) -> None:
        self._invalidate()   # 寫入途中失敗就保持失效，下次重讀
        atomic_write_text(self.metadata_path, "".join(
            json.dumps(r.to_dict(), ensure_ascii=False, allow_nan=False) + "\n" for r in records))
        sig = self._sig()
        if sig is not None:
            self._cache = {r.file: copy.copy(r) for r in records}
            self._cache_sig = sig

    def _unique(self, path: Path) -> Path:
        base = path
        i = 0
        while path.exists() or path.with_name(path.stem + "_proc.jpg").exists():
            i += 1
            path = base.with_name(f"{base.stem}_{i}{base.suffix}")
        return path

    def save(self, frame: Frame, class_id: str, *, image=None, result=None, pipeline=None,
             source=None, controls=None, quality=None, group_id=None, burst_id=None, note="") -> SampleRecord:
        self._class(class_id)
        raw = frame.jpeg if frame.jpeg else encode_jpeg(frame.image)
        processed = encode_jpeg(image) if image is not None and self.save_processed else None
        stamp = datetime.fromtimestamp(frame.t_wall).astimezone()
        with self._lock:
            path = self._unique(dataset_path(self.dataset_dir,
                f"{class_id}/{stamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]}_{frame.seq}.jpg"))
            proc = path.with_name(path.stem + "_proc.jpg") if processed is not None else None
            record = SampleRecord(
                file=path.relative_to(self.dataset_dir).as_posix(), class_id=class_id,
                time={"iso": stamp.isoformat(timespec="milliseconds"), "epoch": frame.t_wall},
                source=_dict(source) if source is not None else {"id": frame.source_id},
                controls=_dict(controls), pipeline=_dict(pipeline),
                quality=_dict(quality) if quality is not None else measure(frame.image).to_dict(),
                result=_dict(result if result is not None else frame.device_result),
                width=frame.width, height=frame.height, sha1=hashlib.sha1(raw).hexdigest(),
                dhash=f"{dhash(frame.image):016x}", group_id=group_id, burst_id=burst_id, note=note,
                processed_file=proc.relative_to(self.dataset_dir).as_posix() if proc else None,
                dataset_dir=self.dataset_dir)
            # 先序列化驗證，避免參數不能存成 JSON 時留下一張孤立照片。
            line = json.dumps(record.to_dict(), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
            SampleRecord.from_dict(record.to_dict(), self.dataset_dir)
            created = []
            try:
                atomic_write_bytes(path, raw)
                created.append(path)
                if proc:
                    atomic_write_bytes(proc, processed)
                    created.append(proc)
                pre = self._sig()
                cache_ok = self._cache is not None and pre == self._cache_sig
                self._append(line)
                try:
                    self._cache_after_append(cache_ok, pre, line)
                except Exception:   # 快取只是加速；紀錄已寫入，不能因此回頭刪照片
                    self._invalidate()
            except BaseException:
                self._invalidate()
                for item in created:
                    item.unlink(missing_ok=True)
                raise
            return record

    def _cache_after_append(self, cache_ok: bool, pre: tuple[int, int, int] | None, line: bytes) -> None:
        """附加一行後同步快取；簽章或大小對不上（可能有外部同時寫入）就放棄快取。"""
        post = self._sig()
        grew = (post[1] - (pre[1] if pre else 0)) if post else -1
        if not cache_ok or post is None or grew not in (len(line), len(line) + 1):
            self._invalidate()
            return
        record = SampleRecord.from_dict(json.loads(line), self.dataset_dir)   # 與重讀檔案得到的內容一致
        self._cache[record.file] = record
        self._cache_sig = post

    def list_samples(self, class_id: str | None = None, *, include_deleted: bool = False) -> list[SampleRecord]:
        with self._lock:
            return [r for r in self._load() if (include_deleted or not r.deleted)
                    and (class_id is None or r.class_id == class_id)]

    def _find(self, records, file, *, deleted=False):
        file = file.file if isinstance(file, SampleRecord) else str(file)
        dataset_path(self.dataset_dir, file)
        for r in records:
            if r.deleted == deleted and (r.file == file or (deleted and r.original_file == file)):
                return r
        raise FileNotFoundError(f"找不到照片 {file}，請重新整理資料集。")

    def _move(self, record, destination, records):
        origin = record.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        moves = [(origin, destination)]
        new_proc = None
        if record.processed_file:
            old_proc = dataset_path(self.dataset_dir, record.processed_file)
            new_proc = destination.with_name(destination.stem + "_proc.jpg")
            moves.append((old_proc, new_proc))
        moved = []
        try:
            for src, dst in moves:
                src.rename(dst)
                moved.append((src, dst))
            record.file = destination.relative_to(self.dataset_dir).as_posix()
            record.processed_file = new_proc.relative_to(self.dataset_dir).as_posix() if new_proc else None
            self._write(records)
        except BaseException:
            self._invalidate()
            for src, dst in reversed(moved):
                dst.rename(src)
            raise
        return record

    def relabel(self, file: str, new_class_id: str) -> SampleRecord:
        self._class(new_class_id)
        with self._lock:
            records = self._load()
            record = self._find(records, file)
            if record.class_id == new_class_id:
                return record
            destination = self._unique(dataset_path(self.dataset_dir, f"{new_class_id}/{record.path.name}"))
            record.class_id = new_class_id
            return self._move(record, destination, records)

    def delete(self, file: str) -> SampleRecord:
        with self._lock:
            records = self._load()
            record = self._find(records, file)
            destination = dataset_path(self.dataset_dir, f"_trash/{uuid.uuid4().hex}/{record.file}")
            record.original_file, record.deleted = record.file, True
            record.deleted_at = time.time()
            return self._move(record, destination, records)

    def restore(self, file: str) -> SampleRecord:
        """接受 delete 回傳的 file 或原路徑；撞名時保留兩張照片。"""
        with self._lock:
            records = self._load()
            record = self._find(records, file, deleted=True)
            self._class(record.class_id)
            destination = self._unique(dataset_path(self.dataset_dir, record.original_file))
            record.deleted, record.original_file, record.deleted_at = False, None, None
            return self._move(record, destination, records)

    def _trash_age_time(self, record: SampleRecord, trash_root: Path) -> float:
        """刪除時間；舊紀錄沒有 deleted_at，用 _trash/<uuid> 目錄的修改時間（刪除當下建立）。"""
        if isinstance(record.deleted_at, (int, float)):
            return float(record.deleted_at)
        parts = PurePosixPath(record.file).parts
        try:
            return (trash_root / parts[1]).stat().st_mtime
        except (OSError, IndexError):
            return 0.0   # 目錄不見了：沒有東西值得保留

    def purge_trash(self, older_than_days: float | None = None) -> int:
        """真的刪除垃圾桶裡的照片，回傳移除的張數。

        older_than_days=None 是「清空垃圾桶」：全部已刪除紀錄與 _trash 下的孤立檔案都清掉。
        給天數則只清超過期限的紀錄與孤立目錄。只會動 dataset/_trash 之內的東西。"""
        with self._lock:
            trash_root = self.dataset_dir / "_trash"
            cutoff = None if older_than_days is None else time.time() - older_than_days * 86400
            records = self._load()
            try:
                trash_real = trash_root.resolve()
                dataset_real = self.dataset_dir.resolve()
            except OSError:
                return 0
            # _trash 本身是符號連結（例如別人做壞的專案資料夾指向家目錄）時完全不動：
            # 否則下面清孤立項目會跟著連結刪掉資料集外的東西，而開專案時的自動清理不需要任何點擊
            if trash_root.is_symlink() or trash_real.parent != dataset_real:
                if trash_root.exists() or trash_root.is_symlink():
                    log.warning("垃圾桶 %s 不是資料集內的一般資料夾，略過清理", trash_root)
                return 0

            def in_trash(rel: str | None) -> Path | None:
                """回傳可安全刪除的路徑；不在 _trash 內或有符號連結越界就回 None。"""
                if not rel:
                    return None
                try:
                    if PurePosixPath(rel).parts[:1] != ("_trash",):
                        return None
                    path = dataset_path(self.dataset_dir, rel)
                    return path if path.resolve().is_relative_to(trash_real) else None
                except (ValueError, OSError):
                    return None

            def uuid_of(rel: str | None) -> str | None:
                parts = PurePosixPath(rel).parts if rel else ()
                return parts[1] if len(parts) > 2 and parts[0] == "_trash" else None

            referenced = {uuid_of(r.file) for r in records if r.deleted}
            # 孤立目錄要在刪檔前先找出來：刪檔會改變目錄的修改時間
            orphans = []
            if trash_root.is_dir():
                for entry in trash_root.iterdir():
                    if entry.name in referenced:
                        continue
                    try:
                        old = cutoff is None or entry.lstat().st_mtime < cutoff
                    except OSError:
                        continue
                    if old:
                        orphans.append(entry)

            keep, removed = [], 0
            for r in records:
                if not (r.deleted and uuid_of(r.file)):
                    keep.append(r)
                    continue
                if cutoff is not None and self._trash_age_time(r, trash_root) >= cutoff:
                    keep.append(r)
                    continue
                targets = [in_trash(r.file)] + ([in_trash(r.processed_file)] if r.processed_file else [])
                if None in targets:
                    log.warning("略過垃圾桶內路徑異常的照片 %s", r.file)
                    keep.append(r)
                    continue
                try:
                    for path in targets:
                        path.unlink(missing_ok=True)
                except OSError as exc:
                    log.warning("無法刪除垃圾桶照片 %s：%s", r.file, exc)
                    keep.append(r)
                    continue
                removed += 1
                # 從檔案所在往上收掉變空的資料夾，停在 _trash 為止
                folder = targets[0].parent
                while folder != trash_root and folder.is_relative_to(trash_root):
                    try:
                        folder.rmdir()
                    except OSError:
                        break
                    folder = folder.parent
            if removed:
                self._write(keep)
            for entry in orphans:
                try:
                    if entry.is_symlink() or entry.is_file():
                        entry.unlink()
                    else:
                        shutil.rmtree(entry)
                except OSError as exc:
                    log.warning("無法清除垃圾桶殘留 %s：%s", entry.name, exc)
            return removed

    def rebuild_metadata(self) -> list[SampleRecord]:
        with self._lock:
            records = self._load()
            known = {r.file for r in records} | {r.processed_file for r in records}
            for cid in self.project.class_ids():
                folder = dataset_path(self.dataset_dir, cid)
                for path in sorted(folder.glob("*")):
                    rel = path.relative_to(self.dataset_dir).as_posix()
                    if rel in known or path.stem.endswith("_proc") or path.suffix.lower() not in {
                            ".jpg", ".jpeg", ".png", ".bmp", ".webp"} or not path.is_file():
                        continue
                    try:
                        dataset_path(self.dataset_dir, rel)
                        image = read_image(path)
                        epoch = path.stat().st_mtime
                        records.append(SampleRecord(
                            file=rel, class_id=cid, width=image.shape[1], height=image.shape[0],
                            time={"iso": datetime.fromtimestamp(epoch).astimezone().isoformat(), "epoch": epoch},
                            sha1=hashlib.sha1(path.read_bytes()).hexdigest(), dhash=f"{dhash(image):016x}",
                            quality=measure(image).to_dict(), dataset_dir=self.dataset_dir))
                    except (OSError, ValueError, cv2.error) as exc:
                        log.warning("跳過無法讀取的照片 %s：%s", path.name, exc)
            self._write(records)
            return [r for r in records if not r.deleted]


_shared: dict[str, DatasetWriter] = {}
_shared_lock = threading.Lock()


def shared_writer(project: Project) -> DatasetWriter:
    """同一個專案資料夾在整個程式裡只用一個 DatasetWriter。

    改類別、刪除、復原會重寫整個 metadata.jsonl；如果拍照面板和資料集瀏覽器各有一個 writer
    （各自的鎖），重寫時可能蓋掉另一個 writer 剛附加的紀錄，照片就從資料集裡「消失」。
    以資料夾路徑為鍵；專案物件重新載入（例如改了類別）時更新 writer.project，不另建新的。"""
    key = str(Path(project.root).resolve())
    with _shared_lock:
        writer = _shared.get(key)
        if writer is None:
            writer = DatasetWriter(project)
            _shared[key] = writer
        else:
            writer.project = project
        return writer
