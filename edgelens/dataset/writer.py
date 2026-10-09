"""資料集照片與可復原的 metadata；同一 writer 的操作由鎖序列化。"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
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


def dataset_path(root: Path, file: str) -> Path:
    """拒絕越界與符號連結，防止搬檔或匯出時誤碰專案外資料。"""
    if not isinstance(file, str) or not file or "\\" in file or ":" in file:
        raise ValueError("照片路徑不正確，請使用資料集內的相對路徑。")
    rel = PurePosixPath(file)
    if rel.is_absolute() or any(part in (".", "..") for part in file.split("/")):
        raise ValueError("照片路徑超出資料集，請重新選取照片。")
    target = root.joinpath(*rel.parts)
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
        known = {f.name for f in fields(cls)} - {"dataset_dir", "extra"}
        if not isinstance(data, dict) or not isinstance(data.get("class_id"), str):
            raise ValueError("照片紀錄缺少類別，請重建資料集紀錄。")
        dataset_path(dataset_dir or Path.cwd(), data.get("file"))
        for key in ("time", "source", "controls", "pipeline", "quality", "result"):
            if key in data and not isinstance(data[key], dict):
                raise ValueError("照片紀錄欄位格式不正確，請重建資料集紀錄。")
        for key in ("processed_file", "original_file"):
            if data.get(key) is not None:
                dataset_path(dataset_dir or Path.cwd(), data[key])
        for key in ("group_id", "burst_id"):
            if data.get(key) is not None and not isinstance(data[key], str):
                raise ValueError("照片分組格式不正確，請使用文字識別碼。")
        return cls(**{k: v for k, v in data.items() if k in known}, dataset_dir=dataset_dir,
                   extra={k: v for k, v in data.items() if k not in known})


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

    def _class(self, class_id: str) -> None:
        if self.project.class_by_id(class_id) is None:
            raise ValueError(f"找不到類別 {class_id}，請先在專案新增這個類別。")

    def _load(self) -> list[SampleRecord]:
        if not self.metadata_path.exists():
            return []
        out = {}
        for number, line in enumerate(self.metadata_path.read_bytes().splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = SampleRecord.from_dict(json.loads(line), self.dataset_dir)
                out[record.file] = record
            except (ValueError, TypeError, KeyError, UnicodeError) as exc:
                log.warning("跳過 metadata 第 %d 行：%s", number, exc)
        return list(out.values())

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
        atomic_write_text(self.metadata_path, "".join(
            json.dumps(r.to_dict(), ensure_ascii=False, allow_nan=False) + "\n" for r in records))

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
                self._append(line)
            except BaseException:
                for item in created:
                    item.unlink(missing_ok=True)
                raise
            return record

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
            return self._move(record, destination, records)

    def restore(self, file: str) -> SampleRecord:
        """接受 delete 回傳的 file 或原路徑；撞名時保留兩張照片。"""
        with self._lock:
            records = self._load()
            record = self._find(records, file, deleted=True)
            self._class(record.class_id)
            destination = self._unique(dataset_path(self.dataset_dir, record.original_file))
            record.deleted, record.original_file = False, None
            return self._move(record, destination, records)

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
