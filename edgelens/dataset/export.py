"""將資料集打包成教學平台可讀的資料夾 ZIP。"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import cv2

from edgelens.core.paths import safe_filename, unique_path
from edgelens.core.project import Project
from edgelens.dataset.dedupe import dhash
from edgelens.dataset.split import PARTS, SplitResult, load_splits, split_dataset
from edgelens.dataset.writer import DatasetWriter, dataset_path, encode_jpeg, read_image
from edgelens.i18n import current_language, tr
from edgelens.metrics.quality import measure


class ExportCancelled(Exception):
    """取消時不發布不完整的 ZIP。"""


def export_zip(project: Project, format: str = "folders", *, use_display_names: bool = False,
               include_processed: bool = False, include_labels: bool = True, include_metadata: bool = True,
               max_edge: int | None = None, progress=None, cancel=None, splits: SplitResult | None = None,
               filename: str | None = None, lang: str | None = None) -> Path:
    """progress(done, total) 按影像數回報；cancel 可為回呼或 threading.Event。"""
    if format not in ("folders", "split_folders"):
        raise ValueError("匯出格式不正確，請選 folders 或 split_folders。")
    if max_edge is not None and (not isinstance(max_edge, int) or max_edge < 1):
        raise ValueError("縮圖最大邊長必須是正整數，請重新設定。")

    def check_cancel():
        if cancel is not None and (cancel() if callable(cancel) else cancel.is_set()):
            raise ExportCancelled(tr("dataset.export.cancelled"))

    check_cancel()
    records = DatasetWriter(project).list_samples()
    labels, used = {}, set()
    for cls in project.classes:
        name = safe_filename(cls.display(lang or current_language()) if use_display_names else cls.id)
        candidate, i = name, 1
        while candidate.casefold() in used:
            candidate = f"{name}_{cls.id}_{i}"
            i += 1
        labels[cls.id] = candidate
        used.add(candidate.casefold())
    if any(r.class_id not in labels for r in records):
        raise ValueError("有照片的類別已從專案移除，請重新標記後再匯出。")
    assignments = {}
    if format == "split_folders":
        if splits is None:
            splits = load_splits(project) if (project.dataset_dir / "splits.json").exists() else split_dataset(project)
        for part in PARTS:
            for file in getattr(splits, part):
                if file in assignments:
                    raise ValueError("切分清單重複，請重新切分資料集。")
                assignments[file] = part
        if set(assignments) != {r.file for r in records}:
            raise ValueError("照片已變更，請重新切分資料集後再匯出。")
    jobs, metadata = [], []
    for record in records:
        prefix = (assignments[record.file] + "/" if assignments else "") + labels[record.class_id]
        raw_name = f"{prefix}/{Path(record.file).name}"
        data = record.to_dict()
        data["file"], data["processed_file"] = raw_name, None
        metadata.append(data)
        jobs.append((record.file, raw_name, data, True))
        if include_processed and record.processed_file:
            proc_name = f"{prefix}/{Path(record.processed_file).name}"
            data["processed_file"] = proc_name
            jobs.append((record.processed_file, proc_name, data, False))
    if len({job[1] for job in jobs}) != len(jobs):
        raise ValueError("匯出檔名重複，請先重新命名照片。")
    name = safe_filename(filename or f"dataset_{datetime.now():%Y%m%d_%H%M%S}.zip")
    if not name.lower().endswith(".zip"):
        name += ".zip"
    target = unique_path(project.dir("exports") / name)
    # 直接寫到磁碟上的暫存檔（資料集可能有幾百 MB，不能整個放在記憶體）；
    # 完成後才改名成正式檔名，取消或出錯時刪掉暫存檔，不留下不完整的 ZIP。
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_export_", suffix=".zip", dir=str(target.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    if progress:
        progress(0, len(jobs))
    try:
        _write_archive(tmp, format, labels, jobs, project, max_edge, metadata, include_labels, include_metadata,
                       progress, check_cancel)
        check_cancel()
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def _write_archive(tmp, format, labels, jobs, project, max_edge, metadata, include_labels, include_metadata,
                   progress, check_cancel) -> None:
    # JPEG／PNG 本身已壓縮，再用 deflate 只會浪費時間，所以影像用 ZIP_STORED，文字檔才壓縮。
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as archive:
        # 即使類別沒有照片，也保留資料夾供教學平台顯示。
        for prefix in (PARTS if format == "split_folders" else ("",)):
            for label in labels.values():
                archive.writestr(f"{prefix + '/' if prefix else ''}{label}/", b"")
        for done, (source, name, meta, original) in enumerate(jobs, 1):
            check_cancel()
            path = dataset_path(project.dataset_dir, source)
            payload = path.read_bytes()
            if max_edge is not None:
                image = read_image(path)
                h, w = image.shape[:2]
                if max(h, w) > max_edge:
                    image = cv2.resize(image, (max(1, round(w * max_edge / max(h, w))),
                                              max(1, round(h * max_edge / max(h, w)))), interpolation=cv2.INTER_AREA)
                    suffix = Path(name).suffix.lower()
                    if suffix in (".jpg", ".jpeg"):
                        payload = encode_jpeg(image)
                    else:
                        ok, encoded = cv2.imencode(suffix, image)
                        if not ok:
                            raise ValueError("縮圖無法編碼，請取消縮圖或換成 JPEG。")
                        payload = encoded.tobytes()
                    if original:
                        meta.update(width=image.shape[1], height=image.shape[0],
                                    sha1=hashlib.sha1(payload).hexdigest(), dhash=f"{dhash(image):016x}",
                                    quality=measure(image).to_dict())
                        meta["export_original_file"] = source
                        meta["export_original_result"] = meta["result"]
                        meta["result"] = {}
            archive.writestr(name, payload)
            if progress:
                progress(done, len(jobs))
        check_cancel()
        if include_labels:
            archive.writestr("labels.txt", "\n".join(labels.values()) + ("\n" if labels else ""),
                             compress_type=zipfile.ZIP_DEFLATED)
        if include_metadata:
            archive.writestr("metadata.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in metadata),
                             compress_type=zipfile.ZIP_DEFLATED)
