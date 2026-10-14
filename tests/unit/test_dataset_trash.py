"""from_dict 不碰檔案系統、存取檔案仍擋越界，以及垃圾桶清除。不依賴時間量測。"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from edgelens.core.project import Project
from edgelens.dataset.writer import TRASH_KEEP_DAYS, DatasetWriter, SampleRecord

DAY = 86400


@pytest.fixture
def writer(tmp_path):
    project = Project(tmp_path / "專案", "專題")
    project.add_class("紅蘋果", "apple")
    project.add_class("香蕉", "banana")
    return DatasetWriter(project)


def _data(file="apple/a.jpg", **kw):
    return {"file": file, "class_id": "apple", **kw}


def test_from_dict_does_not_resolve(tmp_path, monkeypatch):
    calls = []
    real = Path.resolve
    monkeypatch.setattr(Path, "resolve", lambda self, *a, **k: calls.append(self) or real(self, *a, **k))
    SampleRecord.from_dict(_data(processed_file="apple/a_proc.jpg", original_file="apple/a.jpg"), tmp_path)
    assert calls == []


@pytest.mark.parametrize("bad", ["", None, 3, "/etc/passwd", "../x.jpg", "a/../../x.jpg", "a\\b.jpg",
                                 "C:/x.jpg", "a/./b.jpg"])
def test_from_dict_rejects_lexically_bad_paths(tmp_path, bad):
    with pytest.raises(ValueError):
        SampleRecord.from_dict(_data(file=bad), tmp_path)
    with pytest.raises(ValueError):
        SampleRecord.from_dict(_data(processed_file=bad if bad is not None else ""), tmp_path)


@pytest.mark.skipif(os.name == "nt", reason="Windows 建立符號連結需要權限")
def test_record_path_rejects_symlink_escape(writer, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.jpg").write_bytes(b"x")
    (writer.dataset_dir / "apple").symlink_to(outside, target_is_directory=True)
    rec = SampleRecord.from_dict(_data(file="apple/secret.jpg"), writer.dataset_dir)   # 解析時不檢查
    with pytest.raises(ValueError):
        _ = rec.path
    with pytest.raises(ValueError):
        writer.delete("apple/secret.jpg")
    assert (outside / "secret.jpg").exists()


def _delete_one(writer, bgr_image, make_frame, cid="apple", seq=1, proc=True):
    r = writer.save(make_frame(bgr_image, seq=seq, t_wall=1700000000 + seq), cid,
                    image=bgr_image if proc else None)
    return writer.delete(r.file)


def test_delete_sets_deleted_at_and_restore_clears(writer, bgr_image, make_frame):
    before = time.time()
    d = _delete_one(writer, bgr_image, make_frame)
    assert before <= d.deleted_at <= time.time()
    assert writer.list_samples(include_deleted=True)[0].deleted_at == d.deleted_at
    assert writer.restore(d.file).deleted_at is None


def test_purge_all_removes_files_dirs_and_records(writer, bgr_image, make_frame):
    live = writer.save(make_frame(bgr_image, seq=9), "banana", image=bgr_image)
    d1 = _delete_one(writer, bgr_image, make_frame, seq=1)
    d2 = _delete_one(writer, bgr_image, make_frame, seq=2, proc=False)
    paths = [writer.dataset_dir / x for x in (d1.file, d1.processed_file, d2.file)]
    assert all(p.exists() for p in paths)
    assert writer.purge_trash() == 2
    assert not any(p.exists() for p in paths)
    assert not list((writer.dataset_dir / "_trash").iterdir())
    assert [r.file for r in writer.list_samples(include_deleted=True)] == [live.file]
    assert live.path.exists() and (writer.dataset_dir / live.processed_file).exists()
    assert writer.purge_trash() == 0


def test_purge_by_age(writer, bgr_image, make_frame):
    old = _delete_one(writer, bgr_image, make_frame, seq=1)
    new = _delete_one(writer, bgr_image, make_frame, seq=2)
    records = writer._load()
    for r in records:
        if r.file == old.file:
            r.deleted_at = time.time() - (TRASH_KEEP_DAYS + 1) * DAY
    writer._write(records)
    assert writer.purge_trash(TRASH_KEEP_DAYS) == 1
    left = writer.list_samples(include_deleted=True)
    assert [r.file for r in left] == [new.file] and (writer.dataset_dir / new.file).exists()
    assert not (writer.dataset_dir / old.file).exists()


def test_purge_legacy_record_uses_trash_dir_mtime(writer, bgr_image, make_frame):
    old = _delete_one(writer, bgr_image, make_frame, seq=1)
    new = _delete_one(writer, bgr_image, make_frame, seq=2)
    records = writer._load()
    for r in records:
        r.deleted_at = None   # 模擬舊版 metadata
    writer._write(records)
    assert "deleted_at" not in "".join(json.dumps({k: v for k, v in r.to_dict().items() if v is not None})
                                       for r in records)
    trash_dir = (writer.dataset_dir / old.file).parents[len(Path(old.file).parts) - 3]
    assert trash_dir.parent.name == "_trash"
    stamp = time.time() - (TRASH_KEEP_DAYS + 5) * DAY
    os.utime(trash_dir, (stamp, stamp))
    assert writer.purge_trash(TRASH_KEEP_DAYS) == 1
    assert [r.file for r in writer.list_samples(include_deleted=True)] == [new.file]


def test_purge_orphans(writer, bgr_image, make_frame):
    trash = writer.dataset_dir / "_trash"
    old_dir, new_dir = trash / "olddir", trash / "newdir"
    for d in (old_dir, new_dir):
        (d / "apple").mkdir(parents=True)
        (d / "apple" / "x.jpg").write_bytes(b"x")
    (trash / "stray.txt").write_text("x")
    stamp = time.time() - (TRASH_KEEP_DAYS + 1) * DAY
    os.utime(old_dir, (stamp, stamp))
    os.utime(trash / "stray.txt", (stamp, stamp))
    assert writer.purge_trash(TRASH_KEEP_DAYS) == 0
    assert not old_dir.exists() and not (trash / "stray.txt").exists() and new_dir.exists()
    writer.purge_trash(None)
    assert not new_dir.exists()


def test_purge_never_touches_outside_trash(writer, bgr_image, make_frame, tmp_path):
    live = writer.save(make_frame(bgr_image, seq=1), "apple", image=bgr_image)
    victim = tmp_path / "victim.jpg"
    victim.write_bytes(b"keep")
    (writer.dataset_dir / "notes.txt").write_text("keep")
    # 偽造的紀錄：deleted 但路徑指到非 _trash 的位置，或用 .. 逃出
    forged = [
        {**live.to_dict(), "file": "banana/zzz.jpg", "deleted": True, "original_file": "banana/zzz.jpg"},
        {**live.to_dict(), "file": "_trash/u1/../../apple/" + Path(live.file).name, "deleted": True},
    ]
    with open(writer.metadata_path, "a", encoding="utf-8") as f:
        for d in forged:
            f.write(json.dumps(d) + "\n")
    writer.purge_trash()
    assert live.path.exists() and victim.exists() and (writer.dataset_dir / "notes.txt").exists()


@pytest.mark.skipif(os.name == "nt", reason="Windows 建立符號連結需要權限")
def test_purge_skips_symlink_escaping_trash(writer, bgr_image, make_frame, tmp_path):
    d = _delete_one(writer, bgr_image, make_frame)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.jpg").write_bytes(b"keep")
    real = writer.dataset_dir / d.file
    real_dir = real.parent
    for item in real_dir.iterdir():
        item.unlink()
    real_dir.rmdir()
    real_dir.symlink_to(outside, target_is_directory=True)
    (outside / real.name).write_bytes(b"keep")
    writer.purge_trash()
    assert (outside / "secret.jpg").exists() and (outside / real.name).exists()


@pytest.mark.skipif(os.name == "nt", reason="Windows 建立符號連結需要權限")
@pytest.mark.parametrize("days", [None, TRASH_KEEP_DAYS])
def test_purge_refuses_symlinked_trash(writer, tmp_path, days):
    """別人做壞的專案資料夾可能把 _trash 換成指向家目錄的連結；清空或開專案時的自動清理都不能跟進去。"""
    victim = tmp_path / "home"
    (victim / "作業").mkdir(parents=True)
    (victim / "作業" / "報告.docx").write_bytes(b"important")
    (victim / "照片.jpg").write_bytes(b"important")
    old = time.time() - 60 * DAY
    for p in (victim / "作業", victim / "作業" / "報告.docx", victim / "照片.jpg"):
        os.utime(p, (old, old))
    writer.dataset_dir.mkdir(parents=True, exist_ok=True)
    (writer.dataset_dir / "_trash").symlink_to(victim, target_is_directory=True)
    assert writer.purge_trash(days) == 0
    assert (victim / "作業" / "報告.docx").read_bytes() == b"important"
    assert (victim / "照片.jpg").read_bytes() == b"important"
