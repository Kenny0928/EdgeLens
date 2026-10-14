"""DatasetWriter 的解析快取：不重讀、外部改動會被發現、回傳值不污染快取。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import edgelens.dataset.writer as wmod
from edgelens.core.project import Project
from edgelens.dataset.writer import DatasetWriter


@pytest.fixture
def writer(tmp_path):
    project = Project(tmp_path / "專案", "專題")
    project.add_class("紅蘋果", "apple")
    project.add_class("香蕉", "banana")
    return DatasetWriter(project)


@pytest.fixture
def parses(monkeypatch):
    """計算 writer 呼叫 json.loads 的次數。"""
    calls = []
    real = json.loads
    monkeypatch.setattr(wmod.json, "loads", lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


def files(w, **kw):
    return sorted(r.file for r in w.list_samples(**kw))


def disk_truth(w):
    """另建一個沒有快取的 writer，當作檔案真實內容。"""
    return files(DatasetWriter(w.project), include_deleted=True)


def test_second_list_does_not_parse(writer, bgr_image, make_frame, parses):
    writer.save(make_frame(bgr_image, seq=1), "apple")
    writer.list_samples()
    parses.clear()
    writer.list_samples()
    writer.list_samples(include_deleted=True)
    assert parses == []


def test_mutations_keep_cache_without_reparse(writer, bgr_image, make_frame, parses):
    writer.list_samples()   # 檔案不存在 → 空快取
    parses.clear()
    a = writer.save(make_frame(bgr_image, seq=1), "apple", image=bgr_image)
    b = writer.save(make_frame(bgr_image, seq=2), "apple")
    assert len(writer.list_samples()) == 2
    r = writer.relabel(a.file, "banana")
    assert [x.class_id for x in writer.list_samples("banana")] == ["banana"]
    d = writer.delete(b.file)
    assert files(writer) == [r.file]
    writer.restore(d.file)
    assert len(writer.list_samples()) == 2
    writer.delete(r.file)
    assert writer.purge_trash() == 1
    assert len(writer.list_samples(include_deleted=True)) == 1
    assert len(parses) == 2   # 只有兩次 save 各解析自己那一行，沒有整份重讀
    assert files(writer, include_deleted=True) == disk_truth(writer)


def test_external_append_and_rewrite_are_noticed(writer, bgr_image, make_frame):
    a = writer.save(make_frame(bgr_image, seq=1), "apple")
    assert len(writer.list_samples()) == 1
    extra = {**a.to_dict(), "file": "apple/external.jpg"}
    with open(writer.metadata_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(extra) + "\n")
    assert len(writer.list_samples()) == 2
    writer.metadata_path.write_text("")
    assert writer.list_samples() == []
    writer.metadata_path.unlink()
    assert writer.list_samples() == []


def test_save_after_external_change_invalidates(writer, bgr_image, make_frame):
    a = writer.save(make_frame(bgr_image, seq=1), "apple")
    writer.list_samples()
    with open(writer.metadata_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({**a.to_dict(), "file": "apple/external.jpg"}) + "\n")
    writer.save(make_frame(bgr_image, seq=2), "apple")   # 簽章與快取不符 → 不能把快取當成最新
    assert len(writer.list_samples()) == 3


def test_mutating_returned_records_does_not_leak(writer, bgr_image, make_frame):
    writer.save(make_frame(bgr_image, seq=1), "apple")
    first = writer.list_samples()[0]
    first.class_id, first.deleted, first.note = "banana", True, "改過"
    again = writer.list_samples()[0]
    assert (again.class_id, again.deleted, again.note) == ("apple", False, "")
    saved = writer.save(make_frame(bgr_image, seq=2), "apple")
    saved.note = "改過"
    assert all(r.note == "" for r in writer.list_samples())


def test_failed_move_leaves_cache_consistent(writer, bgr_image, make_frame, monkeypatch):
    a = writer.save(make_frame(bgr_image, seq=1), "apple", image=bgr_image)
    writer.list_samples()
    real = Path.rename
    count = []

    def flaky(self, target):
        count.append(1)
        if len(count) == 2:   # 第二個檔（_proc）搬移失敗
            raise OSError("boom")
        return real(self, target)

    monkeypatch.setattr(Path, "rename", flaky)
    with pytest.raises(OSError):
        writer.relabel(a.file, "banana")
    monkeypatch.setattr(Path, "rename", real)
    assert [r.class_id for r in writer.list_samples()] == ["apple"]
    assert files(writer, include_deleted=True) == disk_truth(writer)
    assert writer.list_samples()[0].path.exists()


def test_same_size_same_mtime_rewrite_is_detected(writer, bgr_image, make_frame):
    """隨身碟（FAT／exFAT）時間精度粗：外部原子改寫成同樣大小、時間又相同時，靠 inode 認出來。"""
    import os

    r = writer.save(make_frame(bgr_image, seq=1), "apple")
    assert files(writer) == [r.file]
    path = writer.metadata_path
    st = path.stat()
    old = path.read_text(encoding="utf-8")
    new = old.replace(r.file, r.file.replace("apple/", "pearr/"), 1)   # 同長度、不同內容
    assert len(new.encode()) == len(old.encode()) and new != old
    tmp = path.with_name("metadata.tmp")
    tmp.write_text(new, encoding="utf-8")
    os.replace(tmp, path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert path.stat().st_size == st.st_size and path.stat().st_mtime_ns == st.st_mtime_ns
    assert files(writer) == [r.file.replace("apple/", "pearr/")]
