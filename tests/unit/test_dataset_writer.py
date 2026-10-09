"""以中文路徑與失敗注入驗證照片和紀錄能一起維護。"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import pytest

from edgelens.core.paths import atomic_write_bytes
from edgelens.core.project import Project
from edgelens.core.results import Classification, InferenceResult
from edgelens.dataset.writer import DatasetWriter, read_image
from edgelens.processing.pipeline import Pipeline
from edgelens.sources.base import SourceInfo


@pytest.fixture
def writer(tmp_path):
    project = Project(tmp_path / "中文專案", "我的專題")
    project.add_class("紅蘋果", "apple")
    project.add_class("香蕉", "banana")
    return DatasetWriter(project)


def test_original_jpeg_metadata(writer, bgr_image, make_frame):
    raw = cv2.imencode(".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    frame = make_frame(bgr_image, jpeg=raw, t_wall=1700000000.123)
    result = InferenceResult("classify", "模型", 1, top_k=[Classification(0, .9, "紅蘋果")])
    source = SourceInfo("synthetic", "test:0", "測試來源")
    r = writer.save(frame, "apple", image=255-bgr_image, source=source, result=result,
                    pipeline=Pipeline(), controls={"亮度": 1}, group_id="組一", burst_id="連拍一", note="備註")
    assert r.path.read_bytes() == raw
    assert r.sha1 == hashlib.sha1(raw).hexdigest()
    assert r.file.startswith("apple/" + datetime.fromtimestamp(frame.t_wall).strftime("%Y%m%d_%H%M%S_123"))
    assert read_image(writer.dataset_dir / r.processed_file).shape == bgr_image.shape
    data = json.loads(writer.metadata_path.read_text())
    assert data["source"] == source.to_dict() and data["result"] == result.to_dict()
    assert data["time"]["epoch"] == frame.t_wall
    assert data["width"] == 64 and data["height"] == 48
    assert data["edgelens_version"] and len(data["dhash"]) == 16
    assert writer.list_samples("apple")[0].note == "備註"
    assert not writer.list_samples("banana")


def test_relabel_delete_restore_reopen(writer, bgr_image, make_frame):
    r = writer.save(make_frame(bgr_image), "apple", image=bgr_image)
    old = r.path
    moved = writer.relabel(r.file, "banana")
    assert not old.exists() and moved.path.exists() and moved.file.startswith("banana/")
    assert writer.relabel(moved.file, "banana").file == moved.file
    trash = writer.delete(moved.file)
    assert trash.path.exists() and trash.file.startswith("_trash/") and not writer.list_samples()
    reopened = DatasetWriter(writer.project)
    restored = reopened.restore(trash.file)
    assert restored.file == moved.file and restored.path.exists()
    assert (writer.dataset_dir / restored.processed_file).exists()
    assert reopened.list_samples()[0].class_id == "banana"


def test_restore_collision(writer, bgr_image, make_frame):
    frame = make_frame(bgr_image)
    first = writer.save(frame, "apple")
    trash = writer.delete(first.file)
    second = writer.save(frame, "apple")
    restored = writer.restore(trash.file)
    assert first.file == second.file and restored.file != second.file
    assert len(writer.list_samples()) == 2


def test_bad_lines_rebuild_preserves_records(writer, bgr_image, make_frame, caplog):
    original = writer.save(make_frame(bgr_image), "apple", note="保留", image=bgr_image)
    trash = writer.delete(writer.save(make_frame(bgr_image), "banana").file)
    with writer.metadata_path.open("ab") as f:
        f.write(b'\n{bad\n[]\n{"file":"../outside.jpg","class_id":"apple"}\n\xff\n')
    assert len(writer.list_samples()) == 1 and "跳過" in caplog.text
    raw = cv2.imencode(".jpg", bgr_image)[1].tobytes()
    atomic_write_bytes(writer.dataset_dir / "banana/外來照片.jpg", raw)
    atomic_write_bytes(writer.dataset_dir / "banana/破損.jpg", b"broken")
    rebuilt = writer.rebuild_metadata()
    assert len(rebuilt) == 2
    assert next(r for r in rebuilt if r.file == original.file).note == "保留"
    assert next(r for r in rebuilt if "外來" in r.file).sha1 == hashlib.sha1(raw).hexdigest()
    assert len(writer.rebuild_metadata()) == 2
    assert writer.restore(trash.file).path.exists()


def test_concurrent_collision_and_no_processed(writer, bgr_image, make_frame):
    writer.save_processed = False
    frame = make_frame(bgr_image)
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(lambda _: writer.save(frame, "apple", image=bgr_image), range(24)))
    assert len({r.file for r in records}) == 24
    assert len(writer.list_samples()) == 24
    assert all(r.processed_file is None for r in records)
    assert len(writer.metadata_path.read_text().splitlines()) == 24


def test_metadata_failure_rolls_back(writer, bgr_image, make_frame, monkeypatch):
    def fail(line):
        raise OSError("模擬磁碟故障")

    monkeypatch.setattr(writer, "_append", fail)
    with pytest.raises(OSError):
        writer.save(make_frame(bgr_image), "apple", image=bgr_image)
    assert not list(writer.dataset_dir.rglob("*.jpg"))


def test_move_failure_rolls_back(writer, bgr_image, make_frame, monkeypatch):
    r = writer.save(make_frame(bgr_image), "apple", image=bgr_image)

    def fail(records):
        raise OSError("模擬磁碟故障")

    monkeypatch.setattr(writer, "_write", fail)
    with pytest.raises(OSError):
        writer.relabel(r.file, "banana")
    assert r.path.exists() and (writer.dataset_dir / r.processed_file).exists()
    assert writer.list_samples()[0].file == r.file


@pytest.mark.parametrize("file", ["../x.jpg", "/tmp/x.jpg", "apple/../../x.jpg", "C:\\x.jpg"])
def test_path_validation(writer, file):
    with pytest.raises(ValueError):
        writer.delete(file)


def test_unknown_class_missing_image_and_invalid_metadata(writer, bgr_image, make_frame):
    with pytest.raises(ValueError):
        writer.save(make_frame(bgr_image), "unknown")
    with pytest.raises(FileNotFoundError):
        writer.restore("apple/missing.jpg")
    with pytest.raises(TypeError):
        writer.save(make_frame(bgr_image), "apple", controls={"bad": object()})
    assert not writer.list_samples()


def test_metadata_append_survives_truncated_line(writer, bgr_image, make_frame):
    """當機留下半行時，下一筆紀錄不能接在半行後面一起壞掉。"""
    writer.save(make_frame(bgr_image, seq=1), "apple")
    with open(writer.metadata_path, "ab") as f:
        f.write(b'{"file": "apple/broken')
    writer.save(make_frame(bgr_image, seq=2), "apple")
    assert len(writer.list_samples()) == 2


def test_many_saves_append_without_rewriting(writer, bgr_image, make_frame):
    for i in range(30):
        writer.save(make_frame(bgr_image, seq=i), "apple")
    lines = writer.metadata_path.read_bytes().splitlines()
    assert len(lines) == 30 and len(writer.list_samples()) == 30


def test_shared_writer_is_one_per_folder(tmp_path):
    """拍照與瀏覽器必須共用同一個 writer，改類別重寫 metadata 才不會蓋掉剛拍的紀錄。"""
    from edgelens.core.project import Workspace
    from edgelens.dataset.writer import shared_writer

    ws = Workspace(tmp_path / "ws")
    p = ws.create_project("p", template="rock_paper_scissors")
    w1 = shared_writer(p)
    again = ws.open_project(p.id)  # 重新載入的專案物件
    w2 = shared_writer(again)
    assert w1 is w2 and w2.project is again
