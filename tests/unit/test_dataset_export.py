"""ZIP 的中文名稱、縮圖、metadata 與取消行為。"""
from __future__ import annotations

import hashlib
import json
import zipfile

import cv2
import numpy as np
import pytest

from edgelens.core.project import Project
from edgelens.dataset.export import ExportCancelled, export_zip
from edgelens.dataset.split import split_dataset
from edgelens.dataset.writer import DatasetWriter


@pytest.fixture
def dataset(tmp_path, bgr_image, make_frame):
    project = Project(tmp_path / "中文專案", "資料集")
    project.add_class("紅蘋果", "apple")
    project.add_class("香蕉", "banana")
    writer = DatasetWriter(project)
    for cid in project.class_ids():
        for i in range(10):
            writer.save(make_frame(bgr_image, seq=i), cid, image=bgr_image)
    return project


def test_folders_chinese_and_originals(dataset):
    progress = []
    path = export_zip(dataset, use_display_names=True, include_processed=True, lang="zh_TW",
                      progress=lambda d, t: progress.append((d, t)))
    assert path.parent == dataset.root / "exports"
    with zipfile.ZipFile(path) as z:
        photos = [i for i in z.infolist() if i.filename.endswith(".jpg")]
        assert len(photos) == 40
        assert {i.filename.split("/")[0] for i in photos} == {"紅蘋果", "香蕉"}
        assert all(i.flag_bits & 0x800 for i in photos)
        assert z.read("labels.txt").decode("utf-8") == "紅蘋果\n香蕉\n"
        meta = [json.loads(line) for line in z.read("metadata.jsonl").splitlines()]
        assert len(meta) == 20
        for r in meta:
            assert hashlib.sha1(z.read(r["file"])).hexdigest() == r["sha1"]
            assert r["processed_file"] in z.namelist()
    assert progress[0] == (0, 40) and progress[-1] == (40, 40)


def test_split_thumbnail(dataset):
    split_dataset(dataset)
    path = export_zip(dataset, "split_folders", max_edge=20)
    with zipfile.ZipFile(path) as z:
        meta = [json.loads(line) for line in z.read("metadata.jsonl").splitlines()]
        assert len(meta) == 20
        assert {r["file"].split("/")[0] for r in meta} == {"train", "val", "test"}
        for r in meta:
            raw = z.read(r["file"])
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            assert max(image.shape[:2]) == 20
            assert (r["width"], r["height"]) == (20, 15)
            assert r["sha1"] == hashlib.sha1(raw).hexdigest()
            assert r["processed_file"] is None


def test_options_and_duplicate_display_names(dataset):
    dataset.classes[1].names = dataset.classes[0].names.copy()
    path = export_zip(dataset, use_display_names=True, include_labels=False, include_metadata=False)
    with zipfile.ZipFile(path) as z:
        assert len([n for n in z.namelist() if n.endswith(".jpg")]) == 20
        assert len({n.split("/")[0] for n in z.namelist()}) == 2
        assert "labels.txt" not in z.namelist() and "metadata.jsonl" not in z.namelist()


def test_cancel_has_no_partial_zip(dataset):
    seen = []
    with pytest.raises(ExportCancelled):
        export_zip(dataset, progress=lambda d, t: seen.append(d), cancel=lambda: bool(seen and seen[-1] >= 2))
    assert not list(dataset.dir("exports").glob("*.zip"))


def test_stale_splits(dataset, bgr_image, make_frame):
    split_dataset(dataset)
    DatasetWriter(dataset).save(make_frame(bgr_image), "apple")
    with pytest.raises(ValueError, match="重新切分"):
        export_zip(dataset, "split_folders")


@pytest.mark.parametrize("options", [{"format": "bad"}, {"max_edge": 0}, {"max_edge": 1.5}])
def test_invalid_options(dataset, options):
    with pytest.raises(ValueError):
        export_zip(dataset, **options)
