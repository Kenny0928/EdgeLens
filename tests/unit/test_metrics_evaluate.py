"""用已知答案檢查混淆矩陣，包含未知標籤與取消。"""
from __future__ import annotations

import csv

import pytest

from edgelens.core.project import Project
from edgelens.core.results import Classification, InferenceResult
from edgelens.dataset.writer import DatasetWriter
from edgelens.metrics.evaluate import evaluate


@pytest.fixture
def records(tmp_path, bgr_image, make_frame):
    project = Project(tmp_path / "中文評估", "評估")
    project.add_class("蘋果", "a")
    project.add_class("香蕉", "b")
    writer = DatasetWriter(project)
    return [writer.save(make_frame(bgr_image, seq=i), cid) for i, cid in enumerate(("a", "a", "b", "b"))]


def prediction(label, cid=0):
    return InferenceResult("classify", "模型", 1, top_k=[Classification(cid, .9, label)])


@pytest.mark.parametrize("mapping", [{"a": "蘋果", "b": "香蕉"}, {"蘋果": "a", "香蕉": "b"}])
def test_known_confusion(records, tmp_path, mapping):
    labels = iter(("蘋果", "香蕉", "香蕉", "香蕉"))
    seen = []
    report = evaluate(records, lambda image: prediction(next(labels)), ["a", "b"],
                      label_mapping=mapping, progress=lambda d, t: seen.append((d, t)))
    assert report.accuracy == .75 and report.confusion_matrix == [[1, 1], [0, 2]]
    assert report.per_class["a"]["precision"] == 1 and report.per_class["a"]["recall"] == .5
    assert report.per_class["b"]["precision"] == pytest.approx(2/3)
    assert report.per_class["b"]["f1"] == .8
    assert report.wrong_files == [records[1].file]
    assert report.latency["n"] == 4 and report.latency["max"] >= 0
    assert seen[0] == (0, 4) and seen[-1] == (4, 4)
    path = report.to_csv(tmp_path / "中文報告.csv")
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len([r for r in rows if r["section"] == "confusion"]) == 4


def test_unknown_empty_prediction_and_index(records):
    outputs = iter([prediction("unknown"), InferenceResult("classify", "m", 1),
                    prediction("", 1), prediction("a")])
    report = evaluate(records, lambda image: next(outputs), ["a", "b"])
    assert report.prediction_labels == ["a", "b", "__unknown__"]
    assert report.confusion_matrix == [[0, 0, 2], [1, 1, 0]] and report.accuracy == .25
    assert report.per_class["a"]["support"] == 2 and report.per_class["a"]["recall"] == 0


def test_cancel_and_empty(records):
    completed = []
    report = evaluate(records, lambda image: prediction("a"), ["a", "b"],
                      progress=lambda d, t: completed.append(d), cancel=lambda: completed[-1] >= 1)
    assert report.cancelled and report.total == 1 and report.requested == 4
    assert evaluate([], lambda image: None, ["a"]).total == 0


def test_invalid_and_missing(records):
    with pytest.raises(ValueError):
        evaluate(records, lambda image: prediction("a"), ["a"])
    with pytest.raises(ValueError):
        evaluate(records, lambda image: InferenceResult("detect", "m", 1), ["a", "b"])
    with pytest.raises(ValueError):
        evaluate(records, lambda image: prediction("a"), ["a", "b"], label_mapping={"a": "b"})
    records[0].path.unlink()
    with pytest.raises(FileNotFoundError):
        evaluate(records, lambda image: prediction("a"), ["a", "b"])
