"""參數格點、動態結果欄位與取消後的部分紀錄。"""
from __future__ import annotations

import csv

import pytest

from edgelens.metrics.sweep import run_sweep


def test_grid_csv_and_progress(tmp_path):
    progress = []
    report = run_sweep({"size": [10, 20], "mode": ["甲", "乙"]},
                       lambda p: {"fps": p["size"] / 2}, tmp_path / "掃描.csv",
                       progress=lambda d, t: progress.append((d, t)))
    assert len(report.rows) == 4 and report.total == 4 and not report.cancelled
    assert progress == [(i, 4) for i in range(5)]
    with report.path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0] == {"param.size": "10", "param.mode": "甲", "fps": "5.0"}
    assert report.path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_cancel_and_empty_grid(tmp_path):
    calls = []

    def run(params):
        calls.append(params)
        return {"score": 1}

    report = run_sweep({"x": [1, 2, 3]}, run, tmp_path / "partial.csv", cancel=lambda: len(calls) == 1)
    assert report.cancelled and len(report.rows) == 1
    assert len(run_sweep({}, run, tmp_path / "one.csv").rows) == 1


def test_failure_preserves_completed_rows(tmp_path):
    def run(params):
        if params["x"] == 2:
            raise RuntimeError("模型錯誤")
        return {"fps": 1}

    path = tmp_path / "failure.csv"
    with pytest.raises(RuntimeError):
        run_sweep({"x": [1, 2]}, run, path)
    assert "1,1" in path.read_text(encoding="utf-8-sig")


def test_invalid(tmp_path):
    with pytest.raises(ValueError):
        run_sweep({"x": []}, lambda p: {}, tmp_path / "bad.csv")
    with pytest.raises(ValueError):
        run_sweep({}, lambda p: {"fps": float("nan")}, tmp_path / "bad.csv")
