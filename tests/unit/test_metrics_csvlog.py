"""BOM、動態欄位與週期寫入不能使舊列欄位錯位。"""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor

import pytest

from edgelens.metrics.csvlog import CsvLogger, write_summary_csv


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_dynamic_columns_and_bom(tmp_path):
    path = tmp_path / "中文.csv"
    with CsvLogger(path, ["name"], flush_interval=0) as log:
        log.add({"name": "第一筆", "fps": 30})
        log.add({"name": "第二筆", "latency": 12})
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert read_rows(path) == [{"name": "第一筆", "fps": "30", "latency": ""},
                               {"name": "第二筆", "fps": "", "latency": "12"}]
    log.close()
    with pytest.raises(ValueError):
        log.add({"name": "已關閉"})


def test_flush_interval_and_summary(tmp_path, monkeypatch):
    from edgelens.metrics import csvlog
    clock = [0]
    monkeypatch.setattr(csvlog.time, "monotonic", lambda: clock[0])
    log = CsvLogger(tmp_path / "週期.csv", ["x"], flush_interval=2)
    log.add({"x": 1})
    assert not read_rows(log.path)
    clock[0] = 2
    log.add({"x": 2})
    assert len(read_rows(log.path)) == 2
    log.close()
    path = write_summary_csv(tmp_path / "摘要.csv", [{"模型": "甲", "fps": 10}, {"模型": "乙", "p95": 3}])
    assert read_rows(path)[1] == {"模型": "乙", "fps": "", "p95": "3"}


def test_threaded_and_invalid(tmp_path):
    with CsvLogger(tmp_path / "threads.csv") as log:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda i: log.add({"i": i}), range(40)))
    assert {int(r["i"]) for r in read_rows(log.path)} == set(range(40))
    with pytest.raises(ValueError):
        CsvLogger(tmp_path / "bad.csv", flush_interval=-1)


def test_append_keeps_rows_and_expands_header_once(tmp_path):
    import csv

    from edgelens.metrics.csvlog import CsvLogger

    path = tmp_path / "錄影.csv"
    with CsvLogger(path, ["t", "fps"], flush_interval=0) as log:
        for i in range(50):
            log.add({"t": i, "fps": 30})
        log.add({"t": 50, "fps": 29, "objs": 2})
        log.add({"t": 51, "fps": 28})
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 52 and rows[0]["objs"] == "" and rows[50]["objs"] == "2"
    assert path.read_bytes().startswith(b"\xef\xbb\xbf") and path.read_bytes().count(b"\xef\xbb\xbf") == 1
