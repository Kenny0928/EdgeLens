"""Recorder：用合成畫面錄影，檢查影片、CSV、佇列滿時丟畫面、中文路徑。"""

from __future__ import annotations

import csv
import time
import types

import cv2
import numpy as np
import pytest

from edgelens.core.frame import Frame
from edgelens.core.results import Classification, InferenceResult
from edgelens.gui import recorder as R
from edgelens.processing.pipeline import ProcessResult


def make_packet(seq: int, size=(160, 120), data=None, meta=None):
    img = np.full((size[1], size[0], 3), (seq * 7) % 255, np.uint8)
    cv2.circle(img, (20 + seq % 50, 40), 15, (0, 255, 0), -1)
    frame = Frame(image=img, seq=seq, t_cap=time.monotonic(), source_id="t:0", meta=meta or {})
    result = ProcessResult(image=img, overlays=[], data=data if data is not None else {"blobs": 2},
                           timings_ms=[], total_ms=1.5)
    return types.SimpleNamespace(frame=frame, result=result,
                                 stats={"fps": 25.0, "latency_ms": 12.0, "process_ms": 1.5, "dropped": 0.0})


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_record_two_seconds(tmp_path):
    rec = R.Recorder(tmp_path / "recordings", "synthetic:box", mode="processed", fps=20)
    rec.start()
    n = 40
    for i in range(n):
        if i == 10:
            rec.add_inference(30, InferenceResult(kind="classify", model_name="m", frame_seq=30,
                                                  top_k=[Classification(1, 0.9, "cat")], timings_ms={"total": 4.0}))
        assert rec.add_packet(make_packet(i, meta={"fps": 19.5, "us": {"cap": 1000}, "mem": 5}))
        time.sleep(0.05)
    assert rec.size_bytes >= 0 and rec.elapsed > 1.5
    res = rec.stop()
    assert res.error == "" and res.frames == n and res.dropped == 0
    assert res.video_path is not None and res.video_path.is_file() and res.video_path.stat().st_size > 0
    assert res.video_path.parent == tmp_path / "recordings"
    cap = cv2.VideoCapture(str(res.video_path))
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    assert n - 3 <= count <= n + 3
    rows = read_csv(res.csv_path)
    assert len(rows) == n
    for col in ("t_s", "seq", "fps", "latency_ms", "process_ms", "data.blobs", "board.fps", "board.us.cap_ms",
                "infer_label", "infer_score"):
        assert col in rows[0]
    assert rows[0]["data.blobs"] == "2" and rows[-1]["infer_label"] == "cat" and rows[-1]["infer_seq"] == "30"
    assert res.csv_path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert R.Recorder.stop(rec) is res   # 可重複呼叫


def test_modes_and_gray(tmp_path):
    rec = R.Recorder(tmp_path, "cam", mode="raw", fps=10)
    rec.start()
    p = make_packet(1)
    p.frame = p.frame.replace(image=cv2.cvtColor(p.frame.image, cv2.COLOR_BGR2GRAY))
    for _ in range(5):
        rec.add_packet(p)
    res = rec.stop()
    assert res.video_path is not None and res.frames == 5
    with pytest.raises(ValueError):
        R.Recorder(tmp_path, "cam", mode="bogus")


def test_data_mode_has_no_video(tmp_path):
    rec = R.Recorder(tmp_path / "rec", "cam", mode="data", csv_dir=tmp_path / "logs")
    rec.start()
    for i in range(10):
        rec.add_packet(make_packet(i))
    res = rec.stop()
    assert res.video_path is None and res.csv_path.parent == tmp_path / "logs"
    assert len(read_csv(res.csv_path)) == 10


def test_full_queue_drops_without_blocking(tmp_path, monkeypatch):
    rec = R.Recorder(tmp_path, "cam", mode="processed", max_queue=3)
    orig = R.Recorder._write

    def slow(self, *a):
        time.sleep(0.05)
        return orig(self, *a)

    monkeypatch.setattr(R.Recorder, "_write", slow)
    rec.start()
    t0 = time.monotonic()
    accepted = sum(rec.add_packet(make_packet(i)) for i in range(60))
    assert time.monotonic() - t0 < 1.0     # 生產端沒有被寫入速度拖住
    assert rec.dropped > 0 and accepted + rec.dropped == 60
    res = rec.stop()
    assert res.frames == accepted and res.dropped == rec.dropped
    assert len(read_csv(res.csv_path)) == accepted


def test_chinese_path(tmp_path):
    out = tmp_path / "專案 一" / "recordings"
    rec = R.Recorder(out, "相機/一號", mode="processed", fps=10)
    rec.start()
    for i in range(8):
        rec.add_packet(make_packet(i))
    res = rec.stop()
    assert res.video_path is not None and res.video_path.parent == out and "相機" in res.video_path.name
    cap = cv2.VideoCapture(str(res.video_path))
    ok, _ = cap.read()
    cap.release()
    assert ok
    assert len(read_csv(res.csv_path)) == 8


def test_add_after_stop_is_ignored(tmp_path):
    rec = R.Recorder(tmp_path, "cam", mode="data")
    assert rec.add_packet(make_packet(1)) is False   # 還沒開始
    rec.start()
    rec.stop()
    assert rec.add_packet(make_packet(1)) is False


def test_helpers():
    assert R.flatten_data({"a": 1, "b": [1, 2], "c": "x", "d": True, "e": float("nan")}) == {
        "data.a": 1, "data.b_count": 2, "data.d": 1}
    assert R.flatten_board({"fps": 20.0, "us": {"kpu": 3000}, "lat": 8000})["board.us.kpu_ms"] == 3.0
    assert R.summarize_inference(InferenceResult(kind="detect", model_name="m", frame_seq=1))["infer_count"] == 0
    assert R.ascii_temp_dir().is_dir()
    assert R.fourcc_candidates()[-1] == (".avi", "MJPG")


def test_qimage_to_bgr():
    from PySide6.QtGui import QColor, QImage

    q = QImage(8, 4, QImage.Format.Format_RGB32)
    q.fill(QColor(255, 0, 0))
    arr = R.qimage_to_bgr(q)
    assert arr.shape == (4, 8, 3) and tuple(arr[0, 0]) == (0, 0, 255)
    assert R.qimage_to_bgr(QImage()) is None
