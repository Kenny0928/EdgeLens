"""device 後端：不在電腦上跑，runner 套用板子算好的結果。"""

from __future__ import annotations

import time

import numpy as np

from edgelens.core.frame import Frame
from edgelens.core.modelcard import ModelCard
from edgelens.core.registry import discover
from edgelens.core.results import Classification, Detection, InferenceResult
from edgelens.inference.backends.device import DeviceBackend
from edgelens.inference.runner import InferenceRunner

discover()


def _card(tmp_path, labels=None):
    card = ModelCard(name="k", file=str(tmp_path / "face.kmodel"), labels=labels or [])
    return card


def _frame(dr=None):
    return Frame(np.zeros((48, 64, 3), np.uint8), 7, time.monotonic(), "k210:0", device_result=dr)


def test_attributes_and_can_load(tmp_path):
    assert DeviceBackend.available() == (True, "")
    assert DeviceBackend.can_load(_card(tmp_path))      # 檔案不存在也可以
    assert DeviceBackend.input_mode == "device"


def test_detect_threshold_and_labels(tmp_path):
    dr = InferenceResult(kind="detect", model_name="board", frame_seq=0, where="device",
                         detections=[Detection((1, 2, 30, 40), 0.9, 1, "x"), Detection((5, 5, 9, 9), 0.1, 0)],
                         timings_ms={"infer": 55.0}, image_size=(64, 48))
    r = InferenceRunner(_card(tmp_path, ["a", "b"]))
    r.load()
    assert r.info.mode == "device"
    res = r.run(_frame(dr))
    assert res.where == "device" and res.frame_seq == 7 and len(res.detections) == 1
    assert res.detections[0].label == "b" and res.timings_ms["infer"] == 55.0 and "total" in res.timings_ms
    r.conf_threshold = 0.05
    assert len(r.run(_frame(dr)).detections) == 2


def test_classify_and_missing_result(tmp_path):
    dr = InferenceResult(kind="classify", model_name="board", frame_seq=0, where="device",
                         top_k=[Classification(0, 0.8, "")])
    r = InferenceRunner(_card(tmp_path))
    r.load()
    res = r.run(_frame(dr))
    assert res.kind == "classify" and res.top_k[0].label == "0"
    empty = r.run(_frame(None))
    assert empty.where == "device" and not empty.has_hit and empty.extra.get("device_missing")
