"""Haar 後端與內附 cascade。"""

from __future__ import annotations

import time

import numpy as np
import pytest

from edgelens.core.frame import Frame
from edgelens.core.modelcard import ModelCard
from edgelens.core.registry import discover
from edgelens.inference.backends.haar import builtin_cascades, cascade_display_name
from edgelens.inference.base import InferenceError
from edgelens.inference.runner import InferenceRunner

discover()


def _frame(img):
    return Frame(img, 3, time.monotonic(), "t")


def test_builtin_cascades_listed():
    c = builtin_cascades()
    assert "frontalface_default" in c and "eye" in c and "smile" in c
    assert all(p.suffix == ".xml" and p.is_file() for p in c.values())
    assert cascade_display_name("frontalface_default") != "frontalface_default"
    assert cascade_display_name("no_such_cascade") == "no_such_cascade"


def test_haar_loads_and_blank_image_is_empty():
    card = ModelCard.load(builtin_cascades()["frontalface_default"])
    r = InferenceRunner(card)
    r.load()
    assert r.info.backend == "haar" and r.info.mode == "image"
    res = r.run(_frame(np.full((120, 160, 3), 90, np.uint8)))
    assert res.kind == "detect" and res.detections == [] and res.image_size == (160, 120)
    assert res.frame_seq == 3 and "total" in res.timings_ms
    # 灰階影像也能處理
    assert r.run(_frame(np.full((120, 160), 90, np.uint8))).detections == []


def test_haar_params_from_card():
    card = ModelCard.load(builtin_cascades()["eye"])
    card.decoder_params = {"min_neighbors": 3, "scale_factor": 1.2, "min_size": [20, 20]}
    r = InferenceRunner(card)
    r.load()
    assert "1.2" in r.info.backend_desc and "3" in r.info.backend_desc


def test_haar_bad_xml(tmp_path):
    p = tmp_path / "bad.xml"
    p.write_text("<nothing/>")
    with pytest.raises(InferenceError) as ei:
        InferenceRunner(ModelCard.load(p)).load()
    assert ei.value.hint


def test_haar_non_ascii_path(tmp_path):
    import shutil

    dst = tmp_path / "中文資料夾"
    dst.mkdir()
    p = dst / "臉.xml"
    shutil.copy(builtin_cascades()["frontalface_default"], p)
    r = InferenceRunner(ModelCard.load(p))
    r.load()
    assert r.info.backend == "haar"
