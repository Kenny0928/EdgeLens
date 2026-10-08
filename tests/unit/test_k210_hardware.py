"""K210 實機測試（需要接上 Maixduino／Maix Bit；CI 不執行）。

執行方式（在 repo 根目錄，板子接上 USB，並關掉其他使用序列埠的程式）：

    K210_PORT=/dev/cu.usbserial-XXXX0 python -m pytest -q -m hardware tests/unit/test_k210_hardware.py -s

沒設定 K210_PORT 時用 VID/PID 自動尋找。板上模型測試另外需要：
    K210_MODEL=/path/to/model.kmodel   K210_MODEL_TASK=detect|classify
    K210_MODEL_ANCHORS="1.889,2.5245,..."（偵測用）  K210_MODEL_SIZE=320x240
"""

from __future__ import annotations

import os
import time

import pytest

from edgelens.core.modelcard import InputSpec, ModelCard
from edgelens.sources.base import SourceInfo
from edgelens.sources.k210.source import K210Source

pytestmark = pytest.mark.hardware


def _info() -> SourceInfo:
    port = os.environ.get("K210_PORT", "")
    if not port:
        found = K210Source.discover()
        if not found:
            pytest.skip("找不到 K210 序列埠")
        return found[0]
    return K210Source.info_from_params({"port": port})


@pytest.fixture
def src():
    s = K210Source(_info())
    s.open()
    yield s
    s.close()


def _fps(s: K210Source, seconds: float) -> float:
    n, t0 = 0, time.monotonic()
    while time.monotonic() - t0 < seconds:
        if s.read(timeout=1.0) is not None:
            n += 1
    return n / seconds


def test_stream_30_seconds_fps(src):
    """串流 30 秒量 FPS。和舊 viewer 比較：同一塊板子、同樣 QVGA、lcd 關閉、send_every=1。"""
    fps = _fps(src, 30)
    print(f"FPS = {fps:.1f}  統計 = {src.stats}  板上 fps = {src.last_meta.get('fps')}")
    assert fps > 8
    assert src.stats["bad_frames"] <= 2


def test_live_controls_acked(src):
    for key, value in (("brightness", 1), ("contrast", -1), ("saturation", 2), ("vflip", True), ("vflip", False),
                       ("threshold", 0.6)):
        res = src.set_control(key, value)
        print(key, value, res)
        assert res.ok and res.verified in (True, None)
    assert src.set_control("brightness", 0).verified is True


def test_restart_controls(src):
    res = src.set_control("framesize", "QQVGA")
    assert res.ok and src.read(timeout=3) is not None
    assert src.read(timeout=3).image.shape[:2] == (120, 160)
    assert src.set_control("framesize", "QVGA").ok


def test_open_close_five_times_without_replug():
    for i in range(5):
        s = K210Source(_info())
        s.open()
        assert s.read(timeout=3) is not None, f"第 {i + 1} 次讀不到畫面"
        s.close()


@pytest.mark.skipif(not os.environ.get("K210_MODEL"), reason="需要 K210_MODEL")
def test_upload_and_run_model(src, tmp_path):
    path = os.environ["K210_MODEL"]
    task = os.environ.get("K210_MODEL_TASK", "detect")
    w, h = (int(x) for x in os.environ.get("K210_MODEL_SIZE", "320x240").split("x"))
    device = {"path": "/flash/" + os.path.basename(path), "task": task}
    if task == "detect":
        device["anchors"] = [float(x) for x in os.environ["K210_MODEL_ANCHORS"].split(",")]
    card = ModelCard(name="hw-test", file=path, device=device, input=InputSpec(size=(w, h)),
                     labels=[str(i) for i in range(int(os.environ.get("K210_MODEL_CLASSES", "2")))])
    progress = []
    res = src.run_action("load_model", card=card, progress=lambda f, t: progress.append(f))
    print(res)
    assert res.ok
    assert progress == sorted(progress)
    deadline = time.monotonic() + 20
    got = None
    while time.monotonic() < deadline and got is None:
        f = src.read(timeout=1.0)
        got = f.device_result if f is not None else None
    assert got is not None and got.where == "device"
    print("板上推論", got.kind, got.timings_ms, got.top_k[:3] or got.detections[:3])
    assert src.run_action("unload_model").ok
