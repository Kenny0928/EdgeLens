"""RTSP 來源：用假的 cv2.VideoCapture 測背景執行緒，再用 FFMPEG 開本機影片檔當「串流」做整合測試。"""

from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np
import pytest

import edgelens.sources.rtsp as rtsp_mod
from edgelens.sources.base import SourceError
from edgelens.sources.rtsp import ENV_KEY, RtspSource

FAST = {"fail_after_s": 1.5, "stall_s": 2.0, "open_wait_s": 3.0}


def net_threads():
    return [t for t in threading.enumerate() if t.name.startswith("edgelens-") and t.is_alive()]


class FakeCapture:
    """假的 VideoCapture：每次 read 回傳一張編號遞增的小圖。"""

    instances: list[FakeCapture] = []
    opened_ok = True
    fail_after: int | None = None   # 讀了這麼多張之後 read 開始失敗（只影響第一個實例）
    read_delay = 0.0
    env_seen: list[str | None] = []
    args_seen: list[tuple] = []

    def __init__(self, url, api=None, params=None):
        type(self).args_seen.append((url, api, params))
        type(self).env_seen.append(os.environ.get(ENV_KEY))
        self.n = 0
        self.released = False
        self.index = len(type(self).instances)
        type(self).instances.append(self)

    def isOpened(self):  # noqa: N802
        return type(self).opened_ok

    def read(self):
        if self.released:
            raise RuntimeError("read after release")
        time.sleep(type(self).read_delay)
        fa = type(self).fail_after
        if fa is not None and self.index == 0 and self.n >= fa:
            return False, None
        self.n += 1
        img = np.full((6, 8, 3), self.n % 256, np.uint8)
        return True, img

    def get(self, prop):
        return 30.0

    def set(self, prop, value):
        return True

    def release(self):
        self.released = True


@pytest.fixture
def fake_cap(monkeypatch):
    FakeCapture.instances = []
    FakeCapture.opened_ok = True
    FakeCapture.fail_after = None
    FakeCapture.read_delay = 0.002
    FakeCapture.env_seen = []
    FakeCapture.args_seen = []
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.delenv(ENV_KEY, raising=False)
    return FakeCapture


def make(url="rtsp://192.168.1.50:554/live", **params):
    return RtspSource(RtspSource.info_from_params({"url": url, **FAST, **params}))


def test_frames_and_meta(fake_cap):
    with make() as src:
        frames = [f for f in (src.read(timeout=1.0) for _ in range(5)) if f is not None]
        assert len(frames) == 5
        f = frames[-1]
        assert f.image.shape == (6, 8, 3)
        assert {"net_fps", "reconnects", "bytes_per_s"} <= set(f.meta)
        assert f.meta["net_fps"] > 10
        assert f.jpeg is None
        assert [x.seq for x in frames] == sorted({x.seq for x in frames})
        assert not src.capabilities().provides_jpeg


def test_keeps_only_latest_frame(fake_cap):
    fake_cap.read_delay = 0.002   # 約 500 張/秒
    with make() as src:
        lags = []
        t_end = time.monotonic() + 2.0
        while time.monotonic() < t_end:
            f = src.read(timeout=1.0)
            cap = fake_cap.instances[0]
            lags.append((cap.n - int(f.image[0, 0, 0])) % 256)
            time.sleep(0.05)
        assert max(lags) < 30, f"落後 {max(lags)} 張，舊畫面沒有被丟掉"
        assert fake_cap.instances[0].n > 100   # 背景執行緒一直在讀


def test_returns_none_on_timeout(fake_cap):
    fake_cap.read_delay = 1.0
    with make(open_wait_s=0.1) as src:
        t0 = time.monotonic()
        assert src.read(timeout=0.2) is None
        assert time.monotonic() - t0 < 0.6


def test_reconnects_after_read_failure(fake_cap):
    fake_cap.fail_after = 5
    with make() as src:
        reconnected = None
        t_end = time.monotonic() + 8
        while time.monotonic() < t_end:
            f = src.read(timeout=0.5)
            if f is not None and f.meta["reconnects"] >= 1:
                reconnected = f
                break
        assert reconnected is not None
        assert len(fake_cap.instances) >= 2
        assert fake_cap.instances[0].released


def test_open_failure_message(fake_cap):
    fake_cap.opened_ok = False
    src = make()
    with pytest.raises(SourceError) as e:
        src.open()
    assert "Wi-Fi" in e.value.message and "rtsp://192.168.1.50:554/live" in e.value.message
    assert e.value.retryable
    assert not net_threads()


def test_error_after_long_failure(fake_cap):
    src = make(fail_after_s=0.8, open_wait_s=0.01)
    src.open()   # open_wait 很短：還沒看到錯誤就回來
    fake_cap.opened_ok = False
    fake_cap.fail_after = 0
    try:
        t0 = time.monotonic()
        with pytest.raises(SourceError):
            while time.monotonic() - t0 < 8:
                src.read(timeout=0.2)
    finally:
        src.close()


def test_close_is_fast_and_releases(fake_cap):
    src = make()
    src.open()
    assert src.read(timeout=1.0) is not None
    t0 = time.monotonic()
    src.close()
    assert time.monotonic() - t0 < 2.0
    assert not net_threads()
    assert all(c.released for c in fake_cap.instances)
    src.close()
    src.open()   # 可再開
    assert src.read(timeout=1.0) is not None
    src.close()
    assert not net_threads()


def test_env_option_set_during_open_and_restored(fake_cap, monkeypatch):
    monkeypatch.setenv(ENV_KEY, "keep;me")
    with make():
        pass
    assert fake_cap.env_seen[0] == "rtsp_transport;tcp"
    assert os.environ[ENV_KEY] == "keep;me"
    monkeypatch.delenv(ENV_KEY)
    with make(transport="udp"):
        pass
    assert fake_cap.env_seen[-1] == "rtsp_transport;udp"
    assert ENV_KEY not in os.environ


def test_timeout_props_passed_when_supported(fake_cap):
    with make():
        pass
    url, api, params = fake_cap.args_seen[0]
    assert api == cv2.CAP_FFMPEG
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        assert cv2.CAP_PROP_OPEN_TIMEOUT_MSEC in params and cv2.CAP_PROP_READ_TIMEOUT_MSEC in params


def test_falls_back_when_params_not_supported(monkeypatch):
    calls = []

    class Old:
        def __init__(self, url, api=None):   # 舊版不收第三個參數
            calls.append(url)

    monkeypatch.setattr(cv2, "VideoCapture", Old)
    assert isinstance(rtsp_mod.open_capture("rtsp://x/y"), Old)


def test_url_validation_and_id():
    with pytest.raises(SourceError) as e:
        RtspSource(RtspSource.info_from_params({"url": "ftp://x/y"})).open()
    assert "rtsp://" in e.value.message
    info = RtspSource.info_from_params({"url": "rtsp://admin:secret@10.0.0.9:554/s1"})
    assert "secret" not in info.id and "secret" not in info.detail and info.id == "rtsp:rtsp://10.0.0.9:554/s1"
    for scheme in ("rtmp://h/app", "http://h/live.ts", "https://h/x.m3u8"):
        assert RtspSource.info_from_params({"url": scheme}).id.endswith(scheme)


def test_param_specs():
    assert [s.key for s in RtspSource.param_specs()] == ["url", "transport"]
    assert RtspSource.manual_add


# ---- 整合：本機影片檔當作串流（FFMPEG 後端）-----------------------------------------

def test_local_video_file_via_ffmpeg(tmp_path):
    info = RtspSource.contract_fixture(tmp_path)
    if info is None:
        pytest.skip("這個 OpenCV 無法寫出 AVI")
    src = RtspSource(info)
    src.open()
    try:
        frames = [f for f in (src.read(timeout=2.0) for _ in range(8)) if f is not None]
        assert len(frames) >= 5
        assert frames[0].image.shape == (120, 160, 3)
        # 影片檔播完會從頭再播
        assert src.read(timeout=3.0) is not None
    finally:
        src.close()
    assert not net_threads()
