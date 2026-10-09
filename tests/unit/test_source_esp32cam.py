"""ESP32-CAM 來源：用仿 CameraWebServer 的 FakeCameraServer 測試。"""

from __future__ import annotations

import threading
import time

import pytest

from edgelens.core.controls import ControlType
from edgelens.i18n import has
from edgelens.sources._fake_http import DEFAULT_STATE, FakeCameraServer
from edgelens.sources.base import SourceError
from edgelens.sources.esp32cam import KNOWN_KEYS, Esp32CamSource, build_specs, parse_ip
from edgelens.sources.mjpeg import open_http

FAST = {"fail_after_s": 2.0, "stall_s": 2.0, "open_wait_s": 4.0}


def net_threads():
    return [t for t in threading.enumerate() if t.name.startswith("edgelens-") and t.is_alive()]


def make(srv, **params):
    return Esp32CamSource(Esp32CamSource.info_from_params(
        {"ip": f"127.0.0.1:{srv.port}", "stream_port": srv.stream_port, **FAST, **params}))


def wait_frame(src, pred, seconds=8.0):
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        f = src.read(timeout=0.5)
        if f is not None and pred(f):
            return f
    return None


def test_controls_generated_from_status():
    with FakeCameraServer(fps=20) as srv:
        src = make(srv)
        assert src.controls() == []   # 開啟前沒有
        src.open()
        try:
            keys = [s.key for s in src.controls()]
            expected = [k for k in KNOWN_KEYS if k in srv.status_json()]
            assert keys == expected
            assert "denoise" not in keys and "led_intensity" not in keys   # 韌體沒回報
            for ignored in ("xclk", "face_detect", "face_recognize"):
                assert ignored not in keys
            by = {s.key: s for s in src.controls()}
            assert by["framesize"].type == ControlType.ENUM and by["framesize"].default == 5
            assert by["framesize"].option_values == list(range(14))
            assert by["brightness"].type == ControlType.INT and (by["brightness"].min, by["brightness"].max) == (-2, 2)
            assert by["vflip"].type == ControlType.BOOL and by["vflip"].default is False
            assert by["special_effect"].option_values == list(range(7))
            assert by["wb_mode"].option_values == list(range(5))
            assert by["agc"].default is True
            assert all(s.readback for s in src.controls())
            assert by["aec2"].advanced and not by["brightness"].advanced
            assert src.get_control("framesize") == 5 and src.get_control("awb") is True
            with pytest.raises(KeyError):
                src.get_control("denoise")
        finally:
            src.close()
        assert src.controls() == []


def test_controls_follow_firmware():
    # 不同韌體／感光元件：只回報一部分，或多了 denoise 和大尺寸
    with FakeCameraServer(fps=20, omit=tuple(k for k in DEFAULT_STATE if k not in ("framesize", "quality", "vflip"))) as srv:
        with make(srv) as src:
            assert [s.key for s in src.controls()] == ["framesize", "quality", "vflip"]
    specs = build_specs({"framesize": 15, "quality": 10, "denoise": 3, "brightness": 3})
    by = {s.key: s for s in specs}
    assert 15 in by["framesize"].option_values and 14 in by["framesize"].option_values
    assert by["brightness"].max == 3   # 範圍會放寬到包含目前值
    assert "denoise" in by


def test_all_labels_translated():
    specs = build_specs({k: DEFAULT_STATE.get(k, 0) for k in KNOWN_KEYS})
    assert [s.key for s in specs] == list(KNOWN_KEYS)
    for s in specs:
        for lang in ("zh_TW", "en"):
            assert has(s.label_key, lang), (s.key, s.label_key)
            if s.help_key:
                assert has(s.help_key, lang), s.help_key
            if s.type == ControlType.ENUM and s.key != "framesize" and s.key != "gainceiling":
                for _, k in s.options:
                    assert has(k, lang), k
    for key in ("control.net.ip.help", "control.net.ip", "control.net.stream_port"):
        assert has(key, "zh_TW") and has(key, "en")
    assert "Wi-Fi" in __import__("edgelens.i18n", fromlist=["tr"]).tr("control.net.ip.help")


def test_set_control_readback_and_effect():
    with FakeCameraServer(fps=25) as srv:
        with make(srv) as src:
            f0 = wait_frame(src, lambda f: True)
            base = float(f0.image.mean())
            res = src.set_control("brightness", 2)
            assert res.ok and res.verified is True and res.value == 2
            assert srv.state["brightness"] == 2 and ("brightness", "2") in srv.control_log
            f1 = wait_frame(src, lambda f: f.image.mean() > base + 20)
            assert f1 is not None, "亮度設定沒有反映在畫面上"
            assert src.get_control("brightness") == 2
            res = src.set_control("vflip", True)
            assert res.ok and res.verified is True and res.value is True and srv.state["vflip"] == 1
            assert src.set_control("brightness", 99).value == 2   # 超出範圍會夾到上限
            bad = src.set_control("no_such", 1)
            assert not bad.ok
            assert not src.set_control("denoise", 1).ok   # 這台韌體沒有 denoise
            assert not src.set_control("brightness", "abc").ok


def test_set_control_unverified_when_firmware_ignores():
    with FakeCameraServer(fps=25, ignore_vars=("contrast",)) as srv:
        with make(srv) as src:
            res = src.set_control("contrast", 2)
            assert res.ok and res.verified is False and res.value == 0
            assert "contrast" in res.message


def test_framesize_change_recovers_stream():
    with FakeCameraServer(fps=25) as srv:
        with make(srv) as src:
            f0 = wait_frame(src, lambda f: True)
            assert f0.image.shape[:2] == (240, 320)
            res = src.set_control("framesize", 8)
            assert res.ok and res.verified is True
            f1 = wait_frame(src, lambda f: f.image.shape[:2] == (480, 640))
            assert f1 is not None, "改解析度後串流沒有恢復"
            assert srv.stream_connections >= 2
            assert (640, 480) in src.capabilities().resolutions


def test_falls_back_to_capture_when_stream_busy():
    with FakeCameraServer(fps=25) as srv:
        # 先用另一個連線（像瀏覽器）占住串流名額
        conn, resp = open_http(srv.stream_url)
        assert resp.status == 200
        try:
            with make(srv) as src:
                f = wait_frame(src, lambda f: True)
                assert f is not None
                assert f.meta["mode"] == "capture"
                assert src.capabilities().extra["mode"] == "capture"
                assert srv.stream_rejected >= 1 and srv.capture_requests >= 1
                # 設定仍然可用
                assert src.set_control("brightness", 1).verified is True
        finally:
            resp.close()
            conn.close()
    assert not net_threads()


def test_returns_to_stream_when_it_becomes_free(monkeypatch):
    monkeypatch.setattr(Esp32CamSource, "STREAM_RETRY_S", 1.0)
    with FakeCameraServer(fps=25) as srv:
        conn, resp = open_http(srv.stream_url)
        with make(srv) as src:
            assert wait_frame(src, lambda f: f.meta["mode"] == "capture") is not None
            resp.close()
            conn.close()
            f = wait_frame(src, lambda f: f.meta["mode"] == "stream", seconds=10)
            assert f is not None, "名額釋放後沒有接回串流"
            assert src.capabilities().extra["mode"] == "stream"


def test_reconnect_after_drop():
    with FakeCameraServer(fps=25) as srv:
        with make(srv) as src:
            assert wait_frame(src, lambda f: True)
            srv.drop_connections()
            f = wait_frame(src, lambda f: f.meta["reconnects"] >= 1)
            assert f is not None


def test_open_errors():
    # 不是 CameraWebServer（mjpeg 模式的伺服器沒有 /status）
    with FakeCameraServer(fps=20, mode="mjpeg") as srv:
        with pytest.raises(SourceError) as e:
            Esp32CamSource(Esp32CamSource.info_from_params({"ip": f"127.0.0.1:{srv.port}"})).open()
        assert "CameraWebServer" in e.value.message and e.value.retryable
    # 連不上
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    t0 = time.monotonic()
    with pytest.raises(SourceError) as e:
        Esp32CamSource(Esp32CamSource.info_from_params({"ip": f"127.0.0.1:{port}"})).open()
    assert time.monotonic() - t0 < 4
    assert "Wi-Fi" in e.value.message and "序列埠" in e.value.message
    assert not net_threads()


def test_params_id_and_discover():
    info = Esp32CamSource.info_from_params({})
    assert info.id == "esp32cam:192.168.4.1" and info.params["stream_port"] == 81
    assert Esp32CamSource.info_from_params({"ip": " http://10.0.0.5/ "}).id == "esp32cam:10.0.0.5"
    assert Esp32CamSource.discover() == [] and Esp32CamSource.discover_is_safe
    assert [s.key for s in Esp32CamSource.param_specs()] == ["ip", "stream_port"]
    assert Esp32CamSource.param_specs()[0].default == "192.168.4.1"
    assert parse_ip("192.168.4.1") == ("192.168.4.1", 80) and parse_ip("h:8080") == ("h", 8080)


def test_close_is_fast_and_clean():
    with FakeCameraServer(fps=25) as srv:
        src = make(srv)
        src.open()
        wait_frame(src, lambda f: True)
        t0 = time.monotonic()
        src.close()
        assert time.monotonic() - t0 < 2.0
        assert not net_threads()
        src.close()


@pytest.mark.hardware
def test_real_esp32cam():
    """需要實體 ESP32-CAM（燒錄 CameraWebServer）。用環境變數 EDGELENS_ESP32_IP 指定位址。"""
    import os
    ip = os.environ.get("EDGELENS_ESP32_IP")
    if not ip:
        pytest.skip("未設定 EDGELENS_ESP32_IP")
    with Esp32CamSource(Esp32CamSource.info_from_params({"ip": ip})) as src:
        assert wait_frame(src, lambda f: True, seconds=10) is not None
        assert src.controls()


def test_demo_source_second_client_falls_back_to_capture():
    """模擬 ESP32-CAM 的 #1、#2 連到同一台模擬板子：#2 因為串流被占用而改抓單張。"""
    import time

    from edgelens.sources.esp32cam import Esp32CamDemoSource

    one, two = (Esp32CamDemoSource(i) for i in Esp32CamDemoSource.discover())
    assert one.source_id != two.source_id
    one.open()
    try:
        assert one.read(timeout=3.0) is not None
        two.open()
        try:
            deadline = time.monotonic() + 8
            mode = None
            while time.monotonic() < deadline:
                f = two.read(timeout=1.0)
                mode = two.capabilities().extra.get("mode")
                if f is not None and mode != "stream":
                    break
            assert mode != "stream"
        finally:
            two.close()
    finally:
        one.close()
