"""MJPEG 來源（含共用的 NetworkSource、MjpegParser）：用 FakeCameraServer 測試，不碰真的相機。"""

from __future__ import annotations

import random
import threading
import time

import cv2
import numpy as np
import pytest

from edgelens.sources._fake_http import FakeCameraServer, jpeg_index
from edgelens.sources.base import SourceError
from edgelens.sources.mjpeg import MjpegParser, MjpegSource, parse_boundary, strip_userinfo

FAST = {"fail_after_s": 1.5, "stall_s": 2.0, "open_wait_s": 3.0}


def net_threads():
    return [t for t in threading.enumerate() if t.name.startswith("edgelens-") and t.is_alive()]


def make_source(url, **params):
    return MjpegSource(MjpegSource.info_from_params({"url": url, **FAST, **params}))


def read_n(src, n, timeout=3.0):
    out = []
    t_end = time.monotonic() + timeout * n
    while len(out) < n and time.monotonic() < t_end:
        f = src.read(timeout=1.0)
        if f is not None:
            out.append(f)
    return out


def jpeg_bytes(w=32, h=24, seed=0):
    img = np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


# ---- 解析 -----------------------------------------------------------------------

@pytest.mark.parametrize("ctype,expected", [
    ("multipart/x-mixed-replace;boundary=frame", "frame"),
    ("multipart/x-mixed-replace; boundary = frame ", "frame"),
    ('multipart/x-mixed-replace;boundary="frame"', "frame"),
    ("multipart/x-mixed-replace;boundary=--frame", "frame"),
    ("MULTIPART/X-MIXED-REPLACE; BOUNDARY=Frame", "Frame"),
    ("multipart/x-mixed-replace", None),
    ("image/jpeg", None),
])
def test_parse_boundary(ctype, expected):
    assert parse_boundary(ctype) == expected


def _feed(data: bytes, boundary, chunk_sizes=(1, 7, 100, 65536)):
    """把 data 用各種切法餵給 parser，回傳解析出的 JPEG 清單。"""
    results = []
    for size in chunk_sizes:
        pos = 0

        def read(size=size):
            nonlocal pos
            if pos >= len(data):
                return b""
            chunk = data[pos:pos + size]
            pos += size
            return chunk

        p = MjpegParser(boundary, read)
        got = []
        try:
            while True:
                got.append(p.next_jpeg())
        except ConnectionError:
            pass
        results.append(got)
    return results


@pytest.mark.parametrize("with_length", [True, False])
@pytest.mark.parametrize("boundary,marker", [("frame", b"--frame"), ("Frame", b"--frame"), ("--frame", b"--frame"),
                                              (" frame ", b"--frame")])
def test_parser_variants(with_length, boundary, marker):
    jpegs = [jpeg_bytes(seed=i) for i in range(3)]
    data = b""
    for j in jpegs:
        head = b"\r\n" + marker + b"\r\nContent-Type: image/jpeg\r\n"
        if with_length:
            head += b"Content-Length: %d\r\n" % len(j)
        data += head + b"\r\n" + j
    data += b"\r\n" + marker + b"--\r\n"
    for got in _feed(data, boundary):
        assert got[:3] == jpegs


def test_parser_without_boundary_scans_soi_eoi():
    jpegs = [jpeg_bytes(seed=i) for i in range(3)]
    data = b"".join(jpegs)
    for got in _feed(data, None):
        assert got == jpegs


def test_parser_skips_garbage_and_exif_thumbnail_eoi():
    # 第一張的前面有垃圾；中間夾一個含 FFD9 的假縮圖（不是結尾）
    j = jpeg_bytes()
    thumb = jpeg_bytes(w=8, h=8, seed=9)
    fake_inner = j[:2] + b"\xff\xe1" + (len(thumb) + 2).to_bytes(2, "big") + thumb + j[2:]
    data = b"garbage--fra" + b"\r\n--frame\r\n\r\n" + fake_inner + b"\r\n--frame\r\n\r\n" + j + b"\r\n--frame\r\n"
    for got in _feed(data, "frame"):
        assert got[0] == fake_inner and got[1] == j


def test_parser_random_chunking():
    rng = random.Random(1)
    jpegs = [jpeg_bytes(w=64, h=48, seed=i) for i in range(5)]
    data = b"".join(b"--frame\r\nContent-Length: %d\r\n\r\n" % len(j) + j + b"\r\n" for j in jpegs)
    pos = 0

    def read():
        nonlocal pos
        n = rng.randint(1, 300)
        chunk = data[pos:pos + n]
        pos += n
        return chunk

    p = MjpegParser("frame", read)
    assert [p.next_jpeg() for _ in jpegs] == jpegs


def test_strip_userinfo():
    assert strip_userinfo("http://bob:secret@10.0.0.2:8080/video?x=1") == "http://10.0.0.2:8080/video?x=1"
    assert strip_userinfo("http://10.0.0.2/video") == "http://10.0.0.2/video"


# ---- 串流 -----------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {},
    {"content_length": False},
    {"boundary_variant": "spaced"},
    {"boundary_variant": "quoted"},
    {"boundary_variant": "dashes"},
    {"boundary_variant": "case"},
    {"content_length": False, "chunked": True},
    {"chunked": True},
])
def test_stream_variants(kwargs):
    with FakeCameraServer(fps=40, mode="mjpeg", **kwargs) as srv:
        src = make_source(srv.stream_url)
        src.open()
        try:
            frames = read_n(src, 5)
            assert len(frames) == 5
            for f in frames:
                assert f.image.shape == (240, 320, 3)
                assert f.jpeg is not None and f.jpeg[:2] == b"\xff\xd8"
                assert jpeg_index(f.jpeg) is not None
                assert {"net_fps", "reconnects", "bytes_per_s", "decode_failures", "mode"} <= set(f.meta)
            seqs = [f.seq for f in frames]
            assert seqs == sorted(set(seqs))
        finally:
            src.close()
    assert not net_threads()


def test_meta_reports_rates():
    with FakeCameraServer(fps=30, mode="mjpeg") as srv:
        with make_source(srv.stream_url) as src:
            frames = read_n(src, 30)
            m = frames[-1].meta
            assert 10 < m["net_fps"] < 60
            assert m["bytes_per_s"] > 1000
            assert m["reconnects"] == 0 and m["mode"] == "stream"


def test_bad_jpeg_is_counted_not_fatal():
    with FakeCameraServer(fps=40, mode="mjpeg", bad_every=3) as srv:
        with make_source(srv.stream_url) as src:
            frames = read_n(src, 8)
            assert len(frames) == 8
            assert frames[-1].meta["decode_failures"] >= 1


def test_snapshot_mode():
    with FakeCameraServer(fps=30, mode="mjpeg") as srv:
        with make_source(srv.url + "/capture", mode="snapshot", snapshot_interval_ms=60) as src:
            frames = read_n(src, 4)
            assert len(frames) == 4
            assert frames[0].meta["mode"] == "snapshot"
            assert src.capabilities().extra["mode"] == "snapshot"
            assert srv.capture_requests >= 4
            assert src.capabilities().provides_jpeg


def test_stream_mode_on_still_image_url_switches_to_snapshot():
    with FakeCameraServer(fps=30, mode="mjpeg") as srv:
        with make_source(srv.url + "/capture", snapshot_interval_ms=60) as src:
            frames = read_n(src, 3)
            assert len(frames) == 3 and frames[-1].meta["mode"] == "snapshot"


def test_basic_auth():
    with FakeCameraServer(fps=30, mode="mjpeg", username="kid", password="pw123") as srv:
        with make_source(srv.stream_url, username="kid", password="pw123") as src:
            assert read_n(src, 2)
        # 帳密寫在網址裡也可以
        url = srv.stream_url.replace("http://", "http://kid:pw123@")
        with make_source(url) as src:
            assert read_n(src, 2)
            assert "pw123" not in src.info.id and "pw123" not in repr(src)
        with pytest.raises(SourceError) as e:
            make_source(srv.stream_url, username="kid", password="wrong").open()
        assert "帳號" in e.value.message and e.value.retryable
        with pytest.raises(SourceError):
            make_source(srv.stream_url).open()
    assert not net_threads()


def test_id_strips_password_and_is_stable():
    a = MjpegSource.info_from_params({"url": "http://u:p@1.2.3.4:8080/v", "password": "x"})
    assert "p@" not in a.id and a.id == "mjpeg:http://1.2.3.4:8080/v"
    assert MjpegSource.info_from_params({"url": "http://u:p@1.2.3.4:8080/v"}).id == a.id


def test_bad_urls_and_unreachable():
    with pytest.raises(SourceError) as e:
        make_source("ftp://x/y").open()
    assert "http://" in e.value.message
    with pytest.raises(SourceError):
        make_source("not a url").open()
    # 連線被拒：open 要很快就回報，而且訊息說明可能原因
    s = socket_free_port()
    t0 = time.monotonic()
    with pytest.raises(SourceError) as e:
        make_source(f"http://127.0.0.1:{s}/stream").open()
    assert time.monotonic() - t0 < 3.0
    assert "Wi-Fi" in e.value.message and "IP" in e.value.message
    assert not net_threads()


def socket_free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_http_error_status():
    with FakeCameraServer(fps=30, mode="mjpeg") as srv:
        with pytest.raises(SourceError) as e:
            make_source(srv.url + "/no_such_path").open()
        assert "404" in e.value.message


# ---- 斷線與重連 -------------------------------------------------------------------

def test_reconnect_after_drop():
    with FakeCameraServer(fps=40, mode="mjpeg") as srv:
        with make_source(srv.stream_url) as src:
            assert read_n(src, 3)
            before = src.read(timeout=1.0)
            srv.drop_connections()
            deadline = time.monotonic() + 8
            after = None
            while time.monotonic() < deadline:
                f = src.read(timeout=0.5)
                if f is not None and f.meta["reconnects"] >= 1:
                    after = f
                    break
            assert after is not None, "斷線後沒有恢復"
            assert after.seq > (before.seq if before else 0)
            assert srv.stream_connections >= 2
            assert after.meta["reconnects"] >= 1


def test_reconnect_after_stall():
    with FakeCameraServer(fps=40, mode="mjpeg") as srv:
        with make_source(srv.stream_url, stall_s=0.8, fail_after_s=6.0) as src:
            assert read_n(src, 3)
            srv.pause(True)
            time.sleep(1.5)
            srv.pause(False)
            deadline = time.monotonic() + 8
            ok = False
            while time.monotonic() < deadline:
                f = src.read(timeout=0.5)
                if f is not None and f.meta["reconnects"] >= 1:
                    ok = True
                    break
            assert ok


def test_server_stopped_raises_source_error():
    srv = FakeCameraServer(fps=40, mode="mjpeg").start()
    src = make_source(srv.stream_url, fail_after_s=1.0)
    src.open()
    try:
        assert read_n(src, 3)
        srv.stop()
        t0 = time.monotonic()
        with pytest.raises(SourceError) as e:
            while time.monotonic() - t0 < 10:
                src.read(timeout=0.3)
        assert 0.8 < time.monotonic() - t0 < 6
        assert "Wi-Fi" in e.value.message
    finally:
        src.close()
    assert not net_threads()


def test_read_timeout_returns_none_when_paused():
    with FakeCameraServer(fps=40, mode="mjpeg") as srv:
        with make_source(srv.stream_url, stall_s=30) as src:
            assert read_n(src, 2)
            srv.pause(True)
            time.sleep(0.2)
            src.read(timeout=0.1)  # 清掉殘留
            t0 = time.monotonic()
            assert src.read(timeout=0.3) is None
            assert 0.25 < time.monotonic() - t0 < 1.0
            srv.pause(False)


def test_read_returns_each_frame_once():
    with FakeCameraServer(fps=5, mode="mjpeg") as srv:
        with make_source(srv.stream_url) as src:
            f1 = src.read(timeout=2.0)
            f2 = src.read(timeout=2.0)
            assert f1 is not None and f2 is not None and f2.seq > f1.seq


def test_slow_network_still_works():
    with FakeCameraServer(fps=10, mode="mjpeg", bandwidth_bps=400_000) as srv:
        with make_source(srv.stream_url) as src:
            assert len(read_n(src, 3, timeout=5)) == 3


# ---- 延遲不會累積 -------------------------------------------------------------------

def _latency_run(seconds: float, sleep_s: float = 0.05):
    with FakeCameraServer(fps=60, mode="mjpeg") as srv:
        with make_source(srv.stream_url) as src:
            ages, lags, t_end = [], [], time.monotonic() + seconds
            while time.monotonic() < t_end:
                f = src.read(timeout=1.0)
                if f is None:
                    continue
                ages.append(f.age_ms())
                lags.append(srv.current_index() - jpeg_index(f.jpeg))
                time.sleep(sleep_s)   # 故意讀得比伺服器慢
    return ages, lags


def _check_latency(ages, lags):
    assert len(ages) > 10
    half = len(ages) // 2
    assert max(ages) < 250, f"age 過大：{max(ages):.0f} ms"
    # 後半段的平均延遲不可以比前半段大很多（累積延遲會一路增加）
    assert np.mean(lags[half:]) <= np.mean(lags[:half]) + 6, (np.mean(lags[:half]), np.mean(lags[half:]))
    assert np.mean(ages[half:]) <= np.mean(ages[:half]) + 40
    assert max(lags) < 40, f"落後 {max(lags)} 張"


def test_latency_does_not_grow_10s():
    _check_latency(*_latency_run(10.0))


@pytest.mark.slow
def test_latency_does_not_grow_5min():
    _check_latency(*_latency_run(300.0))


# ---- 關閉 ------------------------------------------------------------------------

def test_close_is_fast_and_idempotent():
    with FakeCameraServer(fps=30, mode="mjpeg") as srv:
        src = make_source(srv.stream_url)
        src.open()
        src.open()
        read_n(src, 2)
        t0 = time.monotonic()
        src.close()
        assert time.monotonic() - t0 < 2.0
        src.close()
        assert not net_threads()
        with pytest.raises(SourceError):
            src.read(timeout=0.1)
        src.open()   # 關閉後可再開
        assert read_n(src, 2)
        src.close()
    assert not net_threads()


def test_close_while_server_is_paused_is_fast():
    with FakeCameraServer(fps=30, mode="mjpeg") as srv:
        src = make_source(srv.stream_url, stall_s=30)
        src.open()
        read_n(src, 2)
        srv.pause(True)
        time.sleep(0.3)
        t0 = time.monotonic()
        src.close()
        assert time.monotonic() - t0 < 2.0
        srv.pause(False)
    assert not net_threads()


def test_param_specs_and_manual_add():
    keys = [s.key for s in MjpegSource.param_specs()]
    assert keys == ["url", "mode", "snapshot_interval_ms", "username", "password"]
    assert MjpegSource.manual_add and MjpegSource.discover() == []
