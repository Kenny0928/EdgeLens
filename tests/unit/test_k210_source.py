"""K210Source：用模擬板子（sim.FakeK210Board）測試 open／read／close、參數、動作與錯誤訊息。"""

from __future__ import annotations

import json
import time

import pytest

from edgelens.core.controls import ControlType
from edgelens.core.modelcard import InputSpec, ModelCard
from edgelens.sources.base import SourceError, SourceInfo
from edgelens.sources.k210 import flash as flash_mod
from edgelens.sources.k210 import source as S
from edgelens.sources.k210.sim import FakeK210Board

ANCHORS = [1.889, 2.5245, 2.9465, 3.94056, 3.99987, 5.3658, 5.155437, 6.92275, 6.718375, 9.01025]


class Env:
    def __init__(self):
        self.boards: list[FakeK210Board] = []
        self.kwargs: dict = {}

    @property
    def board(self) -> FakeK210Board:
        return self.boards[-1]


@pytest.fixture
def env(monkeypatch):
    e = Env()

    def factory(port, baud):
        b = FakeK210Board(**e.kwargs)
        b.time_scale = 0.0   # 不真的等待
        e.boards.append(b)
        return b

    monkeypatch.setattr(S, "open_serial", factory)
    return e


def make(env, **params) -> S.K210Source:
    return S.K210Source(SourceInfo(kind="k210", id="k210:test", name="t", params={"port": "sim:", **params}))


@pytest.fixture
def src(env):
    s = make(env)
    s.open()
    yield s
    s.close()


def frame_of(s, tries=30):
    for _ in range(tries):
        f = s.read(timeout=0.5)
        if f is not None:
            return f
    raise AssertionError("讀不到畫面")


def model_card(tmp_path, task="detect", labels=("cat", "dog"), path="/flash/m.kmodel", **dev):
    f = tmp_path / "m.kmodel"
    f.write_bytes(b"KMODEL" * 700)
    device = {"path": path, "task": task, **dev}
    if task == "detect":
        device.setdefault("anchors", ANCHORS)
    return ModelCard(name="m", file=str(f), labels=list(labels), device=device,
                     input=InputSpec(size=(320, 240)), conf_threshold=0.4, nms_threshold=0.25)


# ---- 開啟、讀取、關閉 -------------------------------------------------------------

def test_open_read_close(env):
    s = make(env)
    with pytest.raises(SourceError):
        s.read(timeout=0.1)
    s.open()
    assert s.is_open
    f = frame_of(s)
    assert f.image.shape == (240, 320, 3) and f.jpeg[:2] == b"\xff\xd8" and f.jpeg[-2:] == b"\xff\xd9"
    json.dumps(f.meta)
    assert f.meta["w"] == 320 and "ack" not in f.meta and "host" in f.meta
    f2 = frame_of(s)
    assert f2.seq > f.seq
    s.close()
    assert not s.is_open and env.board.is_open is False
    s.close()  # 重複關閉不出錯
    with pytest.raises(SourceError):
        s.read(timeout=0.1)


def test_board_info_parsed(src):
    info = src.board_info
    assert info["caps"]["sensor"] == "GC0328" and info["caps"]["kpu"] is True
    assert info["fw"].startswith("v0.6.3_2")
    assert info["config"]["framesize"] == "QVGA"
    assert info["mem"]["mem_free"] > 0
    caps = src.capabilities()
    assert caps.device_inference and caps.provides_jpeg and (320, 240) in caps.resolutions
    assert caps.extra["sensor"] == "GC0328"
    assert set(caps.actions) == {"load_model", "unload_model", "upload_file", "set_cpu_freq", "flash_firmware"}


def test_open_sends_p_and_script(env):
    s = make(env, lcd=True)
    s.open()
    p = env.board.params
    assert p["framesize"] == "QVGA" and p["baud"] == S.STREAM_BAUD and p["lcd"] is True and p["ops"] == []
    assert env.board.starts == 1
    s.close()


def test_open_twice_is_noop(env):
    s = make(env)
    s.open()
    s.open()
    assert len(env.boards) == 1
    s.close()


def test_only_latest_frame_returned_and_counted(env):
    env.kwargs["frames_per_read"] = 5
    s = make(env)
    s.open()
    f = frame_of(s)
    assert f.meta["board_seq"] == 4
    assert s.stats["dropped_frames"] >= 4
    s.close()


def test_data_only_packets_update_result_without_frame(env):
    s = make(env, send_every=0)
    s.open()
    env.board.objs = [[10, 20, 30, 40, 0.9, 0]]
    s._card = None
    s.board_info["model_path"] = "/flash/x.kmodel"
    env.board.params["ops"] = ["yolo"]
    assert s.read(timeout=0.2) is None
    assert s.stats["packets"] > 0
    assert s.last_result is not None and s.last_result.kind == "detect"
    assert s.last_result.detections[0].box == (10, 20, 40, 60)
    s.close()


def test_text_from_board_recorded(src, env):
    env.board._out += b"NOTE hello from board\r\n"
    frame_of(src)
    assert "hello from board" in src.last_text


def test_board_error_text_raises_when_stream_stops(src, env):
    env.board._mode = "raw"   # 板子上的程式結束了
    env.board._out += b"\nERROR at seq 3:\r\nMemoryError\r\n"
    src.read(timeout=0.05)
    src._last_rx -= 5
    with pytest.raises(SourceError, match="MemoryError"):
        src.read(timeout=0.5)


def test_stall_raises(src, env):
    env.board.no_response = True
    src._last_rx -= 100
    with pytest.raises(SourceError) as e:
        src.read(timeout=0.2)
    assert e.value.retryable and "RESET" in e.value.hint


def test_serial_lost_raises(src, env):
    def boom(n):
        raise OSError("device disconnected")

    env.board.read = boom
    with pytest.raises(SourceError, match="中斷"):
        src.read(timeout=0.2)
    src.close()  # 板子拔掉後也要能關


# ---- 裝置推論結果 -----------------------------------------------------------------

def test_load_detect_model_and_device_result(src, env, tmp_path):
    card = model_card(tmp_path)
    seen = []
    res = src.run_action("load_model", card=card, progress=lambda f, t: seen.append(f))
    assert res.ok and res.verified is True, res.message
    assert seen[-1] == 1.0 and seen == sorted(seen)
    b = env.board
    assert bytes(b.files["/flash/m.kmodel"]) == card.model_path.read_bytes()
    p = b.params
    assert p["model"] == "/flash/m.kmodel" and p["ops"] == ["yolo"] and p["classes"] == 2
    assert p["anchor"] == ANCHORS and p["input_size"] == [320, 240]
    assert p["threshold"] == 0.4 and p["nms"] == 0.25
    b.objs = [[10, 20, 30, 40, 0.9, 1], [1, 2, 3, 4, 0.5, 0]]
    f = frame_of(src)
    r = f.device_result
    assert r.kind == "detect" and r.where == "device" and r.frame_seq == f.seq and r.image_size == (320, 240)
    assert r.detections[0].box == (10, 20, 40, 60) and r.detections[0].label == "dog"
    assert r.detections[1].label == "cat"
    json.dumps(r.to_dict())


def test_load_classify_model(src, env, tmp_path):
    card = model_card(tmp_path, task="classify", labels=("a", "b", "c", "d"), softmax=False, topk=3)
    card.input = InputSpec(size=(224, 224))
    assert src.run_action("load_model", card=card).ok
    p = env.board.params
    assert p["ops"] == ["classify"] and p["topk"] == 3 and p["softmax"] is False and p["input_size"] == [224, 224]
    assert "anchor" not in p
    env.board.cls = [[3, 0.2], [1, 0.7], [0, 0.1]]
    r = frame_of(src).device_result
    assert r.kind == "classify" and [c.class_id for c in r.top_k] == [1, 3, 0]
    assert r.top_k[0].label == "b" and r.best.class_id == 1


def test_load_model_skips_upload_when_same_size(src, env, tmp_path):
    card = model_card(tmp_path)
    assert src.run_action("load_model", card=card).ok
    writes_before = sum(1 for e in env.board.events if e[0] == "write" and b"a2b_base64" in e[2])
    assert src.run_action("load_model", card=card).ok
    writes_after = sum(1 for e in env.board.events if e[0] == "write" and b"a2b_base64" in e[2])
    assert writes_before == writes_after > 0


def test_load_model_without_local_file_uses_board_copy(src, env, tmp_path):
    env.board.files["/flash/there.kmodel"] = bytearray(b"x")
    card = ModelCard(name="t", file=str(tmp_path / "missing.kmodel"), device={"path": "/flash/there.kmodel",
                                                                              "task": "classify"})
    assert src.run_action("load_model", card=card).ok


def test_load_model_missing_everywhere_fails_and_keeps_streaming(src, env, tmp_path):
    card = ModelCard(name="t", file=str(tmp_path / "missing.kmodel"), device={"path": "/flash/nope.kmodel",
                                                                              "task": "detect"})
    res = src.run_action("load_model", card=card)
    assert not res.ok and "/flash/nope.kmodel" in res.message
    assert frame_of(src) is not None


def test_load_model_without_device_path(src, tmp_path):
    card = ModelCard(name="t", file=str(tmp_path / "m.kmodel"))
    res = src.run_action("load_model", card=card)
    assert not res.ok and "device.path" in res.message


def test_load_model_start_failure_restores_stream(src, env, tmp_path):
    card = model_card(tmp_path)
    src.run_action("load_model", card=card)
    other = model_card(tmp_path, path="/flash/other.kmodel")
    real = env.board._start_script

    def flaky(code):   # 只有帶新模型的啟動失敗，退回舊設定時正常
        if "other.kmodel" in code.split("\n", 1)[0]:
            env.board._out += b"OK\x04MemoryError: allocation failed\x04>"
            return None
        return real(code)

    env.board._start_script = flaky
    res = src.run_action("load_model", card=other)
    assert not res.ok and "記憶體" in res.message
    assert src._card is card
    assert frame_of(src).device_result is not None or env.board.params["model"] == "/flash/m.kmodel"


def test_load_model_cancel_during_upload(src, env, tmp_path):
    card = model_card(tmp_path)
    n = {"c": 0}

    def cancel():
        n["c"] += 1
        return n["c"] > 2

    res = src.run_action("load_model", card=card, cancel=cancel)
    assert not res.ok and "取消" in res.message
    assert "/flash/m.kmodel" not in env.board.files
    assert frame_of(src) is not None


def test_load_model_to_sd_without_card(src, env, tmp_path):
    card = model_card(tmp_path, path="/sd/m.kmodel")
    src.board_info["caps"]["sd"] = False
    res = src.run_action("load_model", card=card)
    assert not res.ok and "SD" in res.message


def test_unload_model(src, env, tmp_path):
    src.run_action("load_model", card=model_card(tmp_path))
    res = src.run_action("unload_model")
    assert res.ok and env.board.params["ops"] == [] and src._card is None
    assert frame_of(src).device_result is None


def test_upload_file_with_progress(src, env, tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"0123456789" * 200)
    seen = []
    res = src.run_action("upload_file", local_path=str(f), remote_path="/flash/data.bin",
                         progress=lambda p, t: seen.append(p))
    assert res.ok and bytes(env.board.files["/flash/data.bin"]) == f.read_bytes()
    assert seen[-1] == 1.0 and seen == sorted(seen)
    assert frame_of(src) is not None
    res = src.run_action("upload_file", local_path=str(tmp_path / "none"), remote_path="/flash/x")
    assert not res.ok


def test_set_cpu_freq_reconnects(src, env):
    res = src.run_action("set_cpu_freq", cpu_mhz=400, kpu_mhz=300)
    assert res.ok, res.message
    assert len(env.boards) == 2
    assert "freq.set(cpu=400, pll1=400, kpu_div=0)" in env.boards[0].freq_calls[0]
    assert frame_of(src) is not None
    assert not src.run_action("set_cpu_freq", cpu_mhz=5, kpu_mhz=400).ok


def test_kpu_div_formula():
    assert S.kpu_div_for(400) == 0 and S.kpu_div_for(200) == 1 and S.kpu_div_for(100) == 3


def test_flash_action_closes_stream_first(src, env, monkeypatch):
    calls = {}

    def fake_flash(port, fw, **kw):
        calls["port"], calls["fw"] = port, fw
        calls["open_during_flash"] = src.is_open
        return flash_mod.FlashResult(True, "done")

    monkeypatch.setattr(flash_mod, "flash_firmware", fake_flash)
    res = src.run_action("flash_firmware", firmware_path="/x/fw.bin")
    assert res.ok and calls["open_during_flash"] is False and calls["port"] == "sim:"
    assert not src.is_open


def test_unknown_action_and_not_open(env):
    s = make(env)
    assert not s.run_action("nope").ok
    assert not s.run_action("unload_model").ok


# ---- 參數 ------------------------------------------------------------------------

def test_controls_layout(src):
    specs = {c.key: c for c in src.controls()}
    live = {k for k, c in specs.items() if c.live and c.type != ControlType.ACTION}
    assert live == {"brightness", "contrast", "saturation", "auto_gain", "gain_db", "auto_exposure",
                    "exposure_us", "auto_whitebal", "vflip", "hmirror", "threshold", "nms"}
    assert {k for k, c in specs.items() if not c.live} == {"framesize", "pixformat", "send_every", "jpeg_quality"}
    assert specs["brightness"].min == -2 and specs["brightness"].max == 2
    assert specs["load_model"].extra["inputs"] == ["card"]


def test_set_control_ack_verified(src, env):
    res = src.set_control("brightness", 1)
    assert res.ok and res.verified is True and res.value == 1
    assert env.board.cmds[-1]["brightness"] == 1
    assert src.get_control("brightness") == 1
    res = src.set_control("brightness", 99)   # 超出範圍會被限制
    assert res.value == 2


def test_set_control_ack_matches_command_id(src, env):
    src.set_control("contrast", 1)
    src.set_control("saturation", -1)
    line = [e[2] for e in env.board.events if e[0] == "write" and e[2].startswith(b"CMD ")]
    ids = [json.loads(x[4:])["_id"] for x in line]
    assert ids == [1, 2]


def test_set_control_rejected_by_board(src, env):
    env.board.reject.add("contrast")
    src.set_control("contrast", 0)
    res = src.set_control("contrast", 2)
    assert res.ok and res.verified is False and res.value == 0 and "rejected" in res.message
    assert src.get_control("contrast") == 0


def test_set_control_without_ack_is_unverified(src, env, monkeypatch):
    monkeypatch.setattr(S, "ACK_TIMEOUT", 0.15)
    env.board.ack_never = True
    res = src.set_control("hmirror", True)
    assert res.ok and res.verified is None


def test_apply_controls_merges_into_one_command(src, env):
    n_before = len(env.board.cmds)
    out = src.apply_controls({"brightness": 1, "contrast": -1, "vflip": True})
    assert len(env.board.cmds) == n_before + 1
    assert env.board.cmds[-1] == {"brightness": 1, "contrast": -1, "vflip": True}
    assert all(r.verified is True for r in out.values())


def test_manual_gain_sent_together_with_auto_flag(src, env):
    src.set_control("gain_db", 20)
    assert env.board.cmds[-1] == {"auto_gain": True, "gain_db": 20.0}
    src.set_control("auto_gain", False)
    assert env.board.cmds[-1] == {"auto_gain": False, "gain_db": 20.0}
    src.set_control("auto_exposure", False)
    assert env.board.cmds[-1] == {"auto_exposure": False, "exposure_us": 10000}


def test_threshold_and_nms_live(src, env):
    assert src.set_control("threshold", 0.7).verified is True
    assert env.board.cmds[-1] == {"threshold": 0.7}


def test_unknown_and_invalid_control(src):
    assert not src.set_control("__no_such__", 1).ok
    assert not src.set_control("load_model", 1).ok
    assert not src.set_control("framesize", "HUGE").ok
    with pytest.raises(KeyError):
        src.get_control("nope")


def test_non_live_control_restarts_stream(src, env):
    src.set_control("brightness", 1)
    res = src.set_control("framesize", "VGA")
    assert res.ok and res.verified is True
    b = env.board
    assert b.starts == 2 and b.stops == 1 and b.params["framesize"] == "VGA"
    assert b.params["camera"]["brightness"] == 1   # 重啟後保留使用者調過的攝影機設定
    assert frame_of(src).image.shape == (480, 640, 3)


@pytest.mark.parametrize("key,value,check", [
    ("send_every", 3, lambda b: b.params["send_every"] == 3),
    ("jpeg_quality", 50, lambda b: b.params["quality"] == 50),
    ("pixformat", "gray", lambda b: b.params["pixformat"] == "gray"),
])
def test_other_restart_controls(src, env, key, value, check):
    res = src.set_control(key, value)
    assert res.ok and res.verified is True and check(env.board)
    frame_of(src)


def test_gray_frames_are_2d(src, env):
    src.set_control("pixformat", "gray")
    assert frame_of(src).image.ndim == 2


def test_controls_set_before_open_are_applied(env):
    s = make(env)
    s.set_control("framesize", "QQVGA")
    s.set_control("brightness", -1)
    s.open()
    assert env.board.params["framesize"] == "QQVGA" and env.board.params["camera"] == {"brightness": -1}
    s.close()


def test_restart_failure_reverts_value(src, env):
    env.board._start_script = lambda code: env.board._out.extend(b"OK\x04MemoryError\x04>")
    res = src.set_control("framesize", "VGA")
    assert not res.ok and src.get_control("framesize") == "QVGA"


# ---- 關閉時序 ---------------------------------------------------------------------

def test_close_timing_sequence(env, monkeypatch):
    s = make(env)
    s.open()
    b = env.board
    b.time_scale = 1.0
    s._repl.time_scale = 1.0
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda x: sleeps.append(round(x, 3)))
    monkeypatch.setattr(time, "time", lambda: s._started_at + 10)
    n = len(b.events)
    s.close()
    ev = b.events[n:]
    writes = [(e[1], e[2]) for e in ev if e[0] == "write"]
    assert writes[0] == (S.STREAM_BAUD, b"\x03")          # 串流鮑率下送 Ctrl+C
    assert writes[-1] == (S.REPL_BAUD, b"\r\x02")          # 切回 115200 後退出 raw REPL
    assert ev[-1] == ("close",)
    sleeps = [x for x in sleeps if x >= 0.1]   # 排除模擬序列埠讀不到資料時的短暫等待
    assert sleeps[:3] == [0.3, 0.5, 0.2]                   # 重送間隔、等鮑率切回、清緩衝前的等待


def test_close_waits_for_baud_switch_when_closed_too_early(env, monkeypatch):
    s = make(env)
    s.open()
    env.board.time_scale = 1.0
    s._repl.time_scale = 1.0
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda x: sleeps.append(round(x, 3)))
    monkeypatch.setattr(time, "time", lambda: s._started_at + 0.1)
    s.close()
    sleeps = [x for x in sleeps if x >= 0.1]
    assert sleeps[0] == pytest.approx(0.5, abs=0.01)   # 板子印出 STREAM_START 後 0.6 秒內不能送 Ctrl+C


def test_resends_ctrl_c_until_stream_stops(env, monkeypatch):
    s = make(env)
    s.open()
    b = env.board
    ctrl_c = []
    orig = b.write

    def lossy(data):   # 前兩次 Ctrl+C 被吃掉
        if data == b"\x03":
            ctrl_c.append(1)
            if len(ctrl_c) <= 2:
                return len(data)
        return orig(data)

    b.write = lossy
    s.close()
    assert len(ctrl_c) == 3 and b.stops == 1


def test_open_close_five_times(env):
    for _ in range(5):
        s = make(env)
        s.open()
        frame_of(s)
        s.close()
    assert len(env.boards) == 5
    assert all(b.starts == 1 and b.stops == 1 and not b.is_open for b in env.boards)


def test_reopen_same_object(env):
    s = make(env)
    for _ in range(3):
        s.open()
        assert frame_of(s) is not None
        s.close()


# ---- 錯誤訊息 ---------------------------------------------------------------------

def _raise_on_open(monkeypatch, exc):
    def boom(port, baud):
        raise exc
    monkeypatch.setattr(S, "open_serial", boom)


def test_error_port_busy(env, monkeypatch):
    monkeypatch.setattr(S.sys, "platform", "darwin")
    _raise_on_open(monkeypatch, OSError(16, "Resource busy"))
    with pytest.raises(SourceError) as e:
        make(env).open()
    assert "別的程式" in e.value.message and e.value.retryable and "關掉" in e.value.hint


def test_error_windows_access_denied_means_busy(env, monkeypatch):
    monkeypatch.setattr(S.sys, "platform", "win32")
    _raise_on_open(monkeypatch, PermissionError(13, "Access is denied"))
    with pytest.raises(SourceError) as e:
        make(env).open()
    assert "別的程式" in e.value.message


def test_error_linux_permission_mentions_dialout(env, monkeypatch):
    monkeypatch.setattr(S.sys, "platform", "linux")
    _raise_on_open(monkeypatch, PermissionError(13, "Permission denied"))
    with pytest.raises(SourceError) as e:
        make(env).open()
    assert "dialout" in e.value.hint and "權限" in e.value.message


def test_error_port_missing(env, monkeypatch):
    _raise_on_open(monkeypatch, FileNotFoundError(2, "No such file"))
    with pytest.raises(SourceError) as e:
        make(env).open()
    assert "找不到" in e.value.message and "USB" in e.value.hint


def test_error_board_not_responding(env):
    env.kwargs["no_response"] = True
    s = make(env)
    s.info.params["port"] = "sim:"
    import edgelens.sources.k210.repl as R

    orig = R.RawRepl.enter
    R.RawRepl.enter = lambda self, timeout=8: orig(self, timeout=0.05)
    try:
        with pytest.raises(SourceError) as e:
            s.open()
    finally:
        R.RawRepl.enter = orig
    assert "沒有回應" in e.value.message and "RESET" in e.value.hint and e.value.retryable
    assert not s.is_open and env.board.is_open is False   # 失敗後序列埠要釋放


def test_error_board_reports_missing_model(env):
    env.kwargs["fail_start"] = "Traceback (most recent call last):\r\nOSError: 找不到模型 /flash/x.kmodel"
    s = make(env)
    with pytest.raises(SourceError) as e:
        s.open()
    assert "找不到這個模型" in e.value.message and "/flash/x.kmodel" in e.value.message
    assert not s.is_open and env.board.is_open is False


def test_error_board_memory(env):
    env.kwargs["fail_start"] = "MemoryError: memory allocation failed"
    with pytest.raises(SourceError, match="記憶體"):
        make(env).open()


def test_error_start_timeout(env, monkeypatch):
    monkeypatch.setattr(S, "START_TIMEOUT", 0.2)
    env.kwargs["no_response"] = False
    s = make(env)
    monkeypatch.setattr(FakeK210Board, "_start_script", lambda self, code: self._out.extend(b"OK"))  # 板子收到程式卻不開始串流
    with pytest.raises(SourceError, match="沒有開始傳送"):
        s.open()
    assert not s.is_open


# ---- 探索與手動新增 ---------------------------------------------------------------

def test_discover_uses_likely_ports(monkeypatch):
    from edgelens.sources.k210 import ports

    monkeypatch.setattr(ports, "likely_ports", lambda: [
        ports.PortCandidate("/dev/cu.usbserial-1", "USB Serial", 0x1A86, 0x7523, location="20-1", chip="CH340",
                            likely=True)])
    infos = S.K210Source.discover()
    assert len(infos) == 1 and infos[0].id == "k210:1a86:7523:20-1" and infos[0].params["port"] == "/dev/cu.usbserial-1"
    json.dumps(infos[0].to_dict())


def test_discover_survives_errors(monkeypatch):
    from edgelens.sources.k210 import ports

    def boom():
        raise RuntimeError("x")

    monkeypatch.setattr(ports, "likely_ports", boom)
    assert S.K210Source.discover() == []


def test_info_from_params_is_stable(monkeypatch):
    from edgelens.sources.k210 import ports

    monkeypatch.setattr(ports, "list_serial_ports", lambda: [
        ports.PortCandidate("COM3", vid=0x1A86, pid=0x7523, serial_number="ABC", likely=True)])
    a = S.K210Source.info_from_params({"port": "COM3"})
    assert a.id == "k210:1a86:7523:ABC" == S.K210Source.info_from_params({"port": "COM3"}).id
    assert S.K210Source.info_from_params({"port": "COM9"}).id == "k210:manual:COM9"
    assert S.K210Source.info_from_params({}).id == "k210:auto"


def test_resolve_port_follows_stable_id(monkeypatch):
    from edgelens.sources.k210 import ports

    monkeypatch.setattr(ports, "list_serial_ports", lambda: [
        ports.PortCandidate("COM7", vid=0x1A86, pid=0x7523, serial_number="ABC", likely=True)])
    s = S.K210Source(SourceInfo(kind="k210", id="k210:1a86:7523:ABC", name="x", params={"port": "COM3"}))
    assert s._resolve_port() == "COM7"


def test_resolve_port_auto(monkeypatch):
    from edgelens.sources.k210 import ports

    one = ports.PortCandidate("COM3", vid=0x1A86, pid=0x7523, likely=True)
    two = ports.PortCandidate("COM4", vid=0x1A86, pid=0x7523, likely=True)
    s = S.K210Source(SourceInfo(kind="k210", id="k210:auto", name="x", params={}))
    monkeypatch.setattr(ports, "list_serial_ports", lambda: [one])
    assert s._resolve_port() == "COM3"
    monkeypatch.setattr(ports, "list_serial_ports", lambda: [])
    with pytest.raises(SourceError, match="找不到"):
        s._resolve_port()
    monkeypatch.setattr(ports, "list_serial_ports", lambda: [one, two])
    with pytest.raises(SourceError, match="COM4"):
        s._resolve_port()


def test_param_specs():
    specs = {s.key: s for s in S.K210Source.param_specs()}
    assert specs["port"].type == ControlType.STR and specs["baud"].default == 115200


def test_contract_fixture_works_without_hardware(tmp_path):
    info = S.K210Source.contract_fixture(tmp_path)
    s = S.K210Source(info)
    s.open()
    try:
        assert frame_of(s) is not None
    finally:
        s.close()


def test_script_upload_is_stripped_and_valid():
    import ast

    code = S.board_script()
    ast.parse(code)
    assert "STREAM_START" in code and "\n\n" not in code


def test_set_controls_merges_live_and_restarts_others(src, env):
    """GUI 一次送多個參數時：即時參數合併成一條 CMD，需要重啟的參數另外處理。"""
    n_before = len(env.board.cmds)
    out = src.set_controls({"brightness": 2, "saturation": 1, "framesize": "QQVGA"})
    assert env.board.cmds[n_before] == {"brightness": 2, "saturation": 1}
    assert set(out) == {"brightness", "saturation", "framesize"}
    assert out["brightness"].verified is True and out["framesize"].ok
