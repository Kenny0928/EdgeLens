"""raw REPL 與檔案上傳：用模擬板子（sim.FakeK210Board）和假序列埠測試。"""

from __future__ import annotations

import hashlib
import time

import pytest

from edgelens.sources.k210.repl import RawRepl, ReplCancelled, ReplError
from edgelens.sources.k210.sim import FakeK210Board


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)


def make(**kw):
    board = FakeK210Board(**kw)
    repl = RawRepl(board)
    repl.enter()
    return board, repl


def test_enter_sends_ctrl_c_until_prompt_then_raw_mode():
    board, repl = make()
    writes = [e[2] for e in board.events if e[0] == "write"]
    assert writes[0] == b"\r\x03\x02"
    assert b"\r\x01" in writes
    assert board._mode == "raw"


def test_enter_times_out_without_response(monkeypatch):
    board = FakeK210Board(no_response=True)
    repl = RawRepl(board)
    clock = iter(range(0, 1000))
    monkeypatch.setattr(time, "time", lambda: next(clock))
    with pytest.raises(TimeoutError, match="RST"):
        repl.enter(timeout=8)


def test_exec_returns_stdout_and_raises_on_stderr():
    _, repl = make()
    assert repl.stat_size("/flash/none") is None
    # 板子端錯誤：用 sim 的 OSError 路徑（寫到沒有 SD 的 /sd）
    with pytest.raises(ReplError, match="ENODEV"):
        repl.exec("f = open('/sd/x', 'wb')")


def test_exit_writes_ctrl_b():
    board, repl = make()
    repl.exit()
    assert board.events[-1][2] == b"\r\x02"


def test_put_uploads_and_verifies_size_and_hash():
    board, repl = make()
    data = bytes(range(256)) * 10 + b"tail"
    seen = []
    repl.put(data, "/flash/m.kmodel", progress=lambda f, t: seen.append(f))
    assert bytes(board.files["/flash/m.kmodel"]) == data
    assert seen[0] == 0.0 and seen[-1] == 1.0
    assert seen == sorted(seen)
    assert 0 < seen[1] < 0.5   # 進度依實際寫入量，不是一次跳到底
    assert repl.remote_sha256("/flash/m.kmodel") == hashlib.sha256(data).hexdigest()


def test_put_progress_is_proportional():
    board, repl = make()
    data = b"x" * 3840  # 10 個 384 bytes 區塊
    seen = []
    repl.put(data, "/flash/a.bin", progress=lambda f, t: seen.append(round(f, 3)))
    transfer = [f for f in seen if f <= 0.9]
    assert transfer[1:] == pytest.approx([0.09 * i for i in range(1, 11)], abs=1e-3)


def test_put_without_uhashlib_still_checks_size():
    board, repl = make(has_sha=False)
    repl.put(b"abc" * 500, "/flash/a.bin")
    assert repl.remote_sha256("/flash/a.bin") is None


def test_put_detects_size_mismatch():
    board, repl = make()
    board.corrupt_upload = True
    with pytest.raises(ReplError, match="大小不一致"):
        repl.put(b"z" * 1000, "/flash/bad.bin")


def test_put_detects_hash_mismatch(monkeypatch):
    board, repl = make()
    monkeypatch.setattr(RawRepl, "remote_sha256", lambda self, remote, timeout=120: "0" * 64)
    with pytest.raises(ReplError, match="檢查碼"):
        repl.put(b"hello", "/flash/h.bin")


def test_put_cancel_removes_partial_file():
    board, repl = make()
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 3

    with pytest.raises(ReplCancelled):
        repl.put(b"q" * 5000, "/flash/c.bin", cancel=cancel)
    assert "/flash/c.bin" not in board.files


def test_put_empty_file():
    board, repl = make()
    repl.put(b"", "/flash/empty.bin")
    assert bytes(board.files["/flash/empty.bin"]) == b""


def test_put_to_sd_without_card_fails():
    board, repl = make(sd=False)
    with pytest.raises(ReplError):
        repl.put(b"x", "/sd/m.kmodel")


@pytest.mark.parametrize("bad", ["relative.bin", "/flash/a\nb"])
def test_bad_remote_path_rejected(bad):
    _, repl = make()
    with pytest.raises(ValueError):
        repl.put(b"x", bad)


def test_read_until_any_returns_first_token():
    class Ser:
        def __init__(self):
            self.data = [b"abc STREAM", b"_START rest\x04"]

        def read(self, n):
            return self.data.pop(0) if self.data else b""

    repl = RawRepl(Ser())
    tok, data = repl.read_until_any((b"STREAM_START", b"\x04"), timeout=2)
    assert tok == b"STREAM_START" and data == b"abc STREAM_START"
    assert repl.buf == b" rest\x04"
