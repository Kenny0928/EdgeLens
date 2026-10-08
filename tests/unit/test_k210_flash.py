"""韌體燒錄：假的 kflash 子行程，不碰序列埠。"""

from __future__ import annotations

import sys
import threading
import time

import pytest

from edgelens.sources.k210 import flash

OUT = (b"[INFO] Trying to Enter the ISP Mode...\n"
       b"Downloading ISP: |=====| 50.0% 10kiB/s\rDownloading ISP: |=====| 100.0% 10kiB/s\n"
       b"[INFO] Greeting Message Detected\n"
       b"Programming BIN: |=====| 25.0% 20kiB/s\rProgramming BIN: |=====| 100.0% 20kiB/s\n"
       b"[INFO] Rebooting...\n")


class FakeStdout:
    def __init__(self, data: bytes, block_until: threading.Event | None = None):
        self.data, self.pos, self.block = data, 0, block_until

    def read(self, n):
        if self.block is not None and self.pos >= len(self.data):
            self.block.wait(5)
            return b""
        if self.pos >= len(self.data):
            return b""
        b = self.data[self.pos:self.pos + n]
        self.pos += n
        return b


class FakeProc:
    def __init__(self, data=OUT, rc=0, block=False):
        self.terminated = threading.Event()
        self.stdout = FakeStdout(data, self.terminated if block else None)
        self.rc = rc

    def terminate(self):
        self.terminated.set()
        self.rc = -15

    def kill(self):
        self.terminate()

    def wait(self, timeout=None):
        return self.rc


@pytest.fixture
def fw(tmp_path, monkeypatch):
    monkeypatch.setattr(flash, "kflash_available", lambda: True)
    p = tmp_path / "fw.bin"
    p.write_bytes(b"\x00" * 16)
    return str(p)


def test_build_command_uses_current_python_and_slow_mode():
    cmd = flash.build_command("/dev/cu.usbserial-1", "fw.bin")
    assert cmd[:3] == [sys.executable, "-m", "kflash"]
    assert "-S" in cmd and cmd[cmd.index("-B") + 1] == "maixduino" and cmd[cmd.index("-b") + 1] == "1500000"
    assert cmd[cmd.index("-p") + 1] == "/dev/cu.usbserial-1" and cmd[-1] == "fw.bin"
    assert "-S" not in flash.build_command("p", "f", slow=False)


@pytest.mark.parametrize("line,expect", [
    ("Downloading ISP: |===| 50.0% 10kiB/s", 0.025),
    ("Downloading ISP: |===| 100.0%", 0.05),
    ("Programming BIN: |===| 0.0%", 0.05),
    ("Programming BIN: |===| 100.0%", 1.0),
    ("[INFO] hello", None),
    ("50% of something else", None),
])
def test_parse_progress(line, expect):
    got = flash.parse_progress(line)
    assert got == pytest.approx(expect) if expect is not None else got is None


def test_flash_success_reports_progress(fw):
    seen = []
    res = flash.flash_firmware("p", fw, popen=lambda *a, **k: FakeProc(),
                               progress=lambda f, t: seen.append(f))
    assert res.ok and res.returncode == 0
    assert seen == sorted(seen) and seen[-1] == 1.0 and 0.5 < max(x for x in seen[:-1]) <= 1.0
    assert any("Rebooting" in ln for ln in res.output)


def test_flash_failure_translated(fw):
    out = b"[ERROR] Failed to find ISP greeting\n"
    res = flash.flash_firmware("p", fw, popen=lambda *a, **k: FakeProc(out, rc=1))
    assert not res.ok and res.returncode == 1 and "USB" in res.message


def test_flash_permission_error_message(fw):
    out = b"PermissionError: [Errno 13] Permission denied: '/dev/ttyUSB0'\n"
    res = flash.flash_firmware("p", fw, popen=lambda *a, **k: FakeProc(out, rc=1))
    assert "權限" in res.message


def test_flash_cancel_terminates_process(fw):
    proc = FakeProc(b"[INFO] start\n", block=True)
    t0 = time.time()
    res = flash.flash_firmware("p", fw, popen=lambda *a, **k: proc, cancel=lambda: True)
    assert res.cancelled and not res.ok and proc.terminated.is_set() and time.time() - t0 < 3
    assert "不完整" in res.message


def test_flash_without_kflash_explains_install(fw, monkeypatch):
    monkeypatch.setattr(flash, "kflash_available", lambda: False)
    res = flash.flash_firmware("p", fw, popen=lambda *a, **k: pytest.fail("不應啟動"))
    assert not res.ok and "pip install kflash" in res.message


def test_flash_missing_firmware_file(monkeypatch):
    monkeypatch.setattr(flash, "kflash_available", lambda: True)
    res = flash.flash_firmware("p", "/no/such.bin", popen=lambda *a, **k: pytest.fail("不應啟動"))
    assert not res.ok and ".bin" in res.message


def test_flash_popen_oserror(fw):
    def boom(*a, **k):
        raise OSError("nope")

    res = flash.flash_firmware("p", fw, popen=boom)
    assert not res.ok and "nope" in res.message
