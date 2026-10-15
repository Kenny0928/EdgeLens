"""卡住偵測：只在卡住時記一次、恢復時記一行；被 SIGTERM 結束前留下堆疊。"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from edgelens.core import hangwatch
from edgelens.core.hangwatch import HANG_FILE, HangWatch, start_hang_watch

ROOT = Path(__file__).resolve().parents[2]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_dumps_once_per_stall_and_logs_recovery(tmp_path):
    clock = Clock()
    path = tmp_path / "hang.txt"
    with open(path, "w", encoding="utf-8") as out:
        w = HangWatch(out, stall_s=15, clock=clock)
        clock.t = 10
        assert not w.check()
        clock.t = 16
        assert w.check()
        clock.t = 30
        assert not w.check()            # 同一次卡住只記一次
        clock.t = 31
        w.beat()
        clock.t = 40
        assert not w.check()
        clock.t = 47
        assert w.check()                # 恢復後再卡住，會再記
    text = path.read_text(encoding="utf-8")
    assert text.count("沒有回應") == 2
    assert "恢復回應（卡了約 31 秒）" in text
    assert "test_dumps_once_per_stall_and_logs_recovery" in text   # 堆疊裡看得到目前的函式


def test_no_log_dir_means_no_watch():
    assert start_hang_watch(None) is None


def test_oversized_hang_file_is_cleared(tmp_path):
    (tmp_path / HANG_FILE).write_bytes(b"x" * (hangwatch.MAX_HANG_BYTES + 1))
    with hangwatch._open_hang_file(tmp_path) as f:
        f.write("new\n")
    assert (tmp_path / HANG_FILE).read_text(encoding="utf-8") == "new\n"


@pytest.mark.skipif(sys.platform == "win32", reason="Windows 沒有 SIGTERM 的 faulthandler.register")
def test_sigterm_leaves_stack_in_log(tmp_path):
    """模擬「程式卡住、被強制結束」：結束碼是 -15（啟動器顯示 241），日誌裡要看得到卡在哪個函式。"""
    code = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(ROOT)!r})
        from edgelens.core.hangwatch import start_hang_watch
        start_hang_watch(__import__("pathlib").Path({str(tmp_path)!r}))
        def import_pack_stuck_here():
            print("ready", flush=True)
            time.sleep(60)
        import_pack_stuck_here()
    """)
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "ready"
        os.kill(proc.pid, signal.SIGTERM)
        assert proc.wait(timeout=10) == -signal.SIGTERM
    finally:
        proc.kill()
    text = (tmp_path / HANG_FILE).read_text(encoding="utf-8")
    assert "import_pack_stuck_here" in text
