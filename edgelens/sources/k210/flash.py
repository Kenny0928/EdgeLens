"""燒錄 MaixPy 韌體：用子行程執行 `python -m kflash`。

kflash 是選配套件，所以只在這裡用子行程呼叫，不 import；不寫死 kflash 路徑，
用 sys.executable 才會用到和 EdgeLens 同一個 Python 環境。燒錄前呼叫端必須先關閉串流
（序列埠同時只能被一個程式開著）。
"""

from __future__ import annotations

import logging
import os
import queue
import re
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from edgelens.core.registry import module_available
from edgelens.i18n import tr

log = logging.getLogger(__name__)

FLASH_BAUD = 1500000
DEFAULT_BOARD = "maixduino"
BOARDS = ("maixduino", "goE", "dan", "bit", "kd233", "trainer")  # kflash -B 的常用選項

ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]

_PCT = re.compile(r"(\d+(?:\.\d+)?)%")
# kflash 先傳 ISP 小程式（約佔前 5%），再寫入韌體本體
_PHASES = (("Downloading ISP", 0.0, 0.05), ("Programming BIN", 0.05, 1.0))


@dataclass
class FlashResult:
    ok: bool
    message: str = ""
    returncode: int | None = None
    cancelled: bool = False
    output: list[str] = field(default_factory=list)


def kflash_available() -> bool:
    return module_available("kflash")


def build_command(port: str, firmware: str, board: str = DEFAULT_BOARD, baud: int = FLASH_BAUD,
                  slow: bool = True) -> list[str]:
    """kflash 參數。-S（慢速）是舊版實測的結果：macOS 用高速模式會在 ISP 階段斷線；
    -n 關掉彩色控制碼，輸出才好解析。"""
    cmd = [sys.executable, "-m", "kflash"]
    if slow:
        cmd.append("-S")
    cmd += ["-n", "-p", port, "-b", str(baud), "-B", board, firmware]
    return cmd


def parse_progress(line: str) -> float | None:
    """從 kflash 輸出的一行取出整體進度（0–1）；不是進度行回傳 None。"""
    m = _PCT.search(line)
    if not m:
        return None
    pct = min(max(float(m.group(1)) / 100.0, 0.0), 1.0)
    for name, lo, hi in _PHASES:
        if name in line:
            return lo + (hi - lo) * pct
    return None


def explain_failure(output: str) -> str:
    """把 kflash 的英文錯誤換成學生看得懂的說明。"""
    low = output.lower()
    if "permission" in low or "access is denied" in low:
        return tr("source.k210.flash.err.permission")
    if "busy" in low or ("resource" in low and "unavailable" in low):
        return tr("source.k210.flash.err.busy")
    if "no such file" in low or "not found" in low:
        return tr("source.k210.flash.err.nofile")
    if "could not open port" in low or "could not configure port" in low or "serialexception" in low:
        return tr("source.k210.flash.err.port")
    if "timeout" in low or "failed to" in low or "no response" in low or "isp" in low:
        return tr("source.k210.flash.err.isp")
    return tr("source.k210.flash.err.generic")


def flash_firmware(port: str, firmware: str, *, board: str = DEFAULT_BOARD, baud: int = FLASH_BAUD,
                   slow: bool = True, progress: ProgressFn | None = None, cancel: CancelFn | None = None,
                   popen: Callable[..., subprocess.Popen] = subprocess.Popen) -> FlashResult:
    """燒錄韌體。progress(比例, 說明)；cancel() 為 True 時終止子行程。

    成功與否看結束碼；燒錄到一半取消會讓板子停在不完整的韌體，需要重新燒錄，
    所以取消的訊息會提醒使用者。"""
    if not kflash_available():
        return FlashResult(False, tr("source.k210.flash.need_kflash"))
    if not os.path.isfile(firmware):
        return FlashResult(False, tr("source.k210.flash.err.nofile"))
    cmd = build_command(port, firmware, board, baud, slow)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    kwargs: dict = {}
    if os.name == "nt":  # 不要彈出黑色視窗
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    log.info("燒錄韌體：%s", " ".join(cmd))
    try:
        proc = popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, **kwargs)
    except OSError as e:
        return FlashResult(False, tr("source.k210.flash.err.start", error=str(e)))

    q: queue.Queue[bytes | None] = queue.Queue()

    def reader() -> None:
        try:
            while True:
                chunk = proc.stdout.read(1) if proc.stdout else b""
                if not chunk:
                    break
                q.put(chunk)
        finally:
            q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    lines: list[str] = []
    cur = bytearray()
    cancelled = False
    done = False

    def emit_line() -> None:
        if not cur:
            return
        line = bytes(cur).decode("utf-8", "replace").strip()
        cur.clear()
        if not line:
            return
        lines.append(line)
        frac = parse_progress(line)
        if frac is not None:
            last[0] = frac
        if progress and (frac is not None or "[INFO]" in line):
            progress(last[0], line)

    last = [0.0]  # 最近一次的進度；沒有百分比的 [INFO] 行沿用它
    while not done:
        if cancel and cancel() and not cancelled:
            cancelled = True
            proc.terminate()
        try:
            item = q.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            done = True
        elif item in (b"\r", b"\n"):
            emit_line()
        else:
            cur += item
    emit_line()
    try:
        rc = proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = proc.wait()
    if cancelled:
        return FlashResult(False, tr("source.k210.flash.cancelled"), rc, True, lines)
    if rc != 0:
        return FlashResult(False, explain_failure("\n".join(lines[-30:])), rc, False, lines)
    if progress:
        progress(1.0, "done")
    return FlashResult(True, tr("source.k210.flash.done"), rc, False, lines)
