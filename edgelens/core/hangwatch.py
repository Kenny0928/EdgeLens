"""卡住時留下證據：主執行緒太久沒回應、或程式被強制結束時，把每個執行緒正在跑哪一行寫進日誌。

學生遇到「畫面卡住、只能強制關閉」時，一般日誌什麼都不會留下（卡住的程式寫不了日誌）。
這裡用 faulthandler 從背景執行緒直接把堆疊寫到檔案，診斷匯出（edgelens.log*）會一起打包。

    watch = start_hang_watch(log_dir)      # GUI 啟動後呼叫；回傳的物件要在主執行緒定時呼叫 beat()
    timer.timeout.connect(watch.beat)

只用標準函式庫，不依賴 Qt，邏輯可以用假時鐘測試。
"""

from __future__ import annotations

import faulthandler
import logging
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO

log = logging.getLogger(__name__)

HANG_FILE = "edgelens.log.hang"   # 名稱以 edgelens.log 開頭，診斷匯出才會一起打包；輪替不會動到它
MAX_HANG_BYTES = 1_000_000        # 啟動時超過就清空，不讓它越積越大
STALL_S = 15.0                    # 主執行緒超過這麼久沒心跳才算卡住（開大檔、載入模型都不會這麼久）
CHECK_S = 2.0


class HangWatch:
    """主執行緒定時呼叫 beat()；check() 在背景執行緒判斷是否卡住。每次卡住只記一次，恢復時再記一行。"""

    def __init__(self, out: IO[str], stall_s: float = STALL_S, clock: Callable[[], float] = time.monotonic):
        self.out = out
        self.stall_s = stall_s
        self.clock = clock
        self._last = clock()
        self._stalled_since: float | None = None

    def beat(self) -> None:
        now = self.clock()
        if self._stalled_since is not None:
            self._write(f"主執行緒恢復回應（卡了約 {now - self._stalled_since:.0f} 秒）\n")
            self._stalled_since = None
        self._last = now

    def check(self) -> bool:
        """卡住且這次還沒記錄過就寫下所有執行緒的堆疊，回傳是否寫了。"""
        now = self.clock()
        if self._stalled_since is not None or now - self._last < self.stall_s:
            return False
        self._stalled_since = self._last
        self._write(f"主執行緒 {now - self._last:.0f} 秒沒有回應，各執行緒目前位置：\n")
        try:
            faulthandler.dump_traceback(file=self.out, all_threads=True)
        except (OSError, ValueError, RuntimeError) as e:
            log.debug("寫入卡住堆疊失敗：%s", e)
        return True

    def _write(self, text: str) -> None:
        try:
            self.out.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}")
            self.out.flush()
        except (OSError, ValueError):
            pass

    def run_forever(self, interval: float = CHECK_S) -> None:
        while True:
            time.sleep(interval)
            self.check()


def _open_hang_file(log_dir: Path) -> IO[str]:
    path = Path(log_dir) / HANG_FILE
    try:
        if path.stat().st_size > MAX_HANG_BYTES:
            path.unlink()
    except OSError:
        pass
    return open(path, "a", encoding="utf-8")   # noqa: SIM115 - 要一直開著給 faulthandler 寫


def start_hang_watch(log_dir: Path | None) -> HangWatch | None:
    """開始監看；log_dir 是 None（日誌寫不了）或任何錯誤時回傳 None，絕不影響啟動。"""
    if log_dir is None:
        return None
    try:
        out = _open_hang_file(log_dir)
        watch = HangWatch(out)
        # 被 SIGTERM 結束（例如從「活動監視器」結束、或卡住後被關掉；啟動器會顯示代碼 241）前先留下堆疊。
        # Windows 沒有 faulthandler.register。chain=True 讓程式照常結束。
        if hasattr(faulthandler, "register") and sys.platform != "win32":
            faulthandler.register(signal.SIGTERM, file=out, all_threads=True, chain=True)
        threading.Thread(target=watch.run_forever, name="edgelens-hangwatch", daemon=True).start()
        return watch
    except Exception:  # noqa: BLE001 - 診斷工具失敗不能擋住程式
        log.exception("無法啟動卡住偵測")
        return None
