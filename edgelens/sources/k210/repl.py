"""MaixPy raw REPL 控制與檔案上傳（移植自舊版 k210.py 的 RawRepl）。

序列埠物件由外部注入（pyserial 的 Serial 或測試用的假序列埠），只需要
write、read、reset_input_buffer、close，以及 baudrate／timeout 屬性。
"""

from __future__ import annotations

import binascii
import hashlib
import time
from collections.abc import Callable
from typing import Any

REPL_BAUD = 115200
CHUNK = 256         # MaixPy UART 緩衝不大，分小塊送出
PUT_BLOCK = 384     # 每次 f.write 的原始位元組數（base64 後約 512 字元）

ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]


class ReplError(Exception):
    """板子端執行程式時回報的錯誤（message 是板子印出的錯誤文字）。"""


class ReplCancelled(Exception):
    """使用者取消了上傳。"""


class RawRepl:
    def __init__(self, ser: Any):
        self.ser = ser
        self.buf = b""  # 已收到但尚未處理的資料
        self.time_scale = 1.0  # 等待時間的倍率；只有模擬板子（sim）會調小以加快測試

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds * self.time_scale)

    # ---- 讀取 ------------------------------------------------------------------

    def read_until(self, token: bytes, timeout: float = 10) -> bytes:
        """讀到 token 為止，回傳含 token 的資料，多讀的部分留在 self.buf。"""
        _, data = self.read_until_any((token,), timeout)
        return data

    def read_until_any(self, tokens: tuple[bytes, ...], timeout: float = 10) -> tuple[bytes, bytes]:
        """讀到任何一個 token 為止，回傳 (先出現的 token, 含 token 的資料)。"""
        end = time.time() + timeout
        while True:
            best: tuple[int, bytes] | None = None
            for tok in tokens:
                i = self.buf.find(tok)
                if i >= 0 and (best is None or i < best[0]):
                    best = (i, tok)
            if best is not None:
                cut = best[0] + len(best[1])
                data, self.buf = self.buf[:cut], self.buf[cut:]
                return best[1], data
            if time.time() > end:
                raise TimeoutError(f"等待 {tokens!r} 逾時，目前收到：{self.buf[-200:]!r}")
            self.buf += self.ser.read(256)

    # ---- 進入與離開 ------------------------------------------------------------

    def enter(self, timeout: float = 8) -> None:
        """中斷正在執行的程式，進入 raw REPL。

        MaixPy 中斷程式需要一點時間，持續送 Ctrl+C 直到看見 ">>> "；
        Ctrl+B 讓停在 raw REPL（提示符號只有 ">"）的板子回到一般 REPL。
        """
        seen = b""
        end = time.time() + timeout
        while b">>> " not in seen:
            if time.time() > end:
                raise TimeoutError("板子沒有回應 Ctrl+C，請按板子 RST 鍵後再試")
            self.ser.write(b"\r\x03\x02")
            self.sleep(0.3)
            seen = (seen + self.ser.read(4096))[-64:]
        self.sleep(0.2)
        self.ser.reset_input_buffer()
        self.buf = b""
        self.ser.write(b"\r\x01")
        self.read_until(b"raw REPL; CTRL-B to exit\r\n>")

    def exit(self) -> None:
        self.ser.write(b"\r\x02")

    def close(self) -> None:
        self.ser.close()

    # ---- 執行程式 --------------------------------------------------------------

    def send_code(self, code: str, wait_ok: bool = True) -> None:
        """送出程式碼並等板子回 OK（表示已編譯完成、開始執行）。
        wait_ok=False 用在會讓板子重開機的指令：板子可能來不及回 OK。"""
        data = code.encode("utf-8")
        for i in range(0, len(data), CHUNK):
            self.ser.write(data[i:i + CHUNK])
            self.sleep(0.01)
        self.ser.write(b"\x04")
        if wait_ok:
            self.read_until(b"OK")

    def exec(self, code: str, timeout: float = 10) -> str:
        """執行並回傳 stdout；板子回報錯誤時丟 ReplError。"""
        self.send_code(code)
        out = self.read_until(b"\x04", timeout)[:-1]
        err = self.read_until(b"\x04", timeout)[:-1]
        self.read_until(b">", timeout)
        if err:
            raise ReplError(err.decode("utf-8", "replace"))
        return out.decode("utf-8", "replace")

    # ---- 檔案 ------------------------------------------------------------------

    def stat_size(self, remote: str) -> int | None:
        """板上檔案大小；檔案不存在時回傳 None。"""
        _check_path(remote)
        out = self.exec(f"import os\ntry:\n    print('SIZE', os.stat({remote!r})[6])\nexcept OSError:\n    print('SIZE -1')")
        for line in out.splitlines():
            if line.startswith("SIZE "):
                n = int(line.split()[1])
                return None if n < 0 else n
        raise ReplError("無法讀取檔案大小：" + out)

    def remote_sha256(self, remote: str, timeout: float = 120) -> str | None:
        """在板上計算 SHA-256；韌體沒有 uhashlib 時回傳 None。"""
        _check_path(remote)
        code = (
            "try:\n"
            "    import uhashlib, ubinascii\n"
            "    h = uhashlib.sha256()\n"
            f"    f = open({remote!r}, 'rb')\n"
            "    while True:\n"
            "        b = f.read(2048)\n"
            "        if not b:\n"
            "            break\n"
            "        h.update(b)\n"
            "    f.close()\n"
            "    print('SHA', ubinascii.hexlify(h.digest()).decode())\n"
            "except ImportError:\n"
            "    print('SHA none')\n"
        )
        out = self.exec(code, timeout=timeout)
        for line in out.splitlines():
            if line.startswith("SHA "):
                val = line.split()[1]
                return None if val == "none" else val
        return None

    def put(self, data: bytes, remote: str, progress: ProgressFn | None = None,
            cancel: CancelFn | None = None, verify_hash: bool = True) -> None:
        """base64 分段上傳 data 到板上 remote，上傳後在板上確認大小（與雜湊）一致。

        進度以實際寫入的位元組計（0–0.9 為傳輸，之後為驗證）。取消時關閉並刪除半成品，
        丟 ReplCancelled。大小或雜湊不一致丟 ReplError。
        """
        _check_path(remote)
        total = len(data)

        def report(frac: float) -> None:
            if progress:
                progress(min(max(frac, 0.0), 1.0), remote)

        report(0.0)
        self.exec(f"import ubinascii, os\nf = open({remote!r}, 'wb')")
        try:
            for i in range(0, total, PUT_BLOCK):
                if cancel and cancel():
                    raise ReplCancelled()
                b64 = binascii.b2a_base64(data[i:i + PUT_BLOCK]).decode().strip()
                self.exec(f"f.write(ubinascii.a2b_base64({b64!r}))")
                report(0.9 * min(total, i + PUT_BLOCK) / total)
        except ReplCancelled:
            try:
                self.exec(f"f.close()\nos.remove({remote!r})")
            except Exception:  # noqa: BLE001 - 取消時盡力清理，失敗也不要蓋掉取消
                pass
            raise
        self.exec("f.close()")
        report(0.92)
        size = self.stat_size(remote)
        if size != total:
            raise ReplError(f"上傳後大小不一致：本機 {total} bytes，板上 {size} bytes")
        if verify_hash:
            remote_hash = self.remote_sha256(remote)
            if remote_hash is not None and remote_hash != hashlib.sha256(data).hexdigest():
                raise ReplError("上傳後內容檢查碼不一致（可能傳輸出錯），請重新上傳")
        report(1.0)


def _check_path(remote: str) -> None:
    if not remote.startswith("/") or "\n" in remote or "\r" in remote:
        raise ValueError(f"板上路徑必須以 / 開頭，例如 /flash/model.kmodel：{remote!r}")
