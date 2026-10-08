"""模擬的 K210 板子：外觀像序列埠，行為像執行 board/stream_v1.py 的 MaixPy 板子。

用途：單元測試與 contract tests（不需要硬體）、沒有開發板時的示範。連接埠名稱以
"sim:" 開頭時，K210Source 會改用這個假板子。它只模擬「通訊行為」（raw REPL、CAPS、
STREAM_START、封包、CMD 與 ack、檔案上傳、Ctrl+C 停止與鮑率切換），不跑 MicroPython，
所以板上程式本身（感光元件、KPU）仍要用實體板子驗證。
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import re
import time
from typing import Any

from edgelens.sources.k210.protocol import encode_packet

STREAM_BAUD = 1500000
REPL_BAUD = 115200
_SIZES = {"QQVGA": (160, 120), "QVGA": (320, 240), "VGA": (640, 480)}
_CMD_KEYS = {"brightness", "contrast", "saturation", "auto_gain", "gain_db", "auto_exposure",
             "exposure_us", "auto_whitebal", "vflip", "hmirror", "threshold", "nms"}


class FakeK210Board:
    """序列埠物件（write／read／reset_input_buffer／close／baudrate／timeout）。

    測試用的開關：
      frames_per_read  每次 read() 產生幾個封包（>1 可測「只回傳最新」）
      objs／cls        之後封包要帶的偵測／分類結果
      reject           CMD 中這些鍵會回報失敗
      ack_never        不回 ack
      no_response      完全不回應（測試「板子沒有回應」）
      fail_start       P 啟動時丟這段錯誤文字（測試啟動失敗）
      has_sha          韌體是否有 uhashlib
      corrupt_upload   上傳時少寫最後一個區塊（測試大小驗證）
      events           依序記錄 ("write", baud, bytes) 等事件，測試檢查時序用
    """

    def __init__(self, *, frames_per_read: int = 1, has_sha: bool = True, sd: bool = False,
                 no_response: bool = False, fail_start: str = "") -> None:
        self.baudrate = REPL_BAUD
        self.timeout = 0.1
        self.is_open = True
        self.frames_per_read = frames_per_read
        self.has_sha = has_sha
        self.sd = sd
        self.no_response = no_response
        self.fail_start = fail_start
        self.reject: set[str] = set()
        self.ack_never = False
        self.corrupt_upload = False
        self.objs: list[list[Any]] = []
        self.cls: list[list[Any]] = []
        self.files: dict[str, bytearray] = {}
        self.freq_calls: list[str] = []
        self.rebooted = 0
        self.params: dict[str, Any] = {}
        self.events: list[tuple[Any, ...]] = []
        self.cmds: list[dict[str, Any]] = []
        self.starts = 0
        self.stops = 0
        self._mode = "normal"        # normal／raw／stream
        self._out = bytearray()
        self._late = bytearray()     # 要等主機切回 115200 才讀得到的資料
        self._code = bytearray()
        self._cmdbuf = b""
        self._cur: str | None = None
        self._seq = 0
        self._cfg: dict[str, Any] = {}
        self._ack: dict[str, Any] = {"n": 0, "id": 0, "ok": [], "err": []}
        self._jpeg_cache: dict[tuple[int, int, bool], bytes] = {}

    # ---- 序列埠介面 ------------------------------------------------------------

    @property
    def in_waiting(self) -> int:
        return len(self._out)

    def reset_input_buffer(self) -> None:
        self._out.clear()

    def close(self) -> None:
        self.is_open = False
        self.events.append(("close",))

    def write(self, data: bytes) -> int:
        data = bytes(data)
        self.events.append(("write", self.baudrate, data))
        if self.no_response:
            return len(data)
        if self._mode == "stream":
            self._write_stream(data)
        elif self._mode == "raw":
            self._write_raw(data)
        else:
            self._write_normal(data)
        return len(data)

    def read(self, size: int = 1) -> bytes:
        if self.no_response:
            time.sleep(0.002)
            return b""
        if self.baudrate == REPL_BAUD and self._late:
            self._out += self._late
            self._late.clear()
        if self._mode == "stream" and self.baudrate == STREAM_BAUD and not self._out:
            for _ in range(self.frames_per_read):
                self._out += self._packet()
        if not self._out:
            time.sleep(0.002)
            return b""
        data = bytes(self._out[:size])
        del self._out[:size]
        return data

    # ---- 一般 REPL ------------------------------------------------------------

    def _write_normal(self, data: bytes) -> None:
        for b in data:
            if b == 3:
                self._out += b"\r\nKeyboardInterrupt\r\n>>> "
            elif b == 1:
                self._mode = "raw"
                self._code.clear()
                self._out += b"\r\nraw REPL; CTRL-B to exit\r\n>"
            elif b == 13:
                self._out += b"\r\n>>> "

    # ---- raw REPL ------------------------------------------------------------

    def _write_raw(self, data: bytes) -> None:
        for b in data:
            if b == 4:
                code = self._code.decode("utf-8", "replace")
                self._code.clear()
                self._execute(code)
                if self._mode != "raw":
                    return
            elif b == 2 and not self._code:
                self._mode = "normal"
                self._out += b"\r\nMicroPython (simulated)\r\n>>> "
            elif b in (3, 13) and not self._code:
                continue
            else:
                self._code.append(b)

    def _reply(self, stdout: str = "", stderr: str = "") -> None:
        self._out += b"OK" + stdout.encode() + b"\x04" + stderr.encode() + b"\x04>"

    def _execute(self, code: str) -> None:
        if code.startswith("P = "):
            self._start_script(code)
            return
        files = self.files
        m = re.search(r"open\((['\"])(.*?)\1, 'wb'\)", code)
        if m:
            path = m.group(2)
            if path.startswith("/sd/") and not self.sd:
                return self._reply(stderr="OSError: [Errno 19] ENODEV")
            files[path] = bytearray()
            self._cur = path
            return self._reply()
        m = re.search(r"a2b_base64\((['\"])(.*?)\1\)", code)
        if m and self._cur is not None:
            raw = base64.b64decode(m.group(2))
            if self.corrupt_upload and len(raw) < 384:
                raw = raw[:-1]
            files[self._cur] += raw
            return self._reply()
        if "SIZE" in code:
            path = re.search(r"os\.stat\((['\"])(.*?)\1\)", code).group(2)  # type: ignore[union-attr]
            n = len(files[path]) if path in files else -1
            return self._reply(f"SIZE {n}\r\n")
        if "uhashlib" in code:
            path = re.search(r"open\((['\"])(.*?)\1, 'rb'\)", code).group(2)  # type: ignore[union-attr]
            if not self.has_sha:
                return self._reply("SHA none\r\n")
            digest = hashlib.sha256(bytes(files.get(path, b""))).hexdigest()
            return self._reply(f"SHA {digest}\r\n")
        if "f.close()" in code and "os.remove" in code:
            if self._cur:
                files.pop(self._cur, None)
            return self._reply()
        if "f.close()" in code:
            return self._reply()
        if "freq.set" in code:
            self.freq_calls.append(code)
            self.rebooted += 1
            self._mode = "normal"          # 重開機後回到一般 REPL
            self._out.clear()
            self._out += b"\r\n>>> "
            return
        self._reply()

    # ---- 串流 ----------------------------------------------------------------

    def _start_script(self, code: str) -> None:
        first = code.split("\n", 1)[0]
        self.params = ast.literal_eval(first[4:])
        if self.fail_start:
            self._out += b"OK\x04" + self.fail_start.encode() + b"\x04>"
            return
        p = self.params
        w, h = _SIZES.get(p.get("framesize", "QVGA"), (320, 240))
        self._cfg = dict(p.get("camera", {}), vflip=p.get("vflip", False), hmirror=p.get("hmirror", False),
                         threshold=p.get("threshold", 0.3), nms=p.get("nms", 0.3))
        self._ack = {"n": 0, "id": 0, "ok": [], "err": []}
        self._seq = 0
        self._cmdbuf = b""
        ops = list(p.get("ops", []))
        cfg = {"framesize": p.get("framesize", "QVGA"), "gray": p.get("pixformat", "rgb") == "gray",
               "dual_buff": False, "ops": ops, "roi": [0, 0, w, h], "send_every": p.get("send_every", 1),
               "lcd": bool(p.get("lcd", False)), "quality": p.get("quality", 70)}
        caps = {"fw": "v0.6.3_2 (simulated)", "sensor": "GC0328", "sensor_id": 157,
                "framesizes": list(_SIZES), "ops": ["stats", "diff", "sharp", "blobs", "yolo", "classify"],
                "kpu": True, "mem_free": 1500000, "heap_free": 900000, "sd": self.sd}
        lines = ["CAMERA ok [] err []"]
        if "yolo" in ops or "classify" in ops:
            if p.get("model") not in self.files:
                self._out += b"OK\x04OSError: model not found " + str(p.get("model")).encode() + b"\x04>"
                return
            lines.append("MODEL heap_before 900000 heap_after 600000")
        lines += ["CAPS " + json.dumps(caps), "FW v0.6.3_2 (simulated)", "CONFIG " + json.dumps(cfg),
                  "MEM mem_free 1400000 heap_free 590000", "STREAM_START"]
        self._out += b"OK" + "\r\n".join(lines).encode() + b"\r\n"
        self._mode = "stream"
        self.starts += 1

    def _write_stream(self, data: bytes) -> None:
        if b"\x03" in data:
            # 板子收到 Ctrl+C：結束串流，鮑率切回 115200 後才印出追蹤訊息
            self._mode = "raw"
            self.stops += 1
            self._out.clear()
            self._late += b"\r\nTraceback (most recent call last):\r\nKeyboardInterrupt:\r\n\x04\x04>"
            return
        self._cmdbuf = (self._cmdbuf + data)[-1024:]
        ok: list[str] = []
        err: list[str] = []
        cmd_id, count = self._ack["id"], self._ack["n"]
        got = False
        while b"\n" in self._cmdbuf:
            line, _, self._cmdbuf = self._cmdbuf.partition(b"\n")
            if not line.strip().startswith(b"CMD "):
                continue
            got = True
            count += 1
            try:
                c = json.loads(line.strip()[4:])
            except ValueError as e:
                err.append(f"json:{e}")
                continue
            cmd_id = c.pop("_id", cmd_id)
            self.cmds.append(dict(c))
            for k, v in c.items():
                if k in self.reject or k not in _CMD_KEYS:
                    err.append(f"{k}:rejected")
                else:
                    self._cfg[k] = v
                    ok.append(k)
        if got and not self.ack_never:
            self._ack = {"n": count, "id": cmd_id, "ok": ok, "err": err}

    def _packet(self) -> bytes:
        p = self.params
        w, h = _SIZES.get(p.get("framesize", "QVGA"), (320, 240))
        gray = p.get("pixformat", "rgb") == "gray"
        every = p.get("send_every", 1)
        meta: dict[str, Any] = {
            "seq": self._seq, "t": int(time.monotonic() * 1000) & 0xFFFFFFF, "w": w, "h": h,
            "fps": 20.0, "us": {"cap": 1000, "jpg": 2000, "loop": 50000}, "lat": 8000,
            "mem": 1000000, "heap": 500000, "cfg": dict(self._cfg), "rb": {}, "ack": dict(self._ack)}
        ops = p.get("ops", [])
        if "yolo" in ops:
            meta["objs"] = [list(o) for o in self.objs]
        if "classify" in ops:
            meta["cls"] = [list(c) for c in self.cls]
        jpeg = self._jpeg(w, h, gray) if every and self._seq % every == 0 else b""
        self._seq += 1
        return encode_packet(meta, jpeg)

    def _jpeg(self, w: int, h: int, gray: bool) -> bytes:
        key = (w, h, gray)
        if key not in self._jpeg_cache:
            import cv2
            import numpy as np

            img = np.zeros((h, w) if gray else (h, w, 3), np.uint8)
            img[:, : w // 2] = 200
            if not gray:
                img[:, :, 2] = 255
            ok, buf = cv2.imencode(".jpg", img)
            assert ok
            self._jpeg_cache[key] = buf.tobytes()
        return self._jpeg_cache[key]
