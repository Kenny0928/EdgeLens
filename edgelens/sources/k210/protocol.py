"""K210 串流封包的解析與編碼。

封包格式（和板上程式 board/stream_v1.py 一致）：

    b"JBF1" + <H 中繼資料長度> + <I JPEG 長度> + 中繼資料(JSON) + JPEG

JPEG 長度為 0 表示這一張只傳數據。板子在串流前後也會印出一般文字（錯誤訊息），
USB 序列線偶爾掉位元組，所以解析器要能：任意切斷、遇到亂碼往後重新找 MAGIC、
把能讀的文字另外回傳、其餘位元組只計數。這個模組不碰序列埠，方便測試。
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

MAGIC = b"JBF1"
HEADER = struct.Struct("<HI")
HEADER_END = len(MAGIC) + HEADER.size
MAX_META = 4096        # 超過視為資料錯亂
MAX_JPEG = 200000
_KEEP = len(MAGIC) - 1  # 保留可能被切斷的 MAGIC 開頭


@dataclass
class Packet:
    """一個完整的封包。jpeg 為 b"" 表示只傳數據。"""

    meta: dict[str, Any]
    jpeg: bytes = b""


@dataclass
class TextChunk:
    """板子印出的文字（例如錯誤訊息）。"""

    text: str


def encode_packet(meta: dict[str, Any], jpeg: bytes = b"") -> bytes:
    """產生封包位元組（測試與模擬板子用）。"""
    body = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    return MAGIC + HEADER.pack(len(body), len(jpeg)) + body + jpeg


class PacketParser:
    """有狀態的串流解析器。feed() 可以餵任意長度、任意切斷位置的位元組。"""

    def __init__(self) -> None:
        self.bad_frames = 0   # JPEG 不完整而丟棄的畫面數
        self.junk_bytes = 0   # 無法辨識而丟棄的位元組數
        self.lost_seq = 0     # 依序號推算，傳輸中遺失的封包數
        self.packets = 0      # 成功解析的封包數
        self.reset()

    def reset(self) -> None:
        """清掉緩衝與序號記錄（重新開始串流時呼叫）；計數器不歸零。"""
        self._buf = b""
        self._text = b""     # 尚未判定的非封包位元組
        self._last_seq: int | None = None

    def reset_counters(self) -> None:
        self.bad_frames = self.junk_bytes = self.lost_seq = self.packets = 0

    # ---- 解析 ------------------------------------------------------------------

    def feed(self, data: bytes) -> list[Packet | TextChunk]:
        out: list[Packet | TextChunk] = []
        self._buf += data
        while True:
            i = self._buf.find(MAGIC)
            if i < 0:
                # 只保留「可能是被切斷的 MAGIC 開頭」的尾端，其餘馬上當文字處理
                keep = next((k for k in range(min(_KEEP, len(self._buf)), 0, -1)
                             if self._buf.endswith(MAGIC[:k])), 0)
                self._text += self._buf[:len(self._buf) - keep]
                self._buf = self._buf[len(self._buf) - keep:]
                break
            self._text += self._buf[:i]
            self._buf = self._buf[i:]
            if len(self._buf) < HEADER_END:
                break
            meta_len, jpg_len = HEADER.unpack_from(self._buf, len(MAGIC))
            if meta_len > MAX_META or jpg_len > MAX_JPEG:
                self._skip_magic()  # 長度不合理：資料錯亂，往後重新找 MAGIC
                continue
            total = HEADER_END + meta_len + jpg_len
            if len(self._buf) < total:
                break
            try:
                meta = json.loads(self._buf[HEADER_END:HEADER_END + meta_len])
                if not isinstance(meta, dict):
                    raise ValueError("meta 不是物件")
            except ValueError:  # JSONDecodeError、UnicodeDecodeError 都是 ValueError
                self._skip_magic()
                continue
            jpg = self._buf[HEADER_END + meta_len:total]
            self._buf = self._buf[total:]
            self._flush_text(out, final=True)
            self._track_seq(meta.get("seq"))
            if not jpg or (jpg[:2] == b"\xff\xd8" and jpg[-2:] == b"\xff\xd9"):
                self.packets += 1
                out.append(Packet(meta, jpg))
            else:
                self.bad_frames += 1
        self._flush_text(out, final=False)
        return out

    def _skip_magic(self) -> None:
        """目前位置的 MAGIC 是假的：當成雜訊丟掉 4 bytes，繼續往後找。"""
        self.junk_bytes += len(MAGIC)
        self._buf = self._buf[len(MAGIC):]

    def _track_seq(self, seq: Any) -> None:
        if not isinstance(seq, int) or isinstance(seq, bool):
            return
        if self._last_seq is not None and seq > self._last_seq + 1:
            self.lost_seq += seq - self._last_seq - 1
        self._last_seq = seq  # 序號倒退（板子重啟）就直接重新計

    def _flush_text(self, out: list[Packet | TextChunk], final: bool) -> None:
        """把累積的非封包位元組分類：可列印文字回傳，其餘計入 junk_bytes。
        final=False 時，尾端被切斷的 UTF-8 字元留到下次再判斷。"""
        data = self._text.replace(b"\x04", b"")
        if not data:
            self._text = b""
            return
        hold = b""
        if not final:
            try:
                data.decode("utf-8")
            except UnicodeDecodeError as e:
                if e.start >= len(data) - 3 and "unexpected end" in e.reason:
                    hold = data[e.start:]
                    data = data[:e.start]
        self._text = hold
        if not data:
            return
        printable = sum(32 <= b < 127 or b in (9, 10, 13) or b >= 0x80 for b in data)
        try:
            text: str | None = data.decode("utf-8")
        except UnicodeDecodeError:
            text = None
        # 傳輸中掉資料留下的二進位雜訊只計數，不當成文字
        if text is None or printable < len(data) * 0.95:
            self.junk_bytes += len(data)
            return
        out.append(TextChunk(text))

    @property
    def stats(self) -> dict[str, int]:
        return {"packets": self.packets, "bad_frames": self.bad_frames,
                "junk_bytes": self.junk_bytes, "lost_seq": self.lost_seq}

