"""用 USB 的 VID/PID 找 K210 開發板的序列埠。

舊程式用名稱（usbserial、ttyUSB）過濾，換了線或換了系統就找不到。這裡改看晶片的
VID/PID：K210 板子常見的 USB 轉序列晶片是 CH340／CH341、FT2232、CP210x、CH552。
符合的排前面，其他序列埠排在後面讓使用者自己選。只列舉，不開啟任何序列埠。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# (VID, PID) -> 晶片名稱
KNOWN_CHIPS: dict[tuple[int, int], str] = {
    (0x1A86, 0x7523): "CH340",
    (0x1A86, 0x5523): "CH341",
    (0x1A86, 0x55D4): "CH552",    # 新版 Maixduino 常見（CH552 系列）
    (0x1A86, 0x55D3): "CH552",
    (0x1A86, 0x5512): "CH341",
    (0x0403, 0x6010): "FT2232",   # Maix Dock／Bit 舊版：一個晶片兩個介面
    (0x10C4, 0xEA60): "CP210x",
}
DUAL_PORT_CHIPS = {(0x0403, 0x6010)}  # 兩個介面，只有第一個接 K210 的序列埠

_IGNORED = ("bluetooth", "debug-console", "wlan-debug")


@dataclass
class PortCandidate:
    device: str
    description: str = ""
    vid: int | None = None
    pid: int | None = None
    serial_number: str | None = None
    location: str | None = None
    chip: str = ""
    likely: bool = False   # VID/PID 符合 K210 常見晶片

    @property
    def stable_id(self) -> str:
        return stable_id(self)

    @property
    def label(self) -> str:
        chip = f"（{self.chip}）" if self.chip else ""
        return f"{self.device}{chip} {self.description}".strip()


def stable_id(p: PortCandidate) -> str:
    """同一塊板子每次插上都一樣的 id：k210:<vid>:<pid>:<序號或 USB 位置>。
    沒有序號（CH340 常見）時用 USB 位置，再不行才用路徑。"""
    vid = f"{p.vid:04x}" if p.vid is not None else "0000"
    pid = f"{p.pid:04x}" if p.pid is not None else "0000"
    return f"k210:{vid}:{pid}:{p.serial_number or p.location or p.device}"


def _interface_index(p: PortCandidate) -> int:
    """雙介面晶片的介面編號（越小越前面）。Linux 看 location 的 ':1.N'，
    macOS 看裝置名稱最後一個數字（usbserial-XXXX0／1），其他系統退回依名稱排序。"""
    if p.location:
        m = re.search(r"\.(\d+)$", p.location)
        if m:
            return int(m.group(1))
    m = re.search(r"usbserial-[^/]*?(\d)$", p.device)
    return int(m.group(1)) if m else 99


def _natural_key(s: str) -> list[Any]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_serial_ports() -> list[PortCandidate]:
    """列出序列埠：K210 常見晶片在前（依穩定順序），其他在後。

    macOS 同一個埠會有 /dev/cu.* 與 /dev/tty.* 兩個名稱，只留 cu（tty 會卡在等待載波）。
    """
    from serial.tools import list_ports  # 延遲匯入：pyserial 沒裝也不影響註冊

    raw = list(list_ports.comports())
    devices = {getattr(p, "device", "") for p in raw}
    cands: list[PortCandidate] = []
    for p in raw:
        dev = getattr(p, "device", "") or ""
        if not dev:
            continue
        if dev.startswith("/dev/tty.") and ("/dev/cu." + dev[len("/dev/tty."):]) in devices:
            continue
        vid, pid = getattr(p, "vid", None), getattr(p, "pid", None)
        text = (dev + " " + (getattr(p, "description", "") or "")).lower()
        if vid is None and any(w in text for w in _IGNORED):
            continue
        if vid is None and re.search(r"/ttyS\d+$", dev):  # Linux 主機板上的內建序列埠
            continue
        chip = KNOWN_CHIPS.get((vid, pid), "") if vid is not None else ""
        cands.append(PortCandidate(
            device=dev, description=getattr(p, "description", "") or "", vid=vid, pid=pid,
            serial_number=getattr(p, "serial_number", None) or None,
            location=getattr(p, "location", None) or None, chip=chip, likely=bool(chip)))

    # 雙介面晶片：同一顆晶片只有第一個介面是 K210 的序列埠，第二個降為「其他」
    groups: dict[tuple[Any, ...], list[PortCandidate]] = {}
    for c in cands:
        if c.likely and (c.vid, c.pid) in DUAL_PORT_CHIPS:
            key = (c.vid, c.pid, c.serial_number or (c.location or "").split(":")[0] or "-")
            groups.setdefault(key, []).append(c)
    for members in groups.values():
        members.sort(key=lambda c: (_interface_index(c), _natural_key(c.device)))
        for extra in members[1:]:
            extra.likely = False

    cands.sort(key=lambda c: (not c.likely, _natural_key(c.device)))
    return cands


def likely_ports() -> list[PortCandidate]:
    return [c for c in list_serial_ports() if c.likely]
