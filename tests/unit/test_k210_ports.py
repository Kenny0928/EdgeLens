"""序列埠挑選：VID/PID 排序、macOS 用 cu 不用 tty、穩定 id。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from edgelens.sources.k210 import ports


def P(device, vid=None, pid=None, serial=None, location=None, desc="n/a"):
    return SimpleNamespace(device=device, vid=vid, pid=pid, serial_number=serial, location=location,
                           description=desc)


@pytest.fixture
def fake_ports(monkeypatch):
    def install(items):
        import serial.tools.list_ports as lp

        monkeypatch.setattr(lp, "comports", lambda *a, **k: list(items))
    return install


def test_known_chips_come_first(fake_ports):
    fake_ports([
        P("/dev/cu.Bluetooth-Incoming-Port", desc="n/a"),
        P("/dev/cu.usbmodem1101", 0x2341, 0x0043, desc="Arduino"),
        P("/dev/cu.usbserial-1410", 0x1A86, 0x7523, location="20-1", desc="USB Serial"),
    ])
    res = ports.list_serial_ports()
    assert [c.device for c in res] == ["/dev/cu.usbserial-1410", "/dev/cu.usbmodem1101"]
    assert res[0].likely and res[0].chip == "CH340" and not res[1].likely


@pytest.mark.parametrize("vid,pid,chip", [(0x1A86, 0x7523, "CH340"), (0x1A86, 0x5523, "CH341"),
                                           (0x10C4, 0xEA60, "CP210x"), (0x1A86, 0x55D4, "CH552"),
                                           (0x0403, 0x6010, "FT2232")])
def test_each_known_vid_pid(fake_ports, vid, pid, chip):
    fake_ports([P("COM3", vid, pid, serial="S1")])
    (c,) = ports.list_serial_ports()
    assert c.likely and c.chip == chip


def test_macos_prefers_cu_over_tty(fake_ports):
    fake_ports([P("/dev/tty.usbserial-1410", 0x1A86, 0x7523), P("/dev/cu.usbserial-1410", 0x1A86, 0x7523)])
    assert [c.device for c in ports.list_serial_ports()] == ["/dev/cu.usbserial-1410"]


def test_tty_kept_when_no_cu_twin(fake_ports):
    fake_ports([P("/dev/tty.usbserial-9", 0x1A86, 0x7523)])
    assert [c.device for c in ports.list_serial_ports()] == ["/dev/tty.usbserial-9"]


def test_linux_builtin_ttyS_ignored(fake_ports):
    fake_ports([P("/dev/ttyS0"), P("/dev/ttyS1"), P("/dev/ttyUSB0", 0x1A86, 0x7523)])
    assert [c.device for c in ports.list_serial_ports()] == ["/dev/ttyUSB0"]


def test_ft2232_only_first_interface_is_likely(fake_ports):
    fake_ports([
        P("/dev/cu.usbserial-AB0JK1", 0x0403, 0x6010, serial="AB0JK", desc="Dual RS232-HS"),
        P("/dev/cu.usbserial-AB0JK0", 0x0403, 0x6010, serial="AB0JK", desc="Dual RS232-HS"),
    ])
    res = ports.list_serial_ports()
    assert [c.device for c in res] == ["/dev/cu.usbserial-AB0JK0", "/dev/cu.usbserial-AB0JK1"]
    assert [c.likely for c in res] == [True, False]


def test_ft2232_linux_location_interface(fake_ports):
    fake_ports([P("/dev/ttyUSB1", 0x0403, 0x6010, location="1-2:1.1"), P("/dev/ttyUSB0", 0x0403, 0x6010,
                                                                       location="1-2:1.0")])
    res = ports.list_serial_ports()
    assert res[0].device == "/dev/ttyUSB0" and res[0].likely and not res[1].likely


def test_two_separate_ft2232_boards_both_likely(fake_ports):
    fake_ports([P("COM3", 0x0403, 0x6010, serial="A"), P("COM5", 0x0403, 0x6010, serial="B")])
    assert [c.likely for c in ports.list_serial_ports()] == [True, True]


def test_natural_sort_of_com_ports(fake_ports):
    fake_ports([P("COM10", 0x1A86, 0x7523, serial="a"), P("COM3", 0x1A86, 0x7523, serial="b")])
    assert [c.device for c in ports.list_serial_ports()] == ["COM3", "COM10"]


def test_stable_id_prefers_serial_then_location_then_device(fake_ports):
    fake_ports([P("COM3", 0x1A86, 0x7523, serial="ABC"), P("COM4", 0x1A86, 0x7523, location="1-4"),
                P("COM5", 0x1A86, 0x7523)])
    ids = [c.stable_id for c in ports.list_serial_ports()]
    assert ids == ["k210:1a86:7523:ABC", "k210:1a86:7523:1-4", "k210:1a86:7523:COM5"]


def test_stable_id_for_unknown_port():
    assert ports.stable_id(ports.PortCandidate("/dev/x")) == "k210:0000:0000:/dev/x"


def test_likely_ports_filters(fake_ports):
    fake_ports([P("/dev/cu.usbmodem1", 0x2341, 0x43), P("/dev/cu.usbserial-1", 0x1A86, 0x7523)])
    assert [c.device for c in ports.likely_ports()] == ["/dev/cu.usbserial-1"]
