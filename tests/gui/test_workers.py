from __future__ import annotations

import threading
import time

import pytest

from edgelens.gui.workers import DebouncedSender, FrameSlot

pytestmark = pytest.mark.gui


def test_frame_slot_keeps_latest_only():
    slot = FrameSlot()
    assert slot.get(0.01) is None
    for i in range(5):
        slot.put(i)
    assert slot.peek() == 4
    assert slot.get(0.1) == 4
    assert slot.get(0.05) is None  # 沒有更新的值
    assert slot.peek() == 4


def test_frame_slot_wakes_waiter():
    slot = FrameSlot()
    got = []
    t = threading.Thread(target=lambda: got.append(slot.get(2.0)))
    t.start()
    time.sleep(0.05)
    slot.put("x")
    t.join(1.0)
    assert got == ["x"]


def test_debounced_sender_merges(qtbot):
    sent = []
    s = DebouncedSender(lambda k, v: sent.append((k, v)), interval_ms=150)
    for v in range(10):
        s.send("a", v)
    s.send("b", 1)
    assert sent == []
    qtbot.waitUntil(lambda: len(sent) == 2, timeout=1000)
    assert dict(sent) == {"a": 9, "b": 1}
    assert len(sent) == 2


def test_debounced_flush_and_cancel(qtbot):
    sent = []
    s = DebouncedSender(lambda k, v: sent.append((k, v)))
    s.send("a", 1)
    s.flush()
    assert sent == [("a", 1)]
    s.send("a", 2)
    s.cancel()
    qtbot.wait(250)
    assert sent == [("a", 1)]
