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


def test_worker_batches_queued_controls(qtbot):
    """佇列裡連續的參數指令要合併成一次 set_controls，同一個 key 只送最後的值。"""
    from edgelens.core.controls import ControlResult
    from edgelens.gui.workers import SourceWorker
    from edgelens.sources.base import SourceInfo

    calls = []

    class Stub:
        def set_controls(self, changes):
            calls.append(dict(changes))
            return {k: ControlResult(ok=True, value=v, verified=True) for k, v in changes.items()}

    worker = SourceWorker(SourceInfo(kind="synthetic", id="synthetic:t", name="t"))
    worker._source = Stub()
    got = []
    worker.controlResult.connect(lambda k, r: got.append((k, r.value)))
    worker.set_control("brightness", 1)
    worker.set_control("noise", 5)
    worker.set_control("brightness", 3)
    worker.set_pipeline({"version": 1, "steps": []})
    worker.set_control("gray", True)
    worker._drain()
    assert calls == [{"noise": 5, "brightness": 3}, {"gray": True}]
    assert sorted(got) == [("brightness", 3), ("gray", True), ("noise", 5)]


def test_prune_orphan_threads_drops_finished():
    from PySide6.QtCore import QThread

    from edgelens.gui import workers

    class Fake(QThread):
        def run(self):
            self.msleep(300)

    done, running = QThread(), Fake()
    running.start()
    saved = list(workers._orphan_threads)
    workers._orphan_threads[:] = [done, running]
    try:
        assert workers.prune_orphan_threads() == 1
        assert workers._orphan_threads == [running]
    finally:
        running.wait(2000)
        workers._orphan_threads[:] = saved
