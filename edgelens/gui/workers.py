"""相機執行緒：SourceWorker 在背景讀取與處理，SourceHandle 是 GUI 端的包裝。

執行緒規則（見 docs/DEVELOPMENT.md）：FrameSource 與 Pipeline 只在 worker 執行緒使用；
GUI 只透過指令佇列送「新的值」，結果用 Qt signal（自動排隊到 GUI 執行緒）回報。

為什麼用 QThread 子類別而不是 moveToThread：worker 的 run() 是一個長迴圈，不需要
事件迴圈；執行緒在 run() 結束時就結束，關閉程式時不會卡在 exec()。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Signal
from PySide6.QtGui import QImage

from edgelens.core.controls import ControlResult, ControlSpec, ControlType
from edgelens.core.frame import Frame
from edgelens.core.settings import Settings
from edgelens.processing.pipeline import Pipeline, ProcessResult
from edgelens.sources.base import FrameSource, SourceCapabilities, SourceError, SourceInfo, create_source

log = logging.getLogger(__name__)

# 無法在時限內停止的執行緒不能被銷毀（會讓程式崩潰），先留著等它自己結束
_orphan_threads: list[QThread] = []

DEBOUNCE_MS = 150


# ---------------------------------------------------------------- FrameSlot

class FrameSlot:
    """執行緒安全、只保存最新值的槽。推論 worker 從這裡拿最新畫面，慢的消費者只會漏掉舊畫面。"""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._value: Any = None
        self._version = 0
        self._seen = 0

    def put(self, value: Any) -> None:
        with self._cond:
            self._value = value
            self._version += 1
            self._cond.notify_all()

    def get(self, timeout: float | None = None) -> Any:
        """等待比上次 get 更新的值；逾時回傳 None。（設計給單一消費者使用。）"""
        with self._cond:
            if self._version == self._seen:
                self._cond.wait_for(lambda: self._version != self._seen, timeout)
            if self._version == self._seen:
                return None
            self._seen = self._version
            return self._value

    def peek(self) -> Any:
        """看目前的最新值，不影響 get 的「是否更新」判斷。"""
        with self._cond:
            return self._value

    @property
    def version(self) -> int:
        return self._version


# ---------------------------------------------------------------- DebouncedSender

class DebouncedSender(QObject):
    """在 GUI 執行緒合併短時間內同一個 key 的多次修改，只送最後的值。

    滑桿拖曳時每秒會產生幾十次變更，合併後相機只收到約每 150 ms 一次。
    """

    def __init__(self, callback, interval_ms: int = DEBOUNCE_MS, parent: QObject | None = None):
        super().__init__(parent)
        self._callback = callback
        self._pending: dict[str, Any] = {}
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.flush)

    def send(self, key: str, value: Any) -> None:
        self._pending[key] = value
        if not self._timer.isActive():
            self._timer.start()

    def flush(self) -> None:
        self._timer.stop()
        pending, self._pending = self._pending, {}
        for k, v in pending.items():
            try:
                self._callback(k, v)
            except Exception:  # noqa: BLE001 - 一個參數失敗不影響其他參數
                log.exception("送出參數 %s 失敗", k)

    def cancel(self) -> None:
        self._timer.stop()
        self._pending.clear()

    @property
    def pending(self) -> dict[str, Any]:
        return dict(self._pending)


# ---------------------------------------------------------------- FramePacket

@dataclass
class FramePacket:
    """worker 交給 GUI 的一張畫面。

    frame      原始 Frame。 result 為處理結果（影像、overlays、data、各步驟耗時）。
    qimage     處理後、要顯示的 QImage；qimage_raw 是未處理的（沒有處理步驟時與 qimage 相同物件）。
    stats      fps、latency_ms（取像到處理完）、process_ms、dropped（GUI 來不及顯示而略過的張數）。
    """

    frame: Frame
    result: ProcessResult
    qimage: QImage
    qimage_raw: QImage
    stats: dict[str, float] = field(default_factory=dict)


def ndarray_to_qimage(img: np.ndarray) -> QImage:
    """BGR／灰階 ndarray → QImage。一律複製一份，避免 numpy buffer 被回收後畫面壞掉。"""
    if img.ndim == 2:
        gray = np.ascontiguousarray(img)
        h, w = gray.shape
        return QImage(gray.data, w, h, gray.strides[0], QImage.Format.Format_Grayscale8).copy()
    rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()


# ---------------------------------------------------------------- SourceWorker

class SourceWorker(QObject):
    """擁有 FrameSource 與 Pipeline 的 worker；run() 在 QThread 裡執行。"""

    frameReady = Signal(object)          # FramePacket
    opened = Signal(dict)                # controls、capabilities、values
    failed = Signal(dict)                # message、hint、retryable
    controlResult = Signal(str, object)  # key、ControlResult
    actionProgress = Signal(str, float, str)
    actionFinished = Signal(str, object)
    pipelineError = Signal(str)
    closed = Signal()

    def __init__(self, info: SourceInfo, initial_values: dict[str, Any] | None = None,
                 initial_pipeline: dict[str, Any] | None = None):
        super().__init__()
        self.info = info
        self._initial_values = dict(initial_values or {})
        self._cmds: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._stop = threading.Event()
        self._cancel = threading.Event()
        self._inflight = False          # 已 emit 但 GUI 還沒處理完
        self._source: FrameSource | None = None
        self._pipeline = Pipeline()
        self._errored = False
        self._times: deque[float] = deque(maxlen=30)
        self._dropped = 0
        self.frame_slot = FrameSlot()  # SourceHandle 會換成共用的槽
        if initial_pipeline:
            try:
                self._pipeline = Pipeline.from_dict(initial_pipeline)
            except Exception as e:  # noqa: BLE001
                log.warning("還原處理流程失敗：%s", e)

    # ---- 執行緒安全的指令（可從任何執行緒呼叫）--------------------------------
    def set_control(self, key: str, value: Any) -> None:
        self._cmds.put(("control", (key, value)))

    def set_pipeline(self, data: dict[str, Any]) -> None:
        self._cmds.put(("pipeline", data))

    def run_action(self, key: str, kwargs: dict[str, Any] | None = None) -> None:
        self._cancel.clear()
        self._cmds.put(("action", (key, dict(kwargs or {}))))

    def cancel_action(self) -> None:
        self._cancel.set()

    def reopen(self) -> None:
        self._cmds.put(("reopen", None))

    def stop(self) -> None:
        self._stop.set()
        self._cancel.set()
        self._cmds.put(("stop", None))

    def acknowledge(self) -> None:
        """GUI 處理完上一張畫面後呼叫，worker 才會送下一張（避免訊號堆積）。"""
        self._inflight = False

    # ---- 執行緒內部 -------------------------------------------------------------
    def run(self) -> None:
        try:
            self._open()
            while not self._stop.is_set():
                self._drain()
                if self._stop.is_set():
                    break
                if self._errored or self._source is None:
                    try:
                        cmd = self._cmds.get(timeout=0.1)
                        self._handle(cmd)
                    except queue.Empty:
                        pass
                    continue
                try:
                    frame = self._source.read(timeout=0.2)
                except SourceError as e:
                    self._fail(e)
                    continue
                except Exception as e:  # noqa: BLE001 - 來源的未預期錯誤也要讓使用者看得到
                    log.exception("讀取畫面失敗")
                    self._fail(SourceError(str(e), retryable=True))
                    continue
                if frame is not None:
                    self._on_frame(frame)
        except Exception:  # noqa: BLE001
            log.exception("相機執行緒發生未預期錯誤")
        finally:
            self._close_source()
            self.closed.emit()

    def _drain(self) -> None:
        """處理佇列裡所有指令。連續的參數指令合併成一次 set_controls（同一個 key 取最後的值），
        K210 這類每條指令都要等回覆的來源才不會讓畫面卡住；其他指令照原本順序執行。"""
        batch: dict[str, Any] = {}
        while True:
            try:
                cmd = self._cmds.get_nowait()
            except queue.Empty:
                break
            if cmd[0] == "control":
                key, value = cmd[1]
                batch.pop(key, None)
                batch[key] = value
                continue
            if batch:
                self._apply_controls(batch)
                batch = {}
            self._handle(cmd)
        if batch:
            self._apply_controls(batch)

    def _apply_controls(self, changes: dict[str, Any]) -> None:
        if self._source is None or self._errored:
            return
        try:
            results = self._source.set_controls(changes)
        except SourceError as e:
            results = {k: ControlResult.failed(e.message, v) for k, v in changes.items()}
        except Exception as e:  # noqa: BLE001
            log.exception("設定參數失敗")
            results = {k: ControlResult.failed(str(e), v) for k, v in changes.items()}
        for key, value in changes.items():
            self.controlResult.emit(key, results.get(key) or ControlResult.failed("", value))

    def _handle(self, cmd: tuple[str, Any]) -> None:
        name, arg = cmd
        if name == "stop":
            return
        if name == "reopen":
            self._close_source()
            self._open()
            return
        if name == "pipeline":
            try:
                self._pipeline = Pipeline.from_dict(arg)
            except Exception as e:  # noqa: BLE001
                log.warning("套用處理流程失敗：%s", e)
                self.pipelineError.emit(str(e))
            return
        if self._source is None or self._errored:
            return
        if name == "control":
            key, value = arg
            try:
                res = self._source.set_control(key, value)
            except SourceError as e:
                res = ControlResult.failed(e.message, value)
            except Exception as e:  # noqa: BLE001
                log.exception("設定參數 %s 失敗", key)
                res = ControlResult.failed(str(e), value)
            self.controlResult.emit(key, res)
        elif name == "action":
            self._do_action(*arg)

    def _do_action(self, key: str, kwargs: dict[str, Any]) -> None:
        def progress(ratio: float, text: str = "") -> None:
            self.actionProgress.emit(key, float(ratio), str(text))

        kwargs.setdefault("progress", progress)
        kwargs.setdefault("cancel", self._cancel.is_set)
        try:
            res = self._source.run_action(key, **kwargs)  # type: ignore[union-attr]
        except SourceError as e:
            res = ControlResult.failed(e.message)
        except Exception as e:  # noqa: BLE001
            log.exception("動作 %s 失敗", key)
            res = ControlResult.failed(str(e))
        self.actionFinished.emit(key, res)

    def _open(self) -> None:
        self._errored = False
        self._times.clear()
        try:
            src = create_source(self.info)
            src.open()
            self._source = src
            self._apply_initial(src)
            self._pipeline.reset()
            self.opened.emit(self._describe(src))
        except SourceError as e:
            self._fail(e)
        except Exception as e:  # noqa: BLE001
            log.exception("開啟相機失敗")
            self._fail(SourceError(str(e), retryable=True))

    def _apply_initial(self, src: FrameSource) -> None:
        """套用上次保存的參數值（只套用和目前不同的，避免對硬體做多餘的寫入）。"""
        specs = {s.key: s for s in src.controls()}
        for key, value in self._initial_values.items():
            spec = specs.get(key)
            if spec is None or spec.type == ControlType.ACTION:
                continue
            try:
                if src.get_control(key) == value:
                    continue
            except Exception:  # noqa: BLE001 - 讀不到就直接設定
                pass
            try:
                src.set_control(key, value)
            except Exception as e:  # noqa: BLE001
                log.warning("還原參數 %s 失敗：%s", key, e)

    @staticmethod
    def _describe(src: FrameSource) -> dict[str, Any]:
        controls = list(src.controls())
        values: dict[str, Any] = {}
        for spec in controls:
            if spec.type == ControlType.ACTION:
                continue
            try:
                values[spec.key] = src.get_control(spec.key)
            except Exception:  # noqa: BLE001
                values[spec.key] = spec.default_value()
        try:
            caps = src.capabilities()
        except Exception:  # noqa: BLE001
            caps = SourceCapabilities()
        return {"controls": controls, "capabilities": caps, "values": values}

    def _fail(self, err: SourceError) -> None:
        self._errored = True
        self._close_source()
        self.failed.emit({"message": err.message, "hint": err.hint, "retryable": err.retryable})

    def _close_source(self) -> None:
        src, self._source = self._source, None
        if src is not None:
            try:
                src.close()
            except Exception:  # noqa: BLE001
                log.exception("關閉相機失敗")

    def _on_frame(self, frame: Frame) -> None:
        now = time.monotonic()
        self._times.append(now)
        result = self._pipeline.process(frame)
        qimage = ndarray_to_qimage(result.image)
        qraw = qimage if result.image is frame.image else ndarray_to_qimage(frame.image)
        fps = 0.0
        if len(self._times) >= 2 and self._times[-1] > self._times[0]:
            fps = (len(self._times) - 1) / (self._times[-1] - self._times[0])
        stats = {"fps": fps, "latency_ms": (time.monotonic() - frame.t_cap) * 1000.0,
                 "process_ms": result.total_ms, "dropped": float(self._dropped)}
        packet = FramePacket(frame=frame, result=result, qimage=qimage, qimage_raw=qraw, stats=stats)
        self.frame_slot.put(packet)
        if self._inflight:
            self._dropped += 1
            return
        self._inflight = True
        self.frameReady.emit(packet)


class _WorkerThread(QThread):
    def __init__(self, worker: SourceWorker, parent: QObject | None = None):
        super().__init__(parent)
        self._worker = worker

    def run(self) -> None:
        self._worker.run()


# ---------------------------------------------------------------- SourceHandle

class SourceHandle(QObject):
    """GUI 端的相機包裝：狀態、參數、最新畫面，並轉發 worker 的 signals。"""

    stateChanged = Signal(str)
    opened = Signal(dict)
    failed = Signal(dict)
    controlResult = Signal(str, object)
    actionProgress = Signal(str, float, str)
    actionFinished = Signal(str, object)
    pipelineError = Signal(str)
    frameReady = Signal(object)
    closed = Signal()

    def __init__(self, info: SourceInfo, settings: Settings, parent: QObject | None = None):
        super().__init__(parent)
        self.info = info
        self.settings = settings
        self.state = "closed"
        self.controls: list[ControlSpec] = []
        self.values: dict[str, Any] = {}
        self.capabilities = SourceCapabilities()
        self.last_packet: FramePacket | None = None
        self.last_error: dict[str, Any] = {}
        self.frame_slot = FrameSlot()
        saved = settings.source_settings(info.id)
        self.pipeline_dict: dict[str, Any] = dict(saved.get("pipeline") or {})
        self._worker: SourceWorker | None = None
        self._thread: _WorkerThread | None = None
        self._sender = DebouncedSender(self._send_control, parent=self)

    # ---- 啟動與停止 -------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        saved = self.settings.source_settings(self.info.id)
        worker = SourceWorker(self.info, saved.get("controls"), self.pipeline_dict or None)
        worker.frame_slot = self.frame_slot
        worker.opened.connect(self._on_opened)
        worker.failed.connect(self._on_failed)
        worker.frameReady.connect(self._on_frame)
        worker.controlResult.connect(self._on_control_result)
        worker.actionProgress.connect(self.actionProgress)
        worker.actionFinished.connect(self.actionFinished)
        worker.pipelineError.connect(self.pipelineError)
        worker.closed.connect(self._on_closed)
        self._worker = worker
        self._thread = _WorkerThread(worker, self)
        self._set_state("opening")
        self._thread.start()

    def request_stop(self) -> None:
        """通知 worker 停止但不等待（關閉多台相機時先全部通知，再逐一等待）。"""
        self._sender.cancel()
        if self._worker is not None:
            self._worker.stop()

    def stop(self, timeout: float = 2.0) -> bool:
        """停止並等待執行緒結束；回傳是否在時限內結束。"""
        self.request_stop()
        th = self._thread
        ok = True
        if th is not None:
            ok = th.wait(int(timeout * 1000))
            if not ok:
                log.error("相機執行緒 %s 在 %.1f 秒內沒有停止", self.info.id, timeout)
                _orphan_threads.append(th)
                th.setParent(None)
        self._thread = None
        self._worker = None
        self._set_state("closed")
        return ok

    @property
    def thread(self) -> QThread | None:
        return self._thread

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    # ---- 指令 -------------------------------------------------------------------
    def set_control(self, key: str, value: Any) -> None:
        """設定參數：立即更新本地值，經 DebouncedSender 合併後才送到相機。"""
        self.values[key] = value
        self._sender.send(key, value)

    def flush_controls(self) -> None:
        self._sender.flush()

    def _send_control(self, key: str, value: Any) -> None:
        if self._worker is not None:
            self._worker.set_control(key, value)

    def set_pipeline(self, data: dict[str, Any]) -> None:
        self.pipeline_dict = dict(data)
        self.update_saved(pipeline=self.pipeline_dict)
        if self._worker is not None:
            self._worker.set_pipeline(self.pipeline_dict)

    def run_action(self, key: str, **kwargs: Any) -> None:
        if self._worker is not None:
            self._worker.run_action(key, kwargs)

    def cancel_action(self) -> None:
        if self._worker is not None:
            self._worker.cancel_action()

    def retry(self) -> None:
        if self._worker is None:
            self.start()
            return
        self._set_state("opening")
        self._worker.reopen()

    def update_saved(self, **entries: Any) -> None:
        """把資料存進 settings.sources.<id>，保留其他面板存的欄位（例如 controls、pipeline）。"""
        data = self.settings.source_settings(self.info.id)
        data.update(entries)
        self.settings.set_source_settings(self.info.id, data)

    # ---- worker 事件（GUI 執行緒）-----------------------------------------------
    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.stateChanged.emit(state)

    def _on_opened(self, info: dict) -> None:
        self.controls = list(info["controls"])
        self.capabilities = info["capabilities"]
        self.values = dict(info["values"])
        self.last_error = {}
        self._set_state("running")
        self.opened.emit(info)

    def _on_failed(self, info: dict) -> None:
        self.last_error = dict(info)
        self._set_state("error")
        self.failed.emit(info)

    def _on_frame(self, packet: FramePacket) -> None:
        self.last_packet = packet
        try:
            self.frameReady.emit(packet)
        finally:
            if self._worker is not None:
                self._worker.acknowledge()

    def _on_control_result(self, key: str, res: ControlResult) -> None:
        if res.ok and res.value is not None:
            if key not in self._sender.pending:  # 使用者還在拖曳時不要用舊值覆蓋
                self.values[key] = res.value
            saved = dict(self.settings.source_settings(self.info.id).get("controls", {}))
            saved[key] = res.value
            self.update_saved(controls=saved)
        self.controlResult.emit(key, res)

    def _on_closed(self) -> None:
        self.closed.emit()
