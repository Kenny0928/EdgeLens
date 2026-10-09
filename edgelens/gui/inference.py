"""背景推論：一台相機 × 一個模型 = 一個 InferenceSession（自己的執行緒）。

為什麼不用 handle.frame_slot.get()：那個槽設計給單一消費者，同一台相機跑兩個模型（A/B 比較）
時會互相搶畫面。這裡改用 slot.version 搭配 slot.peek() 輪詢，每個 session 各自取最新的一張，
推論比相機慢時舊畫面直接略過（不排隊），延遲才不會越積越多。

runner 的 load／warmup／run 全部在 session 自己的執行緒裡執行（一個 runner 只在一條執行緒用）。
結果經由內部 signal 轉回 GUI 執行緒；GUI 來不及處理時只保留最新的一筆。
"""

from __future__ import annotations

import copy
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Signal

from edgelens.core.modelcard import ModelCard
from edgelens.core.project import DEFAULT_COLORS
from edgelens.core.results import InferenceResult
from edgelens.i18n import tr
from edgelens.inference.base import InferenceError
from edgelens.inference.runner import InferenceRunner, RunnerInfo

log = logging.getLogger(__name__)

# 第二個模型（A/B 比較）沒有對應專案類別時用的色盤，和預設色盤明顯不同
ALT_COLORS = ["#FF4FA3", "#00D2FF", "#B6FF00", "#FF7A00", "#B28DFF", "#FFE600"]
MAX_SESSIONS_PER_CAMERA = 2
_STATS_WINDOW = 30
_ids = iter(range(1, 1_000_000))


class InferenceSession(QObject):
    """一台相機 × 一個模型。背景執行緒從 handle.frame_slot 取最新畫面推論（只做最新，不排隊）。"""

    loaded = Signal(object)       # RunnerInfo
    failed = Signal(dict)         # message、hint
    resultReady = Signal(object)  # InferenceResult（frame_seq 對應畫面）
    rawReady = Signal(object)     # list[(TensorInfo, ndarray)]；取不到時是空 list
    infoChanged = Signal()        # 解碼器改變（第一次推論後或手動指定）
    stopped = Signal()
    _event = Signal(str, object)  # 執行緒 → GUI 的橋（排隊）

    def __init__(self, handle, card: ModelCard, *, backend: str | None = None, use_processed: bool = False,
                 parent: QObject | None = None):
        super().__init__(parent)
        self.handle = handle
        self.card = copy.deepcopy(card)          # 自己一份：改門檻、存說明檔都不影響別的 session
        self.backend = backend
        self.session_id = next(_ids)
        self.label = tr("gui.model.session_label", model=self.card.name, camera=handle.info.name)
        self.style_index = 0                     # 0＝主要；1 以上＝比較用（畫面用不同顏色與虛線）
        self.state = "idle"                      # idle／loading／running／failed／stopped
        self.use_processed = bool(use_processed)
        self.chosen_params: dict[str, Any] = {}  # 學生手動選解碼器時帶的參數（存說明檔用）
        self.builtin = False                     # 來自 OpenCV 內附偵測器（說明檔不存在原資料夾）
        self.info: RunnerInfo = RunnerInfo()
        self.last_result: InferenceResult | None = None
        self.last_error: dict[str, str] = {}
        self.raw_outputs: list = []
        self._times: deque[float] = deque(maxlen=_STATS_WINDOW)
        self._infer_ms: deque[float] = deque(maxlen=_STATS_WINDOW)
        self._age = 0
        self._lock = threading.Lock()
        self._th: dict[str, Any] = {"conf": self.card.conf_threshold, "nms": self.card.nms_threshold,
                                    "max_det": self.card.max_det, "class_agnostic": self.card.class_agnostic_nms}
        self._cmds: queue.Queue = queue.Queue()
        self._raw_req = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: InferenceResult | None = None
        self._notified = False
        self._last_fail = ""
        self._event.connect(self._dispatch)

    # ---- 啟動與停止 -------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self.state = "loading"
        self._thread = threading.Thread(target=self._run, name=f"infer-{self.session_id}", daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        self._stop.set()

    def stop(self, timeout: float = 2.0) -> bool:
        """停止並等待執行緒結束；回傳是否在時限內結束。"""
        self._stop.set()
        th = self._thread
        ok = True
        if th is not None and th is not threading.current_thread():
            th.join(max(0.0, timeout))
            ok = not th.is_alive()
            if not ok:
                log.error("推論執行緒 %s 在 %.1f 秒內沒有停止", self.label, timeout)
        self._thread = None
        if self.state != "stopped":
            self.state = "stopped"
            self.stopped.emit()
        return ok

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---- 設定（可從 GUI 執行緒呼叫）-----------------------------------------------
    def set_thresholds(self, conf: float | None = None, nms: float | None = None,
                       max_det: int | None = None, class_agnostic: bool | None = None) -> None:
        with self._lock:
            for key, val in (("conf", conf), ("nms", nms), ("max_det", max_det), ("class_agnostic", class_agnostic)):
                if val is not None:
                    self._th[key] = val

    @property
    def thresholds(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._th)

    def set_param(self, key: str, value: Any) -> None:
        """改 decoder_params（例如 Haar 的 min_neighbors）；後端每次推論都會重新讀取。"""
        self.card.decoder_params[key] = value

    def set_decoder(self, kind: str, params: dict | None = None) -> None:
        self._cmds.put(("decoder", kind, params))

    def request_raw(self) -> None:
        """下一張畫面時順便取模型的原始輸出（rawReady）。"""
        self._raw_req.set()

    def reset_stats(self) -> None:
        self._times.clear()
        self._infer_ms.clear()

    @property
    def stats(self) -> dict[str, float]:
        """infer_fps、infer_ms（平均總耗時）、age_frames（結果是幾張以前的）。"""
        fps = 0.0
        t = list(self._times)
        if len(t) >= 2 and t[-1] > t[0]:
            fps = (len(t) - 1) / (t[-1] - t[0])
        if t and time.monotonic() - t[-1] > 2.0:
            fps = 0.0
        ms = list(self._infer_ms)
        return {"infer_fps": fps, "infer_ms": sum(ms) / len(ms) if ms else 0.0, "age_frames": float(self._age)}

    # ---- 執行緒 -----------------------------------------------------------------
    def _post(self, kind: str, payload: Any = None) -> None:
        try:
            self._event.emit(kind, payload)
        except RuntimeError:  # QObject 已被刪除
            pass

    def _fail(self, message: str, hint: str = "") -> None:
        if message == self._last_fail:
            return
        self._last_fail = message
        self._post("failed", {"message": message, "hint": hint})

    def _run(self) -> None:
        runner = InferenceRunner(self.card, self.backend)
        try:
            try:
                runner.load()
                runner.warmup()
            except InferenceError as e:
                self._fail(e.message, e.hint)
                return
            except Exception as e:  # noqa: BLE001 - 後端套件可能丟任何例外，要轉成學生看得懂的訊息
                log.exception("載入模型失敗")
                self._fail(tr("gui.model.err.unexpected", error=str(e)[:200]), tr("gui.model.err.unexpected_hint"))
                return
            self.info = runner.info
            self._post("loaded", runner.info)
            self._loop(runner)
        finally:
            try:
                runner.close()
            except Exception:  # noqa: BLE001
                log.exception("關閉推論後端失敗")

    def _loop(self, runner: InferenceRunner) -> None:
        slot = self.handle.frame_slot
        last_version = 0
        last_decoder = runner.info.decoder
        while not self._stop.is_set():
            self._apply_commands(runner)
            version = slot.version
            packet = slot.peek() if version != last_version else None
            if packet is None:
                self._stop.wait(0.004)
                continue
            last_version = version
            with self._lock:
                th = dict(self._th)
            runner.conf_threshold, runner.nms_threshold = float(th["conf"]), float(th["nms"])
            runner.max_det, runner.class_agnostic = int(th["max_det"]), bool(th["class_agnostic"])
            try:
                frame = packet.frame
                if self.use_processed:
                    frame = frame.replace(image=packet.result.image)
                if self._raw_req.is_set():
                    self._raw_req.clear()
                    try:
                        self._post("raw", runner.raw(frame))
                    except InferenceError:
                        self._post("raw", [])
                result = runner.run(frame)
            except InferenceError as e:
                self._fail(e.message, e.hint)
                self._stop.wait(0.05)
                continue
            except Exception as e:  # noqa: BLE001
                log.exception("推論失敗")
                self._fail(tr("gui.model.err.unexpected", error=str(e)[:200]), tr("gui.model.err.unexpected_hint"))
                self._stop.wait(0.2)
                continue
            self._last_fail = ""
            result.extra["session_id"] = self.session_id
            result.extra["style"] = self.style_index
            if self.style_index >= 1:
                result.extra["dashed"] = True
            if runner.info.decoder != last_decoder:
                last_decoder = runner.info.decoder
                self._post("info", None)
            self._post_result(result)

    def _apply_commands(self, runner: InferenceRunner) -> None:
        while True:
            try:
                cmd = self._cmds.get_nowait()
            except queue.Empty:
                return
            if cmd[0] == "decoder":
                try:
                    runner.set_decoder(cmd[1], cmd[2])
                    self._last_fail = ""
                    self._post("info", None)
                except InferenceError as e:
                    self._fail(e.message, e.hint)

    def _post_result(self, result: InferenceResult) -> None:
        with self._lock:
            self._latest = result
            if self._notified:
                return                 # GUI 還沒處理上一筆：只保留最新的
            self._notified = True
        self._post("result", None)

    # ---- GUI 執行緒 -------------------------------------------------------------
    def _dispatch(self, kind: str, payload: Any) -> None:
        if self._stop.is_set() and kind != "failed":
            return
        if kind == "loaded":
            self.state = "running"
            self.info = payload
            self.loaded.emit(payload)
        elif kind == "failed":
            self.state = "failed"
            self.last_error = dict(payload)
            self.failed.emit(dict(payload))
        elif kind == "result":
            with self._lock:
                result, self._latest, self._notified = self._latest, None, False
            if result is None:
                return
            self.state = "running"
            self.last_result = result
            self._times.append(time.monotonic())
            self._infer_ms.append(float(result.timings_ms.get("total", 0.0)))
            pkt = self.handle.last_packet
            self._age = max(0, pkt.frame.seq - result.frame_seq) if pkt is not None else 0
            self.resultReady.emit(result)
        elif kind == "raw":
            self.raw_outputs = payload
            self.rawReady.emit(payload)
        elif kind == "info":
            self.infoChanged.emit()


class InferenceManager(QObject):
    """所有 session 的集合；主視窗、模型面板、比較面板共用。"""

    sessionsChanged = Signal()
    resultReady = Signal(object, object, object)   # handle、session、InferenceResult

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._sessions: list[InferenceSession] = []

    def add(self, handle, card: ModelCard, **kw: Any) -> InferenceSession:
        s = InferenceSession(handle, card, parent=self, **kw)
        s.style_index = len(self.sessions_for(handle))
        s.resultReady.connect(lambda r, s=s: self.resultReady.emit(s.handle, s, r))
        self._sessions.append(s)
        s.start()
        self.sessionsChanged.emit()
        return s

    def add_device(self, handle, card: ModelCard, *, on_progress: Callable[[float, str], None] | None = None,
                   on_done: Callable[[InferenceSession | None, Any], None] | None = None) -> None:
        """K210 板上推論：先讓板子載入模型（run_action("load_model")），成功才建立 session。"""

        def progress(key: str, ratio: float, text: str) -> None:
            if key == "load_model" and on_progress is not None:
                on_progress(ratio, text)

        def finished(key: str, res: Any) -> None:
            if key != "load_model":
                return
            for sig, slot in ((handle.actionProgress, progress), (handle.actionFinished, finished)):
                try:
                    sig.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
            session = self.add(handle, card, backend="device") if res.ok else None
            if on_done is not None:
                on_done(session, res)

        handle.actionProgress.connect(progress)
        handle.actionFinished.connect(finished)
        handle.run_action("load_model", card=card)

    def remove(self, session: InferenceSession, *, unload: bool = True) -> None:
        if session not in self._sessions:
            return
        session.stop()
        self._sessions.remove(session)
        if unload and session.backend == "device":
            try:
                session.handle.run_action("unload_model")
            except Exception:  # noqa: BLE001 - 相機已關閉等情況
                log.warning("卸載板上模型失敗", exc_info=True)
        for i, s in enumerate(self.sessions_for(session.handle)):
            s.style_index = i
        self.sessionsChanged.emit()
        session.deleteLater()

    def remove_for_handle(self, handle, *, unload: bool = False) -> None:
        for s in self.sessions_for(handle):
            self.remove(s, unload=unload)

    def sessions(self) -> list[InferenceSession]:
        return list(self._sessions)

    def sessions_for(self, handle) -> list[InferenceSession]:
        return [s for s in self._sessions if s.handle is handle]

    def primary(self, handle) -> InferenceSession | None:
        found = self.sessions_for(handle)
        return found[0] if found else None

    def last_result(self, handle) -> InferenceResult | None:
        s = self.primary(handle)
        return s.last_result if s is not None else None

    def shutdown(self, timeout: float = 2.0) -> bool:
        """全部通知停止，再共用同一個期限逐一等待（總時間不會變成相加）。"""
        for s in self._sessions:
            s.request_stop()
        deadline = time.monotonic() + timeout
        ok = True
        for s in list(self._sessions):
            ok = s.stop(max(0.05, deadline - time.monotonic())) and ok
        self._sessions.clear()
        self.sessionsChanged.emit()
        return ok


def make_color_resolver(ctx) -> Callable[[InferenceResult, int, str], str | None]:
    """VideoView 用的上色函式：模型的類別名稱和專案類別（id 或任一語言的顯示名稱）相同就用專案顏色，
    否則用色盤（比較用的第二個模型用另一組色盤，才分得出是哪個模型畫的）。"""

    def resolve(res: InferenceResult, class_id: int, label: str) -> str | None:
        project = ctx.project
        key = (label or "").strip().casefold()
        if project is not None and key:
            for c in project.classes:
                names = [c.id, *c.names.values()]
                if any(n and n.strip().casefold() == key for n in names):
                    return c.color
        style = int(res.extra.get("style", 0)) if isinstance(res.extra, dict) else 0
        palette = DEFAULT_COLORS if style == 0 else ALT_COLORS
        return palette[class_id % len(palette)]

    return resolve
