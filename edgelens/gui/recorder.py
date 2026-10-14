"""錄影與數據記錄（非 GUI 邏輯，方便單元測試）。

Recorder 在背景執行緒把畫面寫成 MP4，同時用 CsvLogger 每張畫面寫一列數據。GUI 執行緒只做
「把畫面放進佇列」這件事，佇列滿了就丟畫面並計數，所以寫檔再慢也不會讓介面卡住、記憶體
也不會越積越多。

Windows 的 cv2.VideoWriter 不支援非 ASCII 路徑（使用者名稱或資料夾是中文時會失敗），
所以影片先寫到 ASCII 的暫存資料夾，結束後再搬到專案的 recordings/。

mode：
- "raw"       原始畫面（相機送來的）
- "processed" 處理後的畫面
- "annotated" 含標註（呼叫端在 GUI 執行緒用 VideoView.render_annotated() 產生，再傳進來）
- "data"      只記錄數據 CSV，不錄影
"""

from __future__ import annotations

import logging
import math
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from edgelens.core.housekeeping import REC_TEMP_PREFIX, temp_base_candidates
from edgelens.core.paths import ensure_dir, safe_filename
from edgelens.metrics.csvlog import CsvLogger

log = logging.getLogger(__name__)

MODES = ("raw", "processed", "annotated", "data")
BASE_FIELDS = ["t_s", "time", "seq", "fps", "latency_ms", "process_ms",
               "infer_label", "infer_score", "infer_count", "infer_ms", "infer_seq"]


@dataclass
class RecordingResult:
    """錄影結束後的摘要。video_path 為 None 表示沒有影片（數據模式或寫入失敗）。"""

    video_path: Path | None
    csv_path: Path | None
    frames: int
    dropped: int
    seconds: float
    size_bytes: int
    error: str = ""


def _is_ascii(text: str) -> bool:
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def ascii_temp_dir() -> Path:
    """找一個路徑全是 ASCII 的暫存資料夾（Windows 使用者名稱是中文時，預設暫存路徑會失敗）。"""
    for base in temp_base_candidates():   # 清單與開機清理共用，兩邊才不會不一致
        if not _is_ascii(str(base)):
            continue
        try:
            ensure_dir(base)
            return Path(tempfile.mkdtemp(prefix=REC_TEMP_PREFIX, dir=str(base)))
        except OSError:
            continue
    return Path(tempfile.mkdtemp(prefix=REC_TEMP_PREFIX))


def fourcc_candidates() -> list[tuple[str, str]]:
    """(副檔名, fourcc)：依平台排順序，最後退回 MJPG 的 .avi（幾乎每台電腦都能寫）。"""
    order = ["avc1", "mp4v"] if sys.platform == "darwin" else ["mp4v", "avc1"]
    return [(".mp4", cc) for cc in order] + [(".avi", "MJPG")]


def qimage_to_bgr(qimage: Any) -> np.ndarray | None:
    """QImage → BGR ndarray（含標註模式用）。只能在 GUI 執行緒呼叫；空圖回傳 None。"""
    from PySide6.QtGui import QImage

    if qimage is None or qimage.isNull():
        return None
    img = qimage.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    arr = np.frombuffer(ptr, np.uint8, count=img.bytesPerLine() * h).reshape(h, img.bytesPerLine())
    rgb = arr[:, : w * 3].reshape(h, w, 3)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _num(v: Any) -> float | int | None:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)) and math.isfinite(v):
        return v
    return None


def flatten_data(data: dict[str, Any]) -> dict[str, Any]:
    """ProcessResult.data → CSV 欄位：數值直接記；清單／字典記成 <名稱>_count（個數）。"""
    out: dict[str, Any] = {}
    for k, v in (data or {}).items():
        n = _num(v)
        if n is not None:
            out[f"data.{k}"] = n
        elif isinstance(v, (list, tuple, dict, set)):
            out[f"data.{k}_count"] = len(v)
    return out


def flatten_board(meta: dict[str, Any]) -> dict[str, Any]:
    """Frame.meta 裡的板上數據（K210）：fps、各步驟耗時（毫秒）、記憶體。沒有就是空的。"""
    out: dict[str, Any] = {}
    for k in ("fps", "mem", "heap"):
        n = _num(meta.get(k))
        if n is not None:
            out[f"board.{k}"] = n
    n = _num(meta.get("lat"))
    if n is not None:
        out["board.lat_ms"] = round(n / 1000.0, 3)
    us = meta.get("us")
    if isinstance(us, dict):
        for k, v in us.items():
            n = _num(v)
            if n is not None:
                out[f"board.us.{k}_ms"] = round(n / 1000.0, 3)
    return out


def summarize_inference(result: Any) -> dict[str, Any]:
    """InferenceResult → 一列用的摘要（最佳類別、分數、偵測數、耗時）。"""
    best = getattr(result, "best", None)
    out: dict[str, Any] = {"infer_label": "", "infer_score": "", "infer_count": 0, "infer_ms": ""}
    if best is not None:
        out["infer_label"] = getattr(best, "label", "") or str(getattr(best, "class_id", ""))
        out["infer_score"] = round(float(best.score), 4)
    kind = getattr(result, "kind", "")
    items = getattr(result, "detections", []) if kind == "detect" else getattr(result, "top_k", [])
    out["infer_count"] = len(items)
    t = (getattr(result, "timings_ms", None) or {}).get("total")
    if t is None:
        t = (getattr(result, "timings_ms", None) or {}).get("infer")
    if t is not None:
        out["infer_ms"] = round(float(t), 2)
    return out


class Recorder:
    def __init__(self, out_dir: str | Path, camera_name: str, *, mode: str = "processed", fps: float = 30.0,
                 max_queue: int = 30, csv_dir: str | Path | None = None, suffix: str = ""):
        if mode not in MODES:
            raise ValueError(f"未知的錄影模式：{mode}")
        self.mode = mode
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self._dir = Path(out_dir)
        self._csv_dir = Path(csv_dir) if csv_dir else self._dir
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.base_name = safe_filename(f"{stamp}_{camera_name}{suffix}")
        self.csv_path = self._csv_dir / f"{self.base_name}.csv"
        self.video_path: Path | None = None
        self._queue: queue.Queue[tuple | None] = queue.Queue(maxsize=max(1, int(max_queue)))
        self._thread: threading.Thread | None = None
        self._tmp: Path | None = None
        self._tmp_video: Path | None = None
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None
        self._csv: CsvLogger | None = None
        self._lock = threading.Lock()
        self._infer: dict[int, dict[str, Any]] = {}
        self._infer_order: deque[int] = deque(maxlen=128)
        self._latest_infer: dict[str, Any] | None = None
        self._t0 = 0.0
        self._t_end = 0.0
        self.frames = 0          # 已寫入的張數
        self.dropped = 0         # 因佇列滿而丟掉的張數
        self.error = ""
        self._video_failed = False
        self._running = False
        self._result: RecordingResult | None = None

    # ---- 狀態 -------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._running

    @property
    def elapsed(self) -> float:
        if not self._t0:
            return 0.0
        return (time.monotonic() if self._running else self._t_end) - self._t0

    @property
    def size_bytes(self) -> int:
        p = self._tmp_video
        try:
            return os.path.getsize(p) if p is not None else 0
        except OSError:
            return 0

    # ---- 開始與停止 -------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        ensure_dir(self._dir)
        ensure_dir(self._csv_dir)
        self._csv = CsvLogger(self.csv_path, BASE_FIELDS, flush_interval=1.0)
        if self.mode != "data":
            self._tmp = ascii_temp_dir()
        self._t0 = time.monotonic()
        self._running = True
        self._thread = threading.Thread(target=self._run, name="Recorder", daemon=True)
        self._thread.start()

    def stop(self) -> RecordingResult:
        """停止、寫完佇列、關閉檔案並把影片搬到目的地。可重複呼叫。"""
        if self._result is not None:
            return self._result
        if self._running:
            self._running = False
            self._t_end = time.monotonic()
            self._queue.put(None)
            if self._thread is not None:
                self._thread.join(timeout=60)
        final = self._finish_files()
        self._result = RecordingResult(
            video_path=final, csv_path=self.csv_path, frames=self.frames, dropped=self.dropped,
            seconds=self.elapsed, size_bytes=self._final_size(final), error=self.error)
        return self._result

    def _final_size(self, final: Path | None) -> int:
        try:
            return os.path.getsize(final) if final is not None else 0
        except OSError:
            return 0

    def _finish_files(self) -> Path | None:
        try:
            if self._writer is not None:
                self._writer.release()
        except Exception as e:  # noqa: BLE001
            self._set_error(f"關閉影片失敗：{e}")
        self._writer = None
        if self._csv is not None:
            try:
                self._csv.close()
            except Exception as e:  # noqa: BLE001
                self._set_error(f"關閉 CSV 失敗：{e}")
        final: Path | None = None
        if self._tmp_video is not None and self._tmp_video.is_file():
            target = self._dir / f"{self.base_name}{self._tmp_video.suffix}"
            try:
                shutil.move(str(self._tmp_video), str(target))
                final = target
            except OSError as e:
                self._set_error(f"無法把影片存到 {target}：{e}。影片留在 {self._tmp_video}")
        self.video_path = final
        if self._tmp is not None and final is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
        return final

    # ---- 加入資料 ---------------------------------------------------------------
    def add_packet(self, packet: Any, image: np.ndarray | None = None) -> bool:
        """加入一張畫面（GUI 執行緒呼叫，不會阻塞）。回傳 False 表示因佇列滿而丟掉。

        image：annotated 模式由呼叫端提供含標註的 BGR 圖；沒有就退回處理後的圖。"""
        if not self._running:
            return False
        frame, result = packet.frame, packet.result
        stats = getattr(packet, "stats", {}) or {}
        row: dict[str, Any] = {
            "t_s": round(time.monotonic() - self._t0, 3),
            "time": time.strftime("%H:%M:%S", time.localtime()) + f".{int(time.time() * 1000) % 1000:03d}",
            "seq": frame.seq,
            "fps": round(float(stats.get("fps", 0.0)), 2),
            "latency_ms": round(float(stats.get("latency_ms", 0.0)), 2),
            "process_ms": round(float(stats.get("process_ms", 0.0)), 3),
        }
        row.update(flatten_data(getattr(result, "data", {})))
        row.update(flatten_board(getattr(frame, "meta", {}) or {}))
        img: np.ndarray | None = None
        if self.mode == "raw":
            img = frame.image
        elif self.mode == "processed":
            img = result.image
        elif self.mode == "annotated":
            img = image if image is not None else result.image
        return self._put((img, row, frame.seq))

    def add_row(self, row: dict[str, Any], image: np.ndarray | None = None, seq: int = -1) -> bool:
        """加入自己組好的一列（測試或不是 FramePacket 的來源用）。"""
        if not self._running:
            return False
        full = {"t_s": round(time.monotonic() - self._t0, 3), "seq": seq, **row}
        return self._put((image, full, seq))

    def _put(self, item: tuple) -> bool:
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def add_inference(self, seq: int, result: Any) -> None:
        """推論面板每算完一張就呼叫。CSV 會記下「寫入該列時最新的推論結果」與它對應的 seq。"""
        summary = summarize_inference(result)
        summary["infer_seq"] = int(seq)
        with self._lock:
            if len(self._infer_order) == self._infer_order.maxlen:
                self._infer.pop(self._infer_order[0], None)
            self._infer_order.append(int(seq))
            self._infer[int(seq)] = summary
            self._latest_infer = summary

    # ---- 背景執行緒 -------------------------------------------------------------
    def _set_error(self, msg: str) -> None:
        log.warning("錄影：%s", msg)
        if not self.error:
            self.error = msg

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                self._write(*item)
            except Exception as e:  # noqa: BLE001 - 單張失敗不能讓整個錄影中斷
                self._set_error(f"寫入失敗：{e}")

    def _write(self, image: np.ndarray | None, row: dict[str, Any], seq: int) -> None:
        if image is not None and self.mode != "data" and not self._video_failed:
            self._write_image(image)
        with self._lock:
            infer = self._infer.get(seq) or self._latest_infer
        if infer:
            row.update(infer)
        assert self._csv is not None
        self._csv.add(row)
        self.frames += 1

    def _write_image(self, image: np.ndarray) -> None:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if self._writer is None:
            h, w = image.shape[:2]
            w, h = w - w % 2, h - h % 2   # 編碼器通常要偶數邊長
            self._size = (max(2, w), max(2, h))
            self._open_writer(self._size)
            if self._writer is None:
                self._video_failed = True
                self._set_error("這台電腦無法建立影片檔。數據 CSV 仍會繼續記錄。")
                return
        assert self._size is not None
        if (image.shape[1], image.shape[0]) != self._size:
            image = cv2.resize(image, self._size)
        self._writer.write(np.ascontiguousarray(image))   # type: ignore[union-attr]

    def _open_writer(self, size: tuple[int, int]) -> None:
        assert self._tmp is not None
        for ext, cc in fourcc_candidates():
            path = self._tmp / f"rec{ext}"
            try:
                w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*cc), self.fps, size)
            except Exception as e:  # noqa: BLE001
                log.info("fourcc %s 不能用：%s", cc, e)
                continue
            if w.isOpened():
                self._writer, self._tmp_video = w, path
                log.info("錄影編碼：%s %s", ext, cc)
                return
            w.release()
