"""Haar cascade 後端：OpenCV 的傳統臉部／物件偵測（.xml）。不需要神經網路，直接吃影像。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np

from edgelens.core.modelcard import ModelCard
from edgelens.core.results import Detection, InferenceResult
from edgelens.i18n import tr, tr_or
from edgelens.inference.backends._common import read_model_bytes
from edgelens.inference.base import Backend, InferenceError, register_backend

DEFAULTS: dict[str, Any] = {"scale_factor": 1.1, "min_neighbors": 5, "min_size": [30, 30], "max_size": None}


def builtin_cascades() -> dict[str, Path]:
    """OpenCV 內附的 Haar cascade：{名稱: 路徑}（正臉、側臉、眼睛、笑臉、全身…），學生不用找檔案。
    名稱是檔名去掉 haarcascade_ 前綴，例如 frontalface_default。"""
    try:
        folder = Path(cv2.data.haarcascades)
    except AttributeError:
        return {}
    out: dict[str, Path] = {}
    for p in sorted(folder.glob("haarcascade_*.xml")):
        out[p.stem.removeprefix("haarcascade_")] = p
    return out


def cascade_display_name(name: str) -> str:
    """內附 cascade 的顯示名稱（有翻譯就翻譯，沒有就用原名）。"""
    return tr_or(f"haar.cascade.{name}", name)


@register_backend
class HaarBackend(Backend):
    kind: ClassVar[str] = "haar"
    display_name_key: ClassVar[str] = "backend.haar.name"
    extensions: ClassVar[tuple[str, ...]] = (".xml",)
    requires: ClassVar[tuple[str, ...]] = ()
    input_mode = "image"
    priority: ClassVar[int] = 10

    def __init__(self, card: ModelCard):
        super().__init__(card)
        self._cascade: cv2.CascadeClassifier | None = None

    def load(self) -> None:
        path = self.card.model_path
        data = read_model_bytes(path)
        cascade = cv2.CascadeClassifier()
        ok = False
        try:
            ok = bool(cascade.load(str(path)))
        except cv2.error:
            ok = False
        if not ok:
            # 中文路徑時 OpenCV 可能讀不到，改從記憶體內容載入
            try:
                fs = cv2.FileStorage(data.decode("utf-8", errors="replace"),
                                     cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
                ok = bool(cascade.read(fs.getFirstTopLevelNode()))
                fs.release()
            except (cv2.error, SystemError, ValueError):
                ok = False
        if not ok or cascade.empty():
            raise InferenceError(tr("backend.err.haar_load", path=path.name), hint=tr("backend.err.haar_load_hint"))
        self._cascade = cascade

    def _param(self, key: str) -> Any:
        v = self.card.decoder_params.get(key)
        return DEFAULTS[key] if v is None else v

    def detect_image(self, image: np.ndarray, frame_seq: int = -1) -> InferenceResult:
        if self._cascade is None:
            raise InferenceError(tr("backend.err.not_loaded"))
        t0 = time.perf_counter()
        img = np.asarray(image)
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        h, w = gray.shape[:2]
        min_size = tuple(int(x) for x in self._param("min_size"))
        max_size = tuple(int(x) for x in self._param("max_size")) if self._param("max_size") else (0, 0)
        try:
            rects, _levels, weights = self._cascade.detectMultiScale3(
                gray, scaleFactor=float(self._param("scale_factor")), minNeighbors=int(self._param("min_neighbors")),
                minSize=min_size, maxSize=max_size, outputRejectLevels=True)
        except cv2.error as e:
            raise InferenceError(tr("backend.err.infer", error=str(e)[:200]), hint=tr("backend.err.infer_hint")) from e
        label = self.card.label(0) if self.card.labels else self._default_label()
        dets = [Detection(box=(float(x), float(y), float(x + bw), float(y + bh)),
                          # Haar 沒有真正的機率；用累積權重換成 0–1 的「大概有多確定」
                          score=float(1.0 - np.exp(-max(float(wt), 0.0) / 3.0)), class_id=0, label=label)
                for (x, y, bw, bh), wt in zip(rects, np.ravel(weights), strict=False)]
        dets.sort(key=lambda d: -d.score)
        ms = (time.perf_counter() - t0) * 1000.0
        return InferenceResult(kind="detect", model_name=self.card.name, frame_seq=frame_seq, where="host",
                               detections=dets, timings_ms={"infer": ms}, image_size=(w, h))

    def _default_label(self) -> str:
        stem = self.card.model_path.stem.removeprefix("haarcascade_")
        return cascade_display_name(stem)

    def describe(self) -> str:
        return tr("backend.haar.desc", scale=self._param("scale_factor"), neighbors=self._param("min_neighbors"),
                  size=self._param("min_size")[0])

    def close(self) -> None:
        self._cascade = None
