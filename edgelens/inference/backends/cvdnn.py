"""OpenCV DNN 後端：不需要額外套件就能跑 ONNX、Darknet（YOLOv3／v4）、TensorFlow、Caffe、TFLite。

OpenCV 讀不到模型的輸出形狀，所以載入時先用全零張量跑一次，記下每個輸出的形狀。
Darknet 的輸出已經由 OpenCV 解碼成 [N, 5+C]（中心、寬高為 0–1 正規化），多個 YOLO 層的輸出
由 runner 合併後交給 yolo_flat 解碼器。
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

import cv2
import numpy as np

from edgelens.core.modelcard import ModelCard
from edgelens.i18n import tr
from edgelens.inference.backends._common import read_model_bytes
from edgelens.inference.base import Backend, InferenceError, TensorInfo, register_backend

_NEEDS_CONFIG = {".weights": "cfg", ".pb": "pbtxt", ".caffemodel": "prototxt"}


def cfg_input_size(cfg_text: str) -> tuple[int, int] | None:
    """從 Darknet .cfg 的 [net] 區段讀 width／height。"""
    net = cfg_text.split("[net]", 1)[-1].split("\n[", 1)[0]
    w = re.search(r"^\s*width\s*=\s*(\d+)", net, re.M)
    h = re.search(r"^\s*height\s*=\s*(\d+)", net, re.M)
    return (int(w.group(1)), int(h.group(1))) if w and h else None


def _onnx_probe(data: bytes) -> tuple[tuple[int | None, ...] | None, list[str]]:
    """有裝 onnx 套件時，讀出輸入形狀與輸出名稱（OpenCV 自己拿不到）；沒裝就回傳 (None, [])。"""
    try:
        import onnx
    except ImportError:
        return None, []
    try:
        model = onnx.load_model_from_string(data)
        init = {i.name for i in model.graph.initializer}
        ins = [i for i in model.graph.input if i.name not in init]
        shape = None
        if ins:
            dims = ins[0].type.tensor_type.shape.dim
            shape = tuple(int(d.dim_value) if d.dim_value > 0 else None for d in dims)
        return shape, [o.name for o in model.graph.output]
    except Exception:  # noqa: BLE001 - 探測失敗不影響載入
        return None, []


@register_backend
class OpenCvDnnBackend(Backend):
    kind: ClassVar[str] = "cvdnn"
    display_name_key: ClassVar[str] = "backend.cvdnn.name"
    extensions: ClassVar[tuple[str, ...]] = (".onnx", ".weights", ".pb", ".caffemodel", ".tflite")
    requires: ClassVar[tuple[str, ...]] = ()
    input_mode = "tensor"
    priority: ClassVar[int] = 10

    def __init__(self, card: ModelCard):
        super().__init__(card)
        self._net: Any = None
        self._out_names: list[str] = []
        self._in = TensorInfo("input", (1, 3, None, None))
        self._out: list[TensorInfo] = []
        self._darknet = False

    @classmethod
    def can_load(cls, card: ModelCard) -> bool:
        ext = card.extension
        if ext not in cls.extensions:
            return False
        if ext in _NEEDS_CONFIG:
            cfg = card.extra_path("config")
            return bool(cfg and cfg.is_file())
        if ext == ".tflite":
            return hasattr(cv2.dnn, "readNetFromTFLite")
        return True

    # ---- 載入 ------------------------------------------------------------------

    def _read_net(self) -> tuple[Any, tuple[int, int] | None, list[str]]:
        card = self.card
        ext = card.extension
        data = read_model_bytes(card.model_path)
        size: tuple[int, int] | None = None
        order: list[str] = []
        if ext == ".onnx":
            shape, order = _onnx_probe(data)
            if shape and len(shape) == 4 and shape[2] and shape[3]:
                size = (shape[3], shape[2])
            return cv2.dnn.readNetFromONNX(_buf(data)), size, order
        cfg_path = card.extra_path("config")
        if ext in _NEEDS_CONFIG and (cfg_path is None or not cfg_path.is_file()):
            raise InferenceError(tr("backend.err.cvdnn_config", ext=ext), hint=tr("backend.err.cvdnn_config_hint"))
        cfg = read_model_bytes(cfg_path) if cfg_path else b""
        if ext == ".weights":
            self._darknet = True
            size = cfg_input_size(cfg.decode("utf-8", errors="replace"))
            return cv2.dnn.readNetFromDarknet(_buf(cfg), _buf(data)), size, order
        if ext == ".pb":
            return cv2.dnn.readNetFromTensorflow(_buf(data), _buf(cfg)), size, order
        if ext == ".caffemodel":
            return cv2.dnn.readNetFromCaffe(_buf(cfg), _buf(data)), size, order
        if ext == ".tflite":
            if not hasattr(cv2.dnn, "readNetFromTFLite"):
                raise InferenceError(tr("backend.err.cvdnn_no_tflite", version=cv2.__version__),
                                     hint=tr("backend.err.cvdnn_no_tflite_hint"))
            return cv2.dnn.readNetFromTFLite(_buf(data)), size, order
        raise InferenceError(tr("backend.err.cvdnn_ext", ext=ext), hint="")

    def load(self) -> None:
        try:
            net, size, order = self._read_net()
        except InferenceError:
            raise
        except cv2.error as e:
            raise InferenceError(tr("backend.err.cvdnn_load", error=_cv_msg(e)),
                                 hint=tr("backend.err.cvdnn_load_hint")) from e
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            names = list(net.getUnconnectedOutLayersNames())
        except cv2.error as e:
            raise InferenceError(tr("backend.err.cvdnn_load", error=_cv_msg(e)),
                                 hint=tr("backend.err.cvdnn_load_hint")) from e
        # 多輸出 ONNX：OpenCV 的輸出順序不一定和模型相同，能對上名稱時照模型的順序排
        if order and sorted(order) == sorted(names):
            names = order
        self._net, self._out_names = net, names

        spec = self.card.input
        w, h = spec.size or size or (640, 640)
        self._in = TensorInfo("input", (1, 3, int(h), int(w)))
        self._probe(int(w), int(h))

    def _probe(self, w: int, h: int) -> None:
        """全零張量跑一次，取得各輸出的形狀。"""
        try:
            outs = self._forward(np.zeros((1, 3, h, w), np.float32))
        except InferenceError as e:
            raise InferenceError(tr("backend.err.cvdnn_probe", w=w, h=h, error=e.message),
                                 hint=tr("backend.err.cvdnn_probe_hint")) from e
        self._out = [TensorInfo(n, tuple(int(d) for d in o.shape), "float32")
                     for n, o in zip(self._out_names, outs, strict=False)]

    def _forward(self, tensor: np.ndarray) -> list[np.ndarray]:
        try:
            self._net.setInput(tensor)
            res = self._net.forward(self._out_names)
        except cv2.error as e:
            raise InferenceError(tr("backend.err.infer", error=_cv_msg(e)), hint=tr("backend.err.infer_hint")) from e
        if isinstance(res, np.ndarray):
            res = [res]
        return [np.asarray(r) for r in res]

    # ---- 推論 ------------------------------------------------------------------

    def inputs(self) -> list[TensorInfo]:
        return [self._in]

    def outputs(self) -> list[TensorInfo]:
        return self._out

    def infer(self, tensor: np.ndarray) -> list[np.ndarray]:
        if self._net is None:
            raise InferenceError(tr("backend.err.not_loaded"))
        return self._forward(np.ascontiguousarray(tensor, dtype=np.float32))

    def metadata(self) -> dict[str, Any]:
        return {"darknet": True} if self._darknet else {}

    def describe(self) -> str:
        return tr("backend.cvdnn.desc", version=cv2.__version__, threads=cv2.getNumThreads())

    def close(self) -> None:
        self._net = None


def _buf(data: bytes) -> np.ndarray:
    """OpenCV 的記憶體載入函式要 uint8 陣列；直接傳 bytes 在部分版本會當機（segfault）。"""
    return np.frombuffer(data, dtype=np.uint8)


def _cv_msg(e: Exception) -> str:
    """OpenCV 的錯誤訊息很長，取 'error:' 後第一行就夠學生看。"""
    text = str(e)
    m = re.search(r"error:\s*\(-?\d+:[^)]*\)\s*(.*)", text)
    line = (m.group(1) if m else text).strip().splitlines()
    return (line[0] if line else repr(e))[:300]
