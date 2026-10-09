"""TFLite 後端：跑 .tflite 模型（SSD MobileNet、Teachable Machine 匯出的分類模型等）。

解譯器依序試 ai_edge_litert（Google 新版）與 tflite_runtime（舊版），都沒有時 available() 回傳 False，
並建議改用 ONNX。
"""

from __future__ import annotations

import io
import zipfile
from typing import Any, ClassVar

import numpy as np

from edgelens.core.modelcard import ModelCard
from edgelens.core.registry import module_available
from edgelens.i18n import tr
from edgelens.inference.backends._common import card_threads, read_model_bytes
from edgelens.inference.base import Backend, InferenceError, TensorInfo, register_backend

INTERPRETER_MODULES = ("ai_edge_litert.interpreter", "tflite_runtime.interpreter")


def _find_interpreter_module() -> str | None:
    for m in INTERPRETER_MODULES:
        if module_available(m):
            return m
    return None


def _info(d: dict[str, Any]) -> TensorInfo:
    sig = d.get("shape_signature")
    shape_src = sig if sig is not None and len(sig) == len(d["shape"]) else d["shape"]
    shape = tuple(int(x) if int(x) > 0 else None for x in shape_src)
    dtype = np.dtype(d["dtype"]).name
    quant = None
    q = d.get("quantization") or (0.0, 0)
    if dtype in ("uint8", "int8", "int16") and q and float(q[0]) != 0.0:
        quant = (float(q[0]), int(q[1]))
    return TensorInfo(str(d["name"]), shape, dtype, quant)


def _labels_from_metadata(data: bytes) -> list[str]:
    """TFLite Task Library 的模型把類別名稱檔用 zip 附在 .tflite 後面，有的話順便讀出來。"""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                if name.lower().endswith(".txt"):
                    lines = z.read(name).decode("utf-8-sig", errors="replace").splitlines()
                    return [ln.strip() for ln in lines if ln.strip()]
    except (zipfile.BadZipFile, OSError, KeyError):
        pass
    return []


@register_backend
class TfliteBackend(Backend):
    kind: ClassVar[str] = "tflite"
    display_name_key: ClassVar[str] = "backend.tflite.name"
    extensions: ClassVar[tuple[str, ...]] = (".tflite",)
    requires: ClassVar[tuple[str, ...]] = ()      # 套件名稱不固定，available() 自己檢查
    input_mode = "tensor"
    priority: ClassVar[int] = 20

    def __init__(self, card: ModelCard):
        super().__init__(card)
        self._interp: Any = None
        self._in: list[TensorInfo] = []
        self._out: list[TensorInfo] = []
        self._in_idx: list[int] = []
        self._out_idx: list[int] = []
        self._threads = card_threads(card)
        self._module = ""
        self._data = b""

    @classmethod
    def available(cls) -> tuple[bool, str]:
        if _find_interpreter_module() is None:
            return False, tr("backend.tflite.missing")
        return True, ""

    def load(self) -> None:
        mod = _find_interpreter_module()
        if mod is None:
            raise InferenceError(tr("backend.tflite.missing"), hint=tr("backend.tflite.missing_hint"))
        import importlib

        self._module = mod.split(".")[0]
        interpreter_cls = importlib.import_module(mod).Interpreter
        self._data = read_model_bytes(self.card.model_path)
        try:
            self._interp = interpreter_cls(model_content=self._data, num_threads=self._threads)
            self._interp.allocate_tensors()
        except Exception as e:  # noqa: BLE001
            raise InferenceError(tr("backend.err.tflite_load", error=str(e).strip().splitlines()[0] if str(e) else repr(e)),
                                 hint=tr("backend.err.tflite_load_hint")) from e
        ins, outs = self._interp.get_input_details(), self._interp.get_output_details()
        self._in = [_info(d) for d in ins]
        self._out = [_info(d) for d in outs]
        self._in_idx = [int(d["index"]) for d in ins]
        self._out_idx = [int(d["index"]) for d in outs]

    def inputs(self) -> list[TensorInfo]:
        return self._in

    def outputs(self) -> list[TensorInfo]:
        return self._out

    def infer(self, tensor: np.ndarray) -> list[np.ndarray]:
        if self._interp is None:
            raise InferenceError(tr("backend.err.not_loaded"))
        try:
            want = tuple(self._interp.get_input_details()[0]["shape"])
            if tuple(tensor.shape) != want:       # 動態輸入大小：依這次的張量重新配置
                self._interp.resize_tensor_input(self._in_idx[0], list(tensor.shape))
                self._interp.allocate_tensors()
            self._interp.set_tensor(self._in_idx[0], tensor)
            self._interp.invoke()
            return [np.array(self._interp.get_tensor(i)) for i in self._out_idx]
        except Exception as e:  # noqa: BLE001
            raise InferenceError(tr("backend.err.infer", error=str(e).strip().splitlines()[0] if str(e) else repr(e)),
                                 hint=tr("backend.err.infer_hint")) from e

    def metadata(self) -> dict[str, Any]:
        names = _labels_from_metadata(self._data) if self._data else []
        return {"names": names} if names else {}

    def describe(self) -> str:
        return tr("backend.tflite.desc", module=self._module or "-", threads=self._threads)

    def close(self) -> None:
        self._interp = None
