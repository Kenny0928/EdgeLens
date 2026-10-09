"""ONNX Runtime 後端：跑 .onnx 模型。預設用 CPU，穩定且每個平台結果一致。"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np

from edgelens.core.modelcard import ModelCard
from edgelens.i18n import tr
from edgelens.inference.backends._common import card_threads, dtype_name, parse_names, read_model_bytes
from edgelens.inference.base import Backend, InferenceError, TensorInfo, register_backend

log = logging.getLogger(__name__)


def _dims(shape: Any) -> tuple[int | None, ...]:
    """ONNX 的動態維度是字串或 None，統一成 None。"""
    return tuple(int(d) if isinstance(d, (int, np.integer)) and d > 0 else None for d in (shape or ()))


@register_backend
class OnnxRuntimeBackend(Backend):
    kind: ClassVar[str] = "onnxrt"
    display_name_key: ClassVar[str] = "backend.onnxrt.name"
    extensions: ClassVar[tuple[str, ...]] = (".onnx",)
    requires: ClassVar[tuple[str, ...]] = ("onnxruntime",)
    input_mode = "tensor"
    priority: ClassVar[int] = 20

    def __init__(self, card: ModelCard):
        super().__init__(card)
        self._sess: Any = None
        self._in: list[TensorInfo] = []
        self._out: list[TensorInfo] = []
        self._threads = card_threads(card)
        self._providers: list[str] = []

    def load(self) -> None:
        ok, reason = self.available()
        if not ok:
            raise InferenceError(tr("backend.err.unavailable", name=tr(self.display_name_key), reason=reason),
                                 hint=tr("backend.err.install_hint"))
        import onnxruntime as ort

        data = read_model_bytes(self.card.model_path)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self._threads
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3          # 只顯示錯誤，避免終端機被警告洗版
        try:
            self._sess = ort.InferenceSession(data, sess_options=opts, providers=["CPUExecutionProvider"])
        except Exception as e:  # noqa: BLE001 - onnxruntime 的錯誤型別很多種
            raise InferenceError(tr("backend.err.onnx_load", error=str(e).strip().splitlines()[0] if str(e) else repr(e)),
                                 hint=tr("backend.err.onnx_load_hint")) from e
        self._providers = list(self._sess.get_providers())
        self._in = [TensorInfo(i.name, _dims(i.shape), dtype_name(i.type)) for i in self._sess.get_inputs()]
        self._out = [TensorInfo(o.name, _dims(o.shape), dtype_name(o.type)) for o in self._sess.get_outputs()]

    def inputs(self) -> list[TensorInfo]:
        return self._in

    def outputs(self) -> list[TensorInfo]:
        return self._out

    def infer(self, tensor: np.ndarray) -> list[np.ndarray]:
        if self._sess is None:
            raise InferenceError(tr("backend.err.not_loaded"), hint="")
        try:
            return list(self._sess.run(None, {self._in[0].name: tensor}))
        except Exception as e:  # noqa: BLE001
            raise InferenceError(tr("backend.err.infer", error=str(e).strip().splitlines()[0] if str(e) else repr(e)),
                                 hint=tr("backend.err.infer_hint")) from e

    def metadata(self) -> dict[str, Any]:
        if self._sess is None:
            return {}
        try:
            custom = dict(self._sess.get_modelmeta().custom_metadata_map or {})
        except Exception:  # noqa: BLE001
            return {}
        meta: dict[str, Any] = {"custom": custom}
        names = parse_names(custom.get("names", ""))
        if names:
            meta["names"] = names
        return meta

    def describe(self) -> str:
        return tr("backend.onnxrt.desc", provider=", ".join(self._providers) or "-", threads=self._threads)

    def close(self) -> None:
        self._sess = None
