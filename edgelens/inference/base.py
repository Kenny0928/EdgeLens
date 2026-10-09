"""推論的共同介面：Backend（跑模型）與 Decoder（解讀輸出）。

資料流（由 runner.py 串起來）：

    Frame.image ──preprocess──▶ tensor ──Backend.infer──▶ raw outputs
          │                                                   │ 反量化成 float32
          │                                                   ▼
          │                                       Decoder.decode（輸入座標）
          │                                                   │ 門檻、NMS（runner 做）
          └──────── PreprocessMeta.to_original ◀──────────────┘
                                ▼
                        InferenceResult（原始畫面座標）

Backend 有三種 input_mode：
  tensor  一般模型（onnxrt、cvdnn、tflite）：runner 做前處理與解碼。
  image   直接吃影像、自己產生結果（haar）：runner 呼叫 detect_image()。
  device  在開發板上算好（K210）：runner 讀 Frame.device_result，只套門檻。

Decoder 回傳的框是「模型輸入影像」的像素座標 (x1, y1, x2, y2)，尚未做 NMS（除非
nms_done=True，例如 YOLOv10 或匯出時已含 NMS 的模型）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np

from edgelens.core.modelcard import ModelCard
from edgelens.core.registry import Registry, module_available
from edgelens.core.results import InferenceResult


class InferenceError(Exception):
    """載入或推論失敗；message 直接顯示給使用者，hint 是建議的解決方式。"""

    def __init__(self, message: str, *, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint


@dataclass
class TensorInfo:
    """模型的一個輸入或輸出。shape 中動態維度為 None。quant 為 (scale, zero_point)，
    只有量化模型（int8／uint8）才有；實際值 = (整數值 - zero_point) * scale。"""

    name: str
    shape: tuple[int | None, ...]
    dtype: str = "float32"
    quant: tuple[float, int] | None = None

    def describe(self) -> str:
        dims = "×".join("?" if d is None else str(d) for d in self.shape)
        q = f" q(scale={self.quant[0]:.4g}, zp={self.quant[1]})" if self.quant else ""
        return f"{self.name}: [{dims}] {self.dtype}{q}"


@dataclass
class PreprocessMeta:
    """前處理造成的座標轉換：輸入座標 = 原始座標 * scale + pad。

    letterbox：scale 兩軸相同、pad 為正（補邊）；stretch：scale 兩軸不同、pad 為 0；
    center_crop：pad 為負（裁掉的部分）。
    """

    orig_size: tuple[int, int]
    input_size: tuple[int, int]
    scale: tuple[float, float] = (1.0, 1.0)
    pad: tuple[float, float] = (0.0, 0.0)
    mode: str = "letterbox"

    def to_original(self, boxes: np.ndarray) -> np.ndarray:
        """(N,4) 輸入座標的 xyxy → 原始畫面座標，並裁在畫面範圍內。"""
        b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        b[:, [0, 2]] = (b[:, [0, 2]] - self.pad[0]) / self.scale[0]
        b[:, [1, 3]] = (b[:, [1, 3]] - self.pad[1]) / self.scale[1]
        w, h = self.orig_size
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
        return b

    def to_input(self, boxes: np.ndarray) -> np.ndarray:
        b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        b[:, [0, 2]] = b[:, [0, 2]] * self.scale[0] + self.pad[0]
        b[:, [1, 3]] = b[:, [1, 3]] * self.scale[1] + self.pad[1]
        return b


@dataclass
class DecodeContext:
    """解碼時需要的資訊。num_classes 來自 labels，可能是 None（由輸出形狀推算）。"""

    card: ModelCard
    input_size: tuple[int, int]
    conf_threshold: float
    num_classes: int | None = None
    output_infos: list[TensorInfo] = field(default_factory=list)


@dataclass
class DecodeOutput:
    """解碼結果。

    detect：boxes (N,4) float32 輸入座標 xyxy、scores (N,)、class_ids (N,) int。
    classify：class_scores (C,)，已經是機率（必要時解碼器要做 softmax）。
    nms_done=True 表示模型輸出已經做過 NMS，runner 不再做。
    """

    task: Literal["detect", "classify"]
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    scores: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float32))
    class_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))
    class_scores: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float32))
    nms_done: bool = False


@dataclass
class MatchResult:
    """Decoder.match 的結果。score 0–1，越高越可能是這種格式；params 會合併進
    card.decoder_params；reason 是給學生看的判斷依據（例如「輸出 [1,84,8400] → YOLOv8 格式」）。"""

    score: float
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class Backend(ABC):
    """推論後端。物件只在一條 InferenceWorker 執行緒裡使用。"""

    kind: ClassVar[str] = ""
    display_name_key: ClassVar[str] = ""
    display_name: ClassVar[str] = ""        # 沒有翻譯檔時的顯示名稱（純文字，外掛用）
    description: ClassVar[str] = ""         # 沒有翻譯檔時的說明（純文字，外掛用）
    extensions: ClassVar[tuple[str, ...]] = ()     # 支援的副檔名，例如 (".onnx",)
    requires: ClassVar[tuple[str, ...]] = ()       # 需要的選配模組，例如 ("onnxruntime",)
    input_mode: ClassVar[Literal["tensor", "image", "device"]] = "tensor"
    priority: ClassVar[int] = 0                    # 多個後端都能載入時，數字大的優先

    def __init__(self, card: ModelCard):
        self.card = card

    @classmethod
    def available(cls) -> tuple[bool, str]:
        missing = [m for m in cls.requires if not module_available(m)]
        if missing:
            return False, f"缺少套件：{', '.join(missing)}"
        return True, ""

    @classmethod
    def can_load(cls, card: ModelCard) -> bool:
        return card.extension in cls.extensions

    @abstractmethod
    def load(self) -> None:
        """載入模型。失敗丟 InferenceError（寫清楚原因與建議）。"""

    def inputs(self) -> list[TensorInfo]:
        return []

    def outputs(self) -> list[TensorInfo]:
        return []

    def infer(self, tensor: np.ndarray) -> list[np.ndarray]:
        """tensor 模式：輸入已前處理的張量，回傳原始輸出（依 outputs() 順序，尚未反量化）。"""
        raise NotImplementedError

    def detect_image(self, image: np.ndarray, frame_seq: int = -1) -> InferenceResult:
        """image 模式：直接處理 BGR／灰階影像，回傳原始畫面座標的結果。"""
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        """模型內嵌的資訊（例如 ONNX metadata 裡的類別名稱 names）。"""
        return {}

    def describe(self) -> str:
        """給診斷面板看的說明（例如使用的 execution provider）。"""
        return self.kind

    def close(self) -> None:
        """釋放資源。可重複呼叫。"""


class Decoder(ABC):
    """把模型輸出解讀成偵測框或分類分數。"""

    kind: ClassVar[str] = ""
    task: ClassVar[Literal["detect", "classify"]] = "detect"
    display_name_key: ClassVar[str] = ""
    description_key: ClassVar[str] = ""
    display_name: ClassVar[str] = ""        # 沒有翻譯檔時的顯示名稱（純文字，外掛用）
    description: ClassVar[str] = ""         # 沒有翻譯檔時的說明（純文字，外掛用）

    def __init__(self, card: ModelCard, params: dict[str, Any] | None = None):
        self.card = card
        self.params: dict[str, Any] = {**card.decoder_params, **(params or {})}

    @classmethod
    def match(cls, outputs: list[TensorInfo], card: ModelCard,
              samples: list[np.ndarray] | None = None) -> MatchResult:
        """這種解碼器適不適合這些輸出。samples 是實際跑一次的輸出（可用來看數值範圍）。
        不可丟例外；完全不符合時回傳 score=0。"""
        return MatchResult(0.0)

    @abstractmethod
    def decode(self, outputs: list[np.ndarray], ctx: DecodeContext) -> DecodeOutput:
        """outputs 已經反量化成 float32。只保留分數 ≥ ctx.conf_threshold 的候選框。"""


BACKENDS: Registry[Backend] = Registry("backends", Backend)
register_backend = BACKENDS.register
DECODERS: Registry[Decoder] = Registry("decoders", Decoder)
register_decoder = DECODERS.register


def dequantize(arr: np.ndarray, info: TensorInfo | None) -> np.ndarray:
    """量化輸出 → float32；沒有量化資訊時只轉型。"""
    if info is not None and info.quant is not None and arr.dtype != np.float32:
        scale, zp = info.quant
        return (arr.astype(np.float32) - float(zp)) * float(scale)
    return arr.astype(np.float32, copy=False)
