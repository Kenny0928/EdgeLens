"""Frame：相機來源交給後續處理的一張畫面。"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from edgelens.core.results import InferenceResult


@dataclass
class Frame:
    """一張畫面。

    image 一律是 uint8：彩色為 BGR 的 HxWx3，灰階為 HxW（OpenCV 慣例）。
    t_cap 用 time.monotonic()，計算延遲用；t_wall 用 time.time()，存檔與顯示用。
    jpeg 是來源原本就壓好的位元組（ESP32-CAM、K210），存檔時直接寫入不重新壓縮；
    來源沒有就是 None。
    meta 放各來源的附加資訊（例如 K210 的板上耗時），內容必須能轉成 JSON。
    device_result 是板上推論的結果（例如 K210 跑 YOLO），電腦端推論不放這裡。
    """

    image: np.ndarray
    seq: int
    t_cap: float
    source_id: str
    jpeg: bytes | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    device_result: InferenceResult | None = None
    t_wall: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not isinstance(self.image, np.ndarray):
            raise TypeError("Frame.image 必須是 numpy.ndarray")
        if self.image.dtype != np.uint8:
            raise TypeError(f"Frame.image 必須是 uint8，收到 {self.image.dtype}")
        if self.image.ndim == 3 and self.image.shape[2] == 1:
            self.image = self.image[:, :, 0]
        if not (self.image.ndim == 2 or (self.image.ndim == 3 and self.image.shape[2] == 3)):
            raise ValueError(f"Frame.image 形狀必須是 HxW 或 HxWx3，收到 {self.image.shape}")

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def size(self) -> tuple[int, int]:
        """(寬, 高)。"""
        return self.width, self.height

    @property
    def is_gray(self) -> bool:
        return self.image.ndim == 2

    def age_ms(self, now: float | None = None) -> float:
        """從取像到現在經過幾毫秒。"""
        return ((time.monotonic() if now is None else now) - self.t_cap) * 1000.0

    def replace(self, **changes: Any) -> Frame:
        """回傳換掉部分欄位的新 Frame（例如處理後的 image）。換了 image 時 jpeg 會清掉，
        因為原始 JPEG 已經不代表新的畫面；需要保留時請明確傳入 jpeg=。"""
        if "image" in changes and "jpeg" not in changes:
            changes["jpeg"] = None
        return dataclasses.replace(self, **changes)
