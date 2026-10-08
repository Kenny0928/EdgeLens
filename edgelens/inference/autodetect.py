"""輸出格式自動判斷：看模型的輸出形狀（與一次實際輸出的數值）猜是哪一種解碼器。

做法：對每個註冊的解碼器呼叫 `Decoder.match`，把分數由高到低排序，GUI 顯示最高的並說明
判斷依據，判斷錯了學生可以改選別的。說明檔的 `decoder` 不是 "auto" 時直接用指定的。

規則對照表（N、K 是候選框數；格點數 = Σ (輸入寬/stride × 輸入高/stride)，strides 為 8/16/32，
P6 模型為 8/16/32/64；輸入大小從說明檔或呼叫者提供，推不出來就試 320、416、512、608、640、1280）：

| 輸出形狀                                 | 解碼器       | 適用                                  |
|------------------------------------------|--------------|---------------------------------------|
| [1, C]                                   | classify     | 分類模型                              |
| [1, K, 6 或 7]，K ≤ 300                  | end2end      | YOLOv10、匯出時含 NMS 的 v5/v7/v8      |
| [1, 4+C, N] 或 [1, N, 4+C]               | yolo_t       | YOLOv8、v9、v11、v12                  |
| [1, N, 5+C]，N = 3 × 格點數              | yolo_flat    | YOLOv5、YOLOv7                        |
| [1, N, 5+C]，N = 格點數                  | yolo_flat 或 yolox_raw | YOLOv6／已解碼 YOLOX，或未解碼 YOLOX；用 samples 的數值區分（已解碼的座標是像素、不為負；未解碼的 xy 偏移、wh 的 log 常為負且很小）。無法判斷時兩個分數相同，讓使用者選 |
| 單一輸出 [1, A×(5+C), H, W]（或其他排列）| yolo_grid    | YOLOv2、tiny-YOLOv2、K210 kmodel      |
| 2–3 個格點輸出                           | yolo_grid    | YOLOv3、YOLOv4（含 tiny）             |
| 4 個輸出（框 [1,N,4]、類別、分數、數量）  | tflite_ssd   | TFLite SSD MobileNet                  |

有說明檔（labels）時，類別數 C 會用來交叉確認（例如 84 = 4 + 80）。
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from edgelens.core.modelcard import ModelCard
from edgelens.core.registry import discover
from edgelens.i18n import tr
from edgelens.inference.base import DECODERS, TensorInfo

log = logging.getLogger(__name__)


@dataclass
class FormatGuess:
    """一個可能的輸出格式。params 要合併進 card.decoder_params；reason 是給學生看的依據。"""

    decoder_kind: str
    score: float
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


def detect_format(outputs: list[TensorInfo], card: ModelCard, samples: list[np.ndarray] | None = None,
                  input_size: tuple[int, int] | None = None) -> list[FormatGuess]:
    """回傳所有分數 > 0 的猜測，分數高的在前面。

    samples：實際跑一次（已反量化）的輸出，用來區分 YOLOv6 與未解碼 YOLOX 等。
    input_size：實際的輸入大小 (寬, 高)；說明檔沒設定 input.size 時，用它來算格點數。
    card.decoder 不是 "auto" 且是已註冊的解碼器時，只回傳該解碼器（score=1）。
    """
    discover()
    if card.decoder and card.decoder != "auto":
        if card.decoder in DECODERS:
            return [FormatGuess(card.decoder, 1.0, {}, tr("autodetect.by_card", kind=card.decoder))]
        log.warning("說明檔指定的解碼器 %r 不存在，改用自動判斷", card.decoder)

    if input_size and not card.input.size:
        card = dataclasses.replace(card, input=dataclasses.replace(card.input, size=tuple(input_size)))

    guesses: list[FormatGuess] = []
    for kind, cls in DECODERS.items():
        try:
            m = cls.match(outputs, card, samples)
        except Exception:  # noqa: BLE001 - 單一解碼器出錯不能讓整個判斷失敗
            log.exception("解碼器 %s 的 match 出錯", kind)
            continue
        if m.score > 0:
            guesses.append(FormatGuess(kind, float(m.score), dict(m.params), m.reason))
    guesses.sort(key=lambda g: (-g.score, g.decoder_kind))
    return guesses


def raw_summary(outputs: list[TensorInfo], arrays: list[np.ndarray]) -> list[str]:
    """給「原始輸出檢視器」：每個輸出一行（名稱、形狀、型別、最小／最大／平均）。"""
    lines: list[str] = []
    for i, arr in enumerate(arrays):
        a = np.asarray(arr)
        name = outputs[i].name if i < len(outputs) else f"output{i}"
        if a.size:
            f = a.astype(np.float32, copy=False)
            lo, hi, mean = f"{float(f.min()):.4g}", f"{float(f.max()):.4g}", f"{float(f.mean()):.4g}"
        else:
            lo = hi = mean = "-"
        lines.append(tr("autodetect.summary", name=name, shape=list(a.shape), dtype=str(a.dtype),
                        min=lo, max=hi, mean=mean))
    return lines
