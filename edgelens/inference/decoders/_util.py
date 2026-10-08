"""解碼器共用的小工具（底線開頭的模組不會被當成解碼器註冊）。"""

from __future__ import annotations

from typing import Any

import numpy as np

from edgelens.core.modelcard import ModelCard
from edgelens.i18n import tr
from edgelens.inference.base import DecodeContext, DecodeOutput, InferenceError
from edgelens.inference.nms import box_convert

COMMON_SIZES = (640, 416, 320, 512, 608, 1280)
STRIDE_SETS = ((8, 16, 32), (8, 16, 32, 64))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def strip_batch(arr: np.ndarray, ndim: int) -> np.ndarray:
    """輸出多了開頭的 batch 維度（大小 1）就拿掉，讓後面統一處理 ndim 維。"""
    a = np.asarray(arr, dtype=np.float32)
    while a.ndim > ndim and a.shape[0] == 1:
        a = a[0]
    return a


def static_shape(info_shape: tuple[int | None, ...]) -> tuple[int | None, ...]:
    return tuple(info_shape)


def shape_text(shape: tuple[int | None, ...]) -> str:
    return "[" + ", ".join("?" if d is None else str(d) for d in shape) + "]"


def want_sigmoid(mode: Any, values: np.ndarray) -> bool:
    """mode：True／False／"auto"。auto：數值超出 0–1 就代表還是 logit，需要 sigmoid；
    有些匯出工具已經做過 sigmoid，再做一次分數會全部擠到 0.5–0.73。"""
    if isinstance(mode, bool):
        return mode
    if values.size == 0:
        return False
    return bool(values.min() < 0.0 or values.max() > 1.0)


def check_classes(ctx: DecodeContext, c: int) -> None:
    """說明檔的類別名稱數量和模型輸出的類別數不同時，用學生看得懂的話報錯。"""
    if ctx.num_classes is not None and ctx.num_classes != c:
        raise InferenceError(
            tr("decoder.err.class_mismatch", model=c, labels=ctx.num_classes),
            hint=tr("decoder.err.class_mismatch_hint"))


def is_normalized(coords: np.ndarray, mode: Any = "auto") -> bool:
    """座標是 0–1 正規化嗎？auto：全部數值都在 0–1.5 之間視為正規化。"""
    if isinstance(mode, bool):
        return mode
    if coords.size == 0:
        return False
    return bool(coords.min() >= -0.5 and coords.max() <= 1.5)


def scale_if_normalized(xyxy: np.ndarray, ctx: DecodeContext, mode: Any = "auto") -> np.ndarray:
    if is_normalized(xyxy, mode):
        w, h = ctx.input_size
        xyxy = xyxy * np.array([w, h, w, h], dtype=np.float32)
    return xyxy


def make_output(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, conf: float,
                nms_done: bool = False) -> DecodeOutput:
    """套用分數門檻並整理成 DecodeOutput。"""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    class_ids = np.asarray(class_ids).reshape(-1).astype(np.int64)
    keep = (scores >= conf) & np.isfinite(scores) & np.isfinite(boxes).all(axis=1)
    return DecodeOutput(task="detect", boxes=boxes[keep], scores=scores[keep], class_ids=class_ids[keep],
                        nms_done=nms_done)


def table_to_output(rows_box: np.ndarray, box_format: str, scores: np.ndarray, class_ids: np.ndarray,
                    ctx: DecodeContext, normalized: Any = "auto", nms_done: bool = False) -> DecodeOutput:
    """已通過門檻的候選框（任一框格式）→ 輸入像素座標 xyxy 的 DecodeOutput。"""
    if len(scores) == 0:
        return DecodeOutput(task="detect", nms_done=nms_done)
    xyxy = box_convert(rows_box, box_format)
    # 正規化判斷用中心／寬高或角點的原始數值，兩者都在 0–1.5 內就一致
    xyxy = scale_if_normalized(xyxy, ctx, normalized) if is_normalized(rows_box, normalized) else xyxy
    return make_output(xyxy, scores, class_ids, ctx.conf_threshold, nms_done)


# ---- 格點數（用來判斷輸出是不是 YOLO 的 N）-----------------------------------------------

def grid_total(size: tuple[int, int], strides: tuple[int, ...]) -> int:
    w, h = size
    return sum((w // s) * (h // s) for s in strides)


def candidate_sizes(card: ModelCard, input_size: tuple[int, int] | None = None) -> list[tuple[int, int]]:
    if input_size:
        return [tuple(input_size)]  # type: ignore[list-item]
    if card.input.size:
        return [tuple(card.input.size)]  # type: ignore[list-item]
    return [(s, s) for s in COMMON_SIZES]


def grid_match(n: int | None, card: ModelCard, input_size: tuple[int, int] | None = None
               ) -> tuple[int, tuple[int, int], tuple[int, ...]] | None:
    """N 等於 1×格點數或 3×格點數嗎？回傳 (倍數, 輸入大小, strides)，都不是就回傳 None。"""
    if not n:
        return None
    for size in candidate_sizes(card, input_size):
        for strides in STRIDE_SETS:
            total = grid_total(size, strides)
            for mult in (1, 3):
                if total * mult == n:
                    return mult, size, strides
    return None


def decoded_likelihood(sample: np.ndarray | None) -> float | None:
    """[1,N,D] 輸出的前 4 欄是「已解碼的框座標」(接近 1) 還是「未解碼的格點偏移／log」(接近 0)？

    已解碼：cx, cy, w, h 都不會是負的，像素座標常大於 20（或 0–1 正規化）。
    未解碼：xy 是格點偏移、wh 是 log，常出現負值，且數值通常小於 10。
    判斷不出來回傳 None。
    """
    if sample is None:
        return None
    a = np.asarray(sample, dtype=np.float32)
    a = a.reshape(-1, a.shape[-1]) if a.ndim >= 2 else a.reshape(1, -1)
    if a.shape[1] < 5 or a.shape[0] == 0:
        return None
    box = a[:, :4]
    if not np.isfinite(box).all():
        return None
    lo, hi = float(box.min()), float(box.max())
    if lo < -0.05:
        return 0.05
    if hi > 20.0:
        return 0.95
    if hi <= 1.5:
        return 0.8
    return None
