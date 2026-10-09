"""分類模型：輸出 [1, C] 或 [C]，代表每個類別的分數。也是寫新解碼器時的參考範例。"""

from __future__ import annotations

import numpy as np

from edgelens.core.modelcard import ModelCard
from edgelens.inference.base import (
    DecodeContext,
    DecodeOutput,
    Decoder,
    MatchResult,
    TensorInfo,
    register_decoder,
)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


@register_decoder
class ClassifyDecoder(Decoder):
    """params：
    activation  "auto"（輸出總和不是 1 或有負值時做 softmax）、"softmax"、"sigmoid"、"none"。
    """

    kind = "classify"
    task = "classify"
    display_name_key = "decoder.classify.name"
    description_key = "decoder.classify.desc"

    @classmethod
    def match(cls, outputs: list[TensorInfo], card: ModelCard,
              samples: list[np.ndarray] | None = None) -> MatchResult:
        if len(outputs) != 1:
            return MatchResult(0.0)
        dims = [d for d in outputs[0].shape if d not in (None, 1)]
        if len(dims) != 1:
            return MatchResult(0.0)
        c = dims[0]
        n = card.num_classes
        if n is not None and n == c:
            return MatchResult(0.95, reason=f"輸出形狀 {list(outputs[0].shape)}，與 {n} 個類別名稱相符 → 分類模型")
        return MatchResult(0.8, reason=f"輸出形狀 {list(outputs[0].shape)} 只有一個維度 → 分類模型（{c} 類）")

    def decode(self, outputs: list[np.ndarray], ctx: DecodeContext) -> DecodeOutput:
        scores = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        act = self.params.get("activation", "auto")
        if act == "auto":
            # 已經是機率（0–1 且總和接近 1）就不再做 softmax。容許範圍放寬到 0.7–1.3：
            # 量化（uint8）模型的機率有捨入誤差，總和常常只有 0.9 多，再做一次 softmax 分數會全擠在一起。
            total = float(scores.sum())
            is_prob = scores.min() >= 0 and scores.max() <= 1 and 0.7 < total < 1.3
            act = "none" if is_prob else "softmax"
        if act == "softmax":
            scores = softmax(scores)
        elif act == "sigmoid":
            scores = 1.0 / (1.0 + np.exp(-scores))
        return DecodeOutput(task="classify", class_scores=scores.astype(np.float32))
