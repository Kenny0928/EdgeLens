"""YOLOX 官方 ONNX（匯出時沒有 decode_in_inference）：座標還沒解碼。"""

from __future__ import annotations

import numpy as np

from edgelens.core.modelcard import ModelCard
from edgelens.i18n import tr
from edgelens.inference.base import (
    DecodeContext,
    DecodeOutput,
    Decoder,
    InferenceError,
    MatchResult,
    TensorInfo,
    register_decoder,
)
from edgelens.inference.decoders._util import (
    STRIDE_SETS,
    check_classes,
    decoded_likelihood,
    grid_match,
    grid_total,
    shape_text,
    sigmoid,
    strip_batch,
    table_to_output,
    want_sigmoid,
)


def yolox_grids(size: tuple[int, int], strides: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """每個候選框對應的格點座標 (N,2)（x, y）與 stride (N,1)；順序和 YOLOX 的 decode_outputs 相同。"""
    w, h = size
    grids, strs = [], []
    for s in strides:
        xv, yv = np.meshgrid(np.arange(w // s), np.arange(h // s))
        g = np.stack((xv, yv), axis=2).reshape(-1, 2)
        grids.append(g)
        strs.append(np.full((len(g), 1), s))
    return np.concatenate(grids).astype(np.float32), np.concatenate(strs).astype(np.float32)


@register_decoder
class YoloxRawDecoder(Decoder):
    """YOLOX 官方 ONNX 的原始輸出 [1, N, 5+C]，N = 格點數（strides 8/16/32 的格點總和）。
    xy 是「格點偏移」、wh 是「log」，要自己解碼：中心 = (偏移 + 格點) × stride、寬高 = exp(log) × stride。
    objectness 與類別分數已經過 sigmoid。

    decoder_params：
      strides   [8, 16, 32]（預設）；P6 模型可寫 [8, 16, 32, 64]。N 對不上時會自動試 P6。
      sigmoid   "auto"（分數超出 0–1 才做 sigmoid）、True、False。
    """

    kind = "yolox_raw"
    task = "detect"
    display_name_key = "decoder.yolox_raw.name"
    description_key = "decoder.yolox_raw.desc"

    @classmethod
    def match(cls, outputs: list[TensorInfo], card: ModelCard,
              samples: list[np.ndarray] | None = None) -> MatchResult:
        try:
            if len(outputs) != 1 or len(outputs[0].shape) != 3:
                return MatchResult(0.0)
            _, nn, d = outputs[0].shape
            if d is None or d < 6 or nn is None:
                return MatchResult(0.0)
            n = card.num_classes
            if n is not None and n != d - 5:
                return MatchResult(0.0)
            gm = grid_match(nn, card)
            if gm is None or gm[0] != 1:
                return MatchResult(0.0)
            p = decoded_likelihood(samples[0] if samples else None)
            p = 0.5 if p is None else p
            score = 0.5 + 0.35 * (1 - 2 * p)
            if n is None and d == 84:
                score -= 0.25
            return MatchResult(score, {"strides": list(gm[2])},
                               tr("decoder.reason.yolox_raw", shape=shape_text(outputs[0].shape), d=d, c=d - 5,
                                  n=nn, size=f"{gm[1][0]}×{gm[1][1]}"))
        except Exception:  # noqa: BLE001
            return MatchResult(0.0)

    def decode(self, outputs: list[np.ndarray], ctx: DecodeContext) -> DecodeOutput:
        a = strip_batch(outputs[0], 2)
        if a.ndim != 2 or a.shape[1] < 6:
            raise InferenceError(tr("decoder.err.bad_shape", shape=list(np.shape(outputs[0])), kind=self.kind),
                                 hint=tr("decoder.err.bad_shape_hint"))
        c = a.shape[1] - 5
        check_classes(ctx, c)
        sets = [tuple(int(s) for s in self.params["strides"])] if self.params.get("strides") else []
        sets += [s for s in STRIDE_SETS if s not in sets]
        strides = next((s for s in sets if grid_total(ctx.input_size, s) == a.shape[0]), None)
        if strides is None:
            raise InferenceError(
                tr("decoder.err.grid_mismatch", n=a.shape[0], size=f"{ctx.input_size[0]}×{ctx.input_size[1]}"),
                hint=tr("decoder.err.grid_mismatch_hint"))
        mode = self.params.get("sigmoid", "auto")
        obj, cls = a[:, 4], a[:, 5:]
        if want_sigmoid(mode, obj):
            obj = sigmoid(obj)
        if want_sigmoid(mode, cls):
            cls = sigmoid(cls)
        keep = obj >= ctx.conf_threshold
        grid, strd = yolox_grids(ctx.input_size, strides)
        ak, gk, sk = a[keep], grid[keep], strd[keep]
        cls_k = cls[keep]
        score = obj[keep] * cls_k.max(axis=1) if keep.any() else np.zeros(0, np.float32)
        ids = cls_k.argmax(axis=1)
        sel = score >= ctx.conf_threshold
        xy = (ak[sel, :2] + gk[sel]) * sk[sel]
        wh = np.exp(np.clip(ak[sel, 2:4], -20.0, 20.0)) * sk[sel]
        boxes = np.concatenate([xy, wh], axis=1)
        return table_to_output(boxes, "cxcywh", score[sel], ids[sel], ctx, normalized=False)
