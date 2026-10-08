"""YOLOv5／v7／v6 與已解碼的 YOLOX：[1, N, 5+C]，有 objectness。"""

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
    check_classes,
    decoded_likelihood,
    grid_match,
    shape_text,
    sigmoid,
    strip_batch,
    table_to_output,
    want_sigmoid,
)


@register_decoder
class YoloFlatDecoder(Decoder):
    """輸出 [1, N, 5+C]：cx, cy, w, h, objectness, 各類別分數；分數 = objectness × 類別分數。
    N = 3 × 格點數：YOLOv5、YOLOv7；N = 格點數：YOLOv6、已解碼的 YOLOX（座標已是影像像素）。

    decoder_params：
      objectness  True（預設，有 objectness 欄）或 False（[N, 4+C]，分數就是類別分數）。
      box_format  "cxcywh"（預設）或 "xyxy"、"xywh"。
      normalized  "auto"（座標都在 0–1.5 內就乘上輸入大小）、True、False。
      sigmoid     "auto"（數值超出 0–1 才做 sigmoid）、True、False。
    """

    kind = "yolo_flat"
    task = "detect"
    display_name_key = "decoder.yolo_flat.name"
    description_key = "decoder.yolo_flat.desc"

    @classmethod
    def match(cls, outputs: list[TensorInfo], card: ModelCard,
              samples: list[np.ndarray] | None = None) -> MatchResult:
        try:
            if len(outputs) != 1 or len(outputs[0].shape) != 3:
                return MatchResult(0.0)
            _, nn, d = outputs[0].shape
            if d is None or d < 6 or (nn is not None and nn < d):
                return MatchResult(0.0)
            c = d - 5
            n = card.num_classes
            text = shape_text(outputs[0].shape)
            if n is not None and n != c:
                return MatchResult(0.0)
            gm = grid_match(nn, card)
            if nn is not None and nn <= 300 and d in (6, 7):
                return MatchResult(0.1, {}, tr("decoder.reason.yolo_flat", shape=text, d=d, c=c))
            if gm is None:
                score = 0.3 if nn is not None else 0.3
                if n is not None:
                    score = 0.55
                return MatchResult(score, {}, tr("decoder.reason.yolo_flat", shape=text, d=d, c=c))
            mult, size, _ = gm
            grid = nn // mult
            if mult == 3:
                score = 0.9 if n is not None else 0.88
                return MatchResult(score, {}, tr("decoder.reason.yolo_flat_v5", shape=text, d=d, c=c,
                                                 n=nn, grid=grid, size=f"{size[0]}×{size[1]}"))
            # N = 格點數：YOLOv6 或 YOLOX（已解碼 or 未解碼）
            p = decoded_likelihood(samples[0] if samples else None)
            p = 0.5 if p is None else p
            score = 0.5 + 0.35 * (2 * p - 1)
            if n is None and d == 84:
                score -= 0.25  # 84 = 4+80，比較像 yolo_t 的轉置排列
            return MatchResult(score, {}, tr("decoder.reason.yolo_flat_v6", shape=text, d=d, c=c, n=nn,
                                             size=f"{size[0]}×{size[1]}"))
        except Exception:  # noqa: BLE001
            return MatchResult(0.0)

    def decode(self, outputs: list[np.ndarray], ctx: DecodeContext) -> DecodeOutput:
        a = strip_batch(outputs[0], 2)
        if a.ndim != 2 or a.shape[1] < 5:
            raise InferenceError(tr("decoder.err.bad_shape", shape=list(np.shape(outputs[0])), kind=self.kind),
                                 hint=tr("decoder.err.bad_shape_hint"))
        has_obj = bool(self.params.get("objectness", True))
        head = 5 if has_obj else 4
        c = a.shape[1] - head
        if c < 1:
            raise InferenceError(tr("decoder.err.bad_shape", shape=list(a.shape), kind=self.kind),
                                 hint=tr("decoder.err.bad_shape_hint"))
        check_classes(ctx, c)
        mode = self.params.get("sigmoid", "auto")
        cls = a[:, head:]
        if want_sigmoid(mode, cls):
            cls = sigmoid(cls)
        if has_obj:
            obj = a[:, 4]
            if want_sigmoid(mode, obj):
                obj = sigmoid(obj)
            keep = obj >= ctx.conf_threshold  # 分數 = obj × cls ≤ obj，先篩掉大部分
            cls_k, obj_k = cls[keep], obj[keep]
            best = cls_k.max(axis=1) * obj_k if len(obj_k) else np.zeros(0, np.float32)
            ids = cls_k.argmax(axis=1)
            rows = a[keep, :4]
        else:
            best_all = cls.max(axis=1)
            keep = best_all >= ctx.conf_threshold
            best, ids, rows = best_all[keep], cls[keep].argmax(axis=1), a[keep, :4]
        sel = best >= ctx.conf_threshold
        return table_to_output(rows[sel], self.params.get("box_format", "cxcywh"), best[sel], ids[sel], ctx,
                               self.params.get("normalized", "auto"))
