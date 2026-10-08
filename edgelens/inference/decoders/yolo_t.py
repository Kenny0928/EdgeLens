"""YOLOv8／v9／v11／v12（ultralytics 匯出）：沒有 objectness 的 4+C 排列。"""

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
    grid_match,
    shape_text,
    sigmoid,
    strip_batch,
    table_to_output,
    want_sigmoid,
)


@register_decoder
class YoloTDecoder(Decoder):
    """輸出 [1, 4+C, N]（或轉置的 [1, N, 4+C]）。前 4 個數是 cx, cy, w, h，後面 C 個是各類別分數，
    沒有 objectness。YOLOv8、v9、v11、v12 用 ultralytics 匯出的 ONNX／TFLite 都是這種格式。

    decoder_params：
      transposed  "auto"（預設，依形狀與類別數判斷）、True（[N, 4+C]）、False（[4+C, N]）。
      normalized  "auto"（座標都在 0–1.5 內就乘上輸入大小）、True、False。
      sigmoid     "auto"（分數超出 0–1 才做 sigmoid）、True、False。
    """

    kind = "yolo_t"
    task = "detect"
    display_name_key = "decoder.yolo_t.name"
    description_key = "decoder.yolo_t.desc"

    @classmethod
    def match(cls, outputs: list[TensorInfo], card: ModelCard,
              samples: list[np.ndarray] | None = None) -> MatchResult:
        try:
            if len(outputs) != 1 or len(outputs[0].shape) != 3:
                return MatchResult(0.0)
            _, a, b = outputs[0].shape
            n = card.num_classes
            text = shape_text(outputs[0].shape)
            # 轉置 [1, N, D]：N 是格點數或候選框數，D = 4 + C
            if a is not None and b is not None and a > b >= 5 or (a is None and b is not None and b >= 5):
                c = b - 4
                if n is not None:
                    return (MatchResult(0.85, {"transposed": True},
                                        tr("decoder.reason.yolo_t_tr", shape=text, d=b, c=c))
                            if c == n else MatchResult(0.0))
                gm = grid_match(a, card)
                if gm is None:
                    score = 0.35
                elif gm[0] == 1:
                    # 沒有類別名稱時，COCO 的 80 類很常見：84 = 4+80 → 比較像 yolo_t，85 → 比較像 yolo_flat
                    score = 0.7 if b == 84 else 0.3 if b == 85 else 0.5
                else:
                    score = 0.1  # N = 3×格點數 → 比較像 YOLOv5／v7
                if b in (6, 7) and a is not None and a <= 300:
                    score = min(score, 0.3)  # 比較像 end2end
                return MatchResult(score, {"transposed": True},
                                   tr("decoder.reason.yolo_t_tr", shape=text, d=b, c=c))
            # 一般 [1, 4+C, N]
            if a is not None and a >= 5 and (b is None or a < b):
                c = a - 4
                if n is not None and c != n:
                    return MatchResult(0.0)
                score = 0.92 if n is not None else (0.9 if grid_match(b, card) else 0.75)
                return MatchResult(score, {"transposed": False},
                                   tr("decoder.reason.yolo_t_cf", shape=text, d=a, c=c))
        except Exception:  # noqa: BLE001 - match 不可丟例外
            pass
        return MatchResult(0.0)

    def decode(self, outputs: list[np.ndarray], ctx: DecodeContext) -> DecodeOutput:
        a = strip_batch(outputs[0], 2)
        if a.ndim != 2:
            raise InferenceError(tr("decoder.err.bad_shape", shape=list(np.shape(outputs[0])), kind=self.kind),
                                 hint=tr("decoder.err.bad_shape_hint"))
        mode = self.params.get("transposed", "auto")
        if isinstance(mode, bool):
            transposed = mode
        else:
            n = ctx.num_classes
            if n is not None and a.shape[0] == 4 + n and a.shape[1] != 4 + n:
                transposed = False
            elif n is not None and a.shape[1] == 4 + n:
                transposed = a.shape[0] != 4 + n or a.shape[0] > a.shape[1]
            else:
                transposed = a.shape[0] > a.shape[1]
        rows = a if transposed else a.T
        c = rows.shape[1] - 4
        if c < 1:
            raise InferenceError(tr("decoder.err.bad_shape", shape=list(a.shape), kind=self.kind),
                                 hint=tr("decoder.err.bad_shape_hint"))
        check_classes(ctx, c)
        cls = rows[:, 4:]
        if want_sigmoid(self.params.get("sigmoid", "auto"), cls):
            cls = sigmoid(cls)
        best = cls.max(axis=1)
        sel = best >= ctx.conf_threshold
        return table_to_output(rows[sel, :4], "cxcywh", best[sel], cls[sel].argmax(axis=1), ctx,
                               self.params.get("normalized", "auto"))
