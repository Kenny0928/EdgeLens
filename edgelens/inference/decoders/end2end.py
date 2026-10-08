"""已經含 NMS 的輸出：YOLOv10、以及匯出時內建 NMS 的 YOLOv5／v7／v8。"""

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
from edgelens.inference.decoders._util import check_classes, shape_text, strip_batch, table_to_output

DEFAULT_COLUMNS = {
    6: ("x1", "y1", "x2", "y2", "score", "class"),
    7: ("batch", "x1", "y1", "x2", "y2", "class", "score"),
}


def _is_int_like(v: np.ndarray) -> bool:
    return bool(v.size) and bool(np.all(np.abs(v - np.round(v)) < 1e-3))


@register_decoder
class End2EndDecoder(Decoder):
    """輸出 [1, K, 6]：x1, y1, x2, y2, 分數, 類別（YOLOv10、ultralytics 內建 NMS）；
    或 [1, K, 7]：batch_id, x1, y1, x2, y2, 類別, 分數（YOLOv7 end2end 匯出）。K 通常 ≤ 300，
    不足的列分數為 0。模型已做過 NMS，所以 nms_done=True。

    decoder_params：
      columns     欄位名稱清單，可用 x1 y1 x2 y2 score class batch skip。預設 6 欄為
                  [x1,y1,x2,y2,score,class]、7 欄為 [batch,x1,y1,x2,y2,class,score]；
                  6 欄時若看起來是 [..., class, score] 會自動對調（"auto"）。
      normalized  "auto"（座標都在 0–1.5 內就乘上輸入大小）、True、False。
    """

    kind = "end2end"
    task = "detect"
    display_name_key = "decoder.end2end.name"
    description_key = "decoder.end2end.desc"

    @classmethod
    def match(cls, outputs: list[TensorInfo], card: ModelCard,
              samples: list[np.ndarray] | None = None) -> MatchResult:
        try:
            if len(outputs) != 1 or len(outputs[0].shape) != 3:
                return MatchResult(0.0)
            _, k, d = outputs[0].shape
            if d not in (6, 7) or (k is not None and k > 300):
                return MatchResult(0.0)
            score = 0.85
            if samples:
                s = np.asarray(samples[0], dtype=np.float32).reshape(-1, d)
                live = s[(np.abs(s[:, 1:5]) > 0).any(axis=1)]
                # 類別欄是整數，是 end2end 的強烈跡象
                if len(live) and (_is_int_like(live[:, 5]) or _is_int_like(live[:, 4])):
                    score = 0.95
            return MatchResult(score, {}, tr("decoder.reason.end2end", shape=shape_text(outputs[0].shape), d=d))
        except Exception:  # noqa: BLE001
            return MatchResult(0.0)

    def _columns(self, rows: np.ndarray) -> tuple[str, ...]:
        user = self.params.get("columns")
        if user and user != "auto":
            if len(user) != rows.shape[1]:
                raise InferenceError(tr("decoder.err.columns", got=len(user), need=rows.shape[1]),
                                     hint=tr("decoder.err.bad_shape_hint"))
            return tuple(user)
        d = rows.shape[1]
        cols = DEFAULT_COLUMNS.get(d)
        if cols is None:
            raise InferenceError(tr("decoder.err.bad_shape", shape=list(rows.shape), kind=self.kind),
                                 hint=tr("decoder.err.bad_shape_hint"))
        if d == 6:
            live = rows[(np.abs(rows) > 0).any(axis=1)]
            # [.., class, score]：第 5 欄是整數、第 6 欄不是
            if len(live) and _is_int_like(live[:, 4]) and not _is_int_like(live[:, 5]) \
                    and float(live[:, 5].max()) <= 1.0:
                return ("x1", "y1", "x2", "y2", "class", "score")
        return cols

    def decode(self, outputs: list[np.ndarray], ctx: DecodeContext) -> DecodeOutput:
        a = strip_batch(outputs[0], 2)
        if a.ndim != 2:
            raise InferenceError(tr("decoder.err.bad_shape", shape=list(np.shape(outputs[0])), kind=self.kind),
                                 hint=tr("decoder.err.bad_shape_hint"))
        cols = self._columns(a)
        idx = {name: i for i, name in enumerate(cols) if name != "skip"}
        scores = a[:, idx["score"]]
        sel = scores >= ctx.conf_threshold
        rows = a[sel]
        boxes = rows[:, [idx["x1"], idx["y1"], idx["x2"], idx["y2"]]]
        ids = np.round(rows[:, idx["class"]]).astype(np.int64)
        if ctx.num_classes is not None and len(ids) and int(ids.max()) >= ctx.num_classes:
            # 類別編號超出名稱數量：多半是類別名稱檔和模型不是同一份
            check_classes(ctx, int(ids.max()) + 1)
        return table_to_output(boxes, "xyxy", scores[sel], ids, ctx, self.params.get("normalized", "auto"),
                               nms_done=True)
