"""TFLite SSD（例如 SSD MobileNet）：模型內建 NMS，輸出 4 個張量。"""

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
from edgelens.inference.decoders._util import shape_text, table_to_output


def _squeezed(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def assign_roles(arrays: list[np.ndarray], names: list[str] | None = None) -> dict[str, int]:
    """判斷 4 個輸出各是 boxes／classes／scores／count，回傳 {角色: 索引}。

    順序因模型而異（TF 的 PostProcess 通常是 boxes, classes, scores, count，TF Hub 的有些是
    scores, boxes, count, classes），所以先看形狀，再看名稱，最後看數值（類別是整數、分數是小數）。
    """
    roles: dict[str, int] = {}
    rest: list[int] = []
    for i, a in enumerate(arrays):
        if a.ndim >= 3 and a.shape[-1] == 4:
            roles["boxes"] = i
        elif a.size == 1:
            roles["count"] = i
        else:
            rest.append(i)
    if "boxes" not in roles or len(rest) != 2:
        raise InferenceError(tr("decoder.err.ssd_roles"), hint=tr("decoder.err.bad_shape_hint"))
    low = [(names[i].lower() if names and i < len(names) else "") for i in rest]
    by_name: dict[str, int] = {}
    for i, nm in zip(rest, low, strict=True):
        if "score" in nm:
            by_name["scores"] = i
        elif "class" in nm:
            by_name["classes"] = i
    if len(by_name) == 2:
        roles.update(by_name)
    elif "scores" in by_name:
        roles["scores"] = by_name["scores"]
        roles["classes"] = next(i for i in rest if i != by_name["scores"])
    elif "classes" in by_name:
        roles["classes"] = by_name["classes"]
        roles["scores"] = next(i for i in rest if i != by_name["classes"])
    else:
        x, y = rest
        ax, ay = _squeezed(arrays[x]), _squeezed(arrays[y])
        int_x = bool(np.all(np.abs(ax - np.round(ax)) < 1e-4))
        int_y = bool(np.all(np.abs(ay - np.round(ay)) < 1e-4))
        if int_x and not int_y:
            roles["classes"], roles["scores"] = x, y
        elif int_y and not int_x:
            roles["classes"], roles["scores"] = y, x
        else:
            roles["classes"], roles["scores"] = x, y
    return roles


@register_decoder
class TfliteSsdDecoder(Decoder):
    """TFLite 物件偵測範例模型（SSD MobileNet）：4 個輸出
    boxes [1,N,4]（ymin, xmin, ymax, xmax，0–1 正規化）、classes [1,N]、scores [1,N]、count [1]。
    模型已做過 NMS（nms_done=True）。輸出順序依名稱、形狀、數值判斷。

    decoder_params：
      class_offset  類別編號要加多少（預設 0）；有些模型的類別名稱檔第 0 列是 "???"，這時寫 1。
      normalized    "auto"（座標都在 0–1.5 內就乘上輸入大小）、True、False。
    """

    kind = "tflite_ssd"
    task = "detect"
    display_name_key = "decoder.tflite_ssd.name"
    description_key = "decoder.tflite_ssd.desc"

    @classmethod
    def match(cls, outputs: list[TensorInfo], card: ModelCard,
              samples: list[np.ndarray] | None = None) -> MatchResult:
        try:
            if len(outputs) != 4:
                return MatchResult(0.0)
            shapes = [tuple(o.shape) for o in outputs]
            boxes = [s for s in shapes if len(s) == 3 and s[-1] == 4]
            counts = [s for s in shapes if all(d in (1, None) for d in s) and len(s) <= 2]
            if len(boxes) != 1 or len(counts) < 1:
                return MatchResult(0.0)
            return MatchResult(0.9, {}, tr("decoder.reason.tflite_ssd",
                                           shapes="、".join(shape_text(s) for s in shapes)))
        except Exception:  # noqa: BLE001
            return MatchResult(0.0)

    def decode(self, outputs: list[np.ndarray], ctx: DecodeContext) -> DecodeOutput:
        arrays = [np.asarray(o, dtype=np.float32) for o in outputs]
        names = [i.name for i in ctx.output_infos] if len(ctx.output_infos) == len(arrays) else None
        roles = assign_roles(arrays, names)
        boxes = arrays[roles["boxes"]].reshape(-1, 4)
        classes = arrays[roles["classes"]].reshape(-1)
        scores = arrays[roles["scores"]].reshape(-1)
        n = min(len(boxes), len(classes), len(scores))
        count = int(round(float(arrays[roles["count"]].reshape(-1)[0])))
        if 0 <= count < n:
            n = count
        boxes, classes, scores = boxes[:n], classes[:n], scores[:n]
        sel = scores >= ctx.conf_threshold
        ids = np.round(classes[sel]).astype(np.int64) + int(self.params.get("class_offset", 0))
        return table_to_output(boxes[sel], "yxyx", scores[sel], ids, ctx,
                               self.params.get("normalized", "auto"), nms_done=True)
