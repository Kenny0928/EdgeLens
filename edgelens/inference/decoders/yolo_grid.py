"""YOLOv2／v3／v4（含 tiny、K210 kmodel）：格點輸出 + anchors，要自己做 sigmoid／exp 解碼。"""

from __future__ import annotations

from typing import Any

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
    make_output,
    shape_text,
    sigmoid,
    softmax,
    strip_batch,
    want_sigmoid,
)

_V3_ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
_TINY_ANCHORS = [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]
_V4_ANCHORS = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]

# masks 與 scale_x_y 都依「stride 由小到大」排列（不管模型輸出的順序，程式會用格點大小判斷 stride）
PRESETS: dict[str, dict[str, Any]] = {
    "tiny_yolov2_voc": {
        "anchors": [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52],
        "anchors_unit": "grid"},
    "yolov2_voc": {
        "anchors": [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071],
        "anchors_unit": "grid"},
    "k210_facedetect": {
        "anchors": [1.889, 2.5245, 2.9465, 3.94056, 3.99987, 5.3658, 5.155437, 6.92275, 6.718375, 9.01025],
        "anchors_unit": "grid"},
    "yolov3_coco": {
        "anchors": _V3_ANCHORS, "anchors_unit": "pixel", "masks": [[0, 1, 2], [3, 4, 5], [6, 7, 8]]},
    "yolov3_tiny_coco": {
        "anchors": _TINY_ANCHORS, "anchors_unit": "pixel", "masks": [[0, 1, 2], [3, 4, 5]]},
    "yolov4_coco": {
        "anchors": _V4_ANCHORS, "anchors_unit": "pixel", "masks": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        "scale_x_y": [1.2, 1.1, 1.05]},
    "yolov4_tiny_coco": {
        "anchors": _TINY_ANCHORS, "anchors_unit": "pixel", "masks": [[0, 1, 2], [3, 4, 5]],
        "scale_x_y": [1.05, 1.05]},
}

LAYOUTS_3D = ("NCHW", "NHWC")        # [X,H,W]、[H,W,X]（strip 掉 batch 後）；X = A×(5+C)
LAYOUTS_4D = ("NAHWC", "NHWAC")      # [A,H,W,D]、[H,W,A,D]；D = 5+C


def default_preset(n_outputs: int, num_classes: int | None = None) -> str:
    """沒有 anchors 資訊時，依輸出個數猜一組預設 anchors。"""
    if n_outputs <= 1:
        return "k210_facedetect" if num_classes == 1 else "tiny_yolov2_voc"
    return "yolov3_tiny_coco" if n_outputs == 2 else "yolov3_coco"


def _layout_3d(shape: tuple[int, int, int], a: int | None, c: int | None, forced: str) -> str:
    if forced in LAYOUTS_3D:
        return forced
    if a and c:
        x = a * (5 + c)
        if shape[0] == x:
            return "NCHW"
        if shape[2] == x:
            return "NHWC"
    if a:  # 通道數必須能被 anchors 數整除，兩種解讀只有一種成立時就用它
        ok0, ok2 = shape[0] % a == 0 and shape[0] // a >= 6, shape[2] % a == 0 and shape[2] // a >= 6
        if ok0 != ok2:
            return "NCHW" if ok0 else "NHWC"
    return "NHWC" if shape[2] > shape[0] else "NCHW"


def _layout_4d(shape: tuple[int, int, int, int], a: int | None, forced: str) -> str:
    if forced in LAYOUTS_4D:
        return forced
    if a:
        if shape[0] == a:
            return "NAHWC"
        if shape[2] == a:
            return "NHWAC"
    return "NHWAC" if shape[2] <= 10 < shape[0] else "NAHWC"


def parse_grid(shape: tuple[int, ...], a_hint: int | None, c_hint: int | None, forced: str = "auto"
               ) -> tuple[str, int, int, int, int] | None:
    """已去掉 batch 的形狀 → (layout, A, H, W, D)，D = 5+C；不是格點輸出回傳 None。
    A、C 提示用來在 NCHW／NHWC 之間做判斷與推算。"""
    forced = forced.upper() if isinstance(forced, str) else "AUTO"
    if any(d is None for d in shape):
        return None
    if len(shape) == 3:
        layout = _layout_3d(shape, a_hint, c_hint, forced)  # type: ignore[arg-type]
        x, h, w = shape if layout == "NCHW" else (shape[2], shape[0], shape[1])
        cands: list[int] = []
        if a_hint:
            cands.append(a_hint)
        cands += [a for a in (3, 5) if a not in cands]
        for a in cands:
            if x % a == 0 and x // a >= 6 and (c_hint is None or x // a == 5 + c_hint):
                return layout, a, h, w, x // a
        return None
    if len(shape) == 4:
        layout = _layout_4d(shape, a_hint, forced)  # type: ignore[arg-type]
        a, h, w, d = shape if layout == "NAHWC" else (shape[2], shape[0], shape[1], shape[3])
        if d < 6 or a > 12:
            return None
        return layout, a, h, w, d
    return None


def _to_ahwd(arr: np.ndarray, layout: str, a: int, h: int, w: int, d: int) -> np.ndarray:
    if layout == "NCHW":
        return arr.reshape(a, d, h, w).transpose(0, 2, 3, 1)
    if layout == "NHWC":
        return arr.reshape(h, w, a, d).transpose(2, 0, 1, 3)
    if layout == "NHWAC":
        return arr.transpose(2, 0, 1, 3)
    return arr


def _anchor_pairs(anchors: Any) -> np.ndarray:
    arr = np.asarray(anchors, dtype=np.float32).reshape(-1)
    if arr.size < 2 or arr.size % 2:
        raise InferenceError(tr("decoder.err.anchors_odd"), hint=tr("decoder.err.anchors_hint"))
    return arr.reshape(-1, 2)


@register_decoder
class YoloGridDecoder(Decoder):
    """格點輸出 + anchors（YOLOv2、v3、v4、tiny 版、K210 kmodel）。每個輸出的形狀可以是
    [1, A×(5+C), H, W]、[1, A, H, W, 5+C]、[1, H, W, A, 5+C]、[1, H, W, A×(5+C)]。
    解碼：中心 = (sigmoid(tx) + 格點) × stride、寬高 = exp(tw) × anchor；分數 = objectness × 類別分數。

    decoder_params：
      anchors_preset  內建 anchors 組合：tiny_yolov2_voc、yolov2_voc、k210_facedetect、yolov3_coco、
                      yolov3_tiny_coco（v4-tiny 也適用 anchors）、yolov4_coco、yolov4_tiny_coco。
                      預設值會被明確寫出的參數蓋過。沒指定時依輸出個數猜。
      anchors         自訂 anchors，平的 [w1,h1,w2,h2,...] 或 [[w,h],...]。
      anchors_unit    "grid"（以格點為單位，YOLOv2／K210）或 "pixel"（以輸入像素為單位，YOLOv3／v4）。
      masks           多個輸出時，每個輸出用哪幾個 anchors：[[0,1,2],[3,4,5],...]，依 stride 由小到大排列。
                      沒寫就把 anchors 平均切成幾份（小的給小 stride）。
      strides         每個輸出的 stride（依模型輸出順序）；沒寫就用 輸入大小 ÷ 格點數。
      scale_x_y       v4 的中心偏移縮放（數字或依 stride 由小到大的清單），預設 1.0。
      layout          "auto" 或 "NCHW"、"NHWC"、"NAHWC"、"NHWAC"。
      sigmoid         "auto"（物件／類別分數超出 0–1 才做 sigmoid／softmax）、True、False（匯出時已做過）。
      class_activation  "auto"（grid anchors 用 softmax、pixel anchors 用 sigmoid）、"softmax"、"sigmoid"、"none"。
    """

    kind = "yolo_grid"
    task = "detect"
    display_name_key = "decoder.yolo_grid.name"
    description_key = "decoder.yolo_grid.desc"

    # ---- 判斷 ------------------------------------------------------------------

    @classmethod
    def match(cls, outputs: list[TensorInfo], card: ModelCard,
              samples: list[np.ndarray] | None = None) -> MatchResult:
        try:
            n_out = len(outputs)
            if not 1 <= n_out <= 3:
                return MatchResult(0.0)
            shapes = []
            for o in outputs:
                sh = tuple(o.shape)
                if len(sh) not in (4, 5):
                    return MatchResult(0.0)
                shapes.append(sh[1:] if sh[0] in (1, None) else sh)
            n = card.num_classes
            p = card.decoder_params
            given = _anchor_pairs(p["anchors"]) if p.get("anchors") else None
            preset = PRESETS.get(str(p.get("anchors_preset", "")))
            if given is None and preset:
                given = _anchor_pairs(preset["anchors"])
            if given is not None:
                masks = p.get("masks") or (preset or {}).get("masks")
                a_hint = len(masks[0]) if masks else len(given) // n_out
            else:
                a_hint = 5 if n_out == 1 else 3
            parsed = [parse_grid(sh, a_hint, n, str(p.get("layout", "auto"))) for sh in shapes]
            if any(x is None for x in parsed):
                return MatchResult(0.0)
            ds = {x[4] for x in parsed}  # type: ignore[index]
            if len(ds) != 1:
                return MatchResult(0.0)
            d = ds.pop()
            c = d - 5
            sizes = sorted((x[2], x[3]) for x in parsed)  # type: ignore[index]
            regular = all(sizes[i + 1][0] == 2 * sizes[i][0] for i in range(len(sizes) - 1))
            score = 0.9 if n is not None else 0.75
            if not regular:
                score -= 0.3
            a0 = parsed[0][1]  # type: ignore[index]
            if n_out == 1 and a0 != 5 and given is None:
                score -= 0.15
            params: dict[str, Any] = {}
            if given is None:
                params["anchors_preset"] = default_preset(n_out, n if n is not None else c)
            hw = f"{sizes[0][0]}×{sizes[0][1]}"
            reason = tr("decoder.reason.yolo_grid", shapes="、".join(shape_text(o.shape) for o in outputs),
                        a=a0, c=c, n_out=n_out, hw=hw)
            if "anchors_preset" in params:
                reason += " " + tr("decoder.reason.yolo_grid_guess", preset=params["anchors_preset"])
            return MatchResult(max(score, 0.05), params, reason)
        except Exception:  # noqa: BLE001
            return MatchResult(0.0)

    # ---- 設定 ------------------------------------------------------------------

    def _config(self, n_outputs: int, num_classes: int | None) -> dict[str, Any]:
        p = self.params
        name = p.get("anchors_preset")
        preset = dict(PRESETS.get(str(name), {})) if name else {}
        if name and not preset:
            raise InferenceError(tr("decoder.err.preset", name=name, names=", ".join(PRESETS)),
                                 hint=tr("decoder.err.anchors_hint"))
        if not p.get("anchors") and not preset:
            preset = dict(PRESETS[default_preset(n_outputs, num_classes)])
        cfg = {**preset, **{k: v for k, v in p.items() if k in ("anchors", "anchors_unit", "masks", "scale_x_y")
                            and v is not None}}
        cfg["anchors"] = _anchor_pairs(cfg["anchors"])
        cfg.setdefault("anchors_unit", "pixel")
        return cfg

    # ---- 解碼 ------------------------------------------------------------------

    def decode(self, outputs: list[np.ndarray], ctx: DecodeContext) -> DecodeOutput:
        n_out = len(outputs)
        cfg = self._config(n_out, ctx.num_classes)
        anchors: np.ndarray = cfg["anchors"]
        masks = cfg.get("masks")
        a_hint = len(masks[0]) if masks else len(anchors) // n_out
        in_w, in_h = ctx.input_size
        layout_param = str(self.params.get("layout", "auto"))

        grids = []
        for i, o in enumerate(outputs):
            a = np.asarray(o, dtype=np.float32)
            a = strip_batch(a, 3) if a.ndim == 4 else strip_batch(a, 4)
            parsed = parse_grid(a.shape, a_hint, ctx.num_classes, layout_param)
            if parsed is None:
                raise InferenceError(tr("decoder.err.bad_shape", shape=list(np.shape(o)), kind=self.kind),
                                     hint=tr("decoder.err.bad_shape_hint"))
            layout, na, h, w, d = parsed
            grids.append((i, _to_ahwd(a, layout, na, h, w, d), h, w))
        ds = {g[1].shape[-1] for g in grids}
        if len(ds) != 1:
            raise InferenceError(tr("decoder.err.bad_shape", shape=[g[1].shape for g in grids], kind=self.kind),
                                 hint=tr("decoder.err.bad_shape_hint"))
        c = ds.pop() - 5
        check_classes(ctx, c)

        user_strides = self.params.get("strides")
        order = sorted(range(n_out), key=lambda k: (in_w / grids[k][3]))  # stride 由小到大
        rank = {k: r for r, k in enumerate(order)}

        scale_xy = cfg.get("scale_x_y", 1.0)
        boxes_l: list[np.ndarray] = []
        scores_l: list[np.ndarray] = []
        ids_l: list[np.ndarray] = []
        unit = cfg["anchors_unit"]
        for i, t, h, w in grids:
            r = rank[i]
            na = t.shape[0]
            if masks:
                mask = masks[r] if r < len(masks) else masks[-1]
            elif n_out == 1:
                mask = list(range(len(anchors)))
            else:
                per = len(anchors) // n_out
                mask = list(range(r * per, (r + 1) * per))
            if len(mask) != na or max(mask) >= len(anchors):
                raise InferenceError(tr("decoder.err.anchor_count", a=na, m=len(mask)),
                                     hint=tr("decoder.err.anchors_hint"))
            anc = anchors[mask]
            sx = (float(user_strides[i]) if user_strides else in_w / w)
            sy = (float(user_strides[i]) if user_strides else in_h / h)
            sxy = float(scale_xy[r] if isinstance(scale_xy, (list, tuple)) else scale_xy)

            mode = self.params.get("sigmoid", "auto")
            tail = t[..., 4:]
            act = want_sigmoid(mode, tail)
            obj = sigmoid(t[..., 4]) if act else t[..., 4]
            sel = np.nonzero(obj >= ctx.conf_threshold)
            if len(sel[0]) == 0:
                continue
            row = t[sel]                                    # (n, 5+C)
            o = obj[sel]
            if act:
                cls = row[:, 5:]
                ca = self.params.get("class_activation", "auto")
                if ca == "auto":
                    ca = "softmax" if unit == "grid" else "sigmoid"
                cls = softmax(cls, axis=1) if ca == "softmax" else sigmoid(cls) if ca == "sigmoid" else cls
                xy = sigmoid(row[:, :2])
            else:
                cls = row[:, 5:]
                xy = row[:, :2]
            xy = xy * sxy - (sxy - 1.0) / 2.0
            ai, yi, xi = sel
            cx = (xy[:, 0] + xi) * sx
            cy = (xy[:, 1] + yi) * sy
            aw, ah = anc[ai, 0], anc[ai, 1]
            if unit == "grid":
                aw, ah = aw * sx, ah * sy
            bw = np.exp(np.clip(row[:, 2], -20.0, 10.0)) * aw
            bh = np.exp(np.clip(row[:, 3], -20.0, 10.0)) * ah
            best = cls.max(axis=1) * o
            ok = best >= ctx.conf_threshold
            boxes_l.append(np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)[ok])
            scores_l.append(best[ok])
            ids_l.append(cls.argmax(axis=1)[ok])
        if not boxes_l:
            return DecodeOutput(task="detect")
        return make_output(np.concatenate(boxes_l), np.concatenate(scores_l), np.concatenate(ids_l),
                           ctx.conf_threshold)
