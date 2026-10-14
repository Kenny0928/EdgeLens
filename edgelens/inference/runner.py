"""InferenceRunner：把前處理、後端、解碼器、NMS 串成「畫面進、結果出」。

GUI 的用法：

    card = load_model_file(path)
    runner = InferenceRunner(card)              # backend=None 時自動挑
    runner.load()                               # 失敗丟 InferenceError（message／hint 可直接顯示）
    runner.warmup()
    result = runner.run(frame)                  # InferenceResult，座標是原始畫面座標
    runner.conf_threshold = 0.4                 # 滑桿隨時可改
    runner.info.guesses                         # 「可能是…」候選與理由
    runner.set_decoder("yolo_t")                # 自動判斷錯了就手動指定
    runner.class_filter = frozenset({0, 7})     # 只顯示這些類別編號；None＝全部

一個 runner 只在一條執行緒裡使用；門檻屬性可由 GUI 執行緒修改（run 開頭複製到區域變數）。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from edgelens.core.frame import Frame
from edgelens.core.modelcard import ModelCard
from edgelens.core.registry import discover
from edgelens.core.results import Classification, Detection, InferenceResult
from edgelens.i18n import current_language, tr
from edgelens.inference.autodetect import FormatGuess, detect_format
from edgelens.inference.base import (
    BACKENDS,
    DECODERS,
    Backend,
    DecodeContext,
    DecodeOutput,
    Decoder,
    InferenceError,
    TensorInfo,
    dequantize,
)
from edgelens.inference.nms import batched_nms
from edgelens.inference.preprocess import describe_input, preprocess, resolve_input

log = logging.getLogger(__name__)

TOP_K = 5
AMBIGUOUS_SCORE = 0.5       # 最佳候選低於這個分數就請使用者確認
AMBIGUOUS_GAP = 0.08        # 前兩名差距小於這個也請使用者確認


@dataclass
class BackendChoice:
    """GUI 後端下拉選單的一列。ok=False 時灰色顯示，reason 說明為什麼不能用。"""

    kind: str
    name: str
    ok: bool
    reason: str = ""
    priority: int = 0


@dataclass
class RunnerInfo:
    """載入後給模型面板顯示的資訊。"""

    backend: str = ""
    backend_desc: str = ""
    mode: str = "tensor"                      # tensor／image／device
    task: str = ""                            # detect／classify
    decoder: str = ""
    decoder_reason: str = ""
    guesses: list[FormatGuess] = field(default_factory=list)
    ambiguous: bool = False
    input_desc: str = ""
    inputs: list[TensorInfo] = field(default_factory=list)
    outputs: list[TensorInfo] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    load_ms: float = 0.0
    loaded: bool = False


def load_model_file(path: str | Path) -> ModelCard:
    """讀模型檔或說明檔 → ModelCard。比 ModelCard.load 多做的事：
    - 傳入 .cfg 時改找同名的 .weights。
    - Darknet 的 .weights 找不到同名 .cfg 時，資料夾裡只有一個 .cfg 就用它。
    - 新建立的 Darknet 說明預設用 stretch 縮放（Darknet 訓練時就是直接拉伸，不補邊）。
    - 說明檔有目前介面語言的類別名稱（labels_i18n）就換成該語言。
    """
    p = Path(path)
    if p.suffix.lower() == ".cfg":
        w = p.with_suffix(".weights")
        if w.is_file():
            p = w
    card = ModelCard.load(p)
    if card.card_path is None and p.suffix.lower() == ".weights":
        if "config" not in card.extra_files:
            cfgs = sorted(p.parent.glob("*.cfg"))
            if len(cfgs) == 1:
                card.extra_files["config"] = str(cfgs[0].resolve())
        card.input.resize = "stretch"
    card.localize_labels(current_language())
    return card


class InferenceRunner:
    def __init__(self, card: ModelCard, backend: str | None = None):
        discover()
        self.card = card
        self._requested = backend or (card.backend if card.backend and card.backend != "auto" else None)
        self._backend: Backend | None = None
        self.info = RunnerInfo()
        self.conf_threshold: float = card.conf_threshold
        self.nms_threshold: float = card.nms_threshold
        self.max_det: int = card.max_det
        self.class_agnostic: bool = card.class_agnostic_nms
        self.class_filter: frozenset[int] | None = frozenset(card.class_filter) if card.class_filter else None
        self._decoder: Decoder | None = None
        self._final = False              # True＝已用真實輸出確認過格式（或使用者手動指定）
        self._manual = False
        self._labels: list[str] = []
        self._darknet = False

    # ---- 後端選擇 --------------------------------------------------------------

    @staticmethod
    def backend_choices(card: ModelCard) -> list[BackendChoice]:
        """這個模型檔可以用哪些後端（副檔名相關的都列出，不可用的標 ok=False 與原因）。"""
        discover()
        choices: list[BackendChoice] = []
        ext = card.extension
        for kind, cls in BACKENDS.items():
            if ext not in cls.extensions:
                continue
            ok, reason = cls.available()
            if ok and not cls.can_load(card):
                ok, reason = False, tr("runner.cannot_load", ext=ext)
            choices.append(BackendChoice(kind, tr(cls.display_name_key), ok, reason, cls.priority))
        choices.sort(key=lambda c: (-int(c.ok), -c.priority, c.kind))
        return choices

    def _pick_backend(self) -> type[Backend]:
        choices = self.backend_choices(self.card)
        if self._requested:
            cls = BACKENDS.find(self._requested)
            if cls is None:
                raise InferenceError(tr("runner.unknown_backend", kind=self._requested),
                                     hint=tr("runner.unknown_backend_hint", kinds=", ".join(BACKENDS.kinds())))
            ch = next((c for c in choices if c.kind == self._requested), None)
            if ch is None or not ch.ok:
                reason = ch.reason if ch else tr("runner.cannot_load", ext=self.card.extension)
                raise InferenceError(tr("runner.backend_unusable", name=tr(cls.display_name_key), reason=reason),
                                     hint=self._alternative_hint(choices))
            return cls
        ok = [c for c in choices if c.ok]
        if not ok:
            if choices:
                msg = tr("runner.none_usable", ext=self.card.extension,
                         reasons="；".join(f"{c.name}：{c.reason}" for c in choices))
                raise InferenceError(msg, hint=tr("runner.none_usable_hint"))
            raise InferenceError(tr("runner.unsupported_ext", ext=self.card.extension or "?"),
                                 hint=tr("runner.unsupported_ext_hint"))
        return BACKENDS.get(ok[0].kind)

    @staticmethod
    def _alternative_hint(choices: list[BackendChoice]) -> str:
        alts = [c.name for c in choices if c.ok]
        return tr("runner.try_other", names="、".join(alts)) if alts else tr("runner.none_usable_hint")

    # ---- 載入 ------------------------------------------------------------------

    def load(self) -> None:
        self.close()
        cls = self._pick_backend()
        backend = cls(self.card)
        t0 = time.perf_counter()
        backend.load()
        load_ms = (time.perf_counter() - t0) * 1000.0
        self._backend = backend
        meta = backend.metadata()
        self._darknet = bool(meta.get("darknet"))
        self._labels = list(self.card.labels) or [str(x) for x in meta.get("names", [])]
        self._decoder, self._final, self._manual = None, False, False
        self.info = RunnerInfo(
            backend=cls.kind, backend_desc=backend.describe(), mode=cls.input_mode, labels=list(self._labels),
            inputs=backend.inputs(), outputs=backend.outputs(), load_ms=load_ms, loaded=True,
            task=self.card.task if self.card.task in ("detect", "classify") else "detect")
        if cls.input_mode == "tensor":
            in0 = backend.inputs()[0] if backend.inputs() else None
            self.info.input_desc = describe_input(in0, self.card.input)
            try:
                self._prepare_decoder(None, backend.outputs())
            except InferenceError:
                log.info("載入時還無法判斷輸出格式，等第一次推論有實際輸出再判斷")
        elif cls.input_mode == "device":
            self.info.decoder_reason = tr("runner.device_reason")
        else:
            self.info.decoder_reason = tr("runner.image_reason")

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            raise InferenceError(tr("runner.not_loaded"), hint=tr("runner.not_loaded_hint"))
        return self._backend

    def close(self) -> None:
        if self._backend is not None:
            try:
                self._backend.close()
            finally:
                self._backend = None
        self.info.loaded = False

    # ---- 格式判斷 --------------------------------------------------------------

    def _input_plan(self):
        ins = self.backend.inputs()
        return resolve_input(self.card.input, ins[0] if ins else None)

    def _effective_infos(self, arrays: list[np.ndarray] | None) -> list[TensorInfo]:
        """後端回報的輸出資訊；形狀有未知維度（None）時用實際輸出補上。"""
        infos = list(self.backend.outputs())
        if arrays is None:
            return infos
        if len(infos) != len(arrays):
            return [TensorInfo(f"output{i}", tuple(a.shape), str(a.dtype)) for i, a in enumerate(arrays)]
        out = []
        for info, a in zip(infos, arrays, strict=True):
            if any(d is None for d in info.shape) or len(info.shape) != a.ndim:
                info = replace(info, shape=tuple(int(d) for d in a.shape))
            out.append(info)
        return out

    def _adapt(self, arrays: list[np.ndarray], infos: list[TensorInfo]) -> tuple[list[np.ndarray], list[TensorInfo]]:
        """Darknet（OpenCV 已解碼）的多個 [N, 5+C] 輸出合併成一個 [1, N_total, 5+C]。"""
        if not self._darknet:
            return arrays, infos
        flat = [a.reshape(-1, a.shape[-1]) for a in arrays if a.ndim >= 2]
        if not flat or len({f.shape[1] for f in flat}) != 1:
            return arrays, infos
        merged = np.concatenate(flat, axis=0)[None]
        return [merged], [TensorInfo("darknet_yolo", tuple(merged.shape), "float32")]

    def _prepare_decoder(self, arrays: list[np.ndarray] | None, infos: list[TensorInfo], final: bool = False) -> None:
        """依輸出形狀（與實際數值）判斷解碼器；arrays 為已反量化、已 _adapt 的輸出（可為 None）。"""
        if self._manual:
            return
        if self._darknet:
            shape = _shape_text(arrays[0].shape) if arrays else "[1, N, 5+C]"
            guesses = [FormatGuess("yolo_flat", 0.95, {"normalized": True, "box_format": "cxcywh"},
                                   tr("runner.darknet_reason", shape=shape))]
        else:
            guesses = detect_format(infos, self.card, samples=arrays, input_size=self._input_plan().size)
        if not guesses:
            raise InferenceError(tr("runner.no_format", shapes="、".join(i.describe() for i in infos)),
                                 hint=tr("runner.no_format_hint"))
        self.info.guesses = guesses
        self._apply_guess(guesses[0])
        self.info.ambiguous = _is_ambiguous(guesses)
        if arrays is not None and final:
            self._final = True

    def _apply_guess(self, g: FormatGuess) -> None:
        cls = DECODERS.get(g.decoder_kind)
        self._decoder = cls(self.card, g.params)
        self.info.decoder = g.decoder_kind
        self.info.decoder_reason = g.reason
        self.info.task = cls.task

    def set_decoder(self, kind: str, params: dict[str, Any] | None = None) -> None:
        """使用者手動指定解碼器（自動判斷有疑義時）。之後不再自動判斷。"""
        cls = DECODERS.find(kind)
        if cls is None:
            raise InferenceError(tr("runner.unknown_decoder", kind=kind),
                                 hint=tr("runner.unknown_decoder_hint", kinds=", ".join(DECODERS.kinds())))
        self._decoder = cls(self.card, params)
        self._manual = True
        self.info.decoder = kind
        self.info.task = cls.task
        self.info.decoder_reason = tr("runner.manual_reason")
        self.info.ambiguous = False

    # ---- 推論 ------------------------------------------------------------------

    def _infer_arrays(self, image: np.ndarray):
        backend = self.backend
        ins = backend.inputs()
        in0 = ins[0] if ins else None
        t0 = time.perf_counter()
        tensor, meta = preprocess(image, self.card.input, in0)
        t1 = time.perf_counter()
        raw = backend.infer(tensor)
        t2 = time.perf_counter()
        infos = self._effective_infos(raw)
        arrays = [dequantize(a, i) for a, i in zip(raw, infos, strict=False)]
        arrays, infos = self._adapt(arrays, infos)
        return arrays, infos, meta, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0

    def raw(self, frame: Frame) -> list[tuple[TensorInfo, np.ndarray]]:
        """原始輸出檢視器：回傳 (輸出資訊, 反量化後的陣列)。"""
        if self.backend.input_mode != "tensor":
            raise InferenceError(tr("runner.no_raw"), hint="")
        raw = self.backend.infer(preprocess(frame.image, self.card.input, (self.backend.inputs() or [None])[0])[0])
        infos = self._effective_infos(raw)
        arrays = [dequantize(a, i) for a, i in zip(raw, infos, strict=False)]
        return list(zip(infos, arrays, strict=True))

    def warmup(self, n: int = 2) -> None:
        """用空白影像先跑幾次（第一次推論通常很慢，會讓畫面卡一下）。"""
        backend = self.backend
        if backend.input_mode != "tensor":
            return
        size = self._input_plan().size
        blank = np.full((size[1], size[0], 3), 114, np.uint8)
        for _ in range(max(0, n)):
            self._infer_arrays(blank)

    def run(self, frame: Frame) -> InferenceResult:
        t_start = time.perf_counter()
        backend = self.backend
        conf, nms_t = float(self.conf_threshold), float(self.nms_threshold)
        max_det, agnostic = int(self.max_det), bool(self.class_agnostic)
        keep = self.class_filter
        mode = backend.input_mode
        if mode == "device":
            return self._run_device(frame, conf, t_start, keep)
        if mode == "image":
            res = backend.detect_image(frame.image, frame.seq)
            return self._finish_image(res, frame, t_start)
        arrays, infos, meta, t_pre, t_inf = self._infer_arrays(frame.image)
        if not self._final and not self._manual:
            self._prepare_decoder(arrays, infos, final=True)   # 第一次有真實輸出：用數值重新判斷並快取
        dec = self._decoder
        assert dec is not None
        t3 = time.perf_counter()
        plan = self._input_plan()
        ctx = DecodeContext(card=self.card, input_size=plan.size, conf_threshold=conf,
                            num_classes=self.card.num_classes, output_infos=infos)
        out = dec.decode(arrays, ctx)
        if out.task == "classify":
            result = InferenceResult(kind="classify", model_name=self.card.name, frame_seq=frame.seq,
                                     top_k=self._top_k(out), image_size=frame.size)
        else:
            result = InferenceResult(kind="detect", model_name=self.card.name, frame_seq=frame.seq,
                                     detections=self._detections(out, meta, nms_t, max_det, agnostic, keep),
                                     image_size=frame.size)
        t4 = time.perf_counter()
        result.timings_ms = {"preprocess": t_pre, "infer": t_inf, "postprocess": (t4 - t3) * 1000.0,
                             "total": (t4 - t_start) * 1000.0}
        return result

    # ---- 後處理 ----------------------------------------------------------------

    def _label(self, class_id: int) -> str:
        return self._labels[class_id] if 0 <= class_id < len(self._labels) else str(class_id)

    def _top_k(self, out: DecodeOutput) -> list[Classification]:
        scores = np.asarray(out.class_scores, dtype=np.float32).reshape(-1)
        order = np.argsort(-scores, kind="stable")[:TOP_K]
        return [Classification(int(i), float(scores[i]), self._label(int(i))) for i in order]

    def _detections(self, out: DecodeOutput, meta, nms_t: float, max_det: int, agnostic: bool,
                    keep: frozenset[int] | None = None) -> list[Detection]:
        if keep is not None and len(out.scores):
            # 在 NMS 之前篩：沒選的類別不會占掉 max_det 的名額，也不會在不分類別 NMS 時壓掉選了的框
            mask = np.isin(out.class_ids, np.fromiter(keep, dtype=np.int64, count=len(keep)))
            out = replace(out, boxes=out.boxes[mask], scores=out.scores[mask], class_ids=out.class_ids[mask])
        if len(out.scores) == 0:
            return []
        if out.nms_done:
            idx = np.argsort(-out.scores, kind="stable")[:max_det]
        else:
            idx = batched_nms(out.boxes, out.scores, out.class_ids, nms_t, max_det, agnostic)
        if len(idx) == 0:
            return []
        boxes = meta.to_original(out.boxes[idx])
        dets = []
        for b, s, c in zip(boxes, out.scores[idx], out.class_ids[idx], strict=True):
            if b[2] - b[0] < 1.0 or b[3] - b[1] < 1.0:        # 完全在畫面外或被裁成一條線
                continue
            dets.append(Detection(box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])), score=float(s),
                                  class_id=int(c), label=self._label(int(c))))
        return dets

    def _finish_image(self, res: InferenceResult, frame: Frame, t_start: float) -> InferenceResult:
        # Haar 沒有真正的機率，信心門檻不套用（靈敏度由 min_neighbors 控制），只套用 max_det
        res.detections = res.detections[: int(self.max_det)]
        res.timings_ms.setdefault("preprocess", 0.0)
        res.timings_ms.setdefault("infer", 0.0)
        res.timings_ms["postprocess"] = 0.0
        res.timings_ms["total"] = (time.perf_counter() - t_start) * 1000.0
        res.image_size = frame.size
        return res

    def _run_device(self, frame: Frame, conf: float, t_start: float,
                    keep: frozenset[int] | None = None) -> InferenceResult:
        dr = frame.device_result
        kind = "classify" if self.card.task == "classify" else "detect"
        if dr is None:
            res = InferenceResult(kind=kind, model_name=self.card.name, frame_seq=frame.seq, where="device",
                                  image_size=frame.size, extra={"device_missing": True})
        else:
            dets = [replace(d, label=self._label(d.class_id) if self._labels else (d.label or str(d.class_id)))
                    for d in dr.detections if d.score >= conf and (keep is None or d.class_id in keep)]
            top = [replace(c, label=self._label(c.class_id) if self._labels else (c.label or str(c.class_id)))
                   for c in dr.top_k]
            res = replace(dr, model_name=self.card.name, frame_seq=frame.seq, where="device",
                          detections=dets, top_k=top, timings_ms=dict(dr.timings_ms),
                          image_size=dr.image_size if dr.image_size != (0, 0) else frame.size)
        self.info.task = res.kind if res.kind in ("detect", "classify") else self.info.task
        res.timings_ms.setdefault("preprocess", 0.0)
        res.timings_ms.setdefault("infer", res.timings_ms.get("infer", 0.0))
        res.timings_ms["postprocess"] = 0.0
        res.timings_ms["total"] = (time.perf_counter() - t_start) * 1000.0
        return res


def _is_ambiguous(guesses: list[FormatGuess]) -> bool:
    if not guesses:
        return True
    if guesses[0].score < AMBIGUOUS_SCORE:
        return True
    return len(guesses) > 1 and guesses[0].score - guesses[1].score < AMBIGUOUS_GAP


def _shape_text(shape: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(d) for d in shape) + "]"
