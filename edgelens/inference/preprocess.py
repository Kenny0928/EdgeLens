"""前處理：把相機畫面（BGR 或灰階 uint8）變成模型要的輸入張量。

用法（runner）：

    tensor, meta = preprocess(frame.image, card.input, backend.inputs()[0])
    outs = backend.infer(tensor)
    ...解碼後的框是「輸入座標」，用 meta.to_original(boxes) 換回原始畫面座標。

需要輸入大小（給 DecodeContext）時用 resolve_input(spec, info).size。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from edgelens.core.modelcard import InputSpec
from edgelens.i18n import tr
from edgelens.inference.base import PreprocessMeta, TensorInfo

DEFAULT_SIZE = (640, 640)


@dataclass
class InputPlan:
    """由說明檔與模型輸入形狀合併出的實際前處理計畫。"""

    size: tuple[int, int]      # (寬, 高)
    layout: str                # "NCHW" 或 "NHWC"
    channels: int              # 1 或 3
    dtype: str                 # "float32"、"float16"、"uint8"、"int8"


def _norm_dtype(name: str) -> str:
    n = (name or "").lower()
    if "uint8" in n:
        return "uint8"
    if "int8" in n:
        return "int8"
    if "float16" in n or "half" in n:
        return "float16"
    return "float32"


def _layout_from_shape(shape: tuple[int | None, ...]) -> str:
    """4 維形狀判斷 NCHW／NHWC：通道數（1 或 3）在第 1 軸就是 NCHW，在最後一軸就是 NHWC。"""
    if len(shape) != 4:
        return "NCHW"
    if shape[1] in (1, 3):
        return "NCHW"
    if shape[3] in (1, 3):
        return "NHWC"
    return "NCHW"


def resolve_input(spec: InputSpec, info: TensorInfo | None) -> InputPlan:
    """決定輸入大小、排列方式、通道數、型別。說明檔明確設定的值優先，其餘看模型輸入形狀。"""
    shape = tuple(info.shape) if info is not None else ()
    layout = spec.layout.upper() if spec.layout and spec.layout.lower() != "auto" else _layout_from_shape(shape)

    size = spec.size
    if size is None and len(shape) == 4:
        h, w = (shape[2], shape[3]) if layout == "NCHW" else (shape[1], shape[2])
        size = (int(w) if w else DEFAULT_SIZE[0], int(h) if h else DEFAULT_SIZE[1])
    if size is None:
        size = DEFAULT_SIZE

    channels = 0
    if len(shape) == 4:
        c = shape[1] if layout == "NCHW" else shape[3]
        if c in (1, 3):
            channels = int(c)
    if channels == 0:
        channels = 1 if spec.color.upper() == "GRAY" else 3

    dtype = _norm_dtype(spec.dtype if spec.dtype != "auto" else (info.dtype if info is not None else "float32"))
    return InputPlan(size=(int(size[0]), int(size[1])), layout=layout, channels=channels, dtype=dtype)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    """灰階、BGRA、單通道 3 維都轉成 3 通道 BGR uint8。"""
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _resize(img: np.ndarray, size: tuple[int, int], mode: str, pad_value: int
            ) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """回傳 (縮放後影像, scale, pad)；輸入座標 = 原始座標 * scale + pad。"""
    oh, ow = img.shape[:2]
    w, h = size
    if mode == "stretch":
        out = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        return out, (w / ow, h / oh), (0.0, 0.0)
    if mode == "center_crop":
        r = max(w / ow, h / oh)
        nw, nh = max(w, round(ow * r)), max(h, round(oh * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        left, top = (nw - w) // 2, (nh - h) // 2
        out = np.ascontiguousarray(resized[top:top + h, left:left + w])
        return out, (nw / ow, nh / oh), (float(-left), float(-top))
    # letterbox
    r = min(w / ow, h / oh)
    nw, nh = max(1, min(w, round(ow * r))), max(1, min(h, round(oh * r)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (ow, oh) else img
    out = np.full((h, w, 3), pad_value, dtype=np.uint8)
    left, top = (w - nw) // 2, (h - nh) // 2
    out[top:top + nh, left:left + nw] = resized
    return out, (nw / ow, nh / oh), (float(left), float(top))


def _per_channel(values: tuple[float, ...], channels: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    if v.size == 0:
        v = np.zeros(1, np.float32)
    if channels == 1:
        return np.array([v.mean()], np.float32)
    if v.size >= channels:
        return v[:channels]
    return np.concatenate([v, np.repeat(v[-1], channels - v.size)])


def preprocess(image: np.ndarray, spec: InputSpec, input_info: TensorInfo | None = None
               ) -> tuple[np.ndarray, PreprocessMeta]:
    """相機畫面 → (輸入張量, PreprocessMeta)。

    image：BGR 或灰階 uint8（相機來源的 Frame.image）。
    spec：card.input；input_info：模型第一個輸入（可為 None）。
    張量有 batch 維度、C-contiguous：
      float 型別   (像素 * scale - mean) / std
      uint8／int8  有 input_info.quant 時依 (scale, zero_point) 量化；
                   uint8 沒有量化資訊 → 直接放 0–255 像素值（許多 TFLite 模型如此）；
                   int8 沒有量化資訊 → 像素值 - 128。
    """
    plan = resolve_input(spec, input_info)
    bgr = _to_bgr(image)
    oh, ow = bgr.shape[:2]
    resized, scale, pad = _resize(bgr, plan.size, spec.resize, spec.pad_value)
    meta = PreprocessMeta(orig_size=(ow, oh), input_size=plan.size, scale=scale, pad=pad, mode=spec.resize)

    color = spec.color.upper()
    if color == "RGB":
        px = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    elif color == "GRAY":
        g = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        px = g[:, :, None]
    else:
        px = resized
    # 模型通道數和顏色設定不一致時以模型為準（灰階圖餵 3 通道模型就複製三份）
    if plan.channels == 1 and px.shape[2] == 3:
        px = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)[:, :, None]
    elif plan.channels == 3 and px.shape[2] == 1:
        px = np.repeat(px, 3, axis=2)

    c = px.shape[2]
    if plan.dtype == "uint8" and (input_info is None or input_info.quant is None):
        arr = px.astype(np.uint8)
    elif plan.dtype == "int8" and (input_info is None or input_info.quant is None):
        arr = (px.astype(np.int16) - 128).astype(np.int8)
    else:
        real = (px.astype(np.float32) * spec.scale - _per_channel(spec.mean, c)) / _per_channel(spec.std, c)
        if plan.dtype in ("uint8", "int8") and input_info is not None and input_info.quant is not None:
            q_scale, zp = input_info.quant
            info = np.iinfo(np.uint8 if plan.dtype == "uint8" else np.int8)
            q = np.round(real / (q_scale or 1.0)) + zp
            arr = np.clip(q, info.min, info.max).astype(np.uint8 if plan.dtype == "uint8" else np.int8)
        elif plan.dtype == "float16":
            arr = real.astype(np.float16)
        else:
            arr = real.astype(np.float32)

    if plan.layout == "NCHW":
        arr = arr.transpose(2, 0, 1)
    return np.ascontiguousarray(arr[None]), meta


def describe_input(info: TensorInfo | None, spec: InputSpec) -> str:
    """給模型面板顯示，例如「640×640 RGB NCHW float32，等比例縮放補邊」。"""
    plan = resolve_input(spec, info)
    color = spec.color.upper() if plan.channels == 3 or spec.color.upper() == "GRAY" else "GRAY"
    if plan.channels == 1:
        color = "GRAY"
    resize = tr(f"preprocess.resize.{spec.resize}") if spec.resize in ("letterbox", "stretch", "center_crop") \
        else spec.resize
    return f"{plan.size[0]}×{plan.size[1]} {color} {plan.layout} {plan.dtype}，{resize}"
