"""非極大值抑制（NMS）與框格式轉換，純 numpy，不依賴 cv2.dnn。

為什麼自己寫：cv2.dnn.NMSBoxes 的參數格式容易搞錯、各版本行為略有不同，而且 EdgeLens 要在
每個平台上得到一樣的結果。這裡的實作對 8000 個候選框約 1–3 毫秒。
"""

from __future__ import annotations

import numpy as np

BOX_FORMATS = ("xyxy", "xywh", "cxcywh", "yxyx")


def box_convert(boxes: np.ndarray, fmt: str) -> np.ndarray:
    """把 (N,4) 的框從 fmt 轉成 xyxy（左上、右下）。

    fmt：
      "xyxy"    (x1, y1, x2, y2)
      "xywh"    (左上 x, 左上 y, 寬, 高)
      "cxcywh"  (中心 x, 中心 y, 寬, 高)，YOLO 系列最常見
      "yxyx"    (y1, x1, y2, x2)，TFLite SSD 的順序
    回傳新的 float32 陣列，不會改到輸入。
    """
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if fmt == "xyxy":
        return b.copy()
    out = np.empty_like(b)
    if fmt == "xywh":
        out[:, 0], out[:, 1] = b[:, 0], b[:, 1]
        out[:, 2], out[:, 3] = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    elif fmt == "cxcywh":
        out[:, 0], out[:, 1] = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2
        out[:, 2], out[:, 3] = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    elif fmt == "yxyx":
        out[:, 0], out[:, 1], out[:, 2], out[:, 3] = b[:, 1], b[:, 0], b[:, 3], b[:, 2]
    else:
        raise ValueError(f"不認識的框格式 {fmt!r}；可用：{', '.join(BOX_FORMATS)}")
    return out


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """xyxy 框的 IoU。a (N,4)、b (M,4) → (N,M)。"""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    lt_x = np.maximum(a[:, None, 0], b[None, :, 0])
    lt_y = np.maximum(a[:, None, 1], b[None, :, 1])
    rb_x = np.minimum(a[:, None, 2], b[None, :, 2])
    rb_y = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(rb_x - lt_x, 0, None) * np.clip(rb_y - lt_y, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45, max_det: int = 100) -> np.ndarray:
    """貪婪式 NMS：分數高的先留下，和它重疊（IoU > 門檻）的刪掉。

    回傳保留框的索引（int64，分數由高到低），最多 max_det 個。
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(boxes) == 0 or max_det <= 0:
        return np.zeros((0,), np.int64)
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    keep: list[int] = []
    while order.size and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if rest.size == 0:
            break
        w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = w * h
        ov = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[ov <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float = 0.45,
                max_det: int = 100, agnostic: bool = False) -> np.ndarray:
    """每個類別各自做 NMS（agnostic=True 時所有類別一起做）。

    技巧：把不同類別的框平移到彼此不重疊的位置（座標加上 類別編號 × 偏移量），
    就能一次 NMS 處理所有類別，不用迴圈。
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros((0,), np.int64)
    if agnostic:
        return nms(boxes, scores, iou_threshold, max_det)
    class_ids = np.asarray(class_ids).reshape(-1)
    span = float(np.abs(boxes).max()) + 1.0
    offset = (class_ids.astype(np.float32) * span)[:, None]
    return nms(boxes + offset, scores, iou_threshold, max_det)
