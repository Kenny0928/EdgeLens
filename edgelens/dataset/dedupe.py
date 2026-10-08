"""以 64 位元 dHash 找相似構圖，不把它當作影像完全相同的證明。"""
from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np

from edgelens.metrics.quality import gray_image


def dhash(image: np.ndarray) -> int:
    gray = cv2.resize(gray_image(image), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (gray[:, 1:] > gray[:, :-1]).reshape(-1)
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def _hash(value: int | str) -> int:
    value = int(value, 16) if isinstance(value, str) else int(value)
    if not 0 <= value < 2**64:
        raise ValueError("雜湊值超出 64 位元，請重新計算照片雜湊。")
    return value


def hamming_distance(a: int | str, b: int | str) -> int:
    return (_hash(a) ^ _hash(b)).bit_count()


def find_near_duplicates(records: Iterable, max_distance: int = 5) -> list[tuple[Any, Any, int]]:
    if not 0 <= max_distance <= 64:
        raise ValueError("相似距離必須介於 0–64，請重新設定。")
    items = [(r, _hash(r.dhash)) for r in records if r.dhash is not None and r.dhash != ""]
    return [(a, b, distance) for i, (a, ha) in enumerate(items) for b, hb in items[i + 1:]
            if (distance := hamming_distance(ha, hb)) <= max_distance]


class RecentHashes:
    """check 不加入紀錄；儲存成功後才 add，避免失敗的拍照佔用視窗。"""

    def __init__(self, capacity: int = 100):
        if capacity < 1:
            raise ValueError("最近照片數必須至少為 1，請重新設定。")
        self._items: deque = deque(maxlen=capacity)
        self._lock = threading.RLock()

    def add(self, record: Any, hash_value: int | str | None = None) -> None:
        value = _hash(record.dhash if hash_value is None else hash_value)
        with self._lock:
            self._items.append((record, value))

    def check(self, image_or_hash: np.ndarray | int | str) -> tuple[Any, int] | None:
        value = dhash(image_or_hash) if isinstance(image_or_hash, np.ndarray) else _hash(image_or_hash)
        with self._lock:
            return min(((r, hamming_distance(value, h)) for r, h in self._items),
                       key=lambda pair: pair[1], default=None)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
