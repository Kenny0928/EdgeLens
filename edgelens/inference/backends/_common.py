"""後端共用的小工具（底線開頭的模組不會被當成後端註冊）。"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

from edgelens.core.modelcard import ModelCard
from edgelens.i18n import tr
from edgelens.inference.base import InferenceError


def default_threads() -> int:
    """預設執行緒數：不要吃光 CPU，讓 GUI 保持順暢。"""
    return max(1, min(4, (os.cpu_count() or 2) - 1 or 1))


def card_threads(card: ModelCard) -> int:
    """說明檔 device.num_threads 可覆寫（給高階使用者），否則用預設值。"""
    try:
        n = int(card.device.get("num_threads", 0))
    except (TypeError, ValueError):
        n = 0
    return n if n > 0 else default_threads()


def read_model_bytes(path: Path) -> bytes:
    """用 Python 讀檔，中文路徑在 Windows 上也不會壞（C++ 函式庫常常讀不到中文路徑）。"""
    if not path.is_file():
        raise InferenceError(tr("backend.err.file_missing", path=str(path)), hint=tr("backend.err.file_missing_hint"))
    try:
        return path.read_bytes()
    except OSError as e:
        raise InferenceError(tr("backend.err.read_failed", path=str(path), error=str(e)),
                             hint=tr("backend.err.file_missing_hint")) from e


def parse_names(value: Any) -> list[str]:
    """把 ultralytics／YOLOv5 寫在模型 metadata 的 names 轉成清單。

    格式是 Python 字串，例如 "{0: 'person', 1: 'bicycle'}" 或 "['a', 'b']"。
    用 ast.literal_eval 安全解析（絕不 eval）；失敗就回傳空清單，由使用者自己提供 labels。
    """
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return []
    if isinstance(value, dict):
        try:
            items = sorted(((int(k), str(v)) for k, v in value.items()), key=lambda kv: kv[0])
        except (TypeError, ValueError):
            return []
        # 編號必須是 0..N-1 連續，否則位置會對不上
        if [k for k, _ in items] != list(range(len(items))):
            return []
        return [v for _, v in items]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return []


def dtype_name(text: str) -> str:
    """'tensor(float)' 之類的 ONNX 型別字串 → numpy 型別名稱。"""
    t = text.lower()
    table = {"float16": "float16", "double": "float64", "float": "float32", "uint8": "uint8", "int8": "int8",
             "uint16": "uint16", "int16": "int16", "int32": "int32", "int64": "int64", "bool": "bool"}
    inner = t[t.find("(") + 1:t.rfind(")")] if "(" in t else t
    return table.get(inner, "float32")
