"""ControlSpec：宣告一個可調整的參數（相機設定、處理步驟參數、模型門檻）。

GUI 依 ControlSpec 自動產生面板，所以 webcam、ESP32-CAM、K210、各處理步驟
都用同一套面板程式；新增參數只要多宣告一個 ControlSpec，不用改 GUI。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ControlType(str, Enum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    ENUM = "enum"      # 從 options 選一個
    STR = "str"        # 文字（網址、序列埠名稱等）
    COLOR = "color"    # "#RRGGBB"
    ACTION = "action"  # 按鈕，沒有值；由 run_action() 執行


@dataclass(frozen=True)
class ControlSpec:
    """一個參數的宣告。

    key        穩定的英文識別名稱，存檔與程式都用它（例如 "brightness"）。
    label_key  i18n 鍵（例如 "control.brightness"）；找不到翻譯時顯示 label，再不行顯示 key。
    options    ENUM 用：((值, i18n 鍵或顯示文字), ...)。
    live       True 表示串流中可以直接改；False 表示要重新開啟來源才會生效。
    readback   True 表示來源能讀回實際值（寫入後可確認）。
    group      面板分組：camera、image、advanced、inference、action、source。
    advanced   True 時預設收在「進階」裡，避免學生一開始看到太多選項。
    """

    key: str
    type: ControlType
    label_key: str = ""
    label: str = ""
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: tuple[tuple[Any, str], ...] = ()
    live: bool = True
    readback: bool = False
    group: str = "camera"
    unit: str = ""
    help_key: str = ""
    advanced: bool = False
    extra: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("ControlSpec.key 不可為空")
        if self.type == ControlType.ENUM and not self.options:
            raise ValueError(f"ENUM 參數 {self.key} 必須提供 options")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"參數 {self.key} 的 min 大於 max")

    @property
    def option_values(self) -> list[Any]:
        return [v for v, _ in self.options]

    def validate(self, value: Any) -> Any:
        """把輸入值轉成正確型別並限制在範圍內；無法轉換時丟出 ValueError。"""
        t = self.type
        if t == ControlType.ACTION:
            return None
        if t == ControlType.BOOL:
            if isinstance(value, str):
                v = value.strip().lower()
                if v in ("1", "true", "yes", "on"):
                    return True
                if v in ("0", "false", "no", "off", ""):
                    return False
                raise ValueError(f"{self.key}：無法把 {value!r} 當成布林值")
            return bool(value)
        if t in (ControlType.INT, ControlType.FLOAT):
            try:
                num = float(value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"{self.key}：{value!r} 不是數字") from e
            if math.isnan(num):
                raise ValueError(f"{self.key}：不可為 NaN")
            if self.min is not None:
                num = max(num, float(self.min))
            if self.max is not None:
                num = min(num, float(self.max))
            return int(round(num)) if t == ControlType.INT else num
        if t == ControlType.ENUM:
            if value in self.option_values:
                return value
            # 允許用字串表示的數字選項（例如設定檔存成 "1"）
            for v in self.option_values:
                if str(v) == str(value):
                    return v
            raise ValueError(f"{self.key}：{value!r} 不在可選範圍 {self.option_values}")
        if t == ControlType.COLOR:
            s = str(value).strip()
            if len(s) == 7 and s.startswith("#"):
                int(s[1:], 16)
                return s.upper()
            raise ValueError(f"{self.key}：顏色格式必須是 #RRGGBB")
        return "" if value is None else str(value)

    def default_value(self) -> Any:
        if self.default is not None:
            return self.validate(self.default)
        if self.type == ControlType.ENUM:
            return self.options[0][0]
        if self.type == ControlType.BOOL:
            return False
        if self.type in (ControlType.INT, ControlType.FLOAT):
            base = self.min if self.min is not None else 0
            return self.validate(base)
        if self.type == ControlType.ACTION:
            return None
        return ""


@dataclass
class ControlResult:
    """set_control／run_action 的結果。

    ok        指令是否送出成功。
    value     實際值：能讀回時是讀回值，否則是送出的值。
    verified  True＝讀回確認一致；False＝讀回不一致（硬體不接受）；None＝無法確認。
    message   要給使用者看的說明（失敗原因、提示）。
    """

    ok: bool
    value: Any = None
    verified: bool | None = None
    message: str = ""

    @classmethod
    def failed(cls, message: str, value: Any = None) -> ControlResult:
        return cls(ok=False, value=value, verified=False, message=message)


def specs_by_key(specs: list[ControlSpec]) -> dict[str, ControlSpec]:
    return {s.key: s for s in specs}


def defaults_of(specs: list[ControlSpec]) -> dict[str, Any]:
    """所有非 ACTION 參數的預設值。"""
    return {s.key: s.default_value() for s in specs if s.type != ControlType.ACTION}


def validate_params(specs: list[ControlSpec], params: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    """以預設值為底，套上 params 並逐項驗證。strict=True 時遇到未知鍵會丟 ValueError，
    否則忽略未知鍵（讀舊版設定檔時用）。"""
    by_key = specs_by_key(specs)
    out = defaults_of(specs)
    for k, v in params.items():
        spec = by_key.get(k)
        if spec is None:
            if strict:
                raise ValueError(f"未知的參數：{k}")
            continue
        if spec.type == ControlType.ACTION:
            continue
        out[k] = spec.validate(v)
    return out
