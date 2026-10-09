"""馬賽克：先縮小再用最近鄰放大，示範整數參數。"""
from __future__ import annotations

import cv2

from edgelens.core.controls import ControlSpec, ControlType
from edgelens.processing.pipeline import ProcessingStep, register_step


@register_step
class Pixelate(ProcessingStep):
    kind = "custom.pixelate"
    category = "custom"
    display_name = "馬賽克（外掛範例）"  # 選單上顯示的名稱（沒有翻譯檔時用這個）
    description = "把畫面變成一格一格的色塊；區塊越大越模糊。"  # 滑鼠停在步驟上時的說明

    @classmethod
    def param_specs(cls):
        # label 是沒有翻譯檔的備援文字；範圍避免除以零或過大參數。
        return [ControlSpec(
            key="block_size", type=ControlType.INT, label="區塊大小",
            default=8, min=1, max=64, step=1, group="image",
        )]

    def apply(self, image, ctx):
        h, w = image.shape[:2]
        block = self.params["block_size"]
        # 小影像也至少縮成 1×1；最後恢復尺寸，座標轉換不變。
        small = cv2.resize(image, (max(1, w // block), max(1, h // block)),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
