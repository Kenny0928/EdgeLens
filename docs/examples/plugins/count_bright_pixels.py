"""計算亮點比例，數值與疊字分開輸出，不把文字寫進原始影像。"""
from __future__ import annotations

import cv2

from edgelens.core.results import Overlay
from edgelens.processing.pipeline import ProcessingStep, register_step


@register_step
class CountBrightPixels(ProcessingStep):
    kind = "custom.count_bright_pixels"
    category = "custom"
    display_name = "亮點比例（外掛範例）"  # 選單上顯示的名稱（沒有翻譯檔時用這個）
    description = "計算畫面裡很亮的像素佔多少比例，並把數字寫在畫面上。"  # 滑鼠停在步驟上時的說明

    def apply(self, image, ctx):
        # OpenCV 彩色順序是 BGR；轉成灰階才能用同一個亮度標準。
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        ratio = float((gray >= 200).mean())
        # Python float 可以存成 JSON；比例是 0–1，不是百分比。
        ctx.data[f"{self.kind}.ratio"] = ratio
        # 前面可能已經裁切或縮放，先換回原始畫面的座標。
        x, y = ctx.to_original(8, 20)
        ctx.overlays.append(Overlay(
            kind="text", data={"xy": [x, y], "text": f">=200: {ratio:.1%}"},
            color="#FFFF00",
        ))
        # 回傳副本；疊字交給顯示層處理，不影響後續分析。
        return image.copy()
