"""最小外掛：用負片認識像素，不需要相機或翻譯檔。"""
from __future__ import annotations

from edgelens.processing.pipeline import ProcessingStep, register_step


@register_step  # 匯入檔案時，讓新增步驟選單認得這個類別。
class InvertColors(ProcessingStep):
    kind = "custom.invert_colors"  # 自訂前綴可避免撞到內建步驟。
    category = "custom"
    display_name = "負片（外掛範例）"  # 選單上顯示的名稱（沒有翻譯檔時用這個）
    description = "把每個像素反過來：黑變白、白變黑。最簡單的外掛範例。"  # 滑鼠停在步驟上時的說明
    # 沿用基底的 param_specs()：空清單表示沒有可調參數。

    def apply(self, image, ctx):
        # 減法建立新陣列，保留 uint8，也不會改到其他步驟的輸入。
        return 255 - image
