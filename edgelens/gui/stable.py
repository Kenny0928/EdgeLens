"""不會讓版面跳動的文字標籤。

面板裡有些文字每秒更新好幾次（推論耗時、偵測到的物體、拍照品質、處理時間）。一般的
QLabel 會依內容長短改變高度（換行數變了、提示一下出現一下消失），下面的按鈕就被推上
推下，畫面一直跳。StableLabel 預先保留固定行數的高度；內容超過時只會長高、不會縮回，
所以外框保持不動。換模型、換相機這類「內容類型改變」的時機可以呼叫 reset_height() 重算。
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QLabel, QWidget


class StableLabel(QLabel):
    def __init__(self, lines: int = 1, text: str = "", parent: QWidget | None = None, *, wrap: bool = True):
        super().__init__(text, parent)
        self._lines = max(1, int(lines))
        self._grown = 0
        self.setWordWrap(wrap)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._apply()

    def set_lines(self, lines: int) -> None:
        """改變保留的行數（例如偵測模型要列 6 行、分類模型只要 1 行）。"""
        self._lines = max(1, int(lines))
        self._grown = 0
        self._apply()

    def reset_height(self) -> None:
        """內容類型改變時（換模型、換相機）把「只長不縮」的高度歸零，回到保留的行數。"""
        self._grown = 0
        self._apply()

    def setText(self, text: str) -> None:  # noqa: N802 - 覆寫 Qt 方法
        if text == self.text():
            return
        super().setText(text)
        if self.wordWrap() and self.width() > 0:
            need = self.heightForWidth(self.width())
        else:
            need = self.sizeHint().height()
        if need > max(self._grown, self.minimumHeight()):
            self._grown = need
            self._apply()

    def _reserved(self) -> int:
        m = self.contentsMargins()
        return self.fontMetrics().lineSpacing() * self._lines + m.top() + m.bottom() + 2

    def _apply(self) -> None:
        self.setMinimumHeight(max(self._reserved(), self._grown))

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._apply()
        super().changeEvent(event)
