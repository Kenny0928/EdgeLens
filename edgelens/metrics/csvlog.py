"""CSV 使用 BOM 供 Excel 讀中文。平常只在檔尾附加新列；出現新欄位時才重寫整個檔案。"""
from __future__ import annotations

import csv
import io
import math
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from edgelens.core.paths import atomic_write_text


def write_summary_csv(path: str | Path, rows: Iterable[Mapping[str, Any]],
                      fieldnames: Iterable[str] | None = None) -> Path:
    rows = list(rows)
    fields = list(dict.fromkeys(fieldnames or []))
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return atomic_write_text(path, buf.getvalue(), encoding="utf-8-sig")


class CsvLogger:
    """每次 add 檢查秒數週期；需要閒置時落盤可由呼叫端呼叫 flush。

    錄影時每張畫面一列、可能錄好幾分鐘，所以 flush 只附加新列（不重寫舊列）；只有新列出現
    新欄位時才讀回舊列、用新的標題列重寫一次（少見）。單一 logger 可跨執行緒，
    同一個 CSV 不應同時交給多個 logger。
    """

    def __init__(self, path: str | Path, fieldnames: Iterable[str] | None = None,
                 *, flush_interval: float = 1.0):
        if not math.isfinite(flush_interval) or flush_interval < 0:
            raise ValueError("寫入週期必須是非負秒數，請重新設定。")
        self.path = Path(path)
        self.fieldnames = list(dict.fromkeys(fieldnames or []))
        self.flush_interval = flush_interval
        self.rows = 0
        self._pending: list[dict] = []
        self._closed = False
        self._lock = threading.RLock()
        self._last_flush = time.monotonic()
        write_summary_csv(self.path, [], self.fieldnames)

    def add(self, row: Mapping[str, Any]) -> None:
        with self._lock:
            if self._closed:
                raise ValueError("CSV 已關閉，請建立新的紀錄檔。")
            self._pending.append(dict(row))
            self.rows += 1
            if time.monotonic() - self._last_flush >= self.flush_interval:
                self.flush()

    def flush(self) -> None:
        with self._lock:
            if not self._pending:
                return
            new_fields = [k for row in self._pending for k in row if k not in self.fieldnames]
            if new_fields:
                with self.path.open(encoding="utf-8-sig", newline="") as f:
                    old = list(csv.DictReader(f))
                fields = list(dict.fromkeys([*self.fieldnames, *new_fields]))
                write_summary_csv(self.path, [*old, *self._pending], fields)
                self.fieldnames = fields
            else:
                with self.path.open("a", encoding="utf-8", newline="") as f:
                    csv.DictWriter(f, fieldnames=self.fieldnames).writerows(self._pending)
            self._pending.clear()
            self._last_flush = time.monotonic()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self.flush()
                self._closed = True

    def __enter__(self) -> CsvLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
