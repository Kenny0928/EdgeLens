"""逐一執行參數格點，取消時保留已完成的 CSV。"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

from edgelens.metrics.csvlog import CsvLogger


@dataclass
class SweepReport:
    rows: list[dict]
    path: Path
    total: int
    cancelled: bool = False


def run_sweep(grid: dict[str, list], run, csv_path: str | Path, *, progress=None, cancel=None) -> SweepReport:
    """參數欄位加 param. 前綴，避免和結果的同名指標互相覆蓋。"""
    if any(not isinstance(values, (list, tuple)) or not values for values in grid.values()):
        raise ValueError("每個參數至少需要一個候選值，請補齊參數格點。")
    keys = list(grid)
    total = math.prod(len(grid[k]) for k in keys)
    report = SweepReport([], Path(csv_path), total)
    if progress:
        progress(0, total)
    with CsvLogger(csv_path, [f"param.{k}" for k in keys], flush_interval=0) as logger:
        for values in itertools.product(*(grid[k] for k in keys)):
            if cancel is not None and (cancel() if callable(cancel) else cancel.is_set()):
                report.cancelled = True
                break
            params = dict(zip(keys, values, strict=True))
            metrics = run(dict(params))
            if not isinstance(metrics, dict) or any(not isinstance(k, str) or k.startswith("param.")
                    or not isinstance(v, (int, float)) or not math.isfinite(v) for k, v in metrics.items()):
                raise ValueError("掃描結果需為有限數值，名稱不可使用 param. 前綴，請檢查 run 回傳值。")
            row = {**{f"param.{k}": v for k, v in params.items()}, **metrics}
            logger.add(row)
            report.rows.append(row)
            if progress:
                progress(len(report.rows), total)
    return report
