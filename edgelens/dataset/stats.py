"""提供類別平衡圖、來源分布與品質直方圖所需的資料。"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class DatasetStats:
    class_counts: dict[str, int]
    max_min_ratio: float
    imbalanced: bool
    source_counts: dict[str, int]
    quality: dict[str, dict]
    total: int


def dataset_stats(records, class_ids=None, *, imbalance_threshold: float = 2.0, bins: int = 10) -> DatasetStats:
    if bins < 1 or not math.isfinite(imbalance_threshold) or imbalance_threshold < 1:
        raise ValueError("直方圖格數需大於零，不平衡門檻需至少為 1，請重新設定。")
    records = [r for r in records if not r.deleted]
    counts = Counter({cid: 0 for cid in (class_ids or [])})
    counts.update(r.class_id for r in records)
    values = list(counts.values())
    maximum, minimum = max(values, default=0), min(values, default=0)
    ratio = maximum / minimum if minimum else (math.inf if maximum else 0.0)
    sources = Counter(str(r.source.get("id") or r.source.get("kind") or "unknown") for r in records)
    quality = {}
    for key in ("sharpness", "brightness"):
        data = [float(r.quality[key]) for r in records
                if isinstance(r.quality.get(key), (float, int)) and math.isfinite(r.quality[key])]
        hist, edges = np.histogram(data, bins=bins)
        quality[key] = {"values": data, "count": len(data), "mean": float(np.mean(data)) if data else None,
                        "min": min(data, default=None), "max": max(data, default=None),
                        "histogram": hist.tolist(), "bin_edges": edges.tolist()}
    return DatasetStats(dict(counts), ratio, ratio > imbalance_threshold, dict(sources), quality, len(records))
