"""先合併跨類別的分組關聯，再盡量符合各類別的切分比例。"""
from __future__ import annotations

import json
import math
import random
import warnings as warnings_module
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from edgelens.core.paths import atomic_write_text
from edgelens.core.project import Project
from edgelens.dataset.writer import DatasetWriter, SampleRecord
from edgelens.i18n import tr

PARTS = ("train", "val", "test")


@dataclass
class SplitResult:
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    seed: int = 42
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"version": 1, **{p: getattr(self, p) for p in PARTS}, "seed": self.seed,
                "ratios": list(self.ratios), "warnings": list(self.warnings)}

    def save(self, project_or_path: Project | str | Path) -> Path:
        path = _path(project_or_path)
        return atomic_write_text(path, json.dumps(self.to_dict(), ensure_ascii=False, indent=2))


def _path(project_or_path):
    return (project_or_path.dataset_dir / "splits.json" if isinstance(project_or_path, Project)
            else Path(project_or_path))


def _ratios(ratios):
    values = tuple(float(v) for v in ratios)
    if len(values) != 3 or any(not math.isfinite(v) or v < 0 for v in values) or not math.isclose(sum(values), 1):
        raise ValueError("切分比例需要三個非負數，總和為 1，請重新設定。")
    return values


def load_splits(project_or_path: Project | str | Path) -> SplitResult:
    data = json.loads(_path(project_or_path).read_text(encoding="utf-8"))
    ratios = _ratios(data["ratios"])
    parts = {p: data[p] for p in PARTS}
    if any(not isinstance(files, list) or any(not isinstance(f, str) for f in files) for files in parts.values()):
        raise ValueError("切分檔內容不正確，請重新切分。")
    files = [f for values in parts.values() for f in values]
    if len(files) != len(set(files)):
        raise ValueError("切分檔有重複照片，請重新切分。")
    return SplitResult(**parts, seed=data["seed"], ratios=ratios, warnings=data.get("warnings", []))


def split_records(records, ratios=(0.7, 0.15, 0.15), *, seed: int = 42) -> SplitResult:
    ratios = _ratios(ratios)
    records = sorted((r for r in records if not r.deleted), key=lambda r: r.file)
    if len({r.file for r in records}) != len(records):
        raise ValueError("照片清單有重複路徑，請先整理資料集。")
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, record in enumerate(records):
        for kind in ("group_id", "burst_id"):
            value = getattr(record, kind)
            if value:
                key = (kind, value)
                if key in seen:
                    parent[root(i)] = root(seen[key])
                else:
                    seen[key] = i
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[root(i)].append(record)
    totals = Counter(r.class_id for r in records)
    target = {}
    active = [i for i, ratio in enumerate(ratios) if ratio > 0]
    for cid, count in totals.items():
        # 最大餘數法；1–2 張時優先留給比例較大的份。
        sizes = [math.floor(count * ratio) for ratio in ratios]
        order = sorted(active, key=lambda i: (-(count * ratios[i] - sizes[i]), -ratios[i], i))
        for i in order[:count - sum(sizes)]:
            sizes[i] += 1
        target[cid] = sizes
    result = SplitResult(seed=seed, ratios=ratios)
    grouped = list(groups.values())
    random.Random(seed).shuffle(grouped)
    grouped.sort(key=lambda group: -len(group))
    counts = {cid: [0, 0, 0] for cid in totals}
    for group in grouped:
        composition = Counter(r.class_id for r in group)

        def cost(part, composition=composition):
            return sum(((counts[cid][part] + n - target[cid][part]) ** 2
                        - (counts[cid][part] - target[cid][part]) ** 2) / totals[cid]
                       for cid, n in composition.items())

        part = min(active, key=lambda i: (cost(i), -ratios[i], i))
        getattr(result, PARTS[part]).extend(r.file for r in group)
        for cid, n in composition.items():
            counts[cid][part] += n
    for cid, count in sorted(totals.items()):
        if count < len(active):
            result.warnings.append(tr("dataset.split.small", class_id=cid, count=count))
        elif any(counts[cid][i] == 0 for i in active) or counts[cid] != target[cid]:
            result.warnings.append(tr("dataset.split.grouped", class_id=cid))
    for part in PARTS:
        getattr(result, part).sort()
    for message in result.warnings:
        warnings_module.warn(message, UserWarning, stacklevel=2)
    return result


def split_dataset(project: Project, ratios=(0.7, 0.15, 0.15), *, seed: int = 42,
                  records: list[SampleRecord] | None = None) -> SplitResult:
    result = split_records(DatasetWriter(project).list_samples() if records is None else records,
                           ratios, seed=seed)
    result.save(project)
    return result
