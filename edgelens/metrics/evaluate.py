"""呼叫外部 predict 評估分類，不相依於模型執行器或硬體。"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edgelens.dataset.writer import dataset_path, read_image
from edgelens.metrics.csvlog import write_summary_csv
from edgelens.metrics.rolling import RollingStats


@dataclass
class EvalReport:
    class_ids: list[str]
    prediction_labels: list[str]
    accuracy: float
    confusion_matrix: list[list[int]]
    per_class: dict[str, dict[str, float | int]]
    latency: dict | None
    wrong_files: list[str]
    predictions: list[dict]
    total: int
    requested: int
    cancelled: bool = False

    def to_csv(self, path: str | Path) -> Path:
        rows = [{"section": "summary", "metric": "accuracy", "value": self.accuracy},
                {"section": "summary", "metric": "total", "value": self.total},
                {"section": "summary", "metric": "requested", "value": self.requested},
                {"section": "summary", "metric": "cancelled", "value": self.cancelled}]
        rows.extend({"section": "latency_ms", "metric": k, "value": v} for k, v in (self.latency or {}).items())
        rows.extend({"section": "class", "class_id": cid, **metrics} for cid, metrics in self.per_class.items())
        rows.extend({"section": "confusion", "actual": actual, "predicted": predicted, "value": value}
                    for actual, counts in zip(self.class_ids, self.confusion_matrix, strict=True)
                    for predicted, value in zip(self.prediction_labels, counts, strict=True))
        rows.extend({"section": "prediction", **p} for p in self.predictions)
        return write_summary_csv(path, rows)


def evaluate(records, predict, class_ids, progress=None, cancel=None, *,
             label_mapping: dict[str, str] | None = None, dataset_dir: str | Path | None = None) -> EvalReport:
    """mapping 接受「顯示名稱 → ID」或「ID → 顯示名稱」。

    writer 回傳的紀錄自帶位置；自行建立紀錄時需提供 dataset_dir。
    未辨識／未知標籤另列 __unknown__ 欄，避免混淆矩陣漏算錯誤。
    延遲為 predict 的實際執行毫秒；取消回傳已完成部分。
    """
    records = [r for r in records if not r.deleted]
    classes = list(class_ids)
    if not classes or len(set(classes)) != len(classes) or "__unknown__" in classes:
        raise ValueError("類別清單不可空白、重複或使用 __unknown__，請重新設定。")
    if any(r.class_id not in classes for r in records):
        raise ValueError("照片類別不在評估清單內，請補齊類別後重試。")
    mapping = {cid: cid for cid in classes}
    for key, value in (label_mapping or {}).items():
        alias, target = (key, value) if value in classes else (value, key)
        if target not in classes or (alias in mapping and mapping[alias] != target):
            raise ValueError("類別名稱對照有衝突，請使用唯一名稱對應類別 ID。")
        mapping[alias] = target
    matrix = np.zeros((len(classes), len(classes) + 1), dtype=int)
    latency = RollingStats(max(1, len(records)))
    predictions, wrong = [], []
    cancelled = False
    if progress:
        progress(0, len(records))
    for record in records:
        if cancel is not None and (cancel() if callable(cancel) else cancel.is_set()):
            cancelled = True
            break
        path = dataset_path(Path(dataset_dir), record.file) if dataset_dir is not None else record.path
        image = read_image(path)
        start = time.perf_counter()
        result = predict(image)
        elapsed = (time.perf_counter() - start) * 1000
        if result.kind != "classify":
            raise ValueError("評估需要分類模型結果，請選擇分類模型後重試。")
        latency.add(elapsed)
        top = result.best
        predicted = None
        if top is not None:
            predicted = mapping.get(top.label)
            if not top.label and 0 <= top.class_id < len(classes):
                predicted = classes[top.class_id]
        actual_idx = classes.index(record.class_id)
        predicted_idx = classes.index(predicted) if predicted in classes else len(classes)
        matrix[actual_idx, predicted_idx] += 1
        correct = predicted == record.class_id
        if not correct:
            wrong.append(record.file)
        predictions.append({"file": record.file, "actual": record.class_id,
                            "predicted": predicted or "__unknown__", "correct": correct, "latency_ms": elapsed})
        if progress:
            progress(len(predictions), len(records))
    per_class = {}
    for i, cid in enumerate(classes):
        tp, support, positive = int(matrix[i, i]), int(matrix[i].sum()), int(matrix[:, i].sum())
        precision, recall = tp / positive if positive else 0.0, tp / support if support else 0.0
        per_class[cid] = {"precision": precision, "recall": recall,
                          "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                          "support": support}
    prediction_labels = list(classes)
    if matrix[:, -1].sum():
        prediction_labels.append("__unknown__")
    else:
        matrix = matrix[:, :-1]
    n = len(predictions)
    return EvalReport(classes, prediction_labels, (n - len(wrong)) / n if n else 0.0,
                      matrix.tolist(), per_class, latency.summary(), wrong, predictions, n, len(records), cancelled)
