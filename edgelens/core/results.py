"""推論結果與畫面疊加圖形。

座標一律是「原始畫面」的像素座標（左上角為原點），不是模型輸入的座標；
letterbox 反推由推論 runner 負責，GUI 和資料集只看到原始座標。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Task = Literal["detect", "classify", "raw"]
Where = Literal["host", "device"]


@dataclass
class Detection:
    """一個偵測框。box 為 (x1, y1, x2, y2)，像素座標。"""

    box: tuple[float, float, float, float]
    score: float
    class_id: int
    label: str = ""

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def center(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


@dataclass
class Classification:
    """分類結果中的一個類別。"""

    class_id: int
    score: float
    label: str = ""


@dataclass
class InferenceResult:
    """一次推論的結果。

    kind 為 detect 時看 detections，classify 時看top_k（依分數由高到低）。
    frame_seq 對應 Frame.seq，GUI 用來顯示「結果是幾張以前的」。
    where 標明是電腦（host）還是開發板（device）算的。
    timings_ms 常用鍵：preprocess、infer、postprocess、total。
    image_size 是座標所對應的畫面大小 (寬, 高)。
    """

    kind: Task
    model_name: str
    frame_seq: int
    where: Where = "host"
    detections: list[Detection] = field(default_factory=list)
    top_k: list[Classification] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    image_size: tuple[int, int] = (0, 0)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def best(self) -> Detection | Classification | None:
        """分數最高的結果；沒有結果時為 None。"""
        items: list[Any] = self.detections if self.kind == "detect" else self.top_k
        return max(items, key=lambda x: x.score) if items else None

    @property
    def best_score(self) -> float:
        b = self.best
        return float(b.score) if b is not None else 0.0

    @property
    def has_hit(self) -> bool:
        return bool(self.detections) if self.kind == "detect" else bool(self.top_k)

    def to_dict(self) -> dict[str, Any]:
        """轉成可存成 JSON 的 dict（存進 metadata.jsonl、錄影 CSV 用）。"""
        d = asdict(self)
        d["image_size"] = list(self.image_size)
        for det in d["detections"]:
            det["box"] = [round(float(v), 2) for v in det["box"]]
            det["score"] = round(float(det["score"]), 4)
        for c in d["top_k"]:
            c["score"] = round(float(c["score"]), 4)
        d["timings_ms"] = {k: round(float(v), 3) for k, v in self.timings_ms.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InferenceResult:
        return cls(
            kind=d["kind"],
            model_name=d.get("model_name", ""),
            frame_seq=int(d.get("frame_seq", -1)),
            where=d.get("where", "host"),
            detections=[
                Detection(box=tuple(x["box"]), score=x["score"], class_id=x["class_id"], label=x.get("label", ""))
                for x in d.get("detections", [])
            ],
            top_k=[Classification(**c) for c in d.get("top_k", [])],
            timings_ms=dict(d.get("timings_ms", {})),
            image_size=tuple(d.get("image_size", (0, 0))),
            extra=dict(d.get("extra", {})),
        )


OverlayKind = Literal["box", "polygon", "polyline", "point", "text", "mask"]


@dataclass
class Overlay:
    """處理步驟想讓 GUI 畫在畫面上的圖形（例如色塊外框、輪廓）。

    data 依 kind 而定（都是原始畫面的像素座標）：
      box      {"box": [x1, y1, x2, y2]}
      polygon  {"points": [[x, y], ...]}（封閉）
      polyline {"points": [[x, y], ...]}
      point    {"xy": [x, y], "radius": 4}
      text     {"xy": [x, y], "text": "..."}
      mask     {"mask": HxW uint8 ndarray}（非 0 處以 color 半透明疊上）
    color 為 "#RRGGBB"。label 會顯示在圖形旁邊。
    """

    kind: OverlayKind
    data: dict[str, Any]
    color: str = "#00FF00"
    label: str = ""
    thickness: int = 2
