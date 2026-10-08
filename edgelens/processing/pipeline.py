"""影像處理流程：依序套用處理步驟（灰階、遮罩、二值化…）。

處理流程存成 JSON：{"version": 1, "steps": [{"kind": "color.grayscale", "enabled": true, "params": {}}]}

執行緒規則：Pipeline 只在 SourceWorker 的執行緒裡執行。GUI 要修改時，送一份新的
dict（to_dict 的格式）給 worker，由 worker 呼叫 Pipeline.from_dict 換掉整條流程，
GUI 不直接改正在執行的 Pipeline。

寫新的處理步驟：

    @register_step
    class Invert(ProcessingStep):
        kind = "color.invert"
        category = "color"
        display_name_key = "step.color.invert.name"

        @classmethod
        def param_specs(cls):
            return []

        def apply(self, image, ctx):
            return 255 - image
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from edgelens.core.controls import ControlSpec, specs_by_key, validate_params
from edgelens.core.frame import Frame
from edgelens.core.paths import atomic_write_text
from edgelens.core.registry import Registry
from edgelens.core.results import Overlay

PIPELINE_VERSION = 1

# 分類順序也是 GUI「新增步驟」選單的順序
CATEGORIES = ("geometry", "color", "filter", "threshold", "morph", "edge", "analyze", "motion", "edge_sim",
              "custom")


@dataclass
class StepContext:
    """處理時共用的資訊。

    frame     原始畫面（未處理）。
    overlays  步驟想畫在畫面上的圖形（例如色塊外框），座標為原始畫面座標。
    data      步驟輸出的數值，鍵建議用 "<kind>.<名稱>"，例如 "motion.diff.score"；
              這些數值會顯示在畫面資訊、寫進錄影 CSV。
    matrix    3×3 仿射矩陣：目前影像座標 → 原始畫面座標。裁切、縮放、旋轉、翻轉等
              幾何步驟呼叫 push_transform() 更新它，後面步驟產生的 overlay 就能用
              to_original() 換算回原始畫面座標。
    """

    frame: Frame
    overlays: list[Overlay] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    matrix: np.ndarray = field(default_factory=lambda: np.eye(3))

    def push_transform(self, new_to_previous: Any) -> None:
        """幾何步驟輸出新影像後呼叫。new_to_previous 是 2×3 或 3×3 矩陣，把「新影像座標」
        對應到「這一步的輸入影像座標」（例如裁切起點 (x, y)：[[1, 0, x], [0, 1, y]]）。"""
        m = np.asarray(new_to_previous, dtype=float)
        if m.shape == (2, 3):
            m = np.vstack([m, [0.0, 0.0, 1.0]])
        if m.shape != (3, 3):
            raise ValueError("push_transform 需要 2×3 或 3×3 的矩陣")
        self.matrix = self.matrix @ m

    def to_original(self, x: float, y: float) -> tuple[float, float]:
        """目前影像座標 → 原始畫面座標。"""
        p = self.matrix @ (float(x), float(y), 1.0)
        return float(p[0]), float(p[1])

    def to_original_points(self, points: Any) -> np.ndarray:
        """(N,2) 的點一次換算（輪廓等大量點用）。"""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        return pts @ self.matrix[:2, :2].T + self.matrix[:2, 2]

    @property
    def offset(self) -> tuple[float, float]:
        """目前影像原點在原始畫面的位置（只有平移與縮放時等於裁切位移）。"""
        return float(self.matrix[0, 2]), float(self.matrix[1, 2])

    @property
    def scale(self) -> tuple[float, float]:
        """目前影像每個像素在原始畫面上的長度 (sx, sy)。"""
        n = np.linalg.norm(self.matrix[:2, :2], axis=0)
        return float(n[0]), float(n[1])


class ProcessingStep(ABC):
    """一個處理步驟。

    kind              註冊名稱，格式 "<分類>.<名稱>"，例如 "color.grayscale"。
    category          CATEGORIES 之一，GUI 依此分組。
    display_name_key  i18n 鍵；說明文字用 description_key。
    stateful          True 表示會記住前一張畫面（例如影格差異），換來源時會 reset()。
    changes_size      True 表示輸出大小可能和輸入不同（裁切、縮放）；幾何步驟要呼叫
                      ctx.push_transform()，overlay 座標才換算得回原始畫面。
    accepts_gray      False 表示只能處理彩色（例如 HSV 遮罩）；遇到灰階輸入時，
                      Pipeline 會先轉成 BGR 再交給它。
    """

    kind: ClassVar[str] = ""
    category: ClassVar[str] = "custom"
    display_name_key: ClassVar[str] = ""
    description_key: ClassVar[str] = ""
    stateful: ClassVar[bool] = False
    changes_size: ClassVar[bool] = False
    accepts_gray: ClassVar[bool] = True

    def __init__(self, enabled: bool = True, **params: Any):
        self.enabled = enabled
        self.params: dict[str, Any] = validate_params(self.param_specs(), params)

    @classmethod
    def param_specs(cls) -> list[ControlSpec]:
        """這個步驟的參數。group 用 "image"；不常用的設 advanced=True。"""
        return []

    def set_param(self, key: str, value: Any) -> Any:
        """設定一個參數並回傳驗證後的值；未知參數丟 KeyError。"""
        spec = specs_by_key(self.param_specs()).get(key)
        if spec is None:
            raise KeyError(f"{self.kind} 沒有參數 {key}")
        self.params[key] = spec.validate(value)
        return self.params[key]

    @abstractmethod
    def apply(self, image: np.ndarray, ctx: StepContext) -> np.ndarray:
        """處理一張影像並回傳結果（uint8，HxW 或 HxWx3）。不可修改輸入的 image。"""

    def reset(self) -> None:
        """清除記住的狀態（stateful 步驟要覆寫）。"""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "enabled": self.enabled, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProcessingStep:
        step_cls = STEPS.get(d["kind"])
        return step_cls(enabled=bool(d.get("enabled", True)), **dict(d.get("params", {})))

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.kind} {self.params}>"


STEPS: Registry[ProcessingStep] = Registry("steps", ProcessingStep)
register_step = STEPS.register


@dataclass
class ProcessResult:
    """Pipeline.process 的結果。timings_ms 為 [(kind, 毫秒), ...]，依步驟順序。"""

    image: np.ndarray
    overlays: list[Overlay]
    data: dict[str, Any]
    timings_ms: list[tuple[str, float]]
    total_ms: float
    errors: list[tuple[str, str]] = field(default_factory=list)


class Pipeline:
    def __init__(self, steps: list[ProcessingStep] | None = None, name: str = ""):
        self.steps: list[ProcessingStep] = list(steps or [])
        self.name = name

    def process(self, frame: Frame) -> ProcessResult:
        """依序套用啟用的步驟。某一步出錯時跳過該步並記在 errors，不讓整條流程中斷。"""
        import cv2

        ctx = StepContext(frame=frame)
        img = frame.image
        timings: list[tuple[str, float]] = []
        errors: list[tuple[str, str]] = []
        t_all = time.perf_counter()
        for step in self.steps:
            if not step.enabled:
                continue
            t0 = time.perf_counter()
            try:
                src = img
                if img.ndim == 2 and not step.accepts_gray:
                    src = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                out = step.apply(src, ctx)
                if not isinstance(out, np.ndarray) or out.dtype != np.uint8 or out.ndim not in (2, 3):
                    raise TypeError(f"{step.kind} 回傳的不是 uint8 影像")
                img = out
            except Exception as e:  # noqa: BLE001 - 步驟出錯不能讓相機畫面停住
                errors.append((step.kind, f"{type(e).__name__}: {e}"))
            timings.append((step.kind, (time.perf_counter() - t0) * 1000.0))
        total = (time.perf_counter() - t_all) * 1000.0
        return ProcessResult(image=img, overlays=ctx.overlays, data=ctx.data, timings_ms=timings, total_ms=total,
                             errors=errors)

    def reset(self) -> None:
        for s in self.steps:
            s.reset()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"version": PIPELINE_VERSION, "steps": [s.to_dict() for s in self.steps]}
        if self.name:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Pipeline:
        version = int(d.get("version", 1))
        if version > PIPELINE_VERSION:
            raise ValueError(f"處理流程檔案版本 {version} 比程式新，請更新 EdgeLens")
        return cls([ProcessingStep.from_dict(s) for s in d.get("steps", [])], name=d.get("name", ""))

    def save(self, path: str | Path) -> None:
        atomic_write_text(path, json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Pipeline:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def __len__(self) -> int:
        return len(self.steps)
