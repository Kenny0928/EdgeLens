"""相機來源的共同介面。

新增一種相機只要寫一個 FrameSource 子類別並用 @register_source 註冊，GUI 會自動
出現在「新增相機」清單，參數面板依 controls() 自動產生。

執行緒規則：一個 FrameSource 物件只會被一條執行緒（SourceWorker）使用，open、read、
set_control、close 都在同一條執行緒呼叫，所以實作不需要加鎖。需要背景讀取的來源
（例如 MJPEG 網路串流）可以自己開執行緒，但對外的方法仍只會從一條執行緒呼叫。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from edgelens.core.controls import ControlResult, ControlSpec, ControlType
from edgelens.core.frame import Frame
from edgelens.core.registry import Registry, module_available


class SourceError(Exception):
    """開啟或讀取失敗。message 直接顯示給使用者，要寫清楚原因與下一步。

    retryable=True 表示可以讓使用者按「重試」（例如 macOS 第一次開相機時跳出權限視窗）。
    hint 為額外提示（例如「請到 系統設定 → 隱私權 → 相機 允許終端機」）。
    """

    def __init__(self, message: str, *, retryable: bool = False, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.hint = hint


@dataclass
class SourceInfo:
    """找到的或使用者手動加入的一個相機。

    kind    註冊名稱（FrameSource.kind）。
    id      這台相機的唯一 ID，同一台相機每次都要一樣（每台相機的設定依此保存）。
            例如 "webcam:FaceTime HD Camera"、"esp32cam:192.168.4.1"、"k210:1a86:7523:ABC"。
    name    顯示名稱。
    params  傳給 open 的參數（必須能存成 JSON），重新開啟同一台相機時用。
    detail  補充說明（例如序列埠路徑、解析度）。
    """

    kind: str
    id: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "id": self.id, "name": self.name, "params": dict(self.params),
                "detail": self.detail}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SourceInfo:
        return cls(kind=d["kind"], id=d["id"], name=d.get("name", d["id"]), params=dict(d.get("params", {})),
                   detail=d.get("detail", ""))


@dataclass
class SourceCapabilities:
    """來源能做什麼，GUI 依此顯示或隱藏功能。

    device_inference  能在板上跑模型（K210）。
    resolutions       可選的解析度 [(寬, 高), ...]；空的表示不能選。
    actions           可執行的動作 key（對應 controls() 裡 type=ACTION 的項目）。
    provides_jpeg     Frame.jpeg 會有原始 JPEG。
    live_controls     串流中可以即時調整參數。
    max_fps           預期最高 FPS（0 表示未知）。
    """

    device_inference: bool = False
    resolutions: list[tuple[int, int]] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    provides_jpeg: bool = False
    live_controls: bool = True
    max_fps: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class FrameSource(ABC):
    """相機來源。子類別必須設定 kind、display_name_key，並實作 open／read／close。"""

    kind: ClassVar[str] = ""
    display_name_key: ClassVar[str] = ""   # i18n 鍵，例如 "source.webcam.name"
    description_key: ClassVar[str] = ""    # i18n 鍵，新增相機對話框的說明
    requires: ClassVar[tuple[str, ...]] = ()  # 需要的選配 Python 模組
    # discover() 是否安全（不會跳權限視窗、不會花超過 1 秒、不需要硬體）；
    # contract tests 只會呼叫安全的 discover()
    discover_is_safe: ClassVar[bool] = False
    # 新增相機時 GUI 是否讓使用者手動輸入參數（例如網址），參數由 param_specs() 宣告
    manual_add: ClassVar[bool] = False

    def __init__(self, info: SourceInfo):
        self.info = info
        self._seq = 0

    # ---- 類別層級：列舉與可用性 -------------------------------------------------

    @classmethod
    def available(cls) -> tuple[bool, str]:
        """這種來源在這台電腦能不能用；不能用時第二個值說明原因。"""
        missing = [m for m in cls.requires if not module_available(m)]
        if missing:
            return False, f"缺少套件：{', '.join(missing)}"
        return True, ""

    @classmethod
    def discover(cls, timeout: float = 2.0) -> list[SourceInfo]:
        """找出這台電腦目前接著的相機。找不到或不支援自動尋找時回傳空清單，不要丟例外。"""
        return []

    @classmethod
    def param_specs(cls) -> list[ControlSpec]:
        """手動加入時要填的參數（例如網址、序列埠、鮑率）；group 建議用 "source"。"""
        return []

    @classmethod
    def info_from_params(cls, params: dict[str, Any]) -> SourceInfo:
        """依手動填入的參數產生 SourceInfo。manual_add=True 的來源要覆寫，產生穩定的 id。"""
        raise NotImplementedError(f"{cls.__name__} 不支援手動加入")

    @classmethod
    def contract_fixture(cls, tmp_path: Path) -> SourceInfo | None:
        """回傳一個不需要硬體就能開啟的 SourceInfo，讓 contract tests 驗證 open/read/close。
        需要硬體的來源回傳 None（改由自己的單元測試與硬體實測驗證）。"""
        return None

    # ---- 物件層級：串流 ---------------------------------------------------------

    @property
    def source_id(self) -> str:
        return self.info.id

    @property
    @abstractmethod
    def is_open(self) -> bool: ...

    @abstractmethod
    def open(self) -> None:
        """開啟相機。失敗時丟 SourceError（訊息寫給學生看）。已開啟時不做事。"""

    @abstractmethod
    def read(self, timeout: float = 1.0) -> Frame | None:
        """回傳最新的一張畫面（舊的畫面要丟掉，不能越積越多）。

        timeout 秒內沒有新畫面時回傳 None；連線中斷且無法恢復時丟 SourceError。
        未開啟就呼叫時丟 SourceError。
        """

    @abstractmethod
    def close(self) -> None:
        """關閉並釋放資源。可以重複呼叫；關閉後可以再 open()。"""

    # ---- 參數 ------------------------------------------------------------------

    def controls(self) -> list[ControlSpec]:
        """目前可調整的參數。只列出這台相機實際可用的（例如 webcam 要先探測），
        所以要在 open() 之後呼叫；未開啟時可以回傳空清單。"""
        return []

    def get_control(self, key: str) -> Any:
        """參數目前的值；能讀回硬體時讀回，否則回傳最後設定的值。"""
        raise KeyError(key)

    def set_control(self, key: str, value: Any) -> ControlResult:
        """設定參數。值先用 ControlSpec.validate 檢查。能讀回時要讀回確認並設定 verified。"""
        return ControlResult.failed(f"不支援參數 {key}")

    def run_action(self, key: str, **kwargs: Any) -> ControlResult:
        """執行 ACTION 參數（例如「上傳模型」）。耗時的動作可以阻塞，worker 會在背景執行緒呼叫。"""
        return ControlResult.failed(f"不支援動作 {key}")

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(actions=[c.key for c in self.controls() if c.type == ControlType.ACTION])

    # ---- 共用工具 --------------------------------------------------------------

    def _next_seq(self) -> int:
        """產生遞增的畫面序號（子類別建立 Frame 時用）。"""
        self._seq += 1
        return self._seq

    def __enter__(self) -> FrameSource:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.info.id!r} open={self.is_open}>"


SOURCES: Registry[FrameSource] = Registry("sources", FrameSource)
register_source = SOURCES.register


def create_source(info: SourceInfo) -> FrameSource:
    """依 SourceInfo.kind 建立對應的來源物件（尚未開啟）。"""
    return SOURCES.get(info.kind)(info)
