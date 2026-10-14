"""元件註冊表：相機來源、處理步驟、推論後端、解碼器都用 decorator 註冊。

    from edgelens.sources.base import register_source, FrameSource

    @register_source
    class MySource(FrameSource):
        kind = "my_source"
        ...

discover() 會匯入內建套件裡的所有模組，讓 decorator 執行；load_plugins() 匯入
工作區 plugins/ 資料夾裡學生自己寫的 .py。contract tests 會檢查每一個註冊的元件。

規則：模組在匯入時不可以 import 選配套件（onnxruntime、tflite、kflash…），要在
用到時才匯入，否則沒裝選配套件的電腦連註冊都會失敗。
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import pkgutil
import sys
import traceback
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


class Registry(Generic[T]):
    """名稱 → 類別 的對照表。類別必須有 class 屬性 kind（英文、小寫、可含 . 和 _）。"""

    def __init__(self, name: str, base: type[T]):
        self.name = name
        self.base = base
        self._items: dict[str, type[T]] = {}
        self._origin: dict[str, str] = {}

    def register(self, cls: type[T]) -> type[T]:
        if not isinstance(cls, type) or not issubclass(cls, self.base):
            raise TypeError(f"{cls!r} 必須繼承 {self.base.__name__}")
        kind = getattr(cls, "kind", None)
        if not kind or not isinstance(kind, str):
            raise TypeError(f"{cls.__name__} 沒有設定 kind")
        if not all(c.isalnum() or c in "._" for c in kind) or kind != kind.lower():
            raise ValueError(f"{cls.__name__}.kind={kind!r} 只能用小寫英數字、. 和 _")
        existing = self._items.get(kind)
        if existing is not None and existing.__qualname__ != cls.__qualname__:
            raise ValueError(f"{self.name} 已經有 kind={kind!r}（{existing.__module__}.{existing.__qualname__}）")
        self._items[kind] = cls
        self._origin[kind] = cls.__module__
        return cls

    __call__ = register

    def unregister(self, kind: str) -> None:
        self._items.pop(kind, None)
        self._origin.pop(kind, None)

    def get(self, kind: str) -> type[T]:
        try:
            return self._items[kind]
        except KeyError:
            raise KeyError(f"{self.name} 沒有 {kind!r}；可用：{', '.join(sorted(self._items))}") from None

    def find(self, kind: str) -> type[T] | None:
        return self._items.get(kind)

    def kinds(self) -> list[str]:
        return sorted(self._items)

    def items(self) -> list[tuple[str, type[T]]]:
        return sorted(self._items.items())

    def origin(self, kind: str) -> str:
        """類別定義所在的模組名稱（外掛為 edgelens_plugins.xxx）。"""
        return self._origin.get(kind, "")

    def __contains__(self, kind: object) -> bool:
        return kind in self._items

    def __iter__(self) -> Iterator[type[T]]:
        return iter(cls for _, cls in self.items())

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class ImportProblem:
    module: str
    error: str
    detail: str


# 每個模組匯入失敗的紀錄，「診斷」面板會顯示
import_problems: list[ImportProblem] = []
_discovered = False

BUILTIN_PACKAGES = (
    "edgelens.sources",
    "edgelens.processing.steps",
    "edgelens.inference.backends",
    "edgelens.inference.decoders",
)


def _import_all(package: str) -> None:
    try:
        pkg = importlib.import_module(package)
    except Exception as e:  # noqa: BLE001 - 要記下任何錯誤
        import_problems.append(ImportProblem(package, repr(e), traceback.format_exc()))
        return
    for info in pkgutil.walk_packages(pkg.__path__, prefix=package + "."):
        name = info.name
        # board/ 底下是要上傳到開發板執行的程式，不是電腦端模組
        if ".board" in name or name.rsplit(".", 1)[-1].startswith("_"):
            continue
        try:
            importlib.import_module(name)
        except Exception as e:  # noqa: BLE001
            log.warning("匯入 %s 失敗：%s", name, e)
            import_problems.append(ImportProblem(name, repr(e), traceback.format_exc()))


def discover(force: bool = False) -> None:
    """匯入所有內建元件模組，讓它們完成註冊。可重複呼叫。"""
    global _discovered
    if _discovered and not force:
        return
    for package in BUILTIN_PACKAGES:
        _import_all(package)
    _discovered = True


def plugin_files(folder: str | Path) -> list[Path]:
    """folder 第一層的外掛檔（*.py，不含 _ 開頭），依檔名排序。載入與信任檢查共用，確保兩邊看到同一批檔案。"""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return [p for p in sorted(folder.glob("*.py")) if not p.name.startswith("_") and p.is_file()]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def plugin_hashes(folder: str | Path) -> dict[str, str]:
    """{檔名: sha256}。讀不到的檔案略過（之後載入時自然會失敗並記錄）。"""
    out: dict[str, str] = {}
    for p in plugin_files(folder):
        try:
            out[p.name] = file_sha256(p)
        except OSError:
            continue
    return out


def load_plugins(folder: str | Path, only: Iterable[str] | None = None) -> list[str]:
    """匯入 folder 裡的每個 .py（不含 _ 開頭的檔案），回傳成功載入的模組名稱。

    外掛就是一般的 Python 檔，用同樣的 decorator 註冊元件，例如學生自己寫的處理步驟。
    同名的外掛重新載入時會先移除舊模組。失敗的外掛記在 import_problems，不會讓程式當掉。
    only 不是 None 時只載入名單內的檔名（信任檢查用：沒被核准的檔案不能執行）。
    """
    folder = Path(folder)
    loaded: list[str] = []
    allowed = None if only is None else set(only)
    for path in plugin_files(folder):
        if allowed is not None and path.name not in allowed:
            continue
        mod_name = f"edgelens_plugins.{path.stem}"
        sys.modules.pop(mod_name, None)
        try:
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"無法載入 {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
            loaded.append(mod_name)
        except Exception as e:  # noqa: BLE001
            sys.modules.pop(mod_name, None)
            log.warning("外掛 %s 載入失敗：%s", path.name, e)
            import_problems.append(ImportProblem(str(path), repr(e), traceback.format_exc()))
    return loaded


def all_registries() -> dict[str, Registry[Any]]:
    """四個內建註冊表。延遲匯入以避免循環相依。"""
    from edgelens.inference.base import BACKENDS, DECODERS
    from edgelens.processing.pipeline import STEPS
    from edgelens.sources.base import SOURCES

    return {"sources": SOURCES, "steps": STEPS, "backends": BACKENDS, "decoders": DECODERS}


def lazy_import(module: str) -> Callable[[], Any]:
    """回傳一個函式，第一次呼叫時才匯入 module（選配套件用）。"""
    cache: dict[str, Any] = {}

    def get() -> Any:
        if "m" not in cache:
            cache["m"] = importlib.import_module(module)
        return cache["m"]

    return get


def module_available(module: str) -> bool:
    """不實際匯入，只檢查 module 是否已安裝。"""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False
