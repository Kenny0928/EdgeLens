"""內建與專案處理流程；單一損壞檔不應讓整個預設清單消失。"""
from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from edgelens.core.paths import atomic_write_text, resource_path, safe_filename
from edgelens.i18n import current_language
from edgelens.processing.pipeline import Pipeline

log = logging.getLogger(__name__)


def _register_steps():
    # 僅載入影像步驟，不因讀預設而探索相機或推論後端。
    for category in ("geometry", "color", "filter", "threshold", "morph", "edge", "analyze", "motion", "edge_sim"):
        importlib.import_module(f"edgelens.processing.steps.{category}")


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
            raise ValueError("最外層須有 steps 清單")
        if any(not isinstance(s, dict) or not isinstance(s.get("kind"), str)
               or not isinstance(s.get("params", {}), dict) for s in data["steps"]):
            raise ValueError("步驟須包含 kind 與參數物件")
        for key in ("title", "description"):
            value = data.get(key, {})
            if not isinstance(value, (str, dict)) or (isinstance(value, dict)
                    and any(not isinstance(v, str) for v in value.values())):
                raise ValueError("名稱與說明須為文字或翻譯物件")
        return data
    except (OSError, ValueError) as exc:
        raise ValueError(f"無法讀取處理流程「{path.name}」，請確認檔案是完整的流程 JSON。") from exc


def builtin_presets() -> dict[str, dict]:
    """鍵為含 .json 的檔名，回傳新物件以免編輯時改到內建原稿。"""
    return {path.name: _read(path) for path in sorted(resource_path("processing", "presets").glob("*.json"))}


def _localized(values, lang, fallback=""):
    if isinstance(values, str):
        return values or fallback
    return values.get(lang) or values.get("en") or values.get("zh_TW") or fallback


@dataclass(frozen=True)
class PresetInfo:
    name: str
    path: Path
    builtin: bool
    titles: dict[str, str] | str = field(default_factory=dict)
    descriptions: dict[str, str] | str = field(default_factory=dict)

    def title(self, lang: str | None = None) -> str:
        return _localized(self.titles, lang or current_language(), self.path.stem)

    def description(self, lang: str | None = None) -> str:
        return _localized(self.descriptions, lang or current_language())


def list_presets(user_dir: Path | None = None) -> list[PresetInfo]:
    found = []
    folders = [(resource_path("processing", "presets"), True)]
    if user_dir is not None:
        folders.append((Path(user_dir), False))
    for folder, builtin in folders:
        for path in sorted(folder.glob("*.json")):
            try:
                data = _read(path)
            except ValueError as exc:
                log.warning("略過損壞的預設：%s", exc)
                continue
            found.append(PresetInfo(path.name, path, builtin, data.get("title", {}), data.get("description", {})))
    return found


def load_preset(name_or_path: str | Path) -> Pipeline:
    """使用者預設傳 path；內建可傳含或不含副檔名的名稱。"""
    path = Path(name_or_path)
    if not path.is_file() and isinstance(name_or_path, str) and path.name == name_or_path:
        path = resource_path("processing", "presets", path.name if path.suffix else f"{path.name}.json")
    data = _read(path)
    _register_steps()
    try:
        return Pipeline.from_dict(data)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"無法套用處理流程「{path.name}」，請檢查步驟名稱、參數或更新程式。") from exc


def save_user_preset(pipeline: Pipeline, folder: Path, name: str, title: str | dict[str, str]) -> Path:
    """名稱只作為檔名，避免含斜線的輸入寫到專案外。"""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("請先輸入處理流程的檔名。")
    if not isinstance(title, (str, dict)) or (isinstance(title, dict)
            and any(not isinstance(k, str) or not isinstance(v, str) for k, v in title.items())):
        raise ValueError("處理流程名稱要是文字，或包含各語言文字的物件。")
    filename = safe_filename(name)
    if filename.lower().endswith(".json"):
        filename = filename[:-5] + ".json"
    else:
        filename += ".json"
    data = pipeline.to_dict()
    data["title"] = {"zh_TW": title, "en": title} if isinstance(title, str) else dict(title)
    data["description"] = {"zh_TW": "", "en": ""}
    return atomic_write_text(Path(folder) / filename, json.dumps(data, ensure_ascii=False, indent=2))
