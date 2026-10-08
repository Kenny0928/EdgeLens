"""工作區與專案。

一個工作區（預設「文件／EdgeLens」）可以有很多專案，每個專案是一個獨立的主題
（例如「筊杯辨識」「剪刀石頭布」「垃圾分類」），各自有類別、資料集、模型與處理流程。
老師可以為每個班級或每組學生開一個專案，彼此不會混在一起。

工作區結構（給 WP7 資料集、WP8 GUI 遵守）：

    <workspace>/
      plugins/                   學生自己寫的處理步驟、解碼器（.py）
      projects/<project_id>/
        project.json             名稱、說明、類別
        dataset/<class_id>/*.jpg 樣本照片（資料夾名稱是英文 ID，中文只是顯示名稱）
        dataset/metadata.jsonl   每張照片一行 JSON
        models/                  模型與 .card.json
        pipelines/*.json         影像處理流程預設組合
        references/              構圖參考照
        recordings/              錄影與每張畫面的結果 CSV
        logs/                    效能測試、評估結果
        exports/                 匯出的 ZIP
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from edgelens.core.paths import (
    atomic_write_text,
    default_workspace,
    ensure_dir,
    is_valid_id,
    resource_path,
    slugify,
    unique_path,
)

PROJECT_FILE = "project.json"
PROJECT_VERSION = 1
SUBDIRS = ("dataset", "models", "pipelines", "references", "recordings", "logs", "exports")

# 類別預設顏色（色盲友善的 Okabe–Ito 色盤，再加幾個）
DEFAULT_COLORS = (
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
    "#D55E00", "#CC79A7", "#999999", "#000000", "#882255",
)


@dataclass
class ClassSpec:
    """一個樣本類別。

    id     英文 ID（小寫英數字與底線），也是資料夾名稱，建立後不建議改。
    names  各語言的顯示名稱，例如 {"zh_TW": "聖筊", "en": "Holy"}。
    hotkey 拍照時切換到這個類別的按鍵（"1"–"9"），空字串表示沒有。
    """

    id: str
    names: dict[str, str] = field(default_factory=dict)
    color: str = "#E69F00"
    hotkey: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not is_valid_id(self.id):
            raise ValueError(f"類別 ID {self.id!r} 只能用小寫英文、數字和底線，且不能以數字開頭")

    def display(self, lang: str = "zh_TW") -> str:
        if lang in self.names and self.names[lang]:
            return self.names[lang]
        for v in self.names.values():
            if v:
                return v
        return self.id

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "names": dict(self.names), "color": self.color,
                "hotkey": self.hotkey, "description": self.description}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ClassSpec:
        names = d.get("names") or {}
        if isinstance(names, str):
            names = {"zh_TW": names}
        return cls(id=d["id"], names=dict(names), color=d.get("color", "#E69F00"),
                   hotkey=str(d.get("hotkey", "")), description=d.get("description", ""))


@dataclass
class Project:
    root: Path
    name: str
    description: str = ""
    classes: list[ClassSpec] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.root.name

    def dir(self, sub: str) -> Path:
        """專案底下的子資料夾（會自動建立）。sub 必須是 SUBDIRS 之一。"""
        if sub not in SUBDIRS:
            raise ValueError(f"未知的專案子資料夾：{sub}")
        return ensure_dir(self.root / sub)

    @property
    def dataset_dir(self) -> Path:
        return self.dir("dataset")

    @property
    def models_dir(self) -> Path:
        return self.dir("models")

    @property
    def pipelines_dir(self) -> Path:
        return self.dir("pipelines")

    def class_by_id(self, class_id: str) -> ClassSpec | None:
        return next((c for c in self.classes if c.id == class_id), None)

    def class_ids(self) -> list[str]:
        return [c.id for c in self.classes]

    def add_class(self, name: str, class_id: str | None = None, lang: str = "zh_TW") -> ClassSpec:
        """新增類別。沒給 ID 時由名稱產生（中文名稱會產生 class_1、class_2…）。"""
        base = class_id or slugify(name, fallback=f"class_{len(self.classes) + 1}")
        cid, i = base, 2
        while self.class_by_id(cid):
            cid = f"{base}_{i}"
            i += 1
        used_keys = {c.hotkey for c in self.classes}
        hotkey = next((str(k) for k in range(1, 10) if str(k) not in used_keys), "")
        spec = ClassSpec(id=cid, names={lang: name}, color=DEFAULT_COLORS[len(self.classes) % len(DEFAULT_COLORS)],
                         hotkey=hotkey)
        self.classes.append(spec)
        return spec

    def remove_class(self, class_id: str) -> None:
        """只移除類別定義，不刪除已拍的照片。"""
        self.classes = [c for c in self.classes if c.id != class_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PROJECT_VERSION,
            "name": self.name,
            "description": self.description,
            "created": self.created,
            "classes": [c.to_dict() for c in self.classes],
            **({"extra": self.extra} if self.extra else {}),
        }

    def save(self) -> None:
        ensure_dir(self.root)
        atomic_write_text(self.root / PROJECT_FILE, json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, root: str | Path) -> Project:
        root = Path(root)
        data = json.loads((root / PROJECT_FILE).read_text(encoding="utf-8"))
        return cls(
            root=root,
            name=data.get("name") or root.name,
            description=data.get("description", ""),
            classes=[ClassSpec.from_dict(c) for c in data.get("classes", [])],
            created=float(data.get("created", time.time())),
            extra=dict(data.get("extra", {})),
        )


def project_templates() -> dict[str, dict[str, Any]]:
    """內建的專案範本（edgelens/presets/projects/*.json），鍵為檔名。"""
    out: dict[str, dict[str, Any]] = {}
    folder = resource_path("presets", "projects")
    if folder.is_dir():
        for p in sorted(folder.glob("*.json")):
            try:
                out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
    return out


class Workspace:
    """使用者的工作區資料夾。"""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else default_workspace()

    @property
    def projects_dir(self) -> Path:
        return ensure_dir(self.root / "projects")

    @property
    def plugins_dir(self) -> Path:
        return ensure_dir(self.root / "plugins")

    def projects(self) -> list[Project]:
        out = []
        for d in sorted(self.projects_dir.iterdir()):
            if (d / PROJECT_FILE).is_file():
                try:
                    out.append(Project.load(d))
                except (OSError, ValueError, KeyError):
                    continue
        return out

    def open_project(self, project_id: str) -> Project:
        return Project.load(self.projects_dir / project_id)

    def create_project(self, name: str, template: str | None = None, project_id: str | None = None,
                       lang: str = "zh_TW") -> Project:
        """建立新專案。template 為 project_templates() 的鍵；None 表示空白專案。"""
        pid = project_id or slugify(name, fallback="project")
        root = unique_path(self.projects_dir / pid)
        project = Project(root=root, name=name)
        if template:
            tpl = project_templates().get(template)
            if tpl is None:
                raise KeyError(f"沒有專案範本 {template!r}")
            project.description = tpl.get("description", {}).get(lang, "") if isinstance(
                tpl.get("description"), dict) else tpl.get("description", "")
            project.classes = [ClassSpec.from_dict(c) for c in tpl.get("classes", [])]
        for sub in SUBDIRS:
            ensure_dir(root / sub)
        project.save()
        return project
