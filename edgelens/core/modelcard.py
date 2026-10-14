"""ModelCard：描述一個模型怎麼用（輸入大小、前處理、輸出格式、類別名稱）。

說明檔放在模型旁邊，檔名為 <模型檔名去掉副檔名>.card.json，例如：

    models/yolov8n.onnx
    models/yolov8n.card.json
    models/labels.txt          （或 yolov8n.labels.txt、yolov8n.names、classes.txt）

沒有說明檔時全部用 "auto"，由推論 runner 依模型的輸入輸出形狀自動判斷；判斷錯誤時
學生可以在模型面板手動改，再存成說明檔。

{
  "version": 1,
  "name": "YOLOv8n COCO",
  "file": "yolov8n.onnx",
  "backend": "auto",                 // onnxrt、cvdnn、tflite、haar、device
  "task": "auto",                    // detect、classify
  "decoder": "auto",                 // yolo_t、yolo_flat、yolo_grid、end2end、tflite_ssd、classify…
  "decoder_params": {},              // 例如 {"anchors": [...], "strides": [8,16,32]}
  "labels": ["person", "bicycle"],   // 或 "labels_file": "labels.txt"
  "labels_i18n": {"zh_TW": "labels.zh_TW.txt"},   // 選填：各語言的類別名稱（檔名或清單），順序同 labels
  "input": {"size": [640, 640], "layout": "auto", "color": "RGB", "resize": "letterbox",
            "scale": 0.00392156862745098, "mean": [0,0,0], "std": [1,1,1], "pad_value": 114, "dtype": "auto"},
  "thresholds": {"conf": 0.25, "nms": 0.45, "max_det": 100, "class_agnostic": false,
                 "classes": [0, 7]},     // 選填：只顯示這些類別編號，省略＝全部
  "extra_files": {"config": "yolov4-tiny.cfg"},
  "device": {},                      // 板上推論設定（例如 K210 的 anchors）
  "notes": ""
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from edgelens.core.paths import atomic_write_text

CARD_SUFFIX = ".card.json"
CARD_VERSION = 1
LABEL_FILE_CANDIDATES = ("{stem}.labels.txt", "{stem}.names", "{stem}.txt", "labels.txt", "classes.txt",
                         "obj.names", "coco.names")


@dataclass
class InputSpec:
    """模型輸入的前處理設定。

    size     (寬, 高)；None 表示由模型的輸入形狀決定（動態大小時預設 640×640）。
    layout   "NCHW"、"NHWC" 或 "auto"（依模型形狀判斷）。
    color    "RGB"、"BGR"、"GRAY"。
    resize   "letterbox"（等比例縮放、補邊）、"stretch"（直接拉伸）、"center_crop"（等比例裁中間）。
    scale／mean／std   張量值 = (像素 * scale - mean) / std，依通道。
    dtype    "auto"、"float32"、"uint8"、"int8"；auto 依模型輸入型別。
    """

    size: tuple[int, int] | None = None
    layout: str = "auto"
    color: str = "RGB"
    resize: str = "letterbox"
    scale: float = 1.0 / 255.0
    mean: tuple[float, ...] = (0.0, 0.0, 0.0)
    std: tuple[float, ...] = (1.0, 1.0, 1.0)
    pad_value: int = 114
    dtype: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return {"size": list(self.size) if self.size else None, "layout": self.layout, "color": self.color,
                "resize": self.resize, "scale": self.scale, "mean": list(self.mean), "std": list(self.std),
                "pad_value": self.pad_value, "dtype": self.dtype}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InputSpec:
        size = d.get("size")
        return cls(
            size=(int(size[0]), int(size[1])) if size else None,
            layout=d.get("layout", "auto"),
            color=str(d.get("color", "RGB")).upper(),
            resize=d.get("resize", "letterbox"),
            scale=float(d.get("scale", 1.0 / 255.0)),
            mean=tuple(float(x) for x in d.get("mean", (0.0, 0.0, 0.0))),
            std=tuple(float(x) for x in d.get("std", (1.0, 1.0, 1.0))),
            pad_value=int(d.get("pad_value", 114)),
            dtype=d.get("dtype", "auto"),
        )


@dataclass
class ModelCard:
    name: str
    file: str
    card_path: Path | None = None
    backend: str = "auto"
    task: str = "auto"
    decoder: str = "auto"
    decoder_params: dict[str, Any] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)
    labels_file: str = ""
    labels_i18n: dict[str, Any] = field(default_factory=dict)
    labels_base: list[str] = field(default_factory=list, repr=False)   # 換成別的語言前的原始名稱（不存檔）
    input: InputSpec = field(default_factory=InputSpec)
    conf_threshold: float = 0.25
    nms_threshold: float = 0.45
    max_det: int = 100
    class_agnostic_nms: bool = False
    class_filter: list[int] = field(default_factory=list)   # 只顯示這些類別編號；空＝全部
    extra_files: dict[str, str] = field(default_factory=dict)
    device: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    # ---- 路徑 ------------------------------------------------------------------

    @property
    def base_dir(self) -> Path:
        if self.card_path is not None:
            return self.card_path.parent
        return Path(self.file).parent

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (self.base_dir / p)

    @property
    def model_path(self) -> Path:
        return self.resolve(self.file)

    @property
    def extension(self) -> str:
        """模型副檔名（小寫，含點）。"""
        return self.model_path.suffix.lower()

    def extra_path(self, key: str) -> Path | None:
        rel = self.extra_files.get(key)
        return self.resolve(rel) if rel else None

    def default_card_path(self) -> Path:
        mp = self.model_path
        return mp.with_name(mp.stem + CARD_SUFFIX)

    # ---- 類別名稱 --------------------------------------------------------------

    def label(self, class_id: int) -> str:
        if 0 <= class_id < len(self.labels):
            return self.labels[class_id]
        return str(class_id)

    @property
    def num_classes(self) -> int | None:
        return len(self.labels) or None

    def _load_labels_from_files(self) -> None:
        candidates = [self.labels_file] if self.labels_file else [
            c.format(stem=self.model_path.stem) for c in LABEL_FILE_CANDIDATES]
        for cand in candidates:
            p = self.resolve(cand)
            labels = _read_label_file(p)
            if labels:
                self.labels = labels
                self.labels_file = str(p.name) if p.parent == self.base_dir else str(p)
                return

    def localize_labels(self, lang: str) -> bool:
        """有 labels_i18n[lang] 就把 labels 換成該語言的名稱（例如繁中）。數量和原本不同時不換，
        免得名稱錯位。只改記憶體裡的 labels，存檔時仍寫原本的 labels_file，回傳是否有換。"""
        entry = self.labels_i18n.get(lang)
        if isinstance(entry, list):
            names = [str(x) for x in entry]
        elif isinstance(entry, str) and entry:
            names = _read_label_file(self.resolve(entry))
        else:
            return False
        if not names or (self.labels and len(names) != len(self.labels)):
            return False
        self.labels_base = self.labels_base or list(self.labels)
        self.labels = names
        return True

    # ---- 存讀 ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": CARD_VERSION,
            "name": self.name,
            "file": self.file,
            "backend": self.backend,
            "task": self.task,
            "decoder": self.decoder,
            "decoder_params": dict(self.decoder_params),
            "input": self.input.to_dict(),
            "thresholds": {"conf": self.conf_threshold, "nms": self.nms_threshold, "max_det": self.max_det,
                           "class_agnostic": self.class_agnostic_nms,
                           **({"classes": sorted(self.class_filter)} if self.class_filter else {})},
            "extra_files": dict(self.extra_files),
            "device": dict(self.device),
            "notes": self.notes,
        }
        if self.labels_file:
            d["labels_file"] = self.labels_file
        else:
            d["labels"] = list(self.labels)
        if self.labels_i18n:
            d["labels_i18n"] = dict(self.labels_i18n)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any], card_path: Path | None = None) -> ModelCard:
        version = int(d.get("version", 1))
        if version > CARD_VERSION:
            raise ValueError(f"模型說明檔版本 {version} 比程式新，請更新 EdgeLens")
        th = d.get("thresholds", {})
        labels = d.get("labels", [])
        card = cls(
            name=d.get("name") or Path(d["file"]).stem,
            file=d["file"],
            card_path=card_path,
            backend=d.get("backend", "auto"),
            task=d.get("task", "auto"),
            decoder=d.get("decoder", "auto"),
            decoder_params=dict(d.get("decoder_params", {})),
            labels=list(labels) if isinstance(labels, list) else [],
            labels_file=d.get("labels_file", labels if isinstance(labels, str) else ""),
            labels_i18n=dict(d.get("labels_i18n") or {}),
            input=InputSpec.from_dict(d.get("input", {})),
            conf_threshold=float(th.get("conf", 0.25)),
            nms_threshold=float(th.get("nms", 0.45)),
            max_det=int(th.get("max_det", 100)),
            class_agnostic_nms=bool(th.get("class_agnostic", False)),
            class_filter=[int(x) for x in th.get("classes") or []],
            extra_files=dict(d.get("extra_files", {})),
            device=dict(d.get("device", {})),
            notes=d.get("notes", ""),
        )
        if not card.labels:
            card._load_labels_from_files()
        return card

    @classmethod
    def load(cls, path: str | Path) -> ModelCard:
        """讀取說明檔或模型檔。

        - 傳入 *.card.json：讀取說明檔。
        - 傳入模型檔：旁邊有同名說明檔就讀它；沒有就產生全部為 auto 的說明（不存檔）。
        Darknet 的 .weights 會自動找同名 .cfg。
        """
        path = Path(path)
        if path.name.endswith(CARD_SUFFIX):
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")), card_path=path)
        sibling = path.with_name(path.stem + CARD_SUFFIX)
        if sibling.is_file():
            return cls.from_dict(json.loads(sibling.read_text(encoding="utf-8")), card_path=sibling)
        card = cls(name=path.stem, file=str(path.resolve()))
        if path.suffix.lower() == ".weights":
            cfg = path.with_suffix(".cfg")
            if cfg.is_file():
                card.extra_files["config"] = str(cfg.resolve())
        card._load_labels_from_files()
        return card

    def save(self, path: str | Path | None = None) -> Path:
        """存成說明檔；預設存在模型旁邊。file 會改成相對路徑（模型和說明檔一起搬也能用）。"""
        target = Path(path) if path else (self.card_path or self.default_card_path())
        folder = target.parent.resolve()

        def rebase(rel: str) -> str:
            full = self.resolve(rel).resolve()
            try:
                return str(full.relative_to(folder))
            except ValueError:
                return str(full)

        # 先全部用舊位置算好再一起改（改了 file 之後 base_dir 就變了）
        file = rebase(self.file)
        extra = {key: rebase(rel) for key, rel in self.extra_files.items()}
        labels_file = rebase(self.labels_file) if self.labels_file else ""
        i18n = {lang: rebase(e) if isinstance(e, str) and e else e for lang, e in self.labels_i18n.items()}
        self.file, self.extra_files, self.labels_file, self.labels_i18n = file, extra, labels_file, i18n
        self.card_path = target
        atomic_write_text(target, json.dumps(self.to_dict(), ensure_ascii=False, indent=2))
        return target


def _read_label_file(p: Path) -> list[str]:
    """一行一個類別名稱；略過空行與 # 開頭的行，也接受 "0 person" 的格式。讀不到回傳空清單。"""
    if not p.is_file() or p.suffix.lower() not in (".txt", ".names", ""):
        return []
    try:
        lines = p.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    labels = [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    if labels and all(ln.split(" ", 1)[0].isdigit() and " " in ln for ln in labels):
        labels = [ln.split(" ", 1)[1].strip() for ln in labels]
    return labels
