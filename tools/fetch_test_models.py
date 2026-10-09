#!/usr/bin/env python
"""下載真實模型到 tests/fixtures/models/（已加入 .gitignore），給 `pytest -m network` 驗證解碼器用。

    python tools/fetch_test_models.py              # 下載全部（約 250 MB；yolov4 ONNX 約 245 MB 要加 --all）
    python tools/fetch_test_models.py --list       # 只列出清單、授權與是否已下載
    python tools/fetch_test_models.py yolov8n yolox_nano
    python tools/fetch_test_models.py --verify     # 檢查已下載檔案的 sha256

每個模型一個子資料夾，旁邊附 labels.txt 與 .card.json（說明檔），內容就是 EdgeLens 使用者會寫的設定，
所以這個腳本同時也是「各種模型該怎麼寫說明檔」的範例。第一次下載後，把印出的 sha256 填進 SHA256 表，
之後每次下載都會驗證。只用標準函式庫，不需要安裝套件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "tests" / "fixtures" / "models"
UA = {"User-Agent": "EdgeLens-test-fetch/1.0"}

COCO80 = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
VOC20 = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/voc.names"
IMAGENET = "https://raw.githubusercontent.com/onnx/models/main/validated/vision/classification/synset.txt"
ZOO = "https://github.com/onnx/models/raw/main/validated/vision"


@dataclass
class Item:
    """一個要下載的檔案。dest 相對於模型資料夾；unzip 指定從 zip 取出的檔案（zip 內路徑 → dest）。"""

    url: str
    dest: str
    unzip: dict[str, str] = field(default_factory=dict)
    transform: str = ""            # 下載後的處理：coco_labels、synset、ssd_labels


@dataclass
class TestModel:
    id: str
    title: str
    license: str
    items: list[Item]
    card: dict                     # 寫成 <id>.card.json 的內容
    default: bool = True
    note: str = ""


def _card(file: str, **kw) -> dict:
    d = {"version": 1, "name": kw.pop("name", file), "file": file}
    d.update(kw)
    return d


MODELS: list[TestModel] = [
    TestModel(
        "yolov8n", "YOLOv8n COCO（ultralytics 匯出，Unity inference-engine-yolo 提供）", "AGPL-3.0（ultralytics）",
        [Item("https://huggingface.co/unity/inference-engine-yolo/resolve/main/models/yolov8n.onnx", "yolov8n.onnx"),
         Item(COCO80, "labels.txt")],
        _card("yolov8n.onnx", input={"size": [640, 640]})),
    TestModel(
        "yolov5n", "YOLOv5n COCO（ultralytics 官方 release）", "AGPL-3.0（ultralytics）",
        [Item("https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx", "yolov5n.onnx"),
         Item(COCO80, "labels.txt")],
        _card("yolov5n.onnx")),
    TestModel(
        "yolov5s", "YOLOv5s COCO（ultralytics 官方 release）", "AGPL-3.0（ultralytics）",
        [Item("https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx", "yolov5s.onnx"),
         Item(COCO80, "labels.txt")],
        _card("yolov5s.onnx")),
    TestModel(
        "yolov10n", "YOLOv10n COCO（onnx-community 轉出）", "AGPL-3.0（THU-MIG／ultralytics）",
        [Item("https://huggingface.co/onnx-community/yolov10n/resolve/main/onnx/model.onnx", "yolov10n.onnx"),
         Item(COCO80, "labels.txt")],
        _card("yolov10n.onnx")),
    TestModel(
        "yolox_nano", "YOLOX-Nano COCO（官方 release，未解碼輸出）", "Apache-2.0（Megvii）",
        [Item("https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx",
              "yolox_nano.onnx"), Item(COCO80, "labels.txt")],
        # YOLOX 的前處理：BGR、0–255 不正規化
        _card("yolox_nano.onnx", input={"color": "BGR", "scale": 1.0})),
    TestModel(
        "yolox_s", "YOLOX-S COCO（官方 release，未解碼輸出）", "Apache-2.0（Megvii）",
        [Item("https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx",
              "yolox_s.onnx"), Item(COCO80, "labels.txt")],
        _card("yolox_s.onnx", input={"color": "BGR", "scale": 1.0})),
    TestModel(
        "tiny_yolov2", "Tiny YOLOv2 VOC（ONNX Model Zoo）", "MIT（ONNX Model Zoo）",
        [Item(f"{ZOO}/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx", "tinyyolov2-8.onnx"),
         Item(VOC20, "labels.txt")],
        # 輸入 0–255 不正規化
        _card("tinyyolov2-8.onnx", input={"scale": 1.0, "resize": "stretch"},
              thresholds={"conf": 0.3, "nms": 0.45})),
    TestModel(
        "yolov4", "YOLOv4 COCO（ONNX Model Zoo，約 245 MB，三個格點輸出）", "MIT（ONNX Model Zoo）",
        [Item(f"{ZOO}/object_detection_segmentation/yolov4/model/yolov4.onnx", "yolov4.onnx"),
         Item(COCO80, "labels.txt")],
        _card("yolov4.onnx", decoder_params={"anchors_preset": "yolov4_coco"}, input={"resize": "stretch"}),
        default=False, note="檔案很大，用 --all 或明確指名才下載"),
    TestModel(
        "mobilenetv2", "MobileNetV2 ImageNet 分類（ONNX Model Zoo）", "Apache-2.0",
        [Item(f"{ZOO}/classification/mobilenet/model/mobilenetv2-12.onnx", "mobilenetv2-12.onnx"),
         Item(IMAGENET, "labels.txt", transform="synset")],
        _card("mobilenetv2-12.onnx", input={"size": [224, 224], "resize": "center_crop",
                                            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})),
    TestModel(
        "yolov4_tiny_darknet", "YOLOv4-tiny COCO（Darknet cfg + weights，AlexeyAB）", "YOLO license / Public Domain（AlexeyAB）",
        [Item("https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
              "yolov4-tiny.weights"),
         Item("https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg", "yolov4-tiny.cfg"),
         Item(COCO80, "labels.txt")],
        _card("yolov4-tiny.weights", backend="cvdnn", extra_files={"config": "yolov4-tiny.cfg"},
              input={"resize": "stretch"})),
    TestModel(
        "ssd_mobilenet_tflite", "SSD MobileNet v1 COCO 量化（TensorFlow 官方範例）", "Apache-2.0（TensorFlow）",
        [Item("https://storage.googleapis.com/download.tensorflow.org/models/tflite/"
              "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip", "_ssd.zip",
              unzip={"detect.tflite": "detect.tflite", "labelmap.txt": "labels.txt"}, transform="ssd_labels")],
        _card("detect.tflite", input={"resize": "stretch"})),
    TestModel(
        "mobilenet_tflite", "MobileNet v1 224 量化分類（TensorFlow 官方範例）", "Apache-2.0（TensorFlow）",
        [Item("https://storage.googleapis.com/download.tensorflow.org/models/tflite/"
              "mobilenet_v1_1.0_224_quant_and_labels.zip", "_mobilenet.zip",
              unzip={"mobilenet_v1_1.0_224_quant.tflite": "mobilenet_v1_1.0_224_quant.tflite",
                     "labels_mobilenet_quant_v1_224.txt": "labels.txt"})],
        # 量化輸入：像素值要換成 -1~1（(像素-127.5)/127.5），再依輸入的量化參數轉成 uint8
        _card("mobilenet_v1_1.0_224_quant.tflite",
              input={"resize": "center_crop", "scale": 1 / 127.5, "mean": [1.0, 1.0, 1.0]})),
]

IMAGES = [
    ("images/dog.jpg", "https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg",
     "Darknet 範例圖（狗、腳踏車、卡車）；YOLO 授權"),
    ("images/messi5.jpg", "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/messi5.jpg",
     "OpenCV 範例圖（足球員，有人臉）；Apache-2.0"),
]

# 第一次下載後把 sha256 填在這裡（key：模型 id／相對路徑），之後每次都會驗證。
SHA256: dict[str, str] = {
    "images/dog.jpg": "5a9522051c3cec2bbd2f6323fccba32e8fbf3ddcc2b3e2fd46b04c720bc6f866",
    "images/messi5.jpg": "1d570e49654e84c7a943918537bd9e5e1ef82920152e147c834006e235be97c9",
    "yolov8n/yolov8n.onnx": "341ad75c98ff88775c63e899e7cbbf497c13161e3393b95b620a6cab65052811",
    "yolov5n/yolov5n.onnx": "04f0e55c26f58d17145b36045780fe1250d5bd2187543e11568e5141d05b3262",
    "yolov5s/yolov5s.onnx": "8bc25a47abdbe2fd23e30eb948746bf856b031ea3cc815166687f52db9f8c659",
    "yolov10n/yolov10n.onnx": "a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b",
    "yolox_nano/yolox_nano.onnx": "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
    "yolox_s/yolox_s.onnx": "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
    "yolov4_tiny_darknet/yolov4-tiny.weights": "cf9fbfd0f6d4869b35762f56100f50ed05268084078805f0e7989efe5bb8ca87",
    "tiny_yolov2/tinyyolov2-8.onnx": "583fb7fdc948435ceac9fa82efc7708701efe8382a859a3dd46526b155f5f2ae",
    "mobilenetv2/mobilenetv2-12.onnx": "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5",
    "ssd_mobilenet_tflite/detect.tflite": "e4b118e5e4531945de2e659742c7c590f7536f8d0ed26d135abcfe83b4779d13",
    "mobilenet_tflite/mobilenet_v1_1.0_224_quant.tflite": "ecc3a67c47c5a609ec35f6a58a7d97532834e43df4cb7d3f1204a8164b7d20dd",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r, part.open("wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    part.replace(dest)


def _transform(kind: str, path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.rstrip() for ln in text.splitlines()]
    if kind == "synset":       # "n01440764 tench, Tinca tinca" → "tench"
        lines = [ln.split(" ", 1)[1].split(",")[0].strip() if " " in ln else ln for ln in lines if ln.strip()]
    elif kind == "ssd_labels":  # 第一行 "???" 是 TF 的背景，模型輸出的類別編號從 0 開始，不含它
        lines = lines[1:]
    else:
        return
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fetch_model(m: TestModel) -> list[tuple[str, int, str]]:
    folder = MODELS_DIR / m.id
    rows = []
    for it in m.items:
        dest = folder / it.dest
        if not dest.exists():
            print(f"  下載 {it.url}")
            download(it.url, dest)
        if it.unzip:
            with zipfile.ZipFile(dest) as z:
                for inner, out in it.unzip.items():
                    target = folder / out
                    if not target.exists():
                        target.write_bytes(z.read(inner))
            if it.transform:
                for out in it.unzip.values():
                    if out == "labels.txt":
                        _transform(it.transform, folder / out)
        elif it.transform and not (folder / (it.dest + ".done")).exists():
            _transform(it.transform, dest)
            (folder / (it.dest + ".done")).write_text("1")
        key = f"{m.id}/{it.dest}"
        for out in (it.unzip.values() if it.unzip else [it.dest]):
            p = folder / out
            if p.suffix in (".onnx", ".tflite", ".weights", ".jpg"):
                digest = sha256_of(p)
                expected = SHA256.get(f"{m.id}/{out}")
                status = "未登錄" if expected is None else ("相符" if expected == digest else "不符！")
                rows.append((f"{m.id}/{out}", p.stat().st_size, f"{digest} ({status})"))
                if expected is not None and expected != digest:
                    raise SystemExit(f"sha256 不符：{key}，預期 {expected}，實際 {digest}")
    card_path = folder / f"{Path(m.card['file']).stem}.card.json"
    card_path.write_text(json.dumps(m.card, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def fetch_images() -> list[tuple[str, int, str]]:
    rows = []
    for rel, url, _lic in IMAGES:
        dest = MODELS_DIR / rel
        if not dest.exists():
            print(f"  下載 {url}")
            download(url, dest)
        digest = sha256_of(dest)
        expected = SHA256.get(rel)
        rows.append((rel, dest.stat().st_size, f"{digest} ({'未登錄' if expected is None else '相符' if expected == digest else '不符！'})"))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="只下載這些模型（預設全部，yolov4 除外）")
    ap.add_argument("--list", action="store_true", help="列出清單與授權")
    ap.add_argument("--all", action="store_true", help="包含大型模型（yolov4）")
    ap.add_argument("--verify", action="store_true", help="只驗證已下載檔案的 sha256")
    args = ap.parse_args(argv)

    if args.list:
        for m in MODELS:
            have = (MODELS_DIR / m.id).is_dir()
            print(f"{m.id:22s} {'[已下載]' if have else '[未下載]'} {m.title}\n{'':22s} 授權：{m.license}"
                  f"{'  ' + m.note if m.note else ''}")
        for rel, _u, lic in IMAGES:
            print(f"{rel:22s} {lic}")
        return 0

    if args.verify:
        bad = 0
        for key, want in SHA256.items():
            p = MODELS_DIR / key
            if not p.is_file():
                print(f"略過（未下載） {key}")
                continue
            ok = sha256_of(p) == want
            bad += not ok
            print(f"{'相符' if ok else '不符！'} {key}")
        return 1 if bad else 0

    chosen = [m for m in MODELS if (m.id in args.names if args.names else (m.default or args.all))]
    unknown = set(args.names) - {m.id for m in MODELS}
    if unknown:
        print(f"不認識的模型：{', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    rows = fetch_images()
    for m in chosen:
        print(f"[{m.id}] {m.title}")
        try:
            rows += fetch_model(m)
        except Exception as e:  # noqa: BLE001 - 單一模型失敗不影響其他
            print(f"  失敗：{e}", file=sys.stderr)
    print("\n檔案                                              大小(bytes)  sha256")
    for name, size, digest in rows:
        print(f"{name:48s} {size:12d}  {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
