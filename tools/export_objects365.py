#!/usr/bin/env python
"""製作 Objects365 下載包（維護者用，學生電腦不需要）：YOLO26 模型轉 ONNX、繁中類別名稱、示範照片。

要在裝了 ultralytics 8.4 以上（含 PyTorch）的環境執行，產出的 zip 上傳到 EdgeLens 的 GitHub Release，
App 從那裡下載。學生電腦只拿到 ONNX，不需要 PyTorch。

    python tools/export_objects365.py models            # 匯出 n、s 兩個模型包
    python tools/export_objects365.py models --sizes n
    python tools/export_objects365.py samples           # 下載驗證集 patch0（約 133 MB），自動挑示範照片
    python tools/export_objects365.py samples --keep picked.txt   # 人工審過、刪掉不要的檔名後，照清單重新打包
    python tools/export_objects365.py all

輸出在 build/objects365/（已在 .gitignore）：
    objects365-yolo26n.zip、objects365-yolo26s.zip、objects365-samples.zip
    samples_preview.jpg   示範照片縮圖總覽，上傳前請人工看過一次
    release.txt           每個 zip 的大小與 sha256，貼進 edgelens/zoo/catalog.py

授權：模型權重是 Ultralytics 的 AGPL-3.0；Objects365 資料集（照片）是 CC BY 4.0。兩者都寫進 zip 裡的說明檔。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "objects365"
NAMES_TSV = Path(__file__).resolve().parent / "objects365" / "names_zh_TW.tsv"
VAL_PATCH = ("https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/"
             "val/images/v1/patch0.tar.gz")
NUM_CLASSES = 365
CARD_CONF = {"n": 0.35, "s": 0.35}

# 示範照片挑選：這些類別出現在照片裡就不選（國中課堂用）
EXCLUDE_CLASSES = {"gun", "cigar", "lighter", "urinal", "toilet"}
SAMPLE_COUNT = 40
SAMPLE_LONG_EDGE = 1280
PICK_CONF = 0.5

MODEL_LICENSE = """\
YOLO26{size} 以 Objects365v1 預訓練（Ultralytics 釋出的 yolo26{size}-objv1-150.pt），由 EdgeLens 維護者轉成 ONNX。

模型權重授權：AGPL-3.0（Ultralytics）。https://github.com/ultralytics/ultralytics/blob/main/LICENSE
訓練資料：Objects365（Shao et al., ICCV 2019），CC BY 4.0。https://www.objects365.org/
類別繁中名稱：EdgeLens 專案翻譯，MIT 授權。

這個模型認得 365 種常見物品，但不是每一種都很準（YOLO26{size} 在 Objects365 驗證集的 mAP50-95 約 {map}）。
"""
MAP = {"n": "16.3", "s": "24.3"}   # Ultralytics 文件公布的 mAPval 50-95

SAMPLES_ATTRIBUTION = """\
這些照片取自 Objects365 資料集驗證集（val v1 patch0），縮小後重新存成 JPEG。

Objects365: A Large-scale, High-quality Dataset for Object Detection (Shao et al., ICCV 2019)
https://www.objects365.org/   授權：CC BY 4.0（https://creativecommons.org/licenses/by/4.0/）

檔名對照（EdgeLens 檔名 → 原始檔名）：
"""


# ---- 共用 ---------------------------------------------------------------------------

def read_names() -> tuple[list[str], list[str]]:
    """讀中英對照表 → (英文, 繁中)。中文欄的 ★ 與全形括號後的說明會去掉。"""
    en, zh = [], []
    for line in NAMES_TSV.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        e, z = line.split("\t")
        en.append(e)
        zh.append(z.replace("★", "").split("（")[0].strip())
    if len(en) != NUM_CLASSES or not all(zh):
        raise SystemExit(f"{NAMES_TSV.name} 應該有 {NUM_CLASSES} 列且中文不能空白，實際 {len(en)} 列")
    return en, zh


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"  下載 {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "EdgeLens-export/1.0"})
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


def make_zip(folder: Path, dest: Path) -> None:
    """folder 裡的檔案平放進 zip（不含外層資料夾），App 解壓到 zoo/<id>/。
    檔案時間固定，內容沒變時重新打包的 sha256 也不變。"""
    tmp = dest.with_name(dest.name + ".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                info = zipfile.ZipInfo(p.relative_to(folder).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                z.writestr(info, p.read_bytes())
    tmp.replace(dest)


def record(dest: Path) -> None:
    line = f"{dest.name}\t{dest.stat().st_size}\t{sha256_of(dest)}"
    rel = OUT / "release.txt"
    rows = [r for r in (rel.read_text(encoding="utf-8").splitlines() if rel.exists() else [])
            if not r.startswith(dest.name + "\t")]
    rel.write_text("\n".join(sorted(rows + [line])) + "\n", encoding="utf-8")
    print(f"  {line}")


# ---- 模型 ---------------------------------------------------------------------------

def export_model(size: str) -> Path:
    from ultralytics import YOLO

    en, zh = read_names()
    weights = OUT / "weights" / f"yolo26{size}-objv1-150.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    print(f"[yolo26{size}] 載入 {weights.name}（第一次會自動下載）")
    model = YOLO(str(weights))
    names = [model.names[i] for i in range(len(model.names))]
    if names != en:
        diff = next(i for i, (a, b) in enumerate(zip(names, en, strict=False)) if a != b)
        raise SystemExit(f"模型類別和 {NAMES_TSV.name} 不一致，第 {diff} 類：{names[diff]!r} ≠ {en[diff]!r}")
    onnx_path = Path(model.export(format="onnx", imgsz=640))
    check_onnx(onnx_path)

    stage = OUT / f"stage-yolo26{size}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    pack = f"objects365-yolo26{size}"
    shutil.copy2(onnx_path, stage / f"{pack}.onnx")
    (stage / "labels.txt").write_text("\n".join(en) + "\n", encoding="utf-8")
    (stage / "labels.zh_TW.txt").write_text("\n".join(zh) + "\n", encoding="utf-8")
    card = {
        "version": 1,
        "name": f"YOLO26{size} Objects365（365 類）",
        "file": f"{pack}.onnx",
        "decoder": "yolo_t",
        "labels_file": "labels.txt",
        "labels_i18n": {"zh_TW": "labels.zh_TW.txt"},
        "input": {"size": [640, 640], "color": "RGB", "resize": "letterbox"},
        "thresholds": {"conf": CARD_CONF[size], "nms": 0.45, "max_det": 100},
        "notes": "Objects365 預訓練，365 類常見物品。",
    }
    (stage / f"{pack}.card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    (stage / "LICENSE-MODEL.txt").write_text(MODEL_LICENSE.format(size=size, map=MAP[size]), encoding="utf-8")
    dest = OUT / f"{pack}.zip"
    make_zip(stage, dest)
    record(dest)
    return dest


def check_onnx(path: Path) -> None:
    """確認輸出是 EdgeLens yolo_t 解碼器吃的 [1, 4+365, N]，且內嵌 365 個類別名稱。

    objv1 權重預設不是 end-to-end（Ultralytics 公布的 mAP 也是用一般輸出加 NMS），所以不硬開 end2end，
    NMS 交給 EdgeLens 做。"""
    import onnx

    m = onnx.load(str(path), load_external_data=False)
    outs = [[d.dim_value for d in o.type.tensor_type.shape.dim] for o in m.graph.output]
    if len(outs) != 1 or len(outs[0]) != 3 or outs[0][1] != 4 + NUM_CLASSES:
        raise SystemExit(f"ONNX 輸出形狀不是 [1, {4 + NUM_CLASSES}, N]：{outs}")
    meta = {p.key: p.value for p in m.metadata_props}
    if "names" not in meta:
        raise SystemExit("ONNX 沒有內嵌類別名稱（metadata names）")
    print(f"  ONNX 輸出 {outs[0]}，大小 {path.stat().st_size / 1e6:.1f} MB")


# ---- 示範照片 -----------------------------------------------------------------------

def candidate_images() -> list[Path]:
    tar = OUT / "cache" / "val_patch0.tar.gz"
    download(VAL_PATCH, tar)
    folder = OUT / "cache" / "val_patch0"
    if not folder.is_dir():
        print("  解壓縮 patch0")
        tmp = folder.with_name(folder.name + ".part")
        shutil.rmtree(tmp, ignore_errors=True)
        with tarfile.open(tar) as t:
            t.extractall(tmp, filter="data")
        tmp.replace(folder)
    return sorted(folder.rglob("*.jpg"))


def pick_samples(images: list[Path], size: str) -> list[Path]:
    """用模型自己的偵測結果挑照片：偵測到的類別越多樣越好，避開不適合課堂的類別，貪婪地讓整組照片涵蓋最多類別。"""
    from ultralytics import YOLO

    model = YOLO(str(OUT / "weights" / f"yolo26{size}-objv1-150.pt"))
    found: dict[Path, set[str]] = {}
    print(f"  用 yolo26{size} 掃描 {len(images)} 張照片")
    for i in range(0, len(images), 16):
        batch = images[i:i + 16]
        for path, r in zip(batch, model.predict([str(p) for p in batch], conf=PICK_CONF, verbose=False),
                           strict=True):
            labels = {model.names[int(c)] for c in r.boxes.cls.tolist()}
            if labels and not labels & EXCLUDE_CLASSES and len(labels) >= 3:
                found[path] = labels
        print(f"\r    {min(i + 16, len(images))} / {len(images)}", end="", flush=True)
    print()
    chosen: list[Path] = []
    covered: set[str] = set()
    pool = dict(found)
    while pool and len(chosen) < SAMPLE_COUNT:
        best = max(pool, key=lambda p: (len(pool[p] - covered), len(pool[p]), p.name))
        chosen.append(best)
        covered |= pool.pop(best)
    print(f"  挑了 {len(chosen)} 張，涵蓋 {len(covered)} 種類別")
    return chosen


def build_samples(size: str, keep: Path | None) -> Path:
    import cv2

    images = candidate_images()
    if keep:
        wanted = {ln.strip() for ln in keep.read_text(encoding="utf-8").splitlines() if ln.strip()}
        chosen = [p for p in images if p.name in wanted]
        missing = wanted - {p.name for p in chosen}
        if missing:
            raise SystemExit(f"清單裡有找不到的檔案：{', '.join(sorted(missing))}")
    else:
        chosen = pick_samples(images, size)
    (OUT / "picked.txt").write_text("\n".join(p.name for p in chosen) + "\n", encoding="utf-8")

    stage = OUT / "stage-samples"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    mapping = []
    thumbs = []
    for i, src in enumerate(chosen, 1):
        img = cv2.imread(str(src))
        h, w = img.shape[:2]
        scale = min(1.0, SAMPLE_LONG_EDGE / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        name = f"objects365_{i:02d}.jpg"
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        assert ok
        (stage / name).write_bytes(buf.tobytes())
        mapping.append(f"{name}\t{src.name}")
        t = cv2.resize(img, (240, round(240 * img.shape[0] / img.shape[1])))
        t = cv2.copyMakeBorder(t, 0, max(0, 180 - t.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:180]
        cv2.putText(t, f"{i:02d} {src.name}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        thumbs.append(t)
    (stage / "ATTRIBUTION.txt").write_text(SAMPLES_ATTRIBUTION + "\n".join(mapping) + "\n", encoding="utf-8")
    write_preview(thumbs, OUT / "samples_preview.jpg")
    dest = OUT / "objects365-samples.zip"
    make_zip(stage, dest)
    record(dest)
    print(f"  請打開 {OUT / 'samples_preview.jpg'} 人工看過；不要的照片從 picked.txt 刪掉後，"
          f"用 --keep {OUT / 'picked.txt'} 重新打包")
    return dest


def write_preview(thumbs: list, dest: Path) -> None:
    import cv2
    import numpy as np

    cols = 6
    blank = np.zeros_like(thumbs[0])
    rows = [np.hstack(thumbs[i:i + cols] + [blank] * (cols - len(thumbs[i:i + cols])))
            for i in range(0, len(thumbs), cols)]
    cv2.imwrite(str(dest), np.vstack(rows))


# ---- 主程式 -------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["models", "samples", "all"])
    ap.add_argument("--sizes", default="ns", help="要匯出的模型大小（預設 ns）")
    ap.add_argument("--pick-with", default="s", help="挑示範照片用哪個模型（預設 s）")
    ap.add_argument("--keep", type=Path, help="照這份檔名清單打包示範照片，不重新挑")
    args = ap.parse_args(argv)
    random.seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.what in ("models", "all"):
        for size in args.sizes:
            export_model(size)
    if args.what in ("samples", "all"):
        if not (OUT / "weights" / f"yolo26{args.pick_with}-objv1-150.pt").exists():
            from ultralytics import YOLO
            YOLO(str(OUT / "weights" / f"yolo26{args.pick_with}-objv1-150.pt"))
        build_samples(args.pick_with, args.keep)
    print(f"\n完成。上傳清單與 sha256：{OUT / 'release.txt'}")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)   # ultralytics／PyTorch 有時留下非 daemon 執行緒，用 sys.exit 會卡在結束
