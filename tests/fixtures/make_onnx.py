"""產生小型假 ONNX 模型：輸入 1x3x64x64，輸出是各 YOLO 家族「已知的固定輸出」。

用途：測試後端與 runner 從頭到尾（Frame → InferenceResult）的流程，不需要下載真實模型。
輸出由 Constant 加上「輸入平均值 x 0」得到，這樣輸出才和輸入有相依（OpenCV DNN 讀不了
只有常數的圖），數值又完全固定。

模型檔在測試時才產生到 tests/fixtures/generated/（已加入 .gitignore），不要放進 repo。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

INPUT_SIZE = 64
LABELS = ["alpha", "beta", "gamma"]
GENERATED_DIR = Path(__file__).resolve().parent / "generated"


@dataclass
class Expect:
    """模型輸入座標（64x64）下應該得到的結果。"""

    kind: str                                   # detect／classify
    class_id: int
    box: tuple[float, float, float, float] | None = None
    min_score: float = 0.5
    decoder: str = ""                           # 自動判斷應選到的解碼器


def _yolo_t() -> np.ndarray:
    a = np.full((1, 4 + 3, 84), 0.0, np.float32)
    a[0, 4:, :] = 0.01
    a[0, :4, 40] = (32, 24, 24, 16)             # cx, cy, w, h
    a[0, 4 + 1, 40] = 0.9
    return a


def _yolo_flat() -> np.ndarray:
    a = np.zeros((1, 252, 8), np.float32)
    a[0, :, 4] = 0.01
    a[0, 100, :4] = (32, 24, 24, 16)
    a[0, 100, 4] = 0.95
    a[0, 100, 5:] = (0.1, 0.9, 0.1)
    return a


def _end2end() -> np.ndarray:
    a = np.zeros((1, 10, 6), np.float32)
    a[0, 0] = (20, 16, 44, 32, 0.88, 2)         # x1, y1, x2, y2, score, class
    return a


def _yolo_grid() -> np.ndarray:
    """tiny-YOLOv2 風格：1 個輸出 [1, 5*(5+3), 4, 4]，stride 16，anchors 用內建 tiny_yolov2_voc。"""
    g = np.zeros((5, 8, 4, 4), np.float32)
    g[:, 4] = -10.0                              # 其餘格點 objectness 很低
    g[0, 4, 1, 2] = 5.0                          # anchor 0、y=1、x=2
    g[0, 5:, 1, 2] = (0.0, 4.0, 0.0)
    return g.reshape(1, 40, 4, 4)


def _yolox_raw() -> np.ndarray:
    a = np.zeros((1, 84, 8), np.float32)
    a[0, :, 0:2] = -0.3                          # 未解碼的 xy 偏移常為負
    a[0, :, 2:4] = -1.0
    a[0, :, 4] = 0.01
    a[0, :, 5:] = 0.01
    a[0, 28, :4] = (0.5, 0.5, np.log(2.0), np.log(2.0))   # stride 8 的格點 (x=4, y=3)
    a[0, 28, 4] = 0.9
    a[0, 28, 5:] = (0.05, 0.05, 0.9)
    return a


def _classify() -> np.ndarray:
    return np.array([[0.1, 0.2, 4.0, 0.3, 0.1]], np.float32)


def _ssd() -> list[np.ndarray]:
    boxes = np.zeros((1, 10, 4), np.float32)
    boxes[0, 0] = (0.25, 0.3125, 0.5, 0.75)      # ymin, xmin, ymax, xmax（0–1）
    classes = np.zeros((1, 10), np.float32)
    classes[0, 0] = 1
    scores = np.zeros((1, 10), np.float32)
    scores[0, 0] = 0.8
    return [boxes, classes, scores, np.array([1.0], np.float32)]


# 名稱 → (輸出清單, 輸出名稱, 期待結果)
SPECS: dict[str, tuple[list[np.ndarray], list[str], Expect]] = {
    "yolo_t": ([_yolo_t()], ["output0"], Expect("detect", 1, (20, 16, 44, 32), 0.85, "yolo_t")),
    "yolo_flat": ([_yolo_flat()], ["output"], Expect("detect", 1, (20, 16, 44, 32), 0.8, "yolo_flat")),
    "end2end": ([_end2end()], ["output0"], Expect("detect", 2, (20, 16, 44, 32), 0.85, "end2end")),
    "yolo_grid": ([_yolo_grid()], ["grid"], Expect("detect", 1, (31.36, 14.48, 48.64, 33.52), 0.9, "yolo_grid")),
    "yolox_raw": ([_yolox_raw()], ["output"], Expect("detect", 2, (28, 20, 44, 36), 0.8, "yolox_raw")),
    "classify": ([_classify()], ["logits"], Expect("classify", 2, None, 0.9, "classify")),
    "ssd": (_ssd(), ["TFLite_Detection_PostProcess", "TFLite_Detection_PostProcess:1",
                     "TFLite_Detection_PostProcess:2", "TFLite_Detection_PostProcess:3"],
            Expect("detect", 1, (20, 16, 48, 32), 0.7, "tflite_ssd")),
}
NAMES = tuple(SPECS)


def build_model(name: str) -> bytes:
    """產生名稱對應的 ONNX 模型位元組。"""
    import onnx
    from onnx import TensorProto, helper
    from onnx import numpy_helper as nh

    outputs, out_names, _ = SPECS[name]
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, INPUT_SIZE, INPUT_SIZE])
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=0),
        helper.make_node("Mul", ["mean", "zero"], ["zeroed"]),
    ]
    inits = [nh.from_array(np.zeros((), np.float32), "zero")]
    outs = []
    for i, (arr, oname) in enumerate(zip(outputs, out_names, strict=True)):
        cname = f"const{i}"
        inits.append(nh.from_array(np.ascontiguousarray(arr, dtype=np.float32), cname))
        nodes.append(helper.make_node("Add", [cname, "zeroed"], [oname]))
        outs.append(helper.make_tensor_value_info(oname, TensorProto.FLOAT, list(arr.shape)))
    graph = helper.make_graph(nodes, f"fake_{name}", [inp], outs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8                          # 舊版 onnxruntime 也讀得懂
    onnx.checker.check_model(model)
    return model.SerializeToString()


def generate_all(out_dir: str | Path | None = None) -> dict[str, Path]:
    """把全部假模型寫到 out_dir（預設 tests/fixtures/generated/），回傳 {名稱: 路徑}。"""
    folder = Path(out_dir) if out_dir else GENERATED_DIR
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in NAMES:
        p = folder / f"fake_{name}.onnx"
        p.write_bytes(build_model(name))
        paths[name] = p
    return paths


if __name__ == "__main__":
    for n, p in generate_all().items():
        print(n, p)
