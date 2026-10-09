# EdgeLens 模型支援說明

給老師與學生：你手上的模型能不能用、怎麼匯出、自動判斷錯了怎麼辦。

**一句話版本**：把模型檔（`.onnx`、`.tflite`…）丟進 EdgeLens，大多數情況它會自己看出是哪一種 YOLO 或分類模型。
看不出來或看錯時，在模型面板選正確的格式，或寫一個 `.card.json` 說明檔。電腦不需要安裝 PyTorch 或 TensorFlow。

## 1. 支援的模型檔與後端

| 副檔名 | 後端 | 需要的套件 | 說明 |
|---|---|---|---|
| `.onnx` | ONNX Runtime（優先） | onnxruntime（內建安裝） | 最推薦。跨平台結果一致，預設用 CPU |
| `.onnx` | OpenCV DNN（備用） | 無（OpenCV 內建） | ONNX Runtime 不能用或讀不懂時改用 |
| `.tflite` | TensorFlow Lite | ai_edge_litert 或 tflite_runtime | 量化模型（uint8／int8）也支援；沒裝時可改把模型轉成 ONNX |
| `.weights` + `.cfg` | OpenCV DNN | 無 | Darknet 的 YOLOv3／v4／v4-tiny，兩個檔案放同一個資料夾，同名最好 |
| `.pb` + `.pbtxt`、`.caffemodel` + `.prototxt` | OpenCV DNN | 無 | 舊式格式，設定檔寫在 `extra_files.config` |
| `.xml` | Haar 級聯 | 無 | OpenCV 傳統臉部偵測，EdgeLens 內附正臉、側臉、眼睛、笑臉、全身等 |
| `.kmodel` | 開發板（K210） | 無 | 模型在板子上跑，電腦只顯示板子送回來的結果 |

自動挑選後端的順序是：能用且優先順序最高的。模型面板的後端下拉選單會列出全部相關的後端，不能用的灰色顯示並說明原因。

## 2. 各 YOLO 版本的支援狀態

「以真實模型實測」表示用真正的官方模型跑過測試照片，並確認框的位置與類別正確（測試：`tests/unit/test_real_models.py`）。
「僅以合成資料驗證」表示解碼規則照該版本的輸出格式寫、用人工製作的輸出驗證過，但沒有拿到官方模型實測。

| 模型 | 狀態 | 實測用的模型 | 輸出格式／解碼器 |
|---|---|---|---|
| YOLOv5 | 以真實模型實測 | yolov5n、yolov5s（ultralytics 官方 v7.0） | `[1, 25200, 85]` → yolo_flat |
| YOLOv8 | 以真實模型實測 | yolov8n（ultralytics 匯出） | `[1, 84, 8400]` → yolo_t |
| YOLOv10 | 以真實模型實測 | yolov10n（onnx-community 轉出） | `[1, 300, 6]` → end2end（已含 NMS） |
| YOLOX | 以真實模型實測 | yolox_nano、yolox_s（官方，未解碼輸出） | `[1, N, 85]` → yolox_raw |
| YOLOv2（tiny） | 以真實模型實測 | Tiny YOLOv2 VOC（ONNX Model Zoo） | `[1, 125, 13, 13]` → yolo_grid |
| YOLOv4-tiny（Darknet） | 以真實模型實測 | AlexeyAB yolov4-tiny.cfg + weights | OpenCV 解碼後合併 → yolo_flat |
| TFLite SSD MobileNet | 以真實模型實測 | TensorFlow 官方 coco_ssd_mobilenet_v1 量化版 | 4 個輸出 → tflite_ssd |
| 分類（MobileNet） | 以真實模型實測 | MobileNetV2（ONNX Zoo）、MobileNet v1 量化 TFLite | `[1, 1000]` → classify |
| YOLOv6 | 僅以合成資料驗證 | — | `[1, N, 85]`，N = 格點數 → yolo_flat |
| YOLOv7 | 僅以合成資料驗證 | — | 與 YOLOv5 同格式 → yolo_flat |
| YOLOv9、YOLO11、YOLO12 | 僅以合成資料驗證 | —（輸出格式與 YOLOv8 相同） | yolo_t |
| YOLOv3、YOLOv4（完整版 ONNX） | 僅以合成資料驗證 | — | 2–3 個格點輸出 → yolo_grid（ONNX Model Zoo 的 yolov4.onnx 約 245 MB，測試腳本以 `--all` 才下載） |
| K210 kmodel（電腦端解讀） | 僅以合成資料驗證 | — | 電腦不解 kmodel；板上推論見第 6 節 |

重現方式：`python tools/fetch_test_models.py` 下載模型（GitHub 下載有時很慢，可重跑），再執行 `pytest -m network tests/unit/test_real_models.py`；沒下載的模型測試會自動略過。

## 3. 自動判斷規則

EdgeLens 看模型輸出的形狀（以及跑一次的實際數值）來猜格式，模型面板會列出「可能是…」與判斷理由；最佳猜測分數太低、
或前兩名很接近時，會標示「請確認」。

| 輸出形狀 | 判斷為 | 常見模型 |
|---|---|---|
| `[1, C]` | classify（分類） | MobileNet、Teachable Machine |
| `[1, K, 6 或 7]`，K ≤ 300 | end2end | YOLOv10、匯出時含 NMS 的 v5／v7／v8 |
| `[1, 4+C, N]` 或 `[1, N, 4+C]` | yolo_t | YOLOv8、v9、v11、v12 |
| `[1, N, 5+C]`，N = 3 × 格點數 | yolo_flat | YOLOv5、YOLOv7 |
| `[1, N, 5+C]`，N = 格點數 | yolo_flat 或 yolox_raw | YOLOv6／已解碼 YOLOX 或未解碼 YOLOX，依數值範圍區分，分不出來就請你選 |
| 單一輸出 `[1, A×(5+C), H, W]` | yolo_grid | YOLOv2、tiny-YOLOv2、K210 |
| 2–3 個格點輸出 | yolo_grid | YOLOv3、YOLOv4（含 tiny） |
| 4 個輸出（框、類別、分數、數量） | tflite_ssd | TFLite SSD MobileNet |

特例：Darknet 的 `.weights` 經 OpenCV 讀入後，輸出已經是解碼過的 `[N, 5+C]`（中心與寬高為 0–1 比例），EdgeLens 會把各層合併後直接當 yolo_flat 處理。

類別名稱的來源順序：說明檔的 `labels` → 模型旁邊的 `labels.txt`／`<模型名>.names`／`coco.names` 等 → 模型內嵌的名稱
（ONNX 的 `names`、TFLite 附的 labelmap）→ 都沒有就顯示數字。類別數量和模型輸出對不上時，EdgeLens 會明確告訴你，不會默默亂標。

## 4. 判斷錯誤時：寫 `.card.json`

說明檔放在模型旁邊，檔名是「模型檔名去掉副檔名 + `.card.json`」（例如 `yolov8n.card.json`）。
也可以在模型面板調好後按「儲存說明檔」，不必自己寫。

### 範例 1：YOLOv8 ONNX（自訂類別，例如只認三種物件）

```json
{
  "version": 1,
  "name": "我的三種零件",
  "file": "best.onnx",
  "decoder": "yolo_t",
  "labels": ["螺絲", "螺帽", "墊片"],
  "input": {"size": [640, 640], "color": "RGB", "resize": "letterbox"},
  "thresholds": {"conf": 0.3, "nms": 0.45, "max_det": 50}
}
```

### 範例 2：Darknet yolov4-tiny

```json
{
  "version": 1,
  "name": "YOLOv4-tiny",
  "file": "yolov4-tiny.weights",
  "backend": "cvdnn",
  "extra_files": {"config": "yolov4-tiny.cfg"},
  "labels_file": "coco.names",
  "input": {"resize": "stretch"}
}
```

Darknet 訓練時是直接拉伸影像，不補邊，所以用 `stretch`。輸入大小從 `.cfg` 的 `width`／`height` 讀取。

### 範例 3：TFLite SSD MobileNet

```json
{
  "version": 1,
  "name": "COCO SSD",
  "file": "detect.tflite",
  "decoder": "tflite_ssd",
  "decoder_params": {"class_offset": 0},
  "labels_file": "labels.txt",
  "input": {"resize": "stretch"}
}
```

若類別名稱檔第一行是 `???`（背景），名稱會差一格：刪掉那一行，或把 `class_offset` 設成 1。
量化的分類模型（uint8）如果 MobileNet 輸入範圍是 -1 到 1，要寫 `"scale": 0.00784, "mean": [1, 1, 1]`（即 (像素 − 127.5) ÷ 127.5）。

### 範例 4：K210 kmodel（板上推論）

```json
{
  "version": 1,
  "name": "K210 臉部偵測",
  "file": "face.kmodel",
  "labels": ["face"],
  "device": {"anchors": [1.889, 2.5245, 2.9465, 3.94056, 3.99987, 5.3658, 5.155437, 6.92275, 6.718375, 9.01025]},
  "thresholds": {"conf": 0.5}
}
```

kmodel 在板子上執行，電腦端只套用信心門檻與類別名稱。

### 其他常用欄位

- 輸入顏色順序 `color`：`RGB`／`BGR`／`GRAY`。YOLOX 官方模型要 `"color": "BGR", "scale": 1.0`（0–255 不正規化）；Tiny YOLOv2（ONNX Zoo）要 `"scale": 1.0`。
- 分類模型的標準化：`"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]`（MobileNetV2 ONNX）。
- 自己指定 YOLOv3／v4 的 anchors：`"decoder": "yolo_grid", "decoder_params": {"anchors_preset": "yolov4_coco"}`，或直接寫 `anchors`。
- Haar 偵測：`"decoder_params": {"scale_factor": 1.1, "min_neighbors": 5, "min_size": [30, 30]}`；`min_neighbors` 越大越嚴格、誤判越少。Haar 沒有真正的信心分數，信心門檻對它不起作用，請調 `min_neighbors`。

## 5. 匯出建議（學生電腦不需要 PyTorch）

- **YOLOv5／v8／v9／v10／11／12（ultralytics）**：在訓練用的電腦（或 Colab）執行 `yolo export model=best.pt format=onnx imgsz=640`，
  把 `best.onnx` 帶到 EdgeLens 的電腦即可。類別名稱會一起寫進 ONNX，通常不用另外準備。想要含 NMS 的版本才加 `nms=True`。
- **YOLOX**：官方 release 的 `.onnx`（不是 `.pth`）；若自己匯出，請不要加 `--decode_in_inference`（EdgeLens 能處理兩種）。
- **Teachable Machine**：選「Tensorflow Lite」匯出（Quantized 或 Floating point 都可以），會得到 `model.tflite` 與 `labels.txt`，放在一起。
- **Darknet 專案**：`.weights`、`.cfg`、`.names` 三個放同資料夾。
- **TFLite**：沒有安裝 TFLite 執行環境時，可以改用 ONNX。用 OpenCV 後端讀量化（uint8）的 TFLite 通常不行，請用 TensorFlow Lite 後端。

## 6. K210 板上推論的限制

K210 的 MaixPy v1 韌體只支援 **YOLOv2 與分類** 兩類模型（KPU 的限制）。YOLOv3／v4／v5／v8 等請在電腦端推論（上面的 ONNX／TFLite），
K210 只負責傳影像。板上推論的結果標示為「開發板」算的，延遲、準確度與電腦端不同，比較時請分開看。
