# EdgeLens 開發說明

給要修改或擴充 EdgeLens 的人（老師、學生、協作者）。使用說明請看 `README.zh-TW.md` 與 `docs/` 其他文件。

## 架構

```
相機來源 FrameSource ──Frame──▶ 影像處理 Pipeline ──▶ 畫面顯示／拍照存檔
   (sources/)                     (processing/)              (gui/、dataset/)
                                        │
                                        ▼
                              推論 InferenceRunner ──▶ InferenceResult
                              (inference/：Backend + Decoder)
```

| 資料夾 | 內容 |
|---|---|
| `edgelens/core/` | 共用型別：`Frame`、`ControlSpec`、`InferenceResult`、`ModelCard`、`Project`、註冊表、路徑、設定 |
| `edgelens/sources/` | 相機來源：合成、檔案、webcam、MJPEG／RTSP、ESP32-CAM、K210 |
| `edgelens/processing/` | `Pipeline` 與處理步驟（`steps/`）、預設組合（`presets/`） |
| `edgelens/inference/` | 前處理、NMS、後端（`backends/`）、輸出解碼器（`decoders/`）、runner |
| `edgelens/dataset/` | 樣本存檔、切分、匯出、統計、相似照片 |
| `edgelens/metrics/` | FPS／延遲統計、CSV、影像品質、偵測率、模型評估 |
| `edgelens/gui/` | PySide6 介面 |
| `edgelens/i18n/` | 翻譯檔 `locales/<語言>/*.json` |

### 擴充點（註冊表）

四種元件都用 decorator 註冊，GUI 會自動列出，不用改 GUI 程式：

| 元件 | 基底類別 | decorator | 範例 |
|---|---|---|---|
| 相機來源 | `sources.base.FrameSource` | `@register_source` | `sources/synthetic.py` |
| 處理步驟 | `processing.pipeline.ProcessingStep` | `@register_step` | `tests/unit/test_core.py` 的 `Invert` |
| 推論後端 | `inference.base.Backend` | `@register_backend` | — |
| 輸出解碼器 | `inference.base.Decoder` | `@register_decoder` | `inference/decoders/classify.py` |

**外掛**：把 `.py` 放到工作區的 `plugins/` 資料夾，用同樣的 decorator 註冊，重新開啟 EdgeLens
就會出現。學生可以用這個方式寫自己的處理步驟，不用改 EdgeLens 本身。

### 工作區與專案

見 `edgelens/core/project.py` 開頭的說明。一個工作區有多個專案，每個專案有自己的類別、
資料集、模型與處理流程。類別資料夾用英文 ID，中文是顯示名稱。

### 執行緒

- 每個相機一條 `SourceWorker` 執行緒：開啟、讀取、處理、關閉都在這條執行緒。`FrameSource`
  與 `Pipeline` 物件不跨執行緒使用，所以不需要加鎖。
- GUI 修改參數或處理流程時，把「新的值」送進 worker 的佇列，由 worker 套用。
- 每組（相機 × 模型）一條 `InferenceWorker` 執行緒，只處理最新的畫面。

## 開發規則

1. **Python 3.10–3.13 都要能跑**：每個檔案開頭加 `from __future__ import annotations`；不要用
   `typing.Self`、`enum.StrEnum`、`tomllib`、`except*` 等 3.11 以後才有的功能。
2. **選配套件延遲匯入**：`onnxruntime`、`ai_edge_litert`、`tflite_runtime`、`kflash`、`onnx`
   只能在函式裡匯入，模組最上面不可以 import，否則沒裝的電腦連註冊都會失敗。用類別屬性
   `requires = ("onnxruntime",)` 宣告。
3. **自給自足**：不可引用 repo 以外的路徑；套件內附檔案用 `core.paths.resource_path()` 取得。
4. **存檔**：一律用 `core.paths.atomic_write_bytes/text`。影像用 `cv2.imencode` 後寫入位元組，
   不要用 `cv2.imwrite`／`cv2.imread`（Windows 中文路徑會失敗），讀圖用
   `cv2.imdecode(np.fromfile(path, np.uint8), ...)`。
5. **給使用者看的文字都要翻譯**：用 `edgelens.i18n.tr("鍵")`，中英文都要加。每個工作包用自己的
   翻譯檔（例如 `locales/zh_TW/sources_web.json` 和 `locales/en/sources_web.json`），兩個語言
   的鍵必須一致。例外與日誌可以只寫中文。
6. **錯誤訊息寫給國中生看**：說明發生什麼事、可以怎麼做（例如「找不到相機。請確認 USB
   線有接好，或到 系統設定 → 隱私權 → 相機 允許終端機」），不要只丟英文例外。
7. **不新增必要套件**：必要套件只有 `requirements/base.txt` 那幾個。需要新套件請先討論。
8. **測試**：每個模組都要有 `tests/unit/test_<模組>.py`；新增的註冊元件會自動被
   `tests/contracts/` 檢查，必須全部通過。需要硬體的測試加 `@pytest.mark.hardware`。
9. **程式風格**：註解與 docstring 用繁體中文，簡潔說明「為什麼」；`ruff check` 必須通過。

## 指令

```bash
# 開發用環境（放在 iCloud／OneDrive 以外的地方）
python3 -m venv ~/Library/Caches/EdgeLens/devvenv
~/Library/Caches/EdgeLens/devvenv/bin/pip install -r requirements/dev.txt

PY=~/Library/Caches/EdgeLens/devvenv/bin/python
$PY -m pytest -q                      # 全部測試
$PY -m pytest -q tests/contracts      # 只跑介面檢查
$PY -m pytest -q -m "not hardware"    # CI 跑的範圍
$PY -m ruff check edgelens tests      # 程式風格
$PY -m edgelens --selftest            # 自我檢查
$PY -m edgelens                       # 開啟介面
```

## 工作包與檔案歸屬

開發時分成幾個工作包（WP）平行進行，每個工作包只改自己的檔案；共用的核心檔案
（`edgelens/core/`、`*/base.py`、`processing/pipeline.py`、`tests/contracts/`）由 WP0 負責，
需要修改請提出，不要直接改。

| WP | 內容 | 擁有的檔案 |
|---|---|---|
| 0 | 核心型別、註冊表、i18n、contract tests、CI | `edgelens/core/`、`*/base.py`、`processing/pipeline.py`、`sources/synthetic.py`、`inference/decoders/classify.py`、`selftest.py`、`__main__.py`、`tests/contracts/`、`tests/conftest.py` |
| 1 | 啟動腳本、bootstrap | `run_*`、`bootstrap.py`、`tests/unit/test_bootstrap.py` |
| 2 | 檔案、webcam 來源、相機列舉 | `sources/file.py`、`sources/webcam.py`、`sources/enumerate.py` |
| 3 | MJPEG、RTSP、ESP32-CAM | `sources/mjpeg.py`、`sources/rtsp.py`、`sources/esp32cam.py` |
| 4 | K210 | `sources/k210/` |
| 5 | 影像處理步驟、預設組合、邊緣裝置模擬 | `processing/steps/`、`processing/presets/` |
| 6a | 前處理、NMS、YOLO 解碼器、自動判斷 | `inference/preprocess.py`、`inference/nms.py`、`inference/autodetect.py`、`inference/decoders/` |
| 6b | 推論後端、runner、測試模型 | `inference/backends/`、`inference/runner.py`、`tests/fixtures/make_onnx.py`、`tools/fetch_test_models.py`、`docs/MODEL_SUPPORT.md` |
| 7 | 資料集、指標 | `dataset/`、`metrics/` |
| 8a／8b／9 | 圖形介面 | `gui/` |
| 10 | 文件 | `README*.md`、`docs/` |
