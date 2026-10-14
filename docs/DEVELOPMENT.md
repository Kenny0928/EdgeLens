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
| `edgelens/zoo.py` | 模型庫：可下載的模型與示範照片清單、下載／sha256 驗證／安裝、離線包匯入 |

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

## 套件版本釘選（constraints）

`requirements/*.txt` 和 `pyproject.toml` 只寫「可接受的範圍」（例如 `numpy>=1.26,<3`），有幾個甚至沒有上限。
如果直接照範圍安裝，全班電腦會裝到「安裝當天的最新版」：一來小套件（kflash、pyserial 這類只有一兩位維護者的）
被入侵或出包時整間教室一起中，二來下學期新版一出就可能裝壞。所以 `requirements/constraints.txt` 把每個
套件（含 onnxruntime、PySide6 的相依套件）釘成**測試過的精確版本**，bootstrap 每一次 `pip install`
與 `pip download`（含 `--make-wheelhouse`、tflite-runtime 備援）都會帶 `-c requirements/constraints.txt`，
CI 也一樣。範圍檔照舊：它說「可以用什麼」，constraints 說「實際裝什麼」。

- constraints 只限制版本，不會多裝東西；改了它，`compute_hash` 就變，每台電腦下次啟動會自動重裝。
- 新版 numpy／onnxruntime 不再支援舊 Python，所以用環境標記分開，例如 `numpy==2.2.6; python_version < "3.12"`。
- macOS 的 wheel 有最低系統版本（opencv 4.14 要 macOS 13、onnxruntime 1.30 要 macOS 14），舊 Mac 會找不到檔案。
  所以 macOS 上這幾個（opencv、onnxruntime）寫成 `>=下限,<=上限`，pip 會挑「這台 Mac 裝得起來的最新版」；其他系統維持 `==`。
- 新增套件到 `requirements/` 時一定要同步加進 constraints（`test_constraints_cover_every_top_level_requirement` 會檢查）。

### 怎麼升版

1. 在新的暫時環境裝最新版、跑完整測試與 `--selftest`，確定沒問題後，把版本改進 `constraints.txt`。
2. 用 pip 的解析器在**不安裝**的情況下，把「Python 3.10–3.13 × Windows／macOS（arm64、x86_64，系統版本 12–14）／
   Linux（x86_64、aarch64，glibc 2.17／2.28／2.31／2.35 或 2.36）」每一種都解析一遍，**有 `-c` 和沒有 `-c` 都要跑**。
   規則：沒有 `-c` 解析得出來的，加了 `-c` 也一定要解析得出來（選配套件在某些平台本來就沒有 wheel，那是正常的）。
3. 注意 pip 的 `--python-version` **不會**改環境標記（`python_version`、`sys_platform` 還是用目前這台的），
   所以要用下面這個小包裝檔把標記也換掉，否則上面那些 `; python_version < ...` 的行會被誤判：

```python
# pin_check.py（放哪都可以）：用法 T_PY=3.10 T_OS=win python pin_check.py install --dry-run ...
# 要模擬 x86_64／aarch64 時另外加 T_MACHINE=x86_64（或 aarch64）
import os, sys
from pip._vendor.packaging import markers
SYS, SYSTEM, MACHINE, OS = {"win": ("win32", "Windows", "AMD64", "nt"), "mac": ("darwin", "Darwin", "arm64", "posix"),
                            "linux": ("linux", "Linux", "x86_64", "posix")}[os.environ["T_OS"]]
MACHINE = os.environ.get("T_MACHINE", MACHINE)
markers.default_environment = lambda: {
    "implementation_name": "cpython", "implementation_version": os.environ["T_PY"] + ".0", "os_name": OS,
    "platform_machine": MACHINE, "platform_release": "", "platform_system": SYSTEM, "platform_version": "",
    "python_full_version": os.environ["T_PY"] + ".0", "platform_python_implementation": "CPython",
    "python_version": os.environ["T_PY"], "sys_platform": SYS, "extra": ""}
from pip._internal.cli.main import main
sys.exit(main(sys.argv[1:]))
```

```bash
# 例：Python 3.10、Windows、必要套件；把 -c 那兩個字拿掉就是「沒有 constraints」的對照組
T_PY=3.10 T_OS=win python pin_check.py install --dry-run --ignore-installed --only-binary=:all: \
  --python-version 3.10 --implementation cp --abi cp310 --platform win_amd64 --target /tmp/pin-test \
  -r requirements/base.txt -c requirements/constraints.txt --report /tmp/pin-report.json
# macOS：--platform macosx_13_0_arm64（x86_64 用 macosx_13_0_x86_64 並加 T_MACHINE=x86_64；T_OS=mac）。
#   pip 會自動展開成更舊的 macOS 版本，所以只要各測 12、13、14 就夠
# 每個 requirements/extra-*.txt 都要各跑一次；tflite 備援則是一個只寫 tflite-runtime 的暫存檔
```

**Linux 一定要給完整的 tag 清單。** pip 的 `--platform manylinux_2_28_x86_64` **不會**往下展開成
`manylinux_2_27`…`manylinux_2_17`，只寫一個 tag 會看不到只標 `manylinux_2_17` 的 wheel
（ai-edge-litert 1.x 就是），結果「沒有 -c 解析得出來、有 -c 卻失敗」的問題在矩陣裡完全看不到。
用這段產生旗標（glibc 2.35 ≈ Ubuntu 22.04、2.36 ≈ Raspberry Pi OS bookworm、2.31 ≈ Ubuntu 20.04，
另外請加測 2.28 與 2.17）：

```bash
linux_tags() {  # 用法：linux_tags x86_64 35   或   linux_tags aarch64 36
  for v in $(seq "$2" -1 17); do printf -- '--platform manylinux_2_%s_%s ' "$v" "$1"; done
  printf -- '--platform manylinux2014_%s --platform manylinux2010_%s --platform manylinux1_%s --platform linux_%s' \
    "$1" "$1" "$1" "$1"
}
T_PY=3.12 T_OS=linux python pin_check.py install --dry-run --ignore-installed --only-binary=:all: \
  --python-version 3.12 --implementation cp --abi cp312 $(linux_tags x86_64 35) --target /tmp/pin-test \
  -r requirements/extra-tflite.txt -c requirements/constraints.txt
```

Linux 上 PySide6／shiboken6、ai-edge-litert、numpy（3.12 以上）在舊 glibc 只有舊版 wheel（例如 aarch64 的 PySide6 在
glibc 2.39 以下只有 6.8.0.2；ai-edge-litert 2.x 在 glibc 2.17 沒有 wheel），所以這幾個在 Linux 是 `>=下限,<=上限`
的範圍行，不是 `==`。

升版完成後，`--report` 的結果裡每個套件的版本都應該等於 constraints 裡的版本（macOS 與 Linux 的範圍行除外）。

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
| 5 | 影像處理步驟、預設組合、邊緣裝置模擬、取色 | `processing/steps/`、`processing/presets/`、`processing/library.py`、`processing/colorpick.py` |
| 6a | 前處理、NMS、YOLO 解碼器、自動判斷 | `inference/preprocess.py`、`inference/nms.py`、`inference/autodetect.py`、`inference/decoders/` |
| 6b | 推論後端、runner、測試模型 | `inference/backends/`、`inference/runner.py`、`tests/fixtures/make_onnx.py`、`tools/fetch_test_models.py`、`docs/MODEL_SUPPORT.md` |
| 7 | 資料集、指標 | `dataset/`、`metrics/` |
| 8a／8b／9 | 圖形介面 | `gui/` |
| 10 | 文件 | `README*.md`、`docs/` |

## 目前功能總覽

以下列出目前程式的主要入口與模組關係；操作說明見[學生手冊](STUDENT_GUIDE.md)、[教師指南](TEACHER_GUIDE.md)與[外掛教學](PLUGINS.md)。模型與比較面板的操作章節保留給後續整合補充。

| 檔案／模組 | 功能與關係 |
|---|---|
| [bootstrap.py](../bootstrap.py)、`run_windows.bat`／`run_macos.command`／`run_linux.sh` | 尋找 Python 3.10–3.13、建立快取環境、安裝必要及選配套件；提供 wheelhouse 準備與離線安裝，最後啟動套件 |
| [__main__.py](../edgelens/__main__.py)、[gui/app.py](../edgelens/gui/app.py) | 命令列、自我檢查、安全模式；建立 Qt app、選工作區、探索內建元件、載入外掛與初始專案 |
| [core/project.py](../edgelens/core/project.py)、[core/settings.py](../edgelens/core/settings.py)、[core/paths.py](../edgelens/core/paths.py) | 工作區／專案／類別、本機設定、跨平台路徑與原子寫檔；`presets/projects/*.json` 提供空白及兩個專題範本 |
| [core/registry.py](../edgelens/core/registry.py)、[core/controls.py](../edgelens/core/controls.py) | 四種註冊表與外掛載入；ControlSpec 驅動 GUI 參數控制項與驗證 |
| [sources/base.py](../edgelens/sources/base.py)、`sources/{synthetic,file,webcam,mjpeg,rtsp,esp32cam}.py`、`sources/k210/` | 統一 FrameSource；本機、檔案、網路與板子來源；測試圖案、ESP32-CAM 與 K210 通訊模擬供無硬體驗證 |
| [processing/pipeline.py](../edgelens/processing/pipeline.py)、`processing/steps/`、[processing/library.py](../edgelens/processing/library.py)、[processing/colorpick.py](../edgelens/processing/colorpick.py) | 依序執行步驟、保存流程、預設組合與吸管取色；StepContext 傳遞數值、疊圖和幾何座標關係 |
| [inference/base.py](../edgelens/inference/base.py)、[inference/runner.py](../edgelens/inference/runner.py)、`inference/backends/`、`inference/decoders/` | 前處理 → 後端 → 解碼 → 門檻／NMS → 原始座標結果；ModelCard 指定模型、類別與輸入輸出規格，autodetect 協助判斷格式 |
| [gui/main_window.py](../edgelens/gui/main_window.py)、[gui/workers.py](../edgelens/gui/workers.py)、[gui/video_view.py](../edgelens/gui/video_view.py) | 主視窗、每來源工作執行緒與畫面；設定透過佇列交給 worker，畫面及處理結果送回 GUI |
| `gui/{workspace_dialog,project_dialog,source_dialog,control_panel}.py` | 工作區選擇、專案與類別編輯、來源探索／手動加入、自動產生參數欄位和確認狀態 |
| `gui/panels/{camera,pipeline,capture,dataset,bench,k210}.py` | 相機設定、影像處理、拍照、資料集、效能／錄影及板子工具；共同面板介面由 `gui/panels/base.py` 定義 |
| [dataset/writer.py](../edgelens/dataset/writer.py)、`dataset/{stats,dedupe,split,export}.py` | 原始與處理影像、metadata、改類別、刪除復原、統計、相似照片、分組切分與 ZIP；由拍照與資料集 GUI 共用 |
| [gui/dataset_browser.py](../edgelens/gui/dataset_browser.py)、[gui/export_dialog.py](../edgelens/gui/export_dialog.py)、[gui/capture_tools.py](../edgelens/gui/capture_tools.py) | 縮圖瀏覽、篩選、資料切分／匯出；品質門檻、近期相似提醒與靜止判斷支援拍照流程 |
| [zoo.py](../edgelens/zoo.py)、[gui/zoo_download.py](../edgelens/gui/zoo_download.py)、[gui/class_filter_dialog.py](../edgelens/gui/class_filter_dialog.py)、[tools/export_objects365.py](../tools/export_objects365.py) | 模型庫：清單（網址、大小、sha256）在 `zoo.CATALOG`，zip 由匯出腳本製作後上傳 GitHub Release `models-v1`；安裝在快取的 `zoo/<id>/`，示範照片由 `FileSource.discover()` 列出；模型面板的類別篩選寫入 `ModelCard.class_filter` |
| `metrics/{rolling,quality,csvlog,sweep,evaluate,detection_rate}.py`、[gui/recorder.py](../edgelens/gui/recorder.py) | FPS／延遲、品質、CSV、參數掃描、模型評估及偵測率；Recorder 保存影片與對應數據 |
| [diagnostics_report.py](../edgelens/diagnostics_report.py)、[gui/diagnostics.py](../edgelens/gui/diagnostics.py)、[selftest.py](../edgelens/selftest.py) | 系統／套件／註冊表／外掛匯入失敗、設定遮蔽、日誌與自我檢查；GUI 可匯出診斷 ZIP |
| [i18n/__init__.py](../edgelens/i18n/__init__.py)、`i18n/locales/{zh_TW,en}/*.json` | 合併翻譯表；`tr` 處理內建文字，`tr_or` 支援指定備援文字；顯示備援仍依各 GUI 呼叫處決定 |
| [tests/unit/test_example_plugins.py](../tests/unit/test_example_plugins.py)、[docs/examples/plugins](examples/plugins/) | 三個無硬體、無翻譯檔外掛；透過 load_plugins 驗證，重用內建步驟執行／序列化／reset 契約，名稱翻譯要求改驗備援行為 |

資料在 `FrameSource → Frame → Pipeline → ProcessResult` 後送往顯示、拍照與效能記錄；推論經 `InferenceRunner` 產生 `InferenceResult`，供疊圖、樣本中繼資料與評估使用。工作區保存專案與共用外掛，本機設定保存連線及目前操作狀態；交接專案時，要把處理流程另存到專案的 `pipelines/`，並另附所需外掛。
