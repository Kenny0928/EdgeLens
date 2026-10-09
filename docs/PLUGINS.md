# 用 Python 寫 EdgeLens 外掛

外掛是放在指定資料夾的 Python 程式。EdgeLens 啟動時會匯入它，讓它登記新的處理步驟、相機來源、推論後端或解碼器。先會寫函式和類別，再從影像處理步驟開始。

## 1. 放在哪裡？

外掛放在**目前工作區**的 `plugins/`，不是程式套件資料夾，也不是某個專案裡的 `plugins/`。

```text
我的工作區/
  plugins/
    invert_colors.py
    count_bright_pixels.py
    pixelate.py
  projects/
    我的專案/
      project.json
```

工作區位置可在「診斷資訊」的「完整報告」確認。路徑規則見 [Workspace.plugins_dir](../edgelens/core/project.py)，載入方式見 [load_plugins](../edgelens/core/registry.py)。載入器只讀該資料夾第一層的 `.py`，跳過底線 `_` 開頭的檔案，不會遞迴找子資料夾。

1. 關閉 EdgeLens，複製[範例負片](examples/plugins/invert_colors.py)到工作區 `plugins/`。
2. 重新開啟 EdgeLens，新增「測試圖案（不需相機）」。
3. 在「影像處理」按「新增步驟」→「自訂與外掛」，選 `custom.invert_colors`。
4. 勾選「畫面顯示處理後的影像（取消勾選就看原始畫面）」。看到負片，代表載入與執行成功。步驟旁會標示「外掛」。

新增、修改、移除外掛後都要**結束並重新開啟程式**。目前沒有給學生使用的即時重載按鈕；切換專案也不會重新匯入檔案。

## 2. 最小處理步驟

完整且有註解的程式在 [invert_colors.py](examples/plugins/invert_colors.py)：

```python
from __future__ import annotations

from edgelens.processing.pipeline import ProcessingStep, register_step


@register_step
class InvertColors(ProcessingStep):
    kind = "custom.invert_colors"
    category = "custom"
    display_name_key = ""

    def apply(self, image, ctx):
        return 255 - image
```

- `ProcessingStep` 是共同基底。`@register_step` 在匯入時把類別加入註冊表。
- `kind` 是穩定的識別名稱，也是流程 JSON 使用的名稱。自己寫的步驟用 `custom.xxx`，限小寫英數字、底線與點；不要和其他外掛重複。改掉它會讓舊流程找不到步驟。
- `category = "custom"` 讓它進入「自訂與外掛」。
- `param_specs()` 預設回傳空清單；這個負片步驟不需要參數。
- `apply(image, ctx)` 每張畫面執行一次。`image` 是 NumPy `uint8` 陣列，彩色為 `H×W×3`、順序 **BGR**；灰階為 `H×W`。
- 必須回傳 `uint8` 的 `H×W` 或 `H×W×3` 影像。**不可修改輸入**，不要用 `image[:] = ...` 或直接在輸入上畫字。需要原地處理時，先 `image.copy()`。

這個減法會產生新陣列：0 變 255，255 變 0，而且彩色與灰階都能用。避免每張畫面讀大檔案、等待網路或存檔，否則相機會變慢。

## 3. 顯示名稱與不需要翻譯檔的參數

參數使用 [ControlSpec](../edgelens/core/controls.py) 宣告。參考 [pixelate.py](examples/plugins/pixelate.py)：

```python
from edgelens.core.controls import ControlSpec, ControlType

# 放在 ProcessingStep 子類別內
@classmethod
def param_specs(cls):
    return [ControlSpec(
        key="block_size", type=ControlType.INT,
        label="區塊大小", default=8, min=1, max=64, step=1,
        group="image",
    )]
```

`key` 是程式用的名稱；GUI 依 `ControlType.INT` 建立整數控制項。`self.params["block_size"]` 取得目前值。基底會驗證參數並限制數值範圍，`set_param()` 可用來修改已建立的步驟。不要直接繞過驗證修改 `self.params`。

**沒有翻譯檔時怎麼設定名稱：**

| 用途 | 怎麼寫 |
|---|---|
| 步驟名稱 | 類別屬性 `display_name = "馬賽克"` |
| 步驟說明（滑鼠停留時顯示） | 類別屬性 `description = "把畫面變成一格一格的色塊"` |
| 參數標籤 | `ControlSpec(label="區塊大小", ...)`，`label_key` 留空 |

`display_name_key`、`description_key`、`label_key` 是給有翻譯檔的內建元件用的，外掛不用設定。沒有設定 `display_name` 時，選單會顯示 `kind`（例如 `custom.pixelate`）。

實作依據：[component_name](../edgelens/i18n/__init__.py)、[參數面板](../edgelens/gui/control_panel.py)、[影像處理面板](../edgelens/gui/panels/pipeline.py)。

馬賽克先把影像縮小，再用最近鄰方法放大回原尺寸。區塊越大，細節越少；區塊大小是近似值，不能整除時邊緣大小會不同。

## 4. 數值與畫面疊字

[count_bright_pixels.py](examples/plugins/count_bright_pixels.py) 將彩色轉灰階，計算亮度大於等於 200 的像素比例。

```python
ratio = float((gray >= 200).mean())
ctx.data[f"{self.kind}.ratio"] = ratio
x, y = ctx.to_original(8, 20)
ctx.overlays.append(Overlay(
    kind="text",
    data={"xy": [x, y], "text": f">=200: {ratio:.1%}"},
    color="#FFFF00",
))
```

上面是 `apply` 裡的片段；完整範例已匯入 `Overlay` 並建立 `gray`。`ctx.data` 的鍵用 `kind` 當前綴，避免覆蓋別人的數值。值用一般 Python `float`、`int` 等可轉 JSON 的型別。比例 `0.5` 代表 50%；顯示文字只含數字與符號，沒有翻譯檔也能使用。

`ctx.overlays` 放顯示用圖形，不會把文字燒進分析影像；選擇含標註的錄影模式時才會將畫面標註一起錄下。`Overlay(kind="text")` 使用 `data={"xy": [x, y], "text": ...}`，顏色是 `#RRGGBB`。其他種類見 [Overlay](../edgelens/core/results.py)。

疊圖座標必須對應**原始畫面**。前面可能已有裁切、縮放，所以範例使用 `ctx.to_original()`。本範例不改影像尺寸；如果你另寫裁切或縮放步驟，要設 `changes_size = True` 並用 `ctx.push_transform()` 更新座標關係。需要記住前一張影像時設 `stateful = True` 並實作 `reset()`；不要把狀態放在所有相機共用的全域變數。

## 5. 三個範例與小挑戰

| 檔案 | 選單識別名稱 | 挑戰 |
|---|---|---|
| [invert_colors.py](examples/plugins/invert_colors.py) | `custom.invert_colors` | 畫一個 0、127、255 的陣列，預測負片結果 |
| [count_bright_pixels.py](examples/plugins/count_bright_pixels.py) | `custom.count_bright_pixels` | 把亮度 200 改成整數參數，測全黑、全白、半黑半白 |
| [pixelate.py](examples/plugins/pixelate.py) | `custom.pixelate` | 比較區塊大小 1、8、32，說明辨識細節如何改變 |

可以同時複製三個檔案；若想保留原版再做變體，檔名、類別名稱及 `kind` 都換成自己的唯一名稱。

## 6. 進階：解碼器是什麼？

模型後端算出張量，解碼器把張量解讀成分類分數或偵測框。它不是訓練模型，也不是像素處理步驟。

參考 [ClassifyDecoder](../edgelens/inference/decoders/classify.py) 與 [Decoder 介面](../edgelens/inference/base.py)：

1. 繼承 `Decoder`，用 `@register_decoder` 登記，設定唯一的 `kind = "custom.xxx"` 和 `task`（`classify` 或 `detect`）。
2. `match(outputs, card, samples=None)` 檢查輸出格式，回傳 `MatchResult`，分數介於 0–1。不符合就回 0，遇到未知或動態形狀不能直接拋例外；判定符合時提供原因。
3. `decode(outputs, ctx)` 接到已反量化的輸出。分類回傳 `DecodeOutput(task="classify", class_scores=...)`，分數陣列為一維。若輸出是 logits，需先轉成機率。
4. 偵測回傳 `boxes`、`scores`、`class_ids`。框為**模型輸入影像**的 `xyxy` 像素座標，不是原始相機座標；保留符合 `ctx.conf_threshold` 的候選。一般由 runner 做 NMS 和座標還原；模型已做 NMS 才設 `nms_done=True`。
5. 先用自己造的張量測試形狀、分數與座標，再用真模型核對。可透過模型說明檔的 `decoder` 指定註冊名稱；格式見[模型支援說明](MODEL_SUPPORT.md)。

解碼器的 GUI 顯示方式不應直接假設與處理步驟相同；這裡只介紹核心介面，不提供模型面板按鈕操作。

## 7. 除錯與測試

找不到外掛時，先看工作區是否選對、檔案是否真的以 `.py` 結尾、是否放在第一層、有沒有加 decorator，以及是否重開程式。`kind` 撞名也可能讓匯入失敗。

到「說明」→「診斷資訊…」→「完整報告」查看匯入失敗清單；文字報告中的 `IMPORT FAILED` 後面有檔案與原因。需要協助時按「匯出成 ZIP…」交給老師。若是執行 `apply` 才失敗，處理清單會有警示，流程會跳過出錯步驟；檢查回傳型別與陣列形狀，先用小陣列重現。

外掛讓程式打不開時，先關閉程式，把問題檔案移出 `plugins/`，再重開。也可請老師以 `bootstrap.py --safe-mode` 啟動，略過外掛。

老師準備好[開發環境](DEVELOPMENT.md)後，在 repo 根目錄執行：

```bash
python -m pytest -q -p no:cacheprovider tests/unit/test_example_plugins.py
python -m ruff check --no-cache docs/examples/plugins tests/unit/test_example_plugins.py
```

[範例測試](../tests/unit/test_example_plugins.py) 透過 `load_plugins` 從暫存工作區載入，重用[內建步驟 contract test](../tests/contracts/test_steps_contract.py) 的執行、序列化與 reset 檢查：彩色／灰階、連續三張、輸入不變、輸出型別與大小都要正確。另外測數值結果、Overlay 座標、參數錯誤、沒有翻譯檔時的名稱，以及載入錯誤紀錄。

## 8. 安全提醒

外掛就是 Python 程式，匯入時便會執行，能讀寫檔案或連線網路。**只放自己看得懂、來源可信的檔案**，不懂的先交老師檢查。不要把操作硬體、刪檔或下載程式放在模組最上層。先使用測試圖案與小陣列，不需要為了學外掛連接實體板子。
