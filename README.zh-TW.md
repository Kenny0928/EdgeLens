# EdgeLens 邊緣視覺相機工具

EdgeLens 是給國中師生的通用影像辨識教學工具：接相機、做影像處理、跑模型、蒐集分類樣本，並研究不同相機與模型的表現。

English: [README.md](README.md)

## 能做什麼

- **接各種影像來源**：電腦內建／USB webcam、由作業系統提供的 iPhone 相機、圖片與影片、HTTP MJPEG、RTSP、ESP32-CAM、K210（MaixPy v1）。沒有硬體也能用測試圖案、模擬 ESP32-CAM 練習。
- **觀察與處理影像**：多相機並排、裁切縮放、灰階、濾波、顏色遮罩、二值化、輪廓與動作分析；可用吸管取色、儲存流程與 Python 外掛。
- **執行分類與偵測模型**：ONNX、TFLite、Darknet（`.weights`＋`.cfg`）、OpenCV DNN 的 TensorFlow／Caffe 格式、Haar 級聯；K210 `.kmodel` 在板上執行。支援常見 YOLO 輸出、SSD、MobileNet 分類；不是所有匯出格式都保證相容，實測與限制見[模型支援說明](docs/MODEL_SUPPORT.md)。
- **蒐集與整理樣本**：類別快捷鍵、連拍、倒數、靜止自動拍、品質與相近照片提醒、構圖參考、多相機拍照、改類別、刪除復原、資料切分與 ZIP 匯出。
- **留下實驗證據**：FPS、延遲、長時間測試、參數掃描、CSV 與錄影。訓練在外部平台完成，EdgeLens 負責蒐集、處理與測試。

## 快速開始

1. 先從 [python.org](https://www.python.org/downloads/) 安裝 **Python 3.10–3.13**。Windows 安裝時勾選 **Add python.exe to PATH**。Linux 也需要該 Python 版本的 `venv` 與 `pip`，可請老師協助。
2. 在本專案的程式碼頁下載 ZIP（例如 GitHub 的 Code → Download ZIP），**完整解壓縮**到本機資料夾。不要在 ZIP 裡直接執行。
3. 依作業系統雙擊啟動檔：

   | 系統 | 檔案 |
   |---|---|
   | Windows | [run_windows.bat](run_windows.bat) |
   | macOS | [run_macos.command](run_macos.command) |
   | Linux | [run_linux.sh](run_linux.sh) |

   macOS 第一次可對 `run_macos.command` 按右鍵 → 打開；若仍被下載隔離屬性擋住，確認 ZIP 來源可信後，在終端機執行 `xattr -dr com.apple.quarantine "/完整路徑/EdgeLens資料夾"`，再開一次。若沒有執行權限，在解壓後資料夾執行 `chmod +x run_macos.command`。Linux 若雙擊只開文字，先允許執行檔案，或在該資料夾執行 `bash run_linux.sh`。
4. 第一次會下載約 **300–400 MB**，可能花幾分鐘，請保持連線並等候。啟動器會在本機快取建立專用 Python 環境。部分平台可能無法安裝 TFLite 等選配套件；基本相機與影像處理仍可使用。
5. 開啟後選擇工作區（放專案與照片的資料夾）。預設是「文件／EdgeLens」；如果文件資料夾有 iCloud／OneDrive 同步，預設會改成家目錄裡的「EdgeLens」。自己選位置時**不要選雲端同步資料夾**，否則照片可能被移到雲端。按「新增相機」，選「測試圖案（不需相機）」，按「尋找」，選一筆結果，再按「開啟」。看到移動圖案就成功了。

接下來照[學生操作手冊](docs/STUDENT_GUIDE.md)建立專案、拍第一組樣本。真實相機的接法見下方硬體指南。

## 離線安裝

先在**與教室電腦相同作業系統、CPU 架構與 Python 次版本**的連網電腦準備套件。在解壓後的 EdgeLens 資料夾開啟終端機：

```bash
python3 bootstrap.py --make-wheelhouse ./wheelhouse
```

Windows 可將 `python3` 換成 `py -3.12`（以實際安裝版本為準）。將整份 EdgeLens 資料夾、`wheelhouse/` 及對應的 Python 安裝程式帶到離線電腦；先裝 Python，再於 EdgeLens 資料夾執行：

```bash
python3 bootstrap.py --wheelhouse ./wheelhouse
```

Windows 同樣改用 `py -3.12`。啟動腳本也接受這些參數，例如 `run_windows.bat --wheelhouse .\wheelhouse`、`bash run_macos.command --wheelhouse ./wheelhouse`、`bash run_linux.sh --wheelhouse ./wheelhouse`。成功安裝後，平常直接雙擊即可。

下載完成不代表每個選配套件都有取得；請查看下載訊息，並先在一台離線電腦測試。wheelhouse 不包含 Python、模型、驅動程式或韌體；這些由老師另外準備。詳見[教師指南](docs/TEACHER_GUIDE.md)。

## 文件索引

| 文件 | 適合誰／內容 |
|---|---|
| [學生操作手冊](docs/STUDENT_GUIDE.md) | 第一次操作到匯出資料集 |
| [教師指南](docs/TEACHER_GUIDE.md) | 教室部署、專案管理、課程與評量 |
| [外掛教學](docs/PLUGINS.md) | 自己寫影像處理步驟，附三個 Python 範例 |
| [模型支援說明](docs/MODEL_SUPPORT.md) | 模型格式、輸出解碼與限制 |
| [Webcam／iPhone](docs/hardware/WEBCAM_IPHONE.md) | 本機相機與手機 |
| [網路相機](docs/hardware/NETWORK_CAMERAS.md) | MJPEG／RTSP |
| [ESP32-CAM](docs/hardware/ESP32CAM.md) | Wi-Fi 相機與模擬來源 |
| [K210](docs/hardware/K210.md) | 序列埠、板上程式與硬體注意事項 |
| [開發說明](docs/DEVELOPMENT.md) | 架構、擴充介面與測試 |

## 授權與致謝

EdgeLens 採 [MIT 授權](LICENSE)。第三方套件、模型及韌體各自依其授權使用。

感謝 Python、Qt／PySide6、OpenCV、NumPy、pySerial、cv2-enumerate-cameras、ONNX Runtime、LiteRT／TensorFlow Lite、Sipeed MaixPy、kflash、Espressif 與相關開源社群，以及提供教學回饋的師生。
