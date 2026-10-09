# ESP32-CAM 硬體連接指南

以 Espressif 提供的 Arduino **CameraWebServer** 官方範例為標準。先讓板子能傳影像，再接 EdgeLens。沒有板子也可用最後的模擬伺服器練習。

## 需要準備的東西

- 已能開啟 EdgeLens 的電腦。
- ESP32-CAM 與相容的鏡頭；常見為 AI Thinker ESP32-CAM。
- ESP32-CAM-MB USB 底板，或 USB-TTL 轉接器、杜邦線與適合板子的穩定電源。
- 可傳資料的 USB 線。
- Arduino IDE、Espressif 的 `esp32` 開發板套件。
- 一個電腦與板子都能加入、且允許裝置互相連線的 Wi-Fi。典型 AI Thinker ESP32-CAM 使用 2.4 GHz Wi-Fi。

接線、電源額定值、RESET／下載按鍵位置都**依你的板子而定**。先看板背型號、針腳標示與廠商接線圖，再請老師確認。不要只憑網路照片的方向接線。

## 步驟

本節是 Arduino IDE 與硬體操作，選單語言可能因 IDE 版本而異；下一節才是 EdgeLens 的按鈕名稱。

1. 從 [Arduino 官方網站](https://www.arduino.cc/en/software) 安裝 Arduino IDE。在額外開發板管理員網址加入：
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json`
   再到開發板管理員搜尋 `esp32`，安裝發行者為 **Espressif Systems** 的套件。
   **成功時：** 開發板選單能找到 ESP32 系列。安裝位置可對照 [Espressif 官方安裝說明](https://docs.espressif.com/projects/arduino-esp32/en/latest/installing.html)。
2. 開啟 `檔案 → 範例 → ESP32 → Camera → CameraWebServer`。英文介面是 `File → Examples → ESP32 → Camera → CameraWebServer`。
   **成功時：** 編輯器打開 `CameraWebServer.ino` 與相機相關分頁。若沒有範例，先選 ESP32 系列開發板。
3. 選相機型號。新版範例在 `board_config.h`；舊版直接在 `CameraWebServer.ino`。搜尋 `CAMERA_MODEL_`，只保留一個型號的 `#define` 沒有被 `//` 註解。常見 AI Thinker 板使用：
   ```cpp
   #define CAMERA_MODEL_AI_THINKER
   ```
   **成功時：** 原本啟用的其他型號已註解，只剩符合板子的型號。請對照 [新版設定檔](https://github.com/espressif/arduino-esp32/blob/master/libraries/ESP32/examples/Camera/CameraWebServer/board_config.h) 或 [舊版 2.0.17 範例](https://github.com/espressif/arduino-esp32/blob/2.0.17/libraries/ESP32/examples/Camera/CameraWebServer/CameraWebServer.ino)。
4. 在主程式的 `ssid` 與 `password` 雙引號內填入 Wi-Fi 名稱、密碼。
   **成功時：** 程式中不再是星號範例值，大小寫與 Wi-Fi 完全一致。
5. 在開發板選單選 `AI Thinker ESP32-CAM`。若老師採用 `ESP32 Dev Module`，要確認板子確有 PSRAM，並啟用 `PSRAM`。`Partition Scheme` 選 `Huge APP`（至少 3 MB APP 空間）。有些板型已固定分割區或 PSRAM，不會顯示相同選單，請確認該板型設定。
   **成功時：** 編譯能通過，沒有程式空間不足。相機型號巨集與 IDE 開發板設定兩者都要正確；至少 3 MB 的要求見上述官方範例設定檔。
6. 斷電後選一種連接方式，讓老師檢查再通電：

   | 方式 | 做什麼 | 看到什麼代表成功 |
   |---|---|---|
   | ESP32-CAM-MB 底板 | 對齊板子與底板針腳，使用底板 USB 接電腦；依該底板說明操作下載／RESET 按鍵。 | IDE 出現對應序列埠。部分底板可自動進下載模式，部分仍需按鍵。 |
   | USB-TTL | 確認 UART 邏輯電位為 3.3 V；TX 接板上 RX、RX 接板上 TX、GND 共地。常見標示為 U0R／U0T，仍要核對板上標示。電源接法依板子規格，不能把 5 V 訊號接到 GPIO。燒錄前把 IO0 接 GND，再 RESET／重新上電。 | IDE 出現轉接器序列埠，上傳開始後有寫入進度。 |

   IO0 在重設時拉低會進入下載模式，依據 [Espressif 啟動模式說明](https://docs.espressif.com/projects/esptool/en/latest/esp32/advanced-topics/boot-mode-selection.html)。USB-TTL 的供電輸出不一定足以帶動相機，供電能力請老師確認。
7. 在 IDE 選正確序列埠，執行上傳。USB-TTL 若停在等待連線，可保持 IO0 接 GND 再按 RESET。上傳完成後，先斷電、移除 IO0 與 GND 的短接，再上電並按 RESET。
   **成功時：** IDE 顯示上傳完成，板子離開下載模式開始執行程式。MB 底板則依其說明回到執行模式。
8. 開啟 Arduino 序列埠監控視窗，速度選 `115200`，必要時再按 RESET。
   **成功時：** 看到 `Camera Ready! Use 'http://…' to connect`，記下實際 IP。一般 Wi-Fi 模式的 IP 由路由器分配，不一定是 `192.168.4.1`。官方程式的位置見 [CameraWebServer 主程式](https://github.com/espressif/arduino-esp32/blob/master/libraries/ESP32/examples/Camera/CameraWebServer/CameraWebServer.ino)。
9. 電腦加入同一個可互通的網路，在瀏覽器開 `http://你的IP/capture`。
   **成功時：** 出現一張照片。若另開了連續影像網頁，先停止串流或關掉該分頁。

## 在 EdgeLens 裡加入相機

1. 按「新增相機」，選「ESP32-CAM」。
   **成功時：** 「手動輸入」出現「IP 位址」。這個來源不會自動掃描網路，請直接手動輸入。
2. 在「IP 位址」填 Arduino 印出的 IP，例如 `192.168.1.20`。若 HTTP 主埠不是 80，可填 `host:port`，例如 `192.168.1.20:8080`。
   **成功時：** 欄位是你的實際位址。不要填 `/stream` 或 `/capture` 路徑。
3. 展開「進階」可看到「串流埠號」。官方範例通常保留 `81`；自訂韌體或模擬伺服器才依實際值修改。
   **成功時：** HTTP 主埠與串流埠分別填好，沒有把 `81` 誤填到主埠。
4. 按「加入」，等畫面出現。
   **成功時：** 移動鏡頭前物品，畫面跟著變化；「相機」面板出現相機回報的參數。

「IP 位址」提示原文：

> 電腦要連到同一個 Wi-Fi。IP 請看 Arduino 的序列埠監控視窗；ESP32-CAM 自己開熱點時是 192.168.4.1。

參數面板來自相機的 `/status`。只有 EdgeLens 認得、韌體也有回報的項目才會顯示。例如「補光燈亮度」（`led_intensity`）只有部分韌體有，沒有這個滑桿不代表板子壞掉。

更改「解析度」（`framesize`）可能短暫中斷串流。請一次改一個設定，等畫面恢復再繼續。提示原文：

> 解析度越大，畫面越細但越慢。換解析度時畫面會停一下。

## 常見問題

以下 `{target}`、`{detail}` 等是變動內容，畫面會代入實際 IP 與原因。

| 訊息原文或現象 | 做什麼 | 看到什麼代表成功 |
|---|---|---|
| 連不上 ESP32-CAM（{target}）。請檢查：IP 有沒有打錯（看 Arduino 序列埠監控視窗）、電腦是不是連到同一個 Wi-Fi、ESP32-CAM 有沒有通電。（詳細原因：{detail}） | 看序列埠監控視窗核對 IP、電源與 Wi-Fi；用電腦瀏覽器開 `/capture`。修正後按「重試」；IP 改了就重新新增。 | 瀏覽器看到照片，EdgeLens 也有影像。 |
| {target} 有回應，但不像是 ESP32-CAM 的「CameraWebServer」程式（讀不到 /status）。請確認有燒錄 Arduino 範例「ESP32 → Camera → CameraWebServer」。 | 確認燒錄的是 CameraWebServer；在瀏覽器開 `http://你的IP/status`。 | 出現含 `framesize` 或 `quality` 的資料，重新加入後有影像。 |
| 設定已送出，但讀不回相機目前的值，無法確認。 | 等網路穩定後再改一次，觀察回報。 | 顯示「相機已確認收到這個設定」。 |
| 相機沒有接受 {key}={want}，目前仍是 {got}。可能是這個感光元件不支援，或需要先關掉相關的自動功能。 | 改回支援值，必要時先關閉相關自動功能。 | 回報值符合設定，畫面有相應變化。 |
| 瀏覽器連續串流正常，EdgeLens 卻比較慢 | 官方範例的串流同時只能一個連線。EdgeLens 連不上串流會自動改抓 `/capture`，畫面變慢是正常的。停止瀏覽器串流，稍等約十幾秒。 | EdgeLens 再次接回連續串流，更新變順。 |
| 板子一直重開，序列埠顯示 `Brownout detector was triggered` | 這是板上訊息，表示供電電壓過低觸發保護。請老師檢查電源、線材、接頭；依板子額定值供電。 | 不再反覆印開機訊息，能穩定顯示 IP。 |
| 序列埠一直印點，沒有 IP | 核對 `ssid`、密碼、2.4 GHz 網路與訊號。需要網頁登入或特殊認證的校園 Wi-Fi 不能直接照一般密碼範例連線。 | 印出可用 IP。 |
| 同一 Wi-Fi 仍連不到 | 校園 Wi-Fi 可能隔離裝置。請老師或網管確認，改用允許互通的教學網路，或下方的板子熱點。 | 電腦能開 `/capture`，再回 EdgeLens 加入。 |

供電訊息依 [Espressif Brownout 說明](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/fatal-errors.html) 整理。

程式的連線提示原文：

> 先用電腦的瀏覽器開開看同一個網址，能看到畫面再回來試。

## 進階（可略過）

### 讓 ESP32-CAM 自己開 Wi-Fi 熱點

這需要老師修改 Arduino 範例再上傳，**不是 EdgeLens 裡的開關**。原範例只會連既有 Wi-Fi；直接填 `192.168.4.1` 不會讓板子自動變成熱點。

1. 保留相機初始化。在 `setup()` 找到 `WiFi.begin(ssid, password)`，移除從它開始到原本印完連線網址的那一段（包含等待 `WL_CONNECTED` 的迴圈與原本的 `startCameraServer()`）。
   **成功時：** 相機初始化仍在，舊的 Wi-Fi 等待迴圈不再阻擋程式。
2. 在原位置換成下列程式。請更換示範熱點密碼，至少 8 個字元：

   ```cpp
   WiFi.mode(WIFI_AP);
   IPAddress apIP(192, 168, 4, 1);
   IPAddress netmask(255, 255, 255, 0);
   if (!WiFi.softAPConfig(apIP, apIP, netmask) ||
       !WiFi.softAP("EdgeLens-CAM", "ChangeMe123")) {
     Serial.println("AP setup failed");
     return;
   }
   startCameraServer();
   Serial.print("Camera Ready! Use 'http://");
   Serial.print(WiFi.softAPIP());
   Serial.println("' to connect");
   ```

   **成功時：** 編譯、上傳後，115200 的序列埠監控視窗印出 `192.168.4.1`。`startCameraServer()` 只呼叫一次。API 依據 [Espressif Wi-Fi AP 文件](https://docs.espressif.com/projects/arduino-esp32/en/latest/api/wifi.html)。此改法仍需老師在實際板子及套件版本驗證。
3. 電腦改連 `EdgeLens-CAM` 熱點，輸入剛設定的密碼，再用瀏覽器測試 `http://192.168.4.1/capture`。
   **成功時：** 即使系統提示此 Wi-Fi 沒有網際網路，仍能看到照片。在 EdgeLens 填「IP 位址」=`192.168.4.1`、「串流埠號」=`81`，按「加入」。

### 沒有 ESP32-CAM：用模擬相機練習

EdgeLens 內建一台「模擬 ESP32-CAM」：在這台電腦上假裝成一台燒錄了 CameraWebServer 的板子，畫面是移動的方塊。不需要板子、Arduino 或網路，適合先熟悉操作，或在硬體還沒準備好的課堂示範。

1. 按「新增相機」，選「模擬 ESP32-CAM（不需要硬體）」，在清單裡選「#1」並加入。
   **成功時：** 畫面出現移動方塊；「相機」面板出現和真的 ESP32-CAM 一樣的參數（例如「亮度」、「特效」）。模擬相機預設沒有「補光燈亮度」。
2. 試著改「解析度」，觀察畫面停一下再恢復；這和真的板子一樣。
3. 想看「同時只能一個串流」的效果：再加入清單裡的「#2」。兩個項目連到同一台模擬板子，所以 #2 會改用定時抓單張，畫面比較慢，這就是真的 ESP32-CAM 被瀏覽器占用串流時的情形。
