"""EdgeLens：跨平台邊緣視覺相機工具。

接上任何相機（webcam、iPhone、ESP32-CAM、K210、RTSP），做影像處理、跑模型、
拍照分類蒐集樣本，並比較不同相機與模型的表現。架構說明見 docs/DEVELOPMENT.md。
"""

import os as _os

# onnxruntime 1.2x 起會在 macOS／Linux 啟動微軟遙測，並在背景上傳使用資料。教學電腦不應該把資料
# 傳出去，而且程式結束時上傳執行緒和收尾程式會互相干擾，偶爾讓 Python 當掉
# （libc++abi: recursive_mutex lock failed）。環境變數要在匯入 onnxruntime 之前設定才有效，
# 所以放在這裡：任何程式（含外掛）只要先匯入 edgelens 就會生效。
_os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

__version__ = "0.1.0.dev0"
APP_NAME = "EdgeLens"
