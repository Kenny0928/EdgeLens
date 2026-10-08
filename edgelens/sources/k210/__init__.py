"""K210（Maixduino／Maix Bit，MaixPy v1）相機來源。

模組分工：
  protocol  串流封包的解析與編碼（純位元組處理，不碰序列埠）
  repl      raw REPL 控制與檔案上傳
  ports     用 VID/PID 找序列埠
  flash     燒錄 MaixPy 韌體（呼叫 kflash）
  source    K210Source（註冊到相機來源清單）
  sim       模擬板子（測試與無硬體示範用）
  board/    上傳到板子執行的 MicroPython 程式（不是電腦端模組，沒有 __init__.py）
"""
