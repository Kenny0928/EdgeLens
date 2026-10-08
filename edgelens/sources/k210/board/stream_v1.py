# ruff: noqa
# EdgeLens K210 板上串流程式（MaixPy v1，韌體 v0.6.3_2）。
# 由電腦端 EdgeLens 送到板子的 raw REPL 執行，電腦端會在檔案前面補上一行 P = {...}。
# 這個檔案是上傳到開發板執行的 MicroPython 程式，不是電腦端的 Python 模組。
# 板子上看到亂碼是正常的：串流是二進位資料。
#
# ---- 封包格式 --------------------------------------------------------------------
#   b"JBF1" + <H 中繼資料長度> + <I JPEG 長度> + 中繼資料(JSON) + JPEG
#   JPEG 長度為 0 表示這一張只傳數據（send_every 設定）。
#
# 中繼資料欄位：
#   seq 影格序號、t 板上時間(ms)、w/h 解析度、fps 整圈 FPS（平滑）
#   us  各步驟耗時(微秒)：cap 取像、kpu 推論、stats 統計、diff 影格差異、
#       sharp 清晰度、blobs 色塊、jpg 壓縮、tx 傳輸、lcd 顯示、gc 記憶體回收、loop 整圈
#       （tx、lcd、gc、loop 發生在送出數據之後，所以記錄的是上一張的值）
#   lat 取像開始到結果送出(微秒)、mem MicroPython 可用記憶體、heap 系統可用記憶體
#   objs 偵測結果（P 的 task 為 detect）[[x, y, w, h, 信心, 類別], ...]，座標對應 w/h 的畫面
#   cls  分類結果（task 為 classify）前 topk 名 [[類別編號, 分數], ...]，分數由高到低
#   nblobs 色塊數；roi_mean / roi_std ROI 亮度平均與標準差；diff ROI 與上一張的差異；sharp 清晰度
#   cfg 目前套用的設定（brightness、contrast、saturation、auto_gain、gain_db、auto_exposure、
#       exposure_us、auto_whitebal、vflip、hmirror、threshold、nms）
#   rb  感光元件讀回值 {gain_db, exp_us, rgb_db}（GC0328 驅動多半不支援）
#   ack 最近一次電腦指令的結果 {n 指令累計數, id 電腦給的指令序號, ok 成功的項目, err 失敗的項目 "名稱:原因"}
#
# ---- 啟動時印出的文字行（STREAM_START 之前）----------------------------------------
#   CAPS {json}   板子能力：fw 韌體版本、sensor 感光元件名稱、sensor_id、framesizes 可用解析度、
#                 ops 可用處理項目、kpu 是否有 KPU、mem_free／heap_free 可用記憶體、sd 是否有 SD 卡
#   FW、CONFIG {json}、MEM、MODEL、CAMERA、NOTE（提示）
#   STREAM_START  之後電腦端切換到串流鮑率（BAUD）
#
# ---- P 參數（電腦端填入；沒給的用預設值）-----------------------------------------
#   framesize "QQVGA"|"QVGA"|"VGA"、pixformat "rgb"|"gray"、quality JPEG 品質、
#   send_every 每幾張傳一張影像（0 = 只傳數據）、lcd 是否同步顯示在板子 LCD、
#   vflip、hmirror、dual_buff、roi (x, y, w, h)、ops ["stats","diff","sharp","blobs","yolo","classify"]、
#   camera {brightness, contrast, saturation, auto_gain, gain_db, auto_exposure, exposure_us, auto_whitebal}、
#   baud 串流鮑率、blob_thresholds、diff_scale
#   模型（ops 含 "yolo" 或 "classify" 時才用；不寫死任何專題的模型）：
#     model 板上模型路徑（例如 /flash/x.kmodel 或 /sd/x.kmodel）
#     input_size [寬, 高] 模型輸入大小；yolo 要求畫面大小剛好相同，classify 會縮放
#     anchor YOLO anchors（10 個數字，5 組）、classes 類別數、threshold／nms 門檻
#     topk 分類回傳前幾名（預設 5）、softmax true|false|"auto"（分數不是機率時才套 softmax）
#   舊版參數 mode = "yolo" 等同 ops 含 "yolo"；預設模型 facedetect 只是範例。
#
# ---- 即時指令 CMD -------------------------------------------------------------------
#   電腦端送一行 b'CMD {"brightness": 1, "_id": 7}\n'（JSON 物件，結尾換行）到同一個序列埠，
#   板子每張影像處理完後檢查一次，不用重新啟動。可用欄位：
#   brightness / contrast / saturation（-2..2）、auto_gain（bool）、gain_db、
#   auto_exposure（bool）、exposure_us、auto_whitebal、vflip、hmirror、
#   threshold、nms（偵測模型，會重新 init_yolo2）、_id（電腦端指令序號，原樣放進 ack.id）
#   注意：指令內不可含 0x01–0x04 這類 REPL 控制字元（JSON 純文字不會有）。

import sensor
import lcd
import time
import gc
import os
import ujson
import ustruct
import math
from machine import UART
from Maix import utils

try:
    P
except NameError:
    P = {}

OPS = list(P.get("ops", []))
if P.get("mode") == "yolo" and "yolo" not in OPS:
    OPS.append("yolo")
# 範例預設值（facedetect）；實際專題一律由電腦端傳入模型路徑與 anchors
MODEL = P.get("model", "/flash/facedetect.kmodel")
ANCHOR = P.get("anchor", (1.889, 2.5245, 2.9465, 3.94056, 3.99987,
                          5.3658, 5.155437, 6.92275, 6.718375, 9.01025))
CLASSES = P.get("classes", 1)
INPUT_SIZE = P.get("input_size", (320, 240))
THRESHOLD = P.get("threshold", 0.3)   # 板子先用低門檻，電腦端再過濾
NMS = P.get("nms", 0.3)
TOPK = P.get("topk", 5)
SOFTMAX = P.get("softmax", "auto")
QUALITY = P.get("quality", 70)
SEND_EVERY = P.get("send_every", 1)
USE_LCD = P.get("lcd", True)
VFLIP = P.get("vflip", False)
HMIRROR = P.get("hmirror", False)
FRAMESIZE = P.get("framesize", "QVGA")
GRAY = P.get("pixformat", "rgb") == "gray"
DUAL_BUFF = P.get("dual_buff", False)
ROI = P.get("roi")
BLOB_THRESHOLDS = P.get("blob_thresholds", [(20, 70, 20, 80, 0, 60)])
# 影格差異用縮小的灰階影像：全彩全尺寸要保留兩份 ROI，MicroPython 記憶體零碎時會 MemoryError
DIFF_SCALE = P.get("diff_scale", 0.5)
BAUD = P.get("baud", 1500000)
CAMERA = dict(P.get("camera", {}))

MAGIC = b"JBF1"
SIZES = {"QQVGA": (sensor.QQVGA, 160, 120), "QVGA": (sensor.QVGA, 320, 240),
         "VGA": (sensor.VGA, 640, 480)}
SENSORS = {0x9D: "GC0328", 0x9B: "GC0308", 0x2642: "OV2640", 0x7742: "OV7740", 0x7725: "OV7725"}
ALL_OPS = ["stats", "diff", "sharp", "blobs", "yolo", "classify"]


def apply_camera(cfg, keys, ok, err):
    """把 cfg 內 keys 指定的攝影機設定套到感光元件；每一項各自 try，成功／失敗記進 ok／err。"""
    def run(name, fn):
        try:
            fn()
            ok.append(name)
        except Exception as e:
            err.append("%s:%s" % (name, e))

    for k, fn in (("brightness", sensor.set_brightness), ("contrast", sensor.set_contrast),
                  ("saturation", sensor.set_saturation)):
        if k in keys and k in cfg:
            def one(k=k, fn=fn):
                cfg[k] = max(-2, min(2, int(cfg[k])))
                fn(cfg[k])
            run(k, one)
    if "auto_gain" in keys or "gain_db" in keys:
        if cfg.get("auto_gain", True):
            run("auto_gain", lambda: sensor.set_auto_gain(1))
        else:
            run("gain_db", lambda: sensor.set_auto_gain(0, gain_db=float(cfg.get("gain_db", 0))))
    if "auto_exposure" in keys or "exposure_us" in keys:
        if cfg.get("auto_exposure", True):
            run("auto_exposure", lambda: sensor.set_auto_exposure(1))
        else:
            run("exposure_us", lambda: sensor.set_auto_exposure(0, exposure_us=int(cfg.get("exposure_us", 10000))))
    if "auto_whitebal" in keys:
        run("auto_whitebal", lambda: sensor.set_auto_whitebal(bool(cfg["auto_whitebal"])))
    if "vflip" in keys:
        run("vflip", lambda: sensor.set_vflip(bool(cfg["vflip"])))
    if "hmirror" in keys:
        run("hmirror", lambda: sensor.set_hmirror(bool(cfg["hmirror"])))


def read_back():
    """讀回感光元件目前的增益、曝光；驅動不支援的項目略過。"""
    rb = {}
    try:
        rb["gain_db"] = round(sensor.get_gain_db(), 3)
    except Exception:
        pass
    try:
        rb["exp_us"] = sensor.get_exposure_us()
    except Exception:
        pass
    try:
        rb["rgb_db"] = [round(v, 2) for v in sensor.get_rgb_gain_db()]
    except Exception:
        pass
    return rb


def sensor_name():
    try:
        sid = sensor.get_id()
        return sid, SENSORS.get(sid, "unknown")
    except Exception:
        return -1, "unknown"


def has_sd():
    try:
        return len(os.listdir("/sd")) >= 0
    except Exception:
        return False


def top_scores(plist, k, mode):
    """分類輸出取前 k 名 [[類別編號, 分數], ...]。
    mode：True 一律套 softmax、False 不套、"auto" 分數看起來不是機率（有負值、超過 1、總和不是 1）才套。"""
    n = len(plist)
    if n == 0:
        return []
    vals = plist
    use = mode is True
    if mode == "auto":
        lo = min(plist)
        hi = max(plist)
        use = lo < 0 or hi > 1.0 or abs(sum(plist) - 1.0) > 0.05
    if use:
        hi = max(plist)
        e = [math.exp(v - hi) for v in plist]
        s = sum(e)
        vals = [v / s for v in e]
    k = min(k, n)
    used = []
    out = []
    for _ in range(k):
        best = -1
        for i in range(n):
            if i not in used and (best < 0 or vals[i] > vals[best]):
                best = i
        used.append(best)
        out.append([best, round(vals[best], 4)])
    return out


def main():
    fs, W, H = SIZES[FRAMESIZE]
    roi = tuple(ROI) if ROI else (0, 0, W, H)
    # ROI 依解析度等比例縮放（設定以 QVGA 座標表示）
    if ROI and FRAMESIZE != "QVGA":
        k = W / 320
        roi = tuple(int(v * k) for v in ROI)

    use_lcd = USE_LCD and FRAMESIZE != "VGA"
    if USE_LCD and not use_lcd:
        print("NOTE VGA 超過 LCD 尺寸，停用 LCD")
    iw, ih = INPUT_SIZE
    if "yolo" in OPS and ((W, H) != (iw, ih) or GRAY):
        OPS.remove("yolo")
        print("NOTE 偵測模型只接受 %dx%d 彩色畫面，已停用偵測" % (iw, ih))
    if "classify" in OPS and GRAY:
        OPS.remove("classify")
        print("NOTE 分類模型只接受彩色畫面，已停用分類")

    if use_lcd:
        lcd.init(freq=15000000)
    sensor.reset(dual_buff=DUAL_BUFF)
    sensor.set_pixformat(sensor.GRAYSCALE if GRAY else sensor.RGB565)
    sensor.set_framesize(fs)
    sensor.set_vflip(VFLIP)
    sensor.set_hmirror(HMIRROR)
    sensor.skip_frames(time=1000)
    sensor.run(1)

    cfg = dict(CAMERA, vflip=VFLIP, hmirror=HMIRROR, threshold=THRESHOLD, nms=NMS)
    ack = {"n": 0, "id": 0, "ok": [], "err": []}
    ok, err = [], []
    apply_camera(cfg, [k for k in CAMERA], ok, err)
    print("CAMERA ok %s err %s" % (ok, err))

    task = None
    kpu = None
    is_yolo = "yolo" in OPS
    is_cls = "classify" in OPS
    kpu_ok = True
    try:
        import KPU as kpu
    except Exception:
        kpu_ok = False
    if is_yolo or is_cls:
        if not kpu_ok:
            raise OSError("這個韌體沒有 KPU，無法執行模型")
        try:
            os.stat(MODEL)
        except OSError:
            raise OSError("找不到模型 %s，請先上傳到板子" % MODEL)
        heap0 = utils.heap_free()
        task = kpu.load(MODEL)
        if is_yolo:
            kpu.init_yolo2(task, THRESHOLD, NMS, len(ANCHOR) // 2, ANCHOR)
        print("MODEL heap_before %d heap_after %d" % (heap0, utils.heap_free()))

    gc.collect()
    sid, sname = sensor_name()
    fw = os.uname().version
    print("CAPS", ujson.dumps({"fw": fw, "sensor": sname, "sensor_id": sid,
                               "framesizes": list(SIZES.keys()), "ops": ALL_OPS, "kpu": kpu_ok,
                               "mem_free": gc.mem_free(), "heap_free": utils.heap_free(),
                               "sd": has_sd()}))
    print("FW", fw)
    print("CONFIG", ujson.dumps({"framesize": FRAMESIZE, "gray": GRAY, "dual_buff": DUAL_BUFF,
                                 "ops": OPS, "roi": roi, "send_every": SEND_EVERY,
                                 "lcd": use_lcd, "quality": QUALITY}))
    print("MEM mem_free %d heap_free %d" % (gc.mem_free(), utils.heap_free()))

    repl = UART.repl_uart()
    print("STREAM_START")              # 電腦端看到這行後切換鮑率
    time.sleep_ms(100)
    repl.init(BAUD, 8, None, 1, read_buf_len=4096)
    time.sleep_ms(300)

    cmd_buf = b""
    ticks = time.ticks_us
    diff_t = time.ticks_diff
    do_stats = "stats" in OPS
    do_diff = "diff" in OPS
    do_sharp = "sharp" in OPS
    do_blobs = "blobs" in OPS
    seq = 0
    fps = 0.0
    prev = None
    t_prev = ticks()
    us_prev = {}
    try:
        while True:
            us = us_prev
            m = {"seq": seq, "t": time.ticks_ms(), "w": W, "h": H}

            t0 = ticks()
            img = sensor.snapshot()
            t = ticks()
            us["cap"] = diff_t(t, t0)

            if task is not None and is_yolo:
                a = t
                objs = []
                for o in kpu.run_yolo2(task, img) or []:
                    x, y, w, h = o.rect()
                    objs.append([x, y, w, h, round(o.value(), 3), o.classid()])
                m["objs"] = objs
                t = ticks()
                us["kpu"] = diff_t(t, a)

            if task is not None and is_cls:
                a = t
                if (W, H) == (iw, ih):
                    src = img
                else:
                    # resize 會改動原圖，所以先複製；縮放後 JPEG 與 LCD 仍用原圖
                    src = img.copy().resize(iw, ih)
                try:
                    src.pix_to_ai()
                except Exception:
                    pass
                fmap = kpu.forward(task, src)
                plist = fmap[:]
                m["cls"] = top_scores(plist, TOPK, SOFTMAX)
                if src is not img:
                    del src
                del fmap
                t = ticks()
                us["kpu"] = diff_t(t, a)

            if do_stats:
                a = t
                s = img.get_statistics(roi=roi)
                m["roi_mean"] = s.l_mean()
                m["roi_std"] = s.l_stdev()
                t = ticks()
                us["stats"] = diff_t(t, a)

            if do_diff:
                a = t
                cur = img.copy(roi=roi, x_scale=DIFF_SCALE, y_scale=DIFF_SCALE)
                if not GRAY:
                    cur = cur.to_grayscale()
                if prev is not None:
                    # prev 用完即丟，直接在它上面算差異，省一次複製
                    m["diff"] = prev.difference(cur).get_statistics().l_mean()
                prev = cur
                t = ticks()
                us["diff"] = diff_t(t, a)

            if do_sharp:
                a = t
                g = img.copy(roi=roi)
                if not GRAY:
                    g = g.to_grayscale()
                m["sharp"] = g.laplacian(1).get_statistics().l_stdev()
                del g
                t = ticks()
                us["sharp"] = diff_t(t, a)

            if do_blobs:
                a = t
                m["nblobs"] = len(img.find_blobs(BLOB_THRESHOLDS, roi=roi,
                                                 pixels_threshold=150, area_threshold=150,
                                                 merge=False))
                t = ticks()
                us["blobs"] = diff_t(t, a)

            jpg = b""
            if SEND_EVERY and seq % SEND_EVERY == 0:
                a = t
                jpg = img.compressed(quality=QUALITY).to_bytes()
                t = ticks()
                us["jpg"] = diff_t(t, a)

            m["cfg"] = cfg
            m["rb"] = read_back()
            m["ack"] = ack
            m["us"] = us
            m["fps"] = round(fps, 2)
            m["lat"] = diff_t(t, t0)
            m["mem"] = gc.mem_free()
            m["heap"] = utils.heap_free()
            meta = ujson.dumps(m)
            a = ticks()
            repl.write(MAGIC + ustruct.pack("<HI", len(meta), len(jpg)) + meta)
            if jpg:
                repl.write(jpg)
            t = ticks()
            tx = diff_t(t, a)

            if use_lcd:
                a = t
                for o in m.get("objs", ()):
                    img.draw_rectangle(o[0], o[1], o[2], o[3], color=(255, 0, 0), thickness=2)
                lcd.display(img)
                t = ticks()
                lcd_us = diff_t(t, a)
            else:
                lcd_us = 0

            a = t
            del jpg
            gc.collect()
            t = ticks()
            gc_us = diff_t(t, a)

            loop_us = diff_t(t, t_prev)
            t_prev = t
            us_prev = {"tx": tx, "lcd": lcd_us, "gc": gc_us, "loop": loop_us}
            if loop_us > 0:
                fps = fps * 0.8 + 200000.0 / loop_us if fps else 1000000.0 / loop_us
            seq += 1

            # 檢查電腦端有沒有送來即時調整指令（每行以換行結尾）
            n = repl.any()
            if n:
                data = repl.read(n)
                if data:
                    if b"\x03" in data:
                        raise KeyboardInterrupt()   # 不讓 Ctrl+C 被吃掉
                    cmd_buf = (cmd_buf + data)[-1024:]
                ok, err = [], []
                cmd_id = ack["id"]
                count = ack["n"]
                while b"\n" in cmd_buf:
                    line, _, cmd_buf = cmd_buf.partition(b"\n")
                    line = line.strip()
                    if not line.startswith(b"CMD "):
                        continue
                    count += 1
                    try:
                        c = ujson.loads(line[4:])
                    except Exception as e:
                        c = None
                        err.append("json:%s" % e)
                    if c:
                        cmd_id = c.pop("_id", cmd_id)
                        old = dict(cfg)
                        e0 = len(err)
                        cfg.update(c)
                        apply_camera(cfg, c, ok, err)
                        if "threshold" in c or "nms" in c:
                            try:
                                cfg["threshold"] = float(cfg["threshold"])
                                cfg["nms"] = float(cfg["nms"])
                                if task is not None and is_yolo:
                                    kpu.init_yolo2(task, cfg["threshold"], cfg["nms"], len(ANCHOR) // 2, ANCHOR)
                                for name in ("threshold", "nms"):
                                    if name in c:
                                        ok.append(name)
                            except Exception as e:
                                err.append("threshold:%s" % e)
                        if len(err) > e0:       # 有失敗就把這次指令的欄位還原，cfg 才不會記著無效值
                            for k in c:
                                if k in old:
                                    cfg[k] = old[k]
                                else:
                                    cfg.pop(k, None)
                ack = {"n": count, "id": cmd_id, "ok": ok, "err": err}
    except Exception as e:
        # 趁鮑率還是串流速度時把錯誤印出，電腦端才讀得到
        import sys
        print("\nERROR at seq %d:" % seq)
        sys.print_exception(e)
        time.sleep_ms(200)
        raise
    finally:
        if task is not None:
            kpu.deinit(task)
        time.sleep_ms(300)
        repl.init(115200, 8, None, 1, read_buf_len=4096)


main()
