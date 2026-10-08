"""模擬網路相機的 HTTP 伺服器：測試用，也能在沒有 ESP32-CAM 時做課堂示範。

    with FakeCameraServer(mode="esp32cam") as srv:
        # ESP32-CAM 來源：ip 填 127.0.0.1:<port>
        info = Esp32CamSource.info_from_params({"ip": f"127.0.0.1:{srv.port}", "stream_port": srv.stream_port})

模式：
- "esp32cam"：仿 Arduino 官方 CameraWebServer。主埠有 /status、/control、/capture；
  另一個埠有 /stream，而且和真的一樣「同時只允許一個串流連線」，第二個連線回 503。
- "mjpeg"：像手機 IP 相機 App，所有路徑（/stream、/video、/capture…）都在同一個埠，不限連線數。

每張 JPEG 的 COM 區段裡會寫入 "idx=<序號>"，測試可以用 jpeg_index() 取出，檢查延遲有沒有累積。
檔名以底線開頭，不會被自動匯入註冊成來源。
"""

from __future__ import annotations

import atexit
import json
import socket
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

# esp32-camera 的 framesize_t 對照（值 → 寬, 高）
FRAME_SIZES: dict[int, tuple[int, int]] = {
    0: (96, 96), 1: (160, 120), 2: (176, 144), 3: (240, 176), 4: (240, 240), 5: (320, 240),
    6: (400, 296), 7: (480, 320), 8: (640, 480), 9: (800, 600), 10: (1024, 768), 11: (1280, 720),
    12: (1280, 1024), 13: (1600, 1200),
}

# 預設狀態（鍵與官方 CameraWebServer 的 /status 一致）
DEFAULT_STATE: dict[str, int] = {
    "framesize": 5, "quality": 12, "brightness": 0, "contrast": 0, "saturation": 0, "sharpness": 0,
    "denoise": 0, "special_effect": 0, "wb_mode": 0, "awb": 1, "awb_gain": 1, "aec": 1, "aec2": 0,
    "ae_level": 0, "aec_value": 204, "agc": 1, "agc_gain": 0, "gainceiling": 0, "bpc": 0, "wpc": 1,
    "raw_gma": 1, "lenc": 1, "hmirror": 0, "vflip": 0, "dcw": 1, "colorbar": 0, "led_intensity": 0,
}
# 官方韌體會回報、但 EdgeLens 不認得（應被忽略）的鍵
EXTRA_STATUS = {"xclk": 20, "face_detect": 0, "face_enroll": 0, "face_recognize": 0}
DEFAULT_OMIT = ("denoise", "led_intensity")

JPEG_QUALITY = 80


def jpeg_index(jpeg: bytes) -> int | None:
    """取出假伺服器寫在 JPEG 註解區段裡的序號；沒有（例如壞圖）回傳 None。"""
    i = jpeg.find(b"idx=", 0, 128)
    if i < 0:
        return None
    j = i + 4
    while j < len(jpeg) and 48 <= jpeg[j] <= 57:
        j += 1
    return int(jpeg[i + 4:j]) if j > i + 4 else None


def _with_comment(jpeg: bytes, text: str) -> bytes:
    data = text.encode("ascii")
    seg = b"\xff\xfe" + (len(data) + 2).to_bytes(2, "big") + data
    return jpeg[:2] + seg + jpeg[2:]


def render_frame(state: dict[str, int], idx: int) -> np.ndarray:
    """依目前設定畫一張合成畫面：移動的方塊，亮度、翻轉、特效都真的會改變畫面。"""
    w, h = FRAME_SIZES.get(int(state.get("framesize", 5)), (320, 240))
    if state.get("colorbar"):
        colors = np.array([[255, 255, 255], [0, 255, 255], [255, 255, 0], [0, 255, 0],
                           [255, 0, 255], [0, 0, 255], [255, 0, 0], [0, 0, 0]], np.uint8)
        img = np.repeat(colors[np.arange(w) * len(colors) // w][None, :, :], h, axis=0)
    else:
        img = np.full((h, w, 3), 90, np.uint8)
        img[:, :, 0] = np.linspace(60, 160, w, dtype=np.uint8)[None, :]
        bw, bh = max(4, w // 5), max(4, h // 4)
        x = int((w - bw) * (idx % 50) / 50)
        y = int((h - bh) * (0.5 + 0.4 * np.sin(idx / 9.0)))
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 200, 255), -1)
    b = int(state.get("brightness", 0))
    if b:
        img = np.clip(img.astype(np.int16) + b * 30, 0, 255).astype(np.uint8)
    effect = int(state.get("special_effect", 0))
    if effect == 1:
        img = 255 - img
    elif effect == 2:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if state.get("hmirror"):
        img = img[:, ::-1]
    if state.get("vflip"):
        img = img[::-1, :]
    return np.ascontiguousarray(img)


class FakeCameraServer:
    """見模組說明。

    fps               串流的張數／秒。
    mode              "esp32cam" 或 "mjpeg"。
    content_length    串流的每個 part 是否帶 Content-Length（False 時測試 SOI／EOI 掃描）。
    boundary_variant  Content-Type 裡 boundary 的寫法："plain"、"spaced"（前後有空白）、
                      "quoted"、"dashes"（含 --）、"case"（大小寫和實際不同）。
    chunked           串流是否用 chunked 傳輸（真的 ESP32 是）。
    bad_every         每 N 張送一張壞 JPEG（0＝不送）。
    username/password 設定後所有路徑都要 Basic 驗證。
    bandwidth_bps     模擬慢速網路，限制每秒送出的位元組（0＝不限）。
    omit              /status 不回報的鍵（模擬不同韌體／感光元件）。
    ignore_vars       /control 回 200 但不真的改（模擬韌體不接受，用來測 verified=False）。
    """

    def __init__(self, port: int = 0, fps: float = 15.0, mode: str = "esp32cam", *, stream_port: int = 0,
                 content_length: bool = True, boundary_variant: str = "plain", chunked: bool | None = None,
                 bad_every: int = 0, username: str | None = None, password: str | None = None,
                 bandwidth_bps: int = 0, omit: tuple[str, ...] = DEFAULT_OMIT,
                 ignore_vars: tuple[str, ...] = (), host: str = "127.0.0.1"):
        if mode not in ("esp32cam", "mjpeg"):
            raise ValueError("mode 必須是 'esp32cam' 或 'mjpeg'")
        self.mode = mode
        self.fps = float(fps)
        self.host = host
        self._want_port = port
        self._want_stream_port = stream_port
        self.content_length = content_length
        self.boundary_variant = boundary_variant
        self.chunked = (mode == "esp32cam") if chunked is None else chunked
        self.bad_every = bad_every
        self.username, self.password = username, password
        self.bandwidth_bps = bandwidth_bps
        self.omit = set(omit)
        self.ignore_vars = set(ignore_vars)
        self.state: dict[str, int] = dict(DEFAULT_STATE)
        self.paused = False
        self.port = 0
        self.stream_port = 0
        # 統計（測試用）
        self.stream_connections = 0
        self.stream_rejected = 0
        self.capture_requests = 0
        self.status_requests = 0
        self.control_log: list[tuple[str, str]] = []
        self._servers: list[ThreadingHTTPServer] = []
        self._threads: list[threading.Thread] = []
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._socks: set[socket.socket] = set()
        self._stream_slot = threading.Lock()
        self._t0 = time.monotonic()
        self._cache: tuple[int, int, bytes] | None = None
        self._state_version = 0

    # ---- 啟動／停止 --------------------------------------------------------

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def stream_url(self) -> str:
        if self.mode == "mjpeg":
            return f"{self.url}/stream"
        return f"http://{self.host}:{self.stream_port}/stream"

    def start(self) -> FakeCameraServer:
        if self._servers:
            return self
        self._stop_evt.clear()
        self._t0 = time.monotonic()
        main = self._make_server(self._want_port, is_stream_port=(self.mode == "mjpeg"))
        self.port = main.server_address[1]
        self._servers = [main]
        if self.mode == "esp32cam":
            side = self._make_server(self._want_stream_port, is_stream_port=True)
            self.stream_port = side.server_address[1]
            self._servers.append(side)
        else:
            self.stream_port = self.port
        for s in self._servers:
            t = threading.Thread(target=s.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True,
                                 name="fake-camera-http")
            t.start()
            self._threads.append(t)
        return self

    def stop(self) -> None:
        self._stop_evt.set()
        self.drop_connections()
        for s in self._servers:
            s.shutdown()
            s.server_close()
        for t in self._threads:
            t.join(2.0)
        self._servers, self._threads = [], []

    def __enter__(self) -> FakeCameraServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _make_server(self, port: int, *, is_stream_port: bool) -> ThreadingHTTPServer:
        owner = self

        class Handler(_Handler):
            fake = owner
            stream_port_flag = is_stream_port

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True
            request_queue_size = 16

        return Server((self.host, port), Handler)

    # ---- 故障模擬 ----------------------------------------------------------

    def drop_connections(self) -> None:
        """立刻切斷所有進行中的串流連線（模擬 Wi-Fi 中斷或裝置重開）。"""
        with self._lock:
            socks = list(self._socks)
        for s in socks:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def pause(self, paused: bool = True) -> None:
        """暫停回應：連線保持著但不再送資料。"""
        self.paused = paused

    # ---- 畫面 --------------------------------------------------------------

    def current_index(self) -> int:
        return int((time.monotonic() - self._t0) * self.fps)

    def jpeg_for(self, idx: int) -> bytes:
        with self._lock:
            c = self._cache
            if c is not None and c[0] == idx and c[1] == self._state_version:
                return c[2]
            state = dict(self.state)
            ver = self._state_version
        img = render_frame(state, idx)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        data = _with_comment(buf.tobytes(), f"idx={idx}") if ok else b""
        with self._lock:
            self._cache = (idx, ver, data)
        return data

    def status_json(self) -> dict[str, int]:
        with self._lock:
            d = {k: v for k, v in self.state.items() if k not in self.omit}
        d.update(EXTRA_STATUS)
        return d

    def apply_control(self, var: str, val: int) -> int:
        """回傳 HTTP 狀態碼。"""
        with self._lock:
            self.control_log.append((var, str(val)))
            if var not in self.state:
                return 404
            if var in self.ignore_vars:
                return 200
            if var == "framesize" and val not in FRAME_SIZES:
                return 500
            self.state[var] = val
            self._state_version += 1
        if var == "framesize":
            self.drop_connections()   # 真的相機換解析度時串流會中斷
        return 200

    def boundary_header(self) -> str:
        v = self.boundary_variant
        b = "frame"
        if v == "spaced":
            return f"multipart/x-mixed-replace; boundary = {b} "
        if v == "quoted":
            return f'multipart/x-mixed-replace;boundary="{b}"'
        if v == "dashes":
            return f"multipart/x-mixed-replace;boundary=--{b}"
        if v == "case":
            return "multipart/x-mixed-replace;boundary=FRAME"
        return f"multipart/x-mixed-replace;boundary={b}"


class _Handler(BaseHTTPRequestHandler):
    fake: FakeCameraServer
    stream_port_flag = False
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002  安靜
        pass

    # ---- 共用 ----------------------------------------------------------------

    def _send(self, code: int, body: bytes = b"", ctype: str = "text/plain", extra: dict[str, str] | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)
        self.close_connection = True

    def _authorized(self) -> bool:
        f = self.fake
        if f.username is None:
            return True
        import base64
        want = "Basic " + base64.b64encode(f"{f.username}:{f.password or ''}".encode()).decode()
        if self.headers.get("Authorization") == want:
            return True
        self._send(401, b"auth required", extra={"WWW-Authenticate": 'Basic realm="fake"'})
        return False

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._route()
        except (BrokenPipeError, ConnectionError, OSError):
            self.close_connection = True

    def _route(self) -> None:
        f = self.fake
        u = urllib.parse.urlsplit(self.path)
        path = u.path
        if not self._authorized():
            return
        if self.stream_port_flag and path in ("/stream", "/video", "/mjpeg", "/live"):
            self._stream()
            return
        if f.mode == "esp32cam" and self.stream_port_flag:
            self._send(404, b"not found")
            return
        if path == "/status" and f.mode == "esp32cam":
            f.status_requests += 1
            self._send(200, json.dumps(f.status_json()).encode(), "application/json")
        elif path == "/control" and f.mode == "esp32cam":
            q = urllib.parse.parse_qs(u.query)
            var = (q.get("var") or [""])[0]
            try:
                val = int((q.get("val") or ["0"])[0])
            except ValueError:
                self._send(400, b"bad val")
                return
            self._send(f.apply_control(var, val))
        elif path in ("/capture", "/snapshot", "/shot.jpg", "/snapshot.jpg"):
            f.capture_requests += 1
            self._wait_unpaused()
            self._send(200, f.jpeg_for(f.current_index()), "image/jpeg")
        elif path == "/":
            self._send(200, b"EdgeLens fake camera", "text/plain")
        else:
            self._send(404, b"not found")

    def _wait_unpaused(self) -> None:
        f = self.fake
        while f.paused and not f._stop_evt.is_set():
            time.sleep(0.02)

    # ---- 串流 --------------------------------------------------------------

    def _write(self, data: bytes) -> None:
        f = self.fake
        if f.chunked:
            data = b"%x\r\n" % len(data) + data + b"\r\n"
        if f.bandwidth_bps <= 0:
            self.wfile.write(data)
            return
        step = 1400
        for i in range(0, len(data), step):
            self.wfile.write(data[i:i + step])
            self.wfile.flush()
            time.sleep(min(step, len(data) - i) / f.bandwidth_bps)

    def _stream(self) -> None:
        f = self.fake
        limited = f.mode == "esp32cam"
        # 稍等一下：剛被切斷的舊連線需要一點時間釋放名額
        if limited and not f._stream_slot.acquire(timeout=0.3):
            f.stream_rejected += 1
            self._send(503, b"stream busy")
            return
        sock = self.connection
        with f._lock:
            f._socks.add(sock)
        f.stream_connections += 1
        try:
            self.send_response(200)
            self.send_header("Content-Type", f.boundary_header())
            if f.chunked:
                self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.flush()
            self.close_connection = True
            n = 0
            period = 1.0 / f.fps
            next_t = time.monotonic()
            last_idx = -1
            while not f._stop_evt.is_set():
                while f.paused and not f._stop_evt.is_set():
                    time.sleep(0.02)
                idx = f.current_index()
                if idx == last_idx:
                    time.sleep(0.001)
                    continue
                last_idx = idx
                n += 1
                jpeg = f.jpeg_for(idx)
                if f.bad_every and n % f.bad_every == 0:
                    jpeg = b"\xff\xd8" + b"not really a jpeg" * 20 + b"\xff\xd9"
                head = b"\r\n--frame\r\nContent-Type: image/jpeg\r\n"
                if f.content_length:
                    head += b"Content-Length: %d\r\n" % len(jpeg)
                head += b"X-Timestamp: %f\r\n\r\n" % time.time()
                self._write(head + jpeg)
                self.wfile.flush()
                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.monotonic()
        finally:
            with f._lock:
                f._socks.discard(sock)
            if limited:
                f._stream_slot.release()


# ---- 給各來源的 contract_fixture 共用 ---------------------------------------

_shared: dict[str, FakeCameraServer] = {}
_shared_lock = threading.Lock()


def shared_server(mode: str) -> FakeCameraServer:
    """整個測試程序共用的伺服器（背景 daemon 執行緒，程式結束時自動關閉）。"""
    with _shared_lock:
        srv = _shared.get(mode)
        if srv is None:
            srv = FakeCameraServer(fps=30.0, mode=mode).start()
            _shared[mode] = srv
            atexit.register(srv.stop)
        return srv
