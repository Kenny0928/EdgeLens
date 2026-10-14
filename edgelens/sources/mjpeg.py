"""網路相機來源的共同基礎，以及 MJPEG（HTTP 影像串流）來源。

本檔提供三層東西，RTSP 與 ESP32-CAM 來源也共用它們，所以不要在別處複製：

- NetworkSource：背景執行緒持續把資料讀完、只保留最新一張、斷線自動重連（指數退避，
  最多每 5 秒一次）。延遲不會越積越多，因為舊畫面一律丟掉。
- HttpJpegSource：用標準函式庫 http.client 讀 multipart/x-mixed-replace 串流或定時抓
  靜態圖（不用 requests）。
- MjpegSource：可接手機的 IP 相機 App、網路攝影機等。
"""

from __future__ import annotations

import base64
import http.client
import logging
import re
import socket
import threading
import time
import urllib.parse
from collections import deque
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from edgelens.core.controls import ControlSpec, ControlType
from edgelens.core.frame import Frame
from edgelens.i18n import tr
from edgelens.sources.base import FrameSource, SourceCapabilities, SourceError, SourceInfo, register_source

log = logging.getLogger(__name__)

MAX_JPEG_BYTES = 16 * 1024 * 1024
_REDIRECTS = (301, 302, 303, 307, 308)


class _Stopped(Exception):
    """close() 要求背景執行緒結束（內部使用）。"""


class HttpStatusError(Exception):
    """伺服器回了非 200 的代碼。"""

    def __init__(self, status: int, reason: str = ""):
        super().__init__(f"HTTP {status} {reason}".strip())
        self.status = status
        self.reason = reason


# ---------------------------------------------------------------------------
# HTTP 小工具（標準函式庫）
# ---------------------------------------------------------------------------

def basic_auth_header(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
    return f"Basic {token}"


def strip_userinfo(url: str) -> str:
    """去掉網址裡的 帳號:密碼@，用來產生 id 與顯示（密碼不可以出現在畫面或 id 裡）。"""
    parts = urllib.parse.urlsplit(url.strip())
    if parts.username is None and parts.password is None:
        return url.strip()
    host = parts.netloc.rsplit("@", 1)[-1]
    return urllib.parse.urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


def open_http(url: str, *, headers: dict[str, str] | None = None, connect_timeout: float = 1.5,
              header_timeout: float = 3.0, read_timeout: float = 1.0,
              on_socket: Callable[[socket.socket], None] | None = None,
              max_redirects: int = 3) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    """送出 GET 並讀完回應標頭。

    自己建立 socket 再交給 http.client，是為了讓 close() 能在連線階段就用 on_socket
    拿到 socket 並 shutdown（否則連不上的 IP 會卡住執行緒到逾時）。
    回傳後 socket 逾時已設成 read_timeout，讀串流時逾時只會丟 TimeoutError。
    """
    origin: tuple[str, str, int] | None = None   # 第一個網址的 (scheme, host, port)
    for _ in range(max_redirects + 1):
        parts = urllib.parse.urlsplit(url.strip())
        scheme = parts.scheme.lower()
        if scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"bad url: {url!r}")
        https = scheme == "https"
        port = parts.port or (443 if https else 80)
        here = (scheme, parts.hostname.lower(), port)
        if origin is None:
            origin = here
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query
        hdrs = {"User-Agent": "EdgeLens", "Accept": "*/*", "Connection": "close"}
        hdrs.update(headers or {})
        # 帳號密碼只屬於原本的主機：轉址到別的來源（或 https 降成 http）時不能跟著送出去，
        # 否則惡意或設定錯誤的轉址會把 Basic 認證洩漏給第三方；新網址自帶帳密則照常使用
        upgraded = here[1] == origin[1] and origin[0] == "http" and scheme == "https"   # 升級成 https 無妨
        if here != origin and not upgraded:
            hdrs = {k: v for k, v in hdrs.items() if k.lower() != "authorization"}
        if parts.username is not None and "Authorization" not in hdrs:
            hdrs["Authorization"] = basic_auth_header(urllib.parse.unquote(parts.username),
                                                      urllib.parse.unquote(parts.password or ""))
        sock = socket.create_connection((parts.hostname, port), timeout=connect_timeout)
        if on_socket is not None:
            on_socket(sock)
        conn: http.client.HTTPConnection | None = None
        try:
            if https:
                import ssl
                # 取捨：區域網路的相機幾乎都用自簽憑證，驗證憑證會讓它們全部連不上，所以不驗證。
                # 代價是帳號密碼（Basic 認證）走這條連線時，同一區網內的人可以做中間人攔截；
                # 教室情境可接受，但不要拿這個來源連到不信任的網路
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                sock = ctx.wrap_socket(sock, server_hostname=parts.hostname)
                if on_socket is not None:
                    on_socket(sock)
                conn = http.client.HTTPSConnection(parts.hostname, port)
            else:
                conn = http.client.HTTPConnection(parts.hostname, port)
            conn.sock = sock
            sock.settimeout(header_timeout)
            conn.request("GET", path, headers=hdrs)
            resp = conn.getresponse()
        except BaseException:
            try:
                (conn or sock).close()
            except OSError:
                pass
            sock.close()
            raise
        if resp.status in _REDIRECTS and resp.getheader("Location"):
            url = urllib.parse.urljoin(url, resp.getheader("Location") or "")
            conn.close()
            continue
        sock.settimeout(read_timeout)
        return conn, resp
    raise HttpStatusError(310, "too many redirects")


def http_fetch(url: str, *, headers: dict[str, str] | None = None, timeout: float = 3.0,
               max_bytes: int = MAX_JPEG_BYTES,
               on_socket: Callable[[socket.socket], None] | None = None) -> tuple[dict[str, str], bytes]:
    """抓一個網址的完整內容。非 200 丟 HttpStatusError。回傳 (小寫標頭, 內容)。"""
    conn, resp = open_http(url, headers=headers, header_timeout=timeout, read_timeout=timeout,
                           on_socket=on_socket)
    try:
        if resp.status != 200:
            raise HttpStatusError(resp.status, resp.reason or "")
        body = resp.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError("response too large")
        return {k.lower(): v for k, v in resp.getheaders()}, body
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# multipart/x-mixed-replace 解析
# ---------------------------------------------------------------------------

def parse_boundary(content_type: str) -> str | None:
    """從 Content-Type 取出 boundary 的核心字串（去掉前後空白、引號與開頭的 --）。"""
    for part in content_type.split(";")[1:]:
        key, _, value = part.partition("=")
        if key.strip().lower() == "boundary":
            core = value.strip().strip('"').strip("'").strip().lstrip("-").strip()
            return core or None
    return None


_HEADER_END = re.compile(rb"\r?\n\r?\n")
_CONTENT_LENGTH = re.compile(rb"content-length:\s*(\d+)", re.IGNORECASE)
SOI = b"\xff\xd8"
EOI = b"\xff\xd9"


class MjpegParser:
    """把位元組流切成一張張 JPEG。

    有 boundary：找到邊界後讀 part 標頭；有 Content-Length 就照長度切，沒有就掃描
    JPEG 的 SOI（FFD8）／EOI（FFD9）。沒有 boundary（直接串接的 JPEG）時只用 SOI／EOI。
    read_chunk() 回傳 b"" 表示對方關閉；逾時請丟 TimeoutError，由 on_idle(秒數) 決定要不要繼續等。
    """

    def __init__(self, boundary: str | None, read_chunk: Callable[[], bytes],
                 on_idle: Callable[[float], None] | None = None):
        self.core = boundary.encode("latin-1", "replace") if boundary else None
        # 大小寫不拘；開頭 1–2 個 - 都接受（有些相機的 boundary 已經含 --）
        self._bre = re.compile(rb"-{1,2}" + re.escape(self.core), re.IGNORECASE) if self.core else None
        self._read = read_chunk
        self._on_idle = on_idle
        self.buf = bytearray()
        self.last_data = time.monotonic()

    def _fill(self) -> None:
        while True:
            try:
                data = self._read()
            except TimeoutError:
                if self._on_idle is not None:
                    self._on_idle(time.monotonic() - self.last_data)
                continue
            if not data:
                raise ConnectionError("stream closed by peer")
            self.last_data = time.monotonic()
            self.buf += data
            return

    def next_jpeg(self) -> bytes:
        """阻塞到取得下一張完整 JPEG。"""
        while True:
            jpeg = self._extract()
            if jpeg is not None:
                return jpeg
            if len(self.buf) > 2 * MAX_JPEG_BYTES:  # 資料亂掉了，丟掉重來
                self.buf.clear()
            self._fill()

    def _extract(self) -> bytes | None:
        buf = self.buf
        body_start = 0
        next_marker = None
        if self._bre is not None:
            m = self._bre.search(buf)
            if m is not None:
                he = _HEADER_END.search(buf, m.end())
                if he is None:
                    if len(buf) - m.end() > 4096:  # 標頭太長，不像 part 標頭：跳過這個邊界
                        del buf[:m.end()]
                    return None
                headers = bytes(buf[m.end():he.start()])
                body_start = he.end()
                cl = _CONTENT_LENGTH.search(headers)
                if cl is not None:
                    n = int(cl.group(1))
                    if 0 < n <= MAX_JPEG_BYTES:
                        if len(buf) < body_start + n:
                            return None
                        jpeg = bytes(buf[body_start:body_start + n])
                        del buf[:body_start + n]
                        return jpeg
                next_marker = self._bre.search(buf, body_start)
            else:
                # 還沒看到邊界：保留尾巴（邊界可能被切成兩半），其餘丟掉
                keep = len(self.core or b"") + 16
                if SOI not in buf and len(buf) > keep:
                    del buf[:len(buf) - keep]
                    return None
        soi = buf.find(SOI, body_start)
        if soi < 0:
            return None
        if next_marker is not None:
            body = bytes(buf[soi:next_marker.start()])
            end = body.rfind(EOI)
            del buf[:next_marker.start()]
            return body[:end + 2] if end >= 0 else body
        # 沒有 Content-Length 也還沒看到下一個邊界：找結尾 FFD9。
        # 縮圖（EXIF）內也可能有 FFD9，所以結尾後面必須是邊界或下一張的開頭。
        tail_ok = (b"--", SOI) + ((self.core,) if self.core else ())
        pos = soi + 2
        while True:
            e = buf.find(EOI, pos)
            if e < 0:
                return None
            rest = bytes(buf[e + 2:e + 2 + 64]).lstrip(b"\r\n \t")
            # 後面還沒收到資料時：只有在前面沒有內嵌縮圖（第二個 SOI）的情況下才能確定結束
            nested = buf.find(SOI, soi + 2, e) >= 0
            if (not rest and not nested) or any(rest.lower().startswith(t.lower()) for t in tail_ok):
                jpeg = bytes(buf[soi:e + 2])
                del buf[:e + 2]
                return jpeg
            pos = e + 2


# ---------------------------------------------------------------------------
# 背景執行緒基底
# ---------------------------------------------------------------------------

class NetworkSource(FrameSource):
    """背景執行緒讀取、只保留最新一張、斷線自動重連的來源基底（本身不註冊）。

    子類別實作 _session()：連線並持續呼叫 _publish()，直到斷線（丟例外）或被要求停止
    （_check_stop() 會丟 _Stopped）。回傳 False 表示「計畫中的換手」，不算重連。
    時間參數可用 info.params 覆寫（fail_after_s、stall_s、open_wait_s），測試與特殊網路用。
    """

    FAIL_AFTER_S = 10.0   # 連續失敗超過這麼久，read() 才丟 SourceError
    STALL_S = 6.0         # 連線中但這麼久沒資料，視為斷線並重連
    OPEN_WAIT_S = 3.0     # open() 最多等第一張畫面或第一個錯誤這麼久
    JOIN_S = 2.0          # close() 等執行緒結束的上限
    BACKOFF_MIN = 0.25
    BACKOFF_MAX = 5.0
    WINDOW_S = 3.0        # net_fps／bytes_per_s 的統計視窗

    def __init__(self, info: SourceInfo):
        super().__init__(info)
        p = info.params
        self.fail_after_s = float(p.get("fail_after_s", self.FAIL_AFTER_S))
        self.stall_s = float(p.get("stall_s", self.STALL_S))
        self.open_wait_s = float(p.get("open_wait_s", self.OPEN_WAIT_S))
        self._open = False
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_sock: socket.socket | None = None
        self._latest: Frame | None = None
        self._returned_seq = 0
        self._last_ok = 0.0
        self._last_error: SourceError | None = None
        self._reconnects = 0
        self._decode_failures = 0
        self._got_frame = False
        self._arrivals: deque[tuple[float, int]] = deque()

    # ---- 子類別要實作／可覆寫 ------------------------------------------------

    def _session(self) -> bool | None:
        raise NotImplementedError

    def _pre_open(self) -> None:
        """open() 啟動執行緒前的檢查（例如 ESP32-CAM 先確認 /status）。失敗丟 SourceError。"""

    def _target(self) -> str:
        """訊息裡顯示的連線目標（不可含密碼）。"""
        return self.info.id

    def _extra_meta(self) -> dict[str, Any]:
        return {}

    # ---- 生命週期 -----------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> None:
        if self._open:
            return
        self._pre_open()
        self._stop.clear()
        with self._cond:
            self._latest = None
            self._returned_seq = 0
            self._last_error = None
            self._last_ok = time.monotonic()
            self._reconnects = 0
            self._decode_failures = 0
            self._arrivals.clear()
        self._open = True
        self._thread = threading.Thread(target=self._run, name=f"edgelens-{self.kind}", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + self.open_wait_s
        with self._cond:
            while self._latest is None and self._last_error is None:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._cond.wait(min(left, 0.05))
            err = self._last_error if self._latest is None else None
        if err is not None:
            self.close()
            raise SourceError(err.message, retryable=True, hint=err.hint)

    def close(self) -> None:
        self._stop.set()
        self._open = False
        self._interrupt()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(self.JOIN_S)
            if t.is_alive():
                log.warning("%s 的背景執行緒沒有在 %.1f 秒內結束，已放棄等待", self.kind, self.JOIN_S)
        self._thread = None
        with self._cond:
            self._latest = None
            self._cond.notify_all()

    def _interrupt(self) -> None:
        """從別的執行緒叫醒卡在 recv／connect 的背景執行緒。"""
        s = self._active_sock
        if s is not None:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def _set_sock(self, sock: socket.socket | None) -> None:
        self._active_sock = sock
        if sock is not None and self._stop.is_set():
            self._interrupt()

    def _check_stop(self) -> None:
        if self._stop.is_set():
            raise _Stopped

    # ---- 背景執行緒 ---------------------------------------------------------

    def _run(self) -> None:
        backoff = self.BACKOFF_MIN
        while not self._stop.is_set():
            self._got_frame = False
            counted = True
            try:
                counted = self._session() is not False
            except _Stopped:
                break
            except Exception as e:  # noqa: BLE001  任何錯誤都要重連，不能讓執行緒默默死掉
                if self._stop.is_set():
                    break
                log.info("%s 連線失敗：%r", self.kind, e)
                self._set_error(self._wrap_error(e))
            else:
                if counted and not self._stop.is_set():
                    self._set_error(self._wrap_error(ConnectionError("connection ended")))
            finally:
                self._active_sock = None
            if self._stop.is_set():
                break
            if counted:
                with self._cond:
                    self._reconnects += 1
            if self._got_frame:
                backoff = self.BACKOFF_MIN
            if self._stop.wait(backoff):
                break
            backoff = min(backoff * 2, self.BACKOFF_MAX)

    def _set_error(self, err: SourceError) -> None:
        with self._cond:
            self._last_error = err
            self._cond.notify_all()

    def _wrap_error(self, e: Exception) -> SourceError:
        target = self._target()
        if isinstance(e, SourceError):
            return e
        if isinstance(e, HttpStatusError):
            if e.status in (401, 403):
                return SourceError(tr("source.net.err.auth", target=target), retryable=True,
                                   hint=tr("source.net.hint.auth"))
            return SourceError(tr("source.net.err.http", target=target, code=e.status), retryable=True,
                               hint=tr("source.net.hint.url"))
        if isinstance(e, ValueError) and "bad url" in str(e):
            return SourceError(tr("source.net.err.bad_url", target=target), retryable=False,
                               hint=tr("source.net.hint.url"))
        detail = str(e) or type(e).__name__
        return SourceError(tr("source.net.err.connect", target=target, detail=detail, seconds=round(self.fail_after_s)),
                           retryable=True, hint=tr("source.net.hint.check"))

    # ---- 畫面交接 -----------------------------------------------------------

    def _publish(self, image: np.ndarray, *, jpeg: bytes | None = None, nbytes: int = 0) -> None:
        now = time.monotonic()
        with self._cond:
            arr = self._arrivals
            arr.append((now, nbytes))
            while arr and now - arr[0][0] > self.WINDOW_S:
                arr.popleft()
            span = arr[-1][0] - arr[0][0]
            fps = (len(arr) - 1) / span if len(arr) >= 2 and span > 1e-3 else 0.0
            bps = sum(b for _, b in list(arr)[1:]) / span if len(arr) >= 2 and span > 1e-3 else 0.0
            meta: dict[str, Any] = {"net_fps": round(fps, 2), "reconnects": self._reconnects,
                                    "bytes_per_s": int(bps), "decode_failures": self._decode_failures}
            meta.update(self._extra_meta())
            self._latest = Frame(image=image, seq=self._next_seq(), t_cap=now, source_id=self.source_id,
                                 jpeg=jpeg, meta=meta)
            self._last_ok = now
            self._last_error = None
            self._got_frame = True
            self._cond.notify_all()

    def _publish_jpeg(self, jpeg: bytes) -> bool:
        """解碼並交出一張 JPEG；解碼失敗只記次數，不中斷。"""
        try:
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        except cv2.error:
            img = None
        if img is None or img.size == 0:
            with self._cond:
                self._decode_failures += 1
            return False
        self._publish(img, jpeg=jpeg, nbytes=len(jpeg))
        return True

    def read(self, timeout: float = 1.0) -> Frame | None:
        if not self._open:
            raise SourceError(tr("source.net.err.not_open"))
        deadline = time.monotonic() + max(timeout, 0.0)
        with self._cond:
            while True:
                f = self._latest
                if f is not None and f.seq != self._returned_seq:
                    self._returned_seq = f.seq
                    return f
                now = time.monotonic()
                err = self._last_error
                if err is not None and now - self._last_ok > self.fail_after_s:
                    raise SourceError(err.message, retryable=True, hint=err.hint)
                left = deadline - now
                if left <= 0:
                    return None
                self._cond.wait(min(left, 0.1))
                if not self._open:
                    raise SourceError(tr("source.net.err.not_open"))

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(live_controls=False)


# ---------------------------------------------------------------------------
# HTTP 影像（串流或定時抓圖）
# ---------------------------------------------------------------------------

class HttpJpegSource(NetworkSource):
    """讀 HTTP 的 JPEG 影像：multipart 串流或定時 GET 單張。MJPEG 與 ESP32-CAM 共用（不註冊）。"""

    CONNECT_S = 1.5
    HEADER_S = 3.0
    READ_S = 1.0       # socket 逾時；close() 另外用 shutdown 立刻叫醒，所以不影響關閉速度
    FETCH_S = 3.0

    _mode_now = "stream"

    def _extra_meta(self) -> dict[str, Any]:
        return {"mode": self._mode_now}

    def _auth_headers(self, username: str = "", password: str = "") -> dict[str, str]:
        return {"Authorization": basic_auth_header(username, password)} if username else {}

    def _on_idle(self, idle_s: float) -> None:
        self._check_stop()
        if idle_s > self.stall_s:
            raise TimeoutError(f"no data for {idle_s:.0f}s")

    def _stream_loop(self, url: str, headers: dict[str, str]) -> None:
        """連上串流並一直讀，直到出錯或被要求停止。照片網址（image/*）會自動改成定時抓圖。"""
        self._check_stop()
        conn, resp = open_http(url, headers=headers, connect_timeout=self.CONNECT_S, header_timeout=self.HEADER_S,
                               read_timeout=self.READ_S, on_socket=self._set_sock)
        try:
            if resp.status != 200:
                raise HttpStatusError(resp.status, resp.reason or "")
            ctype = resp.getheader("Content-Type") or ""
            if ctype.lower().startswith("image/"):
                log.info("%s 是單張圖片網址，改用定時抓圖", url)
                conn.close()
                self._snapshot_loop(url, headers, self._interval_s())
                return
            parser = MjpegParser(parse_boundary(ctype), lambda: resp.read1(65536), self._on_idle)
            self._mode_now = "stream"
            while True:
                self._check_stop()
                jpeg = parser.next_jpeg()
                self._publish_jpeg(jpeg)
        finally:
            conn.close()

    def _interval_s(self) -> float:
        return 0.5

    def _snapshot_loop(self, url: str, headers: dict[str, str], interval_s: float,
                       until: float | None = None, label: str = "snapshot") -> None:
        """定時 GET 一張靜態圖。until（monotonic）到了就正常結束。label 是 meta 裡的模式名稱。"""
        self._mode_now = label
        while True:
            self._check_stop()
            t0 = time.monotonic()
            _, body = http_fetch(url, headers=headers, timeout=self.FETCH_S, on_socket=self._set_sock)
            self._publish_jpeg(body)
            if until is not None and time.monotonic() >= until:
                return
            wait = interval_s - (time.monotonic() - t0)
            if wait > 0 and self._stop.wait(wait):
                raise _Stopped


# ---------------------------------------------------------------------------
# MJPEG 來源
# ---------------------------------------------------------------------------

@register_source
class MjpegSource(HttpJpegSource):
    kind = "mjpeg"
    display_name_key = "source.mjpeg.name"
    description_key = "source.mjpeg.desc"
    discover_is_safe = True
    manual_add = True

    DEFAULT_URL = "http://192.168.1.100:8080/video"

    @classmethod
    def param_specs(cls) -> list[ControlSpec]:
        return [
            ControlSpec("url", ControlType.STR, "control.net.url", default=cls.DEFAULT_URL, live=False,
                        group="source", help_key="control.net.url.help"),
            ControlSpec("mode", ControlType.ENUM, "control.net.mode", default="stream", live=False, group="source",
                        options=(("stream", "control.net.mode.stream"), ("snapshot", "control.net.mode.snapshot"))),
            ControlSpec("snapshot_interval_ms", ControlType.INT, "control.net.snapshot_interval", default=500,
                        min=50, max=60000, unit="ms", live=False, group="source", advanced=True),
            ControlSpec("username", ControlType.STR, "control.net.username", default="", live=False,
                        group="source", advanced=True),
            ControlSpec("password", ControlType.STR, "control.net.password", default="", live=False,
                        group="source", advanced=True, extra={"secret": True}),
        ]

    @classmethod
    def info_from_params(cls, params: dict[str, Any]) -> SourceInfo:
        url = str(params.get("url") or cls.DEFAULT_URL).strip()
        clean = strip_userinfo(url)
        host = urllib.parse.urlsplit(clean).netloc or clean
        return SourceInfo(kind=cls.kind, id=f"mjpeg:{clean}", name=f"MJPEG {host}", params=dict(params),
                          detail=clean)

    @classmethod
    def contract_fixture(cls, tmp_path: Any) -> SourceInfo | None:
        from edgelens.sources._fake_http import shared_server
        srv = shared_server("mjpeg")
        return cls.info_from_params({"url": srv.stream_url})

    def __init__(self, info: SourceInfo):
        super().__init__(info)
        p = info.params
        self._url = str(p.get("url") or self.DEFAULT_URL).strip()
        self._mode = str(p.get("mode") or "stream")
        self._interval = max(0.05, float(p.get("snapshot_interval_ms", 500)) / 1000.0)
        parts = urllib.parse.urlsplit(self._url)
        user = str(p.get("username") or "") or urllib.parse.unquote(parts.username or "")
        pwd = str(p.get("password") or "") or urllib.parse.unquote(parts.password or "")
        self._headers = self._auth_headers(user, pwd)
        self._mode_now = "snapshot" if self._mode == "snapshot" else "stream"

    def _target(self) -> str:
        return strip_userinfo(self._url)

    def _interval_s(self) -> float:
        return self._interval

    def _pre_open(self) -> None:
        parts = urllib.parse.urlsplit(self._url)
        if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
            raise SourceError(tr("source.net.err.bad_url", target=self._target()), hint=tr("source.net.hint.url"))

    def _session(self) -> bool | None:
        if self._mode == "snapshot":
            self._snapshot_loop(self._url, self._headers, self._interval)
            return None
        self._stream_loop(self._url, self._headers)
        return None

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(provides_jpeg=True, live_controls=False, extra={"mode": self._mode_now})
