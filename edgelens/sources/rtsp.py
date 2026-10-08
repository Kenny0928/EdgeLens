"""RTSP／RTMP／HTTP 影片串流來源（例如 AMB82 的 RTSP）。

用 OpenCV 的 FFMPEG 後端開啟，背景執行緒不停地 read()、只保留最新一張，所以
延遲不會越積越多。也可以填本機影片檔的路徑（課堂示範、測試用）。
"""

from __future__ import annotations

import logging
import os
import threading
import urllib.parse
from pathlib import Path
from typing import Any

import cv2

from edgelens.core.controls import ControlSpec, ControlType
from edgelens.i18n import tr
from edgelens.sources.base import SourceCapabilities, SourceError, SourceInfo, register_source
from edgelens.sources.mjpeg import NetworkSource, strip_userinfo

log = logging.getLogger(__name__)

STREAM_SCHEMES = ("rtsp", "rtsps", "rtmp", "http", "https")
ENV_KEY = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
# OPENCV_FFMPEG_CAPTURE_OPTIONS 是整個程序共用的環境變數，且只在「開啟」那一刻被讀取。
# 用鎖避免兩台 RTSP 相機同時開啟時互相蓋掉對方的設定。
_ENV_LOCK = threading.Lock()


def is_local_file(url: str) -> bool:
    return "://" not in url and os.path.isfile(url)


def open_capture(url: str, transport: str = "tcp", timeout_ms: int = 5000) -> Any:
    """用 FFMPEG 後端開啟 VideoCapture。

    rtsp 預設用 TCP：UDP 在學校 Wi-Fi 常掉封包造成畫面破碎。環境變數開啟後立刻還原，
    不影響程式裡其他用到 OpenCV 的地方。開啟／讀取逾時在 OpenCV 支援時一併設定。
    """
    transport = transport if transport in ("tcp", "udp") else "tcp"
    use_env = urllib.parse.urlsplit(url).scheme.lower().startswith("rtsp")
    with _ENV_LOCK:
        old = os.environ.get(ENV_KEY)
        if use_env:
            os.environ[ENV_KEY] = f"rtsp_transport;{transport}"
        try:
            props = []
            for name in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
                prop = getattr(cv2, name, None)
                if prop is not None:
                    props += [prop, int(timeout_ms)]
            try:
                return cv2.VideoCapture(url, cv2.CAP_FFMPEG, props) if props else cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            except (TypeError, cv2.error):   # 舊版 OpenCV 不接受參數清單
                return cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        finally:
            if use_env:
                if old is None:
                    os.environ.pop(ENV_KEY, None)
                else:
                    os.environ[ENV_KEY] = old


@register_source
class RtspSource(NetworkSource):
    kind = "rtsp"
    display_name_key = "source.rtsp.name"
    description_key = "source.rtsp.desc"
    discover_is_safe = True
    manual_add = True
    requires = ()

    OPEN_WAIT_S = 8.0   # 略長於 OpenCV 的開啟逾時（5 秒）
    DEFAULT_URL = "rtsp://192.168.1.100:554/live"

    @classmethod
    def param_specs(cls) -> list[ControlSpec]:
        return [
            ControlSpec("url", ControlType.STR, "control.net.url", default=cls.DEFAULT_URL, live=False,
                        group="source", help_key="control.net.rtsp_url.help"),
            ControlSpec("transport", ControlType.ENUM, "control.net.transport", default="tcp", live=False,
                        group="source", advanced=True, help_key="control.net.transport.help",
                        options=(("tcp", "control.net.transport.tcp"), ("udp", "control.net.transport.udp"))),
        ]

    @classmethod
    def info_from_params(cls, params: dict[str, Any]) -> SourceInfo:
        url = str(params.get("url") or cls.DEFAULT_URL).strip()
        clean = strip_userinfo(url) if "://" in url else url
        label = urllib.parse.urlsplit(clean).netloc if "://" in clean else Path(clean).name
        return SourceInfo(kind=cls.kind, id=f"rtsp:{clean}", name=f"RTSP {label or clean}", params=dict(params),
                          detail=clean)

    @classmethod
    def contract_fixture(cls, tmp_path: Path) -> SourceInfo | None:
        import numpy as np
        path = Path(tmp_path) / "rtsp_fixture.avi"
        if not path.exists():
            w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (160, 120))
            if not w.isOpened():
                return None
            for i in range(60):
                img = np.full((120, 160, 3), 40, np.uint8)
                cv2.rectangle(img, (i * 2, 30), (i * 2 + 30, 70), (0, 200, 255), -1)
                w.write(img)
            w.release()
        return cls.info_from_params({"url": str(path)})

    def __init__(self, info: SourceInfo):
        super().__init__(info)
        p = info.params
        self._url = str(p.get("url") or self.DEFAULT_URL).strip()
        self._transport = str(p.get("transport") or "tcp")
        self._timeout_ms = int(p.get("timeout_ms", 5000))

    def _target(self) -> str:
        return strip_userinfo(self._url) if "://" in self._url else self._url

    def _pre_open(self) -> None:
        scheme = urllib.parse.urlsplit(self._url).scheme.lower() if "://" in self._url else ""
        if not (scheme in STREAM_SCHEMES or is_local_file(self._url)):
            raise SourceError(tr("source.rtsp.err.bad_url", target=self._target()), hint=tr("source.rtsp.hint.url"))

    def _session(self) -> bool | None:
        self._check_stop()
        cap = open_capture(self._url, self._transport, self._timeout_ms)
        try:
            if not cap.isOpened():
                raise SourceError(tr("source.rtsp.err.open", target=self._target()), retryable=True,
                                  hint=tr("source.rtsp.hint.check"))
            local = is_local_file(self._url)
            fps = 0.0
            if local:
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                fps = fps if 1.0 <= fps <= 240.0 else 30.0
            while True:
                self._check_stop()
                ok, img = cap.read()
                if (not ok or img is None) and local:   # 影片檔播完就從頭再來
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, img = cap.read()
                if not ok or img is None:
                    raise ConnectionError("read failed")
                self._publish(img)   # OpenCV 看不到網路位元組數，bytes_per_s 固定為 0
                if local and self._stop.wait(1.0 / fps):   # 檔案讀得比即時快，要自己控制速度
                    self._check_stop()
        finally:
            # release 只在背景執行緒自己做：別的執行緒對正在 read() 的 FFMPEG 做 release 可能當掉
            cap.release()

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(provides_jpeg=False, live_controls=False,
                                  extra={"transport": self._transport})
