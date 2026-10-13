"""相機列舉：找出這台電腦接了哪些相機（給 webcam 來源與「診斷」面板用）。

重點：
- 列舉本身**不開啟相機**（macOS 一開相機就會跳權限視窗）。只有明確要求
  ``list_cameras(probe=True)`` 或呼叫 ``probe_cameras()`` 才會真的去開。
- 每台相機有穩定 id：``webcam:<名稱>``；同名多台時依 index 排序加 ``#2``、``#3``，
  這樣插拔 USB 後 index 變了，仍能用名稱找回同一台。
- ``cv2_enumerate_cameras`` 是選配的：沒裝或失敗時 Linux 退回掃 ``/sys/class/video4linux``，
  其他平台回傳空清單（需要時由 ``probe_cameras()`` 探測相機 0–7）。
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from edgelens.i18n import tr

log = logging.getLogger(__name__)

# OpenCV 後端名稱 → 編號（舊版 OpenCV 缺常數時用固定值）
API_IDS: dict[str, int] = {
    "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", 1200),
    "msmf": getattr(cv2, "CAP_MSMF", 1400),
    "dshow": getattr(cv2, "CAP_DSHOW", 700),
    "v4l2": getattr(cv2, "CAP_V4L2", 200),
}
API_NAMES: dict[int, str] = {v: k for k, v in API_IDS.items()}

SYSFS_V4L = Path("/sys/class/video4linux")


def current_platform() -> str:
    """"darwin"／"win"／"linux"／"other"。獨立成函式，測試才能假裝成別的平台。"""
    p = sys.platform
    if p == "darwin":
        return "darwin"
    if p.startswith("win"):
        return "win"
    if p.startswith("linux"):
        return "linux"
    return "other"


def default_api(platform: str | None = None) -> str:
    """各平台預設的 OpenCV 後端名稱；不認識的平台回傳 "auto"。"""
    return {"darwin": "avfoundation", "win": "msmf", "linux": "v4l2"}.get(platform or current_platform(), "auto")


@dataclass
class CameraDevice:
    """一台被找到的相機。

    index     給 cv2.VideoCapture(index, API) 用的編號（該後端自己的編號）。
    backend   "avfoundation"／"msmf"／"dshow"／"v4l2"／""（探測時不知道）。
    path      Linux 是 /dev/videoN；macOS 是裝置的 uniqueID；Windows 是裝置路徑。
    id        穩定 id（list_cameras 依同名順序補上 #2、#3）。
    role      "iphone"（接續互通相機）、"deskview"（桌上視角）或 ""。
    alternatives  同一台相機在其他後端的編號，例如 {"dshow": 1}（Windows 去重複後保留，
                  MSMF 打不開時可以改用 DSHOW，也才有「驅動程式設定視窗」）。
    """

    index: int
    name: str
    backend: str = ""
    path: str = ""
    vid: int | None = None
    pid: int | None = None
    id: str = ""
    dup: int = 1
    role: str = ""
    alternatives: dict[str, int] = field(default_factory=dict)

    @property
    def is_iphone(self) -> bool:
        return self.role == "iphone"

    @property
    def vid_pid(self) -> str:
        if self.vid is None or self.pid is None:
            return ""
        return f"{self.vid:04X}:{self.pid:04X}"

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "name": self.name, "backend": self.backend, "path": self.path,
                "vid": self.vid, "pid": self.pid, "id": self.id, "dup": self.dup, "role": self.role,
                "alternatives": dict(self.alternatives)}


def classify_role(name: str, model_id: str = "") -> str:
    """辨識 iPhone「接續互通相機」與 macOS 的「桌上視角」。

    iPhone 被改名時（例如「Kenny相機」）名稱裡不會有 iPhone，所以 macOS 上優先看
    system_profiler 回報的型號（例如 "iPhone18,5"）。"""
    low = name.lower()
    if "iphone" in low or model_id.lower().startswith("iphone"):
        return "iphone"
    if "desk view" in low or "桌上視角" in name or "桌上視覺" in name:
        return "deskview"
    return ""


def role_detail(role: str) -> str:
    """role 的說明文字（已翻譯）。"""
    if role == "iphone":
        return tr("source.webcam.detail.iphone")
    if role == "deskview":
        return tr("source.webcam.detail.deskview")
    return ""


def stable_id(name: str, dup: int = 1) -> str:
    return f"webcam:{name}" if dup <= 1 else f"webcam:{name}#{dup}"


# ---- 列舉 -----------------------------------------------------------------------


def _apis_for(platform: str) -> list[str]:
    return {"darwin": ["avfoundation"], "win": ["msmf", "dshow"], "linux": ["v4l2"]}.get(platform, [])


def _from_library(platform: str) -> list[CameraDevice] | None:
    """用 cv2_enumerate_cameras 列舉。沒裝或失敗回傳 None（呼叫端退回備案）。"""
    apis = _apis_for(platform)
    if not apis:
        return None
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except Exception as e:  # noqa: BLE001 - 選配套件，缺少或載入失敗都退回備案
        log.info("cv2_enumerate_cameras 不能用：%s", e)
        return None
    result: list[CameraDevice] = []
    got_any_api = False
    for api in apis:
        try:
            infos = list(enumerate_cameras(API_IDS[api]))
        except Exception as e:  # noqa: BLE001
            log.info("列舉 %s 相機失敗：%s", api, e)
            continue
        got_any_api = True
        for info in infos:
            result.append(CameraDevice(index=int(info.index), name=str(info.name or "").strip(), backend=api,
                                       path=str(info.path or ""), vid=info.vid, pid=info.pid))
    return result if got_any_api else None


def _from_sysfs(root: Path | None = None) -> list[CameraDevice]:
    """Linux 備案：掃 /sys/class/video4linux/videoN/name。
    同一台相機常有兩個節點（影像與中繼資料），只保留 index 檔內容為 0 的那個。"""
    root = root or SYSFS_V4L
    result: list[CameraDevice] = []
    if not root.is_dir():
        return result
    for node in sorted(root.glob("video*"), key=lambda p: _trailing_int(p.name)):
        m = re.fullmatch(r"video(\d+)", node.name)
        if not m:
            continue
        try:
            name = (node / "name").read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        try:
            if (node / "index").read_text().strip() not in ("0", ""):
                continue
        except OSError:
            pass
        vid = pid = None
        for rel in ("device/../idVendor", "device/idVendor"):
            try:
                vid = int((node / rel).read_text().strip(), 16)
                pid = int((node / rel.replace("idVendor", "idProduct")).read_text().strip(), 16)
                break
            except (OSError, ValueError):
                continue
        result.append(CameraDevice(index=int(m.group(1)), name=name or f"video{m.group(1)}", backend="v4l2",
                                   path=f"/dev/{node.name}", vid=vid, pid=pid))
    return result


def _macos_model_ids(timeout: float = 3.0) -> dict[str, str]:
    """macOS：相機 uniqueID → 型號（例如 iPhone 接續互通相機是 "iPhone18,5"）。

    用 system_profiler 查詢，不會開啟相機；失敗或逾時回傳空 dict。"""
    try:
        out = subprocess.run(["system_profiler", "-json", "SPCameraDataType"], capture_output=True,
                             timeout=timeout, check=False).stdout
        items = json.loads(out or b"{}").get("SPCameraDataType", [])
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        log.info("查詢相機型號失敗：%s", e)
        return {}
    return {str(i.get("spcamera_unique-id", "")): str(i.get("spcamera_model-id", ""))
            for i in items if isinstance(i, dict) and i.get("spcamera_unique-id")}


def _trailing_int(s: str) -> int:
    m = re.search(r"(\d+)$", s)
    return int(m.group(1)) if m else 0


def _dedupe(devices: list[CameraDevice]) -> list[CameraDevice]:
    """Windows 的 MSMF 與 DSHOW 會各列一次同一台：同名（再加同 VID/PID）視為同一台，
    保留先列舉到的後端（MSMF），另一個後端的編號放進 alternatives。
    同名多台時用「第幾個同名」配對，所以兩台一樣的相機不會被誤併成一台。"""
    kept: list[CameraDevice] = []
    seen: dict[tuple[str, int], CameraDevice] = {}
    ordinal: dict[tuple[str, str], int] = {}
    for d in devices:
        n = ordinal.get((d.backend, d.name.lower()), 0) + 1
        ordinal[(d.backend, d.name.lower())] = n
        first = seen.get((d.name.lower(), n))
        same_ids = first is not None and (
            first.vid is None or d.vid is None or (first.vid, first.pid) == (d.vid, d.pid))
        if first is not None and same_ids and first.backend != d.backend:
            first.alternatives[d.backend] = d.index
            continue
        seen.setdefault((d.name.lower(), n), d)
        kept.append(d)
    return kept


def _finalize(devices: list[CameraDevice], model_ids: dict[str, str] | None = None) -> list[CameraDevice]:
    """補上 role、同名編號（依 index 排序）與穩定 id。model_ids 為 uniqueID → 型號（macOS）。"""
    devices = sorted(devices, key=lambda d: (d.index, d.name))
    counts: dict[str, int] = {}
    for d in devices:
        d.role = classify_role(d.name, (model_ids or {}).get(d.path, ""))
        counts[d.name] = counts.get(d.name, 0) + 1
        d.dup = counts[d.name]
        d.id = stable_id(d.name, d.dup)
    return devices


def probe_cameras(indices: range | list[int] = range(8), api: str | None = None) -> list[CameraDevice]:
    """逐一嘗試開啟相機編號並馬上釋放，回傳打得開的。

    **會真的開相機**（macOS 會跳權限視窗、相機燈會亮一下），只在列舉不到時由使用者按
    「掃描」才呼叫。"""
    api = api or default_api()
    result: list[CameraDevice] = []
    for i in indices:
        cap = None
        try:
            cap = cv2.VideoCapture(i, API_IDS[api]) if api in API_IDS else cv2.VideoCapture(i)
            if cap.isOpened():
                result.append(CameraDevice(index=i, name=tr("source.webcam.camera_n", index=i),
                                           backend=api if api in API_IDS else ""))
        except Exception as e:  # noqa: BLE001
            log.info("探測相機 %s 失敗：%s", i, e)
        finally:
            if cap is not None:
                cap.release()
    return _finalize(result)


def list_cameras(probe: bool = False) -> list[CameraDevice]:
    """列出相機。預設（probe=False）不會開啟任何相機。

    probe=True：列舉不到任何相機時，才退回探測相機 0–7（會開相機）。"""
    platform = current_platform()
    devices = _from_library(platform)
    if devices is None:
        devices = _from_sysfs() if platform == "linux" else []
    if platform == "win":
        devices = _dedupe(devices)
    devices = [d for d in devices if d.name]
    if not devices and probe:
        return probe_cameras()
    model_ids = _macos_model_ids() if platform == "darwin" and devices else None
    return _finalize(devices, model_ids)


def find_camera(name: str, dup: int = 1, devices: list[CameraDevice] | None = None) -> CameraDevice | None:
    """依名稱（與同名的第幾台）找相機；插拔 USB 後 index 變了用這個找回來。"""
    devices = list_cameras() if devices is None else devices
    same = [d for d in devices if d.name == name]
    if not same:
        return None
    for d in same:
        if d.dup == dup:
            return d
    return same[0] if len(same) == 1 else None


def linux_device_path(index: int) -> str:
    return f"/dev/video{index}"   # Linux 專用路徑，固定用 /，不跟著執行平台變
