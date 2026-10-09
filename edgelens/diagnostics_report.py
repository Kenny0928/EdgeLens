"""診斷資訊收集（非 GUI）：學生遇到問題時匯出給老師看，也可以給命令列使用。

    from edgelens import diagnostics_report as dr
    report = dr.collect()      # dict，可轉成 JSON
    print(dr.to_text(report))  # 給人看的文字

收集時不會開啟任何相機，也不會連接開發板。設定檔裡的密碼等秘密欄位一律遮蔽
（SourceInfo.params 的 password、ControlSpec.extra["secret"]、網址裡的 user:pass@）。
"""

from __future__ import annotations

import json
import logging
import platform
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

from edgelens import __version__

log = logging.getLogger(__name__)

MASK = "***"
LOG_TAIL_LINES = 200
PACKAGES = ("numpy", "opencv-python", "opencv-python-headless", "PySide6-Essentials", "onnxruntime", "onnx",
            "ai-edge-litert", "tflite-runtime", "kflash", "pyserial", "cv2-enumerate-cameras")
_SECRET_KEY = re.compile(r"(^|_)pass(word|wd)?$|^pwd$|secret|token|api_?key|authorization|credential", re.I)
_URL_CRED = re.compile(r"(\b[a-zA-Z][a-zA-Z0-9+.-]*://)[^/@\s:]+:[^/@\s]*@")


# ---- 去敏 -----------------------------------------------------------------------

def secret_keys() -> set[str]:
    """所有已註冊來源宣告為秘密的參數鍵（extra["secret"]）。"""
    from edgelens.core.registry import discover
    from edgelens.sources.base import SOURCES, secret_param_keys

    out: set[str] = set()
    try:
        discover()
        for kind in SOURCES.kinds():
            out |= secret_param_keys(kind)
    except Exception:  # noqa: BLE001 - 去敏不能因為註冊表壞掉而失敗，名稱規則還是會擋
        log.exception("取得秘密參數鍵失敗")
    return out


def mask_text(text: str) -> str:
    """遮蔽文字裡網址的帳號密碼（rtsp://user:pass@host → rtsp://***@host）。"""
    return _URL_CRED.sub(r"\1" + MASK + "@", text)


def sanitize(obj: Any, secrets: set[str] | None = None) -> Any:
    """遞迴複製 obj，把秘密欄位的值換成 ***。不修改原物件。"""
    secrets = secret_keys() if secrets is None else secrets
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and (k in secrets or _SECRET_KEY.search(k)) and v not in ("", None):
                out[k] = MASK
            else:
                out[k] = sanitize(v, secrets)
        return out
    if isinstance(obj, (list, tuple)):
        return [sanitize(v, secrets) for v in obj]
    if isinstance(obj, str):
        return mask_text(obj)
    return obj


# ---- 收集 -----------------------------------------------------------------------

def _safe(fn, default: Any = None) -> Any:
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - 診斷本身不能壞掉；錯誤也是診斷資訊
        return default if default is not None else {"error": f"{type(e).__name__}: {e}"}


def package_versions() -> dict[str, str | None]:
    from importlib import metadata

    out: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def read_extras() -> dict[str, Any]:
    path = Path(sys.prefix) / "extras.json"
    try:
        return {"path": str(path), "data": json.loads(path.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return {"path": str(path), "data": None}


def run_selftest() -> list[dict[str, Any]]:
    from dataclasses import asdict

    from edgelens import selftest

    out = []
    for fn in selftest.CHECKS:
        try:
            out.append(asdict(fn()))
        except Exception as e:  # noqa: BLE001
            out.append({"name": fn.__name__.strip("_"), "ok": False, "detail": f"{type(e).__name__}: {e}",
                        "required": True})
    return out


def registry_info() -> dict[str, Any]:
    from edgelens.core.registry import all_registries, discover, import_problems

    discover()
    return {"counts": {k: len(v) for k, v in all_registries().items()},
            "import_problems": [{"module": p.module, "error": p.error} for p in import_problems]}


def plugin_info(workspace_root: str | Path | None) -> dict[str, Any]:
    files: list[str] = []
    if workspace_root:
        folder = Path(workspace_root) / "plugins"
        if folder.is_dir():
            files = sorted(p.name for p in folder.glob("*.py"))
    loaded = sorted(m for m in sys.modules if m.startswith("edgelens_plugins."))
    return {"files": files, "loaded": loaded}


def workspace_info(path: str | Path | None) -> dict[str, Any]:
    from edgelens.core.paths import cloud_sync_provider

    if not path:
        return {"path": "", "cloud_sync": None}
    return {"path": str(path), "exists": Path(path).exists(), "cloud_sync": cloud_sync_provider(path)}


def camera_list() -> list[dict[str, Any]]:
    from edgelens.sources.enumerate import list_cameras

    return [c.to_dict() for c in list_cameras(probe=False)]   # probe=False：不會開啟相機


def serial_ports() -> list[dict[str, Any]]:
    from dataclasses import asdict

    from edgelens.sources.k210 import ports

    return [{**asdict(p), "stable_id": p.stable_id} for p in ports.list_serial_ports()]


def log_tail(path: Path | None = None, lines: int = LOG_TAIL_LINES) -> list[str]:
    from edgelens.core.log import current_log_file

    path = path or current_log_file()
    if path is None or not Path(path).is_file():
        return []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [mask_text(ln) for ln in text.splitlines()[-lines:]]


def settings_dict(settings: Any = None) -> dict[str, Any]:
    from edgelens.core.settings import Settings

    s = settings if settings is not None else Settings()
    return sanitize(s.as_dict() if hasattr(s, "as_dict") else s)


def collect(settings: Any = None, workspace: Any = None, *, selftest: bool = True) -> dict[str, Any]:
    """收集診斷資訊，回傳可轉成 JSON 的 dict。

    settings：Settings 物件（沒給就讀目前的設定檔）。workspace：Workspace、路徑或 None（取設定檔的值）。
    selftest=False 可略過自我檢查（較快）。"""
    from edgelens.core.settings import Settings

    settings = settings if settings is not None else _safe(Settings, None)
    ws_path: Any = getattr(workspace, "root", workspace)
    if not ws_path and hasattr(settings, "get"):
        ws_path = settings.get("workspace.path", "")
    report: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "edgelens_version": __version__,
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation(),
                   "executable": sys.executable, "prefix": sys.prefix},
        "os": {"system": platform.system(), "release": platform.release(), "version": platform.version(),
               "platform": platform.platform()},
        "cpu": {"machine": platform.machine(), "processor": platform.processor()},
        "packages": _safe(package_versions),
        "extras": _safe(read_extras),
        "selftest": _safe(run_selftest, []) if selftest else [],
        "registries": _safe(registry_info),
        "plugins": _safe(lambda: plugin_info(ws_path)),
        "workspace": _safe(lambda: workspace_info(ws_path)),
        "cameras": _safe(camera_list, []),
        "serial_ports": _safe(serial_ports, []),
        "log_file": "",
        "log_tail": [],
        "settings": _safe(lambda: settings_dict(settings)),
    }
    from edgelens.core.log import current_log_file

    lf = _safe(current_log_file, None)
    report["log_file"] = str(lf) if isinstance(lf, Path) else ""
    report["log_tail"] = _safe(log_tail, [])
    return report


# ---- 文字 -----------------------------------------------------------------------

def to_text(report: dict[str, Any]) -> str:
    """把 collect() 的結果轉成容易閱讀的純文字（給學生貼給老師）。"""
    L: list[str] = []

    def head(title: str) -> None:
        L.append("")
        L.append(f"== {title} ==")

    L.append(f"EdgeLens {report.get('edgelens_version', '?')}  ({report.get('generated', '')})")
    py, osd, cpu = report.get("python", {}), report.get("os", {}), report.get("cpu", {})
    head("System")
    L.append(f"Python: {py.get('implementation', '')} {py.get('version', '')}  {py.get('executable', '')}")
    L.append(f"OS: {osd.get('platform', '')}")
    L.append(f"CPU: {cpu.get('machine', '')} {cpu.get('processor', '')}".rstrip())
    head("Packages")
    pk = report.get("packages", {})
    for k, v in (pk.items() if isinstance(pk, dict) else []):
        L.append(f"{k}: {v if v else '(not installed)'}")
    ex = report.get("extras", {})
    head("Bootstrap extras")
    L.append(f"{ex.get('path', '')}: {json.dumps(ex.get('data'), ensure_ascii=False) if ex.get('data') else '(none)'}")
    head("Selftest")
    for c in report.get("selftest", []) or []:
        mark = "OK  " if c.get("ok") else ("FAIL" if c.get("required", True) else "--  ")
        L.append(f"[{mark}] {c.get('name')}: {c.get('detail', '')}")
    head("Registries")
    reg = report.get("registries", {})
    L.append(", ".join(f"{k}={v}" for k, v in (reg.get("counts") or {}).items()))
    for p in reg.get("import_problems") or []:
        L.append(f"IMPORT FAILED {p.get('module')}: {p.get('error')}")
    head("Plugins")
    pl = report.get("plugins", {})
    L.append(f"files: {', '.join(pl.get('files') or []) or '-'}")
    L.append(f"loaded: {', '.join(pl.get('loaded') or []) or '-'}")
    head("Workspace")
    ws = report.get("workspace", {})
    L.append(f"{ws.get('path', '')}")
    if ws.get("cloud_sync"):
        L.append(f"WARNING: workspace is inside a cloud-sync folder ({ws['cloud_sync']})")
    head("Cameras")
    for c in report.get("cameras") or []:
        L.append(f"{c.get('name')}  index={c.get('index')} backend={c.get('backend')} id={c.get('id')}")
    if not report.get("cameras"):
        L.append("-")
    head("Serial ports")
    for p in report.get("serial_ports") or []:
        L.append(f"{p.get('device')}  {p.get('description', '')}  chip={p.get('chip', '')} likely={p.get('likely')}")
    if not report.get("serial_ports"):
        L.append("-")
    head("Settings (secrets masked)")
    L.append(json.dumps(report.get("settings"), ensure_ascii=False, indent=2))
    head(f"Log tail ({report.get('log_file', '')})")
    L.extend(report.get("log_tail") or ["-"])
    return "\n".join(L) + "\n"


# ---- ZIP ------------------------------------------------------------------------

def default_export_path() -> Path:
    """預設存到桌面（有的話），否則文件資料夾。"""
    from edgelens.core.paths import documents_dir

    desktop = Path.home() / "Desktop"
    base = desktop if desktop.is_dir() else documents_dir()
    return base / f"EdgeLens_diagnostics_{time.strftime('%Y%m%d_%H%M%S')}.zip"


def export_zip(path: str | Path, report: dict[str, Any] | None = None, settings: Any = None) -> Path:
    """匯出 ZIP：report.json、report.txt、settings.json（已去敏）、完整日誌檔（網址密碼已遮蔽）。"""
    from edgelens.core.log import current_log_file
    from edgelens.core.paths import atomic_write_bytes

    report = report if report is not None else collect(settings)
    path = Path(path)
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.json", json.dumps(report, ensure_ascii=False, indent=2))
        z.writestr("report.txt", to_text(report))
        z.writestr("settings.json", json.dumps(report.get("settings", {}), ensure_ascii=False, indent=2))
        lf = current_log_file()
        if lf is not None:
            for p in sorted(lf.parent.glob(lf.name + "*")):
                try:
                    z.writestr(f"logs/{p.name}", mask_text(p.read_text(encoding="utf-8", errors="replace")))
                except OSError as e:
                    z.writestr(f"logs/{p.name}.error.txt", str(e))
    return atomic_write_bytes(path, buf.getvalue())
