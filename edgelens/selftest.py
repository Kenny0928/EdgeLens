"""自我檢查：bootstrap 安裝完後執行 `python -m edgelens --selftest`，CI 也會跑。

每個檢查回傳 Check；required=True 的檢查失敗時結束碼為 1。選配套件只報告狀態，
不算失敗。新增檢查：在 CHECKS 加一個函式。
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass

from edgelens import __version__


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    required: bool = True


def _python() -> Check:
    ok = (3, 10) <= sys.version_info[:2] <= (3, 13)
    return Check("python", ok, f"{platform.python_implementation()} {platform.python_version()} ({sys.executable})")


def _imports() -> Check:
    missing = []
    versions = []
    for mod, attr in (("numpy", "__version__"), ("cv2", "__version__"), ("PySide6", "__version__"),
                      ("serial", "VERSION")):
        try:
            m = importlib.import_module(mod)
            versions.append(f"{mod} {getattr(m, attr, '?')}")
        except Exception as e:  # noqa: BLE001
            missing.append(f"{mod}: {e}")
    return Check("required_packages", not missing, "; ".join(missing) if missing else ", ".join(versions))


def _optional() -> Check:
    from edgelens.core.registry import module_available

    status = {m: module_available(m) for m in ("onnxruntime", "ai_edge_litert", "tflite_runtime", "kflash",
                                                "cv2_enumerate_cameras")}
    return Check("optional_packages", True, ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in status.items()),
                 required=False)


def _registries() -> Check:
    from edgelens.core.registry import all_registries, discover, import_problems

    discover()
    counts = {name: len(reg) for name, reg in all_registries().items()}
    problems = [p for p in import_problems if p.module.startswith("edgelens")]
    detail = ", ".join(f"{k}={v}" for k, v in counts.items())
    if problems:
        detail += " | 匯入失敗：" + "; ".join(f"{p.module}: {p.error}" for p in problems)
    return Check("registries", not problems and counts["sources"] > 0, detail)


def _i18n() -> Check:
    from edgelens.i18n import LANGUAGES, load_language

    sizes = {lang: len(load_language(lang)) for lang in LANGUAGES}
    keys = [set(load_language(lang)) for lang in LANGUAGES]
    diff = set.symmetric_difference(*keys) if len(keys) == 2 else set()
    detail = ", ".join(f"{k}={v}" for k, v in sizes.items())
    if diff:
        detail += f" | 只在一種語言出現的鍵：{sorted(diff)[:5]}"
    return Check("i18n", not diff and all(sizes.values()), detail)


def _synthetic_pipeline() -> Check:
    from edgelens.core.registry import discover
    from edgelens.processing.pipeline import Pipeline
    from edgelens.sources.base import SourceInfo, create_source

    discover()
    src = create_source(SourceInfo(kind="synthetic", id="synthetic:selftest", name="selftest",
                                   params={"fps": 120, "resolution": "320x240"}))
    src.open()
    try:
        frames = [src.read(timeout=1.0) for _ in range(3)]
    finally:
        src.close()
    if any(f is None for f in frames):
        return Check("synthetic_source", False, "讀不到畫面")
    res = Pipeline().process(frames[-1])  # type: ignore[arg-type]
    return Check("synthetic_source", res.image.shape == (240, 320, 3), f"3 frames, last seq={frames[-1].seq}")


def _writable() -> Check:
    from edgelens.core.paths import app_cache_dir, app_config_dir, atomic_write_text

    problems = []
    for d in (app_cache_dir(), app_config_dir()):
        try:
            p = atomic_write_text(d / ".selftest", str(time.time()))
            p.unlink()
        except OSError as e:
            problems.append(f"{d}: {e}")
    return Check("writable_dirs", not problems, "; ".join(problems) or f"{app_cache_dir()} | {app_config_dir()}")


def _qt() -> Check:
    """建立一個 offscreen 的 QApplication，確認 Qt 外掛能載入。"""
    env_before = os.environ.get("QT_QPA_PLATFORM")
    try:
        if not env_before:
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        return Check("qt", True, f"platform={app.platformName()}")
    except Exception as e:  # noqa: BLE001
        return Check("qt", False, f"{e}")
    finally:
        if not env_before:
            os.environ.pop("QT_QPA_PLATFORM", None)


CHECKS: list[Callable[[], Check]] = [_python, _imports, _optional, _writable, _i18n, _registries,
                                     _synthetic_pipeline, _qt]


def run(as_json: bool = False) -> int:
    results: list[Check] = []
    for fn in CHECKS:
        try:
            results.append(fn())
        except Exception as e:  # noqa: BLE001
            results.append(Check(fn.__name__.strip("_"), False, f"{e}\n{traceback.format_exc()}"))
    failed = [c for c in results if c.required and not c.ok]
    if as_json:
        print(json.dumps({"version": __version__, "ok": not failed, "checks": [asdict(c) for c in results]},
                         ensure_ascii=False, indent=2))
    else:
        from edgelens.i18n import tr

        print(f"{tr('cli.selftest.start')} — EdgeLens {__version__}")
        for c in results:
            mark = "OK  " if c.ok else ("FAIL" if c.required else "--  ")
            print(f"  [{mark}] {c.name}: {c.detail}")
        print(tr("cli.selftest.ok") if not failed else tr("cli.selftest.failed", count=len(failed)))
    return 1 if failed else 0
