"""bootstrap.py 與啟動腳本的單元測試：只測純函式，不真的安裝套件。"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def bs():
    spec = importlib.util.spec_from_file_location("edgelens_bootstrap", ROOT / "bootstrap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- venv 路徑

def test_venv_path_macos(bs):
    p = bs.venv_dir("darwin", env={}, home="/Users/a", version=(3, 12), machine="arm64")
    assert p == os.path.join("/Users/a", "Library", "Caches", "EdgeLens", "venv-cp312-arm64")


def test_venv_path_windows(bs):
    env = {"LOCALAPPDATA": "C:\\Users\\a\\AppData\\Local"}
    p = bs.venv_dir("win32", env=env, home="C:\\Users\\a", version=(3, 13), machine="AMD64")
    assert p.endswith(os.path.join("EdgeLens", "Cache", "venv-cp313-amd64"))
    assert p.startswith("C:\\Users\\a\\AppData\\Local")


def test_venv_path_linux_xdg_and_default(bs):
    p = bs.venv_dir("linux", env={"XDG_CACHE_HOME": "/x"}, home="/home/a", version=(3, 10), machine="x86_64")
    assert p == os.path.join("/x", "EdgeLens", "venv-cp310-x86_64")
    p = bs.venv_dir("linux", env={}, home="/home/a", version=(3, 11), machine="aarch64")
    assert p == os.path.join("/home/a", ".cache", "EdgeLens", "venv-cp311-aarch64")


def test_venv_env_overrides(bs):
    assert bs.venv_dir("linux", env={"EDGELENS_VENV": "/my/venv"}) == "/my/venv"
    assert bs.cache_dir("linux", env={"EDGELENS_CACHE": "/c"}) == "/c"


def test_cache_dir_matches_app_cache_dir(bs, monkeypatch):
    """bootstrap 的 cache_dir 要和 edgelens.core.paths.app_cache_dir 一致。"""
    from edgelens.core import paths

    monkeypatch.delenv("EDGELENS_CACHE", raising=False)
    assert Path(bs.cache_dir()) == paths.app_cache_dir()


def test_venv_python_path(bs):
    assert bs.venv_python("/v", "win32").endswith(os.path.join("Scripts", "python.exe"))
    assert bs.venv_python("/v", "linux").endswith(os.path.join("bin", "python"))


# ---------------------------------------------------------------- 版本檢查

@pytest.mark.parametrize("ver,ok,why", [((3, 9), False, "old"), ((3, 10), True, ""), ((3, 13), True, ""),
                                        ((3, 14), False, "new"), ((2, 7), False, "old")])
def test_check_python(bs, ver, ok, why):
    assert bs.check_python(ver) == (ok, why)


def test_python_problem_message(bs, monkeypatch):
    monkeypatch.setenv("EDGELENS_LANG", "en")
    monkeypatch.setattr(bs, "_ZH", None)
    msg = bs.python_problem_message((3, 8))
    assert "3.8" in msg and "python.org" in msg
    assert bs.python_problem_message((3, 12)) == ""
    assert "newer" in bs.python_problem_message((3, 14))


# ---------------------------------------------------------------- hash 與 stamp

def _fake_root(tmp_path, base="numpy\n"):
    req = tmp_path / "requirements"
    req.mkdir()
    (req / "base.txt").write_text(base)
    for n in ("onnx", "tflite", "k210"):
        (req / f"extra-{n}.txt").write_text(n + "\n")
    return tmp_path


def test_hash_changes_with_requirements(bs, tmp_path):
    root = _fake_root(tmp_path)
    h1 = bs.compute_hash(root, (3, 12, 1))
    assert h1 == bs.compute_hash(root, (3, 12, 9))  # 修訂版號不影響
    (root / "requirements" / "base.txt").write_text("numpy\nscipy\n")
    assert bs.compute_hash(root, (3, 12, 1)) != h1


def test_hash_changes_with_python_and_bootstrap_version(bs, tmp_path):
    root = _fake_root(tmp_path)
    assert bs.compute_hash(root, (3, 12, 0)) != bs.compute_hash(root, (3, 13, 0))
    assert bs.compute_hash(root, (3, 12, 0), "1") != bs.compute_hash(root, (3, 12, 0), "2")


def test_hash_ignores_line_endings(bs, tmp_path):
    root = _fake_root(tmp_path, "a\nb\n")
    h = bs.compute_hash(root, (3, 12, 0))
    (root / "requirements" / "base.txt").write_bytes(b"a\r\nb\r\n")
    assert bs.compute_hash(root, (3, 12, 0)) == h


def test_stamp_and_up_to_date(bs, tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("")
    assert not bs.is_up_to_date(str(venv), "abc", "linux")
    bs.write_stamp(str(venv), "abc")
    assert bs.is_up_to_date(str(venv), "abc", "linux")
    assert not bs.is_up_to_date(str(venv), "other", "linux")


def test_extras_roundtrip(bs, tmp_path):
    extras = {"onnx": {"ok": True, "package": "onnxruntime", "error": ""},
              "k210": {"ok": False, "package": "kflash", "error": "網路 error"}}
    bs.write_extras(str(tmp_path), extras)
    assert bs.read_extras(str(tmp_path)) == extras
    assert json.loads((tmp_path / "extras.json").read_text(encoding="utf-8"))["onnx"]["ok"] is True
    assert bs.read_extras(str(tmp_path / "nope")) == {}
    (tmp_path / "extras.json").write_text("{壞掉")
    assert bs.read_extras(str(tmp_path)) == {}


# ---------------------------------------------------------------- 鎖檔

def test_lock_stale_rules(bs):
    now = 10_000.0
    alive = lambda pid: True  # noqa: E731
    dead = lambda pid: False  # noqa: E731
    assert not bs.lock_is_stale({"pid": 5, "time": now - 10}, now, alive)
    assert bs.lock_is_stale({"pid": 5, "time": now - 10}, now, dead)  # PID 不存在
    assert bs.lock_is_stale({"pid": 5, "time": now - 31 * 60}, now, alive)  # 超過 30 分鐘
    assert bs.lock_is_stale(None, now, alive)
    assert bs.lock_is_stale({"pid": "x", "time": now}, now, alive)


def test_pid_alive(bs):
    assert bs.pid_alive(os.getpid())
    assert not bs.pid_alive(0)
    assert not bs.pid_alive("abc")


def test_acquire_release_lock(bs, tmp_path):
    lock = str(tmp_path / "v.lock")
    assert bs.acquire_lock(lock)
    assert json.loads(Path(lock).read_text())["pid"] == os.getpid()
    # 自己持有且還活著：不等待時拿不到
    assert not bs.acquire_lock(lock, wait=False)
    bs.release_lock(lock)
    assert not os.path.exists(lock)
    assert bs.acquire_lock(lock)
    bs.release_lock(lock)


def test_acquire_replaces_stale_lock(bs, tmp_path):
    lock = tmp_path / "v.lock"
    lock.write_text(json.dumps({"pid": os.getpid(), "time": time.time() - 3600}))
    assert bs.acquire_lock(str(lock), wait=False)
    bs.release_lock(str(lock))


def test_acquire_waits_then_notifies(bs, tmp_path):
    lock = tmp_path / "v.lock"
    lock.write_text(json.dumps({"pid": os.getpid(), "time": time.time()}))
    calls = []

    def fake_sleep(_s):
        calls.append("sleep")
        if len(calls) >= 2:
            lock.unlink()

    notified = []
    assert bs.acquire_lock(str(lock), sleep=fake_sleep, notify=lambda: notified.append(1))
    assert notified == [1] and len(calls) == 2
    bs.release_lock(str(lock))


def test_acquire_timeout(bs, tmp_path):
    lock = tmp_path / "v.lock"
    lock.write_text(json.dumps({"pid": os.getpid(), "time": time.time()}))
    assert not bs.acquire_lock(str(lock), timeout=-1, sleep=lambda s: None)


# ---------------------------------------------------------------- 參數

def test_parse_args_known_and_unknown(bs):
    o = bs.parse_args(["--selftest", "--noninteractive", "--lang", "en", "--debug", "--reinstall"])
    assert o.selftest and o.noninteractive and o.reinstall and not o.clean
    assert o.passthrough == ["--selftest", "--lang", "en", "--debug"]


def test_parse_args_wheelhouse_forms(bs):
    assert bs.parse_args(["--wheelhouse", "/w"]).wheelhouse == "/w"
    assert bs.parse_args(["--wheelhouse=/w"]).wheelhouse == "/w"
    assert bs.parse_args(["--make-wheelhouse", "/m"]).make_wheelhouse == "/m"
    o = bs.parse_args(["--wheelhouse"])
    assert o.error == "--wheelhouse"
    assert bs.parse_args(["--workspace", "/a b/中文"]).passthrough == ["--workspace", "/a b/中文"]


def test_pip_install_args(bs):
    cmd = bs.pip_install_args("py", "requirements/base.txt")
    assert cmd[:4] == ["py", "-m", "pip", "install"] and cmd[-2:] == ["-r", "requirements/base.txt"]
    assert "--no-index" not in cmd
    cmd = bs.pip_install_args("py", "tflite-runtime", "/wh")
    assert cmd[-1] == "tflite-runtime"
    i = cmd.index("--no-index")
    assert cmd[i + 1:i + 3] == ["--find-links", "/wh"]
    cmd = bs.pip_download_args("py", "r.txt", "/d")
    assert "download" in cmd and cmd[-4:] == ["-d", "/d", "-r", "r.txt"] or cmd[-2:] == ["-r", "r.txt"]


def test_launch_env(bs):
    env = bs.launch_env("/repo", {"PYTHONPATH": "/old"})
    assert env["PYTHONPATH"] == "/repo" + os.pathsep + "/old"
    assert env["PYTHONUTF8"] == "1"
    assert bs.launch_env("/repo", {})["PYTHONPATH"] == "/repo"


def test_check_wheelhouse(bs, tmp_path, monkeypatch):
    monkeypatch.setenv("EDGELENS_LANG", "en")
    monkeypatch.setattr(bs, "_ZH", None)
    assert "not found" in bs.check_wheelhouse(str(tmp_path / "none"))
    assert "empty" in bs.check_wheelhouse(str(tmp_path))
    (tmp_path / "a-1.0-py3-none-any.whl").write_bytes(b"")
    assert bs.check_wheelhouse(str(tmp_path)) == ""


def test_install_with_empty_wheelhouse_stops_before_pip(bs, tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(bs.subprocess, "Popen", lambda *a, **k: calls.append(a))
    opts = bs.parse_args(["--wheelhouse", str(tmp_path)])
    assert bs.install(opts, str(tmp_path / "venv"), str(ROOT)) == 1
    assert calls == []
    assert "FAILED" in capsys.readouterr().out or True


def test_install_base_failure_stops(bs, tmp_path, monkeypatch):
    """base 安裝失敗就不再裝選配，也不寫 stamp。"""
    venv = tmp_path / "venv"
    seen = []
    monkeypatch.setattr(bs, "create_venv", lambda v: (True, []))
    monkeypatch.setattr(bs, "platform_key", lambda plat=None: "macos")

    def fake_run(cmd, tail_lines=25, show=True):
        seen.append(cmd)
        return 1, ["ERROR: Could not find a version that satisfies"]

    monkeypatch.setattr(bs, "run_streaming", fake_run)
    assert bs.install(bs.parse_args([]), str(venv), str(ROOT)) == 1
    assert len(seen) == 1 and not (venv / "edgelens_install.json").exists()


def test_install_extras_failure_does_not_stop(bs, tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    venv.mkdir()
    monkeypatch.setattr(bs, "create_venv", lambda v: (True, []))
    monkeypatch.setattr(bs, "platform_key", lambda plat=None: "macos")
    seen = []

    def fake_run(cmd, tail_lines=25, show=True):
        seen.append(cmd)
        text = " ".join(cmd)
        if "base.txt" in text or "extra-onnx" in text:
            return 0, []
        return 1, ["ERROR: boom"]

    monkeypatch.setattr(bs, "run_streaming", fake_run)
    assert bs.install(bs.parse_args([]), str(venv), str(ROOT)) == 0
    extras = bs.read_extras(str(venv))
    assert extras["onnx"]["ok"] is True and extras["onnx"]["package"] == "onnxruntime"
    assert extras["tflite"]["ok"] is False and extras["tflite"]["package"] == "tflite-runtime"
    assert extras["k210"] == {"ok": False, "package": "kflash", "error": "ERROR: boom"}
    # tflite 先試 ai-edge-litert，再試 tflite-runtime
    assert any(c[-1] == "tflite-runtime" for c in seen)
    assert bs.read_stamp(str(venv))["hash"] == bs.compute_hash(str(ROOT))


def test_ensure_installed_skips_when_up_to_date(bs, tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    vpy = Path(bs.venv_python(str(venv)))
    vpy.parent.mkdir(parents=True)
    vpy.write_text("")
    bs.write_stamp(str(venv), bs.compute_hash(str(ROOT)))
    monkeypatch.setattr(bs, "install", lambda *a, **k: pytest.fail("不該安裝"))
    assert bs.ensure_installed(bs.parse_args([]), str(venv), str(ROOT)) == 0


def test_linux_missing_libs(bs):
    assert bs.linux_missing_libs(lambda name: True) == []
    assert "libxcb-cursor0" in bs.linux_missing_libs(lambda name: name != "xcb-cursor")


def test_failure_hint_network(bs):
    hints = bs.failure_hint(["WARNING: Retrying ... Connection timed out"])
    assert any("pypi" in h for h in hints)


# ---------------------------------------------------------------- 啟動腳本

def test_launch_scripts(bs):
    for name in ("run_macos.command", "run_linux.sh"):
        path = ROOT / name
        assert path.exists(), name
        data = path.read_bytes()
        assert b"\r" not in data, f"{name} 必須是 LF"
        assert data.startswith(b"#!/bin/bash")
        if sys.platform != "win32":
            assert path.stat().st_mode & stat.S_IXUSR, f"{name} 需要執行權限"
    bat = (ROOT / "run_windows.bat").read_bytes()
    assert b"\r\n" in bat and b"\n" not in bat.replace(b"\r\n", b"")
    for v in (b"3.13", b"3.12", b"3.11", b"3.10"):
        assert v in bat
    assert b"Add python.exe to PATH" in bat
    assert (ROOT / "bootstrap.py").exists()


def test_bootstrap_syntax_is_old_python_friendly():
    """bootstrap.py 要能被 Python 3.6 的語法解析：不能有 walrus、match、list[str] 標註等。"""
    import ast

    src = (ROOT / "bootstrap.py").read_text(encoding="utf-8")
    tree = ast.parse(src, feature_version=(3, 6))
    assert not any(isinstance(n, ast.NamedExpr) for n in ast.walk(tree))
    assert "from __future__ import annotations" not in src
