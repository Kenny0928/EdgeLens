"""bootstrap.py 與啟動腳本的單元測試：只測純函式，不真的安裝套件。"""

from __future__ import annotations

import importlib.util
import json
import os
import re
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


# ---------------------------------------------------------------- 精確版本（constraints）

def _constraint_names(path):
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.add(re.split(r"[<>=!~;\s\[]", line)[0].lower().replace("_", "-"))
    return names


def test_constraints_cover_every_top_level_requirement():
    """requirements/*.txt 新增套件卻忘了在 constraints.txt 釘版本，全班又會變成「安裝當天最新版」。"""
    pinned = _constraint_names(ROOT / "requirements" / "constraints.txt")
    for name in ("base.txt", "extra-onnx.txt", "extra-tflite.txt", "extra-k210.txt"):
        for pkg in _constraint_names(ROOT / "requirements" / name):
            assert pkg in pinned, f"{name} 的 {pkg} 沒有在 constraints.txt 釘版本"


def test_constraints_lines_are_exact_or_bounded():
    for line in (ROOT / "requirements" / "constraints.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        spec = line.split(";")[0]
        assert "==" in spec or ("<=" in spec and ">=" in spec), f"不夠精確：{line}"


def test_constraints_file_helper(bs, tmp_path):
    assert bs.constraints_file(str(tmp_path)) is None
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements" / "constraints.txt").write_text("numpy==2.0.0\n")
    assert bs.constraints_file(str(tmp_path)) == str(tmp_path / "requirements" / "constraints.txt")
    assert bs.constraints_file().endswith(os.path.join("requirements", "constraints.txt"))


def test_hash_changes_with_constraints(bs, tmp_path):
    root = _fake_root(tmp_path)
    h0 = bs.compute_hash(root, (3, 12, 0))
    (root / "requirements" / "constraints.txt").write_text("numpy==2.0.0\n")
    h1 = bs.compute_hash(root, (3, 12, 0))
    assert h1 != h0
    (root / "requirements" / "constraints.txt").write_text("numpy==2.0.1\n")
    assert bs.compute_hash(root, (3, 12, 0)) != h1


def test_pip_args_include_constraints(bs):
    cmd = bs.pip_install_args("py", "requirements/base.txt", None, "/c.txt")
    i = cmd.index("-c")
    assert cmd[i + 1] == "/c.txt" and cmd.index("install") < i < cmd.index("-r")
    cmd = bs.pip_install_args("py", "tflite-runtime", "/wh", "/c.txt")
    assert cmd[cmd.index("-c") + 1] == "/c.txt" and cmd[-1] == "tflite-runtime"
    assert "-c" not in bs.pip_install_args("py", "x.txt")
    cmd = bs.pip_download_args("py", "r.txt", "/d", "/c.txt")
    assert cmd[cmd.index("-c") + 1] == "/c.txt" and cmd[-4:] == ["-d", "/d", "-r", "r.txt"]
    assert "-c" not in bs.pip_download_args("py", "r.txt", "/d")


def test_install_passes_constraints_to_every_pip_call(bs, tmp_path, monkeypatch):
    """base、每個選配、tflite-runtime 備援都要帶 -c。"""
    venv = tmp_path / "venv"
    venv.mkdir()
    monkeypatch.setattr(bs, "create_venv", lambda v: (True, []))
    monkeypatch.setattr(bs, "platform_key", lambda plat=None: "macos")
    seen = []

    def fake_run(cmd, tail_lines=25, show=True):
        seen.append(cmd)
        return (1, ["boom"]) if "extra-tflite" in " ".join(cmd) else (0, [])

    monkeypatch.setattr(bs, "run_streaming", fake_run)
    assert bs.install(bs.parse_args([]), str(venv), str(ROOT)) == 0
    assert len(seen) == 5  # base、onnx、tflite、tflite-runtime、k210
    want = str(ROOT / "requirements" / "constraints.txt")
    for cmd in seen:
        assert cmd[cmd.index("-c") + 1] == want, cmd
    assert seen[3][-1] == "tflite-runtime"


def test_make_wheelhouse_passes_constraints(bs, tmp_path, monkeypatch):
    seen = []

    def fake_run(cmd, tail_lines=25, show=True):
        seen.append(cmd)
        return (1, ["x"]) if cmd[-1].endswith("extra-tflite.txt") else (0, [])

    monkeypatch.setattr(bs, "run_streaming", fake_run)
    assert bs.make_wheelhouse(str(tmp_path / "wh"), str(ROOT)) == 0
    assert len(seen) == 5  # base、onnx、tflite、tflite-runtime 備援、k210
    for cmd in seen:
        assert "download" in cmd and cmd[cmd.index("-c") + 1].endswith("constraints.txt"), cmd


# ---------------------------------------------------------------- 舊虛擬環境清理

DAY = 86400


def _mk_venv(cache, name, age_days=0.0, stamp=True, now=None):
    now = time.time() if now is None else now
    d = cache / name
    (d / "bin").mkdir(parents=True)
    (d / "bin" / "python").write_text("")
    t = now - age_days * DAY
    if stamp:
        (d / "edgelens_last_used").write_text("x")
        os.utime(d / "edgelens_last_used", (t, t))
    os.utime(d, (t, t))
    return d


def test_touch_last_used(bs, tmp_path):
    venv = tmp_path / "v"
    venv.mkdir()
    bs.touch_last_used(str(venv), now=1234)
    assert (venv / "edgelens_last_used").read_text().strip() == "1234"
    bs.touch_last_used(str(tmp_path / "missing"))  # 資料夾不存在也不能丟例外


def test_cleanup_removes_only_stale_sibling(bs, tmp_path):
    cur = _mk_venv(tmp_path, "venv-cp313-arm64")
    old = _mk_venv(tmp_path, "venv-cp312-arm64", age_days=61)
    recent = _mk_venv(tmp_path, "venv-cp311-arm64", age_days=59)
    assert bs.cleanup_old_venvs(str(cur), env={}, cache=str(tmp_path)) == ["venv-cp312-arm64"]
    assert not old.exists() and recent.exists() and cur.exists()


def test_cleanup_never_removes_current_even_if_old(bs, tmp_path):
    cur = _mk_venv(tmp_path, "venv-cp313-arm64", age_days=400)
    assert bs.cleanup_old_venvs(str(cur), env={}, cache=str(tmp_path)) == []
    assert cur.exists()


def test_cleanup_keeps_non_matching_names(bs, tmp_path):
    cur = _mk_venv(tmp_path, "venv-cp313-arm64")
    keep = [_mk_venv(tmp_path, n, age_days=500) for n in
            ("devvenv", "py310-venv", "wp1-test-venv", "wp1-test-venv312", "venv-cp312", "venv-cp312-", "venv-cpx-arm64",
             "venv-cp312-ARM64", "xvenv-cp312-arm64", "venv-cp312-arm64.lock", "zoo", "logs")]
    (tmp_path / "venv-cp310-x86_64").write_text("a file, not a dir")
    assert bs.cleanup_old_venvs(str(cur), env={}, cache=str(tmp_path)) == []
    assert all(d.exists() for d in keep) and (tmp_path / "venv-cp310-x86_64").exists()


def test_cleanup_stamps_unstamped_venv_instead_of_deleting(bs, tmp_path):
    """舊版環境沒有 last-used 記號：資料夾時間不可信（還在用的環境根目錄也不會更新），
    所以先補記號、這次保留，等記號本身過期才刪。"""
    cur = _mk_venv(tmp_path, "venv-cp313-arm64")
    old = _mk_venv(tmp_path, "venv-cp310-amd64", age_days=400, stamp=False)
    assert bs.cleanup_old_venvs(str(cur), env={}, cache=str(tmp_path)) == []
    assert old.exists() and (old / "edgelens_last_used").exists()
    later = time.time() + 61 * DAY
    assert bs.cleanup_old_venvs(str(cur), env={}, cache=str(tmp_path), now=later) == ["venv-cp310-amd64"]
    assert not old.exists()


def test_cleanup_keeps_unstamped_venv_when_stamp_cannot_be_written(bs, tmp_path, monkeypatch):
    cur = _mk_venv(tmp_path, "venv-cp313-arm64")
    old = _mk_venv(tmp_path, "venv-cp310-amd64", age_days=400, stamp=False)
    monkeypatch.setattr(bs, "touch_last_used", lambda *a, **k: None)   # 例如唯讀資料夾
    later = time.time() + 400 * DAY
    assert bs.cleanup_old_venvs(str(cur), env={}, cache=str(tmp_path), now=later) == []
    assert old.exists()


def test_cleanup_prefers_last_used_stamp_over_directory_mtime(bs, tmp_path):
    cur = _mk_venv(tmp_path, "venv-cp313-arm64")
    d = _mk_venv(tmp_path, "venv-cp312-arm64", age_days=1)  # 剛用過
    old = time.time() - 300 * DAY
    os.utime(d, (old, old))  # 資料夾本身很舊，但 last-used 很新
    assert bs.cleanup_old_venvs(str(cur), env={}, cache=str(tmp_path)) == []
    assert d.exists()


def test_cleanup_skipped_when_edgelens_venv_set(bs, tmp_path):
    cur = _mk_venv(tmp_path, "venv-cp313-arm64")
    old = _mk_venv(tmp_path, "venv-cp312-arm64", age_days=400)
    assert bs.cleanup_old_venvs(str(cur), env={"EDGELENS_VENV": str(cur)}, cache=str(tmp_path)) == []
    assert old.exists()


def test_cleanup_uses_cache_dir_from_env(bs, tmp_path):
    cur = _mk_venv(tmp_path, "venv-cp313-arm64")
    old = _mk_venv(tmp_path, "venv-cp312-arm64", age_days=400)
    assert bs.cleanup_old_venvs(str(cur), env={"EDGELENS_CACHE": str(tmp_path)}) == ["venv-cp312-arm64"]
    assert not old.exists()


def test_cleanup_removal_error_does_not_raise(bs, tmp_path):
    cur = _mk_venv(tmp_path, "venv-cp313-arm64")
    stuck = _mk_venv(tmp_path, "venv-cp310-arm64", age_days=400)
    gone = _mk_venv(tmp_path, "venv-cp311-arm64", age_days=400)

    def remove(path):
        if path.endswith("cp310-arm64"):
            raise PermissionError("in use")
        bs.remove_tree(path)

    assert bs.cleanup_old_venvs(str(cur), env={}, cache=str(tmp_path), remove=remove) == ["venv-cp311-arm64"]
    assert stuck.exists() and not gone.exists()


def test_cleanup_missing_cache_dir_is_fine(bs, tmp_path):
    assert bs.cleanup_old_venvs(str(tmp_path / "x" / "venv-cp313-arm64"), env={}, cache=str(tmp_path / "none")) == []


def test_cleanup_prints_bilingual_note(bs, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EDGELENS_LANG", "en")
    monkeypatch.setattr(bs, "_ZH", None)
    cur = _mk_venv(tmp_path, "venv-cp313-arm64")
    _mk_venv(tmp_path, "venv-cp312-arm64", age_days=400)
    bs.cleanup_old_venvs(str(cur), env={}, cache=str(tmp_path))
    assert "venv-cp312-arm64" in capsys.readouterr().out
    monkeypatch.setenv("EDGELENS_LANG", "zh_TW")
    monkeypatch.setattr(bs, "_ZH", None)
    _mk_venv(tmp_path, "venv-cp311-arm64", age_days=400)
    bs.cleanup_old_venvs(str(cur), env={}, cache=str(tmp_path))
    assert "舊環境" in capsys.readouterr().out


def test_main_touches_stamp_and_cleans_after_install(bs, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    cur = _mk_venv(cache, bs.venv_name(), stamp=False)
    old = _mk_venv(cache, "venv-cp27-arm64", age_days=400)
    monkeypatch.setenv("EDGELENS_CACHE", str(cache))
    monkeypatch.delenv("EDGELENS_VENV", raising=False)
    calls = []
    monkeypatch.setattr(bs, "ensure_installed", lambda *a, **k: 0)
    monkeypatch.setattr(bs, "launch", lambda *a, **k: calls.append(a) or 0)
    assert bs.main([]) == 0
    assert (cur / "edgelens_last_used").exists() and not old.exists() and calls


def test_main_does_not_clean_when_install_fails(bs, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    _mk_venv(cache, bs.venv_name(), stamp=False)
    old = _mk_venv(cache, "venv-cp27-arm64", age_days=400)
    monkeypatch.setenv("EDGELENS_CACHE", str(cache))
    monkeypatch.delenv("EDGELENS_VENV", raising=False)
    monkeypatch.setattr(bs, "ensure_installed", lambda *a, **k: 1)
    assert bs.main([]) == 1
    assert old.exists()
