#!/usr/bin/env python3
# ruff: noqa: UP004, UP021, UP031
# 上面三個規則會要求新語法（text=、% 改 format 等）；bootstrap 刻意保持老派寫法以相容舊 Python。
"""EdgeLens 啟動程式：找到（或建立）虛擬環境、安裝套件，然後開啟 EdgeLens。

雙擊 run_windows.bat／run_macos.command／run_linux.sh 就會執行這個檔案。
- 只用標準函式庫，不能 import edgelens（那時候套件還沒裝）。
- 語法保持在 Python 3.6 也能解析（不用 walrus、list[str] 之類的新語法），
  這樣太舊的 Python 才能執行到版本檢查，印出友善的說明而不是 SyntaxError。
- 虛擬環境放在使用者的快取資料夾，不放在專案資料夾裡：避免 iCloud 把 . 開頭資料夾裡的
  檔案設成隱藏、也避免 OneDrive 同步幾萬個小檔案。
"""

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time

BOOTSTRAP_VERSION = "1"
MIN_PY = (3, 10)
MAX_PY = (3, 13)
APP_DIR_NAME = "EdgeLens"
PYTHON_URL = "https://www.python.org/downloads/"
LOCK_MAX_AGE = 30 * 60  # 鎖檔超過 30 分鐘視為過期
LOCK_POLL = 2.0
STAMP_NAME = "edgelens_install.json"
EXTRAS_NAME = "extras.json"
LAST_USED_NAME = "edgelens_last_used"  # 每次成功啟動就更新；用來判斷舊環境是不是已經沒人用
VENV_NAME_RE = re.compile(r"^venv-cp\d+-[a-z0-9_-]+$")  # 只清 venv_name() 產生的名稱，其他資料夾一律不碰
STALE_VENV_DAYS = 60
EXTRA_KEYS = ("onnx", "tflite", "k210")
TOTAL_STEPS = 5

ROOT = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- 訊息

def _detect_zh():
    """系統語言是中文就回傳 True。EDGELENS_LANG=en／zh_TW 可強制指定。"""
    forced = os.environ.get("EDGELENS_LANG", "").lower()
    if forced:
        return forced.startswith("zh")
    for name in ("LC_ALL", "LC_MESSAGES", "LANGUAGE", "LANG"):
        val = os.environ.get(name, "")
        if val and val not in ("C", "POSIX"):
            return val.lower().startswith("zh")
    try:
        if sys.platform.startswith("win"):
            import ctypes

            return (ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0xFF) == 0x04
        if sys.platform == "darwin":
            out = subprocess.run(["defaults", "read", "-g", "AppleLocale"], stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, universal_newlines=True, timeout=5).stdout
            return out.strip().lower().startswith("zh")
    except Exception:  # noqa: BLE001 - 偵測不到就用英文
        pass
    return False


_ZH = None


def t(zh, en):
    """依系統語言回傳中文或英文。"""
    global _ZH
    if _ZH is None:
        _ZH = _detect_zh()
    return zh if _ZH else en


def say(zh, en=None):
    print(t(zh, en if en is not None else zh))
    sys.stdout.flush()


def step(i, zh, en):
    say("[%d/%d] %s" % (i, TOTAL_STEPS, zh), "[%d/%d] %s" % (i, TOTAL_STEPS, en))


def _setup_stdio():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- 純函式（有單元測試）

def check_python(version=None):
    """回傳 (ok, 原因代碼)。原因：'old'、'new' 或 ''。"""
    v = tuple(version if version is not None else sys.version_info[:2])[:2]
    if v < MIN_PY:
        return False, "old"
    if v > MAX_PY:
        return False, "new"
    return True, ""


def python_problem_message(version):
    """版本不符時要顯示的說明（中英文依系統語言）。"""
    ok, why = check_python(version)
    cur = "%d.%d" % tuple(version[:2])
    if ok:
        return ""
    if why == "old":
        return t(
            "這台電腦的 Python 是 %s，太舊了。EdgeLens 需要 Python 3.10 到 3.13。\n"
            "請到 %s 下載 Python 3.12，安裝時記得勾選「Add python.exe to PATH」，再重新雙擊。" % (cur, PYTHON_URL),
            "Your Python is %s, which is too old. EdgeLens needs Python 3.10 to 3.13.\n"
            "Download Python 3.12 from %s (on Windows tick \"Add python.exe to PATH\") and try again." % (cur, PYTHON_URL))
    return t(
        "這台電腦的 Python 是 %s，比 EdgeLens 支援的新（需要 3.10 到 3.13）。\n"
        "請到 %s 另外安裝 Python 3.12（舊版可以並存），再重新雙擊。" % (cur, PYTHON_URL),
        "Your Python is %s, which is newer than EdgeLens supports (3.10 to 3.13).\n"
        "Install Python 3.12 from %s alongside it and try again." % (cur, PYTHON_URL))


def platform_key(plat=None):
    plat = plat if plat is not None else sys.platform
    if plat.startswith("win"):
        return "windows"
    if plat == "darwin":
        return "macos"
    return "linux"


def cache_dir(plat=None, env=None, home=None):
    """要和 edgelens/core/paths.py 的 app_cache_dir() 保持一致（改一邊就要改另一邊）。"""
    env = os.environ if env is None else env
    home = home if home is not None else os.path.expanduser("~")
    if env.get("EDGELENS_CACHE"):
        return env["EDGELENS_CACHE"]
    p = platform_key(plat)
    if p == "windows":
        base = env.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
        return os.path.join(base, APP_DIR_NAME, "Cache")
    if p == "macos":
        return os.path.join(home, "Library", "Caches", APP_DIR_NAME)
    base = env.get("XDG_CACHE_HOME") or os.path.join(home, ".cache")
    return os.path.join(base, APP_DIR_NAME)


def venv_name(version=None, machine=None):
    """例如 venv-cp312-arm64。Python 版本或 CPU 不同要用不同的環境。"""
    v = version if version is not None else sys.version_info[:2]
    m = (machine if machine is not None else platform.machine() or "unknown").lower()
    m = "".join(c if c.isalnum() or c in "_-" else "_" for c in m) or "unknown"
    return "venv-cp%d%d-%s" % (v[0], v[1], m)


def venv_dir(plat=None, env=None, home=None, version=None, machine=None):
    env = os.environ if env is None else env
    if env.get("EDGELENS_VENV"):
        return env["EDGELENS_VENV"]
    return os.path.join(cache_dir(plat, env, home), venv_name(version, machine))


def venv_python(venv, plat=None):
    if platform_key(plat) == "windows":
        return os.path.join(venv, "Scripts", "python.exe")
    return os.path.join(venv, "bin", "python")


def requirements_files(root=None):
    """會影響安裝結果的 requirements 檔（dev.txt 是開發用，不算）。"""
    req = os.path.join(root or ROOT, "requirements")
    names = ["base.txt", "extra-onnx.txt", "extra-tflite.txt", "extra-k210.txt", "constraints.txt"]
    return [os.path.join(req, n) for n in names]


def constraints_file(root=None):
    """精確版本檔的路徑；檔案不存在（例如被刪掉）就回傳 None，等於不限制版本。
    每個 pip install／download 都要帶 -c 這個檔案，全班才會裝到同一組、測試過的版本。"""
    path = os.path.join(root or ROOT, "requirements", "constraints.txt")
    return path if os.path.isfile(path) else None


def compute_hash(root=None, version=None, bootstrap_version=BOOTSTRAP_VERSION):
    """requirements 內容 + Python 版本 + bootstrap 版本 的雜湊；相同就不用再安裝。"""
    v = version if version is not None else sys.version_info[:3]
    h = hashlib.sha256()
    h.update(("bootstrap=%s\npython=%d.%d\n" % (bootstrap_version, v[0], v[1])).encode("utf-8"))
    for path in requirements_files(root):
        h.update(("\n--%s--\n" % os.path.basename(path)).encode("utf-8"))
        try:
            with open(path, "rb") as f:
                h.update(f.read().replace(b"\r\n", b"\n"))
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()


def read_stamp(venv):
    try:
        with open(os.path.join(venv, STAMP_NAME), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_stamp(venv, digest):
    _write_json(os.path.join(venv, STAMP_NAME), {"hash": digest, "bootstrap": BOOTSTRAP_VERSION,
                                                 "python": platform.python_version(), "time": time.time()})


def is_up_to_date(venv, digest, plat=None):
    return (os.path.exists(venv_python(venv, plat)) and read_stamp(venv).get("hash") == digest)


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def write_extras(venv, extras):
    _write_json(os.path.join(venv, EXTRAS_NAME), extras)


def read_extras(venv):
    try:
        with open(os.path.join(venv, EXTRAS_NAME), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


class Options(object):
    """命令列參數。沒看過的參數全部原樣放進 passthrough，交給 python -m edgelens。"""

    def __init__(self):
        self.selftest = False
        self.reinstall = False
        self.clean = False
        self.noninteractive = False
        self.wheelhouse = None
        self.make_wheelhouse = None
        self.passthrough = []
        self.error = ""


def parse_args(argv):
    o = Options()
    args = list(argv)
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--selftest":
            o.selftest = True
            o.passthrough.append(a)  # edgelens 本身也要知道
        elif a == "--reinstall":
            o.reinstall = True
        elif a == "--clean":
            o.clean = True
        elif a == "--noninteractive":
            o.noninteractive = True
        elif a in ("--wheelhouse", "--make-wheelhouse") or a.startswith(("--wheelhouse=", "--make-wheelhouse=")):
            name, eq, val = a.partition("=")
            if not eq:
                if i + 1 >= len(args):
                    o.error = name
                    break
                i += 1
                val = args[i]
            setattr(o, name[2:].replace("-", "_"), val)
        else:
            o.passthrough.append(a)
        i += 1
    return o


def pip_install_args(python, target, wheelhouse=None, constraints=None):
    """target：requirements 檔路徑（以 -r 安裝）或單一套件規格（例如 tflite-runtime）。"""
    cmd = [python, "-m", "pip", "install", "--disable-pip-version-check", "--progress-bar", "off"]
    if constraints:
        cmd += ["-c", constraints]
    if wheelhouse:
        cmd += ["--no-index", "--find-links", wheelhouse]
    if os.path.isfile(target) or target.endswith(".txt"):
        cmd += ["-r", target]
    else:
        cmd.append(target)
    return cmd


def pip_download_args(python, requirements_file, dest, constraints=None):
    cmd = [python, "-m", "pip", "download", "--disable-pip-version-check", "--progress-bar", "off"]
    if constraints:
        cmd += ["-c", constraints]
    return cmd + ["-d", dest, "-r", requirements_file]


def launch_env(root=None, env=None):
    new = dict(os.environ if env is None else env)
    root = root or ROOT
    old = new.get("PYTHONPATH")
    new["PYTHONPATH"] = root + (os.pathsep + old if old else "")
    new["PYTHONUTF8"] = "1"
    return new


# --------------------------------------------------------------------------- 鎖檔

def pid_alive(pid):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if sys.platform.startswith("win"):
        try:
            import ctypes

            k32 = ctypes.windll.kernel32
            handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return False
            code = ctypes.c_ulong()
            ok = k32.GetExitCodeProcess(handle, ctypes.byref(code))
            k32.CloseHandle(handle)
            return bool(ok) and code.value == 259  # STILL_ACTIVE
        except Exception:  # noqa: BLE001
            return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def lock_is_stale(info, now=None, alive=None, max_age=LOCK_MAX_AGE):
    """鎖檔內容不完整、PID 已不存在、或超過 max_age 秒，都算過期。"""
    now = time.time() if now is None else now
    alive = pid_alive if alive is None else alive
    if not isinstance(info, dict):
        return True
    pid, stamp = info.get("pid"), info.get("time")
    if not isinstance(pid, int) or not isinstance(stamp, (int, float)):
        return True
    if now - stamp > max_age:
        return True
    return not alive(pid)


def _read_lock(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def acquire_lock(path, wait=True, timeout=LOCK_MAX_AGE, sleep=time.sleep, notify=None):
    """取得安裝鎖。別的實例正在安裝時等待；成功回傳 True，等太久回傳 False。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    deadline = time.time() + timeout
    told = False
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            info = _read_lock(path)
            try:
                fresh = time.time() - os.path.getmtime(path) < 5
            except OSError:
                fresh = False
            if lock_is_stale(info) and not (info is None and fresh):  # 剛建立還沒寫完的鎖不算過期
                try:
                    os.remove(path)
                except OSError:
                    pass
                continue
            if not wait or time.time() > deadline:
                return False
            if not told and notify:
                notify()
                told = True
            sleep(LOCK_POLL)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "time": time.time()}, f)
        return True


def release_lock(path):
    info = _read_lock(path)
    if isinstance(info, dict) and info.get("pid") not in (None, os.getpid()):
        return  # 不是我的鎖
    try:
        os.remove(path)
    except OSError:
        pass


# --------------------------------------------------------------------------- 執行 pip 與子程序

def run_streaming(cmd, tail_lines=25, show=True):
    """執行指令，回傳 (結束碼, 輸出最後幾行)。只印 pip 的重點進度，避免洗版。"""
    tail = []
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                                encoding="utf-8", errors="replace")
    except OSError as e:
        return 127, [str(e)]
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        tail.append(line)
        del tail[:-tail_lines]
        if show and line.startswith(("Collecting", "Downloading", "Installing collected", "Successfully")):
            print("    " + line[:100])
            sys.stdout.flush()
    proc.wait()
    return proc.returncode, tail


def failure_hint(tail, wheelhouse=None):
    text = "\n".join(tail).lower()
    hints = []
    if wheelhouse:
        hints.append(t("wheelhouse 資料夾裡缺少需要的 wheel 檔；請用同一種作業系統與 Python 版本的電腦執行 "
                       "--make-wheelhouse 重新準備。",
                       "The wheelhouse folder is missing needed wheels. Re-create it with --make-wheelhouse on a "
                       "computer with the same OS and Python version."))
    if any(k in text for k in ("connection", "timed out", "network", "resolution", "ssl", "retry", "proxy")):
        hints.append(t("網路好像有問題：確認網路連線（學校網路可能擋住 pypi.org），或改用 --wheelhouse 離線安裝。",
                       "Looks like a network problem: check your connection (school networks may block pypi.org) "
                       "or use --wheelhouse for an offline install."))
    if "no space" in text or "errno 28" in text:
        hints.append(t("磁碟空間不夠：安裝需要約 1 GB，請清出空間後再試。",
                       "Not enough disk space: the install needs about 1 GB. Free some space and try again."))
    if "no matching distribution" in text or "could not find a version" in text:
        hints.append(t("找不到適合這個 Python 的套件：請使用 Python 3.10 到 3.13（建議 3.12）。",
                       "No matching package for this Python: use Python 3.10 to 3.13 (3.12 recommended)."))
    if not hints:
        hints.append(t("請確認網路、磁碟空間與 Python 版本（3.10 到 3.13），然後再雙擊一次。",
                       "Check your network, disk space and Python version (3.10 to 3.13), then try again."))
    return hints


def remove_tree(path):
    def _onerror(func, p, _exc):
        try:
            os.chmod(p, 0o700)
            func(p)
        except OSError:
            pass

    if os.path.isdir(path):
        shutil.rmtree(path, onerror=_onerror)


# --------------------------------------------------------------------------- 舊環境清理

def touch_last_used(venv, now=None):
    """記下「這個環境最近一次成功啟動」。失敗（唯讀、被防毒擋住）不影響啟動。"""
    try:
        with open(os.path.join(venv, LAST_USED_NAME), "w") as f:
            f.write("%d\n" % (time.time() if now is None else now))
    except (OSError, ValueError):
        pass


def _last_used_time(path, now):
    """最近使用時間（last-used 檔的修改時間）。

    舊版建立的環境沒有這個檔：不能改看資料夾修改時間，因為 venv 根目錄在平常使用時不會更新，
    還在用的環境也會看起來很舊。所以第一次遇到時先補一個「現在」的記號、這次不刪；
    之後真的 STALE_VENV_DAYS 天沒被啟動才會被清掉。補不上記號就永遠不刪（寧可留著）。"""
    stamp = os.path.join(path, LAST_USED_NAME)
    try:
        return os.path.getmtime(stamp)
    except OSError:
        pass
    touch_last_used(path, now)
    return None


def stale_venvs(cache, current, now=None, days=STALE_VENV_DAYS):
    """cache 底下名稱符合 venv-cpXYZ-架構、不是 current、且超過 days 天沒用過的資料夾（完整路徑）。
    名稱用嚴格比對：開發者自己放在同一個快取資料夾的 devvenv、py310-venv 等絕對不會被選到。"""
    now = time.time() if now is None else now
    try:
        names = sorted(os.listdir(cache))
    except OSError:
        return []
    cur = os.path.realpath(current)
    found = []
    for name in names:
        path = os.path.join(cache, name)
        if not VENV_NAME_RE.match(name) or os.path.islink(path) or not os.path.isdir(path):
            continue
        if os.path.realpath(path) == cur:
            continue
        used = _last_used_time(path, now)
        if used is not None and now - used > days * 86400:
            found.append(path)
    return found


def cleanup_old_venvs(current, env=None, cache=None, now=None, remove=None):
    """Python 升級後，舊版的虛擬環境（約 650 MB）不會再被用到，超過 60 天就刪掉。
    自訂 EDGELENS_VENV 時完全不動（那是使用者自己管的）。任何錯誤都吞掉：清理失敗絕不能擋住啟動。
    回傳實際刪掉的資料夾名稱。"""
    env = os.environ if env is None else env
    if env.get("EDGELENS_VENV"):
        return []
    remove = remove_tree if remove is None else remove
    removed = []
    try:
        for path in stale_venvs(cache if cache is not None else cache_dir(env=env), current, now):
            try:
                remove(path)
            except Exception:  # noqa: BLE001 - 刪不掉（檔案被占用等）下次再試
                continue
            if not os.path.exists(path):
                removed.append(os.path.basename(path))
    except Exception:  # noqa: BLE001
        pass
    if removed:
        say("已刪除超過 %d 天沒用過的舊環境（釋放硬碟空間）：%s" % (STALE_VENV_DAYS, ", ".join(removed)),
            "Removed old environments unused for over %d days (frees disk space): %s"
            % (STALE_VENV_DAYS, ", ".join(removed)))
    return removed


# --------------------------------------------------------------------------- Linux 檢查

_LINUX_LIBS = (
    ("xcb-cursor", "libxcb-cursor0"),
    ("xkbcommon-x11", "libxkbcommon-x11-0"),
    ("EGL", "libegl1"),
    ("GL", "libgl1"),
)


def _lib_present(name):
    try:
        import ctypes.util

        if ctypes.util.find_library(name):
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        out = subprocess.run(["ldconfig", "-p"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             universal_newlines=True, timeout=10).stdout
        return ("lib%s.so" % name) in out
    except Exception:  # noqa: BLE001
        return False


def linux_missing_libs(present=_lib_present):
    """回傳缺少的 apt 套件名稱清單。"""
    return [apt for lib, apt in _LINUX_LIBS if not present(lib)]


def in_dialout_group():
    """回傳 True／False；系統沒有 dialout 群組時回傳 None。"""
    try:
        import grp

        gid = grp.getgrnam("dialout").gr_gid
    except (ImportError, KeyError):
        return None
    try:
        return gid in os.getgroups()
    except OSError:
        return None


def have_venv_module():
    try:
        import importlib.util

        return importlib.util.find_spec("venv") is not None and importlib.util.find_spec("ensurepip") is not None
    except Exception:  # noqa: BLE001
        return False


def venv_apt_hint():
    return "sudo apt install python3-venv python%d.%d-venv" % sys.version_info[:2]


def linux_checks(env=None):
    """提示 Linux 常見問題。只提示，不中斷（缺 venv 模組由建立 venv 時處理）。"""
    env = os.environ if env is None else env
    missing = linux_missing_libs()
    if missing:
        say("提示：缺少 Qt 需要的系統函式庫，視窗可能打不開。請執行：\n    sudo apt install " + " ".join(missing),
            "Note: some system libraries Qt needs are missing, so the window may not open. Run:\n"
            "    sudo apt install " + " ".join(missing))
    if in_dialout_group() is False:
        say("提示：你不在 dialout 群組，連接 K210 開發板時會沒有權限。請執行：\n"
            "    sudo usermod -aG dialout $USER   （然後登出再登入）",
            "Note: you are not in the dialout group, so serial boards (K210) will be denied. Run:\n"
            "    sudo usermod -aG dialout $USER   (then log out and in)")
    if env.get("XDG_SESSION_TYPE", "").lower() == "wayland":
        say("提示：目前是 Wayland。如果畫面怪怪的，可以改用：QT_QPA_PLATFORM=xcb ./run_linux.sh",
            "Note: you are on Wayland. If the window looks wrong, try: QT_QPA_PLATFORM=xcb ./run_linux.sh")


# --------------------------------------------------------------------------- 安裝

def fail(zh, en, hints=(), tail=()):
    print("")
    say("[失敗] " + zh, "[FAILED] " + en)
    if tail:
        say("pip 最後的訊息：", "Last lines from pip:")
        for line in tail:
            print("    " + line)
    for h in hints:
        print("  -> " + h)
    sys.stdout.flush()
    return 1


def create_venv(venv):
    """建立虛擬環境。已存在但壞掉（例如 Python 被移走）時先刪掉重建。"""
    vpy = venv_python(venv)
    if os.path.exists(vpy):
        rc = subprocess.call([vpy, "-c", "import sys"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if rc == 0:
            return True, []
        remove_tree(venv)
    os.makedirs(os.path.dirname(venv) or ".", exist_ok=True)
    rc, tail = run_streaming([sys.executable, "-m", "venv", venv], show=False)
    return rc == 0 and os.path.exists(vpy), tail


def install_extra(key, vpy, root, wheelhouse):
    """安裝一個選配套件；失敗只記錄，不中斷。回傳 {'ok','package','error'}。"""
    req = os.path.join(root, "requirements", "extra-%s.txt" % key)
    attempts = [(req, _first_package(req) or key)]
    if key == "tflite":
        attempts.append(("tflite-runtime", "tflite-runtime"))
    error = ""
    for target, name in attempts:
        rc, tail = run_streaming(pip_install_args(vpy, target, wheelhouse, constraints_file(root)))
        if rc == 0:
            return {"ok": True, "package": name, "error": ""}
        error = (tail[-1] if tail else "pip failed")[:300]
        say("    %s 安裝失敗，繼續..." % name, "    %s failed, continuing..." % name)
    return {"ok": False, "package": attempts[-1][1] if key == "tflite" else attempts[0][1], "error": error}


def _first_package(req_path):
    try:
        with open(req_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(("#", "-")):
                    for sep in ("<", ">", "=", "!", "~", ";", "["):
                        line = line.split(sep)[0]
                    return line.strip()
    except OSError:
        pass
    return ""


def check_wheelhouse(path):
    """wheelhouse 必須是有檔案的資料夾；否則回傳錯誤說明。"""
    if not os.path.isdir(path):
        return t("找不到 wheelhouse 資料夾：%s" % path, "Wheelhouse folder not found: %s" % path)
    if not [n for n in os.listdir(path) if n.endswith((".whl", ".tar.gz", ".zip"))]:
        return t("wheelhouse 資料夾是空的：%s\n請先在有網路的電腦執行 bootstrap.py --make-wheelhouse 資料夾" % path,
                 "The wheelhouse folder is empty: %s\nOn a computer with internet run "
                 "bootstrap.py --make-wheelhouse FOLDER first." % path)
    return ""


def install(opts, venv, root=None):
    """完整安裝流程。成功回傳 0。"""
    root = root or ROOT
    wh = opts.wheelhouse
    if wh:
        problem = check_wheelhouse(wh)
        if problem:
            return fail(problem, problem)
    if platform_key() == "linux":
        if not have_venv_module():
            return fail("這台電腦的 Python 沒有 venv 模組。", "This Python has no venv module.",
                        [t("請執行：", "Please run: ") + venv_apt_hint()])
        linux_checks()
    say("第一次執行需要安裝套件，下載約 300–400 MB，可能要幾分鐘，請不要關閉視窗。",
        "First run: installing packages (about 300-400 MB download, may take several minutes). "
        "Please keep this window open.")
    step(1, "建立虛擬環境...", "Creating the virtual environment...")
    ok, tail = create_venv(venv)
    if not ok:
        hints = [t("確認磁碟空間足夠、資料夾可以寫入：", "Check disk space and write access: ") + venv]
        if platform_key() == "linux":
            hints.append(t("請執行：", "Please run: ") + venv_apt_hint())
        return fail("無法建立虛擬環境。", "Could not create the virtual environment.", hints, tail[-8:])
    vpy = venv_python(venv)

    step(2, "安裝必要套件（最花時間）...", "Installing required packages (the slow part)...")
    rc, tail = run_streaming(pip_install_args(vpy, os.path.join(root, "requirements", "base.txt"), wh,
                                              constraints_file(root)))
    if rc != 0:
        return fail("必要套件安裝失敗，EdgeLens 無法啟動。", "Required packages failed to install.",
                    failure_hint(tail, wh), tail[-8:])

    extras = {}
    labels = {"onnx": ("ONNX 模型推論", "ONNX inference"), "tflite": ("TFLite 模型推論", "TFLite inference"),
              "k210": ("K210 燒錄工具", "K210 flashing tool")}
    for n, key in enumerate(EXTRA_KEYS, start=3):
        step(n, "安裝選配：%s" % labels[key][0], "Installing optional: %s" % labels[key][1])
        extras[key] = install_extra(key, vpy, root, wh)
    write_extras(venv, extras)
    write_stamp(venv, compute_hash(root))
    bad = [k for k in EXTRA_KEYS if not extras[k]["ok"]]
    if bad:
        say("注意：選配套件沒裝成功（%s），相關功能會停用，其他功能不受影響。" % ", ".join(bad),
            "Note: optional packages not installed (%s); those features are off, everything else works." % ", ".join(bad))
    say("安裝完成！", "Installation finished!")
    return 0


def ensure_installed(opts, venv, root=None):
    """需要時才安裝（用雜湊判斷）；用鎖檔避免兩個視窗同時安裝。成功回傳 0。"""
    root = root or ROOT
    digest = compute_hash(root)
    if is_up_to_date(venv, digest):
        return 0
    lock = venv.rstrip("/\\") + ".lock"

    def _notify():
        say("另一個視窗正在安裝中，請稍候...", "Another window is installing, please wait...")

    if not acquire_lock(lock, wait=True, notify=_notify):
        return fail("等了很久，另一個安裝還沒結束。", "Waited a long time but the other install is still running.",
                    [t("請稍後再試；如果確定沒有其他安裝，刪除這個檔案：", "Try later; if nothing else is installing delete: ") + lock])
    try:
        if is_up_to_date(venv, digest):  # 等待期間別人已經裝好了
            return 0
        return install(opts, venv, root)
    finally:
        release_lock(lock)


def make_wheelhouse(dest, root=None):
    """在有網路的電腦下載所有 wheel 檔，之後可用 --wheelhouse 離線安裝（作業系統與 Python 版本要相同）。"""
    root = root or ROOT
    os.makedirs(dest, exist_ok=True)
    say("正在下載套件到 %s（僅適用相同作業系統與 Python %d.%d）..." % ((dest,) + sys.version_info[:2]),
        "Downloading packages to %s (only valid for the same OS and Python %d.%d)..." % ((dest,) + sys.version_info[:2]))
    cons = constraints_file(root)
    rc, tail = run_streaming(pip_download_args(sys.executable, os.path.join(root, "requirements", "base.txt"), dest,
                                               cons))
    if rc != 0:
        return fail("下載必要套件失敗。", "Downloading required packages failed.", failure_hint(tail), tail[-8:])
    for key in EXTRA_KEYS:
        req = os.path.join(root, "requirements", "extra-%s.txt" % key)
        rc, _ = run_streaming(pip_download_args(sys.executable, req, dest, cons))
        if rc != 0 and key == "tflite":
            tmp = os.path.join(dest, ".tflite-req.txt")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("tflite-runtime\n")
            rc, _ = run_streaming(pip_download_args(sys.executable, tmp, dest, cons))
            os.remove(tmp)
        if rc != 0:
            say("    選配 %s 下載失敗，略過。" % key, "    Optional %s failed to download, skipping." % key)
    say("完成！把資料夾 %s 一起帶到要安裝的電腦，執行：bootstrap.py --wheelhouse %s" % (dest, dest),
        "Done! Bring the folder %s to the target computer and run: bootstrap.py --wheelhouse %s" % (dest, dest))
    return 0


def launch(vpy, passthrough, root=None):
    root = root or ROOT
    cmd = [vpy, "-m", "edgelens"] + list(passthrough)
    return subprocess.call(cmd, cwd=root, env=launch_env(root))


def main(argv=None):
    _setup_stdio()
    argv = sys.argv[1:] if argv is None else list(argv)
    ok, _why = check_python()
    if not ok:
        print(python_problem_message(sys.version_info[:2]))
        return 1
    opts = parse_args(argv)
    if opts.error:
        print("%s needs a folder / 需要指定資料夾" % opts.error)
        return 2
    if opts.make_wheelhouse:
        return make_wheelhouse(opts.make_wheelhouse)
    venv = venv_dir()
    if opts.clean:
        remove_tree(venv)
        say("已刪除虛擬環境：%s" % venv, "Removed the virtual environment: %s" % venv)
        return 0
    if opts.reinstall:
        say("重新安裝：刪除舊的虛擬環境...", "Reinstall: removing the old environment...")
        remove_tree(venv)
    started = time.time()
    rc = ensure_installed(opts, venv)
    if rc != 0:
        return rc
    vpy = venv_python(venv)
    touch_last_used(venv)
    cleanup_old_venvs(venv)  # 確認目前的環境可用之後才清舊的，不會把人留在沒有環境的狀態
    if os.environ.get("EDGELENS_BOOTSTRAP_DEBUG"):
        print("bootstrap check: %.2fs" % (time.time() - started))
    rc = launch(vpy, opts.passthrough)
    if rc != 0 and platform_key() == "linux" and not opts.selftest:
        linux_checks()
    return rc


if __name__ == "__main__":
    sys.exit(main())
