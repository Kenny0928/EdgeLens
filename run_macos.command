#!/bin/bash
# EdgeLens 啟動腳本（macOS）：雙擊即可。找到合適的 Python 3.10-3.13 後交給 bootstrap.py。
cd "$(dirname "$0")" || exit 1

NONINT=0
for a in "$@"; do [ "$a" = "--noninteractive" ] && NONINT=1; done

pause_if_needed() {
    if [ "$NONINT" = "0" ]; then
        read -r -p "按 Enter 關閉 / Press Enter to close... " _
    fi
}

py_ok() {
    "$1" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' >/dev/null 2>&1
}

PY=""
for v in 3.13 3.12 3.11 3.10; do
    for cand in \
        "$(command -v python$v 2>/dev/null)" \
        "/Library/Frameworks/Python.framework/Versions/$v/bin/python3" \
        "/opt/homebrew/bin/python$v" \
        "/usr/local/bin/python$v"; do
        if [ -n "$cand" ] && [ -x "$cand" ] && py_ok "$cand"; then PY="$cand"; break 2; fi
    done
done
if [ -z "$PY" ]; then
    # 最後才試 python3；跳過 /usr/bin/python3（macOS 內建，太舊，沒裝開發工具時還會跳出安裝視窗）
    cand="$(command -v python3 2>/dev/null)"
    if [ -n "$cand" ] && [ "$cand" != "/usr/bin/python3" ] && py_ok "$cand"; then PY="$cand"; fi
fi

if [ -z "$PY" ]; then
    echo "找不到可用的 Python（需要 3.10 到 3.13）。"
    echo "Could not find Python 3.10-3.13."
    echo ""
    echo "請到 https://www.python.org/downloads/macos/ 下載並安裝 Python 3.12，再重新雙擊這個檔案。"
    echo "Download and install Python 3.12 from https://www.python.org/downloads/macos/ then run this again."
    pause_if_needed
    exit 1
fi

"$PY" bootstrap.py "$@"
rc=$?
if [ $rc -ne 0 ]; then
    echo ""
    echo "EdgeLens 結束了（代碼 ${rc}）。 / EdgeLens exited with code ${rc}."
    pause_if_needed
fi
exit $rc
