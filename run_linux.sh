#!/bin/bash
# EdgeLens 啟動腳本（Linux）：./run_linux.sh。找到合適的 Python 3.10-3.13 後交給 bootstrap.py。
cd "$(dirname "$0")" || exit 1

NONINT=0
for a in "$@"; do [ "$a" = "--noninteractive" ] && NONINT=1; done

pause_if_needed() {
    if [ "$NONINT" = "0" ] && [ -t 0 ]; then
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
        "/usr/bin/python$v" \
        "/usr/local/bin/python$v" \
        "/opt/homebrew/bin/python$v"; do
        if [ -n "$cand" ] && [ -x "$cand" ] && py_ok "$cand"; then PY="$cand"; break 2; fi
    done
done
if [ -z "$PY" ]; then
    cand="$(command -v python3 2>/dev/null)"
    if [ -n "$cand" ] && py_ok "$cand"; then PY="$cand"; fi
fi

if [ -z "$PY" ]; then
    echo "找不到可用的 Python（需要 3.10 到 3.13）。"
    echo "Could not find Python 3.10-3.13."
    echo ""
    echo "Ubuntu / Debian 請執行： sudo apt install python3 python3-venv python3-pip"
    echo "Run: sudo apt install python3 python3-venv python3-pip   (or see https://www.python.org/downloads/)"
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
