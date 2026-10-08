"""命令列進入點：python -m edgelens [選項]。不加選項時開啟圖形介面。"""

from __future__ import annotations

import argparse
import sys

from edgelens import APP_NAME, __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="edgelens", description=f"{APP_NAME}：跨平台邊緣視覺相機工具")
    p.add_argument("--version", action="store_true", help="顯示版本後結束")
    p.add_argument("--selftest", action="store_true", help="執行自我檢查後結束（結束碼 0 表示通過）")
    p.add_argument("--json", action="store_true", help="自我檢查結果用 JSON 輸出")
    p.add_argument("--workspace", metavar="DIR", help="工作區資料夾（預設：文件/EdgeLens）")
    p.add_argument("--lang", choices=["zh_TW", "en"], help="介面語言")
    p.add_argument("--safe-mode", action="store_true", help="不載入外掛、不自動開相機（程式打不開時用）")
    p.add_argument("--debug", action="store_true", help="輸出除錯訊息")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.version:
        print(f"{APP_NAME} {__version__}")
        return 0

    from edgelens.core.log import setup_logging

    setup_logging(debug=args.debug)
    if args.lang:
        from edgelens.i18n import set_language

        set_language(args.lang)

    if args.selftest:
        from edgelens import selftest

        return selftest.run(as_json=args.json)

    try:
        from edgelens.gui.app import run
    except ImportError as e:
        from edgelens.i18n import tr

        print(tr("cli.gui_missing"), f"({e})", file=sys.stderr)
        return 2
    return int(run(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
