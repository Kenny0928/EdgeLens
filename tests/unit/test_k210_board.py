"""板上程式 board/stream_v1.py：不能在電腦端執行，只檢查語法與文件約定。"""

from __future__ import annotations

import ast
import importlib.resources

from edgelens.sources.k210.source import board_script


def _text() -> str:
    return importlib.resources.files("edgelens.sources.k210").joinpath("board", "stream_v1.py").read_text(
        encoding="utf-8")


def test_board_script_is_valid_python():
    ast.parse(_text())
    ast.parse(board_script())


def test_board_dir_is_not_a_package():
    root = importlib.resources.files("edgelens.sources.k210").joinpath("board")
    assert not root.joinpath("__init__.py").is_file()


def test_documented_fields_exist():
    text = _text()
    header = text.split("import sensor")[0]
    for word in ("CAPS", "cls", "objs", "ack", "CMD", "STREAM_START", "JBF1", "classify", "topk", "softmax",
                 "input_size", "anchor", "send_every"):
        assert word in header, f"文件註解缺少 {word}"
    for word in ("CAPS", "kpu.forward", "fmap[:]", '"cls"', "STREAM_START"):
        assert word in text


def test_only_maixpy_modules_imported():
    tree = ast.parse(_text())
    allowed = {"sensor", "lcd", "time", "gc", "os", "ujson", "ustruct", "math", "machine", "Maix", "KPU", "sys"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert {a.name.split(".")[0] for a in node.names} <= allowed
        elif isinstance(node, ast.ImportFrom):
            assert node.module.split(".")[0] in allowed


def test_no_project_specific_model_hardcoded():
    text = _text().lower()
    for word in ("筊", "jiao", "cup"):
        assert word not in text


def test_no_fstrings_or_walrus_for_old_micropython():
    tree = ast.parse(_text())
    assert not any(isinstance(n, (ast.JoinedStr, ast.NamedExpr)) for n in ast.walk(tree))


def test_stripped_script_is_smaller_and_has_no_comment_lines():
    s = board_script()
    assert not any(ln.lstrip().startswith("#") for ln in s.splitlines())
    assert len(s) < len(_text())
