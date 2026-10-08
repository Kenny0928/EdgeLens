"""翻譯檔檢查：兩種語言的鍵一致、程式碼裡 tr("...") 用到的鍵都存在、{參數} 一致。"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from edgelens import i18n
from edgelens.i18n import LANGUAGES, load_language, tr

PKG = Path(i18n.__file__).resolve().parents[1]
PLACEHOLDER = re.compile(r"\{(\w+)\}")


def test_languages_have_same_keys():
    tables = {lang: load_language(lang) for lang in LANGUAGES}
    zh, en = set(tables["zh_TW"]), set(tables["en"])
    assert not (zh - en), f"只有中文有的鍵：{sorted(zh - en)}"
    assert not (en - zh), f"只有英文有的鍵：{sorted(en - zh)}"


def test_placeholders_match():
    zh, en = load_language("zh_TW"), load_language("en")
    for key in zh:
        assert set(PLACEHOLDER.findall(zh[key])) == set(PLACEHOLDER.findall(en[key])), f"{key} 的 {{參數}} 不一致"


def test_locale_files_paired():
    zh_files = {p.name for p in (i18n.LOCALES_DIR / "zh_TW").glob("*.json")}
    en_files = {p.name for p in (i18n.LOCALES_DIR / "en").glob("*.json")}
    assert zh_files == en_files, "zh_TW 和 en 的翻譯檔名稱要成對"


def _literal_tr_keys() -> list[tuple[str, str]]:
    found = []
    for path in PKG.rglob("*.py"):
        if "board" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and node.args:
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
                if name in ("tr",) and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    found.append((node.args[0].value, f"{path.relative_to(PKG)}:{node.lineno}"))
    return found


def _class_attr_keys() -> list[tuple[str, str]]:
    """類別屬性 display_name_key／description_key／*_key = "..." 都要有翻譯。"""
    found = []
    for path in PKG.rglob("*.py"):
        if "board" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for stmt in node.body:
                    if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                            and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id.endswith("_key")
                            and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)
                            and stmt.value.value):
                        found.append((stmt.value.value, f"{path.relative_to(PKG)}:{stmt.lineno}"))
    return found


@pytest.mark.parametrize("lang", list(LANGUAGES))
def test_all_used_keys_exist(lang):
    table = load_language(lang)
    missing = [(k, where) for k, where in _literal_tr_keys() + _class_attr_keys() if k not in table]
    assert not missing, "程式用到但翻譯檔沒有的鍵：\n" + "\n".join(f"  {k}  ({w})" for k, w in missing)


def test_tr_fallbacks():
    assert tr("app.name") == "EdgeLens"
    assert tr("no.such.key") == "no.such.key"
    i18n.set_language("en")
    assert tr("common.ok") == "OK"
    assert tr("app.version", version="1.2") == "Version 1.2"
    assert tr("app.version") == "Version {version}"


def test_language_listener():
    seen = []
    remove = i18n.on_language_changed(seen.append)
    i18n.set_language("en")
    i18n.set_language("en")
    i18n.set_language("zh_TW")
    remove()
    i18n.set_language("en")
    assert seen == ["en", "zh_TW"]
    with pytest.raises(ValueError):
        i18n.set_language("fr")
