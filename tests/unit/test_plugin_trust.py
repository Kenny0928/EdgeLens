"""外掛信任檢查：信任存取、20 個資料夾上限、提示流程與 switch_workspace 的限制。"""

from __future__ import annotations

import sys
import types

import pytest

from edgelens.core import registry
from edgelens.core.settings import Settings
from edgelens.gui import plugin_trust as pt
from edgelens.processing.pipeline import STEPS


@pytest.fixture(autouse=True)
def _restore_globals(monkeypatch):
    monkeypatch.setattr(STEPS, "_items", dict(STEPS.items()))
    monkeypatch.setattr(STEPS, "_origin", dict(STEPS._origin))
    monkeypatch.setattr(registry, "import_problems", [])
    before = {k for k in sys.modules if k.startswith("edgelens_plugins.")}
    yield
    for k in list(sys.modules):
        if k.startswith("edgelens_plugins.") and k not in before:
            del sys.modules[k]


@pytest.fixture
def settings(tmp_path):
    return Settings(tmp_path / "settings.json")


def write(folder, name, body="X = 1\n"):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(body, encoding="utf-8")


def prompts(monkeypatch, answer):
    calls = []

    def fake(parent, pending):
        calls.append(list(pending))
        return answer

    monkeypatch.setattr(pt, "ask_trust", fake)
    return calls


def test_trust_roundtrip_and_cap(settings, tmp_path):
    folder = tmp_path / "p"
    write(folder, "a.py")
    pt.record_trust(settings, folder, registry.plugin_hashes(folder))
    settings.save()
    again = Settings(settings.path)
    assert pt.trusted_hashes(again, folder) == registry.plugin_hashes(folder)
    for i in range(25):
        pt.record_trust(settings, tmp_path / f"f{i}", {"a.py": "x"})
    store = settings.get(pt.TRUST_KEY)
    assert len(store) == pt.MAX_TRUSTED_FOLDERS
    assert pt._folder_key(tmp_path / "f24") in store
    assert pt._folder_key(tmp_path / "f0") not in store
    assert pt._folder_key(folder) not in store


def test_trusted_unchanged_loads_without_prompt(settings, tmp_path, monkeypatch):
    folder = tmp_path / "p"
    write(folder, "trust_a.py")
    pt.record_trust(settings, folder, registry.plugin_hashes(folder))
    calls = prompts(monkeypatch, False)
    assert pt.load_workspace_plugins(settings, folder, interactive=True) == ["edgelens_plugins.trust_a"]
    assert calls == []


def test_new_file_prompts_and_approval_records(settings, tmp_path, monkeypatch):
    folder = tmp_path / "p"
    write(folder, "trust_b.py")
    calls = prompts(monkeypatch, True)
    assert pt.load_workspace_plugins(settings, folder, interactive=True) == ["edgelens_plugins.trust_b"]
    assert calls == [[("trust_b.py", False)]]
    calls.clear()
    pt.load_workspace_plugins(settings, folder, interactive=True)
    assert calls == []
    write(folder, "trust_b.py", "X = 2\n")  # 改過內容 → 再問一次，並標為已修改
    pt.load_workspace_plugins(settings, folder, interactive=True)
    assert calls == [[("trust_b.py", True)]]


def test_decline_loads_only_trusted(settings, tmp_path, monkeypatch):
    folder = tmp_path / "p"
    write(folder, "trust_ok.py")
    pt.record_trust(settings, folder, registry.plugin_hashes(folder))
    write(folder, "trust_new.py")
    prompts(monkeypatch, False)
    assert pt.load_workspace_plugins(settings, folder, interactive=True) == ["edgelens_plugins.trust_ok"]
    assert "trust_new.py" not in pt.trusted_hashes(settings, folder)


def test_non_interactive_loads_everything(settings, tmp_path, monkeypatch):
    folder = tmp_path / "p"
    write(folder, "trust_c.py")
    calls = prompts(monkeypatch, False)
    assert pt.load_workspace_plugins(settings, folder, interactive=False) == ["edgelens_plugins.trust_c"]
    assert calls == []


@pytest.mark.gui
@pytest.mark.parametrize("mode", ["safe", "disabled", "ok"])
def test_switch_workspace_respects_flags(qtbot, tmp_path, monkeypatch, mode):
    from edgelens.core.project import Workspace
    from edgelens.gui import main_window as mw
    from edgelens.gui.app import create_app

    args = types.SimpleNamespace(workspace=str(tmp_path / "ws"), lang="zh_TW", safe_mode=(mode == "safe"))
    _, ctx, window = create_app(args)
    qtbot.addWidget(window)
    new_root = tmp_path / "ws2"
    write(Workspace(new_root).plugins_dir, "trust_sw.py")
    if mode == "disabled":
        ctx.settings.set("plugins.enabled", False)
    window.interactive = True
    calls = prompts(monkeypatch, True)
    dlg = types.SimpleNamespace(exec=lambda: True, chosen_path=lambda: new_root)
    monkeypatch.setattr(mw, "WorkspaceDialog", lambda *a, **k: dlg)
    window.switch_workspace()
    loaded = "edgelens_plugins.trust_sw" in sys.modules
    assert loaded == (mode == "ok")
    assert bool(calls) == (mode == "ok")
    window.close()
