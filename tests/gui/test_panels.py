from __future__ import annotations

import sys
import textwrap

import pytest

from edgelens.gui.panels import discover_panels
from edgelens.gui.panels.base import Panel

pytestmark = pytest.mark.gui


def test_discover_finds_camera_panel():
    panels = discover_panels()
    ids = [p.panel_id for p in panels]
    assert "camera" in ids
    orders = [(p.order, p.panel_id) for p in panels]
    assert orders == sorted(orders)
    assert all(issubclass(p, Panel) for p in panels)


def test_discover_custom_package_with_broken_and_good_modules(tmp_path, monkeypatch):
    pkg = tmp_path / "fake_panels_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "good_b.py").write_text(textwrap.dedent("""
        from edgelens.gui.panels.base import Panel
        class B(Panel):
            panel_id = "b"
            order = 20
        class NotAPanel:
            panel_id = "zzz"
    """))
    (pkg / "good_a.py").write_text(textwrap.dedent("""
        from edgelens.gui.panels.base import Panel
        class A(Panel):
            panel_id = "a"
            order = 5
    """))
    (pkg / "broken.py").write_text("raise RuntimeError('boom')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        found = discover_panels("fake_panels_pkg")
    finally:
        for name in [n for n in sys.modules if n.startswith("fake_panels_pkg")]:
            del sys.modules[name]
    assert [p.panel_id for p in found] == ["a", "b"]
