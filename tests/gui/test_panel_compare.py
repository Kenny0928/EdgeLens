"""比較面板：兩台 synthetic 相機加一個模型的表格、重設、匯出 CSV。"""

from __future__ import annotations

import csv
import types

import pytest

from edgelens.gui.app import create_app
from edgelens.gui.panels.compare import COLUMNS
from edgelens.inference.runner import load_model_file
from edgelens.sources.base import SourceInfo
from edgelens.sources.synthetic import SyntheticSource
from tests.fixtures.make_onnx import LABELS, generate_all

pytestmark = pytest.mark.gui


def synth(**params) -> SourceInfo:
    return SyntheticSource.info_from_params({"fps": 60, "resolution": "160x120", **params})


@pytest.fixture
def env(qtbot, tmp_path):
    folder = tmp_path / "m"
    paths = generate_all(folder)
    paths["yolo_t"].with_name("fake_yolo_t.labels.txt").write_text("\n".join(LABELS), encoding="utf-8")
    args = types.SimpleNamespace(workspace=str(tmp_path / "ws"), lang="zh_TW", safe_mode=True)
    app, ctx, window = create_app(args)
    qtbot.addWidget(window)
    ctx.set_project(ctx.workspace.create_project("RPS", "rock_paper_scissors"))
    h1 = window.add_source(synth(pattern="bars"))
    h2 = window.add_source(synth(pattern="moving_box"))
    qtbot.waitUntil(lambda: h1.last_packet is not None and h2.last_packet is not None, timeout=5000)
    s = ctx.inference.add(h1, load_model_file(paths["yolo_t"]))
    qtbot.waitUntil(lambda: s.last_result is not None, timeout=8000)
    yield ctx, window, (h1, h2), s, window.panel("compare")
    window.close()


def test_table_rows_and_columns(qtbot, env):
    ctx, window, (h1, h2), s, panel = env
    assert panel.order == 60
    qtbot.waitUntil(lambda: s.stats["infer_fps"] > 0, timeout=4000)
    panel.refresh()
    # 指標在列、每組（相機 × 模型）一欄
    assert panel.table.columnCount() == 2 and panel.table.rowCount() == len(COLUMNS) - 2 == 8
    rows = panel.rows_data()
    assert rows[0]["camera"] == h1.info.name and rows[0]["model"] == "fake_yolo_t"
    assert rows[1]["model"] == "" and rows[1]["infer_fps"] is None
    assert rows[0]["infer_fps"] > 0 and rows[0]["infer_ms"] > 0
    assert rows[0]["cam_fps"] > 0 and rows[0]["sharpness"] is not None and rows[0]["brightness"] is not None
    assert "beta" in rows[0]["best"]
    qtbot.waitUntil(lambda: panel.rows_data()[0]["rate"] is not None, timeout=3000)
    assert panel.rows_data()[0]["rate"] == pytest.approx(100.0)
    from edgelens.gui.panels.compare import METRICS

    assert panel.table.item(METRICS.index("rate"), 0).text().endswith("%")
    assert panel.table.item(METRICS.index("infer_fps"), 1).text() == "-"
    assert panel.table.horizontalHeaderItem(0).text().startswith(h1.info.name)
    assert panel.table.verticalHeaderItem(0).text() == "相機 FPS"


def test_second_model_adds_row_and_close_removes(qtbot, env, tmp_path):
    ctx, window, (h1, h2), s, panel = env
    from edgelens.inference.runner import load_model_file

    ctx.inference.add(h2, load_model_file(tmp_path / "m" / "fake_classify.onnx"))
    panel.refresh()
    assert panel.table.columnCount() == 2
    ctx.inference.add(h1, load_model_file(tmp_path / "m" / "fake_classify.onnx"))
    panel.refresh()
    assert panel.table.columnCount() == 3
    window.close_source(h2)
    panel.refresh()
    assert panel.table.columnCount() == 2


def test_reset_stats(qtbot, env):
    ctx, window, (h1, h2), s, panel = env
    qtbot.waitUntil(lambda: panel.rows_data()[0]["rate"] is not None, timeout=3000)
    panel.reset_button.click()
    assert s.stats["infer_fps"] == 0.0
    assert panel._rates[s.session_id].summary().total <= 1
    assert "重設" in panel.status_label.text()


def test_export_csv_bom_and_location(qtbot, env):
    ctx, window, (h1, h2), s, panel = env
    path = panel.export_csv()
    assert path is not None and ctx.project.dir("logs") in path.parents
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and "清晰度" in rows[0] and rows[0]["模型"] == "fake_yolo_t"


def test_export_without_project_shows_hint(qtbot, env):
    ctx, window, (h1, h2), s, panel = env
    ctx.project = None
    assert panel.export_csv() is None
    assert "專案" in panel.status_label.text()


def test_grid_button_and_help(qtbot, env):
    ctx, window, handles, s, panel = env
    panel.grid_button.click()
    assert window.grid.layout_mode == "grid"
    assert "同一個場景" in panel.fair_label.text()
