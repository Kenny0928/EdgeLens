"""效能／錄影面板：統計、記錄 CSV、錄影、長時間測試、參數掃描（synthetic 來源，不碰硬體）。"""

from __future__ import annotations

import csv
import types

import pytest

from edgelens.gui.app import create_app
from edgelens.sources.synthetic import SyntheticSource

pytestmark = pytest.mark.gui


def synth(**params):
    return SyntheticSource.info_from_params({"pattern": "moving_box", "fps": 60, "resolution": "160x120", **params})


@pytest.fixture
def gui(qtbot, tmp_path):
    args = types.SimpleNamespace(workspace=str(tmp_path / "ws"), lang="zh_TW", safe_mode=True)
    app, ctx, window = create_app(args)
    qtbot.addWidget(window)
    window.show()
    yield ctx, window, window.panel("bench")
    window.close()


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def start_cam(qtbot, window):
    h = window.add_source(synth())
    qtbot.waitUntil(lambda: h.last_packet is not None, timeout=3000)
    return h


def test_panel_registered_and_empty(gui):
    ctx, window, bench = gui
    assert bench is not None and bench.order == 70
    assert bench.empty_label.isVisibleTo(bench)
    assert not bench.record_button.isEnabled()
    assert any(a.text() for a in window.menuBar().actions())


def _num(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return 0.0


def test_stats_update(qtbot, gui):
    ctx, window, bench = gui
    h = start_cam(qtbot, window)
    qtbot.waitUntil(lambda: bench._stat_values["samples"].text() not in ("-", "0"), timeout=3000)
    # 第一筆樣本出來時 fps 可能還是 0，慢的 CI 機器上更明顯；等它算出來
    qtbot.waitUntil(lambda: _num(bench._stat_values["fps"].text()) > 5, timeout=3000)
    assert "p95" in bench._stat_values["latency"].text()
    bench.window_spin.setValue(20)
    assert bench.fps_stats.window == 20
    qtbot.waitUntil(lambda: int(bench._stat_values["samples"].text()) >= 2, timeout=3000)
    assert not bench.board_table.isVisibleTo(bench)   # synthetic 沒有板上數據
    assert h is ctx.active_source


def test_log_data_csv(qtbot, gui):
    ctx, window, bench = gui
    start_cam(qtbot, window)
    assert bench.start_log()
    qtbot.wait(700)
    res = bench.stop_log()
    assert res is not None and res.frames > 5 and res.video_path is None
    rows = read_csv(res.csv_path)
    assert len(rows) == res.frames and float(rows[-1]["fps"]) > 0
    assert res.csv_path.parent == ctx.project.root / "logs"
    assert bench.log_button.text() == "開始記錄數據"


def test_record_video_modes(qtbot, gui):
    ctx, window, bench = gui
    start_cam(qtbot, window)
    bench.mode_combo.setCurrentIndex(2)   # 含標註
    assert bench.start_record()
    qtbot.wait(900)
    assert "錄影中" in bench.record_label.text()
    res = bench.stop_record()
    assert res.video_path is not None and res.video_path.is_file() and res.frames > 5
    assert res.video_path.parent == ctx.project.root / "recordings"
    assert len(read_csv(res.csv_path)) == res.frames
    assert "已存成" in bench.record_label.text()


def test_shutdown_closes_recording(qtbot, gui):
    ctx, window, bench = gui
    start_cam(qtbot, window)
    bench.start_record()
    bench.start_log()
    qtbot.wait(300)
    window.close()
    assert bench._rec is None and bench._log is None
    assert bench.last_recording.video_path is not None and bench.last_recording.video_path.is_file()


def test_long_test_summary(qtbot, gui):
    ctx, window, bench = gui
    start_cam(qtbot, window)
    assert bench.start_long(seconds=1.5)
    assert bench.long_progress.isVisibleTo(bench)
    qtbot.waitUntil(lambda: bench._long is None, timeout=6000)
    assert bench.last_long_csv is not None and bench.last_long_csv.parent == ctx.project.root / "logs"
    rows = {r["metric"]: r for r in read_csv(bench.last_long_csv)}
    assert {"fps", "latency_ms", "process_ms"} <= set(rows)
    for col in ("mean", "p50", "p95", "max"):
        assert float(rows["fps"][col]) > 0
    assert bench.long_table.isVisibleTo(bench) and bench.long_table.rowCount() >= 3
    assert "完成" in bench.long_status.text()


def test_long_test_cancel(qtbot, gui):
    ctx, window, bench = gui
    start_cam(qtbot, window)
    bench.start_long(seconds=30)
    qtbot.wait(200)
    bench.long_button.click()
    assert bench._long is None and "取消" in bench.long_status.text()


def test_sweep_brightness_restores(qtbot, gui):
    ctx, window, bench = gui
    h = start_cam(qtbot, window)
    qtbot.waitUntil(lambda: any(s.key == "brightness" for s in h.controls), timeout=2000)
    h.set_control("brightness", 7)
    h.flush_controls()
    qtbot.waitUntil(lambda: ctx.settings.source_settings(h.info.id).get("controls", {}).get("brightness") == 7,
                    timeout=2000)
    idx = bench.sweep_param.findData("brightness")
    assert idx >= 0
    bench.sweep_param.setCurrentIndex(idx)
    bench.sweep_from.setValue(-50)
    bench.sweep_to.setValue(50)
    bench.sweep_count.setValue(3)
    bench.sweep_dwell.setValue(0.5)
    seen = []
    h.controlResult.connect(lambda k, r: seen.append((k, r.value)) if k == "brightness" else None)
    assert bench.start_sweep()
    qtbot.waitUntil(lambda: bench._sweep is None, timeout=8000)
    rows = read_csv(bench.last_sweep_csv)
    assert [float(r["value"]) for r in rows] == [-50, 0, 50]
    assert all(int(r["frames"]) > 3 and float(r["fps_mean"]) > 0 for r in rows)
    assert bench.sweep_table.rowCount() == 3
    assert h.values["brightness"] == 7
    qtbot.waitUntil(lambda: ctx.settings.source_settings(h.info.id)["controls"]["brightness"] == 7
                    and [v for _, v in seen][-1] == 7, timeout=2000)
    assert {-50, 0, 50} <= {v for _, v in seen}
    assert "還原" in bench.sweep_status.text()


def test_sweep_cancel_restores(qtbot, gui):
    ctx, window, bench = gui
    h = start_cam(qtbot, window)
    qtbot.waitUntil(lambda: any(s.key == "noise" for s in h.controls), timeout=2000)
    bench.sweep_param.setCurrentIndex(bench.sweep_param.findData("noise"))
    bench.sweep_count.setValue(4)
    bench.sweep_dwell.setValue(0.5)
    assert bench.start_sweep()
    assert not bench.sweep_param.isEnabled()
    qtbot.waitUntil(lambda: h.values["noise"] != 0, timeout=3000)
    bench.sweep_button.click()
    assert bench._sweep is None and h.values["noise"] == 0
    assert "取消" in bench.sweep_status.text()
    qtbot.waitUntil(lambda: ctx.settings.source_settings(h.info.id)["controls"].get("noise") == 0, timeout=2000)


def test_sweep_enum_values(qtbot, gui):
    ctx, window, bench = gui
    h = start_cam(qtbot, window)
    qtbot.waitUntil(lambda: any(s.key == "pattern" for s in h.controls), timeout=2000)
    bench.sweep_param.setCurrentIndex(bench.sweep_param.findData("pattern"))
    assert bench.sweep_options.isVisibleTo(bench) and bench.sweep_options.count() == 5
    from PySide6.QtCore import Qt
    for i in range(2, 5):
        bench.sweep_options.item(i).setCheckState(Qt.CheckState.Unchecked)
    assert bench._checked_options() == ["bars", "moving_box"]
    bench.sweep_dwell.setValue(0.4)
    bench.start_sweep()
    qtbot.waitUntil(lambda: bench._sweep is None, timeout=6000)
    assert [r["value"] for r in read_csv(bench.last_sweep_csv)] == ["bars", "moving_box"]
    assert h.values["pattern"] == "moving_box"


def test_open_folder_and_language(qtbot, gui, monkeypatch):
    ctx, window, bench = gui
    opened = []
    monkeypatch.setattr(type(bench), "_open_folder", lambda self, p: opened.append(p))
    bench.open_logs_button.click()
    bench.open_rec_button.click()
    assert [p.name for p in opened] == ["logs", "recordings"]
    ctx.set_language("en")
    assert bench.record_button.text() == "Start recording"
    assert bench.title == "Performance / Recording"


def test_add_inference_reaches_csv(qtbot, gui):
    from edgelens.core.results import Classification, InferenceResult
    ctx, window, bench = gui
    h = start_cam(qtbot, window)
    bench.start_log()
    bench.add_inference(h, 1, InferenceResult(kind="classify", model_name="m", frame_seq=1,
                                              top_k=[Classification(2, 0.8, "dog")]))
    qtbot.wait(500)
    res = bench.stop_log()
    assert read_csv(res.csv_path)[-1]["infer_label"] == "dog"
