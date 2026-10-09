"""背景推論整合：session、manager、主視窗（結果疊加、HUD、關閉相機／程式、拍照與錄影串接）。"""

from __future__ import annotations

import time
import types

import pytest

from edgelens.gui.app import create_app
from edgelens.sources.base import SourceInfo
from edgelens.sources.synthetic import SyntheticSource
from tests.fixtures.make_onnx import LABELS, generate_all

pytestmark = pytest.mark.gui


def synth(**params) -> SourceInfo:
    return SyntheticSource.info_from_params({"pattern": "bars", "fps": 60, "resolution": "160x120", **params})


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    folder = tmp_path_factory.mktemp("models")
    paths = generate_all(folder)
    for p in paths.values():
        p.with_name(p.stem + ".labels.txt").write_text("\n".join(LABELS), encoding="utf-8")
    return paths


@pytest.fixture
def env(qtbot, tmp_path):
    args = types.SimpleNamespace(workspace=str(tmp_path / "ws"), lang="zh_TW", safe_mode=True)
    app, ctx, window = create_app(args)
    qtbot.addWidget(window)
    handle = window.add_source(synth())
    qtbot.waitUntil(lambda: handle.last_packet is not None, timeout=5000)
    yield ctx, window, handle
    window.close()


def card_of(path):
    from edgelens.inference.runner import load_model_file

    return load_model_file(path)


def wait_result(qtbot, session, timeout=8000):
    qtbot.waitUntil(lambda: session.last_result is not None, timeout=timeout)
    return session.last_result


def test_session_runs_and_draws(qtbot, env, models):
    ctx, window, handle = env
    s = ctx.inference.add(handle, card_of(models["yolo_t"]))
    res = wait_result(qtbot, s)
    assert res.kind == "detect" and res.detections[0].label == "beta"
    assert s.info.loaded and s.state == "running" and s.label.endswith(handle.info.name)
    view = window.video_view_for(handle)
    qtbot.waitUntil(lambda: bool(view._results), timeout=3000)
    assert view._results[0].detections
    qtbot.waitUntil(lambda: s.stats["infer_fps"] > 0, timeout=3000)
    st = s.stats
    assert st["infer_ms"] > 0 and st["age_frames"] >= 0
    qtbot.waitUntil(lambda: "推論" in "".join(window._hud_extra.get(handle, {}).values()), timeout=3000)
    # 模型類別名稱和專案類別名稱相同時，用專案類別的顏色
    assert view._color_resolver is not None


def test_color_resolver_matches_project_class(qtbot, env, models):
    ctx, window, handle = env
    project = ctx.workspace.create_project("RPS", "rock_paper_scissors")
    ctx.set_project(project)
    first = project.classes[0]
    from edgelens.core.results import InferenceResult

    res = InferenceResult(kind="detect", model_name="m", frame_seq=1)
    assert window._color_resolver(res, 7, first.id) == first.color
    from edgelens.core.project import DEFAULT_COLORS
    from edgelens.gui.inference import ALT_COLORS

    assert window._color_resolver(res, 7, "nothing") == DEFAULT_COLORS[7 % len(DEFAULT_COLORS)]
    res.extra["style"] = 1
    assert window._color_resolver(res, 0, "zzz") == ALT_COLORS[0]
    assert window._color_resolver(res, 0, first.display("zh_TW")) == first.color


def test_thresholds_change_results_live(qtbot, env, models):
    ctx, window, handle = env
    s = ctx.inference.add(handle, card_of(models["yolo_t"]))
    assert wait_result(qtbot, s).detections
    s.set_thresholds(conf=0.95)
    qtbot.waitUntil(lambda: s.last_result is not None and not s.last_result.detections, timeout=5000)
    s.set_thresholds(conf=0.3)
    qtbot.waitUntil(lambda: bool(s.last_result.detections), timeout=5000)


def test_load_error_has_message_and_hint(qtbot, env, tmp_path):
    ctx, window, handle = env
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"this is not a model")
    s = ctx.inference.add(handle, card_of(bad))
    with qtbot.waitSignal(s.failed, timeout=8000) as sig:
        pass
    err = sig.args[0]
    assert err["message"] and err["hint"] and s.state == "failed"
    assert s.last_result is None


def test_two_sessions_on_one_camera(qtbot, env, models):
    ctx, window, handle = env
    a = ctx.inference.add(handle, card_of(models["yolo_t"]))
    b = ctx.inference.add(handle, card_of(models["classify"]))
    wait_result(qtbot, a)
    wait_result(qtbot, b)
    assert ctx.inference.primary(handle) is a and ctx.inference.sessions_for(handle) == [a, b]
    assert (a.style_index, b.style_index) == (0, 1)
    view = window.video_view_for(handle)
    qtbot.waitUntil(lambda: len(view._results) == 2, timeout=3000)
    assert view._results[1].extra.get("dashed")
    ctx.inference.remove(a)
    qtbot.waitUntil(lambda: len(view._results) == 1, timeout=3000)
    assert b.style_index == 0


def test_haar_builtin_session(qtbot, env):
    from edgelens.inference.backends.haar import builtin_cascades

    ctx, window, handle = env
    path = builtin_cascades()["frontalface_default"]
    s = ctx.inference.add(handle, card_of(path))
    res = wait_result(qtbot, s)
    assert s.info.mode == "image" and res.kind == "detect"


def test_use_processed_runs(qtbot, env, models):
    ctx, window, handle = env
    s = ctx.inference.add(handle, card_of(models["yolo_t"]), use_processed=True)
    wait_result(qtbot, s)
    assert s.use_processed


def test_raw_request(qtbot, env, models):
    ctx, window, handle = env
    s = ctx.inference.add(handle, card_of(models["yolo_t"]))
    wait_result(qtbot, s)
    with qtbot.waitSignal(s.rawReady, timeout=5000) as sig:
        s.request_raw()
    info, arr = sig.args[0][0]
    assert arr.shape == (1, 7, 84) and info.name


def test_set_decoder_applies_in_thread(qtbot, env, models):
    ctx, window, handle = env
    s = ctx.inference.add(handle, card_of(models["yolo_t"]))
    wait_result(qtbot, s)
    s.set_decoder("yolo_t", {})
    qtbot.waitUntil(lambda: s.info.decoder_reason != "" and "yolo_t" in s.info.decoder, timeout=3000)


def test_close_camera_stops_sessions(qtbot, env, models):
    ctx, window, handle = env
    s = ctx.inference.add(handle, card_of(models["yolo_t"]))
    wait_result(qtbot, s)
    thread = s._thread
    window.close_source(handle)
    assert ctx.inference.sessions_for(handle) == []
    assert not thread.is_alive()
    assert s.state == "stopped"


def test_close_window_stops_threads_quickly(qtbot, tmp_path, models):
    args = types.SimpleNamespace(workspace=str(tmp_path / "ws"), lang="zh_TW", safe_mode=True)
    app, ctx, window = create_app(args)
    qtbot.addWidget(window)
    h1 = window.add_source(synth())
    h2 = window.add_source(synth(pattern="moving_box"))
    qtbot.waitUntil(lambda: h1.last_packet is not None and h2.last_packet is not None, timeout=5000)
    sessions = [ctx.inference.add(h, card_of(models["yolo_t"])) for h in (h1, h2)]
    for s in sessions:
        wait_result(qtbot, s)
    threads = [s._thread for s in sessions]
    t0 = time.monotonic()
    window.close()
    assert time.monotonic() - t0 < 4.0
    assert all(not t.is_alive() for t in threads)


def test_last_inference_and_capture_metadata(qtbot, env, models):
    ctx, window, handle = env
    assert window.last_inference(handle) is None
    s = ctx.inference.add(handle, card_of(models["yolo_t"]))
    res = wait_result(qtbot, s)
    got = window.last_inference(handle)
    assert got is not None and got.kind == "detect" and got.model_name == res.model_name


def test_bench_add_inference_called(qtbot, env, models):
    ctx, window, handle = env
    bench = window.panel("bench")
    calls = []
    bench.add_inference = lambda h, seq, result: calls.append((h, seq, result))
    s = ctx.inference.add(handle, card_of(models["yolo_t"]))
    wait_result(qtbot, s)
    qtbot.waitUntil(lambda: bool(calls), timeout=3000)
    h, seq, result = calls[0]
    assert h is handle and seq == result.frame_seq


def test_project_switch_keeps_inference(qtbot, env, models):
    ctx, window, handle = env
    s = ctx.inference.add(handle, card_of(models["yolo_t"]))
    wait_result(qtbot, s)
    ctx.set_project(ctx.workspace.create_project("Another", "blank"))
    assert ctx.inference.sessions_for(handle) == [s] and s.is_running
