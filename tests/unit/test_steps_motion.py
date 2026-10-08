"""連續影格、重設與自動拍照所需的靜止訊號。"""
from __future__ import annotations

import numpy as np
import pytest

from edgelens.processing.pipeline import StepContext
from edgelens.processing.steps.motion import BackgroundSub, FrameDiff, StillDetect


def test_frame_difference_baseline_and_second_frame(make_frame):
    first = np.zeros((10, 12), np.uint8)
    second = np.full_like(first, 30)
    ctx = StepContext(make_frame(first))
    step = FrameDiff()
    assert not step.apply(first, ctx).any()
    assert ctx.data["motion.frame_diff.score"] == 0 and not ctx.data["motion.frame_diff.ready"]
    assert np.all(step.apply(second, ctx) == 30)
    assert ctx.data["motion.frame_diff.score"] == 30
    first[:] = 255
    assert not step.apply(second, ctx).any()
    step.set_param("show_diff", False)
    assert np.array_equal(step.apply(first, ctx), first)
    step.reset()
    step.apply(second, ctx)
    assert not ctx.data["motion.frame_diff.ready"]


def test_still_requires_n_differences_and_motion_resets(make_frame):
    image = np.full((10, 10), 50, np.uint8)
    ctx = StepContext(make_frame(image))
    step = StillDetect(frames=3, threshold=2)
    for expected in (False, False, False, True, True):
        assert np.array_equal(step.apply(image, ctx), image)
        assert ctx.data["motion.still_detect.still"] is expected
    step.apply(image + 2, ctx)
    assert not ctx.data["motion.still_detect.still"]
    assert ctx.data["motion.still_detect.count"] == 0
    step.set_param("frames", 1)
    step.apply(image, ctx)
    assert not ctx.data["motion.still_detect.still"]
    step.apply(image, ctx)
    assert ctx.data["motion.still_detect.still"]
    step.reset()
    step.apply(image, ctx)
    assert not ctx.data["motion.still_detect.still"]


@pytest.mark.parametrize("step", [FrameDiff(), StillDetect(frames=1), BackgroundSub()])
def test_state_reset_on_shape_and_source(step, make_frame):
    step.reset()
    first = np.zeros((20, 20), np.uint8)
    ctx = StepContext(make_frame(first))
    step.apply(first, ctx)
    step.apply(first, ctx)
    different = np.full((10, 30, 3), 255, np.uint8)
    out = step.apply(different, ctx)
    assert out.shape[:2] == different.shape[:2]
    if step.kind != "motion.background_sub":
        assert ctx.data[f"{step.kind}.score"] == 0
    other_ctx = StepContext(make_frame(different, source_id="other"))
    step.apply(different, other_ctx)
    if isinstance(step, FrameDiff):
        assert not other_ctx.data["motion.frame_diff.ready"]
    if isinstance(step, StillDetect):
        assert not other_ctx.data["motion.still_detect.still"]


@pytest.mark.parametrize("method", ["mog2", "knn"])
def test_background_learns_static_scene_then_finds_foreground(method, make_frame):
    image = np.zeros((40, 40, 3), np.uint8)
    ctx = StepContext(make_frame(image))
    step = BackgroundSub(method=method, history=10, learning_rate=0.5)
    for _ in range(30):
        out = step.apply(image, ctx)
    assert not out.any()
    image[10:30, 10:30] = 255
    step.set_param("learning_rate", 0)
    out = step.apply(image, ctx)
    assert np.all(out[10:30, 10:30] == 255)
    assert ctx.data["motion.background_sub.foreground_ratio"] == pytest.approx(0.25)
    step.set_param("method", "knn" if method == "mog2" else "mog2")
    assert step.apply(image, ctx).shape == (40, 40)
