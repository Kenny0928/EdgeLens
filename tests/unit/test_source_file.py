"""FileSource 單元測試：影片、圖片資料夾（含中文路徑）、單張圖、循環、播完、暫停與逐張。"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from edgelens.core.controls import ControlType
from edgelens.sources import file as fs
from edgelens.sources.base import SourceError
from edgelens.sources.file import FileSource


def write_img(path: Path, value: int, ext: str = ".jpg", size=(64, 48)) -> bytes:
    img = np.full((size[1], size[0], 3), value, np.uint8)
    ok, buf = cv2.imencode(ext, img)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())   # 不用 imwrite，中文路徑才安全
    return buf.tobytes()


def make_video(path: Path, n: int = 10, fps: float = 30.0) -> Path:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    assert w.isOpened()
    for i in range(n):
        w.write(np.full((48, 64, 3), (20 + i * 10) % 256, np.uint8))
    w.release()
    return path


def src_for(path, **params) -> FileSource:
    return FileSource(FileSource.info_from_params({"path": str(path), **params}))


def read_all(src, limit=50, timeout=1.0):
    out = []
    for _ in range(limit):
        f = src.read(timeout=timeout)
        if f is None:
            if src.ended:
                break
            continue
        out.append(f)
    return out


# ---- 影片 -------------------------------------------------------------------------


def test_video_plays_and_ends(tmp_path):
    p = make_video(tmp_path / "clip.avi", n=8, fps=120)
    s = src_for(p, loop=False)
    s.open()
    frames = read_all(s)
    assert len(frames) == 8
    assert [f.seq for f in frames] == list(range(1, 9))
    assert [f.meta["index"] for f in frames] == list(range(1, 9))
    assert all(f.meta["file"] == "clip.avi" and f.meta["kind"] == "video" for f in frames)
    assert [f.meta["ended"] for f in frames] == [False] * 7 + [True]
    assert s.ended
    assert s.read(timeout=0.1) is None          # 播完：None，不是錯誤
    assert s.read(timeout=0.1) is None
    s.close()


def test_video_default_fps_follows_video_and_is_adjustable(tmp_path):
    s = src_for(make_video(tmp_path / "v.avi", fps=25))
    s.open()
    spec = {c.key: c for c in s.controls()}["fps"]
    assert spec.default == pytest.approx(25, abs=1)
    assert s.get_control("fps") == pytest.approx(25, abs=1)
    assert s.set_control("fps", 100).value == 100
    s.close()


def test_video_loop(tmp_path):
    s = src_for(make_video(tmp_path / "v.avi", n=5, fps=200), loop=True)
    s.open()
    frames = [s.read(timeout=1.0) for _ in range(13)]
    assert all(f is not None for f in frames)
    assert [f.meta["index"] for f in frames] == [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3]
    assert not any(f.meta["ended"] for f in frames) and not s.ended
    seqs = [f.seq for f in frames]
    assert seqs == sorted(set(seqs))
    s.close()


def test_turning_loop_off_ends_at_last_and_on_again_restarts(tmp_path):
    s = src_for(make_video(tmp_path / "v.avi", n=4, fps=200), loop=True)
    s.open()
    s.read()
    assert s.set_control("loop", False).ok
    rest = read_all(s)
    assert [f.meta["index"] for f in rest][-1] == 4 and rest[-1].meta["ended"]
    assert s.ended
    s.set_control("loop", True)
    f = s.read(timeout=1.0)
    assert f is not None and f.meta["index"] == 1 and not s.ended
    s.close()


def test_restart_action(tmp_path):
    s = src_for(make_video(tmp_path / "v.avi", n=4, fps=200), loop=False)
    s.open()
    read_all(s)
    assert s.ended
    assert s.run_action("restart").ok
    assert not s.ended
    frames = read_all(s)
    assert len(frames) == 4 and frames[0].meta["index"] == 1
    assert frames[0].seq > 4
    s.close()


def test_fps_pacing_limits_speed(tmp_path):
    s = src_for(make_video(tmp_path / "v.avi", n=30, fps=30), loop=False)
    s.open()
    s.set_control("fps", 20)
    t0 = time.monotonic()
    got = [s.read(timeout=1.0) for _ in range(6)]
    elapsed = time.monotonic() - t0
    assert all(got)
    assert elapsed >= 4.5 / 20, f"5 個間隔至少 0.25 秒，實際 {elapsed:.3f}"
    assert elapsed < 1.0
    s.close()


def test_read_returns_none_when_not_due_within_timeout(tmp_path):
    s = src_for(make_video(tmp_path / "v.avi", n=30, fps=30), loop=False)
    s.open()
    s.set_control("fps", 1)
    assert s.read(timeout=1.0) is not None
    t0 = time.monotonic()
    assert s.read(timeout=0.05) is None          # 下一張要 1 秒後，timeout 內沒有
    assert time.monotonic() - t0 < 0.5
    s.close()


# ---- 圖片資料夾（含中文）---------------------------------------------------------------


def test_image_folder_with_chinese_names(tmp_path):
    folder = tmp_path / "測試圖片資料夾"
    raws = {}
    for i, name in enumerate(["第一張.jpg", "第二張.jpg", "第十張.jpg"]):
        raws[name] = write_img(folder / name, 50 + i * 60)
    write_img(folder / "三.png", 200, ".png")
    (folder / "備註.txt").write_text("不是圖片", encoding="utf-8")
    write_img(folder / "._hidden.jpg", 1)
    s = src_for(folder, loop=False, fps=500)
    s.open()
    frames = read_all(s)
    names = [f.meta["file"] for f in frames]
    assert names == sorted(["第一張.jpg", "第二張.jpg", "第十張.jpg", "三.png"], key=str.lower)
    assert all(f.meta["total"] == 4 and f.meta["kind"] == "folder" for f in frames)
    for f in frames:
        if f.meta["file"].endswith(".jpg"):
            assert f.jpeg == raws[f.meta["file"]], "jpg 必須帶原始位元組"
        else:
            assert f.jpeg is None
        assert f.image.shape == (48, 64, 3) and f.image.dtype == np.uint8
    assert frames[-1].meta["ended"]
    assert s.capabilities().provides_jpeg
    s.close()


def test_natural_sort_order(tmp_path):
    for n in (1, 2, 10, 3):
        write_img(tmp_path / f"img{n}.png", n * 10, ".png")
    s = src_for(tmp_path, loop=False, fps=500)
    s.open()
    assert [f.meta["file"] for f in read_all(s)] == ["img1.png", "img2.png", "img3.png", "img10.png"]
    s.close()


def test_folder_default_speed_and_images_in_order(tmp_path):
    for i in range(3):
        write_img(tmp_path / f"{i}.png", 100 + i * 50, ".png")
    s = src_for(tmp_path)
    s.open()
    assert s.get_control("fps") == fs.DEFAULT_IMAGE_FPS
    s.set_control("fps", 500)
    vals = [int(s.read(timeout=1.0).image[0, 0, 0]) for _ in range(5)]
    assert vals == [100, 150, 200, 100, 150]      # 預設循環
    s.close()


def test_single_image_and_chinese_path(tmp_path):
    p = tmp_path / "中文資料夾" / "貓咪照片.jpg"
    raw = write_img(p, 123)
    s = src_for(p, fps=500)
    s.open()
    frames = [s.read(timeout=1.0) for _ in range(3)]
    assert all(f is not None and f.jpeg == raw and f.meta["total"] == 1 for f in frames)
    assert [f.seq for f in frames] == [1, 2, 3]
    s.close()
    s2 = src_for(p, loop=False, fps=500)
    s2.open()
    got = read_all(s2)
    assert len(got) == 1 and got[0].meta["ended"]
    s2.close()


def test_exif_rotated_jpeg_drops_raw_bytes(tmp_path):
    p = tmp_path / "r.jpg"
    raw = write_img(p, 100, size=(64, 48))
    # 手工插入 EXIF 方向 = 6（需要旋轉）
    tiff = (b"MM\x00\x2a\x00\x00\x00\x08" + b"\x00\x01" + b"\x01\x12\x00\x03\x00\x00\x00\x01\x00\x06\x00\x00"
            + b"\x00\x00\x00\x00")
    exif = b"Exif\x00\x00" + tiff
    seg = b"\xff\xe1" + (len(exif) + 2).to_bytes(2, "big") + exif
    p.write_bytes(raw[:2] + seg + raw[2:])
    assert fs._exif_orientation(p.read_bytes()) == 6
    assert fs._exif_orientation(raw) is None
    got = fs.read_image(p)
    assert got is not None and got[1] is None


def test_png_decoded_color(tmp_path):
    p = tmp_path / "a.png"
    gray = np.full((20, 30), 90, np.uint8)
    p.write_bytes(cv2.imencode(".png", gray)[1].tobytes())
    img, jpeg = fs.read_image(p)
    assert img.ndim == 3 and jpeg is None


def test_broken_image_is_skipped(tmp_path):
    write_img(tmp_path / "a.png", 10, ".png")
    (tmp_path / "b.png").write_bytes(b"not an image")
    write_img(tmp_path / "c.png", 30, ".png")
    s = src_for(tmp_path, loop=False, fps=500)
    s.open()
    assert [f.meta["file"] for f in read_all(s)] == ["a.png", "c.png"]
    s.close()


# ---- 暫停與逐張 -------------------------------------------------------------------


def test_pause_and_step(tmp_path):
    for i in range(4):
        write_img(tmp_path / f"{i}.png", 50 + i * 40, ".png")
    s = src_for(tmp_path, loop=False, fps=500)
    s.open()
    assert s.read(timeout=1.0).meta["index"] == 1
    s.set_control("paused", True)
    assert s.read(timeout=0.1) is None and s.read(timeout=0.1) is None
    assert s.run_action("step").ok
    f = s.read(timeout=0.1)
    assert f is not None and f.meta["index"] == 2
    assert s.read(timeout=0.1) is None            # 一次 step 只前進一張
    s.run_action("step")
    assert s.read(timeout=0.1).meta["index"] == 3
    s.set_control("paused", False)
    rest = read_all(s)
    assert [f.meta["index"] for f in rest] == [4] and s.ended


def test_step_requires_pause(tmp_path):
    write_img(tmp_path / "a.png", 1, ".png")
    s = src_for(tmp_path)
    s.open()
    res = s.run_action("step")
    assert not res.ok and "暫停" in res.message
    assert not s.run_action("nope").ok
    s.close()


def test_paused_param_starts_paused(tmp_path):
    write_img(tmp_path / "a.png", 1, ".png")
    s = src_for(tmp_path, paused=True)
    s.open()
    assert s.read(timeout=0.1) is None
    s.run_action("step")
    assert s.read(timeout=0.1) is not None
    s.close()


# ---- 錯誤與參數 -------------------------------------------------------------------


def test_errors(tmp_path):
    with pytest.raises(SourceError) as ei:
        src_for("").open()
    assert "選" in ei.value.message
    with pytest.raises(SourceError) as ei:
        src_for(tmp_path / "不存在.mp4").open()
    assert "不存在.mp4" in ei.value.message
    (tmp_path / "x.xyz").write_text("a")
    with pytest.raises(SourceError) as ei:
        src_for(tmp_path / "x.xyz").open()
    assert "x.xyz" in ei.value.message
    empty = tmp_path / "空"
    empty.mkdir()
    with pytest.raises(SourceError) as ei:
        src_for(empty).open()
    assert "沒有圖片" in ei.value.message
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00" * 100)
    with pytest.raises(SourceError) as ei:
        src_for(bad).open()
    assert ei.value.hint
    s = src_for(tmp_path / "x.xyz")
    with pytest.raises(SourceError):
        s.read(timeout=0.1)


def test_info_and_specs(tmp_path):
    info = FileSource.info_from_params({"path": str(tmp_path / "a.mp4")})
    assert info.id == f"file:{tmp_path / 'a.mp4'}" and info.name == "a.mp4"
    assert FileSource.info_from_params({"path": str(tmp_path / "a.mp4")}).id == info.id
    spec = FileSource.param_specs()[0]
    assert spec.key == "path" and spec.type == ControlType.STR and spec.group == "source"
    assert FileSource.manual_add
    s = src_for(make_video(tmp_path / "v.avi"))
    assert s.controls() == []
    s.open()
    by_key = {c.key: c for c in s.controls()}
    assert set(by_key) == {"loop", "fps", "paused", "restart", "step"}
    assert by_key["restart"].type == ControlType.ACTION and by_key["paused"].type == ControlType.BOOL
    assert s.capabilities().actions == ["restart", "step"]
    assert s.capabilities().extra["kind"] == "video"
    assert not s.set_control("nope", 1).ok and not s.set_control("restart", 1).ok
    with pytest.raises(KeyError):
        s.get_control("nope")
    s.close()


def test_close_and_reopen(tmp_path):
    s = src_for(make_video(tmp_path / "v.avi", n=5, fps=200), loop=False)
    s.open()
    read_all(s)
    s.close()
    s.close()
    s.open()
    assert s.read(timeout=1.0).meta["index"] == 1 and not s.ended
    s.close()


# ---- Windows 非 ASCII 影片路徑 -------------------------------------------------------


def test_windows_nonascii_path_uses_short_path(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "_is_windows", lambda: True)
    monkeypatch.setattr(fs, "_short_path", lambda p: "C:\\SHORT~1\\V~1.AVI")
    assert fs.openable_video_path(str(tmp_path / "影片.avi")) == ("C:\\SHORT~1\\V~1.AVI", None)
    assert fs.openable_video_path("C:\\video.avi") == ("C:\\video.avi", None)


def test_windows_nonascii_path_copies_to_temp_when_no_short_path(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "_is_windows", lambda: True)
    monkeypatch.setattr(fs, "_short_path", lambda p: None)
    src = make_video(tmp_path / "中文影片.avi", n=3, fps=200)
    real, tmpdir = fs.openable_video_path(str(src))
    try:
        assert real.isascii() and Path(real).is_file() and tmpdir
    finally:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)
    # 完整流程：複製後可以播放，關閉後暫存資料夾被刪除
    s = src_for(src, loop=False)
    s.open()
    reader_tmp = s._reader.tmpdir
    assert reader_tmp and Path(reader_tmp).is_dir()
    assert len(read_all(s)) == 3
    s.close()
    assert not Path(reader_tmp).exists()


def test_non_windows_uses_original_path(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "_is_windows", lambda: False)
    assert fs.openable_video_path("/tmp/中文.mp4") == ("/tmp/中文.mp4", None)
