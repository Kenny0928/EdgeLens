from __future__ import annotations

import os
import time

from edgelens.core import housekeeping as hk


def _make(p, size, age_days=0.0):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)
    t = time.time() - age_days * 86400
    os.utime(p, (t, t))


def test_prune_thumb_cache(tmp_path, monkeypatch):
    from edgelens.core.paths import app_cache_dir

    root = app_cache_dir() / "thumbs"
    for i in range(10):
        _make(root / "p1" / f"{i}.jpg", 100, age_days=10 - i)   # 0 最舊
    _make(root / "p2" / "old.jpg", 100, age_days=100)
    outside = app_cache_dir() / "other.bin"
    _make(outside, 5000)
    assert hk.prune_thumb_cache(max_bytes=5000, target_bytes=1000) == 0   # 沒超過
    assert hk.prune_thumb_cache(max_bytes=1000, target_bytes=500) == 6
    left = sorted(p.name for p in root.rglob("*.jpg"))
    assert left == ["5.jpg", "6.jpg", "7.jpg", "8.jpg", "9.jpg"]
    assert not (root / "p2").exists()      # 空資料夾被移除
    assert outside.exists()


def test_sweep_stale_temp(tmp_path, monkeypatch):
    base = tmp_path / "tmp"
    base.mkdir()
    monkeypatch.setattr(hk, "temp_base_candidates", lambda: [base, base])
    old = 10 * 86400

    def mk(name, age=old, files=()):
        d = base / name
        d.mkdir()
        for f in files:
            (d / f).write_bytes(b"1")
        t = time.time() - age
        os.utime(d, (t, t))
        return d

    stale_rec = mk("edgelens_rec_ab12cd34", files=["a.avi"])
    stale_vid = mk("edgelens_zz99yy88", files=["video.mp4"])
    fresh = mk("edgelens_rec_ab12cd35", age=3600)
    other = mk("edgelens_mydata")                      # 名稱不符 mkdtemp 格式
    foreign = mk("edgelens_qq11ww22", files=["notes.txt"])   # 內容不是影片複本
    unrelated = mk("something_ab12cd34")
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("k")
    link = base / "edgelens_rec_link0000"
    link.symlink_to(victim, target_is_directory=True)

    assert hk.sweep_stale_temp(7) == 2
    assert not stale_rec.exists() and not stale_vid.exists()
    for keep in (fresh, other, foreign, unrelated, victim / "keep.txt"):
        assert keep.exists()
    assert link.is_symlink()


def test_run_startup_housekeeping_never_raises(monkeypatch):
    def boom():
        raise RuntimeError("x")

    monkeypatch.setattr(hk, "prune_thumb_cache", boom)
    called = []
    monkeypatch.setattr(hk, "sweep_stale_temp", lambda: called.append(1) or 0)
    th = hk.run_startup_housekeeping()
    th.join(5)
    assert called == [1] and th.daemon
