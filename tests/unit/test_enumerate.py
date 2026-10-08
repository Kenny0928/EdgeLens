"""相機列舉測試：用假的 cv2_enumerate_cameras，絕不開啟相機。"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import cv2
import pytest

from edgelens.sources import enumerate as en


def cam(index, name, backend, path="", vid=None, pid=None):
    return SimpleNamespace(index=index, name=name, backend=backend, path=path, vid=vid, pid=pid)


@pytest.fixture
def no_open(monkeypatch):
    """列舉絕對不能開相機：誰呼叫 cv2.VideoCapture 就讓測試失敗。"""
    def boom(*a, **k):
        raise AssertionError("列舉不可以開啟相機")

    monkeypatch.setattr(cv2, "VideoCapture", boom)


def fake_lib(monkeypatch, per_api: dict[int, list] | Exception):
    mod = types.ModuleType("cv2_enumerate_cameras")

    def enumerate_cameras(api=0):
        if isinstance(per_api, Exception):
            raise per_api
        return list(per_api.get(api, []))

    mod.enumerate_cameras = enumerate_cameras
    monkeypatch.setitem(sys.modules, "cv2_enumerate_cameras", mod)


def plat(monkeypatch, name):
    monkeypatch.setattr(en, "current_platform", lambda: name)


def test_macos_stable_ids_iphone_and_deskview(monkeypatch, no_open):
    plat(monkeypatch, "darwin")
    av = en.API_IDS["avfoundation"]
    fake_lib(monkeypatch, {av: [
        cam(0, "FaceTime HD Camera", av, "AAA", 1452, 34304),
        cam(1, "Kenny 的 iPhone 相機", av, "BBB"),
        cam(2, "Kenny 的 iPhone 桌上視角", av, "CCC"),
        cam(3, "Desk View Camera", av, "DDD"),
    ]})
    devs = en.list_cameras()
    assert [d.id for d in devs] == ["webcam:FaceTime HD Camera", "webcam:Kenny 的 iPhone 相機",
                                    "webcam:Kenny 的 iPhone 桌上視角", "webcam:Desk View Camera"]
    assert [d.role for d in devs] == ["", "iphone", "iphone", "deskview"]
    assert devs[1].is_iphone and devs[0].backend == "avfoundation"
    assert devs[0].vid_pid == "05AC:8600" and devs[1].vid_pid == ""
    assert devs[0].path == "AAA"
    assert en.role_detail("iphone") and en.role_detail("") == ""


def test_same_name_gets_suffix_by_index_order(monkeypatch, no_open):
    plat(monkeypatch, "linux")
    v = en.API_IDS["v4l2"]
    fake_lib(monkeypatch, {v: [cam(4, "USB Cam", v, "/dev/video4"), cam(0, "USB Cam", v, "/dev/video0"),
                               cam(2, "USB Cam", v, "/dev/video2"), cam(6, "Other", v, "/dev/video6")]})
    devs = en.list_cameras()
    assert [(d.index, d.id, d.dup) for d in devs] == [
        (0, "webcam:USB Cam", 1), (2, "webcam:USB Cam#2", 2), (4, "webcam:USB Cam#3", 3), (6, "webcam:Other", 1)]
    assert devs[0].path == "/dev/video0"


def test_windows_dedupes_msmf_and_dshow(monkeypatch, no_open):
    plat(monkeypatch, "win")
    ms, ds = en.API_IDS["msmf"], en.API_IDS["dshow"]
    fake_lib(monkeypatch, {
        ms: [cam(0, "HD Webcam", ms, "p0", 0x046D, 0x0825), cam(1, "HD Webcam", ms, "p1", 0x046D, 0x0825),
             cam(2, "Integrated Camera", ms, "p2", 0x0BDA, 0x5678)],
        ds: [cam(0, "HD Webcam", ds, "p0", 0x046D, 0x0825), cam(1, "HD Webcam", ds, "p1", 0x046D, 0x0825),
             cam(2, "OBS Virtual Camera", ds, "", None, None)],
    })
    devs = en.list_cameras()
    ids = sorted(d.id for d in devs)
    assert ids == ["webcam:HD Webcam", "webcam:HD Webcam#2", "webcam:Integrated Camera",
                   "webcam:OBS Virtual Camera"]
    hd = next(d for d in devs if d.id == "webcam:HD Webcam")
    assert hd.backend == "msmf" and hd.alternatives == {"dshow": 0}
    hd2 = next(d for d in devs if d.id == "webcam:HD Webcam#2")
    assert hd2.alternatives == {"dshow": 1}
    obs = next(d for d in devs if "OBS" in d.name)
    assert obs.backend == "dshow" and obs.alternatives == {}


def test_windows_same_name_different_vidpid_not_merged(monkeypatch, no_open):
    plat(monkeypatch, "win")
    ms, ds = en.API_IDS["msmf"], en.API_IDS["dshow"]
    fake_lib(monkeypatch, {ms: [cam(0, "Camera", ms, "a", 1, 2)], ds: [cam(0, "Camera", ds, "b", 3, 4)]})
    assert len(en.list_cameras()) == 2


def test_library_failure_falls_back_to_sysfs(monkeypatch, tmp_path, no_open):
    plat(monkeypatch, "linux")
    fake_lib(monkeypatch, RuntimeError("壞了"))
    for n, name, idx in (("video0", "Integrated Webcam", "0"), ("video1", "Integrated Webcam", "1"),
                         ("video2", "USB Camera", "0"), ("video10", "Cam10", "0")):
        d = tmp_path / n
        d.mkdir()
        (d / "name").write_text(name + "\n")
        (d / "index").write_text(idx + "\n")
    (tmp_path / "videoX").mkdir()
    monkeypatch.setattr(en, "SYSFS_V4L", tmp_path)
    devs = en.list_cameras()
    assert [(d.index, d.name, d.path) for d in devs] == [
        (0, "Integrated Webcam", "/dev/video0"), (2, "USB Camera", "/dev/video2"), (10, "Cam10", "/dev/video10")]
    assert devs[0].backend == "v4l2"


def test_missing_library_on_linux_and_empty_sysfs(monkeypatch, tmp_path, no_open):
    plat(monkeypatch, "linux")
    monkeypatch.setitem(sys.modules, "cv2_enumerate_cameras", None)   # import 會失敗
    monkeypatch.setattr(en, "SYSFS_V4L", tmp_path / "none")
    assert en.list_cameras() == []


def test_other_platform_does_not_probe_by_default(monkeypatch, no_open):
    plat(monkeypatch, "other")
    monkeypatch.setitem(sys.modules, "cv2_enumerate_cameras", None)
    assert en.list_cameras() == []
    assert en.list_cameras(probe=False) == []


def test_probe_opens_indices_and_releases(monkeypatch):
    plat(monkeypatch, "other")
    monkeypatch.setitem(sys.modules, "cv2_enumerate_cameras", None)
    opened, released = [], []

    class Cap:
        def __init__(self, index, api=None):
            self.index = index
            opened.append(index)

        def isOpened(self):
            return self.index in (0, 3)

        def release(self):
            released.append(self.index)

    monkeypatch.setattr(cv2, "VideoCapture", Cap)
    devs = en.list_cameras(probe=True)
    assert opened == list(range(8)) and sorted(released) == list(range(8))
    assert [d.index for d in devs] == [0, 3] and devs[0].id == "webcam:相機 0"
    # 列舉到相機時，probe=True 不會再去開
    plat(monkeypatch, "darwin")
    av = en.API_IDS["avfoundation"]
    fake_lib(monkeypatch, {av: [cam(0, "Cam", av)]})
    opened.clear()
    assert [d.name for d in en.list_cameras(probe=True)] == ["Cam"] and opened == []


def test_find_camera_by_name_and_dup(monkeypatch, no_open):
    plat(monkeypatch, "linux")
    v = en.API_IDS["v4l2"]
    fake_lib(monkeypatch, {v: [cam(0, "A", v), cam(1, "A", v), cam(2, "B", v)]})
    assert en.find_camera("B").index == 2
    assert en.find_camera("A", 2).index == 1
    assert en.find_camera("A", 1).index == 0
    assert en.find_camera("C") is None
    # 同名兩台但要找的第三台不存在：不可亂猜
    assert en.find_camera("A", 3) is None


def test_empty_names_skipped_and_dict(monkeypatch, no_open):
    plat(monkeypatch, "linux")
    v = en.API_IDS["v4l2"]
    fake_lib(monkeypatch, {v: [cam(0, "", v), cam(1, "X", v, "/dev/video1", 0x1234, 0xABCD)]})
    devs = en.list_cameras()
    assert len(devs) == 1
    d = devs[0].to_dict()
    assert d["vid"] == 0x1234 and d["id"] == "webcam:X"


def test_default_api():
    assert en.default_api("darwin") == "avfoundation"
    assert en.default_api("win") == "msmf"
    assert en.default_api("linux") == "v4l2"
    assert en.default_api("other") == "auto"


@pytest.mark.hardware
def test_hardware_real_enumeration_does_not_crash():
    devs = en.list_cameras()
    print([(d.index, d.name, d.backend, d.role) for d in devs])
    assert isinstance(devs, list)


def test_iphone_detected_by_model_id_when_renamed():
    """iPhone 改名後（例如「Kenny相機」）名稱沒有 iPhone，要靠 macOS 回報的型號辨識。"""
    devs = en._finalize([en.CameraDevice(index=0, name="MacBook Air相機", path="A"),
                         en.CameraDevice(index=1, name="Kenny相機", path="B")],
                        {"A": "MacBook Air相機", "B": "iPhone18,5"})
    assert [d.role for d in devs] == ["", "iphone"]
    assert en.classify_role("Kenny相機") == ""
