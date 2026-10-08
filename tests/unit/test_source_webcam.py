"""WebcamSource 單元測試：全部用假的 cv2.VideoCapture，不會開啟實體相機。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from edgelens.core.controls import ControlType
from edgelens.sources import webcam as wc
from edgelens.sources.base import SourceError, SourceInfo
from edgelens.sources.enumerate import CameraDevice
from edgelens.sources.webcam import WebcamSource

P = cv2
BRIGHTNESS, CONTRAST, SATURATION, GAIN = P.CAP_PROP_BRIGHTNESS, P.CAP_PROP_CONTRAST, P.CAP_PROP_SATURATION, P.CAP_PROP_GAIN
EXPOSURE, AUTO_EXPOSURE, FOCUS, AUTOFOCUS = P.CAP_PROP_EXPOSURE, P.CAP_PROP_AUTO_EXPOSURE, P.CAP_PROP_FOCUS, P.CAP_PROP_AUTOFOCUS


STREAM_PROPS = {P.CAP_PROP_FRAME_WIDTH, P.CAP_PROP_FRAME_HEIGHT, P.CAP_PROP_FPS, P.CAP_PROP_FOURCC,
                P.CAP_PROP_BUFFERSIZE, P.CAP_PROP_SETTINGS}


@dataclass
class Feature:
    value: float
    effective: bool = True      # False：set 回傳 True 但值不會變
    accept: bool = True         # False：set 回傳 False
    snap: float = 0.0           # 讀回值會吸附到 snap 的倍數（模擬驅動只接受特定步階）
    requires: tuple[int, float] | None = None   # (屬性, 值)：該屬性不是這個值時設定沒有作用
    nan: bool = False           # get 回傳 NaN（讀不回來）


class World:
    """假的相機世界：決定哪些編號打得開、有哪些屬性。"""

    def __init__(self):
        self.openable: set[int] = {0}
        self.features: dict[int, Feature] = {}
        self.fixed_res: tuple[int, int] | None = None
        self.fail_grab = False
        self.no_frames = False
        self.created: list[FakeCap] = []
        self.calls: list[tuple[int, int | None]] = []


class FakeCap:
    def __init__(self, world: World, index: int, api: int | None):
        self.world, self.index, self.api = world, index, api
        self.opened = index in world.openable
        self.props: dict[int, float] = {P.CAP_PROP_FRAME_WIDTH: 640, P.CAP_PROP_FRAME_HEIGHT: 480, P.CAP_PROP_FPS: 30}
        self.set_log: list[tuple[int, float]] = []
        self.grabs = 0
        self.released = False
        world.created.append(self)

    def isOpened(self):
        return self.opened and not self.released

    def release(self):
        self.released = True

    def _shape(self):
        if self.world.fixed_res:
            return self.world.fixed_res
        return int(self.props[P.CAP_PROP_FRAME_WIDTH]), int(self.props[P.CAP_PROP_FRAME_HEIGHT])

    def _frame(self):
        w, h = self._shape()
        return np.full((h, w, 3), 7, np.uint8)

    def read(self):
        if self.world.no_frames or self.world.fail_grab:
            return False, None
        return True, self._frame()

    def grab(self):
        self.grabs += 1
        return not self.world.fail_grab

    def retrieve(self):
        return True, self._frame()

    def get(self, pid):
        f = self.world.features.get(pid)
        if f is not None:
            return math.nan if f.nan else f.value
        return self.props.get(pid, 0.0)

    def set(self, pid, value):
        self.set_log.append((pid, value))
        f = self.world.features.get(pid)
        if f is None:
            # 沒列在 features 的屬性：只有串流屬性能設（和真的 OpenCV 一樣，其餘 set 回傳 False）
            if pid in STREAM_PROPS:
                self.props[pid] = value
                return True
            return False
        if not f.accept:
            return False
        if f.requires is not None and self.world.features[f.requires[0]].value != f.requires[1]:
            return True
        if f.effective:
            f.value = round(value / f.snap) * f.snap if f.snap else value
        return True


@pytest.fixture
def world(monkeypatch):
    w = World()

    def factory(index, api=None):
        w.calls.append((index, api))
        return FakeCap(w, index, api)

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    monkeypatch.setattr(wc, "WARMUP_SECONDS", 0.2)
    monkeypatch.setattr(wc, "list_cameras", lambda probe=False: [])
    return w


def platform(monkeypatch, name):
    monkeypatch.setattr(wc, "current_platform", lambda: name)


def make(params=None, id="webcam:test") -> WebcamSource:
    return WebcamSource(SourceInfo(kind="webcam", id=id, name="t", params=params or {}))


def keys(src):
    return [c.key for c in src.controls()]


# ---- 參數探測 ---------------------------------------------------------------------


def test_probe_lists_only_working_controls(world, monkeypatch):
    platform(monkeypatch, "linux")
    world.features = {
        BRIGHTNESS: Feature(128),
        CONTRAST: Feature(32, effective=False),      # 設了不會變
        SATURATION: Feature(60, accept=False),       # set 失敗
        GAIN: Feature(math.nan, nan=True),           # 讀不到
    }
    src = make({"api": "v4l2"})
    src.open()
    assert keys(src) == ["resolution", "fps", "brightness"]
    spec = next(c for c in src.controls() if c.key == "brightness")
    assert spec.readback and spec.type == ControlType.INT
    assert world.features[BRIGHTNESS].value == 128, "探測後必須還原原本的值"
    src.close()
    assert src.controls() == []


def test_probe_restores_values_and_auto_switch(world, monkeypatch):
    platform(monkeypatch, "linux")
    # V4L2：auto_exposure 3＝自動、1＝手動；自動時 exposure 設了沒有作用
    world.features = {
        AUTO_EXPOSURE: Feature(3),
        EXPOSURE: Feature(300, requires=(AUTO_EXPOSURE, 1)),
    }
    src = make({"api": "v4l2"})
    src.open()
    assert "exposure" in keys(src) and "auto_exposure" in keys(src)
    assert world.features[AUTO_EXPOSURE].value == 3
    assert world.features[EXPOSURE].value == 300
    ctrl = {c.key: c for c in src.controls()}
    assert ctrl["auto_exposure"].type == ControlType.BOOL and ctrl["auto_exposure"].default is True
    # 自動開著時改曝光 → 沒有效果，提示要先關自動
    res = src.set_control("exposure", 500)
    assert res.ok and res.verified is False and res.message
    assert src.set_control("auto_exposure", False).verified is True
    assert world.features[AUTO_EXPOSURE].value == 1
    res = src.set_control("exposure", 500)
    assert res.verified is True and res.value == 500


def test_dshow_auto_values_and_windows_exposure_range(world, monkeypatch):
    platform(monkeypatch, "win")
    world.features = {AUTO_EXPOSURE: Feature(0.75), EXPOSURE: Feature(-6)}
    src = make({"api": "dshow"})
    src.open()
    ctrl = {c.key: c for c in src.controls()}
    assert ctrl["exposure"].type == ControlType.FLOAT
    assert ctrl["exposure"].min == -13 and ctrl["exposure"].max == -1
    assert src.set_control("auto_exposure", False).verified is True
    assert world.features[AUTO_EXPOSURE].value == 0.25


def test_range_expands_to_readback(world, monkeypatch):
    platform(monkeypatch, "linux")
    world.features = {BRIGHTNESS: Feature(300)}   # 超過預設的 0–255
    src = make({"api": "v4l2"})
    src.open()
    spec = next(c for c in src.controls() if c.key == "brightness")
    assert spec.max >= 300 and spec.min <= 0


def test_range_expands_after_set(world, monkeypatch):
    platform(monkeypatch, "linux")
    world.features = {BRIGHTNESS: Feature(100), GAIN: Feature(10)}
    src = make({"api": "v4l2"})
    src.open()
    world.features[BRIGHTNESS].snap = 1000   # 驅動把值吸附到 1000：超出範圍
    res = src.set_control("brightness", 200)
    assert res.verified is False
    spec = next(c for c in src.controls() if c.key == "brightness")
    assert spec.max >= 0 and spec.max >= res.value


def test_set_control_verification(world, monkeypatch):
    platform(monkeypatch, "linux")
    world.features = {BRIGHTNESS: Feature(100), GAIN: Feature(10), FOCUS: Feature(50)}
    src = make({"api": "v4l2"})
    src.open()
    res = src.set_control("brightness", 150)
    assert res.ok and res.verified is True and res.value == 150
    world.features[GAIN].snap = 50          # 驅動只接受 50 的倍數
    res = src.set_control("gain", 120)
    assert res.ok and res.verified is False and res.value == 100
    world.features[FOCUS].nan = True        # 讀不回來
    res = src.set_control("focus", 80)
    assert res.ok and res.verified is None
    world.features[BRIGHTNESS].accept = False
    res = src.set_control("brightness", 10)
    assert not res.ok
    assert not src.set_control("nope", 1).ok
    assert src.get_control("brightness") == 150


def test_set_control_clamps_to_spec(world, monkeypatch):
    platform(monkeypatch, "linux")
    world.features = {BRIGHTNESS: Feature(100)}
    src = make({"api": "v4l2"})
    src.open()
    assert src.set_control("brightness", 99999).value == 255


def test_dshow_settings_dialog(world, monkeypatch):
    platform(monkeypatch, "win")
    src = make({"api": "dshow"})
    src.open()
    spec = next(c for c in src.controls() if c.key == "settings_dialog")
    assert spec.type == ControlType.ACTION
    assert src.capabilities().actions == ["settings_dialog"]
    cap = world.created[-1]
    assert src.run_action("settings_dialog").ok
    assert (cv2.CAP_PROP_SETTINGS, 1.0) in cap.set_log
    # 其他後端沒有
    src.close()
    src2 = make({"api": "msmf"})
    src2.open()
    assert "settings_dialog" not in keys(src2)
    assert not src2.run_action("settings_dialog").ok


def test_macos_avfoundation_has_no_probe(world, monkeypatch):
    platform(monkeypatch, "darwin")
    world.features = {BRIGHTNESS: Feature(100)}
    src = make({"api": "avfoundation"})
    src.open()
    assert keys(src) == ["resolution", "fps"]
    caps = src.capabilities()
    assert "macOS" in caps.extra["note"]
    assert caps.resolutions and (640, 480) in caps.resolutions
    # AVFoundation 不設 FOURCC
    assert all(pid != cv2.CAP_PROP_FOURCC for pid, _ in world.created[0].set_log)


def test_stream_props_mjpg_and_buffersize(world, monkeypatch):
    platform(monkeypatch, "linux")
    src = make({"api": "v4l2", "resolution": "1280x720", "fps": 15})
    src.open()
    log = dict(world.created[0].set_log)
    assert log[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter_fourcc(*"MJPG")
    assert (log[cv2.CAP_PROP_FRAME_WIDTH], log[cv2.CAP_PROP_FRAME_HEIGHT]) == (1280, 720)
    assert log[cv2.CAP_PROP_FPS] == 15
    assert log[cv2.CAP_PROP_BUFFERSIZE] == 1


# ---- 解析度 -----------------------------------------------------------------------


def test_resolution_change_reopens(world, monkeypatch):
    platform(monkeypatch, "linux")
    world.features = {BRIGHTNESS: Feature(100)}
    src = make({"api": "v4l2"})
    src.open()
    first = src.read()
    spec = next(c for c in src.controls() if c.key == "resolution")
    assert spec.live is False and "1280x720" in spec.option_values
    res = src.set_control("resolution", "1280x720")
    assert res.ok and res.verified is True and res.value == "1280x720"
    f = src.read()
    assert f.image.shape[:2] == (720, 1280) and f.meta == {"width": 1280, "height": 720}
    assert f.seq > first.seq
    assert len(world.created) == 2 and world.created[0].released
    assert "brightness" in keys(src), "重新開啟後要重新探測參數"
    assert src.capabilities().resolutions


def test_resolution_not_supported_reports_actual(world, monkeypatch):
    platform(monkeypatch, "linux")
    world.fixed_res = (640, 480)
    src = make({"api": "v4l2"})
    src.open()
    res = src.set_control("resolution", "1920x1080")
    assert res.ok and res.verified is False and res.value == "640x480" and res.message


def test_fps_change(world, monkeypatch):
    platform(monkeypatch, "linux")
    src = make({"api": "v4l2"})
    src.open()
    res = src.set_control("fps", 15)
    assert res.ok and res.value == 15 and res.verified is True


# ---- 開啟失敗 ---------------------------------------------------------------------


def test_open_failure_macos(world, monkeypatch):
    platform(monkeypatch, "darwin")
    world.openable = set()
    src = make({"index": 0, "api": "avfoundation"})
    with pytest.raises(SourceError) as ei:
        src.open()
    e = ei.value
    assert e.retryable and "系統設定" in e.hint and "相機" in e.hint and "終端機" in e.hint
    assert "權限" in e.hint and not src.is_open


def test_open_failure_windows(world, monkeypatch):
    platform(monkeypatch, "win")
    world.openable = set()
    with pytest.raises(SourceError) as ei:
        make({"index": 1, "api": "msmf"}).open()
    assert ei.value.retryable and "Teams" in ei.value.hint and "1" in ei.value.message


def test_open_failure_linux(world, monkeypatch):
    platform(monkeypatch, "linux")
    world.openable = set()
    monkeypatch.setattr(wc, "_device_exists", lambda p: False)
    with pytest.raises(SourceError) as ei:
        make({"index": 3, "api": "v4l2"}).open()
    assert "/dev/video3" in ei.value.message

    monkeypatch.setattr(wc, "_device_exists", lambda p: True)
    monkeypatch.setattr(wc, "_device_access", lambda p: False)
    with pytest.raises(SourceError) as ei:
        make({"index": 3, "api": "v4l2"}).open()
    assert "video 群組" in ei.value.hint and "usermod" in ei.value.hint

    monkeypatch.setattr(wc, "_device_access", lambda p: True)
    with pytest.raises(SourceError) as ei:
        make({"index": 3, "api": "v4l2"}).open()
    assert ei.value.hint


def test_open_failure_in_english(world, monkeypatch):
    from edgelens import i18n

    i18n.set_language("en")
    platform(monkeypatch, "darwin")
    world.openable = set()
    with pytest.raises(SourceError) as ei:
        make({"api": "avfoundation"}).open()
    assert "System Settings" in ei.value.hint


def test_opened_but_no_frames(world, monkeypatch):
    platform(monkeypatch, "darwin")
    world.no_frames = True
    with pytest.raises(SourceError) as ei:
        make({"api": "avfoundation"}).open()
    assert ei.value.retryable and ei.value.hint
    assert world.created[0].released


def test_windows_msmf_falls_back_to_dshow(world, monkeypatch):
    platform(monkeypatch, "win")
    dev = CameraDevice(index=0, name="USB Cam", backend="msmf", id="webcam:USB Cam", alternatives={"dshow": 1})
    monkeypatch.setattr(wc, "list_cameras", lambda probe=False: [dev])
    opened = {1}   # 只有 DSHOW 的編號 1 打得開
    world.openable = opened
    src = make({"index": 0, "api": "auto", "name": "USB Cam"})
    src.open()
    assert src._cur_api == "dshow" and src._cur_index == 1


# ---- 讀取與斷線 -------------------------------------------------------------------


def test_read_before_open_and_meta(world, monkeypatch):
    platform(monkeypatch, "linux")
    src = make({"api": "v4l2"})
    with pytest.raises(SourceError):
        src.read(timeout=0.1)
    src.open()
    seqs = []
    for _ in range(3):
        f = src.read(timeout=1.0)
        assert f.meta == {"width": 640, "height": 480} and f.source_id == "webcam:test"
        seqs.append(f.seq)
    assert seqs == [1, 2, 3]
    assert world.created[0].grabs >= 2, "之後的畫面要用 grab() 取最新的"


def test_disconnect_detected(world, monkeypatch):
    platform(monkeypatch, "linux")
    src = make({"api": "v4l2"})
    src.open()
    assert src.read(timeout=0.5) is not None
    world.fail_grab = True
    got = None
    with pytest.raises(SourceError) as ei:
        for _ in range(wc.MAX_GRAB_FAILS + 2):
            got = src.read(timeout=0.05)
    assert got is None and ei.value.retryable
    assert not src.is_open
    world.fail_grab = False
    src.open()                 # 重新插好後可以再開
    assert src.read(timeout=0.5) is not None


def test_single_grab_failure_is_not_disconnect(world, monkeypatch):
    platform(monkeypatch, "linux")
    src = make({"api": "v4l2"})
    src.open()
    src.read(timeout=0.5)
    world.fail_grab = True
    assert src.read(timeout=0.05) is None
    assert src.is_open
    world.fail_grab = False
    assert src.read(timeout=0.5) is not None


# ---- 依名稱重新找 index -------------------------------------------------------------


def test_reopen_by_name_when_index_changed(world, monkeypatch):
    platform(monkeypatch, "linux")
    dev = CameraDevice(index=2, name="Logi Cam", backend="v4l2", id="webcam:Logi Cam")
    monkeypatch.setattr(wc, "list_cameras", lambda probe=False: [dev])
    world.openable = {2}
    src = make({"index": 0, "api": "auto", "name": "Logi Cam", "dup": 1})
    src.open()
    assert world.calls[0][0] == 2


def test_name_not_found_does_not_open_other_camera(world, monkeypatch):
    platform(monkeypatch, "linux")
    other = CameraDevice(index=0, name="Other Cam", backend="v4l2", id="webcam:Other Cam")
    monkeypatch.setattr(wc, "list_cameras", lambda probe=False: [other])
    src = make({"index": 0, "name": "Logi Cam"})
    with pytest.raises(SourceError) as ei:
        src.open()
    assert "Logi Cam" in ei.value.message and ei.value.retryable
    assert world.calls == [], "找不到名稱時不可以亂開別台相機"


def test_same_name_uses_dup_number(world, monkeypatch):
    platform(monkeypatch, "linux")
    devs = [CameraDevice(index=0, name="Cam", backend="v4l2", id="webcam:Cam", dup=1),
            CameraDevice(index=4, name="Cam", backend="v4l2", id="webcam:Cam#2", dup=2)]
    monkeypatch.setattr(wc, "list_cameras", lambda probe=False: devs)
    world.openable = {0, 4}
    make({"index": 1, "name": "Cam", "dup": 2}).open()
    assert world.calls[0][0] == 4


def test_enumeration_empty_falls_back_to_stored_index(world, monkeypatch):
    platform(monkeypatch, "linux")
    world.openable = {5}
    make({"index": 5, "name": "Cam"}).open()
    assert world.calls[0][0] == 5


# ---- 探索與手動加入 -----------------------------------------------------------------


def test_discover_builds_stable_infos(monkeypatch):
    devs = [CameraDevice(index=0, name="FaceTime HD Camera", backend="avfoundation", id="webcam:FaceTime HD Camera",
                         vid=1452, pid=34304, role=""),
            CameraDevice(index=1, name="Kenny's iPhone Camera", backend="avfoundation",
                         id="webcam:Kenny's iPhone Camera", role="iphone"),
            CameraDevice(index=2, name="Cam", backend="avfoundation", id="webcam:Cam#2", dup=2)]
    monkeypatch.setattr(wc, "list_cameras", lambda probe=False: devs)
    infos = WebcamSource.discover()
    assert [i.id for i in infos] == ["webcam:FaceTime HD Camera", "webcam:Kenny's iPhone Camera", "webcam:Cam#2"]
    assert "接續互通" in infos[1].detail and "05AC:8600" in infos[0].detail
    assert infos[2].name == "Cam #2" and infos[2].params["dup"] == 2
    assert infos[0].params["api"] == "avfoundation"


def test_discover_never_raises(monkeypatch):
    def boom(probe=False):
        raise RuntimeError("x")

    monkeypatch.setattr(wc, "list_cameras", boom)
    assert WebcamSource.discover() == []


def test_manual_add_info_is_stable():
    a = WebcamSource.info_from_params({"index": 2, "api": "dshow"})
    b = WebcamSource.info_from_params({"index": 2, "api": "dshow"})
    assert a.id == b.id == "webcam:camera2@dshow"
    assert WebcamSource.info_from_params({"index": 0, "api": "auto"}).id == "webcam:camera0"
    specs = {s.key: s for s in WebcamSource.param_specs()}
    assert specs["index"].type == ControlType.INT
    assert specs["api"].option_values == ["auto", "avfoundation", "msmf", "dshow", "v4l2"]
    assert not WebcamSource.discover_is_safe


# ---- 實機測試（CI 不執行；由老師的電腦驗證）-----------------------------------------


@pytest.mark.hardware
def test_hardware_open_index0_read_30():
    src = WebcamSource(WebcamSource.info_from_params({"index": 0, "api": "auto"}))
    src.open()
    try:
        frames = [f for f in (src.read(timeout=2.0) for _ in range(30)) if f is not None]
        assert len(frames) >= 25
        assert all(f.image.ndim == 3 for f in frames)
    finally:
        src.close()


@pytest.mark.hardware
def test_hardware_list_controls():
    src = WebcamSource(WebcamSource.info_from_params({"index": 0, "api": "auto"}))
    src.open()
    try:
        specs = src.controls()
        print("可用參數：", [s.key for s in specs])
        assert {"resolution", "fps"} <= {s.key for s in specs}
        for s in specs:
            if s.type != ControlType.ACTION:
                res = src.set_control(s.key, s.default_value())
                print(s.key, res)
    finally:
        src.close()


@pytest.mark.hardware
def test_hardware_iphone_camera_if_present():
    iphones = [i for i in WebcamSource.discover() if "iphone" in i.name.lower()]
    if not iphones:
        pytest.skip("沒有 iPhone 接續互通相機")
    assert "接續互通" in iphones[0].detail
    src = WebcamSource(iphones[0])
    src.open()
    try:
        assert src.read(timeout=3.0) is not None
    finally:
        src.close()
