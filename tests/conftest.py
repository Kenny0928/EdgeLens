"""共用測試設定：所有設定、快取、工作區都導到暫存資料夾，不碰使用者真正的檔案；
Qt 一律用 offscreen 模式。"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("EDGELENS_CONFIG", str(tmp_path / "_config"))
    monkeypatch.setenv("EDGELENS_CACHE", str(tmp_path / "_cache"))
    monkeypatch.setenv("EDGELENS_WORKSPACE", str(tmp_path / "_workspace"))
    yield


@pytest.fixture(autouse=True)
def _reset_language():
    from edgelens import i18n

    i18n.set_language("zh_TW")
    yield
    i18n.set_language("zh_TW")


@pytest.fixture(scope="session")
def registries():
    from edgelens.core.registry import all_registries, discover

    discover()
    return all_registries()


@pytest.fixture
def bgr_image():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    img[10:30, 20:40] = (0, 0, 255)
    return img


@pytest.fixture
def gray_image(bgr_image):
    import cv2

    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)


@pytest.fixture
def make_frame():
    import time

    from edgelens.core.frame import Frame

    def _make(image, seq=1, source_id="test:0", **kw):
        return Frame(image=image, seq=seq, t_cap=time.monotonic(), source_id=source_id, **kw)

    return _make
