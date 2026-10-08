"""前處理測試：座標來回、版面、顏色、量化、灰階。"""

from __future__ import annotations

import numpy as np
import pytest

from edgelens.core.modelcard import InputSpec
from edgelens.inference.base import TensorInfo
from edgelens.inference.preprocess import describe_input, preprocess, resolve_input

H, W = 480, 640   # 原始畫面


def make_image(h=H, w=W, color=(10, 20, 30)):
    img = np.full((h, w, 3), 0, np.uint8)
    img[:] = color
    return img


def spec(**kw):
    base = {"size": (320, 320), "layout": "NCHW", "color": "RGB", "resize": "letterbox", "scale": 1.0 / 255}
    base.update(kw)
    return InputSpec(**base)


POINTS = np.array([[0, 0, 640, 480], [100, 50, 300, 400], [320, 240, 321, 241]], np.float32)


@pytest.mark.parametrize("mode", ["letterbox", "stretch", "center_crop"])
@pytest.mark.parametrize("size", [(320, 320), (416, 256), (200, 500)])
def test_coordinates_round_trip(mode, size):
    _, meta = preprocess(make_image(), spec(size=size, resize=mode))
    back = meta.to_original(meta.to_input(POINTS))
    assert np.allclose(back, POINTS, atol=1e-3)
    assert meta.input_size == size and meta.orig_size == (W, H) and meta.mode == mode


def test_letterbox_content_and_padding():
    img = make_image(color=(0, 0, 0))
    img[100:200, 200:300] = 255
    t, meta = preprocess(img, spec(size=(640, 640), pad_value=114, color="BGR", scale=1.0))
    assert t.shape == (1, 3, 640, 640)
    assert meta.pad[0] == 0 and meta.pad[1] == 80 and meta.scale == (1.0, 1.0)
    assert t[0, 0, 0, 0] == 114                      # 補邊
    x1, y1, x2, y2 = meta.to_input(np.array([[200, 100, 300, 200]], np.float32))[0].astype(int)
    assert t[0, 0, (y1 + y2) // 2, (x1 + x2) // 2] == 255
    assert t[0, 0, y1 - 3, (x1 + x2) // 2] == 0


def test_center_crop_negative_pad_and_content():
    img = make_image(color=(0, 0, 0))
    img[:, 320:] = 200
    t, meta = preprocess(img, spec(size=(240, 240), resize="center_crop", color="BGR", scale=1.0))
    assert meta.pad[0] < 0 and meta.pad[1] == 0 and meta.scale[0] == pytest.approx(0.5)
    assert t[0, 0, 100, 5] == 0 and t[0, 0, 100, 235] == 200
    x = meta.to_input(np.array([[320, 0, 320, 0]], np.float32))[0, 0]
    assert 118 <= x <= 122   # 原圖中線落在裁切後的中間


def test_stretch_scales_differ():
    _, meta = preprocess(make_image(), spec(size=(320, 160), resize="stretch"))
    assert meta.scale == (0.5, pytest.approx(160 / 480)) and meta.pad == (0.0, 0.0)


def test_layouts_and_batch_and_contiguous():
    img = make_image()
    for layout, shape in (("NCHW", (1, 3, 320, 320)), ("NHWC", (1, 320, 320, 3))):
        t, _ = preprocess(img, spec(layout=layout))
        assert t.shape == shape and t.flags["C_CONTIGUOUS"] and t.dtype == np.float32
    # auto：依模型輸入形狀
    t, _ = preprocess(img, spec(size=None, layout="auto"), TensorInfo("i", (1, 224, 192, 3)))
    assert t.shape == (1, 224, 192, 3)
    t, _ = preprocess(img, spec(size=None, layout="auto"), TensorInfo("i", (1, 3, 224, 192)))
    assert t.shape == (1, 3, 224, 192)


def test_size_resolution():
    assert resolve_input(spec(size=None), TensorInfo("i", (1, 3, None, None))).size == (640, 640)
    assert resolve_input(spec(size=None), None).size == (640, 640)
    assert resolve_input(spec(size=None, layout="auto"), TensorInfo("i", (1, 3, 256, 384))).size == (384, 256)
    assert resolve_input(spec(size=(100, 50), layout="auto"), TensorInfo("i", (1, 3, 256, 384))).size == (100, 50)
    p = resolve_input(spec(size=None, layout="auto"), TensorInfo("i", (1, None, None, 3)))
    assert p.layout == "NHWC" and p.size == (640, 640) and p.channels == 3


def test_color_orders():
    img = make_image(color=(10, 20, 30))  # BGR
    rgb, _ = preprocess(img, spec(color="RGB", scale=1.0))
    bgr, _ = preprocess(img, spec(color="BGR", scale=1.0))
    assert rgb[0, :, 160, 160].tolist() == [30, 20, 10]
    assert bgr[0, :, 160, 160].tolist() == [10, 20, 30]


def test_gray_and_gray_input_images():
    img = make_image(color=(10, 20, 30))
    t, _ = preprocess(img, spec(color="GRAY", scale=1.0), TensorInfo("i", (1, 1, 320, 320)))
    assert t.shape == (1, 1, 320, 320)
    g = np.full((H, W), 77, np.uint8)
    t3, _ = preprocess(g, spec(color="RGB", scale=1.0), TensorInfo("i", (1, 3, 320, 320)))
    assert t3.shape == (1, 3, 320, 320) and (t3[0, :, 160, 160] == 77).all()
    t1, _ = preprocess(g, spec(color="GRAY", scale=1.0, layout="auto"), TensorInfo("i", (1, 320, 320, 1)))
    assert t1.shape == (1, 320, 320, 1)
    t, _ = preprocess(g, spec(color="GRAY", scale=1.0))     # 無模型資訊：GRAY → 1 通道
    assert t.shape[1] == 1
    bgra = np.dstack([make_image(), np.full((H, W), 255, np.uint8)])
    assert preprocess(bgra, spec())[0].shape == (1, 3, 320, 320)


def test_mean_std_normalization():
    img = make_image(color=(10, 20, 30))
    t, _ = preprocess(img, spec(color="RGB", scale=1 / 255, mean=(0.5, 0.4, 0.3), std=(0.5, 0.2, 0.1)))
    exp = [(30 / 255 - 0.5) / 0.5, (20 / 255 - 0.4) / 0.2, (10 / 255 - 0.3) / 0.1]
    assert t[0, :, 160, 160].tolist() == pytest.approx(exp, abs=1e-5)


def test_quantized_inputs():
    img = make_image(color=(10, 20, 30))
    info = TensorInfo("i", (1, 3, 320, 320), "int8", quant=(1 / 128, -1))
    t, _ = preprocess(img, spec(color="RGB"), info)
    assert t.dtype == np.int8
    assert t[0, :, 160, 160].tolist() == [round(30 / 255 * 128) - 1, round(20 / 255 * 128) - 1,
                                          round(10 / 255 * 128) - 1]
    uq = TensorInfo("i", (1, 320, 320, 3), "uint8", quant=(1 / 255, 0))
    t, _ = preprocess(img, spec(layout="NHWC"), uq)
    assert t.dtype == np.uint8 and t[0, 160, 160].tolist() == [30, 20, 10]
    big = TensorInfo("i", (1, 3, 320, 320), "int8", quant=(0.001, 0))
    t, _ = preprocess(img, spec(), big)
    assert t.max() == 127   # 超出範圍會夾住
    # uint8 沒有量化資訊 → 直接放像素值
    t, _ = preprocess(img, spec(), TensorInfo("i", (1, 3, 320, 320), "uint8"))
    assert t.dtype == np.uint8 and t[0, :, 160, 160].tolist() == [30, 20, 10]
    t, _ = preprocess(img, spec(), TensorInfo("i", (1, 3, 320, 320), "int8"))
    assert t.dtype == np.int8 and t[0, 0, 160, 160] == 30 - 128
    t, _ = preprocess(img, spec(dtype="float16"))
    assert t.dtype == np.float16


def test_describe_input():
    s = describe_input(TensorInfo("i", (1, 3, 640, 640)), spec(size=None, layout="auto"))
    assert "640×640" in s and "RGB" in s and "NCHW" in s and "float32" in s
    assert "NHWC" in describe_input(TensorInfo("i", (1, 96, 96, 1), "uint8"), spec(size=None, layout="auto"))
    assert "GRAY" in describe_input(TensorInfo("i", (1, 96, 96, 1), "uint8"), spec(size=None, layout="auto"))
