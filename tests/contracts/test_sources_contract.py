"""每個註冊的相機來源都必須通過的檢查（新來源的驗收門檻）。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from edgelens.core.controls import ControlResult, ControlSpec, ControlType
from edgelens.core.frame import Frame
from edgelens.core.registry import discover
from edgelens.i18n import LANGUAGES, has
from edgelens.sources.base import SOURCES, FrameSource, SourceCapabilities, SourceError, SourceInfo

discover()
KINDS = SOURCES.kinds()


def test_at_least_one_source():
    assert "synthetic" in KINDS


@pytest.mark.parametrize("kind", KINDS)
def test_class_attributes(kind):
    cls = SOURCES.get(kind)
    assert issubclass(cls, FrameSource)
    assert cls.kind == kind
    for lang in LANGUAGES:
        assert has(cls.display_name_key, lang), f"{kind}: 缺少 {lang} 翻譯 {cls.display_name_key!r}"
        if cls.description_key:
            assert has(cls.description_key, lang), f"{kind}: 缺少 {lang} 翻譯 {cls.description_key!r}"
    ok, reason = cls.available()
    assert isinstance(ok, bool) and isinstance(reason, str)
    for spec in cls.param_specs():
        assert isinstance(spec, ControlSpec)


@pytest.mark.parametrize("kind", KINDS)
def test_safe_discover(kind):
    cls = SOURCES.get(kind)
    if not cls.discover_is_safe:
        pytest.skip("discover() 需要硬體或可能跳出權限視窗")
    infos = cls.discover(timeout=1.0)
    assert isinstance(infos, list)
    for info in infos:
        assert isinstance(info, SourceInfo) and info.kind == kind
        json.dumps(info.to_dict())  # 必須能存成 JSON


@pytest.mark.parametrize("kind", KINDS)
def test_manual_add(kind):
    cls = SOURCES.get(kind)
    if not cls.manual_add:
        pytest.skip("不支援手動加入")
    params = {s.key: s.default_value() for s in cls.param_specs() if s.type != ControlType.ACTION}
    info = cls.info_from_params(params)
    assert info.kind == kind and info.id
    assert cls.info_from_params(params).id == info.id, "同樣的參數必須產生同樣的 id"
    json.dumps(info.to_dict())


def _fixture_or_skip(kind, tmp_path):
    cls = SOURCES.get(kind)
    ok, reason = cls.available()
    if not ok:
        pytest.skip(reason)
    info = cls.contract_fixture(tmp_path)
    if info is None:
        pytest.skip("需要硬體；由該來源自己的單元測試與硬體實測驗證")
    return cls, info


@pytest.mark.parametrize("kind", KINDS)
def test_open_read_close(kind, tmp_path):
    cls, info = _fixture_or_skip(kind, tmp_path)
    src = cls(info)
    assert not src.is_open
    with pytest.raises(SourceError):
        src.read(timeout=0.1)
    src.open()
    try:
        assert src.is_open
        src.open()  # 重複開啟不出錯
        frames = []
        for _ in range(20):
            f = src.read(timeout=2.0)
            if f is not None:
                frames.append(f)
            if len(frames) >= 3:
                break
        assert len(frames) >= 3, "讀不到 3 張畫面"
        seqs = [f.seq for f in frames]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), f"序號必須遞增：{seqs}"
        for f in frames:
            assert isinstance(f, Frame)
            assert f.image.dtype == np.uint8 and f.image.ndim in (2, 3)
            assert f.source_id == info.id
            json.dumps(f.meta)
        caps = src.capabilities()
        assert isinstance(caps, SourceCapabilities)
    finally:
        src.close()
    src.close()  # 重複關閉不出錯
    assert not src.is_open
    # 關閉後可以再開
    src.open()
    try:
        assert src.read(timeout=2.0) is not None or src.read(timeout=2.0) is not None
    finally:
        src.close()


@pytest.mark.parametrize("kind", KINDS)
def test_controls_roundtrip(kind, tmp_path):
    cls, info = _fixture_or_skip(kind, tmp_path)
    src = cls(info)
    src.open()
    try:
        specs = src.controls()
        keys = [s.key for s in specs]
        assert len(keys) == len(set(keys)), "參數 key 不可重複"
        for spec in specs:
            assert isinstance(spec, ControlSpec)
            for lang in LANGUAGES:
                assert spec.label_key and has(spec.label_key, lang) or spec.label, \
                    f"{kind}.{spec.key}: 缺少 {lang} 翻譯 {spec.label_key!r}"
            if spec.type == ControlType.ENUM:
                for _, opt_key in spec.options:
                    assert has(opt_key, "zh_TW") or not opt_key.count("."), f"{kind}.{spec.key}: 選項缺翻譯 {opt_key}"
            if spec.type == ControlType.ACTION:
                continue
            res = src.set_control(spec.key, spec.default_value())
            assert isinstance(res, ControlResult)
            assert res.ok, f"{kind}.{spec.key} 設為預設值失敗：{res.message}"
            assert res.verified in (True, False, None)
        res = src.set_control("__no_such_control__", 1)
        assert isinstance(res, ControlResult) and not res.ok
        assert src.read(timeout=2.0) is not None or src.read(timeout=2.0) is not None
    finally:
        src.close()
