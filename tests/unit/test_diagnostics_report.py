"""diagnostics_report：收集不出錯、去敏、文字與 ZIP。"""

from __future__ import annotations

import json
import logging
import logging.handlers
import zipfile

import pytest

from edgelens import diagnostics_report as dr
from edgelens.core.settings import Settings


@pytest.fixture(autouse=True)
def _no_system_calls(monkeypatch):
    # 不呼叫 system_profiler／序列埠列舉，也不跑會建立 QApplication 的自我檢查
    monkeypatch.setattr(dr, "camera_list", lambda: [{"name": "Fake cam", "index": 0, "backend": "x", "id": "fake"}])
    monkeypatch.setattr(dr, "serial_ports", lambda: [{"device": "/dev/ttyFAKE", "description": "d", "chip": "CH340",
                                                       "likely": True}])


def secret_settings(tmp_path) -> Settings:
    s = Settings(tmp_path / "settings.json")
    s.set("workspace.path", str(tmp_path / "ws"))
    s.set_source_settings("mjpeg:cam", {"controls": {"password": "hunter2", "brightness": 3}})
    s.set("session.sources", [{"kind": "mjpeg", "id": "mjpeg:cam", "name": "c",
                               "params": {"url": "rtsp://bob:hunter2@10.0.0.5/live", "password": "hunter2",
                                          "username": "bob"}}])
    return s


def test_collect_structure(tmp_path):
    r = dr.collect(Settings(tmp_path / "s.json"), tmp_path / "ws", selftest=False)
    for key in ("edgelens_version", "python", "os", "cpu", "packages", "extras", "registries", "plugins",
                "workspace", "cameras", "serial_ports", "log_tail", "settings"):
        assert key in r
    assert r["registries"]["counts"]["sources"] > 0
    assert "numpy" in r["packages"] and r["cameras"][0]["name"] == "Fake cam"
    json.dumps(r)   # 必須能轉成 JSON
    text = dr.to_text(r)
    assert "EdgeLens" in text and "numpy" in text and "/dev/ttyFAKE" in text


def test_collect_with_selftest_runs(tmp_path):
    r = dr.collect(Settings(tmp_path / "s.json"), tmp_path / "ws", selftest=True)
    names = {c["name"] for c in r["selftest"]}
    assert {"python", "registries", "i18n"} <= names


def test_collect_survives_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "package_versions", lambda: 1 / 0)
    r = dr.collect(Settings(tmp_path / "s.json"), None, selftest=False)
    assert "ZeroDivisionError" in json.dumps(r["packages"])
    assert "Packages" in dr.to_text(r)


def test_secrets_are_masked(tmp_path):
    s = secret_settings(tmp_path)
    r = dr.collect(s, None, selftest=False)
    blob = json.dumps(r, ensure_ascii=False)
    assert "hunter2" not in blob
    assert "bob:" not in blob            # 網址帳密
    assert "hunter2" not in dr.to_text(r)
    assert r["settings"]["sources"]["mjpeg:cam"]["controls"]["brightness"] == 3
    assert r["workspace"]["path"] == str(tmp_path / "ws")
    # 原設定物件不能被改到
    assert s.get("session.sources")[0]["params"]["password"] == "hunter2"


def test_sanitize_helpers():
    assert dr.sanitize({"password": "x", "a": {"token": "y", "n": 1}, "empty_password": ""},
                       set()) == {"password": dr.MASK, "a": {"token": dr.MASK, "n": 1}, "empty_password": ""}
    assert dr.sanitize({"custom_key": "v"}, {"custom_key"}) == {"custom_key": dr.MASK}
    assert dr.mask_text("open rtsp://u:p@h/x ok") == "open rtsp://***@h/x ok"
    assert dr.mask_text("http://example.com/a") == "http://example.com/a"
    assert "secret" in dr.secret_keys() or "password" in dr.secret_keys()


def test_log_tail_and_zip(tmp_path, monkeypatch):
    logf = tmp_path / "logs" / "edgelens.log"
    logf.parent.mkdir()
    logf.write_text("\n".join(f"line {i} rtsp://u:pw123@h/x" for i in range(300)), encoding="utf-8")
    (tmp_path / "logs" / "edgelens.log.1").write_text("old", encoding="utf-8")
    handler = logging.handlers.RotatingFileHandler(logf)
    logging.getLogger().addHandler(handler)
    try:
        s = secret_settings(tmp_path)
        r = dr.collect(s, None, selftest=False)
        assert len(r["log_tail"]) == 200 and r["log_tail"][-1].startswith("line 299")
        assert "pw123" not in json.dumps(r)
        out = dr.export_zip(tmp_path / "out" / "diag.zip", r)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert {"report.json", "report.txt", "settings.json", "logs/edgelens.log", "logs/edgelens.log.1"} <= names
        everything = "".join(z.read(n).decode("utf-8") for n in names)
        assert "hunter2" not in everything and "pw123" not in everything
        assert len(z.read("logs/edgelens.log").decode().splitlines()) == 300
        assert json.loads(z.read("report.json"))["edgelens_version"]


def test_default_export_path():
    p = dr.default_export_path()
    assert p.suffix == ".zip" and "EdgeLens_diagnostics" in p.name
