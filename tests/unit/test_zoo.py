"""模型庫：下載（用 file:// 網址模擬，不連網）、sha256 驗證、解壓安全、離線包、取消。"""

from __future__ import annotations

import hashlib
import json
import threading
import zipfile

import pytest

from edgelens import zoo


def _pack(path, files: dict[str, bytes]):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def release(tmp_path, monkeypatch):
    """假的 Release 資料夾 + 只有一個模型、一包照片的清單。"""
    folder = tmp_path / "release"
    folder.mkdir()
    card = json.dumps({"version": 1, "file": "demo.onnx", "labels_file": "labels.txt"}).encode()
    model_sha = _pack(folder / "demo.zip", {"demo.card.json": card, "demo.onnx": b"onnx", "labels.txt": b"cup\n"})
    photo_sha = _pack(folder / "photos.zip", {"a.jpg": b"jpg", "ATTRIBUTION.txt": b"CC BY"})
    items = (
        zoo.ZooItem("demo", "model", "demo.zip", (folder / "demo.zip").stat().st_size, model_sha, "MIT"),
        zoo.ZooItem("photos", "samples", "photos.zip", (folder / "photos.zip").stat().st_size, photo_sha, "CC BY"),
    )
    monkeypatch.setattr(zoo, "RELEASE_URL", folder.as_uri() + "/")
    monkeypatch.setattr(zoo, "CATALOG", items)
    return folder


def test_install_downloads_verifies_and_extracts(release):
    item = zoo.find("demo")
    assert not zoo.is_installed(item)
    seen = []
    target = zoo.install(item, progress=lambda done, total: seen.append((done, total)))
    assert zoo.is_installed(item)
    assert target == zoo.item_dir(item)
    assert zoo.card_path(item).is_file() and (target / "demo.onnx").read_bytes() == b"onnx"
    assert seen[-1] == (item.size, item.size)
    assert not list((zoo.zoo_dir() / "_downloads").iterdir()), "下載暫存檔要清掉"
    assert zoo.install(item) == target       # 已安裝就不再下載


def test_installed_card_loads(release):
    from edgelens.inference.runner import load_model_file

    item = zoo.find("demo")
    zoo.install(item)
    card = load_model_file(zoo.card_path(item))
    assert card.model_path == zoo.item_dir(item) / "demo.onnx"
    assert card.labels == ["cup"]


def test_checksum_mismatch_installs_nothing(release, monkeypatch):
    bad = zoo.ZooItem("demo", "model", "demo.zip", 1, "0" * 64, "MIT")
    monkeypatch.setattr(zoo, "CATALOG", (bad,))
    with pytest.raises(zoo.ZooError) as e:
        zoo.install(bad)
    assert e.value.hint
    assert not zoo.is_installed(bad) and not zoo.item_dir(bad).exists()


def test_missing_file_is_network_error(release, monkeypatch):
    item = zoo.find("demo")
    monkeypatch.setattr(zoo, "RELEASE_URL", (release / "nowhere").as_uri() + "/")
    with pytest.raises(zoo.ZooError) as e:
        zoo.install(item)
    assert e.value.message and e.value.hint
    assert not zoo.is_installed(item)


def test_cancel_stops_download(release):
    item = zoo.find("demo")
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(zoo.Cancelled):
        zoo.install(item, cancel=cancel)
    assert not zoo.is_installed(item)


def test_unpublished_item_explains(monkeypatch):
    item = zoo.ZooItem("later", "samples", "later.zip", 0, "", "CC BY")
    monkeypatch.setattr(zoo, "CATALOG", (item,))
    with pytest.raises(zoo.ZooError):
        zoo.install(item)
    assert zoo.installed_samples() == []


def test_new_catalog_version_counts_as_not_installed(release, monkeypatch):
    item = zoo.find("demo")
    zoo.install(item)
    newer = zoo.ZooItem("demo", "model", "demo.zip", item.size, "f" * 64, "MIT")
    assert not zoo.is_installed(newer)


def test_offline_pack_identified_by_checksum(release, tmp_path):
    copy = tmp_path / "隨便取的名字.zip"
    copy.write_bytes((release / "photos.zip").read_bytes())
    item = zoo.install_zip(copy)
    assert item.id == "photos" and zoo.is_installed(item)
    assert [it.id for it, _ in zoo.installed_samples()] == ["photos"]


def test_offline_pack_unknown(release, tmp_path):
    other = tmp_path / "other.zip"
    _pack(other, {"x.txt": b"x"})
    with pytest.raises(zoo.ZooError) as e:
        zoo.install_zip(other)
    assert e.value.hint


@pytest.mark.parametrize("evil", ["../escape.txt", "/abs.txt", "C:/win.txt"])
def test_zip_cannot_escape_install_folder(release, monkeypatch, tmp_path, evil):
    path = release / "evil.zip"
    sha = _pack(path, {evil: b"x", "demo.card.json": b"{}"})
    item = zoo.ZooItem("evil", "model", "evil.zip", path.stat().st_size, sha, "MIT")
    monkeypatch.setattr(zoo, "CATALOG", (item,))
    with pytest.raises(zoo.ZooError):
        zoo.install(item)
    assert not zoo.item_dir(item).exists()
    assert not (zoo.zoo_dir() / "escape.txt").exists()


def test_model_pack_without_card_rejected(release, monkeypatch):
    path = release / "nocard.zip"
    sha = _pack(path, {"model.onnx": b"x"})
    item = zoo.ZooItem("nocard", "model", "nocard.zip", path.stat().st_size, sha, "MIT")
    monkeypatch.setattr(zoo, "CATALOG", (item,))
    with pytest.raises(zoo.ZooError):
        zoo.install(item)
    assert not zoo.is_installed(item)


def test_remove(release):
    item = zoo.find("photos")
    zoo.install(item)
    zoo.remove(item)
    assert not zoo.is_installed(item)


def test_file_source_discovers_installed_samples(release):
    from edgelens.sources.file import SAMPLES_FPS, FileSource

    assert FileSource.discover() == []
    item = zoo.find("photos")
    zoo.install(item)
    (info,) = FileSource.discover()
    assert info.params["path"] == str(zoo.item_dir(item))
    assert info.params["fps"] == SAMPLES_FPS
    assert info.name == item.title


def test_real_catalog_titles_translated():
    from edgelens.i18n import has

    for item in zoo.CATALOG:
        assert has(f"zoo.item.{item.id}", "zh_TW") and has(f"zoo.item.{item.id}", "en")
        assert item.url.startswith("https://")
