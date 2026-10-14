"""模型庫的 GUI：模型面板「內附模型」選單、下載對話框、類別篩選、新增相機的示範照片。"""

from __future__ import annotations

import hashlib
import json
import types
import zipfile

import pytest
from PySide6.QtCore import Qt

from edgelens import zoo
from edgelens.gui.app import create_app
from edgelens.gui.class_filter_dialog import ClassFilterDialog
from edgelens.sources.synthetic import SyntheticSource
from tests.fixtures.make_onnx import LABELS, generate_all

pytestmark = pytest.mark.gui


def _zip(path, files: dict[str, bytes]) -> str:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def release(tmp_path, monkeypatch):
    """假的 Release：一個 3 類的 YOLOv8 格式模型（附中文類別名稱）與一包示範照片。"""
    import cv2
    import numpy as np

    folder = tmp_path / "release"
    folder.mkdir()
    onnx = generate_all(tmp_path / "onnx")["yolo_t"].read_bytes()
    card = {"version": 1, "name": "Demo", "file": "demo.onnx", "decoder": "yolo_t", "labels_file": "labels.txt",
            "labels_i18n": {"zh_TW": "labels.zh_TW.txt"}}
    model_sha = _zip(folder / "demo.zip", {
        "demo.onnx": onnx, "demo.card.json": json.dumps(card).encode(),
        "labels.txt": "\n".join(LABELS).encode(), "labels.zh_TW.txt": "甲\n乙\n丙\n".encode()})
    ok, jpg = cv2.imencode(".jpg", np.full((48, 64, 3), 128, np.uint8))
    photo_sha = _zip(folder / "photos.zip", {"p1.jpg": jpg.tobytes(), "p2.jpg": jpg.tobytes()})
    items = (
        zoo.ZooItem("demo", "model", "demo.zip", (folder / "demo.zip").stat().st_size, model_sha, "MIT"),
        zoo.ZooItem("photos", "samples", "photos.zip", (folder / "photos.zip").stat().st_size, photo_sha, "CC BY"),
    )
    monkeypatch.setattr(zoo, "RELEASE_URL", folder.as_uri() + "/")
    monkeypatch.setattr(zoo, "CATALOG", items)
    return items


@pytest.fixture
def env(qtbot, tmp_path, release):
    args = types.SimpleNamespace(workspace=str(tmp_path / "ws"), lang="zh_TW", safe_mode=True)
    app, ctx, window = create_app(args)
    qtbot.addWidget(window)
    handle = window.add_source(SyntheticSource.info_from_params({"pattern": "bars", "fps": 60,
                                                                 "resolution": "160x120"}))
    qtbot.waitUntil(lambda: handle.last_packet is not None, timeout=5000)
    panel = window.panel("model")
    yield ctx, window, panel
    window.close()


def _menu_texts(panel) -> list[str]:
    return [a.text() for a in panel.builtin_menu.actions()]


def test_menu_lists_downloadable_models_and_haar(env, release):
    _ctx, _window, panel = env
    texts = _menu_texts(panel)
    item = release[0]
    assert f"{item.title}　下載（約 {item.size_text}）" in texts
    assert release[1].title not in " ".join(texts), "示範照片在新增相機對話框下載，不放模型選單"
    assert "匯入離線模型包…" in texts
    assert any("正臉" in t or "frontalface" in t for t in texts), "Haar 偵測器還在"


def test_download_dialog_installs_then_loads_localized(qtbot, env, release, monkeypatch):
    """沒裝過：按選單 → 進度對話框下載 → 自動載入，類別顯示中文。"""
    _ctx, _window, panel = env
    item = release[0]
    assert panel.use_zoo_item(item)
    assert zoo.is_installed(item)
    qtbot.waitUntil(lambda: panel._session is not None and panel._session.last_result is not None, timeout=8000)
    assert panel._session.builtin
    assert panel._session.info.labels == ["甲", "乙", "丙"]
    assert "甲" in panel._info_values["classes"].text()
    assert not any("下載" in t for t in _menu_texts(panel) if item.title in t)


def test_download_failure_shows_message(qtbot, env, monkeypatch, tmp_path):
    _ctx, _window, panel = env
    monkeypatch.setattr(zoo, "RELEASE_URL", (tmp_path / "missing").as_uri() + "/")
    shown = []
    monkeypatch.setattr("edgelens.gui.zoo_download.QMessageBox.warning",
                        lambda _p, title, text: shown.append(text))
    assert not panel.use_zoo_item(zoo.CATALOG[0])
    assert shown and "網路" in shown[0]


def test_import_offline_pack(qtbot, env, release, monkeypatch, tmp_path):
    _ctx, _window, panel = env
    pack = tmp_path / "老師給的.zip"
    pack.write_bytes((tmp_path / "release" / "demo.zip").read_bytes())
    monkeypatch.setattr("edgelens.gui.zoo_download.QFileDialog.getOpenFileName", lambda *a, **k: (str(pack), ""))
    monkeypatch.setattr("edgelens.gui.zoo_download.QMessageBox.information", lambda *a, **k: None)
    panel.import_zoo_pack()
    assert zoo.is_installed(release[0])
    qtbot.waitUntil(lambda: panel._session is not None and panel._session.state == "running", timeout=8000)


def test_class_filter_button_and_dialog(qtbot, env, release, monkeypatch):
    _ctx, _window, panel = env
    zoo.install(release[0])
    assert panel.use_zoo_item(release[0])
    qtbot.waitUntil(lambda: panel._session is not None and panel._session.last_result is not None, timeout=8000)
    s = panel._session
    assert panel.filter_button.isVisibleTo(panel) and "全部" in panel.filter_button.text()
    assert any(d.class_id == 1 for d in s.last_result.detections)

    class Pick:
        def __init__(self, labels, alt, selected, parent):
            assert labels == ["甲", "乙", "丙"] and alt == list(LABELS) and selected is None

        def exec(self):
            return True

        def selected(self):
            return [0]

    monkeypatch.setattr("edgelens.gui.panels.model.ClassFilterDialog", Pick)
    panel.choose_classes()
    assert s.class_filter == frozenset({0}) and s.card.class_filter == [0]
    assert "1 / 3" in panel.filter_button.text()
    seq = s.last_result.frame_seq
    qtbot.waitUntil(lambda: s.last_result.frame_seq > seq + 1, timeout=5000)
    assert all(d.class_id == 0 for d in s.last_result.detections)
    card = panel.build_card(s)
    assert card.class_filter == [0]


def test_class_filter_dialog_search_and_select(qtbot):
    dlg = ClassFilterDialog(["杯子", "椅子", "人"], ["cup", "chair", "person"], [0, 2])
    qtbot.addWidget(dlg)
    items = [dlg.list.item(i) for i in range(dlg.list.count())]
    assert [it.checkState() == Qt.CheckState.Checked for it in items] == [True, False, True]
    assert dlg.selected() == [0, 2]
    dlg.search.setText("CHA")                    # 英文也找得到
    assert [it.isHidden() for it in items] == [True, False, True]
    dlg.all_button.click()                       # 只作用在搜尋到的
    assert dlg.selected() is None                # 全選＝不篩選
    dlg.search.setText("")
    dlg.none_button.click()
    assert dlg._checked() == []
    assert not dlg.buttons.button(dlg.buttons.StandardButton.Ok).isEnabled()
    dlg.search.setText("杯")
    assert [it.isHidden() for it in items] == [False, True, True]


def test_source_dialog_downloads_and_lists_samples(qtbot, tmp_path, release):
    from edgelens.gui.source_dialog import SourceDialog

    args = types.SimpleNamespace(workspace=str(tmp_path / "ws"), lang="zh_TW", safe_mode=True)
    app, ctx, window = create_app(args)
    qtbot.addWidget(window)
    dlg = SourceDialog(ctx, window)
    qtbot.addWidget(dlg)
    dlg.select_kind("synthetic")
    assert not dlg.samples_button.isVisibleTo(dlg)
    dlg.select_kind("file")
    assert dlg.samples_button.isVisibleTo(dlg) and release[1].title in dlg.samples_button.text()
    dlg.download_samples()
    assert zoo.is_installed(release[1]) and not dlg.samples_button.isVisibleTo(dlg)
    qtbot.waitUntil(lambda: dlg.result_list.count() == 1, timeout=5000)
    dlg._accept_selected()
    info = dlg.selected_info()
    assert info is not None and info.kind == "file" and info.name == release[1].title
    window.close()
