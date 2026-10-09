"""資料集面板、瀏覽器、切分與匯出（offscreen，只用合成照片）。"""

from __future__ import annotations

import json
import time
import zipfile

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QObject, Qt, Signal

from edgelens.core.frame import Frame
from edgelens.core.project import Project
from edgelens.dataset.writer import DatasetWriter
from edgelens.gui.dataset_browser import ALL, DELETED, DatasetBrowser
from edgelens.gui.export_dialog import ExportDialog
from edgelens.gui.panels.dataset import DatasetPanel
from edgelens.gui.thumbs import ThumbLoader, shared_writer

pytestmark = pytest.mark.gui


class StubCtx(QObject):
    projectChanged = Signal(object)
    languageChanged = Signal(str)
    datasetChanged = Signal()

    def __init__(self, project):
        super().__init__()
        self.project = project


def noise(seed: int, w: int = 64, h: int = 48) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return cv2.resize(rng.integers(0, 256, (6, 8, 3), dtype=np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def frame(img, seq, t):
    return Frame(image=img, seq=seq, t_cap=time.monotonic(), source_id="cam:0", t_wall=t)


@pytest.fixture
def project(tmp_path):
    p = Project(tmp_path / "ws" / "projects" / "中文專案", "中文專案")
    p.add_class("紅蘋果", "apple")
    p.add_class("香蕉", "banana")
    p.add_class("橘子", "orange")
    p.save()
    return p


@pytest.fixture
def filled(project):
    """30 張：蘋果 12（含 4 張連拍組）、香蕉 10、橘子 8；其中有 2 張幾乎一樣、1 張很暗。"""
    w = DatasetWriter(project)
    t0 = time.time() - 1000
    n = 0
    for cid, count in (("apple", 12), ("banana", 10), ("orange", 8)):
        for i in range(count):
            img = noise(n * 7 + 1)
            kw = {}
            if cid == "apple" and i < 4:
                kw["burst_id"] = "burst1"
            if cid == "banana" and i < 2:
                kw["group_id"] = "g1"
            if n == 3:
                img = (img * 0.05).astype(np.uint8)   # 太暗
            w.save(frame(img, n, t0 + n), cid, image=img, source={"id": "cam:0" if n % 2 else "cam:1", "name": f"相機{n % 2}"},
                   **kw)
            n += 1
    # 兩張幾乎一樣的相似照片
    base = noise(999)
    w.save(frame(base, 100, t0 + 100), "orange")
    w.save(frame(np.clip(base.astype(int) + 1, 0, 255).astype(np.uint8), 101, t0 + 101), "orange")
    return project


def wait_thumbs(qtbot, browser, timeout=8000):
    """模擬整個清單都看得到：對每一列要一次縮圖，等背景載入完成。"""
    n = min(browser.model.rowCount(), 30)
    for i in range(n):
        browser.model.data(browser.model.index(i), Qt.ItemDataRole.DecorationRole)
    qtbot.waitUntil(lambda: browser.loader.pending() == 0 and all(
        browser.model.pixmap(i) is not None for i in range(n)), timeout=timeout)


def make_browser(qtbot, project):
    b = DatasetBrowser(StubCtx(project), project)
    qtbot.addWidget(b)
    b.show()
    return b


def idle(qtbot, b, timeout=5000):
    qtbot.waitUntil(lambda: not b.busy, timeout=timeout)


def test_panel_counts_and_hint(qtbot, filled):
    ctx = StubCtx(filled)
    panel = DatasetPanel(ctx, None)
    qtbot.addWidget(panel)
    panel.show()
    assert panel.count_labels["apple"].text() == "12"
    assert panel.count_labels["banana"].text() == "10"
    assert panel.count_labels["orange"].text() == "10"
    assert "32" in panel.total_label.text()
    assert panel.recent_model.rowCount() == 9
    # 中文類別名稱出現在列上
    assert any(w.text() == "紅蘋果" for w in panel.grid_host.findChildren(type(panel.total_label)))
    panel.shutdown()


def test_panel_detects_change_without_signal(qtbot, project, bgr_image):
    ctx = StubCtx(project)
    panel = DatasetPanel(ctx, None)
    qtbot.addWidget(panel)
    panel.show()
    assert "0" in panel.total_label.text()
    shared_writer(project).save(frame(bgr_image, 1, time.time()), "apple")
    panel.refresh()
    assert panel.count_labels["apple"].text() == "1"
    panel.shutdown()


def test_panel_project_switch(qtbot, filled, project, tmp_path):
    other = Project(tmp_path / "ws" / "projects" / "other", "other")
    other.add_class("A", "a")
    other.save()
    panel = DatasetPanel(StubCtx(filled), None)
    qtbot.addWidget(panel)
    panel.on_project_changed(other)
    assert list(panel.count_labels) == ["a"]
    panel.on_project_changed(None)
    assert not panel.browser_button.isEnabled()
    panel.shutdown()


def test_panel_rebuild_picks_up_dropped_photos(qtbot, project, bgr_image):
    ok, buf = cv2.imencode(".jpg", bgr_image)
    (project.dataset_dir / "apple").mkdir(parents=True, exist_ok=True)
    (project.dataset_dir / "apple" / "學生的照片.jpg").write_bytes(buf.tobytes())
    panel = DatasetPanel(StubCtx(project), None)
    qtbot.addWidget(panel)
    assert panel.count_labels["apple"].text() == "0"
    panel.rebuild()
    qtbot.waitUntil(lambda: panel._rebuild_task is None, timeout=5000)
    assert panel.count_labels["apple"].text() == "1"
    panel.shutdown()


def test_browser_thumbs_and_relabel_delete_restore(qtbot, filled):
    b = make_browser(qtbot, filled)
    assert b.model.rowCount() == 32
    wait_thumbs(qtbot, b)
    b.set_class_filter("banana")
    assert b.model.rowCount() == 10
    first = [r.file for r in b.model.records[:2]]
    b.select_files(first)
    assert len(b.selected_records()) == 2
    b.relabel_selected("apple")
    idle(qtbot, b)
    assert len(shared_writer(filled).list_samples("apple")) == 14
    assert b.model.rowCount() == 8
    # 刪除再復原
    b.select_files([b.model.records[0].file])
    b.delete_selected()
    idle(qtbot, b)
    assert b.model.rowCount() == 7
    b.set_class_filter(DELETED)
    assert b.model.rowCount() == 1
    b.select_files([b.model.records[0].file])
    assert b.restore_button.isEnabled() and not b.delete_button.isVisibleTo(b)
    b.restore_selected()
    idle(qtbot, b)
    assert b.model.rowCount() == 0
    b.set_class_filter(ALL)
    assert b.model.rowCount() == 32
    b.shutdown()


def test_browser_keyboard_hotkey_and_delete(qtbot, filled):
    b = make_browser(qtbot, filled)
    b.activateWindow()
    qtbot.waitUntil(b.isActiveWindow, timeout=3000)
    b.set_class_filter("orange")
    b.select_files([b.model.records[0].file])
    qtbot.keyClick(b.view, Qt.Key.Key_1)      # apple 的快捷鍵
    idle(qtbot, b)
    assert len(shared_writer(filled).list_samples("apple")) == 13
    b.set_class_filter("apple")
    b.select_files([b.model.records[0].file])
    qtbot.keyClick(b.view, Qt.Key.Key_Delete)
    idle(qtbot, b)
    assert len(shared_writer(filled).list_samples("apple")) == 12
    b.shutdown()


def test_browser_filters_and_sort(qtbot, filled):
    b = make_browser(qtbot, filled)
    b.set_quality_filter("dark")
    assert 1 <= b.model.rowCount() <= 3
    b.set_quality_filter("all")
    b.set_burst_filter("burst1")
    assert b.model.rowCount() == 4
    b.set_burst_filter("any")
    assert b.model.rowCount() == 4
    b.set_burst_filter("all")
    b.set_camera_filter("cam:1")
    n1 = b.model.rowCount()
    b.set_camera_filter("cam:0")
    assert 0 < n1 < 32 and n1 + b.model.rowCount() <= 32
    b.set_camera_filter(None)
    b.set_sort("time_asc")
    times = [r.time["epoch"] for r in b.model.records]
    assert times == sorted(times)
    b.set_sort("sharp_desc")
    sh = [r.quality["sharpness"] for r in b.model.records]
    assert sh == sorted(sh, reverse=True)
    b.shutdown()


def test_browser_similar_groups(qtbot, filled):
    b = make_browser(qtbot, filled)
    b.find_similar(True)
    qtbot.waitUntil(lambda: b._similar is not None, timeout=5000)
    assert b.model.rowCount() >= 2
    assert any(len(g) >= 2 for g in b._similar)
    # 最後那兩張幾乎一樣的要相鄰
    files = [r.file for r in b.model.records]
    last_two = [r.file for r in shared_writer(filled).list_samples("orange")[-2:]]
    assert abs(files.index(last_two[0]) - files.index(last_two[1])) == 1
    b.find_similar(False)
    assert b.model.rowCount() == 32
    b.shutdown()


def test_browser_selection_preview_and_metadata(qtbot, filled):
    b = make_browser(qtbot, filled)
    b.select_files([next(r.file for r in b.model.records if r.processed_file)])
    text = b.meta.toPlainText()
    assert "紅蘋果" in text or "香蕉" in text or "橘子" in text
    assert b.preview.pixmap() is not None and not b.preview.pixmap().isNull()
    assert b.processed_button.isEnabled()
    b.processed_button.setChecked(True)
    assert not b.preview.pixmap().isNull()
    b.shutdown()


def test_browser_picks_up_external_change(qtbot, filled, bgr_image):
    b = make_browser(qtbot, filled)
    assert b.model.rowCount() == 32
    shared_writer(filled).save(frame(bgr_image, 500, time.time()), "banana")
    b._poll()
    assert b.model.rowCount() == 33
    b.shutdown()


def test_thumb_cache_not_regenerated(qtbot, filled):
    b = make_browser(qtbot, filled)
    wait_thumbs(qtbot, b)
    assert b.loader.generated >= 30
    b.shutdown()
    b2 = make_browser(qtbot, filled)
    wait_thumbs(qtbot, b2)
    assert b2.loader.generated == 0
    assert b2.loader.disk_hits >= 30
    b2.shutdown()


def test_thumb_cache_survives_relabel(qtbot, filled):
    loader = ThumbLoader(filled)
    from edgelens.gui.thumbs import thumb_key
    rec = shared_writer(filled).list_samples("apple")[0]
    k1 = thumb_key(rec)
    moved = shared_writer(filled).relabel(rec.file, "banana")
    assert thumb_key(moved) == k1
    loader.stop()


def test_split_table_and_zip_export(qtbot, filled):
    dlg = ExportDialog(StubCtx(filled), filled)
    qtbot.addWidget(dlg)
    result = dlg.do_split()
    assert result is not None
    assert dlg.table.rowCount() == 3
    total = sum(int(dlg.table.item(r, c).text()) for r in range(3) for c in (1, 2, 3))
    assert total == 32
    assert (filled.dataset_dir / "splits.json").exists()
    dlg.chk_names.setChecked(True)
    dlg.start_export()
    qtbot.waitUntil(lambda: dlg.last_zip is not None, timeout=10000)
    with zipfile.ZipFile(dlg.last_zip) as z:
        names = z.namelist()
        assert sum(n.endswith(".jpg") for n in names) == 32  # 預設只有原圖
    dlg.close()


def test_zip_export_default_counts(qtbot, filled):
    dlg = ExportDialog(StubCtx(filled), filled)
    qtbot.addWidget(dlg)
    dlg.chk_names.setChecked(True)
    dlg.start_export()
    qtbot.waitUntil(lambda: dlg.last_zip is not None, timeout=10000)
    with zipfile.ZipFile(dlg.last_zip) as z:
        names = z.namelist()
        assert sum(n.endswith(".jpg") and "_proc" not in n for n in names) == 32
        assert {n.split("/")[0] for n in names if "/" in n} == {"紅蘋果", "香蕉", "橘子"}
        assert z.read("labels.txt").decode("utf-8").split() == ["紅蘋果", "香蕉", "橘子"]
        assert len(z.read("metadata.jsonl").splitlines()) == 32
    assert "exports" in str(dlg.last_zip) and dlg.reveal_button.isVisibleTo(dlg)
    assert dlg.next_label.text()
    dlg.close()


def test_split_folders_export_with_edge(qtbot, filled):
    dlg = ExportDialog(StubCtx(filled), filled)
    qtbot.addWidget(dlg)
    dlg.radio_split.setChecked(True)
    dlg.edge_combo.setCurrentIndex(dlg.edge_combo.findData(224))
    dlg.chk_processed.setChecked(True)
    dlg.start_export()
    qtbot.waitUntil(lambda: dlg.last_zip is not None, timeout=10000)
    with zipfile.ZipFile(dlg.last_zip) as z:
        parts = {n.split("/")[0] for n in z.namelist() if n.endswith(".jpg")}
        assert parts == {"train", "val", "test"}
        meta = [json.loads(line) for line in z.read("metadata.jsonl").splitlines()]
        assert len(meta) == 32
    dlg.close()


def test_cancel_export_leaves_no_zip(qtbot, filled):
    dlg = ExportDialog(StubCtx(filled), filled)
    qtbot.addWidget(dlg)
    dlg.start_export()
    dlg.cancel_export()          # 立刻取消
    qtbot.waitUntil(lambda: not dlg.busy, timeout=10000)
    exports = filled.root / "exports"
    leftovers = list(exports.glob("*")) if exports.exists() else []
    if dlg.last_zip is None:     # 取消成功（極少數情況匯出已完成就不檢查）
        assert leftovers == []
        assert dlg.result_label.text()
    dlg.close()


def test_ratio_must_sum_to_100(qtbot, filled):
    dlg = ExportDialog(StubCtx(filled), filled)
    qtbot.addWidget(dlg)
    dlg.spin_test.setValue(30)
    assert dlg.ratio_warn.text() and not dlg.split_button.isEnabled()
    dlg.spin_test.setValue(15)
    assert dlg.split_button.isEnabled()


def test_previous_split_shown(qtbot, filled):
    d1 = ExportDialog(StubCtx(filled), filled)
    qtbot.addWidget(d1)
    d1.spin_train.setValue(60)
    d1.spin_val.setValue(20)
    d1.spin_test.setValue(20)
    d1.seed_spin.setValue(7)
    d1.do_split()
    d2 = ExportDialog(StubCtx(filled), filled)
    qtbot.addWidget(d2)
    assert d2.spin_train.value() == 60 and d2.seed_spin.value() == 7
    assert d2.table.rowCount() == 3


def test_english_labels(qtbot, filled):
    from edgelens import i18n
    i18n.set_language("en")
    b = make_browser(qtbot, filled)
    assert b.class_list.item(0).text().startswith("All")
    b.shutdown()


def test_browser_2000_photos_fast(qtbot, project):
    """2000 張小圖：建立 model 要在 1 秒內，縮圖是看到才載入。"""
    w = DatasetWriter(project)
    img = noise(1, 16, 12)
    for i in range(2000):
        w.save(frame(img, i, 1_700_000_000 + i), ("apple", "banana", "orange")[i % 3], image=None)
    t = time.perf_counter()
    b = DatasetBrowser(StubCtx(project), project)
    qtbot.addWidget(b)
    b.show()
    elapsed = time.perf_counter() - t
    assert b.model.rowCount() == 2000
    assert elapsed < 3.0   # 含載入 metadata；單純 set_records 見下
    t = time.perf_counter()
    b.model.set_records(list(b.model.records))
    assert time.perf_counter() - t < 1.0
    # 沒捲到的縮圖不會被要求（佇列遠小於 2000）
    assert b.loader.pending() < 2000
    b.shutdown()


def test_drag_to_class_list_relabels(qtbot, filled):
    from PySide6.QtCore import QMimeData
    b = make_browser(qtbot, filled)
    b.set_class_filter("apple")
    files = [r.file for r in b.model.records[:3]]
    mime = b.model.mimeData([b.model.index(i) for i in range(3)])
    assert isinstance(mime, QMimeData) and bytes(mime.data("application/x-edgelens-samples")).decode().split("\n") == files
    b.class_list.samplesDropped.emit("banana", files)
    idle(qtbot, b)
    assert len(shared_writer(filled).list_samples("banana")) == 13
    b.shutdown()


def test_painting_requests_visible_thumbs_only(qtbot, filled):
    b = make_browser(qtbot, filled)
    b.grab()   # 畫到的格子才會要求縮圖
    qtbot.waitUntil(lambda: b.model.pixmap(0) is not None, timeout=5000)
    b.shutdown()


def test_dataset_changed_signal_refreshes_panel(qtbot, project):
    """拍照面板存檔後發出 ctx.datasetChanged，資料集面板立刻更新張數（不用等輪詢）。"""
    from edgelens.dataset.writer import shared_writer
    from edgelens.gui.panels.dataset import DatasetPanel

    ctx = StubCtx(project)
    panel = DatasetPanel(ctx, None)
    qtbot.addWidget(panel)
    panel.on_project_changed(project)
    before = len(panel._records)
    cid = project.classes[0].id
    shared_writer(project).save(frame(noise(99), 1, time.time()), cid)
    ctx.datasetChanged.emit()
    qtbot.waitUntil(lambda: len(panel._records) == before + 1, timeout=3000)
