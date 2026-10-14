"""資料集面板隱藏時不讀 metadata、啟動自動清理、瀏覽器「清空垃圾桶」按鈕。"""
from __future__ import annotations

import time

import pytest
from PySide6.QtWidgets import QMessageBox

from edgelens.dataset.writer import TRASH_KEEP_DAYS, DatasetWriter
from edgelens.gui.dataset_browser import DELETED
from edgelens.gui.panels.dataset import DatasetPanel
from edgelens.gui.thumbs import shared_writer
from tests.gui.test_dataset_panel import StubCtx, frame, idle, make_browser, noise, project  # noqa: F401

pytestmark = pytest.mark.gui


def test_hidden_panel_only_marks_stale(qtbot, project, monkeypatch):  # noqa: F811
    ctx = StubCtx(project)
    panel = DatasetPanel(ctx, None)
    qtbot.addWidget(panel)
    calls = []
    writer = shared_writer(project)
    real = writer.list_samples
    monkeypatch.setattr(writer, "list_samples", lambda *a, **k: calls.append(1) or real(*a, **k))
    panel.on_project_changed(project)
    calls.clear()
    writer.save(frame(noise(1), 1, time.time()), project.classes[0].id)
    ctx.datasetChanged.emit()
    assert calls == [] and panel._stale
    panel.show()
    assert calls and not panel._stale
    assert len(panel._records) == 1
    calls.clear()
    ctx.datasetChanged.emit()   # 可見時照常更新
    assert calls


def test_project_load_purges_old_trash(qtbot, project):  # noqa: F811
    w = shared_writer(project)
    cid = project.classes[0].id
    r = w.save(frame(noise(2), 1, time.time()), cid)
    d = w.delete(r.file)
    records = w._load()
    records[0].deleted_at = time.time() - (TRASH_KEEP_DAYS + 1) * 86400
    w._write(records)
    panel = DatasetPanel(StubCtx(project), None)
    qtbot.addWidget(panel)
    qtbot.waitUntil(lambda: panel._purge_task is None, timeout=5000)
    assert not (project.dataset_dir / d.file).exists()
    assert w.list_samples(include_deleted=True) == []


def test_empty_trash_button(qtbot, project, monkeypatch):  # noqa: F811
    w = DatasetWriter(project)
    cid = project.classes[0].id
    for i in range(3):
        w.delete(w.save(frame(noise(i), i, time.time() + i), cid).file)
    b = make_browser(qtbot, project)
    assert not b.empty_trash_button.isVisibleTo(b)
    b.set_class_filter(DELETED)
    assert b.empty_trash_button.isVisibleTo(b) and b.trash_hint.isVisibleTo(b)
    assert str(TRASH_KEEP_DAYS) in b.trash_hint.text()
    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: asked.append(a[2]) or QMessageBox.StandardButton.No)
    b.empty_trash()
    assert "3" in asked[0] and len(w.list_samples(include_deleted=True)) == 3
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    b.empty_trash()
    idle(qtbot, b)
    assert w.list_samples(include_deleted=True) == [] and b.model.rowCount() == 0
    assert not list((project.dataset_dir / "_trash").iterdir())
    b.shutdown()
