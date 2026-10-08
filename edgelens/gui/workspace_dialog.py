"""工作區對話框：第一次啟動或切換工作區時選擇資料夾，並提醒雲端同步的風險。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from edgelens.core.paths import cloud_sync_provider, default_workspace, ensure_dir
from edgelens.i18n import tr


class WorkspaceDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, current: Path | None = None, first_run: bool = False):
        super().__init__(parent)
        self.setWindowTitle(tr("gui.workspace.title"))
        self.resize(560, 260)
        self._first_run = first_run
        lay = QVBoxLayout(self)
        intro = QLabel(tr("gui.workspace.intro_first" if first_run else "gui.workspace.intro"))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        row = QHBoxLayout()
        self.path_edit = QLineEdit(str(current or default_workspace()))
        self.path_edit.textChanged.connect(self._update_warning)
        browse = QPushButton(tr("common.browse"))
        browse.clicked.connect(self._browse)
        row.addWidget(self.path_edit, 1)
        row.addWidget(browse)
        lay.addLayout(row)
        self.warning = QLabel()
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color:#C77700;")
        lay.addWidget(self.warning)
        lay.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr("common.ok"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr("common.cancel"))
        buttons.accepted.connect(self._ok)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self._update_warning()

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("gui.workspace.title"), self.path_edit.text())
        if folder:
            self.path_edit.setText(folder)

    def _update_warning(self) -> None:
        provider = cloud_sync_provider(self.path_edit.text()) if self.path_edit.text().strip() else None
        self.warning.setText(tr("gui.workspace.cloud_warning", provider=provider) if provider else "")

    def _ok(self) -> None:
        text = self.path_edit.text().strip()
        if not text:
            return
        try:
            ensure_dir(text)
        except OSError as e:
            QMessageBox.warning(self, tr("gui.workspace.title"), tr("gui.workspace.cannot_create", error=str(e)))
            return
        self.accept()

    def chosen_path(self) -> Path:
        return Path(self.path_edit.text().strip()).expanduser()
