from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QCheckBox
)
from pathlib import Path

class ExportDialog(QDialog):
    """Minimal export dialog used by main window.

    Methods expected by main_window:
    - get_export_dir() -> str
    - get_options() -> dict
    """
    Accepted = QDialog.Accepted

    def __init__(self, format_type=None, config=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导出设置")
        self.resize(400, 120)

        self._dir_edit = QLineEdit()
        browse = QPushButton("浏览")
        browse.clicked.connect(self._browse)

        self._create_config_cb = QCheckBox("生成数据集配置文件")
        self._create_config_cb.setChecked(True)

        ok = QPushButton("确定")
        cancel = QPushButton("取消")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)

        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("导出目录:"))
        row.addWidget(self._dir_edit)
        row.addWidget(browse)
        layout.addLayout(row)

        layout.addWidget(self._create_config_cb)

        btn_row = QHBoxLayout()
        btn_row.addWidget(ok)
        btn_row.addWidget(cancel)
        layout.addLayout(btn_row)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "选择导出目录", "")
        if d:
            self._dir_edit.setText(d)

    def get_export_dir(self) -> str:
        text = self._dir_edit.text().strip()
        return text if text else None

    def get_options(self) -> dict:
        return {'create_config': bool(self._create_config_cb.isChecked())}
