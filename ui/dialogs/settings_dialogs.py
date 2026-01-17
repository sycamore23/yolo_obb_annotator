"""
简单的设置对话框占位实现。提供最小的 UI 以便在 `main_window` 中被调用。
"""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QCheckBox
from PyQt5.QtCore import Qt

from config import Config


class SettingsDialog(QDialog):
    """最小化设置对话框，用于显示并修改一些全局设置。

    仅实现基本结构：接受 `Config` 实例并在确认时调用其保存方法。
    """

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.config = config

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 显示一个简单的自动保存选项作为示例
        self.auto_save_checkbox = QCheckBox("启用自动保存")
        try:
            self.auto_save_checkbox.setChecked(bool(self.config.app_config.auto_save))
        except Exception:
            # 如果 config 中不存在该字段则忽略
            self.auto_save_checkbox.setChecked(False)

        layout.addWidget(self.auto_save_checkbox)

        # 占位说明
        layout.addWidget(QLabel("（这是一个简易设置对话占位实现）"))

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(self._on_accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

    def _on_accept(self):
        # 将选项写入 config（如果该属性存在）
        try:
            if hasattr(self.config, 'app_config') and hasattr(self.config.app_config, 'auto_save'):
                self.config.app_config.auto_save = bool(self.auto_save_checkbox.isChecked())
        except Exception:
            pass

        self.accept()


__all__ = ["SettingsDialog"]
