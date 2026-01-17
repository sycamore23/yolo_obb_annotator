from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QListWidget, QDialogButtonBox, QMessageBox,
    QCheckBox, QHBoxLayout
)
from PyQt5.QtCore import Qt
from pathlib import Path


class RenameHistoryDialog(QDialog):
    """列出 `rename_*` 备份目录并允许选择一个进行撤销。

    使用示例:
        dialog = RenameHistoryDialog(config, parent)
        if dialog.exec_() == QDialog.Accepted:
            record_path = dialog.get_selected_record()
            project_manager.undo_last_rename(record_path)
    """

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("重命名历史")
        self.resize(600, 400)

        self.config = config

        self.layout = QVBoxLayout(self)

        self.layout.addWidget(QLabel("请选择要撤销的重命名记录（按时间排序，最近在上方）："))

        # Top: list of records
        self.list_widget = QListWidget()
        self.list_widget.setFixedHeight(140)
        self.list_widget.itemSelectionChanged.connect(self._on_record_selected)
        self.layout.addWidget(self.list_widget)

        # Middle: preview mappings
        self.layout.addWidget(QLabel("映射预览（原名  ->  新名）："))
        self.mapping_list = QListWidget()
        self.layout.addWidget(self.mapping_list)

        # Options
        opts_layout = QHBoxLayout()
        self.save_after_undo_cb = QCheckBox("撤销后保存项目")
        self.save_after_undo_cb.setChecked(True)
        opts_layout.addWidget(self.save_after_undo_cb)
        opts_layout.addStretch()
        self.layout.addLayout(opts_layout)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

        self._populate()

    def _populate(self):
        self.list_widget.clear()
        backup_root = Path(self.config.get_backup_dir())
        if not backup_root.exists():
            return

        # list rename_* subdirs sorted by name descending (timestamp suffix)
        items = sorted([p for p in backup_root.iterdir() if p.is_dir() and p.name.startswith('rename_')], reverse=True)

        for p in items:
            record_file = p / 'rename_record.json'
            display = p.name
            if record_file.exists():
                try:
                    # try to read mappings length for nicer display
                    import json
                    with open(record_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        n = len(data.get('mappings', []))
                        display = f"{p.name}  —  {n} files"
                except Exception:
                    pass

            self.list_widget.addItem(display)
            # store the path as data on the QListWidgetItem
            self.list_widget.item(self.list_widget.count() - 1).setData(Qt.UserRole, str(record_file))

        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    def _on_accept(self):
        if not self.list_widget.currentItem():
            QMessageBox.warning(self, "未选择", "请先选择一条重命名记录")
            return

        self.accept()

    def get_selected_record(self):
        item = self.list_widget.currentItem()
        if not item:
            return None
        return item.data(Qt.UserRole)

    def should_save_project(self):
        return bool(self.save_after_undo_cb.isChecked())

    def _on_record_selected(self):
        """当用户选择记录时，加载并显示映射预览。"""
        self.mapping_list.clear()
        item = self.list_widget.currentItem()
        if not item:
            return

        record_file = item.data(Qt.UserRole)
        if not record_file:
            return

        try:
            import json
            with open(record_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                mappings = data.get('mappings', [])
                # show original -> new mapping
                for m in mappings:
                    old = m.get('old') or ''
                    new = m.get('new') or ''
                    self.mapping_list.addItem(f"{Path(old).name}  ->  {Path(new).name}")
        except Exception:
            pass
