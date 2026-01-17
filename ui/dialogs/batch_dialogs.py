from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSpinBox,
    QListWidget, QDialogButtonBox, QCheckBox, QListWidgetItem
)
from PyQt5.QtCore import Qt
from pathlib import Path


class BatchRenameDialog(QDialog):
    """简单的批量重命名对话框。

    提供：前缀、起始索引、是否保留原始序号宽度和预览。
    使用 `get_new_files()` 获取重命名后的完整路径列表（与传入顺序对应）。
    """

    def __init__(self, image_files, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量重命名图片")
        self.image_files = list(image_files)

        self.layout = QVBoxLayout(self)

        # Options
        opts_layout = QHBoxLayout()
        opts_layout.addWidget(QLabel("前缀:"))
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("例如: img_")
        opts_layout.addWidget(self.prefix_edit)

        opts_layout.addWidget(QLabel("起始序号:"))
        self.start_spin = QSpinBox()
        self.start_spin.setMinimum(0)
        self.start_spin.setMaximum(999999)
        self.start_spin.setValue(1)
        opts_layout.addWidget(self.start_spin)

        self.keep_width_cb = QCheckBox("保留原始序号宽度")
        self.keep_width_cb.setChecked(True)
        opts_layout.addWidget(self.keep_width_cb)

        self.layout.addLayout(opts_layout)

        # Preview list
        self.preview_list = QListWidget()
        self.layout.addWidget(QLabel("重命名预览:"))
        self.layout.addWidget(self.preview_list)

        # Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

        # Signals
        self.prefix_edit.textChanged.connect(self._update_preview)
        self.start_spin.valueChanged.connect(self._update_preview)
        self.keep_width_cb.stateChanged.connect(self._update_preview)

        # Initial preview
        self._update_preview()

    def _update_preview(self):
        self.preview_list.clear()
        prefix = self.prefix_edit.text() or ""
        start = int(self.start_spin.value())
        keep_width = bool(self.keep_width_cb.isChecked())

        # Determine width from longest numeric index if keeping width
        if keep_width:
            total = len(self.image_files)
            width = max(1, len(str(start + total - 1)))
        else:
            width = 0

        for i, path in enumerate(self.image_files):
            p = Path(path)
            idx = start + i
            if width:
                idx_str = str(idx).zfill(width)
            else:
                idx_str = str(idx)

            new_name = f"{prefix}{idx_str}{p.suffix}"
            item = QListWidgetItem(f"{p.name}  ->  {new_name}")
            self.preview_list.addItem(item)

    def get_new_files(self):
        """返回按预览生成的新文件完整路径（Path 对象），不执行重命名。

        主流程应在用户确认后使用返回的路径替换项目中的路径，
        并由外部负责执行物理文件重命名（如果需要）。
        """
        prefix = self.prefix_edit.text() or ""
        start = int(self.start_spin.value())
        keep_width = bool(self.keep_width_cb.isChecked())

        if keep_width:
            total = len(self.image_files)
            width = max(1, len(str(start + total - 1)))
        else:
            width = 0

        new_paths = []
        for i, path in enumerate(self.image_files):
            p = Path(path)
            idx = start + i
            idx_str = str(idx).zfill(width) if width else str(idx)
            new_name = f"{prefix}{idx_str}{p.suffix}"
            new_paths.append(str(p.with_name(new_name)))

        return new_paths
