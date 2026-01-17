from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QFileDialog, QDoubleSpinBox, QGroupBox, QMessageBox
)
from PyQt5.QtCore import Qt

class DatasetSplitDialog(QDialog):
    """数据集划分对话框"""
    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("自动划分数据集 - 划分比例设置")
        self.resize(450, 300)
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # 1. 导出目录
        dir_group = QGroupBox("导出目录")
        dir_layout = QHBoxLayout()
        self.dir_edit = QLineEdit()
        # 默认使用配置中的输出目录
        if self.config:
            self.dir_edit.setText(str(self.config.get_workspace_path() / "split_dataset"))
        
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse)
        dir_layout.addWidget(self.dir_edit)
        dir_layout.addWidget(browse_btn)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)
        
        # 2. 比例设置
        ratio_group = QGroupBox("划分比例 (总和需为 1.0)")
        ratio_layout = QVBoxLayout()
        
        # 训练集
        train_row = QHBoxLayout()
        train_row.addWidget(QLabel("训练集 (Train):"))
        self.train_spin = QDoubleSpinBox()
        self.train_spin.setRange(0, 1.0)
        self.train_spin.setSingleStep(0.1)
        self.train_spin.setValue(0.7)
        train_row.addWidget(self.train_spin)
        ratio_layout.addLayout(train_row)
        
        # 验证集
        val_row = QHBoxLayout()
        val_row.addWidget(QLabel("验证集 (Val):"))
        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(0, 1.0)
        self.val_spin.setSingleStep(0.1)
        self.val_spin.setValue(0.2)
        val_row.addWidget(self.val_spin)
        ratio_layout.addLayout(val_row)
        
        # 测试集
        test_row = QHBoxLayout()
        test_row.addWidget(QLabel("测试集 (Test):"))
        self.test_spin = QDoubleSpinBox()
        self.test_spin.setRange(0, 1.0)
        self.test_spin.setSingleStep(0.1)
        self.test_spin.setValue(0.1)
        test_row.addWidget(self.test_spin)
        ratio_layout.addLayout(test_row)
        
        ratio_group.setLayout(ratio_layout)
        layout.addWidget(ratio_group)
        
        # 3. 按钮
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("开始划分")
        ok_btn.clicked.connect(self._on_ok)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "选择导出目录", "")
        if d:
            self.dir_edit.setText(d)
            
    def _on_ok(self):
        total = self.train_spin.value() + self.val_spin.value() + self.test_spin.value()
        if abs(total - 1.0) > 1e-5:
            QMessageBox.warning(self, "警告", f"比例总和必须为 1.0 (当前: {total:.2f})")
            return
        
        if not self.dir_edit.text().strip():
            QMessageBox.warning(self, "警告", "请选择导出目录")
            return
            
        self.accept()
        
    def get_data(self):
        return {
            "export_dir": self.dir_edit.text().strip(),
            "ratios": {
                "train": self.train_spin.value(),
                "val": self.val_spin.value(),
                "test": self.test_spin.value()
            }
        }
