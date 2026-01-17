from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QFileDialog, QSpinBox, QComboBox, QGroupBox, 
    QTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSlot

class TrainDialog(QDialog):
    """模型训练参数设置对话框"""
    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("一键训练模型 - 参数设置")
        self.resize(500, 400)
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # 1. 数据集目录
        dir_group = QGroupBox("数据集配置")
        dir_layout = QVBoxLayout()
        
        # 数据集 yaml 路径
        yaml_row = QHBoxLayout()
        yaml_row.addWidget(QLabel("数据集 YAML:"))
        self.yaml_edit = QLineEdit()
        # 默认尝试寻找 split_dataset 里的 yaml
        if self.config:
            default_yaml = self.config.get_workspace_path() / "split_dataset" / "dataset.yaml"
            if default_yaml.exists():
                self.yaml_edit.setText(str(default_yaml))
            else:
                self.yaml_edit.setText(str(self.config.get_workspace_path() / "dataset.yaml"))
        
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse_yaml)
        yaml_row.addWidget(self.yaml_edit)
        yaml_row.addWidget(browse_btn)
        dir_layout.addLayout(yaml_row)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)
        
        # 2. 训练参数
        param_group = QGroupBox("训练超参数")
        param_layout = QVBoxLayout()
        
        # 模型类型
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("基础模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"])
        if self.config:
            idx = self.model_combo.findText(getattr(self.config.app_config, 'train_model_type', 'yolov8n.pt'))
            if idx >= 0: self.model_combo.setCurrentIndex(idx)
        model_row.addWidget(self.model_combo)
        param_layout.addLayout(model_row)
        
        # Epochs
        epoch_row = QHBoxLayout()
        epoch_row.addWidget(QLabel("迭代次数 (Epochs):"))
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 10000)
        self.epoch_spin.setValue(getattr(self.config.app_config, 'train_epochs', 100) if self.config else 100)
        epoch_row.addWidget(self.epoch_spin)
        param_layout.addLayout(epoch_row)
        
        # Batch Size
        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel("批次大小 (Batch):"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(getattr(self.config.app_config, 'train_batch', 16) if self.config else 16)
        batch_row.addWidget(self.batch_spin)
        param_layout.addLayout(batch_row)
        
        # Image Size
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("图片尺寸 (Imgsz):"))
        self.size_spin = QSpinBox()
        self.size_spin.setRange(32, 2048)
        self.size_spin.setSingleStep(32)
        self.size_spin.setValue(getattr(self.config.app_config, 'train_imgsz', 640) if self.config else 640)
        size_row.addWidget(self.size_spin)
        param_layout.addLayout(size_row)
        
        # Device
        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("训练设备 (Device):"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["0", "cpu", "1", "2"])
        if self.config:
            idx = self.device_combo.findText(getattr(self.config.app_config, 'train_device', '0'))
            if idx >= 0: self.device_combo.setCurrentIndex(idx)
        device_row.addWidget(self.device_combo)
        param_layout.addLayout(device_row)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # 3. 按钮
        btn_layout = QHBoxLayout()
        start_btn = QPushButton("开始训练")
        start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
        start_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(start_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
    def _browse_yaml(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择数据集 YAML 文件", "", "YAML 文件 (*.yaml *.yml)")
        if f:
            self.yaml_edit.setText(f)
            
    def get_params(self):
        return {
            "data": self.yaml_edit.text().strip(),
            "model": self.model_combo.currentText(),
            "epochs": self.epoch_spin.value(),
            "batch": self.batch_spin.value(),
            "imgsz": self.size_spin.value(),
            "device": self.device_combo.currentText()
        }

class TrainingLogDialog(QDialog):
    """训练日志实时显示对话框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("正在训练模型...")
        self.resize(800, 600)
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: 'Consolas', monospace;")
        layout.addWidget(self.log_text)
        
        self.close_btn = QPushButton("关闭")
        self.close_btn.setEnabled(False)
        self.close_btn.clicked.connect(self.accept)
        layout.addWidget(self.close_btn, 0, Qt.AlignRight)
        
    @pyqtSlot(str)
    def append_log(self, text):
        self.log_text.append(text)
        # 自动滚动到底部
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        
    def set_finished(self, success, message):
        self.append_log(f"\n{'='*20}\n{message}\n")
        self.close_btn.setEnabled(True)
        if success:
            self.setWindowTitle("训练已完成")
            QMessageBox.information(self, "成功", message)
        else:
            self.setWindowTitle("训练失败")
            QMessageBox.critical(self, "错误", message)
