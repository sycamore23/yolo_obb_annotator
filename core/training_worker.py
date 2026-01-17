import sys
import subprocess
import os
from PyQt5.QtCore import QThread, pyqtSignal

class TrainingWorker(QThread):
    """
    后台训练工作类
    通过 subprocess 调用 yolo 命令，以便实时捕获 stdout 日志。
    """
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, data_yaml, model_type="yolov8n.pt", epochs=100, imgsz=640, batch=16, device="cpu"):
        super().__init__()
        self.data_yaml = data_yaml
        self.model_type = model_type
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.device = device
        self._is_running = True

    def run(self):
        try:
            # 构造 Python 代码调用 API，这比直接调用模块更可靠
            # 需要处理路径中的反斜杠，确保在字符串中正确转义
            data_path = self.data_yaml.replace("\\", "/")
            model_path = self.model_type.replace("\\", "/")
            
            # 检查 device 是否为数字字符串，如果是则保持原样，否则加引号
            device_val = f"'{self.device}'" if not str(self.device).isdigit() else self.device

            python_code = (
                f"from ultralytics import YOLO; "
                f"model = YOLO('{model_path}'); "
                f"model.train(data='{data_path}', epochs={self.epochs}, "
                f"imgsz={self.imgsz}, batch={self.batch}, device={device_val})"
            )
            
            cmd = [sys.executable, "-c", python_code]
            
            self.log_signal.emit(f"启动训练线程...\n")
            self.log_signal.emit(f"数据集: {self.data_yaml}\n")
            self.log_signal.emit(f"模型: {self.model_type}\n\n")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1,
                universal_newlines=True
            )
            
            while self._is_running:
                line = process.stdout.readline()
                if not line:
                    break
                self.log_signal.emit(line)
                
            process.wait()
            
            if process.returncode == 0:
                self.finished_signal.emit(True, "训练成功完成！")
            else:
                self.finished_signal.emit(False, f"训练异常退出，退出码: {process.returncode}")
                
        except Exception as e:
            self.finished_signal.emit(False, f"启动训练失败: {str(e)}")

    def _check_old_version(self):
        # 简单判定，通常最新版直接 python -m ultralytics 即可
        return False

    def stop(self):
        self._is_running = False
