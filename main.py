#!/usr/bin/env python3
"""
YOLO-OBB专业标注工具 - 主程序入口
"""

import sys
import os
import logging
from pathlib import Path

# 添加项目根目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, QTimer

from ui.main_window import YOLOOBBAnnotatorPro
from config import Config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yolo_annotator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SplashScreen(QSplashScreen):
    """启动画面"""
    def __init__(self):
        # 创建启动画面
        pixmap = QPixmap(400, 300)
        pixmap.fill(Qt.white)
        
        super().__init__(pixmap)
        
        # 设置消息颜色
        self.setStyleSheet("color: #2C3E50; font-size: 12px;")
    
    def show_message(self, message):
        """显示消息"""
        self.showMessage(
            message,
            Qt.AlignBottom | Qt.AlignHCenter,
            Qt.black
        )
        QApplication.processEvents()

def setup_application():
    """设置应用程序"""
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO-OBB标注工具")
    app.setOrganizationName("sycamore")
    app.setApplicationVersion("2.0.0")
    
    # 设置高DPI支持
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # 设置样式
    app.setStyle("Fusion")
    
    # 加载样式表
    style_file = current_dir / "resources" / "styles.qss"
    if style_file.exists():
        with open(style_file, 'r', encoding='utf-8') as f:
            app.setStyleSheet(f.read())
    
    return app

def show_splash_and_load(splash):
    """显示启动画面并加载"""
    splash.show_message("初始化配置...")
    
    # 加载配置
    config = Config()
    
    splash.show_message("创建主窗口...")
    
    # 创建主窗口（在后台）
    main_window = YOLOOBBAnnotatorPro(config)
    
    splash.show_message("准备启动...")
    
    return main_window

def main():
    """主函数"""
    try:
        # 创建应用
        app = setup_application()
        
        # 显示启动画面
        splash = SplashScreen()
        splash.show()
        
        # 显示启动消息
        splash.show_message("启动YOLO-OBB标注工具...")
        
        # 延迟加载
        QTimer.singleShot(100, lambda: None)
        
        # 加载主窗口
        main_window = show_splash_and_load(splash)
        
        # 显示主窗口
        main_window.show()
        
        # 关闭启动画面
        splash.finish(main_window)
        
        # 记录启动成功
        logger.info("应用程序启动成功")
        
        # 运行应用
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"应用程序启动失败: {e}", exc_info=True)
        
        # 显示错误对话框
        from PyQt5.QtWidgets import QMessageBox
        
        error_msg = QMessageBox()
        error_msg.setIcon(QMessageBox.Critical)
        error_msg.setWindowTitle("启动错误")
        error_msg.setText("应用程序启动失败")
        error_msg.setInformativeText(str(e))
        error_msg.setDetailedText(str(e.__traceback__))
        error_msg.exec_()
        
        sys.exit(1)

if __name__ == "__main__":
    main()