"""
主窗口模块
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QMessageBox,
    QFileDialog, QInputDialog, QToolBar, QStatusBar, QMenuBar,
    QAction, QDockWidget, QTabWidget, QGroupBox, QCheckBox,
    QSpinBox, QComboBox, QLineEdit, QTextEdit, QProgressBar,
    QApplication, QMenu, QStyleFactory, QShortcut, QGridLayout, QDialog
)
from PyQt5.QtWidgets import QProgressDialog
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize, QPoint, QRect, QObject, QThread
from PyQt5.QtGui import (
    QIcon, QKeySequence, QFont, QColor, QPixmap, QImage,
    QPainter, QPen, QBrush, QPalette
)

from config import Config
from models.annotation_item import AnnotationItem
from models.class_item import ClassItem
from models.enums import AnnotationMode, ExportFormat
from ui.canvas import ImageCanvas
from ui.dialogs.class_dialogs import ClassEditDialog, ClassManagerDialog
from ui.dialogs.export_dialogs import ExportDialog
from ui.dialogs.settings_dialogs import SettingsDialog
from ui.widgets.annotation_list import AnnotationListWidget
from ui.widgets.class_list import ClassListWidget
from utils.file_utils import FileUtils
from utils.export_utils import ExportUtils
from utils.annotation_utils import auto_annotate_image, auto_annotate_all_images
from core.project_manager import ProjectManager
from core.annotation_manager import AnnotationManager
from ui.dialogs.selection_dialog import LabelSelectDialog
from ui.dialogs.split_dialog import DatasetSplitDialog
from ui.dialogs.train_dialogs import TrainDialog, TrainingLogDialog
from core.training_worker import TrainingWorker
import random

class YOLOOBBAnnotatorPro(QMainWindow):
    """YOLO-OBB专业标注工具主窗口"""
    
    # 信号定义
    project_loaded = pyqtSignal(str)  # 项目加载完成
    project_saved = pyqtSignal(str)   # 项目保存完成
    image_changed = pyqtSignal(int)   # 图片切换
    annotation_added = pyqtSignal(AnnotationItem)  # 标注添加
    annotation_updated = pyqtSignal(AnnotationItem)  # 标注更新
    annotation_removed = pyqtSignal(str)  # 标注删除
    
    def __init__(self, config: Config, parent=None):
        """
        初始化主窗口
        
        Args:
            config: 配置管理器
            parent: 父窗口
        """
        super().__init__(parent)
        
        # 保存配置
        self.config = config
        self.project_manager = ProjectManager(config)
        self.annotation_manager = AnnotationManager(config)
        
        # 初始化状态变量
        self._init_state_variables()
        
        # 初始化UI
        self._init_ui()
        
        # 初始化信号连接
        self._init_signals()
        
        # 初始化菜单和工具栏
        self._init_menus()
        self._init_toolbars()
        # 初始化全局快捷键（确保即使焦点在画布也能响应）
        try:
            self._init_shortcuts()
        except Exception:
            pass
        
        # 初始化状态栏
        self._init_statusbar()
        
        # 加载最近项目
        self._load_recent_projects()
        
        # 设置窗口属性
        self._setup_window()
        
        # 显示欢迎消息
        QTimer.singleShot(100, self._show_welcome_message)

        # thread references to avoid GC
        self._project_load_thread: Optional[QThread] = None
        self._project_load_worker: Optional[QObject] = None
    
    def _init_state_variables(self):
        """初始化状态变量"""
        # 项目状态
        self.project_modified = False
        self.project_name = "未命名项目"
        self.current_image_path = None
        self.current_image_index = -1
        
        # 标注状态
        self.current_annotations: List[AnnotationItem] = []
        self.selected_annotation: Optional[AnnotationItem] = None
        
        # 类别状态
        self.classes: List[ClassItem] = []
        self.current_class_index = 0
        
        # 视图状态
        self.show_labels = True
        self.show_confidence = True
        self.show_grid = False
        self.show_statistics = True
        
        # 自动保存计时器
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self._auto_save)
        
        # 检查自动保存设置
        if self.config.app_config.auto_save:
            interval = self.config.app_config.auto_save_interval * 60 * 1000  # 转换为毫秒
            self.auto_save_timer.start(interval)
        
        # 用于在编辑前保存选中标注的快照，以便记录修改前状态
        self._last_selected_snapshot: Optional[AnnotationItem] = None
        
        # 当前图片标签自动保存（每1秒保存一次当前图片的标签文件）
        self.label_autosave_timer = QTimer()
        self.label_autosave_timer.setInterval(1000)  # 1秒
        # connect to saving current annotations; keep it lightweight
        self.label_autosave_timer.timeout.connect(self._save_current_annotations)
    def _init_ui(self):
        """初始化用户界面"""
        # 设置窗口标题
        self.setWindowTitle(f"YOLO-OBB标注工具 - {self.project_name}")
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧面板
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # 中心画布
        self.canvas = ImageCanvas(self)
        splitter.addWidget(self.canvas)
        
        # 右侧面板
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        # 设置分割器比例
        splitter.setSizes([300, 800, 300])
        main_layout.addWidget(splitter)
        
        # 设置样式
        self._apply_styles()
    
    def _create_left_panel(self) -> QWidget:
        """创建左侧面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # 项目信息组
        project_group = QGroupBox("项目信息")
        project_layout = QVBoxLayout()
        
        self.project_name_label = QLabel(f"项目: {self.project_name}")
        self.project_name_label.setWordWrap(True)
        project_layout.addWidget(self.project_name_label)
        
        self.image_info_label = QLabel("图片: 0/0")
        project_layout.addWidget(self.image_info_label)
        
        self.annotation_info_label = QLabel("标注: 0")
        project_layout.addWidget(self.annotation_info_label)
        
        # 项目按钮
        project_buttons_layout = QGridLayout()
        
        self.new_project_btn = QPushButton("新建")
        self.new_project_btn.clicked.connect(self._new_project)
        project_buttons_layout.addWidget(self.new_project_btn, 0, 0)
        
        self.open_project_btn = QPushButton("打开")
        self.open_project_btn.clicked.connect(self._open_project)
        project_buttons_layout.addWidget(self.open_project_btn, 0, 1)
        
        self.save_project_btn = QPushButton("保存")
        self.save_project_btn.clicked.connect(self._save_project)
        project_buttons_layout.addWidget(self.save_project_btn, 1, 0)
        
        self.save_as_btn = QPushButton("另存为")
        self.save_as_btn.clicked.connect(self._save_project_as)
        project_buttons_layout.addWidget(self.save_as_btn, 1, 1)
        
        project_layout.addLayout(project_buttons_layout)
        project_group.setLayout(project_layout)
        layout.addWidget(project_group)
        
        # 图片列表组
        image_group = QGroupBox("图片列表")
        image_layout = QVBoxLayout()
        
        self.image_list = QListWidget()
        self.image_list.itemSelectionChanged.connect(self._on_image_selected)
        image_layout.addWidget(self.image_list)
        
        # 图片操作按钮
        image_buttons_layout = QGridLayout()
        
        self.open_folder_btn = QPushButton("打开文件夹")
        self.open_folder_btn.clicked.connect(self._open_image_folder)
        image_buttons_layout.addWidget(self.open_folder_btn, 0, 0, 1, 2)
        
        self.add_images_btn = QPushButton("添加图片")
        self.add_images_btn.clicked.connect(self._add_images)
        image_buttons_layout.addWidget(self.add_images_btn, 1, 0, 1, 2)
        
        self.prev_image_btn = QPushButton("上一张")
        self.prev_image_btn.clicked.connect(self._prev_image)
        image_buttons_layout.addWidget(self.prev_image_btn, 2, 0)
        
        self.next_image_btn = QPushButton("下一张")
        self.next_image_btn.clicked.connect(self._next_image)
        image_buttons_layout.addWidget(self.next_image_btn, 2, 1)
        
        image_layout.addLayout(image_buttons_layout)
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)
        
        # 显示选项组
        display_group = QGroupBox("显示选项")
        display_layout = QVBoxLayout()
        
        self.show_labels_cb = QCheckBox("显示标签")
        self.show_labels_cb.setChecked(self.show_labels)
        self.show_labels_cb.stateChanged.connect(self._toggle_show_labels)
        display_layout.addWidget(self.show_labels_cb)
        
        self.show_confidence_cb = QCheckBox("显示置信度")
        self.show_confidence_cb.setChecked(self.show_confidence)
        self.show_confidence_cb.stateChanged.connect(self._toggle_show_confidence)
        display_layout.addWidget(self.show_confidence_cb)
        
        self.show_grid_cb = QCheckBox("显示网格")
        self.show_grid_cb.setChecked(self.show_grid)
        self.show_grid_cb.stateChanged.connect(self._toggle_show_grid)
        display_layout.addWidget(self.show_grid_cb)
        
        self.show_statistics_cb = QCheckBox("显示统计")
        self.show_statistics_cb.setChecked(self.show_statistics)
        self.show_statistics_cb.stateChanged.connect(self._toggle_show_statistics)
        display_layout.addWidget(self.show_statistics_cb)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # 添加弹簧
        layout.addStretch()
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """创建右侧面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # 类别管理组
        class_group = QGroupBox("类别管理")
        class_layout = QVBoxLayout()
        
        # 类别列表
        self.class_list_widget = ClassListWidget()
        self.class_list_widget.class_selected.connect(self._on_class_selected)
        self.class_list_widget.class_double_clicked.connect(self._edit_class)
        class_layout.addWidget(self.class_list_widget)
        
        # 类别操作按钮
        class_buttons_layout = QGridLayout()
        
        self.add_class_btn = QPushButton("添加")
        self.add_class_btn.clicked.connect(self._add_class)
        class_buttons_layout.addWidget(self.add_class_btn, 0, 0)
        
        self.edit_class_btn = QPushButton("编辑")
        self.edit_class_btn.clicked.connect(self._edit_class)
        class_buttons_layout.addWidget(self.edit_class_btn, 0, 1)
        
        self.delete_class_btn = QPushButton("删除")
        self.delete_class_btn.clicked.connect(self._delete_class)
        class_buttons_layout.addWidget(self.delete_class_btn, 1, 0)
        
        self.import_classes_btn = QPushButton("导入")
        self.import_classes_btn.clicked.connect(self._import_classes)
        class_buttons_layout.addWidget(self.import_classes_btn, 1, 1)
        
        class_layout.addLayout(class_buttons_layout)
        class_group.setLayout(class_layout)
        layout.addWidget(class_group)
        
        # 标注列表组
        annotation_group = QGroupBox("标注列表")
        annotation_layout = QVBoxLayout()
        
        self.annotation_list_widget = AnnotationListWidget()
        self.annotation_list_widget.annotation_selected.connect(self._on_annotation_list_selected)
        self.annotation_list_widget.annotation_deleted.connect(self._delete_selected_annotation)
        annotation_layout.addWidget(self.annotation_list_widget)
        
        # 标注操作按钮
        annotation_buttons_layout = QGridLayout()
        
        self.clear_annotations_btn = QPushButton("清空")
        self.clear_annotations_btn.clicked.connect(self._clear_annotations)
        annotation_buttons_layout.addWidget(self.clear_annotations_btn, 0, 0, 1, 2)
        
        self.copy_annotation_btn = QPushButton("复制")
        self.copy_annotation_btn.clicked.connect(self._copy_annotation)
        annotation_buttons_layout.addWidget(self.copy_annotation_btn, 1, 0)
        
        self.paste_annotation_btn = QPushButton("粘贴")
        self.paste_annotation_btn.clicked.connect(self._paste_annotation)
        annotation_buttons_layout.addWidget(self.paste_annotation_btn, 1, 1)
        
        annotation_layout.addLayout(annotation_buttons_layout)
        annotation_group.setLayout(annotation_layout)
        layout.addWidget(annotation_group)
        
        # 标注工具组
        tool_group = QGroupBox("标注工具")
        tool_layout = QVBoxLayout()
        
        self.select_tool_btn = QPushButton("选择工具 (S)")
        self.select_tool_btn.setCheckable(True)
        self.select_tool_btn.setChecked(True)
        self.select_tool_btn.clicked.connect(lambda: self._set_annotation_mode(AnnotationMode.NONE))
        tool_layout.addWidget(self.select_tool_btn)
        
        self.rectangle_tool_btn = QPushButton("矩形框 (R)")
        self.rectangle_tool_btn.setCheckable(True)
        self.rectangle_tool_btn.clicked.connect(lambda: self._set_annotation_mode(AnnotationMode.HORIZONTAL))
        tool_layout.addWidget(self.rectangle_tool_btn)
        
        self.rotated_tool_btn = QPushButton("旋转框 (O)")
        self.rotated_tool_btn.setCheckable(True)
        self.rotated_tool_btn.clicked.connect(lambda: self._set_annotation_mode(AnnotationMode.ROTATED))
        tool_layout.addWidget(self.rotated_tool_btn)
        
        self.polygon_tool_btn = QPushButton("多边形 (P)")
        self.polygon_tool_btn.setCheckable(True)
        self.polygon_tool_btn.clicked.connect(lambda: self._set_annotation_mode(AnnotationMode.POLYGON))
        tool_layout.addWidget(self.polygon_tool_btn)
        
        tool_group.setLayout(tool_layout)
        layout.addWidget(tool_group)
        
        # 导出组
        export_group = QGroupBox("导出")
        export_layout = QVBoxLayout()
        
        self.export_yolo_btn = QPushButton("导出YOLO格式")
        self.export_yolo_btn.clicked.connect(lambda: self._export_annotations(ExportFormat.YOLO))
        export_layout.addWidget(self.export_yolo_btn)
        
        self.export_coco_btn = QPushButton("导出COCO格式")
        self.export_coco_btn.clicked.connect(lambda: self._export_annotations(ExportFormat.COCO))
        export_layout.addWidget(self.export_coco_btn)
        
        self.export_voc_btn = QPushButton("导出VOC格式")
        self.export_voc_btn.clicked.connect(lambda: self._export_annotations(ExportFormat.VOC))
        export_layout.addWidget(self.export_voc_btn)
        
        self.create_config_btn = QPushButton("创建配置文件")
        self.create_config_btn.clicked.connect(self._create_dataset_config)
        export_layout.addWidget(self.create_config_btn)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # 添加弹簧
        layout.addStretch()
        
        return panel
    
    def _apply_styles(self):
        """应用样式"""
        # 设置调色板
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(45, 45, 45))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.black)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)
        
        # 设置字体
        font = QFont("Segoe UI", 9)
        self.setFont(font)
    
    def _init_signals(self):
        """初始化信号连接"""
        # 画布信号
        self.canvas.annotation_added.connect(self._on_annotation_added)
        self.canvas.annotation_updated.connect(self._on_annotation_updated)
        self.canvas.annotation_selected.connect(self._on_annotation_selected)
        self.canvas.annotation_label_clicked.connect(self._prompt_label_selection)
        
        # 项目信号
        self.project_loaded.connect(self._on_project_loaded)
        self.project_saved.connect(self._on_project_saved)
        self.image_changed.connect(self._on_image_changed)
        
        # 标注信号
        self.annotation_added.connect(self._update_annotation_list)
        self.annotation_added.connect(self._update_annotation_list)
        self.annotation_updated.connect(self._update_annotation_list)
        self.annotation_removed.connect(self._update_annotation_list)
        
        # 模式切换信号
        self.canvas.mode_changed.connect(self._on_canvas_mode_changed)

    def _sync_canvas_class_colors(self):
        """同步当前 `self.classes` 到画布的颜色映射与类别列表引用。"""
        try:
            color_map = {}
            for cls in self.classes:
                try:
                    color_map[cls.id] = cls.color
                except Exception:
                    pass
            if hasattr(self, 'canvas'):
                try:
                    self.canvas.class_colors = color_map
                    self.canvas.classes = list(self.classes)
                    # force repaint so color changes appear immediately
                    self.canvas.update()
                except Exception:
                    pass
        except Exception:
            pass

    def _on_project_loaded(self, file_path: str):
        """项目加载完成事件处理（用于菜单更新等）"""
        try:
            self._load_recent_projects()
        except Exception:
            pass

    def _on_project_saved(self, file_path: str):
        """项目保存完成事件处理"""
        try:
            self._load_recent_projects()
        except Exception:
            pass

    def _on_image_changed(self, index: int):
        """图片切换事件处理占位符"""
        pass

    def _update_annotation_list(self):
        """更新标注列表 UI。"""
        try:
            if hasattr(self, 'annotation_list_widget'):
                self.annotation_list_widget.set_annotations(self.current_annotations)
            if hasattr(self, 'annotation_info_label'):
                self.annotation_info_label.setText(f"标注: {len(self.current_annotations)}")
        except Exception:
            pass

    def _update_annotation_selection(self):
        """在 UI 中同步当前选中的标注（列表与画布）。"""
        try:
            if self.selected_annotation and hasattr(self, 'annotation_list_widget'):
                try:
                    self.annotation_list_widget.select_annotation(self.selected_annotation.id)
                except Exception:
                    pass

            if self.selected_annotation and hasattr(self, 'canvas'):
                try:
                    self.canvas.select_annotation(self.selected_annotation)
                except Exception:
                    pass
            else:
                # 如果没有选中，确保列表中的选择被清除
                try:
                    if hasattr(self, 'annotation_list_widget'):
                        self.annotation_list_widget._list.clearSelection()
                except Exception:
                    pass
        except Exception:
            pass

    def _update_image_list(self):
        """刷新图片列表 UI。"""
        try:
            self.image_list.clear()
            for i, image_path in enumerate(self.project_manager.image_files):
                file_name = os.path.basename(image_path)
                has_annotation = False
                try:
                    has_annotation = self.project_manager.has_annotation(image_path)
                except Exception:
                    has_annotation = False

                item = QListWidgetItem(file_name)
                if has_annotation:
                    item.setForeground(QColor(0, 200, 0))
                    item.setText(f"{file_name} ✓")

                self.image_list.addItem(item)

            if hasattr(self, 'image_info_label'):
                try:
                    self.image_info_label.setText(f"图片: {self.project_manager.image_count}")
                except Exception:
                    self.image_info_label.setText("图片: 0/0")
        except Exception:
            pass
    
    def _init_menus(self):
        """初始化菜单"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        new_action = QAction("新建项目", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._new_project)
        file_menu.addAction(new_action)
        
        open_action = QAction("打开项目", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_project)
        file_menu.addAction(open_action)
        
        save_action = QAction("保存项目", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_project)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("另存为", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self._save_project_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        open_image_action = QAction("打开图片文件夹", self)
        open_image_action.setShortcut("Ctrl+Shift+O")
        open_image_action.triggered.connect(self._open_image_folder)
        file_menu.addAction(open_image_action)
        
        add_images_action = QAction("添加图片", self)
        add_images_action.setShortcut("Ctrl+Shift+A")
        add_images_action.triggered.connect(self._add_images)
        file_menu.addAction(add_images_action)
        
        file_menu.addSeparator()
        
        export_menu = file_menu.addMenu("导出")
        
        export_yolo_action = QAction("YOLO格式", self)
        export_yolo_action.triggered.connect(lambda: self._export_annotations(ExportFormat.YOLO))
        export_menu.addAction(export_yolo_action)
        
        export_coco_action = QAction("COCO格式", self)
        export_coco_action.triggered.connect(lambda: self._export_annotations(ExportFormat.COCO))
        export_menu.addAction(export_coco_action)
        
        export_voc_action = QAction("VOC格式", self)
        export_voc_action.triggered.connect(lambda: self._export_annotations(ExportFormat.VOC))
        export_menu.addAction(export_voc_action)
        
        file_menu.addSeparator()
        
        settings_action = QAction("设置", self)
        settings_action.triggered.connect(self._open_settings)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 编辑菜单
        edit_menu = menubar.addMenu("编辑")
        
        undo_action = QAction("撤销", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self._undo)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("重做", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(self._redo)
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        copy_action = QAction("复制标注", self)
        copy_action.setShortcut("Ctrl+C")
        copy_action.triggered.connect(self._copy_annotation)
        edit_menu.addAction(copy_action)
        
        paste_action = QAction("粘贴标注", self)
        paste_action.setShortcut("Ctrl+V")
        paste_action.triggered.connect(self._paste_annotation)
        edit_menu.addAction(paste_action)
        
        delete_action = QAction("删除标注", self)
        delete_action.setShortcut("Delete")
        delete_action.triggered.connect(self._delete_selected_annotation)
        edit_menu.addAction(delete_action)
        
        edit_menu.addSeparator()
        
        clear_action = QAction("清空所有标注", self)
        clear_action.triggered.connect(self._clear_annotations)
        edit_menu.addAction(clear_action)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图")
        
        zoom_in_action = QAction("放大", self)
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(self.canvas.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction("缩小", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.canvas.zoom_out)
        view_menu.addAction(zoom_out_action)
        
        zoom_actual_action = QAction("实际大小", self)
        zoom_actual_action.setShortcut("Ctrl+0")
        zoom_actual_action.triggered.connect(self.canvas.zoom_actual)
        view_menu.addAction(zoom_actual_action)
        
        fit_window_action = QAction("适应窗口", self)
        fit_window_action.setShortcut("Ctrl+1")
        fit_window_action.triggered.connect(self.canvas.fit_to_view)
        view_menu.addAction(fit_window_action)
        
        view_menu.addSeparator()
        
        show_labels_action = QAction("显示标签", self)
        show_labels_action.setCheckable(True)
        show_labels_action.setChecked(self.show_labels)
        show_labels_action.toggled.connect(self._toggle_show_labels)
        view_menu.addAction(show_labels_action)
        
        show_grid_action = QAction("显示网格", self)
        show_grid_action.setCheckable(True)
        show_grid_action.setChecked(self.show_grid)
        show_grid_action.toggled.connect(self._toggle_show_grid)
        view_menu.addAction(show_grid_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu("工具")
        
        manage_classes_action = QAction("管理类别", self)
        manage_classes_action.triggered.connect(self._manage_classes)
        tools_menu.addAction(manage_classes_action)
        
        tools_menu.addSeparator()
        
        batch_rename_action = QAction("批量重命名图片", self)
        batch_rename_action.triggered.connect(self._batch_rename_images)
        tools_menu.addAction(batch_rename_action)
        
        batch_export_action = QAction("批量导出标注", self)
        batch_export_action.triggered.connect(self._batch_export_annotations)
        tools_menu.addAction(batch_export_action)

        rename_history_action = QAction("重命名历史", self)
        rename_history_action.triggered.connect(self._open_rename_history)
        tools_menu.addAction(rename_history_action)

        split_dataset_action = QAction("自动划分数据集", self)
        split_dataset_action.triggered.connect(self._split_dataset)
        tools_menu.addAction(split_dataset_action)

        train_model_action = QAction("一键训练模型", self)
        train_model_action.triggered.connect(self._start_training)
        tools_menu.addAction(train_model_action)

        # 智能标注子菜单（区分：使用模型 / 使用已有标注迁移）
        ai_menu = tools_menu.addMenu("智能标注")

        ai_curr_model = QAction("当前图片（使用模型）", self)
        ai_curr_model.triggered.connect(self._ai_annotate_current_with_model)
        ai_menu.addAction(ai_curr_model)

        ai_curr_labels = QAction("当前图片（从已有标注迁移）", self)
        ai_curr_labels.triggered.connect(self._ai_annotate_current_from_labels)
        ai_menu.addAction(ai_curr_labels)

        ai_all_model = QAction("所有图片（使用模型）", self)
        ai_all_model.triggered.connect(self._ai_annotate_all_with_model)
        ai_menu.addAction(ai_all_model)

        ai_all_labels = QAction("所有图片（从已有标注迁移）", self)
        ai_all_labels.triggered.connect(self._ai_annotate_all_from_labels)
        ai_menu.addAction(ai_all_labels)

        find_label_action = QAction("查找包含标签的图片", self)
        find_label_action.triggered.connect(self._find_images_by_label_prompt)
        tools_menu.addAction(find_label_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        
        help_action = QAction("使用说明", self)
        help_action.setShortcut("F1")
        help_action.triggered.connect(self._show_help)
        help_menu.addAction(help_action)
        
        about_action = QAction("关于", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _init_toolbars(self):
        """初始化工具栏"""
        # 主工具栏
        main_toolbar = self.addToolBar("主工具栏")
        main_toolbar.setMovable(False)
        main_toolbar.setIconSize(QSize(24, 24))
        
        # 添加工具栏动作
        main_toolbar.addAction(self.findChild(QAction, "新建项目"))
        main_toolbar.addAction(self.findChild(QAction, "打开项目"))
        main_toolbar.addAction(self.findChild(QAction, "保存项目"))
        main_toolbar.addSeparator()
        
        main_toolbar.addAction(self.findChild(QAction, "打开图片文件夹"))
        main_toolbar.addSeparator()
        
        main_toolbar.addAction(self.findChild(QAction, "复制标注"))
        main_toolbar.addAction(self.findChild(QAction, "粘贴标注"))
        main_toolbar.addAction(self.findChild(QAction, "删除标注"))
        main_toolbar.addSeparator()
        
        main_toolbar.addAction(self.findChild(QAction, "放大"))
        main_toolbar.addAction(self.findChild(QAction, "缩小"))
        main_toolbar.addAction(self.findChild(QAction, "适应窗口"))
        
        # 标注工具栏
        annotation_toolbar = self.addToolBar("标注工具栏")
        annotation_toolbar.setMovable(False)
        
        annotation_toolbar.addWidget(QLabel("标注工具: "))
        annotation_toolbar.addWidget(self.select_tool_btn)
        annotation_toolbar.addWidget(self.rectangle_tool_btn)
        annotation_toolbar.addWidget(self.rotated_tool_btn)
        annotation_toolbar.addWidget(self.polygon_tool_btn)

    def _init_shortcuts(self):
        """初始化全局快捷键，使用 ApplicationShortcut 上下文以避免焦点丢失导致不响应。"""
        # 使用 try/except 包裹以保证在不支持的平台或测试环境下不抛异常
        try:
            # 空格：下一张
            self._space_next_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
            self._space_next_shortcut.setContext(Qt.ApplicationShortcut)
            self._space_next_shortcut.activated.connect(self._next_image)

            # a：上一张
            self._a_prev_shortcut = QShortcut(QKeySequence(Qt.Key_A), self)
            self._a_prev_shortcut.setContext(Qt.ApplicationShortcut)
            self._a_prev_shortcut.activated.connect(self._prev_image)

            # f：下一张（与空格等价）
            self._f_next_shortcut = QShortcut(QKeySequence(Qt.Key_F), self)
            self._f_next_shortcut.setContext(Qt.ApplicationShortcut)
            self._f_next_shortcut.activated.connect(self._next_image)
        except Exception:
            pass
    
    def _init_statusbar(self):
        """初始化状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label, 1)
        
        # 鼠标位置标签
        self.mouse_position_label = QLabel("x: 0, y: 0")
        self.status_bar.addPermanentWidget(self.mouse_position_label)
        
        # 图像信息标签
        self.image_size_label = QLabel("大小: 0x0")
        self.status_bar.addPermanentWidget(self.image_size_label)
        
        # 缩放比例标签
        self.zoom_label = QLabel("缩放: 100%")
        self.status_bar.addPermanentWidget(self.zoom_label)
        
        # 连接画布的鼠标位置信号
        self.canvas.mouse_position_changed.connect(self._update_mouse_position)
    
    def _setup_window(self):
        """设置窗口属性"""
        # 设置窗口大小和位置
        self.resize(
            self.config.app_config.window_width,
            self.config.app_config.window_height
        )
        
        if self.config.app_config.window_maximized:
            self.showMaximized()
        
        # 设置窗口图标
        icon_path = Path(__file__).parent.parent / "resources" / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
    
    def _load_recent_projects(self):
        """加载最近项目"""
        recent_projects = self.config.get_recent_projects()
        
        # 更新文件菜单
        file_menu = self.menuBar().findChild(QMenu, "文件")
        if file_menu:
            # 查找最近项目子菜单
            for action in file_menu.actions():
                if action.text() == "最近项目":
                    file_menu.removeAction(action)
                    break
            
            if recent_projects:
                recent_menu = file_menu.addMenu("最近项目")
                for project_path in recent_projects:
                    action = QAction(os.path.basename(project_path), self)
                    action.setData(project_path)
                    action.triggered.connect(lambda checked, path=project_path: self._open_recent_project(path))
                    recent_menu.addAction(action)
    
    # ==================== 项目操作 ====================
    
    def _new_project(self):
        """新建项目"""
        if self.project_modified:
            reply = QMessageBox.question(
                self, "保存修改",
                "当前项目已修改，是否保存？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                if not self._save_project():
                    return
            elif reply == QMessageBox.Cancel:
                return
        
        # 重置项目状态
        self.project_modified = False
        self.project_name = "未命名项目"
        
        # 清空数据
        self.project_manager.new_project()
        self.annotation_manager.clear()
        
        # 更新UI
        self._update_project_ui()
        self._update_image_list()
        self._update_annotation_list()
        
        # 清空画布
        self.canvas.clear()
        
        # 更新状态
        self.status_label.setText("已创建新项目")
    
    def _open_project(self):
        """打开项目"""
        if self.project_modified:
            reply = QMessageBox.question(
                self, "保存修改",
                "当前项目已修改，是否保存？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                if not self._save_project():
                    return
            elif reply == QMessageBox.Cancel:
                return
        
        # 选择项目文件
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开项目",
            "", "项目文件 (*.json);;所有文件 (*.*)"
        )
        
        if file_path:
            self._load_project(file_path)
    
    def _open_recent_project(self, file_path):
        """打开最近项目"""
        if self.project_modified:
            reply = QMessageBox.question(
                self, "保存修改",
                "当前项目已修改，是否保存？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                if not self._save_project():
                    return
            elif reply == QMessageBox.Cancel:
                return
        
        if os.path.exists(file_path):
            self._load_project(file_path)
        else:
            QMessageBox.warning(self, "错误", f"项目文件不存在: {file_path}")
            # 从最近项目中移除
            self.config.get_recent_projects()
    
    def _load_project(self, file_path):
        """加载项目 — 快速响应 UI 并在后台执行耗时检查（例如是否有标签）"""
        try:
            if not self.project_manager.load_project(file_path):
                QMessageBox.warning(self, "错误", "无法加载项目文件")
                return

            # quick UI update
            self.project_name = self.project_manager.project_name
            self.project_modified = False
            try:
                self.config.add_recent_project(file_path)
            except Exception:
                pass

            self._update_project_ui()

            # Populate image list quickly without per-file IO
            try:
                self.image_list.clear()
                for i, image_path in enumerate(self.project_manager.image_files):
                    file_name = os.path.basename(image_path)
                    item = QListWidgetItem(file_name)
                    self.image_list.addItem(item)
                if hasattr(self, 'image_info_label'):
                    self.image_info_label.setText(f"图片: {self.project_manager.image_count}")
            except Exception:
                pass

            # Load classes immediately
            try:
                self.classes = self.project_manager.get_classes()
                self.class_list_widget.set_classes(self.classes)
                self._sync_canvas_class_colors()
            except Exception:
                self.classes = []

            # Background worker: check per-image annotation existence and then load first image
            try:
                class _ProjectLoadWorker(QObject):
                    progress = pyqtSignal(int, int)
                    annotation_found = pyqtSignal(int, bool)
                    finished = pyqtSignal()

                    def __init__(self, pm):
                        super().__init__()
                        self.pm = pm

                    def run(self):
                        files = list(self.pm.image_files)
                        total = len(files)
                        for idx, img in enumerate(files):
                            try:
                                has = self.pm.has_annotation(img)
                            except Exception:
                                has = False
                            self.annotation_found.emit(idx, has)
                            self.progress.emit(idx + 1, total)
                        self.finished.emit()

                # avoid starting another load if one is running
                if self._project_load_thread and self._project_load_thread.isRunning():
                    return

                worker = _ProjectLoadWorker(self.project_manager)
                thread = QThread(self)
                worker.moveToThread(thread)
                thread.started.connect(worker.run)
                worker.annotation_found.connect(self._on_worker_annotation_found)
                worker.progress.connect(self._on_worker_progress)
                worker.finished.connect(thread.quit)
                worker.finished.connect(worker.deleteLater)
                thread.finished.connect(thread.deleteLater)

                # when worker finishes, load first image on main thread
                def _on_finished_and_load():
                    try:
                        if self.project_manager.image_count > 0:
                            self._load_image(0)
                    except Exception:
                        pass
                    self.project_loaded.emit(file_path)

                worker.finished.connect(_on_finished_and_load)

                # store refs
                self._project_load_worker = worker
                self._project_load_thread = thread

                thread.start()
            except Exception:
                # fallback synchronous behavior
                try:
                    self._update_image_list()
                    if self.project_manager.image_count > 0:
                        self._load_image(0)
                    self.project_loaded.emit(file_path)
                except Exception:
                    pass

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载项目失败: {str(e)}")

    def _on_worker_annotation_found(self, idx: int, has: bool):
        """Worker signal handler: update image list item when annotation presence is known."""
        try:
            item = self.image_list.item(idx)
            if item is None:
                return
            text = item.text()
            if has:
                item.setForeground(QColor(0, 200, 0))
                if not text.endswith(' ✓'):
                    item.setText(f"{text} ✓")
            else:
                if text.endswith(' ✓'):
                    item.setText(text[:-2])
        except Exception:
            pass

    def _on_worker_progress(self, processed: int, total: int):
        try:
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"正在加载项目: {processed}/{total}")
        except Exception:
            pass
    
    def _save_project(self) -> bool:
        """保存项目"""
        try:
            if self.project_manager.save_project():
                self.project_modified = False
                self.project_name = self.project_manager.project_name
                
                # 更新UI
                self._update_project_ui()
                
                # 更新状态
                self.status_label.setText(f"项目已保存: {self.project_name}")
                
                # 发射信号
                self.project_saved.emit(self.project_manager.project_path)
                
                return True
            else:
                return False
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存项目失败: {str(e)}")
            return False
    
    def _save_project_as(self) -> bool:
        """另存项目"""
        try:
            if self.project_manager.save_project_as():
                self.project_modified = False
                self.project_name = self.project_manager.project_name
                
                # 更新配置中的最近项目
                self.config.add_recent_project(self.project_manager.project_path)
                
                # 更新UI
                self._update_project_ui()
                
                # 更新状态
                self.status_label.setText(f"项目已另存为: {self.project_name}")
                
                return True
            else:
                return False
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"另存项目失败: {str(e)}")
            return False
    
    def _auto_save(self):
        """自动保存"""
        if self.project_modified and self.project_manager.has_images():
            try:
                self.project_manager.auto_save()
                self.status_label.setText("已自动保存")
            except Exception as e:
                print(f"自动保存失败: {e}")
    
    # ==================== 图片操作 ====================
    def _open_image_folder(self):
        """打开图片文件夹"""
        folder = QFileDialog.getExistingDirectory(
            self, "选择图片文件夹", ""
        )
        
        if folder:
            # 获取文件夹中的图片文件
            image_files = FileUtils.get_image_files(folder)
            
            if image_files:
                # 添加到项目
                self.project_manager.add_image_files(image_files)
                self.project_modified = True
                
                # 更新UI
                self._update_image_list()
                
                # 加载第一张图片
                if self.project_manager.image_count > 0:
                    # 自动尝试从图片目录下加载类别
                    img_path = self.project_manager.get_image_path(0)
                    if img_path:
                        self._auto_load_classes(str(Path(img_path).parent))
                    self._load_image(0)
                
                # 更新状态
                self.status_label.setText(f"已加载 {len(image_files)} 张图片")
            else:
                QMessageBox.warning(self, "警告", "文件夹中没有找到图片文件")
    
    def _add_images(self):
        """添加图片"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片文件", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp);;所有文件 (*.*)"
        )
        
        if files:
            # 添加到项目
            self.project_manager.add_image_files(files)
            self.project_modified = True
            
            # 更新UI
            self._update_image_list()
            
            # 如果没有当前图片，加载第一张
            if self.current_image_index < 0 and self.project_manager.image_count > 0:
                self._load_image(0)
            
            # 更新状态
            self.status_label.setText(f"已添加 {len(files)} 张图片")
    
    def _load_image(self, index: int):
        """加载指定索引的图片"""
        if 0 <= index < self.project_manager.image_count:
            try:
                # 停止标签自动保存，防止与手动/切换保存冲突
                try:
                    if hasattr(self, 'label_autosave_timer'):
                        self.label_autosave_timer.stop()
                except Exception:
                    pass

                # 保存当前标注（保存上一张图片的标签）
                self._save_current_annotations()
                
                # 加载图片
                image_path = self.project_manager.get_image_path(index)
                image = FileUtils.load_image(image_path)
                
                if image is not None:
                    # 设置当前索引
                    self.current_image_index = index
                    self.current_image_path = image_path
                    
                    # 自动加载当前图片文件夹下的标签类别
                    self._auto_load_classes(str(Path(image_path).parent))
                    
                    # 设置到画布
                    self.canvas.set_image(image)
                    
                    # 加载标注
                    self._load_image_annotations(image_path)
                    
                    # 更新UI
                    self._update_image_selection()
                    self._update_image_info()
                    
                    # 发射信号
                    self.image_changed.emit(index)
                    
                    # 启动当前图片的标签自动保存（每1秒保存一次）
                    try:
                        if hasattr(self, 'label_autosave_timer'):
                            self.label_autosave_timer.start()
                    except Exception:
                        pass

                    # 更新状态
                    self.status_label.setText(f"图片 {index + 1}/{self.project_manager.image_count}")
                else:
                    QMessageBox.warning(self, "错误", f"无法加载图片: {image_path}")
                    
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载图片失败: {str(e)}")
    
    def _load_image_annotations(self, image_path: str):
        """加载图片标注"""
        # 获取图片对应的标注文件路径
        label_path = self.project_manager.get_label_path(image_path)
        
        # 加载标注
        if os.path.exists(label_path):
            image_size = self.canvas.get_image_size()
            self.current_annotations = FileUtils.load_yolo_annotations(
                label_path, image_size, self.classes
            )
        else:
            self.current_annotations = []
        
        # 设置到画布
        self.canvas.set_annotations(self.current_annotations)
        
        # 更新标注列表
        self._update_annotation_list()
    
    def _save_current_annotations(self):
        """保存当前图片的标注"""
        if self.current_image_path:
            try:
                # 获取标注文件路径
                label_path = self.project_manager.get_label_path(self.current_image_path)
                
                # 获取图片尺寸
                image_size = self.canvas.get_image_size()
                # Use canvas as the source of truth for latest annotations to avoid
                # timing/race conditions between canvas signals and UI actions.
                anns = getattr(self.canvas, 'annotations', None)
                if anns is None:
                    anns = self.current_annotations if self.current_annotations is not None else []

                # 保存标注（确保立即写入磁盘）
                FileUtils.save_yolo_annotations(
                    label_path, anns, image_size
                )

                # Keep UI state in sync
                self.current_annotations = list(anns)
                
                # 标记项目已修改
                self.project_modified = True
                
                # 更新图片列表中的标记
                self._update_image_list()
                
            except Exception as e:
                print(f"保存标注失败: {e}")
    
    def _prev_image(self):
        """上一张图片"""
        if self.current_image_index > 0:
            self._load_image(self.current_image_index - 1)
    
    def _next_image(self):
        """下一张图片"""
        if self.current_image_index < self.project_manager.image_count - 1:
            self._load_image(self.current_image_index + 1)
    
    def _on_image_selected(self):
        """图片选择事件"""
        selected_items = self.image_list.selectedItems()
        if selected_items:
            index = self.image_list.row(selected_items[0])
            if index != self.current_image_index:
                self._load_image(index)
    
    # ==================== 标注操作 ====================
    
    def _on_annotation_added(self, annotation: AnnotationItem):
        """标注添加事件"""
        try:
            self.annotation_manager.record_add(annotation)
        except Exception:
            pass

        # 将标注加入当前列表
        self.current_annotations.append(annotation)

        # 检查是否已有关联类别（例如来自粘贴或自动标注）
        if getattr(annotation, 'class_name', None):
            mapped = False
            for c in self.classes:
                if c.name.lower() == annotation.class_name.lower():
                    annotation.class_id = c.id
                    annotation.class_name = c.name
                    mapped = True
                    break

            if not mapped and not self.classes:
                # 自动创建第一个类别
                name_to_create = annotation.class_name.strip() if annotation.class_name.strip() else "未命名"
                new_id = len(self.classes)
                qc = self._generate_unique_color()
                new_cls = ClassItem(new_id, name_to_create, qc)
                self.classes.append(new_cls)
                try:
                    self.class_list_widget.set_classes(self.classes)
                    self._save_classes()
                    self._sync_canvas_class_colors()
                except Exception: pass
                annotation.class_id = new_cls.id
                annotation.class_name = new_cls.name

        # 如果没有标签信息，提示用户选择。如果是新画的框且取消选择，则删除该框。
        if not getattr(annotation, 'class_name', None):
            if not self._prompt_label_selection(annotation, is_new=True):
                return

        # 通知 UI 更新
        try:
            self.canvas.update()
            self.annotation_updated.emit(annotation)
        except Exception:
            pass

        self.annotation_added.emit(annotation)
        self.project_modified = True

    def _prompt_label_selection(self, annotation: AnnotationItem, is_new: bool = False) -> bool:
        """
        弹出标签选择对话框（支持点击已有标签或新画框完成时调用）。
        Args:
            annotation: 要编辑标签的标注对象
            is_new: 是否为刚创建的新标注（若是，则取消选择时会删除该标注）
        Returns:
            bool: 用户是否确认了选择
        """
        if not annotation:
            return False
            
        try:
            # 1. 如果当前项目还没有任何标签，提示输入第一个
            if not self.classes:
                from PyQt5.QtWidgets import QInputDialog
                name, ok = QInputDialog.getText(self, "输入标签名", "该项目暂无标签，请输入第一个标签名称:")
                if ok and name.strip():
                    new_id = len(self.classes)
                    qc = self._generate_unique_color()
                    new_cls = ClassItem(new_id, name.strip(), qc)
                    self.classes.append(new_cls)
                    try:
                        self.class_list_widget.set_classes(self.classes)
                        self._save_classes()
                        self._sync_canvas_class_colors()
                    except Exception: pass
                    annotation.class_id = new_id
                    annotation.class_name = name.strip()
                else:
                    if is_new:
                        self._delete_selected_annotation(annotation.id)
                    return False
            else:
                # 2. 弹出自定义选择对话框
                choices = [c.name for c in self.classes] + ["<新建标签>"]
                dialog = LabelSelectDialog(self, choices, "选择标签")
                if dialog.exec_() == QDialog.Accepted:
                    if hasattr(dialog, 'get_selected_result'):
                        action, value = dialog.get_selected_result()
                    else:
                        raw_item = dialog.get_selected_item()
                        if raw_item == "<新建标签>": action, value = "MANUAL_NEW", None
                        elif raw_item.startswith("新建: "): action, value = "NEW", raw_item[4:].strip()
                        else: action, value = "EXISTING", raw_item

                    name_to_create = None
                    if action == "MANUAL_NEW":
                        from PyQt5.QtWidgets import QInputDialog
                        text, ok = QInputDialog.getText(self, "新建标签", "请输入新标签名称:")
                        if ok and text:
                            name_to_create = text.strip()
                    elif action == "NEW":
                        name_to_create = value
                    else:
                        # 选中已有标签
                        names = [c.name for c in self.classes]
                        try:
                            idx = names.index(value)
                            annotation.class_id = self.classes[idx].id
                            annotation.class_name = self.classes[idx].name
                        except ValueError:
                            pass

                    # 3. 处理新建标签逻辑
                    if name_to_create:
                        exists_cls = next((c for c in self.classes if c.name.lower() == name_to_create.lower()), None)
                        if exists_cls:
                            annotation.class_id = exists_cls.id
                            annotation.class_name = exists_cls.name
                        else:
                            new_id = len(self.classes)
                            qc = self._generate_unique_color()
                            new_cls = ClassItem(new_id, name_to_create, qc)
                            self.classes.append(new_cls)
                            try:
                                self.class_list_widget.set_classes(self.classes)
                                self._save_classes()
                                self._sync_canvas_class_colors()
                            except Exception: pass
                            annotation.class_id = new_id
                            annotation.class_name = name_to_create
                            
                else:
                    # 用户取消
                    if is_new:
                        self._delete_selected_annotation(annotation.id)
                    return False

            # 4. 更新 UI 和状态
            self.canvas.update()
            self.annotation_updated.emit(annotation)
            self._update_annotation_list()
            self.project_modified = True
            return True
            
        except Exception as e:
            print(f"Error prompting label selection: {e}")
            return False
    
    def _on_annotation_updated(self, annotation: AnnotationItem):
        """标注更新事件"""
        # 如果之前保存了选中标注的快照，则认为此次为修改操作，记录修改前后的状态
        try:
            if self._last_selected_snapshot and self._last_selected_snapshot.id == annotation.id:
                self.annotation_manager.record_modify(self._last_selected_snapshot, annotation)
        except Exception:
            pass
        finally:
            # 清空快照
            self._last_selected_snapshot = None

        self.annotation_updated.emit(annotation)
        self.project_modified = True
    
    def _on_annotation_selected(self, annotation: AnnotationItem):
        """标注选中事件"""
        # 在修改开始前保存快照
        try:
            self._last_selected_snapshot = annotation.copy() if annotation else None
        except Exception:
            self._last_selected_snapshot = None

        self.selected_annotation = annotation
        self._update_annotation_selection()
    
    def _on_annotation_list_selected(self, annotation_id: str):
        """标注列表选择事件"""
        # 在标注列表中查找标注
        for annotation in self.current_annotations:
            if annotation.id == annotation_id:
                # 设置选中状态
                self.selected_annotation = annotation
                self.canvas.select_annotation(annotation)
                break
    
    def _copy_annotation(self):
        """复制标注 (支持多选)"""
        try:
            # 优先从画布获取所有选中的标注
            selected_anns = self.canvas.get_selected_annotations()
            
            if not selected_anns and self.selected_annotation:
                selected_anns = [self.selected_annotation]

            if selected_anns:
                self.annotation_manager.copy_annotation(selected_anns)
                self.status_label.setText(f"已复制 {len(selected_anns)} 个标注")
            else:
                self.status_label.setText("未选中标注")
        except Exception as e:
            print(f"Copy error: {e}")
    
    def _paste_annotation(self):
        """粘贴标注"""
        pasted = self.annotation_manager.paste_annotation()
        if not pasted:
            return

        # 检查并映射类别 (保持原逻辑)
        for ann in pasted:
            mapped = False
            if ann.class_name:
                for c in self.classes:
                    if c.name.lower() == ann.class_name.lower():
                        ann.class_id = c.id
                        ann.class_name = c.name
                        mapped = True
                        break

            if not mapped:
                if self.classes:
                    ann.class_id = self.classes[0].id
                    ann.class_name = self.classes[0].name
                else:
                    name_to_create = ann.class_name if ann.class_name else "未命名"
                    new_id = len(self.classes)
                    qc = self._generate_unique_color()
                    new_cls = ClassItem(new_id, name_to_create, qc)
                    self.classes.append(new_cls)
                    try:
                        self.class_list_widget.set_classes(self.classes)
                        self._save_classes()
                        self._sync_canvas_class_colors()
                    except Exception:
                        pass
                    ann.class_id = new_cls.id
                    ann.class_name = new_cls.name

        # 将粘贴项加入列表并通知画布
        self.current_annotations.extend(pasted)
        for ann in pasted:
            try:
                self.canvas.add_annotation(ann)
            except Exception:
                self.annotation_added.emit(ann)

        self._update_annotation_list()
        self.project_modified = True
        self.status_label.setText(f"已粘贴 {len(pasted)} 个标注")
    
    def _delete_selected_annotation(self, annotation_id: str = None):
        """删除选中的标注项 (支持多选)"""
        # 1. 确定要删除的标注列表
        targets = []
        if annotation_id:
            # 指定删除某一个 (例如从列表右键)
            targets = [a for a in self.current_annotations if a.id == annotation_id]
        else:
            # 删除所有选中的
            targets = self.canvas.get_selected_annotations()
            if not targets and self.selected_annotation:
                targets = [self.selected_annotation]
        
        if not targets:
            return

        # 2. 记录到撤销管理器 (批量)
        try:
            # 记录索引以便撤销
            indices = []
            for t in targets:
                try:
                    idx = self.current_annotations.index(t)
                    indices.append(idx)
                except ValueError:
                    indices.append(-1)
            self.annotation_manager.record_remove(targets, indices=indices)
        except Exception as e:
            print(f"Record remove error: {e}")

        # 3. 同步删除
        target_ids = [t.id for t in targets]
        self.current_annotations = [a for a in self.current_annotations if a.id not in target_ids]
        
        try:
            self.canvas.annotations = [a for a in self.canvas.annotations if a.id not in target_ids]
            # 清理单选指针
            if self.selected_annotation and self.selected_annotation.id in target_ids:
                self.selected_annotation = None
            self.canvas.update()
        except Exception as e:
            print(f"Error syncing deletion to canvas: {e}")

        # 4. 发射信号并标记修改
        for id_ in target_ids:
            self.annotation_removed.emit(id_)
        
        self.project_modified = True
        self._update_annotation_list()
        self.status_label.setText(f"已删除 {len(targets)} 个标注")
    
    def _clear_annotations(self):
        """清空所有标注"""
        if not self.current_annotations:
            return
        
        reply = QMessageBox.question(
            self, "确认清空",
            "确定要清空当前图片的所有标注吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.current_annotations.clear()
            self.canvas.clear_annotations()
            self._update_annotation_list()
            self.project_modified = True
            self.status_label.setText("已清空所有标注")
    
    def _on_canvas_mode_changed(self, mode):
        """画布模式改变事件"""
        self._set_annotation_mode(mode)

    def _set_annotation_mode(self, mode: AnnotationMode):
        """设置标注模式"""
        self.canvas.set_annotation_mode(mode)
        
        # 更新按钮状态
        self.select_tool_btn.setChecked(mode == AnnotationMode.NONE)
        self.rectangle_tool_btn.setChecked(mode == AnnotationMode.HORIZONTAL)
        self.rotated_tool_btn.setChecked(mode == AnnotationMode.ROTATED)
        self.polygon_tool_btn.setChecked(mode == AnnotationMode.POLYGON)
        
        # 更新状态
        mode_names = {
            AnnotationMode.NONE: "选择模式",
            AnnotationMode.HORIZONTAL: "矩形标注模式",
            AnnotationMode.ROTATED: "旋转框标注模式",
            AnnotationMode.POLYGON: "多边形标注模式"
        }
        self.status_label.setText(f"切换到: {mode_names.get(mode, '未知模式')}")
    
    def _generate_unique_color(self):
        """生成与其他类别区分明显的颜色"""
        from PyQt5.QtGui import QColor
        existing_colors = [c.color for c in self.classes]
        
        for _ in range(50):
            r = random.randint(50, 255)
            g = random.randint(50, 255)
            b = random.randint(50, 255)
            
            # 简单的距离检查
            too_close = False
            for ec in existing_colors:
                if abs(ec.red() - r) + abs(ec.green() - g) + abs(ec.blue() - b) < 100:
                    too_close = True
                    break
            
            if not too_close:
                return QColor(r, g, b)
                
        return QColor(random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
    
    # ==================== 类别操作 ====================
    
    def _add_class(self):
        """添加类别"""
        dialog = ClassEditDialog(self)
        if dialog.exec_() == ClassEditDialog.Accepted:
            class_name = dialog.get_class_name()
            color = dialog.get_color()
            
            if class_name:
                # 创建新类别
                class_id = len(self.classes)
                try:
                    from PyQt5.QtGui import QColor
                    qc = QColor(color)
                    if not qc.isValid():
                        qc = QColor(0, 200, 0)
                except Exception:
                    from PyQt5.QtGui import QColor
                    qc = QColor(0, 200, 0)

                new_class = ClassItem(class_id, class_name, qc)
                self.classes.append(new_class)
                
                # 更新UI
                self.class_list_widget.set_classes(self.classes)
                # 同步颜色
                self._sync_canvas_class_colors()
                
                # 保存类别文件
                self._save_classes()
                
                # 更新状态
                self.status_label.setText(f"添加类别: {class_name}")
    
    def _edit_class(self, class_item=None):
        """编辑类别"""
        # 支持从信号传入或获取选中
        selected_class = class_item if isinstance(class_item, ClassItem) else self.class_list_widget.get_selected_class()
        
        if selected_class:
            dialog = ClassEditDialog(self)
            dialog.set_class(selected_class)
            
            if dialog.exec_() == ClassEditDialog.Accepted:
                class_name = dialog.get_class_name()
                color = dialog.get_color()
                
                if class_name:
                    # 更新类别
                    selected_class.name = class_name
                    try:
                        from PyQt5.QtGui import QColor
                        qc = QColor(color)
                        if not qc.isValid():
                            qc = QColor(0, 200, 0)
                    except Exception:
                        from PyQt5.QtGui import QColor
                        qc = QColor(0, 200, 0)
                    selected_class.color = qc

                    # 更新UI
                    self.class_list_widget.update_class(selected_class)

                    # 更新所有使用此类别的当前标注
                    if self.current_annotations:
                        updated = False
                        for ann in self.current_annotations:
                            if ann.class_id == selected_class.id:
                                ann.class_name = class_name
                                updated = True
                        
                        if updated:
                            # 刷新列表和画布
                            self._update_annotation_list()
                            self.canvas.update()
                            self.project_modified = True

                    # 保存类别文件
                    self._save_classes()

                    # 更新状态
                    self.status_label.setText(f"更新类别: {class_name}")
                    # 同步颜色到画布
                    self._sync_canvas_class_colors()
                    self.canvas.update()
    
    def _delete_class(self):
        """删除类别"""
        selected_class = self.class_list_widget.get_selected_class()
        if selected_class:
            reply = QMessageBox.question(
                self, "确认删除",
                f"确定要删除类别 '{selected_class.name}' 吗？\n\n注意：这将同时删除所有使用该类别的标注！",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # 删除类别
                class_id = selected_class.id
                self.classes.pop(class_id)
                
                # 重新编号
                for i, cls in enumerate(self.classes):
                    cls.id = i
                
                # 更新当前标注中的类别ID
                for annotation in self.current_annotations:
                    if annotation.class_id == class_id:
                        annotation.class_id = 0
                    elif annotation.class_id > class_id:
                        annotation.class_id -= 1
                
                # 更新UI
                self.class_list_widget.set_classes(self.classes)
                
                # 更新画布
                self.canvas.set_annotations(self.current_annotations)
                
                # 保存类别文件
                self._save_classes()
                
                # 更新状态
                self.status_label.setText(f"删除类别: {selected_class.name}")
    
    def _import_classes(self):
        """导入类别"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "导入类别", "",
            "文本文件 (*.txt);;JSON文件 (*.json);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                # 导入类别
                imported_classes = FileUtils.import_classes(file_path)
                
                if imported_classes:
                    # 合并类别
                    start_id = len(self.classes)
                    for i, cls in enumerate(imported_classes):
                        cls.id = start_id + i
                        self.classes.append(cls)
                    
                    # 更新UI
                    self.class_list_widget.set_classes(self.classes)
                    
                    # 保存类别文件
                    self._save_classes()
                    
                    # 更新状态
                    self.status_label.setText(f"导入了 {len(imported_classes)} 个类别")
                else:
                    QMessageBox.warning(self, "警告", "没有找到可导入的类别")
                    
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导入类别失败: {str(e)}")
    
    def _manage_classes(self):
        """管理类别"""
        dialog = ClassManagerDialog(self.classes, self)
        if dialog.exec_() == ClassManagerDialog.Accepted:
            # 获取更新后的类别
            self.classes = dialog.get_classes()
            
            # 更新UI
            self.class_list_widget.set_classes(self.classes)
            
            # 保存类别文件
            self._save_classes()

            # 同步到画布
            self._sync_canvas_class_colors()
            
            # 更新状态
            self.status_label.setText("类别已更新")
    
    def _on_class_selected(self, class_obj):
        """类别选择事件，接受 `ClassItem` 或 索引。"""
        try:
            # 如果传入的是索引
            if isinstance(class_obj, int):
                idx = class_obj
            else:
                # 如果是 ClassItem，查找对应索引
                idx = None
                for i, c in enumerate(self.classes):
                    if hasattr(class_obj, 'id') and c.id == class_obj.id:
                        idx = i
                        break

            if idx is not None and 0 <= idx < len(self.classes):
                self.current_class_index = idx
                self.canvas.set_current_class(self.classes[idx])
                self.status_label.setText(f"当前类别: {self.classes[idx].name}")
                # 如果有选中的标注，把它的类别改为当前类别
                try:
                    if self.selected_annotation:
                        cls = self.classes[idx]
                        self.selected_annotation.class_id = cls.id
                        self.selected_annotation.class_name = cls.name
                        # 同步颜色映射并刷新画布
                        self._sync_canvas_class_colors()
                        try:
                            self.canvas.update()
                        except Exception:
                            pass
                        # 记录修改并发出更新
                        try:
                            self.annotation_manager.record_modify(self.selected_annotation.copy(), self.selected_annotation)
                        except Exception:
                            pass
                        try:
                            self.annotation_updated.emit(self.selected_annotation)
                        except Exception:
                            pass
                        self.project_modified = True
                except Exception:
                    pass
        except Exception:
            pass
    
    def _save_classes(self):
        """保存类别文件"""
        try:
            # 确保类别名称干净
            for cls in self.classes:
                cls.name = str(cls.name).strip()

            # 1. 保存到工作空间根目录 (classes.txt)
            workspace_classes = self.config.get_workspace_path() / "classes.txt"
            workspace_classes.parent.mkdir(parents=True, exist_ok=True)
            FileUtils.save_classes(str(workspace_classes), self.classes)
            
            # 2. 同步保存到 labels 目录 (classes.txt)
            labels_classes = self.config.get_labels_dir() / "classes.txt"
            labels_classes.parent.mkdir(parents=True, exist_ok=True)
            FileUtils.save_classes(str(labels_classes), self.classes)
            
            # 3. 同步保存到当前图片所在的文件夹 (classes.txt)
            if self.current_image_path:
                img_dir = Path(self.current_image_path).parent
                if img_dir.exists():
                    img_dir_classes = img_dir / "classes.txt"
                    FileUtils.save_classes(str(img_dir_classes), self.classes)

            self.project_modified = True
            self.status_label.setText("类别文件已同步保存")
        except Exception as e:
            print(f"保存类别失败: {e}")
            self.status_label.setText(f"保存类别失败: {e}")

    def _auto_load_classes(self, directory: str):
        """尝试从指定目录自动加载类别"""
        if not directory or not os.path.exists(directory):
            return
            
        classes_txt = Path(directory) / "classes.txt"
        classes_json = Path(directory) / "classes.json"
        
        if classes_txt.exists() or classes_json.exists():
            try:
                new_classes = self.project_manager.get_classes(directory)
                if new_classes:
                    # 检查是否真的有变化（不仅仅是对象地址，而是内容）
                    current_names = [c.name for c in self.classes]
                    new_names = [c.name for c in new_classes]
                    
                    if current_names != new_names:
                        self.classes = new_classes
                        # 更新所有引用
                        self.class_list_widget.set_classes(self.classes)
                        self._sync_canvas_class_colors()
                        self.status_label.setText(f"已自动从目录加载 {len(self.classes)} 个类别")
            except Exception as e:
                print(f"自动加载类别失败: {e}")
    
    # ==================== 导出操作 ====================
    
    def _export_annotations(self, format_type: ExportFormat):
        """导出标注"""
        if not self.project_manager.has_images():
            QMessageBox.warning(self, "警告", "没有图片可导出")
            return
        
        dialog = ExportDialog(format_type, self.config, self)
        if dialog.exec_() == ExportDialog.Accepted:
            export_dir = dialog.get_export_dir()
            options = dialog.get_options()
            
            if export_dir:
                try:
                    # 显示进度对话框
                    progress_dialog = QProgressDialog("正在导出...", "取消", 0, 100, self)
                    progress_dialog.setWindowTitle("导出进度")
                    progress_dialog.setWindowModality(Qt.WindowModal)
                    progress_dialog.show()
                    
                    # 导出标注
                    if format_type == ExportFormat.YOLO:
                        ExportUtils.export_yolo_format(
                            export_dir, 
                            self.project_manager.image_files,
                            self.config.get_labels_dir(),
                            self.classes
                        )
                    elif format_type == ExportFormat.COCO:
                        ExportUtils.export_coco_format(
                            export_dir,
                            self.project_manager.image_files,
                            self.config.get_labels_dir(),
                            self.classes
                        )
                    elif format_type == ExportFormat.VOC:
                        ExportUtils.export_voc_format(
                            export_dir,
                            self.project_manager.image_files,
                            self.config.get_labels_dir(),
                            self.classes
                        )
                    
                    progress_dialog.close()
                    
                    # 创建数据集配置文件
                    if options.get('create_config', True):
                        class_names = [cls.name for cls in self.classes]
                        self.config.create_dataset_config(export_dir, class_names)
                    
                    QMessageBox.information(
                        self, "导出完成",
                        f"标注已成功导出到:\n{export_dir}"
                    )
                    
                except Exception as e:
                    progress_dialog.close()
                    QMessageBox.critical(self, "导出错误", f"导出失败: {str(e)}")
    
    def _batch_export_annotations(self):
        """批量导出标注"""
        self._export_annotations(ExportFormat.YOLO)
    
    def _create_dataset_config(self):
        """创建数据集配置文件"""
        if not self.classes:
            QMessageBox.warning(self, "警告", "请先添加类别")
            return
        
        export_dir = QFileDialog.getExistingDirectory(
            self, "选择导出目录", ""
        )
        
        if export_dir:
            try:
                class_names = [cls.name for cls in self.classes]
                config_path = self.config.create_dataset_config(export_dir, class_names)
                
                QMessageBox.information(
                    self, "配置文件已创建",
                    f"数据集配置文件已创建:\n{config_path}"
                )
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"创建配置文件失败: {str(e)}")
    
    def _split_dataset(self):
        """自动划分数据集"""
        if not self.project_manager.has_images():
            QMessageBox.warning(self, "警告", "没有图片可划分")
            return
            
        dialog = DatasetSplitDialog(self.config, self)
        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_data()
            export_dir = data["export_dir"]
            ratios = data["ratios"]
            
            try:
                # 过滤出已打标的图片，并构造 (图片路径, 标签路径) 元组
                labeled_pairs = []
                for img in self.project_manager.image_files:
                    lbl = self.project_manager.get_label_path(img)
                    if lbl and lbl.exists():
                        labeled_pairs.append((img, str(lbl)))
                
                if not labeled_pairs:
                    QMessageBox.warning(self, "警告", "当前项目中没有已标注的图片，无法划分数据集")
                    return

                # 显示进度对话框
                progress_dialog = QProgressDialog("正在划分数据集...", "取消", 0, 100, self)
                progress_dialog.setWindowTitle("处理进度")
                progress_dialog.setWindowModality(Qt.WindowModal)
                progress_dialog.show()
                
                def update_progress(idx, total):
                    progress_dialog.setValue(int(idx / total * 100))
                    QApplication.processEvents()
                    if progress_dialog.wasCanceled():
                        raise InterruptedError("用户取消了划分操作")
                
                # 执行划分
                ExportUtils.split_dataset(
                    export_dir,
                    labeled_pairs,
                    self.classes,
                    ratios,
                    progress_callback=update_progress
                )
                
                # 创建数据集 YAML 配置文件
                class_names = [cls.name for cls in self.classes]
                # 这里复用 create_dataset_config，但需要传具体的 split_info 吗？
                # 目前 Config.create_dataset_config 默认逻辑是处理单一路径，
                # 我们划分后变成了 images/train 等，需要改进 Config.create_dataset_config 或者手动补充。
                
                # 方案：调用 Config.create_dataset_config 并传入自定义 split 信息
                # 让 yaml 指向 images/train 和 images/val
                split_info = {"train": [], "val": [], "test": []} # 仅占位以触发 yaml 结构
                self.config.create_dataset_config(export_dir, class_names, split_info=split_info)
                
                progress_dialog.close()
                QMessageBox.information(
                    self, "划分完成",
                    f"数据集已成功划分并保存到:\n{export_dir}"
                )
                
            except InterruptedError as e:
                QMessageBox.information(self, "提示", str(e))
            except Exception as e:
                if 'progress_dialog' in locals():
                    progress_dialog.close()
                QMessageBox.critical(self, "错误", f"划分数据集失败: {str(e)}")
    
    def _start_training(self):
        """开始一键训练模型"""
        dialog = TrainDialog(self.config, self)
        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_params()
            
            # 校验 YAML 存在
            if not os.path.exists(params["data"]):
                QMessageBox.critical(self, "错误", f"找不到数据集配置文件: {params['data']}")
                return
            
            # 保存这些参数到配置中以便下次使用
            self.config.app_config.train_model_type = params["model"]
            self.config.app_config.train_epochs = params["epochs"]
            self.config.app_config.train_batch = params["batch"]
            self.config.app_config.train_imgsz = params["imgsz"]
            self.config.app_config.train_device = params["device"]
            self.config.save_config()
            
            # 创建日志窗口
            log_dialog = TrainingLogDialog(self)
            
            # 创建并启动 Worker
            worker = TrainingWorker(
                data_yaml=params["data"],
                model_type=params["model"],
                epochs=params["epochs"],
                imgsz=params["imgsz"],
                batch=params["batch"],
                device=params["device"]
            )
            
            # 连接信号
            worker.log_signal.connect(log_dialog.append_log)
            worker.finished_signal.connect(log_dialog.set_finished)
            
            # 记录 worker 防止被销毁
            self._training_worker = worker
            
            # 启动
            log_dialog.show() # 非模态显示
            worker.start()
            
            # 提示
            self.status_label.setText("模型训练已启动")
    
    # ==================== 批量操作 ====================
    
    def _batch_rename_images(self):
        """批量重命名图片"""
        if not self.project_manager.has_images():
            QMessageBox.warning(self, "警告", "没有图片可重命名")
            return
        
        from ui.dialogs.batch_dialogs import BatchRenameDialog
        
        dialog = BatchRenameDialog(self.project_manager.image_files, self)
        if dialog.exec_() == BatchRenameDialog.Accepted:
            # 执行物理重命名，并生成可撤销的备份记录
            new_files = dialog.get_new_files()
            if not new_files:
                return

            try:
                record_path = self.project_manager.rename_images_with_backup(new_files)

            except Exception as e:
                QMessageBox.critical(self, "错误", f"批量重命名失败: {str(e)}")
                return

            # 更新 UI 状态
            self.project_modified = True
            self._update_image_list()
            if self.current_image_index >= 0:
                try:
                    self.current_image_path = self.project_manager.get_image_path(self.current_image_index)
                except Exception:
                    self.current_image_path = None

            # 提示并提供撤销选项
            reply = QMessageBox.question(
                self, "重命名完成",
                f"已重命名 {len(new_files)} 张图片。是否立即撤销？",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                try:
                    self.project_manager.undo_last_rename(record_path)
                    self._update_image_list()
                    # 恢复当前图片路径
                    if self.current_image_index >= 0:
                        try:
                            self.current_image_path = self.project_manager.get_image_path(self.current_image_index)
                        except Exception:
                            self.current_image_path = None
                    QMessageBox.information(self, "已撤销", "已撤销最近一次重命名操作")
                except Exception as e:
                    QMessageBox.critical(self, "撤销失败", f"撤销失败: {e}")
            else:
                QMessageBox.information(self, "完成", f"已重命名 {len(new_files)} 张图片")
    
    # ==================== 设置操作 ====================
    
    def _open_settings(self):
        """打开设置对话框"""
        dialog = SettingsDialog(self.config, self)
        if dialog.exec_() == SettingsDialog.Accepted:
            # 保存配置
            self.config.save_config()
            
            # 更新UI
            self._apply_settings()
            
            # 更新状态
            self.status_label.setText("设置已保存")

    def _open_rename_history(self):
        """打开重命名历史对话框，允许撤销此前的重命名操作"""
        try:
            from ui.dialogs.rename_history_dialog import RenameHistoryDialog

            dialog = RenameHistoryDialog(self.config, self)
            if dialog.exec_() == RenameHistoryDialog.Accepted:
                record = dialog.get_selected_record()
                save_after = dialog.should_save_project()
                if record:
                    try:
                        self.project_manager.undo_last_rename(record)
                        self._update_image_list()
                        # 恢复当前图片路径
                        if self.current_image_index >= 0:
                            try:
                                self.current_image_path = self.project_manager.get_image_path(self.current_image_index)
                            except Exception:
                                self.current_image_path = None

                        if save_after:
                            try:
                                self._save_project()
                            except Exception:
                                pass

                        QMessageBox.information(self, "撤销完成", "已撤销所选重命名记录")
                    except Exception as e:
                        QMessageBox.critical(self, "撤销失败", f"撤销失败: {e}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开重命名历史失败: {e}")
    
    def _apply_settings(self):
        """应用设置"""
        # 更新自动保存计时器
        if self.config.app_config.auto_save:
            interval = self.config.app_config.auto_save_interval * 60 * 1000
            self.auto_save_timer.start(interval)
        else:
            self.auto_save_timer.stop()
        
        # 更新视图设置
        self.show_labels = self.config.app_config.show_labels
        self.show_confidence = self.config.app_config.show_confidence
        self.show_grid = self.config.app_config.show_grid
        
        # 应用到画布
        self.canvas.set_show_labels(self.show_labels)
        self.canvas.set_show_grid(self.show_grid)
        self.canvas.set_grid_size(self.config.app_config.grid_size)
        
        # 更新复选框
        self.show_labels_cb.setChecked(self.show_labels)
        self.show_confidence_cb.setChecked(self.show_confidence)
        self.show_grid_cb.setChecked(self.show_grid)
    
    # ==================== 视图操作 ====================
    
    def _toggle_show_labels(self, checked: bool):
        """切换显示标签"""
        self.show_labels = checked
        self.canvas.set_show_labels(checked)
        self.config.app_config.show_labels = checked
        self.config.save_config()
    
    def _toggle_show_confidence(self, checked: bool):
        """切换显示置信度"""
        self.show_confidence = checked
        self.canvas.set_show_confidence(checked)
        self.config.app_config.show_confidence = checked
        self.config.save_config()
    
    def _toggle_show_grid(self, checked: bool):
        """切换显示网格"""
        self.show_grid = checked
        self.canvas.set_show_grid(checked)
        self.config.app_config.show_grid = checked
        self.config.save_config()
    
    def _toggle_show_statistics(self, checked: bool):
        """切换显示统计"""
        self.show_statistics = checked
        self.canvas.set_show_statistics(checked)
    
    def _update_mouse_position(self, x: int, y: int):
        """更新鼠标位置"""
        self.mouse_position_label.setText(f"x: {x}, y: {y}")
    
    def _update_image_info(self):
        """更新图片信息"""
        if self.canvas.has_image():
            size = self.canvas.get_image_size()
            self.image_size_label.setText(f"大小: {size[0]}x{size[1]}")
            
            zoom = self.canvas.get_zoom_level()
            self.zoom_label.setText(f"缩放: {zoom:.0f}%")
    
    # ==================== 帮助操作 ====================
    
    def _show_help(self):
        """显示帮助"""
        help_text = """
        YOLO-OBB标注工具使用说明
        
        基本操作:
        1. 新建或打开项目
        2. 添加图片或打开图片文件夹
        3. 添加和管理类别
        4. 使用标注工具进行标注
        5. 保存项目
        
        快捷键:
        S - 选择工具
        R - 矩形标注工具
        O - 旋转框标注工具
        P - 多边形标注工具
        
        Ctrl+S - 保存项目
        Ctrl+O - 打开项目
        Ctrl+Shift+O - 打开图片文件夹
        
        Ctrl+C - 复制标注
        Ctrl+V - 粘贴标注
        Delete - 删除选中标注
        
        Ctrl++ - 放大
        Ctrl+- - 缩小
        Ctrl+0 - 实际大小
        Ctrl+1 - 适应窗口
        
        更多帮助请查看文档。
        """
        
        QMessageBox.information(self, "使用说明", help_text)
    
    def _show_about(self):
        """显示关于信息"""
        about_text = f"""
        YOLO-OBB标注工具
        
        版本: 2.0.0
        开发者: sycamore
        
        功能特性:
        - 支持多种标注类型（矩形、旋转框、多边形）
        - 支持YOLO、COCO、VOC格式导出
        - 智能标注辅助
        - 批量处理功能
        - 自动保存和备份
        
        项目目录: {self.config.get_workspace_path()}
        配置文件: {self.config.config_file}
        
        © 2026 保留所有权利
        """
        
        QMessageBox.about(self, "关于", about_text)
    
    def _show_welcome_message(self):
        """显示欢迎消息"""
        if not self.project_manager.has_images():
            welcome_text = """
            欢迎使用YOLO-OBB标注工具！
            
            快速开始:
            1. 点击"打开图片文件夹"或"添加图片"加载图片
            2. 在右侧面板中添加类别
            3. 选择标注工具开始标注
            4. 保存项目以保留进度
            
            提示: 使用快捷键可以提高工作效率。
            """
            
            QMessageBox.information(self, "欢迎", welcome_text)
        # 智能标注的具体操作在类方法中实现
    
    # ==================== 撤销/重做 ====================
    
    def _undo(self):
        """撤销操作"""
        if self.annotation_manager.can_undo():
            action = self.annotation_manager.undo()
            if action:
                self._apply_action(action, is_undo=True)
                self._update_annotation_list()
                self.status_label.setText("已撤销操作")
    
    def _redo(self):
        """重做操作"""
        if self.annotation_manager.can_redo():
            action = self.annotation_manager.redo()
            if action:
                self._apply_action(action, is_undo=False)
                self._update_annotation_list()
                self.status_label.setText("已重做操作")

    def _apply_action(self, action: dict, is_undo: bool = False):
        """根据 manager 返回的 action 描述在 UI 上执行对应的修改。

        action 格式示例:
            {'type': 'remove', 'annotation': AnnotationItem}
            {'type': 'add', 'annotation': AnnotationItem, 'index': Optional[int]}
            {'type': 'modify', 'annotation': AnnotationItem}
        """
        try:
            t = action.get('type')
            ann: AnnotationItem = action.get('annotation')

            if t == 'remove':
                # 从当前列表移除具有相同 id 的标注
                if not ann:
                    return
                self.current_annotations = [a for a in self.current_annotations if a.id != ann.id]
                try:
                    self.canvas.remove_annotation(ann)
                except Exception:
                    pass
                self.annotation_removed.emit(ann.id)

            elif t == 'add':
                # 在指定位置插入或追加
                idx = action.get('index')
                if idx is None or idx < 0 or idx > len(self.current_annotations):
                    self.current_annotations.append(ann)
                else:
                    self.current_annotations.insert(idx, ann)

                try:
                    self.canvas.add_annotation(ann)
                except Exception:
                    self.annotation_added.emit(ann)

            elif t == 'modify':
                # 将目标标注恢复到 ann 的状态（ann 是要恢复的状态）
                if not ann:
                    return
                for i, a in enumerate(self.current_annotations):
                    if a.id == ann.id:
                        # 替换对象的主要属性
                        self.current_annotations[i] = ann
                        break
                try:
                    # 通知画布更新
                    self.annotation_updated.emit(ann)
                except Exception:
                    pass

        except Exception:
            pass

    # ==================== 智能标注实现（模型/无模型两种模式） ====================
    def _ai_annotate_current_with_model(self):
        """使用外部模型为当前图片生成标注（若未配置模型则提示错误）。"""
        current_path = self.project_manager.get_current_image_path()
        if not current_path:
            QMessageBox.warning(self, "警告", "当前没有打开的图片")
            return

        try:
            from utils.annotation_utils import auto_annotate_image_with_model
            anns = auto_annotate_image_with_model(self.config, current_path)
        except Exception as e:
            QMessageBox.critical(self, "AI 标注失败", str(e))
            return

        if not anns:
            QMessageBox.information(self, "AI 标注", "未检测到任何目标")
            return

        try:
            import cv2
            img = cv2.imread(current_path)
            h, w = img.shape[:2]
        except Exception:
            w, h = 0, 0

        for ann in anns:
            try:
                self.annotation_manager.record_add(ann)
            except Exception:
                pass
            self.current_annotations.append(ann)
            try:
                self.canvas.add_annotation(ann)
            except Exception:
                self.annotation_added.emit(ann)

        # 写标签文件
        try:
            labels_dir = self.config.get_labels_dir()
            labels_dir.mkdir(parents=True, exist_ok=True)
            image_stem = os.path.splitext(os.path.basename(current_path))[0]
            label_path = labels_dir / f"{image_stem}.txt"
            lines = []
            for ann in anns:
                try:
                    lines.append(ann.to_yolo_format(w, h))
                except Exception:
                    pass
            label_path.write_text("\n".join(lines), encoding='utf-8')
        except Exception:
            pass

        self._update_annotation_list()
        self.project_modified = True
        QMessageBox.information(self, "AI 标注", f"已为当前图片生成 {len(anns)} 个标注")

    def _ai_annotate_current_from_labels(self):
        """基于已有标注迁移为当前图片生成标注（不依赖模型）。"""
        current_path = self.project_manager.get_current_image_path()
        if not current_path:
            QMessageBox.warning(self, "警告", "当前没有打开的图片")
            return

        try:
            from utils.annotation_utils import auto_annotate_image_from_labels
            anns, src, score = auto_annotate_image_from_labels(self.config, current_path)
        except Exception as e:
            QMessageBox.critical(self, "标注迁移失败", str(e))
            return

        if not anns:
            QMessageBox.information(self, "标注迁移", "未能从已有标注迁移出结果")
            return

        try:
            import cv2
            img = cv2.imread(current_path)
            h, w = img.shape[:2]
        except Exception:
            w, h = 0, 0

        for ann in anns:
            try:
                self.annotation_manager.record_add(ann)
            except Exception:
                pass
            self.current_annotations.append(ann)
            try:
                self.canvas.add_annotation(ann)
            except Exception:
                self.annotation_added.emit(ann)

        # 写标签文件
        try:
            labels_dir = self.config.get_labels_dir()
            labels_dir.mkdir(parents=True, exist_ok=True)
            image_stem = os.path.splitext(os.path.basename(current_path))[0]
            label_path = labels_dir / f"{image_stem}.txt"
            lines = []
            for ann in anns:
                try:
                    lines.append(ann.to_yolo_format(w, h))
                except Exception:
                    pass
            label_path.write_text("\n".join(lines), encoding='utf-8')
        except Exception:
            pass

        self._update_annotation_list()
        self.project_modified = True
        msg = f"已为当前图片生成 {len(anns)} 个标注（迁移）"
        if src:
            msg += f"\n来源图像: {os.path.basename(src)}\n匹配置信度: {score:.2f}"
        QMessageBox.information(self, "标注迁移", msg)

    def _ai_annotate_all_with_model(self):
        """对项目中的所有图片使用外部模型进行标注并保存标签文件。"""
        if not self.project_manager.image_files:
            QMessageBox.warning(self, "警告", "当前项目没有图片")
            return

        reply = QMessageBox.question(self, "确认", "将对所有图片使用模型进行 AI 标注并覆盖/创建标签文件，是否继续？")
        if reply != QMessageBox.Yes:
            return

        try:
            from utils.annotation_utils import auto_annotate_all_images_with_model
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法启动 AI 标注模块: {e}")
            return

        image_list = list(self.project_manager.image_files)
        total = len(image_list)

        def progress_cb(idx, tot):
            self.status_label.setText(f"AI 标注进度: {idx}/{tot}")

        try:
            results = auto_annotate_all_images_with_model(self.config, image_list, progress_callback=progress_cb)
        except Exception as e:
            QMessageBox.critical(self, "AI 标注失败", str(e))
            return

        saved = 0
        try:
            import cv2
            labels_dir = self.config.get_labels_dir()
            labels_dir.mkdir(parents=True, exist_ok=True)

            for img_path, anns in results.items():
                image_stem = os.path.splitext(os.path.basename(img_path))[0]
                label_path = labels_dir / f"{image_stem}.txt"

                try:
                    img = cv2.imread(img_path)
                    h, w = img.shape[:2]
                except Exception:
                    w, h = 0, 0

                lines = []
                for ann in anns:
                    try:
                        lines.append(ann.to_yolo_format(w, h))
                    except Exception:
                        pass

                label_path.write_text("\n".join(lines), encoding='utf-8')
                if lines:
                    saved += 1

        except Exception:
            pass

        self.status_label.setText("AI 标注完成（模型）")
        QMessageBox.information(self, "AI 标注完成", f"为 {saved} 张图片生成了标签（{total} 张图片已处理）")

    def _ai_annotate_all_from_labels(self):
        """对项目中的所有图片使用已有标注迁移策略进行标注并保存标签文件。"""
        if not self.project_manager.image_files:
            QMessageBox.warning(self, "警告", "当前项目没有图片")
            return

        reply = QMessageBox.question(self, "确认", "将对所有图片使用已有标注迁移并覆盖/创建标签文件，是否继续？")
        if reply != QMessageBox.Yes:
            return

        try:
            from utils.annotation_utils import auto_annotate_all_images_from_labels
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法启动标注迁移模块: {e}")
            return

        image_list = list(self.project_manager.image_files)
        total = len(image_list)

        def progress_cb(idx, tot):
            self.status_label.setText(f"标注迁移进度: {idx}/{tot}")

        try:
            results = auto_annotate_all_images_from_labels(self.config, image_list, progress_callback=progress_cb)
        except Exception as e:
            QMessageBox.critical(self, "标注迁移失败", str(e))
            return

        saved = 0
        try:
            import cv2
            labels_dir = self.config.get_labels_dir()
            labels_dir.mkdir(parents=True, exist_ok=True)

            for img_path, anns in results.items():
                image_stem = os.path.splitext(os.path.basename(img_path))[0]
                label_path = labels_dir / f"{image_stem}.txt"

                try:
                    img = cv2.imread(img_path)
                    h, w = img.shape[:2]
                except Exception:
                    w, h = 0, 0

                lines = []
                for ann in anns:
                    try:
                        lines.append(ann.to_yolo_format(w, h))
                    except Exception:
                        pass

                label_path.write_text("\n".join(lines), encoding='utf-8')
                if lines:
                    saved += 1

        except Exception:
            pass

        self.status_label.setText("标注迁移完成")
        QMessageBox.information(self, "标注迁移完成", f"为 {saved} 张图片生成了标签（{total} 张图片已处理）")

    # ==================== 查找标签功能 ====================
    def _find_images_by_label_prompt(self):
        """弹出对话框输入标签名并执行查找"""
        text, ok = QInputDialog.getText(self, "查找标签", "请输入标签名称或文本片段:")
        if ok and text:
            matches = self._find_images_by_label(text.strip())
            if not matches:
                QMessageBox.information(self, "查找结果", "未找到包含该标签的图片")
                return
            # 选中第一个匹配并高亮所有匹配
            first_idx = matches[0]
            # 清选并设置选中项
            self.image_list.clearSelection()
            for idx in matches:
                item = self.image_list.item(idx)
                if item:
                    item.setSelected(True)
            self._load_image(first_idx)

    def _find_images_by_label(self, label_text: str) -> list:
        """在项目的标签文件中查找包含给定文本的图片，返回匹配的图片索引列表。"""
        matches = []
        labels_dir = self.config.get_labels_dir()
        for i, image_path in enumerate(self.project_manager.image_files):
            image_stem = os.path.splitext(os.path.basename(image_path))[0]
            label_path = labels_dir / f"{image_stem}.txt"
            if label_path.exists():
                try:
                    text = label_path.read_text(encoding='utf-8')
                    if label_text in text:
                        matches.append(i)
                except Exception:
                    continue
        return matches
    
    # ==================== UI更新方法 ====================
    
    def _update_project_ui(self):
        """更新项目UI"""
        self.setWindowTitle(f"YOLO-OBB标注工具 - {self.project_name}")
        self.project_name_label.setText(f"项目: {self.project_name}")
    
        # 自动标注迁移（已禁用默认自动运行，因为对于大项目会阻塞/弹出持续对话框）
        # 如果你确实希望在打开/更新项目时自动迁移，请在配置中设置 `app_config.auto_migrate_on_open = True`。
        # ensure defaults so later UI code can safely reference them
        results = {}
        saved = 0
        total = 0

        if getattr(self.config.app_config, 'auto_migrate_on_open', False):
            try:
                from utils.annotation_utils import auto_annotate_all_images_from_labels
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法启动标注迁移模块: {e}")
                return

            image_list = list(self.project_manager.image_files)
            total = len(image_list)

            # 使用进度对话框显示批量进度（可取消）
            progress = QProgressDialog("正在迁移标注...", "取消", 0, total, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setWindowTitle("标注迁移进度")
            progress.show()

            def progress_cb(idx, tot):
                try:
                    progress.setValue(idx)
                    progress.setLabelText(f"正在迁移: {idx}/{tot}")
                    QApplication.processEvents()
                except Exception:
                    pass

            try:
                results = auto_annotate_all_images_from_labels(self.config, image_list, progress_callback=progress_cb)
            except Exception as e:
                progress.close()
                QMessageBox.critical(self, "标注迁移失败", str(e))
                return

            saved = 0
            try:
                import cv2
                labels_dir = self.config.get_labels_dir()
                labels_dir.mkdir(parents=True, exist_ok=True)

                for idx, (img_path, meta) in enumerate(results.items(), start=1):
                    if progress.wasCanceled():
                        break

                    anns = meta.get('annotations', [])
                    src = meta.get('source')
                    score = meta.get('score', 0.0)

                    image_stem = os.path.splitext(os.path.basename(img_path))[0]
                    label_path = labels_dir / f"{image_stem}.txt"

                    try:
                        img = cv2.imread(img_path)
                        h, w = img.shape[:2]
                    except Exception:
                        w, h = 0, 0

                    lines = []
                    for ann in anns:
                        try:
                            lines.append(ann.to_yolo_format(w, h))
                        except Exception:
                            pass

                    try:
                        label_path.write_text("\n".join(lines), encoding='utf-8')
                        if lines:
                            saved += 1
                    except Exception:
                        pass

                progress.close()
            except Exception:
                progress.close()
                pass

        # 展示简单统计与示例匹配信息
        sample_info = []
        for img_path, meta in list(results.items())[:5]:
            if meta.get('source'):
                sample_info.append(f"{os.path.basename(img_path)} <- {os.path.basename(meta.get('source'))} (score={meta.get('score'):.2f})")

        info_msg = f"为 {saved} 张图片生成了标签（{total} 张图片已处理）"
        if sample_info:
            info_msg += "\n示例匹配:\n" + "\n".join(sample_info)

        self.status_label.setText("标注迁移完成")
        QMessageBox.information(self, "标注迁移完成", info_msg)
    
    def _update_image_selection(self):
        """选中并滚动到当前图片列表项（在图片切换后调用）。"""
        try:
            if 0 <= self.current_image_index < self.project_manager.image_count:
                item = self.image_list.item(self.current_image_index)
                if item:
                    self.image_list.setCurrentRow(self.current_image_index)
                    self.image_list.scrollToItem(item)
        except Exception:
            pass
    def keyPressEvent(self, event):
        """键盘按下事件"""
        # 处理快捷键
        # 翻页快捷键: 'a' = 上一张, 'f' = 下一张
        if event.key() == Qt.Key_A and event.modifiers() == Qt.NoModifier:
            self._prev_image()
            return
        elif event.key() == Qt.Key_F and event.modifiers() == Qt.NoModifier:
            self._next_image()
            return

        if event.key() == Qt.Key_S and event.modifiers() == Qt.NoModifier:
            self._set_annotation_mode(AnnotationMode.NONE)
        elif event.key() == Qt.Key_R and event.modifiers() == Qt.NoModifier:
            self._set_annotation_mode(AnnotationMode.HORIZONTAL)
        elif event.key() == Qt.Key_O and event.modifiers() == Qt.NoModifier:
            self._set_annotation_mode(AnnotationMode.ROTATED)
        elif event.key() == Qt.Key_P and event.modifiers() == Qt.NoModifier:
            self._set_annotation_mode(AnnotationMode.POLYGON)
        
        # 调用父类处理其他快捷键
        super().keyPressEvent(event)