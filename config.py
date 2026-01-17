"""
配置管理模块
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
    """应用配置"""
    # 路径配置
    output_dir: str = "yolo_obb_dataset"
    
    # 自动保存配置
    auto_save: bool = True
    auto_save_interval: int = 5  # 分钟
    save_labels_in_image_dir: bool = True
    backup_count: int = 10
    max_backup_age_days: int = 30
    
    # 视图配置
    show_labels: bool = True
    show_confidence: bool = False
    show_grid: bool = False
    grid_size: int = 50
    default_zoom: float = 1.0
    annotation_opacity: int = 128  # 0-255
    
    # 标注配置
    default_class_colors: List[str] = field(default_factory=lambda: [
        "#FF5252", "#FF4081", "#E040FB", "#7C4DFF", "#536DFE",
        "#448AFF", "#40C4FF", "#18FFFF", "#64FFDA", "#69F0AE",
        "#B2FF59", "#EEFF41", "#FFFF00", "#FFD740", "#FFAB40"
    ])
    annotation_line_width: int = 2
    selected_line_width: int = 3
    
    # 智能标注配置
    ai_model_path: str = ""
    ai_confidence_threshold: float = 0.25
    ai_iou_threshold: float = 0.45
    ai_max_detections: int = 300
    
    # 训练配置
    train_epochs: int = 100
    train_imgsz: int = 640
    train_batch: int = 16
    train_device: str = "0"  # 0 or "cpu"
    train_model_type: str = "yolov8n.pt"
    
    # 快捷键配置
    shortcuts: Dict[str, str] = field(default_factory=lambda: {
        "select_tool": "S",
        "rectangle_tool": "R",
        "rotated_tool": "O",
        "polygon_tool": "P",
        "points_tool": "K",
        "zoom_in": "Ctrl++",
        "zoom_out": "Ctrl+-",
        "fit_window": "Ctrl+0",
        "actual_size": "Ctrl+1",
        "delete_annotation": "Delete",
        "copy_annotation": "Ctrl+C",
        "paste_annotation": "Ctrl+V",
        "undo": "Ctrl+Z",
        "redo": "Ctrl+Y",
        "save_project": "Ctrl+S",
        "open_project": "Ctrl+O",
        "next_image": "Right",
        "prev_image": "Left",
        "toggle_grid": "Ctrl+G"
    })
    
    # 窗口配置
    window_width: int = 1400
    window_height: int = 900
    window_maximized: bool = False
    splitter_sizes: List[int] = field(default_factory=lambda: [300, 800, 300])
    
    # 最近文件
    recent_projects: List[str] = field(default_factory=list)
    max_recent_projects: int = 10
    
    # 导出配置
    export_yolo_format: bool = True
    export_coco_format: bool = False
    export_voc_format: bool = False
    export_split_ratio: Dict[str, float] = field(default_factory=lambda: {
        "train": 0.7,
        "val": 0.2,
        "test": 0.1
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 处理特殊类型
        data['shortcuts'] = dict(data['shortcuts'])
        data['export_split_ratio'] = dict(data['export_split_ratio'])
        return data
    
    def update_from_dict(self, data: Dict[str, Any]):
        """从字典更新配置"""
        for key, value in data.items():
            if hasattr(self, key):
                # 处理特殊字段
                if key == 'shortcuts' and isinstance(value, dict):
                    self.shortcuts.update(value)
                elif key == 'export_split_ratio' and isinstance(value, dict):
                    self.export_split_ratio.update(value)
                else:
                    setattr(self, key, value)
    
    def validate(self) -> Tuple[bool, str]:
        """验证配置"""
        errors = []
        
        # 验证路径
        if not self.output_dir:
            errors.append("输出目录不能为空")
        
        # 验证自动保存间隔
        if self.auto_save_interval < 1 or self.auto_save_interval > 60:
            errors.append("自动保存间隔应在1-60分钟之间")
        
        # 验证备份数量
        if self.backup_count < 1 or self.backup_count > 100:
            errors.append("备份数量应在1-100之间")
        
        # 验证标注不透明度
        if self.annotation_opacity < 0 or self.annotation_opacity > 255:
            errors.append("标注不透明度应在0-255之间")
        
        # 验证快捷键
        for key, shortcut in self.shortcuts.items():
            if not shortcut:
                errors.append(f"快捷键 '{key}' 不能为空")
        
        if errors:
            return False, "; ".join(errors)
        return True, ""

class Config:
    """配置管理器"""
    
    def __init__(self, config_file: str = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径，如果为None则使用默认路径
        """
        self.config_file = config_file or "config.json"
        self.app_config = AppConfig()
        self._loaded = False
        
        # 确保配置目录存在
        self._ensure_config_dir()
        
        # 加载配置
        self.load_config()
    
    def _ensure_config_dir(self):
        """确保配置目录存在"""
        config_dir = os.path.dirname(self.config_file)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
    
    def load_config(self) -> bool:
        """加载配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.app_config.update_from_dict(data)
                
                # 验证配置
                valid, message = self.app_config.validate()
                if not valid:
                    logger.warning(f"配置验证失败: {message}")
                    # 重置为默认配置
                    self.app_config = AppConfig()
                
                self._loaded = True
                logger.info(f"配置已从 {self.config_file} 加载")
                return True
                
            except json.JSONDecodeError as e:
                logger.error(f"配置文件格式错误: {e}")
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
        
        logger.info("使用默认配置")
        return False
    
    def save_config(self) -> bool:
        """保存配置文件"""
        try:
            # 验证配置
            valid, message = self.app_config.validate()
            if not valid:
                logger.error(f"配置验证失败: {message}")
                return False
            
            # 创建备份
            if os.path.exists(self.config_file):
                backup_file = f"{self.config_file}.backup"
                try:
                    import shutil
                    shutil.copy2(self.config_file, backup_file)
                except Exception as e:
                    logger.warning(f"创建配置备份失败: {e}")
            
            # 保存配置
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.app_config.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置已保存到 {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            return False
    
    def get_workspace_path(self) -> Path:
        """获取工作空间路径"""
        return Path(self.app_config.output_dir).expanduser().resolve()
    
    def get_images_dir(self) -> Path:
        """获取图片目录"""
        return self.get_workspace_path() / "images"
    
    def get_labels_dir(self) -> Path:
        """获取标签目录"""
        return self.get_workspace_path() / "labels"
    
    def get_backup_dir(self) -> Path:
        """获取备份目录"""
        return self.get_workspace_path() / "backups"
    
    def get_export_dir(self) -> Path:
        """获取导出目录"""
        return self.get_workspace_path() / "exports"
    
    def get_temp_dir(self) -> Path:
        """获取临时目录"""
        return self.get_workspace_path() / "temp"
    
    def ensure_dirs(self):
        """确保所有目录存在"""
        dirs = [
            self.get_workspace_path(),
            self.get_images_dir(),
            self.get_labels_dir(),
            self.get_backup_dir(),
            self.get_export_dir(),
            self.get_temp_dir()
        ]
        
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
    
    def add_recent_project(self, project_path: str):
        """添加最近项目"""
        # 规范化路径
        project_path = os.path.abspath(project_path)
        
        # 如果已在列表中，先移除
        if project_path in self.app_config.recent_projects:
            self.app_config.recent_projects.remove(project_path)
        
        # 添加到列表开头
        self.app_config.recent_projects.insert(0, project_path)
        
        # 限制最近项目数量
        if len(self.app_config.recent_projects) > self.app_config.max_recent_projects:
            self.app_config.recent_projects = self.app_config.recent_projects[:self.app_config.max_recent_projects]
        
        # 保存配置
        self.save_config()
    
    def clear_recent_projects(self):
        """清空最近项目"""
        self.app_config.recent_projects.clear()
        self.save_config()
    
    def get_recent_projects(self) -> List[str]:
        """获取最近项目列表（过滤不存在的项目）"""
        recent_projects = []
        for project_path in self.app_config.recent_projects:
            if os.path.exists(project_path):
                recent_projects.append(project_path)
        
        # 更新列表
        if len(recent_projects) != len(self.app_config.recent_projects):
            self.app_config.recent_projects = recent_projects
            self.save_config()
        
        return recent_projects
    
    def create_dataset_config(self, export_dir: str, class_names: List[str], 
                            split_info: Optional[Dict[str, List[str]]] = None) -> str:
        """
        创建数据集配置文件
        
        Args:
            export_dir: 导出目录
            class_names: 类别名称列表
            split_info: 分割信息，格式为 {"train": [...], "val": [...], "test": [...]}
            
        Returns:
            配置文件路径
        """
        try:
            config_path = os.path.join(export_dir, "dataset.yaml")
            
            config = {
                "path": os.path.abspath(export_dir),
                "train": "images/train" if split_info and 'train' in split_info else "images",
                "val": "images/val" if split_info and 'val' in split_info else "images",
                "test": "images/test" if split_info and 'test' in split_info else "",
                "nc": len(class_names),
                "names": class_names
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write("# YOLO Dataset Configuration\n")
                f.write(f"# Generated by YOLO-OBB Annotator\n")
                f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 使用 yaml.dump 确保格式正确，特别是数值类型
                yaml.dump(config, f, allow_unicode=True, sort_keys=False)
                
                # 添加分割信息注释
                if split_info:
                    f.write("\n# Split Information\n")
                    for split_name, file_list in split_info.items():
                        f.write(f"#{split_name}: {len(file_list)} images\n")
            
            logger.info(f"数据集配置文件已创建: {config_path}")
            return config_path
            
        except Exception as e:
            logger.error(f"创建数据集配置文件失败: {e}")
            raise
    
    def cleanup_old_backups(self, max_age_days: Optional[int] = None):
        """清理旧的备份文件"""
        backup_dir = self.get_backup_dir()
        if not backup_dir.exists():
            return
        
        max_age_days = max_age_days or self.app_config.max_backup_age_days
        current_time = datetime.now().timestamp()
        
        for file_path in backup_dir.iterdir():
            if file_path.is_file():
                # 获取文件修改时间
                mtime = file_path.stat().st_mtime
                age_days = (current_time - mtime) / (24 * 3600)
                
                if age_days > max_age_days:
                    try:
                        file_path.unlink()
                        logger.info(f"删除旧备份文件: {file_path}")
                    except Exception as e:
                        logger.warning(f"删除备份文件失败 {file_path}: {e}")
    
    def reset_to_defaults(self):
        """重置为默认配置"""
        self.app_config = AppConfig()
        self.save_config()
        logger.info("配置已重置为默认值")
    
    def __getitem__(self, key: str) -> Any:
        """获取配置项"""
        return getattr(self.app_config, key)
    
    def __setitem__(self, key: str, value: Any):
        """设置配置项"""
        setattr(self.app_config, key, value)
    
    def __contains__(self, key: str) -> bool:
        """检查配置项是否存在"""
        return hasattr(self.app_config, key)