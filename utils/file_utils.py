"""
文件操作工具模块
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np

from models.annotation_item import AnnotationItem
from models.class_item import ClassItem
from models.enums import BBoxType

class FileUtils:
    """文件操作工具类"""
    
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """
        加载图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            OpenCV图像或None
        """
        try:
            image = cv2.imread(image_path)
            if image is not None:
                return image
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
        
        return None
    
    @staticmethod
    def get_image_files(directory: str) -> List[str]:
        """
        获取目录下的所有图片文件
        
        Args:
            directory: 目录路径
            
        Returns:
            图片文件路径列表
        """
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
        
        image_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    image_files.append(os.path.join(root, file))
        
        return sorted(image_files)
    
    @staticmethod
    def load_classes(classes_file: str) -> List[ClassItem]:
        """
        加载类别文件
        
        Args:
            classes_file: 类别文件路径
            
        Returns:
            类别项列表
        """
        classes = []
        
        if os.path.exists(classes_file):
            try:
                with open(classes_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        name = line.strip()
                        if name:
                            color = QColor(np.random.randint(50, 255), 
                                         np.random.randint(50, 255), 
                                         np.random.randint(50, 255))
                            classes.append(ClassItem(i, name, color))
            except Exception as e:
                print(f"加载类别失败: {e}")
        
        return classes
    
    @staticmethod
    def save_classes(classes_file: str, classes: List[ClassItem]):
        """
        保存类别文件 (严格遵循 YOLO 规范：按 ID 排序，每行一个类别)
        
        Args:
            classes_file: 类别文件路径
            classes: 类别项列表
        """
        try:
            # 确保按 ID 从小到大排序
            sorted_classes = sorted(classes, key=lambda x: x.id)
            
            with open(classes_file, 'w', encoding='utf-8', newline='\n') as f:
                for cls in sorted_classes:
                    name = str(cls.name).strip()
                    if name:
                        f.write(f"{cls.id}:{name}\n")
        except Exception as e:
            print(f"保存类别失败: {e}")
    
    @staticmethod
    def load_yolo_annotations(label_path: str, image_size: Tuple[int, int], 
                            classes: List[ClassItem]) -> List[AnnotationItem]:
        """
        加载YOLO格式标注
        
        Args:
            label_path: 标签文件路径
            image_size: 图像尺寸 (width, height)
            classes: 类别列表
            
        Returns:
            标注项列表
        """
        annotations = []
        
        if not os.path.exists(label_path):
            return annotations
        
        img_width, img_height = image_size
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 1:
                        continue
                    
                    class_id = int(parts[0])
                    
                    # 获取类别名称
                    class_name = ""
                    if 0 <= class_id < len(classes):
                        class_name = classes[class_id].name
                    
                    if len(parts) == 5:
                        # 水平矩形框
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # 转换为像素坐标
                        x1 = (x_center - width/2) * img_width
                        y1 = (y_center - height/2) * img_height
                        x2 = (x_center + width/2) * img_width
                        y2 = (y_center + height/2) * img_height
                        
                        pts = [(float(x1), float(y1)), (float(x2), float(y1)), (float(x2), float(y2)), (float(x1), float(y2))]
                        annotation = AnnotationItem(
                            points=pts,
                            bbox_type=BBoxType.HORIZONTAL,
                            class_id=class_id,
                            class_name=class_name
                        )
                        annotations.append(annotation)
                    
                    elif len(parts) >= 9:
                        # 旋转矩形框或多边形
                        points = list(map(float, parts[1:]))

                        # 转换为像素坐标，构造成点元组列表
                        pixel_pts = []
                        for i in range(0, min(len(points), 8), 2):
                            x = points[i] * img_width
                            y = points[i + 1] * img_height
                            pixel_pts.append((float(x), float(y)))

                        bbox_type = BBoxType.ROTATED if len(pixel_pts) == 4 else BBoxType.POLYGON

                        annotation = AnnotationItem(
                            points=pixel_pts,
                            bbox_type=bbox_type,
                            class_id=class_id,
                            class_name=class_name
                        )
                        annotations.append(annotation)
                        
        except Exception as e:
            print(f"加载标注失败 {label_path}: {e}")
        
        return annotations
    
    @staticmethod
    def save_yolo_annotations(label_path: str, annotations: List[AnnotationItem], 
                            image_size: Tuple[int, int]):
        """
        保存YOLO格式标注
        
        Args:
            label_path: 标签文件路径
            annotations: 标注项列表
            image_size: 图像尺寸 (width, height)
        """
        img_width, img_height = image_size
        
        try:
            # Ensure directory exists
            label_dir = os.path.dirname(label_path)
            if label_dir and not os.path.exists(label_dir):
                os.makedirs(label_dir, exist_ok=True)

            with open(label_path, 'w') as f:
                for ann in annotations:
                    try:
                        if ann.bbox_type == BBoxType.HORIZONTAL:
                            x1, y1, x2, y2 = ann.get_bbox()

                            x_center = (x1 + x2) / 2 / img_width
                            y_center = (y1 + y2) / 2 / img_height
                            bbox_width = (x2 - x1) / img_width
                            bbox_height = (y2 - y1) / img_height

                            f.write(f"{ann.class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

                        elif ann.bbox_type in (BBoxType.ROTATED, BBoxType.POLYGON):
                            pts = ann.get_points()
                            line = f"{ann.class_id}"
                            for (x, y) in pts:
                                x_norm = x / img_width if img_width else 0.0
                                y_norm = y / img_height if img_height else 0.0
                                line += f" {x_norm:.6f} {y_norm:.6f}"
                            f.write(f"{line}\n")
                    except Exception:
                        # 忽略单个标注格式化错误
                        continue

                # Ensure data is flushed and synced to disk so switching images sees the latest file
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    # os.fsync may fail on some environments (e.g., some network FS), ignore safely
                    pass

        except Exception as e:
            print(f"保存标注失败 {label_path}: {e}")