"""
导出工具模块
"""

import os
import json
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

from models.annotation_item import AnnotationItem
from models.class_item import ClassItem
from models.enums import ExportFormat

class ExportUtils:
    """导出工具类"""
    
    @staticmethod
    def export_yolo_format(export_dir: str, image_files: List[str], 
                          annotations_dir: str, classes: List[ClassItem]):
        """
        导出为YOLO格式
        
        Args:
            export_dir: 导出目录
            image_files: 图片文件列表
            annotations_dir: 标注文件目录
            classes: 类别列表
        """
        images_dir = os.path.join(export_dir, "images")
        labels_dir = os.path.join(export_dir, "labels")
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # 保存类别文件
        classes_file = os.path.join(export_dir, "classes.txt")
        with open(classes_file, 'w', encoding='utf-8') as f:
            for cls in classes:
                f.write(f"{cls.name}\n")
        
        # 导出图片和标注
        for image_path in image_files:
            try:
                # 复制图片
                image_name = os.path.basename(image_path)
                dest_image_path = os.path.join(images_dir, image_name)
                shutil.copy2(image_path, dest_image_path)
                
                # 复制标注
                label_name = os.path.splitext(image_name)[0] + ".txt"
                src_label_path = os.path.join(annotations_dir, label_name)
                dest_label_path = os.path.join(labels_dir, label_name)
                
                if os.path.exists(src_label_path):
                    shutil.copy2(src_label_path, dest_label_path)
                    
            except Exception as e:
                print(f"导出失败 {image_path}: {e}")
    
    @staticmethod
    def export_coco_format(export_dir: str, image_files: List[str], 
                          annotations_dir: str, classes: List[ClassItem]) -> str:
        """
        导出为COCO格式
        
        Args:
            export_dir: 导出目录
            image_files: 图片文件列表
            annotations_dir: 标注文件目录
            classes: 类别列表
            
        Returns:
            导出的JSON文件路径
        """
        import cv2
        
        coco_data = {
            "info": {
                "description": "YOLO-OBB标注工具导出",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat()
            },
            "licenses": [{"id": 1, "name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # 添加类别
        for i, cls in enumerate(classes):
            coco_data["categories"].append({
                "id": i + 1,
                "name": cls.name,
                "supercategory": "none"
            })
        
        # 图片和标注ID计数器
        image_id = 1
        annotation_id = 1
        
        images_dir = os.path.join(export_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        for image_path in image_files:
            try:
                # 读取图片信息
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                height, width = image.shape[:2]
                image_name = os.path.basename(image_path)
                
                # 添加图片信息
                image_info = {
                    "id": image_id,
                    "file_name": image_name,
                    "width": width,
                    "height": height,
                    "date_captured": datetime.now().isoformat()
                }
                coco_data["images"].append(image_info)
                
                # 复制图片
                dest_image_path = os.path.join(images_dir, image_name)
                shutil.copy2(image_path, dest_image_path)
                
                # 添加标注信息
                label_name = os.path.splitext(image_name)[0] + ".txt"
                label_path = os.path.join(annotations_dir, label_name)
                
                if os.path.exists(label_path):
                    annotations = ExportUtils._load_yolo_annotations_for_coco(
                        label_path, (width, height), classes
                    )
                    
                    for ann in annotations:
                        annotation_data = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": ann["category_id"],
                            "bbox": ann["bbox"],
                            "area": ann["area"],
                            "segmentation": ann.get("segmentation", []),
                            "iscrowd": 0
                        }
                        coco_data["annotations"].append(annotation_data)
                        annotation_id += 1
                
                image_id += 1
                
            except Exception as e:
                print(f"处理图片失败 {image_path}: {e}")
        
        # 保存COCO格式JSON文件
        coco_file = os.path.join(export_dir, "annotations.json")
        with open(coco_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        
        return coco_file
    
    @staticmethod
    def _load_yolo_annotations_for_coco(label_path: str, image_size: Tuple[int, int], 
                                      classes: List[ClassItem]) -> List[Dict[str, Any]]:
        """
        为COCO格式加载YOLO标注
        
        Args:
            label_path: 标签文件路径
            image_size: 图像尺寸
            classes: 类别列表
            
        Returns:
            COCO格式标注列表
        """
        annotations = []
        img_width, img_height = image_size
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 1:
                        continue
                    
                    class_id = int(parts[0]) + 1  # COCO类别ID从1开始
                    
                    if len(parts) == 5:
                        # 水平矩形框
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # 转换为COCO格式的bbox [x, y, width, height]
                        x = (x_center - width/2) * img_width
                        y = (y_center - height/2) * img_height
                        w = width * img_width
                        h = height * img_height
                        
                        annotation = {
                            "category_id": class_id,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "segmentation": []
                        }
                        annotations.append(annotation)
                        
        except Exception as e:
            print(f"加载标注失败 {label_path}: {e}")
        
        return annotations
    
    @staticmethod
    def export_voc_format(export_dir: str, image_files: List[str], 
                         annotations_dir: str, classes: List[ClassItem]):
        """
        导出为VOC格式
        
        Args:
            export_dir: 导出目录
            image_files: 图片文件列表
            annotations_dir: 标注文件目录
            classes: 类别列表
        """
        import cv2
        
        images_dir = os.path.join(export_dir, "JPEGImages")
        annotations_dir_voc = os.path.join(export_dir, "Annotations")
        image_sets_dir = os.path.join(export_dir, "ImageSets", "Main")
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(annotations_dir_voc, exist_ok=True)
        os.makedirs(image_sets_dir, exist_ok=True)
        
        image_names = []
        
        for image_path in image_files:
            try:
                # 读取图片
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                height, width, depth = image.shape
                image_name = os.path.basename(image_path)
                image_base_name = os.path.splitext(image_name)[0]
                
                # 复制图片
                dest_image_path = os.path.join(images_dir, image_name)
                shutil.copy2(image_path, dest_image_path)
                
                # 创建XML标注文件
                xml_file = os.path.join(annotations_dir_voc, f"{image_base_name}.xml")
                ExportUtils._create_voc_xml(xml_file, image_name, width, height, depth, 
                                          annotations_dir, image_base_name, classes)
                
                image_names.append(image_base_name)
                
            except Exception as e:
                print(f"处理图片失败 {image_path}: {e}")
        
        # 创建ImageSets文件
        ExportUtils._create_voc_image_sets(image_sets_dir, image_names)
    
    @staticmethod
    def _create_voc_xml(xml_file: str, image_name: str, width: int, height: int, 
                       depth: int, annotations_dir: str, image_base_name: str, 
                       classes: List[ClassItem]):
        """
        创建VOC格式XML文件
        
        Args:
            xml_file: XML文件路径
            image_name: 图片名称
            width: 图片宽度
            height: 图片高度
            depth: 图片通道数
            annotations_dir: 标注文件目录
            image_base_name: 图片基础名称
            classes: 类别列表
        """
        root = ET.Element("annotation")
        
        # 文件夹信息
        folder = ET.SubElement(root, "folder")
        folder.text = "JPEGImages"
        
        # 文件名
        filename = ET.SubElement(root, "filename")
        filename.text = image_name
        
        # 路径
        path = ET.SubElement(root, "path")
        path.text = f"JPEGImages/{image_name}"
        
        # 图片信息
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = str(depth)
        
        # 标注信息
        label_path = os.path.join(annotations_dir, f"{image_base_name}.txt")
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 1:
                            continue
                        
                        class_id = int(parts[0])
                        if class_id >= len(classes):
                            continue
                        
                        class_name = classes[class_id].name
                        
                        if len(parts) == 5:
                            # 水平矩形框
                            x_center, y_center, box_width, box_height = map(float, parts[1:5])
                            
                            # 转换为VOC格式 [xmin, ymin, xmax, ymax]
                            xmin = (x_center - box_width/2) * width
                            ymin = (y_center - box_height/2) * height
                            xmax = (x_center + box_width/2) * width
                            ymax = (y_center + box_height/2) * height
                            
                            # 创建object节点
                            obj = ET.SubElement(root, "object")
                            ET.SubElement(obj, "name").text = class_name
                            ET.SubElement(obj, "pose").text = "Unspecified"
                            ET.SubElement(obj, "truncated").text = "0"
                            ET.SubElement(obj, "difficult").text = "0"
                            
                            bndbox = ET.SubElement(obj, "bndbox")
                            ET.SubElement(bndbox, "xmin").text = str(int(xmin))
                            ET.SubElement(bndbox, "ymin").text = str(int(ymin))
                            ET.SubElement(bndbox, "xmax").text = str(int(xmax))
                            ET.SubElement(bndbox, "ymax").text = str(int(ymax))
                            
            except Exception as e:
                print(f"处理标注失败 {label_path}: {e}")
        
        # 写入XML文件
        tree = ET.ElementTree(root)
        tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    
    @staticmethod
    def _create_voc_image_sets(image_sets_dir: str, image_names: List[str]):
        """
        创建VOC ImageSets文件
        
        Args:
            image_sets_dir: ImageSets目录
            image_names: 图片名称列表
        """
        if not image_names:
            return
        
        # 分割训练集和验证集 (80%训练，20%验证)
        split_idx = int(len(image_names) * 0.8)
        train_names = image_names[:split_idx]
        val_names = image_names[split_idx:]
        
        # 写入train.txt
        train_file = os.path.join(image_sets_dir, "train.txt")
        with open(train_file, 'w') as f:
            for name in train_names:
                f.write(f"{name}\n")
        
        # 写入val.txt
        val_file = os.path.join(image_sets_dir, "val.txt")
        with open(val_file, 'w') as f:
            for name in val_names:
                f.write(f"{name}\n")
        # 写入trainval.txt (所有图片)
        trainval_file = os.path.join(image_sets_dir, "trainval.txt")
        with open(trainval_file, 'w') as f:
            for name in image_names:
                f.write(f"{name}\n")

    @staticmethod
    def split_dataset(export_dir: str, data_pairs: List[Tuple[str, str]], 
                     classes: List[ClassItem], ratios: Dict[str, float], 
                     progress_callback=None):
        """
        自动划分数据集为 训练集、平衡验证集、测试集 (YOLO格式)
        
        Args:
            export_dir: 导出目录
            data_pairs: (图片路径, 标签路径) 元组列表
            classes: 类别列表
            ratios: 划分比例, 如 {"train": 0.7, "val": 0.2, "test": 0.1}
            progress_callback: 进度回调函数 (idx, total)
        """
        # 1. 准备目录结构
        subsets = ["train", "val", "test"]
        for subset in subsets:
            os.makedirs(os.path.join(export_dir, "images", subset), exist_ok=True)
            os.makedirs(os.path.join(export_dir, "labels", subset), exist_ok=True)
        
        # 2. 随机打乱
        shuffled_pairs = list(data_pairs)
        random.shuffle(shuffled_pairs)
        
        total = len(shuffled_pairs)
        train_count = int(total * ratios.get("train", 0.7))
        val_count = int(total * ratios.get("val", 0.2))
        
        # 3. 划分索引
        train_pairs = shuffled_pairs[:train_count]
        val_pairs = shuffled_pairs[train_count:train_count+val_count]
        test_pairs = shuffled_pairs[train_count+val_count:]
        
        dataset_map = {
            "train": train_pairs,
            "val": val_pairs,
            "test": test_pairs
        }
        
        # 4. 执行复制
        current_idx = 0
        for subset, pairs in dataset_map.items():
            subset_img_dir = os.path.join(export_dir, "images", subset)
            subset_lbl_dir = os.path.join(export_dir, "labels", subset)
            
            for img_path, lbl_path in pairs:
                try:
                    img_name = os.path.basename(img_path)
                    # 复制图片
                    shutil.copy2(img_path, os.path.join(subset_img_dir, img_name))
                    
                    # 复制对应标注 (lbl_path 可能是 None 如果之前没过滤干净，但这里应该已经过滤了)
                    if lbl_path and os.path.exists(lbl_path):
                        lbl_name = os.path.basename(lbl_path)
                        shutil.copy2(lbl_path, os.path.join(subset_lbl_dir, lbl_name))
                except Exception as e:
                    print(f"Error splitting {img_path}: {e}")
                
                current_idx += 1
                if progress_callback:
                    progress_callback(current_idx, total)
        
        # 5. 保存 classes.txt (YOLO标准)
        classes_file = os.path.join(export_dir, "classes.txt")
        with open(classes_file, 'w', encoding='utf-8') as f:
            for cls in classes:
                f.write(f"{cls.name}\n")