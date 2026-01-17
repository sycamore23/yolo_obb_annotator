# -*- coding: utf-8 -*-
"""
Core project manager for YOLO OBB annotator.
Provides project load/save and utilities around images/labels/backups.
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from config import Config
from models.class_item import ClassItem
from PyQt5.QtGui import QColor


class ProjectManager:
    """Project manager."""

    def __init__(self, config: Config):
        self.config = config
        self.project_name: str = "untitled"
        self.project_path: Optional[str] = None
        self.project_modified: bool = False
        self.image_files: List[str] = []
        self.current_image_index: int = 0

    def new_project(self):
        self.project_name = "untitled"
        self.project_path = None
        self.project_modified = False
        self.image_files = []
        self.current_image_index = 0
        self._clear_workspace()

    def open_project(self, file_path: str) -> bool:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.project_name = data.get('name', Path(file_path).stem)
            self.project_path = file_path
            self.image_files = data.get('image_files', [])
            self.current_image_index = 0
            self.project_modified = False

            # add to recent
            try:
                self.config.add_recent_project(file_path)
            except Exception:
                pass

            return True
        except Exception as e:
            print(f"Failed to open project: {e}")
            return False

    def save_project(self) -> bool:
        if not self.project_path or self.project_name == "untitled":
            return self.save_project_as()
        return self._save_project_file(self.project_path)

    def save_project_as(self) -> bool:
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(None, "Save project", "", "Project Files (*.json)")
        if file_path:
            self.project_path = file_path
            self.project_name = Path(file_path).stem
            return self._save_project_file(file_path)
        return False

    def _save_project_file(self, file_path: str) -> bool:
        try:
            project_data = {
                'name': self.project_name,
                'image_files': self.image_files,
                'save_time': datetime.now().isoformat()
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, indent=2, ensure_ascii=False)
            self.project_modified = False
            try:
                self.config.add_recent_project(file_path)
            except Exception:
                pass
            return True
        except Exception as e:
            print(f"Failed to save project: {e}")
            return False

    def _clear_workspace(self):
        # preserve config and classes files
        output = Path(self.config.app_config.output_dir)
        config_files = {
            os.path.abspath(self.config.config_file),
            os.path.abspath(str(output / "classes.txt")),
            os.path.abspath(str(output / "config.json")),
        }
        for root, dirs, files in os.walk(self.config.app_config.output_dir):
            for fname in files:
                fp = os.path.abspath(os.path.join(root, fname))
                if fp not in config_files:
                    try:
                        os.remove(fp)
                    except Exception:
                        pass

    def get_current_image_path(self) -> Optional[str]:
        if 0 <= self.current_image_index < len(self.image_files):
            return self.image_files[self.current_image_index]
        return None

    def has_images(self) -> bool:
        return len(self.image_files) > 0

    @property
    def image_count(self) -> int:
        return len(self.image_files)

    def get_image_path(self, index: int) -> Optional[str]:
        if 0 <= index < len(self.image_files):
            return self.image_files[index]
        return None

    def get_label_path(self, image_path: str) -> Path:
        stem = Path(image_path).stem
        if getattr(self.config.app_config, 'save_labels_in_image_dir', False):
            return Path(image_path).parent / f"{stem}.txt"
        return self.config.get_labels_dir() / f"{stem}.txt"

    def has_annotation(self, image_path: str) -> bool:
        try:
            return self.get_label_path(image_path).exists()
        except Exception:
            return False

    def get_images_dir(self) -> Path:
        return self.config.get_images_dir()

    def get_labels_dir(self) -> Path:
        return self.config.get_labels_dir()

    def get_backup_dir(self) -> Path:
        return self.config.get_backup_dir()

    def rename_images_with_backup(self, new_paths: List[str]) -> str:
        if len(new_paths) != len(self.image_files):
            raise ValueError("new_paths length must match current image_files length")

        backup_root = self.get_backup_dir()
        backup_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_sub = backup_root / f"rename_{timestamp}"
        backup_sub.mkdir(parents=True, exist_ok=True)

        records = []
        moved = []
        labels_dir = self.get_labels_dir()
        try:
            # backup originals
            for old in self.image_files:
                p_old = Path(old)
                if p_old.exists():
                    shutil.copy2(str(p_old), str(backup_sub / p_old.name))
                try:
                    label_old = labels_dir / f"{p_old.stem}.txt"
                    if label_old.exists():
                        shutil.copy2(str(label_old), str(backup_sub / f"{p_old.stem}.txt"))
                except Exception:
                    pass

            for old, new in zip(self.image_files, new_paths):
                p_old = Path(old)
                p_new = Path(new)
                p_new.parent.mkdir(parents=True, exist_ok=True)
                if p_new.exists():
                    raise FileExistsError(f"Target file already exists: {p_new}")
                if p_old.exists():
                    shutil.move(str(p_old), str(p_new))
                    moved.append((p_old, p_new))
                try:
                    label_old = labels_dir / f"{p_old.stem}.txt"
                    label_new = labels_dir / f"{p_new.stem}.txt"
                    if label_old.exists():
                        if label_new.exists():
                            raise FileExistsError(f"Target label already exists: {label_new}")
                        shutil.move(str(label_old), str(label_new))
                        moved.append((label_old, label_new))
                except Exception:
                    raise

                records.append({'old': str(p_old), 'new': str(p_new)})

            self.image_files = list(new_paths)
            self.project_modified = True

            record_path = backup_sub / "rename_record.json"
            with open(record_path, 'w', encoding='utf-8') as f:
                json.dump({'timestamp': timestamp, 'mappings': records}, f, ensure_ascii=False, indent=2)

            return str(record_path)
        except Exception:
            # attempt rollback
            for src, dst in reversed(moved):
                try:
                    if dst.exists():
                        shutil.move(str(dst), str(src))
                except Exception:
                    pass
            # restore from backup copies
            try:
                for file in backup_sub.iterdir():
                    try:
                        target = Path(self.config.get_images_dir()) / file.name
                        if not target.exists():
                            shutil.copy2(str(file), str(target))
                    except Exception:
                        pass
            except Exception:
                pass
            raise

    def undo_last_rename(self, record_file: Optional[str] = None) -> bool:
        backup_root = self.get_backup_dir()
        if record_file is None:
            candidates = sorted(backup_root.glob('rename_*'), reverse=True)
            if not candidates:
                raise FileNotFoundError("No rename backup found")
            record_dir = candidates[0]
            record_path = record_dir / "rename_record.json"
        else:
            record_path = Path(record_file)
            record_dir = record_path.parent

        if not record_path.exists():
            raise FileNotFoundError(f"Record file does not exist: {record_path}")

        with open(record_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        mappings = data.get('mappings', [])
        labels_dir = self.get_labels_dir()

        moved_back = []
        try:
            for entry in reversed(mappings):
                old = Path(entry['old'])
                new = Path(entry['new'])
                if new.exists():
                    old.parent.mkdir(parents=True, exist_ok=True)
                    if old.exists():
                        try:
                            old.unlink()
                        except Exception:
                            pass
                    shutil.move(str(new), str(old))
                    moved_back.append((new, old))
                try:
                    label_old = labels_dir / f"{old.stem}.txt"
                    label_new = labels_dir / f"{new.stem}.txt"
                    if label_new.exists():
                        if label_old.exists():
                            try:
                                label_old.unlink()
                            except Exception:
                                pass
                        shutil.move(str(label_new), str(label_old))
                        moved_back.append((label_new, label_old))
                except Exception:
                    pass

            old_list = [entry['old'] for entry in mappings]
            self.image_files = old_list
            self.project_modified = True
            return True
        except Exception:
            for src, dst in reversed(moved_back):
                try:
                    if dst.exists():
                        shutil.move(str(dst), str(src))
                except Exception:
                    pass
            raise

    def add_image_files(self, file_paths: List[str]):
        for file_path in file_paths:
            if file_path not in self.image_files:
                self.image_files.append(file_path)
        self.image_files.sort()
        self.project_modified = True

    def remove_image_file(self, index: int):
        if 0 <= index < len(self.image_files):
            self.image_files.pop(index)
            self.project_modified = True

    def load_project(self, file_path: str) -> bool:
        return self.open_project(file_path)

    def get_classes(self, directory: Optional[str] = None) -> List[ClassItem]:
        """Load classes from classes.json or classes.txt. Returns a list of ClassItem.

        Supported formats:
        - classes.json: list of strings or list of objects with 'name' and optional 'color'
        - classes.txt: each line 'name' or 'name,color' (comma or tab separated)
        """
        classes: List[ClassItem] = []
        try:
            if directory:
                workspace = Path(directory)
            else:
                workspace = self.config.get_workspace_path()
            
            classes_txt = workspace / "classes.txt"
            classes_json = workspace / "classes.json"
            default_colors = getattr(self.config.app_config, 'default_class_colors', [])

            if classes_json.exists():
                try:
                    with open(classes_json, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        for idx, entry in enumerate(data):
                            if isinstance(entry, str):
                                name = entry.strip()
                                color_value = default_colors[idx % len(default_colors)] if default_colors else '#FF5252'
                            elif isinstance(entry, dict):
                                name = str(entry.get('name', '')).strip()
                                color_value = entry.get('color') or (default_colors[idx % len(default_colors)] if default_colors else '#FF5252')
                            else:
                                continue
                            if not name:
                                continue
                            try:
                                color = QColor(color_value)
                                if not color.isValid():
                                    color = QColor(default_colors[idx % len(default_colors)] if default_colors else '#FF5252')
                            except Exception:
                                color = QColor(default_colors[idx % len(default_colors)] if default_colors else '#FF5252')
                            classes.append(ClassItem(idx, name, color))
                except Exception:
                    pass

            if not classes and classes_txt.exists():
                with open(classes_txt, 'r', encoding='utf-8') as f:
                    for idx, raw in enumerate(f):
                        line = raw.strip()
                        if not line:
                            continue
                        
                        # 支持 ID:Name, ID,Name, Name 格式
                        name = ""
                        color_value = None
                        
                        if ':' in line:
                            # 优先处理 ID:Name 格式
                            parts = line.split(':', 1)
                            name = parts[1].strip()
                        elif '\t' in line:
                            parts = line.split('\t')
                            name = parts[0].strip()
                            if len(parts) > 1: color_value = parts[1].strip()
                        elif ',' in line:
                            parts = [p.strip() for p in line.split(',')]
                            # 检查是否为 ID,Name 格式
                            if parts[0].isdigit() and len(parts) > 1:
                                name = parts[1].strip()
                                if len(parts) > 2: color_value = parts[2].strip()
                            else:
                                name = parts[0].strip()
                                if len(parts) > 1: color_value = parts[1].strip()
                        else:
                            name = line
                            
                        if not name:
                            continue
                        if not color_value:
                            color_value = default_colors[idx % len(default_colors)] if default_colors else '#FF5252'
                        try:
                            color = QColor(color_value)
                            if not color.isValid():
                                color = QColor(default_colors[idx % len(default_colors)] if default_colors else '#FF5252')
                        except Exception:
                            color = QColor(default_colors[idx % len(default_colors)] if default_colors else '#FF5252')
                        classes.append(ClassItem(idx, name, color))

            if not classes:
                color_value = default_colors[0] if default_colors else '#FF5252'
                try:
                    color = QColor(color_value)
                except Exception:
                    color = QColor('#FF5252')
                classes.append(ClassItem(0, 'object', color))
        except Exception:
            try:
                color = QColor('#FF5252')
                return [ClassItem(0, 'object', color)]
            except Exception:
                return []
        return classes

    def create_backup(self):
        backup_dir = self.get_backup_dir()
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_image_path = self.get_current_image_path()
        if current_image_path:
            image_name = Path(current_image_path).stem
            label_path = self.get_labels_dir() / f"{image_name}.txt"
            if label_path.exists():
                backup_path = backup_dir / f"{timestamp}_{image_name}.txt"
                shutil.copy2(label_path, backup_path)
        self._cleanup_old_backups()

    def _cleanup_old_backups(self):
        backup_dir = self.get_backup_dir()
        if not backup_dir.exists():
            return
        backups = sorted(backup_dir.iterdir())
        if len(backups) > self.config.app_config.backup_count:
            for backup in backups[:-self.config.app_config.backup_count]:
                try:
                    backup.unlink()
                except Exception:
                    pass