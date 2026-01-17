"""
增强的 AnnotationManager：支持剪贴板、记录 add/remove/modify 操作并返回可用于 UI 的 undo/redo 指令。

支持单项或多项操作。
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import uuid

from models.annotation_item import AnnotationItem


@dataclass
class ActionRecord:
    """记录一次用户操作以便撤销/重做。

    action: 'add' | 'remove' | 'modify'
    annotations: List[AnnotationItem]    # 包含该次操作受影响的所有标注（多选支持）
    indices: Optional[List[int]] = None  # 记录原始插入位置，用于撤销 remove
    prev_states: Optional[List[AnnotationItem]] = None # modify 时的旧状态
    """
    action: str
    annotations: List[AnnotationItem]
    indices: Optional[List[int]] = None
    prev_states: Optional[List[AnnotationItem]] = None


class AnnotationManager:
    """增强的标注管理：支持剪贴板、以及记录 add/remove/modify 操作用于撤销/重做。"""

    def __init__(self, config=None):
        self.config = config

        # 剪贴板中存放最近一次复制的标注列表（深拷贝）
        self._clipboard: List[AnnotationItem] = []

        # 操作栈
        self._undo_stack: List[ActionRecord] = []
        self._redo_stack: List[ActionRecord] = []

    def clear(self):
        """清理管理器状态。"""
        self._clipboard = []
        self._undo_stack.clear()
        self._redo_stack.clear()

    # ---- 剪贴板操作 ----
    def copy_annotation(self, annotations: Union[AnnotationItem, List[AnnotationItem]]):
        """将给定标注或标注列表复制到内部剪贴板（深拷贝）。"""
        if annotations is None:
            self._clipboard = []
            return

        if isinstance(annotations, list):
            self._clipboard = [a.copy() for a in annotations if a is not None]
        else:
            self._clipboard = [annotations.copy()]

    def paste_annotation(self) -> List[AnnotationItem]:
        """从剪贴板创建新的标注实例列表并返回。同时也记录为 add 操作。"""
        if not self._clipboard:
            return []

        new_list: List[AnnotationItem] = []
        for src in self._clipboard:
            new_ann = src.copy()
            new_ann.id = str(uuid.uuid4())
            try:
                new_ann.update()
            except Exception:
                pass
            new_list.append(new_ann)

        # 记录为批量 add 操作
        if new_list:
            self.record_add(new_list)
            
        return new_list

    # ---- 记录操作 ----
    def record_add(self, annotations: Union[AnnotationItem, List[AnnotationItem]], indices: Optional[List[int]] = None):
        if not isinstance(annotations, list):
            annotations = [annotations]
        rec = ActionRecord(
            action='add', 
            annotations=[a.copy() for a in annotations], 
            indices=indices
        )
        self._undo_stack.append(rec)
        self._redo_stack.clear()

    def record_remove(self, annotations: Union[AnnotationItem, List[AnnotationItem]], indices: Optional[List[int]] = None):
        if not isinstance(annotations, list):
            annotations = [annotations]
        rec = ActionRecord(
            action='remove', 
            annotations=[a.copy() for a in annotations], 
            indices=indices
        )
        self._undo_stack.append(rec)
        self._redo_stack.clear()

    def record_modify(self, prev_states: Union[AnnotationItem, List[AnnotationItem]], new_states: Union[AnnotationItem, List[AnnotationItem]]):
        if not isinstance(prev_states, list):
            prev_states = [prev_states]
        if not isinstance(new_states, list):
            new_states = [new_states]
            
        rec = ActionRecord(
            action='modify', 
            annotations=[a.copy() for a in new_states], 
            prev_states=[a.copy() for a in prev_states]
        )
        self._undo_stack.append(rec)
        self._redo_stack.clear()

    # ---- 撤销/重做 ----
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def undo(self) -> Optional[Dict[str, Any]]:
        """执行撤销，返回需要应用的操作描述。"""
        if not self.can_undo():
            return None

        rec = self._undo_stack.pop()
        self._redo_stack.append(rec)

        if rec.action == 'add':
            # 撤销 add -> 移除这些标注
            return {'type': 'remove_batch', 'annotations': rec.annotations}

        if rec.action == 'remove':
            # 撤销 remove -> 把这些标注加回去
            return {'type': 'add_batch', 'annotations': rec.annotations, 'indices': rec.indices}

        if rec.action == 'modify':
            # 撤销 modify -> 回到 prev 状态
            return {'type': 'modify_batch', 'annotations': rec.prev_states}

        return None

    def redo(self) -> Optional[Dict[str, Any]]:
        """执行重做。"""
        if not self.can_redo():
            return None

        rec = self._redo_stack.pop()
        self._undo_stack.append(rec)

        if rec.action == 'add':
            return {'type': 'add_batch', 'annotations': rec.annotations, 'indices': rec.indices}

        if rec.action == 'remove':
            return {'type': 'remove_batch', 'annotations': rec.annotations}

        if rec.action == 'modify':
            return {'type': 'modify_batch', 'annotations': rec.annotations}

        return None


__all__ = ["AnnotationManager"]
