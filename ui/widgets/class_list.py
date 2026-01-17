"""
类别列表控件 - 简化实现，提供基本的添加/移除与选择信号。
"""

from typing import List, Optional
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QListWidget, QListWidgetItem
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor

from models.class_item import ClassItem


class ClassListWidget(QWidget):
    """显示类别的简单列表控件并提供选择信号。

    信号:
        class_selected(ClassItem)
    """

    class_selected = pyqtSignal(object)
    class_double_clicked = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._classes: List[ClassItem] = []

        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_current_changed)
        self._list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._list.itemClicked.connect(self._on_item_single_clicked)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._list)

    def _on_item_double_clicked(self, item: QListWidgetItem):
        data = item.data(32)
        if isinstance(data, ClassItem):
            self.class_double_clicked.emit(data)

    def _on_item_single_clicked(self, item: QListWidgetItem):
        # Allow re-selecting the same item to apply it (e.g. to a selected annotation)
        data = item.data(32)
        if isinstance(data, ClassItem):
            self.class_selected.emit(data)

    def _on_current_changed(self, current: QListWidgetItem, previous: QListWidgetItem):
        if current is None:
            return
        data = current.data(32)  # Qt.UserRole
        if isinstance(data, ClassItem):
            self.class_selected.emit(data)

    def set_classes(self, classes: List[ClassItem]):
        """用给定的类别列表填充控件。"""
        self._classes = list(classes)
        self._list.clear()
        for c in self._classes:
            item = QListWidgetItem(c.name)
            # 保存对象以便选择时使用
            item.setData(32, c)
            # 设置前景颜色为类别颜色
            try:
                item.setForeground(c.color)
            except Exception:
                pass
            self._list.addItem(item)

    def add_class(self, class_item: ClassItem):
        self._classes.append(class_item)
        item = QListWidgetItem(class_item.name)
        item.setData(32, class_item)
        try:
            item.setForeground(class_item.color)
        except Exception:
            pass
        self._list.addItem(item)

    def remove_selected(self):
        idx = self._list.currentRow()
        if idx >= 0:
            self._list.takeItem(idx)
            try:
                self._classes.pop(idx)
            except Exception:
                pass

    def get_selected(self) -> Optional[ClassItem]:
        item = self._list.currentItem()
        if not item:
            return None
        data = item.data(32)
        if isinstance(data, ClassItem):
            return data
        return None

    # Compatibility helpers expected by main_window
    def get_selected_class(self) -> Optional[ClassItem]:
        return self.get_selected()

    def update_class(self, class_item: ClassItem):
        # find and update the corresponding list item
        for i in range(self._list.count()):
            item = self._list.item(i)
            data = item.data(32)
            if isinstance(data, ClassItem) and data.id == class_item.id:
                item.setText(class_item.name)
                item.setData(32, class_item)
                try:
                    item.setForeground(class_item.color)
                except Exception:
                    pass
                break


__all__ = ["ClassListWidget"]
