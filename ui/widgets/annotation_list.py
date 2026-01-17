from typing import List
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QPushButton, QHBoxLayout, QAbstractItemView
from PyQt5.QtCore import pyqtSignal, Qt

from models.annotation_item import AnnotationItem


class AnnotationListWidget(QWidget):
    annotation_selected = pyqtSignal(str)
    annotation_deleted = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._list = QListWidget()
        # Allow multi-selection so users can copy multiple annotations
        self._list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._list.itemClicked.connect(self._on_item_clicked)

        del_btn = QPushButton("删除")
        del_btn.clicked.connect(self._on_delete_clicked)

        layout = QVBoxLayout(self)
        layout.addWidget(self._list)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(del_btn)
        layout.addLayout(btn_row)

        self._id_map = {}

    def set_annotations(self, anns: List[AnnotationItem]):
        self._list.clear()
        self._id_map.clear()
        for ann in anns:
            text = f"[{ann.class_name}] {ann.id}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, ann.id)
            self._list.addItem(item)
            self._id_map[ann.id] = ann

    def _on_item_clicked(self, item: QListWidgetItem):
        aid = item.data(Qt.UserRole)
        if aid:
            self.annotation_selected.emit(aid)

    def get_selected_annotation_ids(self):
        """返回当前选中的标注 id 列表（顺序与列表中显示一致）"""
        ids = []
        for it in self._list.selectedItems():
            aid = it.data(Qt.UserRole)
            if aid:
                ids.append(aid)
        return ids

    def _on_delete_clicked(self):
        item = self._list.currentItem()
        if not item:
            return
        aid = item.data(Qt.UserRole)
        if aid:
            self.annotation_deleted.emit(aid)

    def select_annotation(self, annotation_id: str):
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.UserRole) == annotation_id:
                self._list.setCurrentItem(item)
                break
