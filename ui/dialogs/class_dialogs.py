from typing import List
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QListWidget, QListWidgetItem, QColorDialog, QMessageBox
)
from PyQt5.QtGui import QColor

from models.class_item import ClassItem
from PyQt5.QtGui import QColor


class ClassEditDialog(QDialog):
    """Simple dialog to add or edit a ClassItem (name + color)."""
    Accepted = QDialog.Accepted

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("编辑类别")
        self.resize(300, 120)

        self._name_edit = QLineEdit()
        self._color_btn = QPushButton("选择颜色")
        self._color_display = QLabel()
        self._color_display.setFixedSize(48, 20)

        self._color = QColor(0, 200, 0)
        self._update_color_display()

        self._color_btn.clicked.connect(self._pick_color)

        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("名称:"))
        row.addWidget(self._name_edit)
        layout.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("颜色:"))
        row2.addWidget(self._color_display)
        row2.addWidget(self._color_btn)
        layout.addLayout(row2)

        btn_row = QHBoxLayout()
        ok = QPushButton("确定")
        cancel = QPushButton("取消")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        btn_row.addWidget(ok)
        btn_row.addWidget(cancel)
        layout.addLayout(btn_row)

    def _pick_color(self):
        c = QColorDialog.getColor(self._color, self, "选择类别颜色")
        if c.isValid():
            self._color = c
            self._update_color_display()

    def _update_color_display(self):
        self._color_display.setStyleSheet(f"background:{self._color.name()};")

    def set_class(self, cls: ClassItem):
        self._name_edit.setText(cls.name)
        try:
            self._color = QColor(cls.color)
            if not self._color.isValid():
                self._color = QColor(0, 200, 0)
        except Exception:
            self._color = QColor(0, 200, 0)
        self._update_color_display()

    def get_class_name(self) -> str:
        return self._name_edit.text().strip()

    def get_color(self) -> str:
        return self._color.name()


class ClassManagerDialog(QDialog):
    """Simple class manager to add/edit/remove classes."""

    def __init__(self, classes: List[ClassItem], parent=None):
        super().__init__(parent)
        self.setWindowTitle("管理类别")
        self.resize(400, 300)

        self._classes: List[ClassItem] = [ClassItem(c.id, c.name, c.color) for c in classes]

        self._list = QListWidget()
        self._refresh_list()

        add_btn = QPushButton("添加")
        edit_btn = QPushButton("编辑")
        del_btn = QPushButton("删除")
        ok_btn = QPushButton("确定")
        cancel_btn = QPushButton("取消")

        add_btn.clicked.connect(self._add)
        edit_btn.clicked.connect(self._edit)
        del_btn.clicked.connect(self._delete)
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._list)

        row = QHBoxLayout()
        row.addWidget(add_btn)
        row.addWidget(edit_btn)
        row.addWidget(del_btn)
        layout.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(ok_btn)
        row2.addWidget(cancel_btn)
        layout.addLayout(row2)

    def _refresh_list(self):
        self._list.clear()
        for c in self._classes:
            item = QListWidgetItem(f"{c.id}: {c.name}")
            item.setData(0, c)
            self._list.addItem(item)

    def _add(self):
        dlg = ClassEditDialog(self)
        if dlg.exec_() == dlg.Accepted:
            name = dlg.get_class_name()
            color = dlg.get_color()
            if name:
                new_id = len(self._classes)
                try:
                    qc = QColor(color)
                    if not qc.isValid():
                        qc = QColor(0, 200, 0)
                except Exception:
                    qc = QColor(0, 200, 0)
                self._classes.append(ClassItem(new_id, name, qc))
                self._refresh_list()
            else:
                QMessageBox.warning(self, "警告", "名称不能为空")

    def _edit(self):
        it = self._list.currentItem()
        if not it:
            return
        cls: ClassItem = it.data(0)
        dlg = ClassEditDialog(self)
        dlg.set_class(cls)
        if dlg.exec_() == dlg.Accepted:
            name = dlg.get_class_name()
            color = dlg.get_color()
            if name:
                cls.name = name
                try:
                    qc = QColor(color)
                    if not qc.isValid():
                        qc = QColor(0, 200, 0)
                except Exception:
                    qc = QColor(0, 200, 0)
                cls.color = qc
                self._refresh_list()
            else:
                QMessageBox.warning(self, "警告", "名称不能为空")

    def _delete(self):
        it = self._list.currentItem()
        if not it:
            return
        cls: ClassItem = it.data(0)
        self._classes = [c for c in self._classes if c.id != cls.id]
        # reassign ids
        for i, c in enumerate(self._classes):
            c.id = i
        self._refresh_list()

    def get_classes(self) -> List[ClassItem]:
        return self._classes
