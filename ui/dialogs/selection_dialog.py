from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLineEdit, QListWidget, 
                             QDialogButtonBox, QAbstractItemView, QListWidgetItem)
from PyQt5.QtCore import Qt, pyqtSignal

class LabelSelectDialog(QDialog):
    def __init__(self, parent=None, items=None, title="选择标签", multi: bool = False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedWidth(300)
        self.items = items or []
        self.multi = multi

        layout = QVBoxLayout(self)

        # 搜索框
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("输入首字母或名称检索...")
        self.search_edit.textChanged.connect(self.filter_items)
        self.search_edit.textEdited.connect(self.filter_items)
        layout.addWidget(self.search_edit)

        # 列表框
        self.list_widget = QListWidget()
        self.list_widget.addItems(self.items)
        if self.multi:
            self.list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        else:
            self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
            self.list_widget.itemDoubleClicked.connect(self.accept)
            self.list_widget.itemActivated.connect(self.accept)
        layout.addWidget(self.list_widget)

        # 按钮
        flags = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.button_box = QDialogButtonBox(flags)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        # 默认选中第一项
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

        self.search_edit.setFocus()

        # 安装事件过滤器以处理上下键
        self.search_edit.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj == self.search_edit and event.type() == event.KeyPress:
            if event.key() == Qt.Key_Down:
                row = self.list_widget.currentRow()
                if row < self.list_widget.count() - 1:
                    self.list_widget.setCurrentRow(row + 1)
                return True
            elif event.key() == Qt.Key_Up:
                row = self.list_widget.currentRow()
                if row > 0:
                    self.list_widget.setCurrentRow(row - 1)
                return True
            elif event.key() == Qt.Key_Escape:
                self.reject()
                return True
            elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                self.accept()
                return True
        return super().eventFilter(obj, event)

    def filter_items(self, text):
        self.list_widget.clear()
        
        # 始终添加标准选项
        current_items = []
        
        # 匹配逻辑
        text_lower = text.lower().strip() if text else ""
        
        exact_match = False
        
        if not text_lower:
            current_items = list(self.items)
        else:
            # 1. 完全匹配/前缀匹配
            for item in self.items:
                if item == "<新建标签>": continue
                if item.lower().startswith(text_lower):
                    current_items.append(item)
                    if item.lower() == text_lower:
                        exact_match = True
            
            # 2. 包含匹配 (如果不重复)
            for item in self.items:
                if item == "<新建标签>": continue
                if text_lower in item.lower() and item not in current_items:
                    current_items.append(item)
        
        # 添加到列表控件
        # 如果没有完全匹配且有输入，添加"新建"选项作为第一项
        if text_lower and not exact_match:
            name_to_create = text.strip()
            item = QListWidgetItem(f"新建: {name_to_create}")
            item.setData(Qt.UserRole, ('NEW', name_to_create))
            self.list_widget.addItem(item)
            
        for name in current_items:
            if name == "<新建标签>": continue
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, ('EXISTING', name))
            self.list_widget.addItem(item)
            
        # 总是保留原始"<新建标签>"作为最后的fallback
        item = QListWidgetItem("<新建标签>")
        item.setData(Qt.UserRole, ('MANUAL_NEW', None))
        self.list_widget.addItem(item)
            
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    def get_selected_result(self):
        """返回选中结果。

        - 当 `multi=False` 时，返回 (action, value) 与旧行为一致。
        - 当 `multi=True` 时，返回列表形式：[ (action, value), ... ]
        """
        if not self.multi:
            if self.list_widget.currentItem():
                data = self.list_widget.currentItem().data(Qt.UserRole)
                if data:
                    return data
                text = self.list_widget.currentItem().text()
                if text == "<新建标签>":
                    return ('MANUAL_NEW', None)
                return ('EXISTING', text)
            return None

        # multi selection: return list of data tuples
        selected = self.list_widget.selectedItems()
        results = []
        for it in selected:
            data = it.data(Qt.UserRole)
            if data:
                results.append(data)
            else:
                text = it.text()
                if text == "<新建标签>":
                    results.append(('MANUAL_NEW', None))
                else:
                    results.append(('EXISTING', text))
        return results
