"""
类别项数据模型
"""

from dataclasses import dataclass, field
from typing import Dict, Any
from PyQt5.QtGui import QColor
import numpy as np

@dataclass
class ClassItem:
    """类别项"""
    id: int = 0
    name: str = ""
    color: QColor = field(default_factory=lambda: QColor(
        np.random.randint(50, 255),
        np.random.randint(50, 255),
        np.random.randint(50, 255)
    ))
    visible: bool = True
    count: int = 0
    shortcut: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color.name(),
            "count": self.count,
            "shortcut": self.shortcut
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClassItem':
        """从字典创建类别项"""
        color = QColor(data.get('color', '#FFFFFF'))
        return cls(
            id=data.get('id', 0),
            name=data.get('name', ''),
            color=color,
            visible=data.get('visible', True),
            count=data.get('count', 0),
            shortcut=data.get('shortcut', '')
        )