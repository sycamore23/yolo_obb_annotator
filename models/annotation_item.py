"""
标注项数据模型
"""

import math
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
import uuid
import json

from PyQt5.QtCore import QPoint, QRect, QRectF
from PyQt5.QtGui import QColor, QPolygonF

from .enums import AnnotationMode, BBoxType

@dataclass
class AnnotationItem:
    """标注项"""
    
    # 基本属性
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    bbox_type: BBoxType = BBoxType.HORIZONTAL
    class_id: int = 0
    class_name: str = ""
    confidence: float = 1.0
    visible: bool = True
    locked: bool = False
    selected: bool = False
    
    # 几何属性
    points: List[Tuple[float, float]] = field(default_factory=list)  # 多边形点列表
    rotation: float = 0.0  # 旋转角度（度）
    center: Optional[Tuple[float, float]] = None  # 中心点
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 跟踪信息
    track_id: Optional[int] = None
    frame_id: Optional[int] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.updated_at is None:
            self.updated_at = self.created_at
        
        # 确保points是浮点数
        self.points = [(float(x), float(y)) for x, y in self.points]
    
    def update(self):
        """更新修改时间"""
        self.updated_at = datetime.now()
    
    def get_bbox(self) -> Tuple[float, float, float, float]:
        """获取边界框 (x1, y1, x2, y2)"""
        if not self.points:
            return (0.0, 0.0, 0.0, 0.0)
        
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))

    def get_points(self) -> List[Tuple[float, float]]:
        """返回标注的点列表（每个点为 (x, y)）。"""
        return list(self.points)
    
    def get_center_point(self) -> Tuple[float, float]:
        """获取中心点"""
        if self.center:
            return self.center
        
        if not self.points:
            return (0.0, 0.0)
        
        # 计算多边形中心
        x_sum = sum(p[0] for p in self.points)
        y_sum = sum(p[1] for p in self.points)
        return (x_sum / len(self.points), y_sum / len(self.points))
    
    def get_area(self) -> float:
        """计算面积"""
        if len(self.points) < 3:
            return 0.0
        
        # 使用鞋带公式计算多边形面积
        area = 0.0
        n = len(self.points)
        
        for i in range(n):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        
        return abs(area) / 2.0
    
    def get_perimeter(self) -> float:
        """计算周长"""
        if len(self.points) < 2:
            return 0.0
        
        perimeter = 0.0
        n = len(self.points)
        
        for i in range(n):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % n]
            perimeter += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        return perimeter
    
    def contains_point(self, point: Tuple[float, float], tolerance: float = 5.0) -> bool:
        """检查点是否在标注内"""
        x, y = point
        
        # 检查是否在边界点附近
        for px, py in self.points:
            if math.sqrt((px - x) ** 2 + (py - y) ** 2) <= tolerance:
                return True
        
        # 检查是否在多边形内部
        return self._point_in_polygon(x, y)
    
    def _point_in_polygon(self, x: float, y: float) -> bool:
        """判断点是否在多边形内（射线法）"""
        if len(self.points) < 3:
            return False
        
        inside = False
        n = len(self.points)
        
        p1x, p1y = self.points[0]
        for i in range(1, n + 1):
            p2x, p2y = self.points[i % n]
            
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            
            p1x, p1y = p2x, p2y
        
        return inside
    
    def get_bounding_rect(self) -> QRectF:
        """获取边界矩形"""
        if not self.points:
            return QRectF()
        
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
    
    def get_qpolygon(self) -> QPolygonF:
        """获取Qt多边形对象"""
        polygon = QPolygonF()
        for x, y in self.points:
            polygon.append(QPointF(x, y))
        return polygon
    
    def translate(self, dx: float, dy: float):
        """平移标注"""
        self.points = [(x + dx, y + dy) for x, y in self.points]
        if self.center:
            self.center = (self.center[0] + dx, self.center[1] + dy)
        self.update()
    
    def rotate(self, angle: float, center: Optional[Tuple[float, float]] = None):
        """旋转标注"""
        if not center:
            center = self.get_center_point()
        
        cx, cy = center
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        # 旋转所有点
        rotated_points = []
        for x, y in self.points:
            # 平移到原点
            x -= cx
            y -= cy
            
            # 旋转
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            
            # 平移回原位置
            rotated_points.append((x_rot + cx, y_rot + cy))
        
        self.points = rotated_points
        self.rotation += angle
        self.update()
    
    def scale(self, sx: float, sy: float, center: Optional[Tuple[float, float]] = None):
        """缩放标注"""
        if not center:
            center = self.get_center_point()
        
        cx, cy = center
        
        # 缩放所有点
        scaled_points = []
        for x, y in self.points:
            # 平移到原点
            x -= cx
            y -= cy
            
            # 缩放
            x_scaled = x * sx
            y_scaled = y * sy
            
            # 平移回原位置
            scaled_points.append((x_scaled + cx, y_scaled + cy))
        
        self.points = scaled_points
        self.update()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "bbox_type": self.bbox_type.value,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "visible": self.visible,
            "locked": self.locked,
            "selected": self.selected,
            "points": self.points,
            "rotation": self.rotation,
            "center": self.center,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "track_id": self.track_id,
            "frame_id": self.frame_id,
            "area": self.get_area(),
            "perimeter": self.get_perimeter()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnnotationItem':
        """从字典创建"""
        # 解析时间字符串
        created_at = datetime.fromisoformat(data.get('created_at', datetime.now().isoformat()))
        updated_at = datetime.fromisoformat(data.get('updated_at', created_at.isoformat()))
        
        # 解析bbox_type
        bbox_type = BBoxType(data.get('bbox_type', 'horizontal'))
        
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            bbox_type=bbox_type,
            class_id=data.get('class_id', 0),
            class_name=data.get('class_name', ''),
            confidence=data.get('confidence', 1.0),
            visible=data.get('visible', True),
            locked=data.get('locked', False),
            selected=data.get('selected', False),
            points=data.get('points', []),
            rotation=data.get('rotation', 0.0),
            center=data.get('center'),
            created_at=created_at,
            updated_at=updated_at,
            metadata=data.get('metadata', {}),
            track_id=data.get('track_id'),
            frame_id=data.get('frame_id')
        )
    
    def to_yolo_format(self, img_width: int, img_height: int) -> str:
        """转换为YOLO格式字符串"""
        if self.bbox_type == BBoxType.HORIZONTAL:
            # 水平框: class_id x_center y_center width height
            x1, y1, x2, y2 = self.get_bbox()
            
            x_center = (x1 + x2) / 2 / img_width
            y_center = (y1 + y2) / 2 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            return f"{self.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        
        elif self.bbox_type == BBoxType.ROTATED:
            # 旋转框: class_id x1 y1 x2 y2 x3 y3 x4 y4
            # 计算旋转矩形的四个角点
            center_x, center_y = self.get_center_point()
            x1, y1, x2, y2 = self.get_bbox()
            width = x2 - x1
            height = y2 - y1
            
            # 计算旋转后的四个角点
            points = [
                (-width/2, -height/2),
                (width/2, -height/2),
                (width/2, height/2),
                (-width/2, height/2)
            ]
            
            rad = math.radians(self.rotation)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            
            rotated_points = []
            for x, y in points:
                x_rot = x * cos_a - y * sin_a + center_x
                y_rot = x * sin_a + y * cos_a + center_y
                rotated_points.extend([x_rot / img_width, y_rot / img_height])
            
            points_str = ' '.join(f'{p:.6f}' for p in rotated_points)
            return f"{self.class_id} {points_str}"
        
        elif self.bbox_type == BBoxType.POLYGON:
            # 多边形: class_id x1 y1 x2 y2 ... xn yn
            points_norm = []
            for x, y in self.points:
                points_norm.extend([x / img_width, y / img_height])
            
            points_str = ' '.join(f'{p:.6f}' for p in points_norm)
            return f"{self.class_id} {points_str}"
        
        else:
            raise ValueError(f"Unsupported bbox_type: {self.bbox_type}")
    
    def copy(self) -> 'AnnotationItem':
        """创建副本"""
        return AnnotationItem.from_dict(self.to_dict())
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"Annotation(id={self.id}, class={self.class_name}, points={len(self.points)}, area={self.get_area():.1f})"
    
    def __repr__(self) -> str:
        """详细表示"""
        return f"AnnotationItem({self.to_dict()})"