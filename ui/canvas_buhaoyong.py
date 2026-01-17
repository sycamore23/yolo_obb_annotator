import math
from typing import List, Optional
from PyQt5.QtWidgets import QWidget, QMenu
from PyQt5.QtCore import Qt, QPoint, QRect, QRectF, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont, QImage, QPixmap, QPainterPath
import cv2
import numpy as np
import logging

_logger = logging.getLogger(__name__)

from models.annotation_item import AnnotationItem
from models.class_item import ClassItem
from models.enums import AnnotationMode, EditMode, BBoxType

class ImageCanvas(QWidget):
    """图像画布控件"""
    
    # 信号定义
    annotation_added = pyqtSignal(AnnotationItem)
    annotation_updated = pyqtSignal(AnnotationItem)
    annotation_removed = pyqtSignal(int)
    annotation_selected = pyqtSignal(object)
    mouse_position_changed = pyqtSignal(int, int)
    mode_changed = pyqtSignal(object)
    
    def __init__(self, parent=None):
        """
        初始化图像画布
        
        Args:
            parent: 父控件
        """
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setFocusPolicy(Qt.StrongFocus)
        
        # 图像相关
        self.image: Optional[np.ndarray] = None
        self.pixmap: Optional[QPixmap] = None
        self.scale_factor: float = 1.0
        self.offset: QPoint = QPoint(0, 0)
        self.panning: bool = False
        self.pan_start: QPoint = QPoint()
        
        # 标注相关
        self.annotations: List[AnnotationItem] = []
        self.classes: List[ClassItem] = []
        self.class_colors: dict = {}
        
        # 状态
        self.annotation_mode: AnnotationMode = AnnotationMode.NONE
        self.edit_mode: EditMode = EditMode.NONE
        self.drawing: bool = False
        self.editing: bool = False
        self.show_bbox_info: bool = True
        self.show_class_names: bool = True
        self.show_grid: bool = False
        self.grid_size: int = 50
        self.snap_to_grid: bool = False
        # 修复：添加缺失的属性
        self.show_confidence: bool = True
        self.show_statistics: bool = False
        self.current_class: Optional[ClassItem] = None
        
        # 当前操作
        self.current_bbox: List[float] = []
        self.current_points: List[QPoint] = []
        self.start_point: Optional[QPoint] = None
        self.selected_annotation: Optional[AnnotationItem] = None
        self.hovered_annotation: Optional[AnnotationItem] = None
        # 编辑相关
        self._edit_handle_index: Optional[int] = None
        self._edit_start_point: Optional[QPoint] = None
        self._original_points: Optional[List[tuple]] = None
        self._original_rotation: float = 0.0
        # 临时预览点（多边形绘制时随鼠标移动）
        self._preview_point: Optional[QPoint] = None
        
        # 设置鼠标跟踪
        self.setMouseTracking(True)
        
        # 初始化样式
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
            }
        """)

        # 颜色管理
        self.class_colors: dict = {}
        self._default_colors = [
            QColor(255, 0, 0),    # 红色
            QColor(0, 255, 0),    # 绿色
            QColor(0, 0, 255),    # 蓝色
            QColor(255, 255, 0),  # 黄色
            QColor(255, 0, 255),  # 洋红
            QColor(0, 255, 255),  # 青色
            QColor(255, 128, 0),  # 橙色
            QColor(128, 0, 255),  # 紫色
            QColor(0, 255, 128),  # 青色绿
            QColor(255, 0, 128),  # 粉红
        ]
        self._next_color_index = 0

    def _get_class_color(self, class_id: int) -> QColor:
        """获取类别颜色，如果不存在则分配一个新颜色"""
        if class_id not in self.class_colors:
            # 从预设颜色列表中获取颜色
            color_index = self._next_color_index % len(self._default_colors)
            # QColor 没有 copy() 方法，直接使用即可
            self.class_colors[class_id] = self._default_colors[color_index]
            self._next_color_index += 1
        return self.class_colors[class_id]

    def set_classes(self, classes: List[ClassItem]):
        """设置类别列表并分配颜色"""
        self.classes = classes
        
        # 为每个类别分配颜色
        for class_item in classes:
            self._get_class_color(class_item.id)  # 确保颜色已分配
        
        self.update()

    def add_annotation(self, annotation: AnnotationItem):
        """添加标注时确保类别颜色已分配"""
        try:
            # 确保颜色已分配
            self._get_class_color(annotation.class_id)
            
            self.annotations.append(annotation)
            self.annotation_added.emit(annotation)
            self.update()
        except Exception as e:
            _logger.error(f"添加标注失败: {e}")

    def set_image(self, image: np.ndarray):
        """
        设置图像
        
        Args:
            image: OpenCV图像 (BGR格式)
        """
        self.image = image
        if image is not None:
            height, width, channel = image.shape
            
            # 转换图像格式
            if channel == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif channel == 4:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # 创建QImage
            bytes_per_line = 3 * width
            qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(qimage)
            
            # 调整显示
            self.fit_to_view()
        
        self.update()
    
    def fit_to_view(self):
        """适应视图"""
        if self.pixmap and not self.pixmap.isNull():
            pixmap_size = self.pixmap.size()
            view_size = self.size()
            
            if pixmap_size.width() > 0 and pixmap_size.height() > 0:
                scale_w = view_size.width() / pixmap_size.width()
                scale_h = view_size.height() / pixmap_size.height()
                self.scale_factor = min(scale_w, scale_h) * 0.95
                
                # 居中显示
                scaled_width = pixmap_size.width() * self.scale_factor
                scaled_height = pixmap_size.height() * self.scale_factor
                ox = int((view_size.width() - scaled_width) / 2)
                oy = int((view_size.height() - scaled_height) / 2)
                self.offset = QPoint(ox, oy)
    
    def zoom_in(self):
        """放大"""
        self.scale_factor *= 1.2
        self.scale_factor = min(10.0, self.scale_factor)
        self.update()

    def zoom_out(self):
        """缩小"""
        self.scale_factor /= 1.2
        self.scale_factor = max(0.01, self.scale_factor)
        self.update()
    
    def zoom_actual(self):
        """实际大小"""
        if self.pixmap:
            self.scale_factor = 1.0
            self.offset = QPoint(0, 0)
            self.update()
    
    def window_to_image(self, window_point: QPoint) -> Optional[QPoint]:
        """
        窗口坐标转换为图像坐标
        
        Args:
            window_point: 窗口坐标
            
        Returns:
            图像坐标或None
        """
        if not self.pixmap or self.pixmap.isNull():
            return None
        
        # 转换为相对于图像的位置
        x = (window_point.x() - self.offset.x()) / self.scale_factor
        y = (window_point.y() - self.offset.y()) / self.scale_factor
        
        # 检查是否在图像范围内
        if (0 <= x < self.pixmap.width() and 
            0 <= y < self.pixmap.height()):
            return QPoint(int(x), int(y))
        return None
    
    def image_to_window(self, image_point: QPoint) -> Optional[QPoint]:
        """
        图像坐标转换为窗口坐标
        
        Args:
            image_point: 图像坐标
            
        Returns:
            窗口坐标或None
        """
        if not self.pixmap or self.pixmap.isNull():
            return None
        
        x = image_point.x() * self.scale_factor + self.offset.x()
        y = image_point.y() * self.scale_factor + self.offset.y()
        return QPoint(int(x), int(y))
    
    def mousePressEvent(self, event):
        """鼠标按下事件 - 修复可能的递归问题"""
        self.setFocus()
        img_pos = self.window_to_image(event.pos())
        
        if event.button() == Qt.LeftButton:
            if not img_pos:
                return

            # ==========================================================
            # 规则1: 点击标注框角点或旋转柄自动切换到选择模式
            # ==========================================================
            # 检测是否点击了选中标注的控制手柄
            if self.selected_annotation:
                handle = self._find_handle_at(img_pos)
                if handle is not None:
                    # 即使当前是绘制模式，也切换回选择模式进行编辑
                    if self.annotation_mode != AnnotationMode.NONE:
                        # 直接设置模式，不发射信号（因为是由用户操作触发的）
                        self.annotation_mode = AnnotationMode.NONE
                        # 这里不发射信号，避免递归
                        # 如果主窗口需要知道这个变化，可以通过其他方式
                        
                        # 清除绘制状态
                        self.drawing = False
                        self.current_bbox = []
                        self.current_points = []
                        self.start_point = None
                        self._preview_point = None
                    
                    # 进入编辑状态
                    if handle == -1:
                        # 旋转
                        self.editing = True
                        self.edit_mode = EditMode.ROTATE
                        self._edit_handle_index = None
                        self._edit_start_point = img_pos
                        self._original_points = list(self.selected_annotation.get_points())
                        self._original_rotation = getattr(self.selected_annotation, 'rotation', 0.0)
                        
                        # 计算旋转中心
                        pts = self.selected_annotation.get_points()
                        if len(pts) == 4:
                            cx = sum(p[0] for p in pts) / len(pts)
                            cy = sum(p[1] for p in pts) / len(pts)
                            self._rotation_center = QPoint(int(cx), int(cy))
                    else:
                        # 缩放
                        self.editing = True
                        self.edit_mode = EditMode.RESIZE
                        self._edit_handle_index = handle
                        self._edit_start_point = img_pos
                        self._original_points = list(self.selected_annotation.get_points())
                    self.update()
                    return

            # ==========================================================
            # 分支处理：绘制模式 vs 选择模式
            # ==========================================================
            if self.annotation_mode != AnnotationMode.NONE:
                # --- 绘制模式 ---
                
                # 如果是多边形模式，且已经开始绘制，继续添加点
                if self.annotation_mode == AnnotationMode.POLYGON and self.drawing:
                    self.current_points.append(img_pos)
                    self.update()
                    return

                # 开始新绘制
                if self.annotation_mode in (AnnotationMode.HORIZONTAL, AnnotationMode.ROTATED):
                    if not self.drawing:
                        self.start_point = img_pos
                        self.current_bbox = [img_pos.x(), img_pos.y(), img_pos.x(), img_pos.y()]
                        self.drawing = True
                    else:
                        # 完成绘制 (通常是再次点击)
                        self.finish_annotation()

                elif self.annotation_mode == AnnotationMode.POLYGON:
                    if not self.drawing:
                        self.start_point = img_pos
                        self.current_points = [img_pos]
                        self.drawing = True
            
            else:
                # --- 选择模式 (AnnotationMode.NONE) ---
                
                # 检查是否点击了某个标注 (用于选择/移动)
                clicked_ann = self._get_annotation_at(img_pos)
                
                if clicked_ann:
                    # 点击在标注上 -> 选择 或 移动
                    
                    # 1. 选中该标注 (如果未选)
                    if self.selected_annotation != clicked_ann:
                        self.select_annotation(clicked_ann)
                    
                    # 2. 准备移动 (Move)
                    self.editing = True
                    self.edit_mode = EditMode.MOVE
                    self._edit_start_point = img_pos
                    self._original_points = list(clicked_ann.get_points())
                
                else:
                    # ==========================================================
                    # 规则2: 如果在选择模式按住鼠标左键拖动自动切换到旋转框模式 (即点击空白处)
                    # ==========================================================
                    
                    # 切换到旋转框模式
                    old_mode = self.annotation_mode
                    self.annotation_mode = AnnotationMode.ROTATED
                    
                    # 只有当模式改变时才发射信号
                    if old_mode != self.annotation_mode:
                        self.mode_changed.emit(self.annotation_mode)
                    
                    # 立即开始绘制
                    self.start_point = img_pos
                    self.current_bbox = [img_pos.x(), img_pos.y(), img_pos.x(), img_pos.y()]
                    self.drawing = True
                    
                    # 清除当前选择
                    if self.selected_annotation is not None:
                        self.select_annotation(None)
        
        elif event.button() == Qt.MiddleButton:
            # 中键拖拽
            self.panning = True
            self.pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        
        self.update()
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        img_pos = self.window_to_image(event.pos())
        
        if img_pos:
            # 发射鼠标位置信号
            self.mouse_position_changed.emit(img_pos.x(), img_pos.y())
        
        # 1. 处理编辑状态
        if getattr(self, 'editing', False) and img_pos:
            if self.edit_mode == EditMode.MOVE:
                # 移动标注
                dx = img_pos.x() - self._edit_start_point.x()
                dy = img_pos.y() - self._edit_start_point.y()
                
                new_points = []
                for x, y in self._original_points:
                    new_points.append((x + dx, y + dy))
                
                self.selected_annotation.points = new_points
                self.annotation_updated.emit(self.selected_annotation)
                

            elif self.edit_mode == EditMode.RESIZE:
                # 调整大小
                # 更稳健的逻辑：基于原对角关系调整
                # 假设点顺序是 [TL, TR, BR, BL] (0,1,2,3)
                # 0 <-> 2, 1 <-> 3 是对角
                if self.selected_annotation and 0 <= self._edit_handle_index < 4:
                    opp_idx = (self._edit_handle_index + 2) % 4
                    opp_pt = self._original_points[opp_idx] # 固定点 (pivot)
                    
                    # 当前鼠标位置
                    curr_pt = (img_pos.x(), img_pos.y())
                    
                    # 计算新的中心和尺寸
                    # 注意：这改变了形状，可能变成任意四边形。
                    # 严格的 OBB Resizing 需要保持矩形约束。
                    # 简单策略：仅移动顶点会导致非矩形。
                    
                    # -----------------------------------------------------------------
                    # OBB Resizing Logic:
                    # 1. Identify the fixed pivot point (opposite to the dragged handle).
                    # 2. Identify the two axes directions from the pivot along adjacent edges.
                    # 3. Project the vector (pivot -> mouse) onto these two axes to determine new dimensions.
                    # 4. Reconstruct the rectangle corners.
                    # -----------------------------------------------------------------
                    
                    def vec(p1, p2): return (p2[0]-p1[0], p2[1]-p1[1])
                    def dot(v1, v2): return v1[0]*v2[0] + v1[1]*v2[1]
                    def add(p, v): return (p[0]+v[0], p[1]+v[1])
                    def mag_sq(v): return v[0]**2 + v[1]**2
                    
                    # Calculate previous and next vertex indices for axes definition
                    prev_idx = (self._edit_handle_index - 1) % 4
                    next_idx = (self._edit_handle_index + 1) % 4
                    
                    # Get original points for reference directions
                    orig_prev = self._original_points[prev_idx]
                    orig_next = self._original_points[next_idx]
                    
                    # Vectors representing the original box axes from the pivot
                    v_prev = vec(opp_pt, orig_prev)
                    v_next = vec(opp_pt, orig_next)
                    
                    # Vector from pivot to current mouse position
                    v_mouse = vec(opp_pt, curr_pt)
                    
                    # Project v_mouse onto the axes defined by v_prev and v_next
                    len_prev_sq = mag_sq(v_prev)
                    len_next_sq = mag_sq(v_next)
                    
                    # Calculate new edge vectors based on projection
                    if len_prev_sq > 1e-6:
                            scale_prev = dot(v_mouse, v_prev) / len_prev_sq
                            new_v_prev = (v_prev[0] * scale_prev, v_prev[1] * scale_prev)
                    else:
                            new_v_prev = (0, 0)
                    
                    if len_next_sq > 1e-6:
                            scale_next = dot(v_mouse, v_next) / len_next_sq
                            new_v_next = (v_next[0] * scale_next, v_next[1] * scale_next)
                    else:
                            new_v_next = (0, 0)
                            
                    # Reconstruct the 4 corners
                    # The sequence is: Pivot, Pivot + NewV_Prev, Pivot + NewV_Prev + NewV_Next, Pivot + NewV_Next
                    # We need to map these back to the original indices
                    
                    p_opp = opp_pt
                    p_prev = add(p_opp, new_v_prev)
                    p_next = add(p_opp, new_v_next)
                    p_curr = add(p_opp, add(new_v_prev, new_v_next))
                    
                    new_pts = [None] * 4
                    new_pts[opp_idx] = p_opp
                    new_pts[prev_idx] = p_prev
                    new_pts[next_idx] = p_next
                    new_pts[self._edit_handle_index] = p_curr
                    
                    self.selected_annotation.points = list(new_pts)

                    self.annotation_updated.emit(self.selected_annotation)

            elif self.edit_mode == EditMode.ROTATE:
                # 旋转
                if self.selected_annotation:
                    # 计算相对于中心的角度变化
                    cx = sum(p[0] for p in self._original_points) / len(self._original_points)
                    cy = sum(p[1] for p in self._original_points) / len(self._original_points)
                    
                    import math
                    # 初始角度
                    dx1 = self._edit_start_point.x() - cx
                    dy1 = self._edit_start_point.y() - cy
                    angle1 = math.atan2(dy1, dx1)
                    
                    # 当前角度
                    dx2 = img_pos.x() - cx
                    dy2 = img_pos.y() - cy
                    angle2 = math.atan2(dy2, dx2)
                    
                    delta_angle = math.degrees(angle2 - angle1)
                    new_rotation = self._original_rotation + delta_angle
                    
                    # 更新旋转
                    self.selected_annotation.rotation = new_rotation
                    
                    # 重新计算点位置
                    rad = math.radians(delta_angle)
                    cos_a = math.cos(rad)
                    sin_a = math.sin(rad)
                    
                    new_points = []
                    for x, y in self._original_points:
                        nx = (x - cx) * cos_a - (y - cy) * sin_a + cx
                        ny = (x - cx) * sin_a + (y - cy) * cos_a + cy
                        new_points.append((nx, ny))
                    self.selected_annotation.points = new_points
                    self.annotation_updated.emit(self.selected_annotation)
            
            self.update()
            return

        # 2. 处理绘制状态
        if self.drawing and img_pos:
            if self.annotation_mode in (AnnotationMode.HORIZONTAL, AnnotationMode.ROTATED):
                # 更新当前 bbox 的结束坐标
                try:
                    self.current_bbox[2] = img_pos.x()
                    self.current_bbox[3] = img_pos.y()
                    self._preview_point = None
                except Exception:
                    # 防止未初始化
                    if self.start_point:
                        self.current_bbox = [self.start_point.x(), self.start_point.y(), img_pos.x(), img_pos.y()]
            
            elif self.annotation_mode == AnnotationMode.POLYGON:
                self._preview_point = img_pos
                
            self.update()
            return

        # 3. 处理悬停/光标状态 (Selection Mode only typically)
        if not self.drawing and not self.editing and img_pos:
            # 检查手柄
            if self.selected_annotation:
                handle = self._find_handle_at(img_pos)
                if handle is not None:
                    if handle == -1:
                        self.setCursor(Qt.PointingHandCursor) # Rotate
                    else:
                        self.setCursor(Qt.SizeAllCursor) # Resize
                    self.update()
                    return

            # 检查标注体
            ann = self._get_annotation_at(img_pos)
            if ann:
                self.setCursor(Qt.OpenHandCursor) # Move candidate
            else:
                self.setCursor(Qt.ArrowCursor)
            
            # 高亮光标下的标注 (Optional)
            # if ann != self.hovered_annotation:
            #     self.hovered_annotation = ann
            #     self.update()

        elif event.button() == Qt.MiddleButton:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)
        
        self.update()
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton:
            if self.panning:
                self.panning = False
                self.setCursor(Qt.ArrowCursor)

            if getattr(self, 'editing', False):
                self.editing = False
                self.edit_mode = EditMode.NONE
                self._edit_handle_index = None
                self._edit_start_point = None
                self._original_points = None
                
                # 发送最终更新信号
                if self.selected_annotation:
                    self.annotation_updated.emit(self.selected_annotation)
        
        elif event.button() == Qt.MiddleButton:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)
        
        self.update()
    
        if self.annotation_mode == AnnotationMode.POLYGON:
                self.finish_polygon()

    def keyPressEvent(self, event):
        """键盘按键事件"""
        if event.key() == Qt.Key_Escape:
            # 取消当前操作
            if self.drawing:
                self.drawing = False
                self.current_bbox = []
                self.current_points = []
                self.start_point = None
                self._preview_point = None
                self.update()
            
            elif self.editing:
                # 恢复原来的状态
                self.editing = False
                self.edit_mode = EditMode.NONE
                self._edit_handle_index = None
                self._edit_start_point = None
                if self.selected_annotation and self._original_points:
                    self.selected_annotation.points = self._original_points
                    if self._original_rotation is not None:
                        self.selected_annotation.rotation = self._original_rotation
                    self.annotation_updated.emit(self.selected_annotation)
                self._original_points = None
                self.update()
            
            elif self.annotation_mode != AnnotationMode.NONE:
                # 切换回选择模式
                self.annotation_mode = AnnotationMode.NONE
                self.mode_changed.emit(self.annotation_mode)
                self.update()

    
    def wheelEvent(self, event):
        """滚轮事件"""
        img_pos = self.window_to_image(event.pos())
        old_scale = self.scale_factor
        
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9
        self.scale_factor *= zoom_factor
        self.scale_factor = max(0.01, min(10.0, self.scale_factor))
        
        # 调整偏移以保持鼠标位置不变
        if old_scale != self.scale_factor and img_pos:
            rel_x = (event.pos().x() - self.offset.x()) / old_scale
            rel_y = (event.pos().y() - self.offset.y()) / old_scale
            
            self.offset.setX(int(event.pos().x() - rel_x * self.scale_factor))
            self.offset.setY(int(event.pos().y() - rel_y * self.scale_factor))
        
        self.update()
    
    def finish_annotation(self):
        """完成标注"""
        if self.current_bbox:
            # 创建标注项
            bbox_type = BBoxType.HORIZONTAL if self.annotation_mode == AnnotationMode.HORIZONTAL else BBoxType.ROTATED
            
            # 确保坐标顺序正确
            if bbox_type == BBoxType.HORIZONTAL and len(self.current_bbox) == 4:
                x1, y1, x2, y2 = self.current_bbox
                x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
                self.current_bbox = [x1, y1, x2, y2]
            
            # 不再自动设置默认类别，以便触发主窗口的打标对话框
            current_class = None
            
            # convert bbox to polygon points (rectangle) for AnnotationItem.points
            pts = []
            if len(self.current_bbox) == 4:
                x1, y1, x2, y2 = self.current_bbox
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                if self.annotation_mode == AnnotationMode.ROTATED:
                    # compute center, width, height, and rotation angle based on drag vector
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    w = abs(x2 - x1)
                    h = abs(y2 - y1)
                    # angle: use vector from start_point to end to get orientation
                    try:
                        import math
                        if self.start_point:
                            ang = math.atan2((self.current_bbox[3] - self.start_point.y()), (self.current_bbox[2] - self.start_point.x()))
                        else:
                            ang = 0.0
                        rot_deg = math.degrees(ang)
                    except Exception:
                        rot_deg = 0.0

                    # compute four corners relative to center and rotate
                    hw = w / 2.0
                    hh = h / 2.0
                    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
                    pts = []
                    try:
                        import math
                        rad = math.radians(rot_deg)
                        cos_a = math.cos(rad)
                        sin_a = math.sin(rad)
                        for ox, oy in corners:
                            rx = ox * cos_a - oy * sin_a + cx
                            ry = ox * sin_a + oy * cos_a + cy
                            pts.append((float(rx), float(ry)))
                    except Exception:
                        pts = [(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)]
                    rotation_val = rot_deg
                else:
                    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            else:
                # fallback: empty points
                pts = []
            annotation = AnnotationItem(
                points=pts,
                bbox_type=bbox_type,
                class_id=current_class.id if current_class else 0,
                class_name=current_class.name if current_class else "",
                rotation=(rotation_val if 'rotation_val' in locals() else 0.0)
            )
            
            self.annotations.append(annotation)
            self.annotation_added.emit(annotation)
            
            # 重置状态
            self.drawing = False
            self.current_bbox = []
            self.start_point = None
            self.selected_annotation = annotation
            annotation.selected = True
            # 清除预览点
            self._preview_point = None
    
    def finish_polygon(self):
        """完成多边形标注"""
        if len(self.current_points) >= 3:
            # 展平点列表
            points = []
            for point in self.current_points:
                points.append((float(point.x()), float(point.y())))
            
            # 不再自动设置默认类别，以便触发主窗口的打标对话框
            current_class = None
            
            annotation = AnnotationItem(
                points=points,
                bbox_type=BBoxType.POLYGON,
                class_id=current_class.id if current_class else 0,
                class_name=current_class.name if current_class else ""
            )
            
            self.annotations.append(annotation)
            self.annotation_added.emit(annotation)
            
            # 重置状态
            self.drawing = False
            self.current_points = []
            self.start_point = None
            self.selected_annotation = annotation
            # 清除预览点
            self._preview_point = None
    
    def _get_annotation_at(self, position: QPoint) -> Optional[AnnotationItem]:
        """获取指定位置的标注（不改变选择状态）"""
        for annotation in reversed(self.annotations):
            try:
                if annotation.visible and annotation.contains_point((position.x(), position.y())):
                    return annotation
            except Exception:
                continue
        return None

    def select_annotation_at(self, position: QPoint):
        """
        选择指定位置的标注
        
        Args:
            position: 选择位置
        """
        annotation = self._get_annotation_at(position)
        
        if annotation:
            if self.selected_annotation != annotation:
                self.select_annotation(annotation)
        else:
            # 如果点击了空白处，取消所有选择
            self.select_annotation(None)

        # Legacy implementation replaced by _get_annotation_at usage
        return

    def _find_handle_at(self, pos: QPoint, tol: int = 6):
        """检测给定图像坐标是否在当前选中标注的角点附近，返回角点索引或 None。"""
        try:
            if not self.selected_annotation:
                return None
            
            # 获取缩放后的容差
            scaled_tol = tol / self.scale_factor
            
            pts = self.selected_annotation.get_points()
            for i, (x, y) in enumerate(pts):
                dx = x - pos.x()
                dy = y - pos.y()
                if (dx * dx + dy * dy) ** 0.5 <= scaled_tol:
                    return i
            
            # 检查旋转柄
            if len(pts) == 4:
                # 计算旋转柄位置
                rot_handle_pos = self._get_rotation_handle_position(pts)
                if rot_handle_pos:
                    dx = rot_handle_pos.x() - pos.x()
                    dy = rot_handle_pos.y() - pos.y()
                    if (dx * dx + dy * dy) ** 0.5 <= scaled_tol * 1.5:  # 旋转柄的容差稍大一点
                        return -1  # 旋转柄索引
        except Exception:
            pass
        return None

    def _get_rotation_handle_position(self, points):
        """计算旋转柄的位置"""
        if len(points) != 4:
            return None
        
        # 找到最上方的两个点（y值最小的两个点）
        sorted_points = sorted(points, key=lambda p: p[1])
        top_points = sorted_points[:2]
        
        # 按x坐标排序
        top_points.sort(key=lambda p: p[0])
        
        # 计算上边缘中点
        p0, p1 = top_points[0], top_points[1]
        mx = (p0[0] + p1[0]) / 2.0
        my = (p0[1] + p1[1]) / 2.0
        
        # 计算上边缘的方向向量
        ex = p1[0] - p0[0]
        ey = p1[1] - p0[1]
        length = (ex**2 + ey**2)**0.5
        
        if length < 1e-6:
            return None
        
        # 计算上边缘的法向量（向上方向）
        # 对于正常的矩形，上边缘的法向量应该是向上（负y方向）
        nx = -ey / length
        ny = ex / length
        
        # 确保法向量指向外部（向上）
        # 计算矩形中心
        cx = sum(p[0] for p in points) / 4
        cy = sum(p[1] for p in points) / 4
        
        # 如果法向量指向矩形内部，则反转
        center_to_mid = (cx - mx, cy - my)
        if nx * center_to_mid[0] + ny * center_to_mid[1] > 0:
            nx = -nx
            ny = -ny
        
        # 旋转柄距离边缘的距离
        handle_distance = 25.0 / self.scale_factor  # 考虑缩放
        
        # 旋转柄位置
        rot_x = mx + nx * handle_distance
        rot_y = my + ny * handle_distance
        
        return QPoint(int(rot_x), int(rot_y))

    def delete_selected_annotations(self):
        """删除选中的标注"""
        annotations_to_remove = []
        for i, annotation in enumerate(self.annotations):
            if annotation.selected:
                annotations_to_remove.append((i, annotation))
        
        # 从后往前删除
        for i, annotation in reversed(annotations_to_remove):
            self.annotations.pop(i)
            self.annotation_removed.emit(annotation.id)
        
        self.selected_annotation = None
        self.update()
    
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制背景
        painter.fillRect(self.rect(), QColor(50, 50, 50))
        
        if self.pixmap and not self.pixmap.isNull():
            # 保存 painter 状态
            painter.save()
            
            # 应用变换
            painter.translate(self.offset.x(), self.offset.y())
            painter.scale(self.scale_factor, self.scale_factor)
            
            # 绘制网格
            if self.show_grid:
                self._draw_grid(painter)
            
            # 绘制图像
            painter.drawPixmap(0, 0, self.pixmap)
            
            # 绘制标注
            for annotation in self.annotations:
                if annotation.visible:
                    self._draw_annotation(painter, annotation)
            
            # 绘制当前正在绘制的标注
            if self.drawing:
                self._draw_current_annotation(painter)
            
            # 恢复 painter 状态
            painter.restore()
    
    def _draw_grid(self, painter: QPainter):
        """绘制网格"""
        if not self.pixmap:
            return
        
        width = self.pixmap.width()
        height = self.pixmap.height()
        
        painter.save()
        pen = QPen(QColor(100, 100, 100, 100), 1)
        painter.setPen(pen)
        
        # 绘制垂直线
        for x in range(0, width + self.grid_size, self.grid_size):
            painter.drawLine(x, 0, x, height)
        
        # 绘制水平线
        for y in range(0, height + self.grid_size, self.grid_size):
            painter.drawLine(0, y, width, y)
        
        painter.restore()
    
    def _draw_annotation(self, painter: QPainter, annotation: AnnotationItem):
        """绘制单个标注"""
        # 获取类别颜色 - 使用新的颜色获取方法
        color = self._get_class_color(annotation.class_id)
        
        # 设置画笔
        pen_width = 3 if annotation.selected else 2
        pen_color = QColor(255, 255, 255) if annotation.locked else color
        
        pen = QPen(pen_color, pen_width)
        if annotation.selected:
            pen.setStyle(Qt.DashLine)
        
        painter.setPen(pen)
        
        # 设置填充（如果是选中状态，使用半透明填充）
        if annotation.selected:
            fill_color = QColor(color)
            fill_color.setAlpha(30)
            painter.setBrush(QBrush(fill_color))
        else:
            painter.setBrush(Qt.NoBrush)
        
        # 根据标注类型绘制
        points = annotation.get_points()
        
        if annotation.bbox_type == BBoxType.HORIZONTAL:
            if len(points) == 4:
                # 绘制矩形
                x1, y1 = points[0]
                x2, y2 = points[2]
                painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
                
                # 绘制角点
                if annotation.selected:
                    self._draw_corners(painter, points)
        
        elif annotation.bbox_type == BBoxType.ROTATED:
            if len(points) == 4:
                # 绘制旋转矩形
                path = QPainterPath()
                path.moveTo(points[0][0], points[0][1])
                for i in range(1, 4):
                    path.lineTo(points[i][0], points[i][1])
                path.closeSubpath()
                painter.drawPath(path)
                
                # 绘制角点
                if annotation.selected:
                    self._draw_corners(painter, points)
        
        elif annotation.bbox_type == BBoxType.POLYGON:
            if len(points) >= 3:
                # 绘制多边形
                path = QPainterPath()
                path.moveTo(points[0][0], points[0][1])
                for i in range(1, len(points)):
                    path.lineTo(points[i][0], points[i][1])
                path.closeSubpath()
                painter.drawPath(path)
                
                # 绘制顶点
                if annotation.selected:
                    for point in points:
                        painter.setBrush(QBrush(QColor(255, 255, 255)))
                        painter.drawEllipse(QRectF(point[0] - 1.5, point[1] - 1.5, 3, 3))
        
        # 绘制标签
        if self.show_class_names or (self.show_bbox_info and annotation.selected):
            self._draw_label(painter, annotation)
    
    def _draw_corners(self, painter: QPainter, points: List[tuple]):
        """绘制角点"""
        # 绘制角点手柄
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        pen = QPen(QColor(0, 0, 0), 1)
        painter.setPen(pen)
        
        handle_size = 8.0 / self.scale_factor  # 根据缩放调整手柄大小
        for point in points:
            painter.drawRect(QRectF(
                point[0] - handle_size/2, 
                point[1] - handle_size/2, 
                handle_size, 
                handle_size
            ))
        
        # 绘制旋转柄
        rot_handle_pos = self._get_rotation_handle_position(points)
        if rot_handle_pos:
            # 绘制连接线
            # 找到上边缘中点
            sorted_points = sorted(points, key=lambda p: p[1])
            top_points = sorted_points[:2]
            top_points.sort(key=lambda p: p[0])
            p0, p1 = top_points[0], top_points[1]
            mx = (p0[0] + p1[0]) / 2.0
            my = (p0[1] + p1[1]) / 2.0
            
            painter.setPen(QPen(QColor(255, 200, 0), 2))
            # drawLine supports numeric coordinates directly
            painter.drawLine(int(mx), int(my), int(rot_handle_pos.x()), int(rot_handle_pos.y()))
            
            # 绘制旋转柄
            painter.setBrush(QBrush(QColor(255, 200, 0)))
            painter.setPen(QPen(QColor(100, 70, 0), 2))
            rot_handle_size = 10.0 / self.scale_factor
            painter.drawEllipse(QRectF(
                rot_handle_pos.x() - rot_handle_size/2,
                rot_handle_pos.y() - rot_handle_size/2,
                rot_handle_size,
                rot_handle_size
            ))

    def _draw_label(self, painter: QPainter, annotation: AnnotationItem):
        """绘制标签"""
        text = annotation.class_name
        
        # 修复：检查是否有 get_area 方法，如果没有就跳过面积显示
        if self.show_bbox_info:
            try:
                # 检查是否存在 get_area 方法
                if hasattr(annotation, 'get_area'):
                    area = annotation.get_area()
                    if area is not None:
                        text = f"{text} ({int(area)}px²)"
            except:
                pass
        
        # 如果不需要显示标签，直接返回
        if not text:
            return
        
        # Determine position and rotation
        pts = annotation.get_points()
        if not pts: 
            return
        
        # Center of the box
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        
        rotation = 0.0
        
        if annotation.bbox_type == BBoxType.ROTATED and len(pts) == 4:
            # Use top edge (0-1) for orientation
            p0 = pts[0]
            p1 = pts[1]
            
            # Midpoint of top edge
            label_x = (p0[0] + p1[0]) / 2.0
            label_y = (p0[1] + p1[1]) / 2.0
            
            # Calculate rotation angle of the edge
            # edge vector
            ex = p1[0] - p0[0]
            ey = p1[1] - p0[1]
            
            import math
            angle_rad = math.atan2(ey, ex)
            rotation = math.degrees(angle_rad)
            
            # Offset slightly "up" (outwards)
            # Normal vector (-ey, ex) as derived before
            length = (ex**2 + ey**2)**0.5
            if length > 0.001:
                nx = -ey / length
                ny = ex / length
                # Direction check
                mcx = cx - label_x
                mcy = cy - label_y
                if (nx*mcx + ny*mcy) > 0:
                     nx = -nx
                     ny = -ny
                
                # Move label up by 15px
                label_x += nx * 15
                label_y += ny * 15
                
        else:
            # Horizontal / Polygon default
            try:
                bbox = annotation.get_bounding_rect()
                label_x = bbox.x()
                label_y = bbox.y() - 15
                if label_y < 0: 
                    label_y = bbox.bottom() + 5
            except:
                # 如果获取边界矩形失败，使用中心点
                label_x = cx
                label_y = cy - 20
            rotation = 0.0

        # Draw
        painter.save()
        painter.translate(label_x, label_y)
        painter.rotate(rotation)
        
        # 确保文本不会颠倒显示
        if rotation > 90 or rotation < -90:
            # 如果旋转角度太大，调整文字方向
            painter.rotate(180)
        
        # Draw text centered at (0,0) (which is label_x, label_y)
        font_metrics = painter.fontMetrics()
        text_rect = font_metrics.boundingRect(text)
        text_rect.adjust(-4, -4, 4, 4)
        
        # Center the rect
        w = text_rect.width()
        h = text_rect.height()
        
        # Background
        color = self._get_class_color(annotation.class_id)
        painter.fillRect(QRectF(-w/2, -h/2, w, h), color)
        
        # Text
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(QRectF(-w/2, -h/2, w, h), Qt.AlignCenter, text)
        
        painter.restore()
    
    def _draw_current_annotation(self, painter: QPainter):
        """绘制当前正在绘制的标注"""
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        painter.setBrush(Qt.NoBrush)
        
        if self.annotation_mode == AnnotationMode.HORIZONTAL and len(self.current_bbox) == 4:
            x1, y1, x2, y2 = self.current_bbox
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
        
        elif self.annotation_mode == AnnotationMode.ROTATED and len(self.current_bbox) == 4:
            # 绘制旋转矩形
            # points logic similar to finish_annotation
            x1, y1, x2, y2 = self.current_bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            
            try:
                import math
                if self.start_point:
                    ang = math.atan2((y2 - self.start_point.y()), (x2 - self.start_point.x()))
                else:
                    ang = 0.0
                rot_deg = math.degrees(ang)
                
                rad = math.radians(rot_deg)
                cos_a = math.cos(rad)
                sin_a = math.sin(rad)
                
                hw = w / 2.0
                hh = h / 2.0
                corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
                
                path = QPainterPath()
                first = True
                for ox, oy in corners:
                    rx = ox * cos_a - oy * sin_a + cx
                    ry = ox * sin_a + oy * cos_a + cy
                    if first:
                        path.moveTo(rx, ry)
                        first = False
                    else:
                        path.lineTo(rx, ry)
                path.closeSubpath()
                painter.drawPath(path)
                
            except Exception:
                painter.drawRect(QRectF(min(x1,x2), min(y1,y2), abs(x1-x2), abs(y1-y2)))
        
        elif self.annotation_mode == AnnotationMode.POLYGON and self.current_points:
            path = QPainterPath()
            path.moveTo(self.current_points[0].x(), self.current_points[0].y())
            for i in range(1, len(self.current_points)):
                path.lineTo(self.current_points[i].x(), self.current_points[i].y())

            # 如果存在预览点，则把它连上以显示临时边
            if getattr(self, '_preview_point', None) is not None:
                p = self._preview_point
                path.lineTo(p.x(), p.y())

            painter.drawPath(path)

            # 绘制点
            painter.setBrush(QBrush(QColor(0, 255, 0)))
            for point in self.current_points:
                painter.drawEllipse(point, 3, 3)
            # 绘制预览点（如果有）
            if getattr(self, '_preview_point', None) is not None:
                painter.drawEllipse(self._preview_point, 3, 3)

    # ----------------- 便捷方法/外部接口 -----------------
    def remove_annotation(self, annotation: AnnotationItem):
        """外部接口：按 id 删除指定标注并发出信号。"""
        try:
            ann_id = annotation.id
            self.annotations = [a for a in self.annotations if a.id != ann_id]
            self.annotation_removed.emit(ann_id)
            self.update()
        except Exception:
            pass

    def set_show_labels(self, enabled: bool):
        self.show_class_names = bool(enabled)
        self.update()

    def set_show_grid(self, enabled: bool):
        self.show_grid = bool(enabled)
        self.update()

    def set_grid_size(self, size: int):
        try:
            self.grid_size = int(size)
        except Exception:
            pass
        self.update()

    def set_show_confidence(self, enabled: bool):
        self.show_confidence = bool(enabled)
        self.update()

    def set_show_statistics(self, enabled: bool):
        # 修复：正确的属性赋值
        self.show_statistics = bool(enabled)
        self.update()

    def set_annotation_mode(self, mode):
        """设置标注模式，避免递归调用"""
        try:
            # 检查模式是否真的改变了
            if self.annotation_mode != mode:
                old_mode = self.annotation_mode
                self.annotation_mode = mode
                
                # 清除当前的绘制状态
                if mode == AnnotationMode.NONE:
                    self.drawing = False
                    self.current_bbox = []
                    self.current_points = []
                    self.start_point = None
                    self._preview_point = None
                
                # 只有当模式确实改变时才发射信号
                if old_mode != mode:
                    self.mode_changed.emit(mode)
                
                self.update()
            else:
                # 模式相同，只更新UI
                self.update()
        except Exception as e:
            _logger.error(f"设置标注模式失败: {e}")
            self.update()

    def select_annotation(self, annotation):
        """选择标注，避免递归调用"""
        try:
            # Check for no change to avoid signal loops
            prev = self.selected_annotation
            
            # Retrieve IDs safely
            prev_id = getattr(prev, 'id', None) if prev else None
            curr_id = getattr(annotation, 'id', None) if annotation else None
            
            if prev_id == curr_id:
                return

            # Clear all selection flags to ensure consistency
            for ann in self.annotations:
                ann.selected = False

            self.selected_annotation = annotation
            if annotation:
                annotation.selected = True

            # 只有当选择确实改变时才发射信号
            if prev != annotation:
                self.annotation_selected.emit(annotation)
            
            self.update()
        except Exception as e:
            _logger.error(f"选择标注失败: {e}")

    def clear(self):
        try:
            self.image = None
            self.pixmap = None
            self.annotations = []
            self.update()
        except Exception:
            pass

    def clear_annotations(self):
        try:
            self.annotations = []
            self.update()
        except Exception:
            pass

    def set_annotations(self, annotations: List[AnnotationItem]):
        try:
            self.annotations = list(annotations) if annotations else []
            self.update()
        except Exception:
            pass

    def set_current_class(self, class_item: ClassItem):
        try:
            self.current_class = class_item
        except Exception:
            pass

    def has_image(self) -> bool:
        return self.image is not None

    def get_image_size(self):
        if self.image is None:
            return (0, 0)
        h, w = self.image.shape[:2]
        return (w, h)

    def get_zoom_level(self) -> float:
        return self.scale_factor * 100.0