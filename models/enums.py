"""
枚举定义
"""

from enum import Enum, IntEnum

class AnnotationMode(IntEnum):
    """标注模式"""
    NONE = 0          # 选择模式
    HORIZONTAL = 1    # 水平矩形框
    ROTATED = 2       # 旋转矩形框
    POLYGON = 3       # 多边形
    POINTS = 4        # 关键点
    LINE = 5          # 线段
    ELLIPSE = 6       # 椭圆

class EditMode(IntEnum):
    """编辑模式"""
    NONE = 0          # 无编辑
    MOVE = 1          # 移动
    RESIZE = 2        # 调整大小
    ROTATE = 3        # 旋转
    ADD_POINT = 4     # 添加点
    REMOVE_POINT = 5  # 删除点

class BBoxType(str, Enum):
    """边界框类型"""
    HORIZONTAL = "horizontal"  # 水平矩形框
    ROTATED = "rotated"        # 旋转矩形框
    POLYGON = "polygon"        # 多边形
    POINTS = "points"          # 关键点
    LINE = "line"              # 线段
    ELLIPSE = "ellipse"        # 椭圆

class ExportFormat(str, Enum):
    """导出格式"""
    YOLO = "yolo"     # YOLO格式
    COCO = "coco"     # COCO格式
    VOC = "voc"       # VOC格式
    JSON = "json"     # 自定义JSON格式
    TXT = "txt"       # 纯文本格式

class ImageFormat(str, Enum):
    """图片格式"""
    JPEG = "jpeg"
    PNG = "png"
    BMP = "bmp"
    TIFF = "tiff"
    WEBP = "webp"

class ProjectState(IntEnum):
    """项目状态"""
    NEW = 0           # 新项目
    OPENED = 1        # 已打开
    MODIFIED = 2      # 已修改
    SAVED = 3         # 已保存

class SnappingMode(IntEnum):
    """吸附模式"""
    NONE = 0          # 无吸附
    GRID = 1          # 网格吸附
    POINTS = 2        # 点吸附
    EDGES = 3         # 边吸附

class DisplayMode(IntEnum):
    """显示模式"""
    NORMAL = 0        # 正常显示
    WIREFRAME = 1     # 线框模式
    OVERLAY = 2       # 叠加模式