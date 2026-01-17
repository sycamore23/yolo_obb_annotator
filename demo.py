#!/usr/bin/env python3
"""
YOLO OBB Annotator 演示脚本
展示如何使用标注工具的核心功能
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def demo_config():
    """演示配置功能"""
    from config import Config

    print("=== 配置演示 ===")
    config = Config()
    print(f"输出目录: {config.app_config.output_dir}")
    print(f"自动保存间隔: {config.app_config.auto_save_interval} 分钟")
    print(f"默认类别颜色数量: {len(config.app_config.default_class_colors)}")
    print()

def demo_project_manager():
    """演示项目管理功能"""
    from config import Config
    from core.project_manager import ProjectManager

    print("=== 项目管理演示 ===")
    config = Config()
    pm = ProjectManager(config)

    # 创建示例类别
    classes = pm.get_classes()
    print(f"当前类别数量: {len(classes)}")
    print()

def demo_annotation_utils():
    """演示标注工具功能"""
    print("=== 标注工具演示 ===")
    print("标注工具功能包括:")
    print("- 自动标注图像")
    print("- 批量处理")
    print("- 导出多种格式")
    print()

def main():
    """主演示函数"""
    print("🎯 YOLO OBB Annotator 演示")
    print("=" * 50)

    try:
        demo_config()
        demo_project_manager()
        demo_annotation_utils()

        print("✅ 演示完成！")
        print("\n🚀 启动完整应用:")
        print("python main.py")

    except Exception as e:
        print(f"❌ 演示出错: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())