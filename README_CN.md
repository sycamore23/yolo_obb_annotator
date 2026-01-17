[English](README.md) | [中文](README_CN.md)

# YOLO OBB Annotator

一个专注于 **YOLO Oriented Bounding Box（OBB，旋转框）** 的专业标注工具，  
用于高精度旋转目标检测数据集的高效构建。

---

## 🚀 项目简介

**YOLO OBB Annotator** 是一款专为 YOLO 系列模型设计的标注工具，  
原生支持 **旋转目标（OBB）标注**，解决了传统标注工具仅支持水平框、  
无法准确描述目标方向的问题。

该工具特别适用于对目标方向、姿态有较高要求的计算机视觉任务。

**适用场景包括：**

- 工业视觉检测  
- 遥感 / 航拍影像分析  
- 自动驾驶  
- 高精度目标检测与姿态感知任务  

---

## ✨ 核心特性

- 🎯 **原生 OBB 旋转框标注**  
  支持精确的旋转矩形标注，完全兼容 YOLO OBB 格式。

- 🤖 **AI 智能辅助标注**  
  集成 YOLOv8 / YOLOv11 模型，自动生成初始标注，大幅提升效率。

- 🎨 **可视化图形界面**  
  基于 PyQt5 的现代化 GUI，交互直观、所见即所得。

- 📦 **批量数据处理**  
  支持大规模数据集的批量导入、标注与导出。

- 🔄 **实时预览与交互**  
  支持缩放、平移、编辑等操作，标注结果即时反馈。

- 📊 **数据集管理功能**  
  提供数据集拆分、备份及基础版本管理能力。

- 🚀 **训练流程集成**  
  可无缝对接 YOLO 训练流程，实现端到端工作流。

- 📈 **多格式导出支持**  
  支持 YOLO、COCO、Pascal VOC 等主流格式。

---

## 📦 安装

### 环境要求

- Python 3.8 及以上  
- Windows / Linux / macOS  

### 安装步骤

```bash
git clone https://github.com/sycamore2323/yolo-obb-annotator.git
cd yolo-obb-annotator
pip install -r requirements.txt
▶️ 快速开始
运行演示程序
python demo.py
启动完整应用
python main.py
🖱️ 基本使用流程
新建项目

选择图像文件目录

配置目标类别信息

进行标注：

手动绘制 OBB 旋转框

或启用 AI 智能自动标注

保存标注结果并导出数据集

📊 示例效果
强烈建议在此处添加 界面截图或 GIF 动画，展示：

标注工具界面

OBB 旋转框标注过程

AI 自动标注效果

🏗️ 项目架构
模块化设计：核心逻辑与界面分离，便于维护与扩展

多线程处理：标注与训练异步执行，保证界面流畅

内存优化：高效的图像加载与缓存机制

自动保存与恢复：有效防止标注数据丢失

🤝 贡献指南
欢迎提交 Issue 和 Pull Request！

git checkout -b feature/YourFeature
git commit -m "Add YourFeature"
git push origin feature/YourFeature
然后在 GitHub 上创建 Pull Request。

📄 License
本项目采用 MIT License 开源协议，
详见 LICENSE 文件。

🙏 致谢
Ultralytics YOLO

PyQt5

OpenCV

⭐ 如果你觉得这个项目对你有帮助，欢迎在 GitHub 上点一个 Star 支持！