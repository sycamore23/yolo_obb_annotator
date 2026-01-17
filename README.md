# YOLO OBB Annotator

一个强大的YOLO模型OBB（Oriented Bounding Box）标注工具，让标注变得简单高效！

## 🚀 项目简介

YOLO OBB Annotator 是一个专为YOLO模型设计的专业标注工具，专注于Oriented Bounding Box (OBB) 标注，帮助开发者快速创建高质量的训练数据集。这个工具解决了传统标注工具不支持旋转框标注的问题，特别适合需要精确物体检测和姿态识别的计算机视觉项目。

适合对象：计算机视觉研究者、AI工程师、自动驾驶开发者、工业检测工程师等需要高精度物体检测标注的专业人士。

## ✨ 特性

- 🎯 **专业OBB标注**：支持旋转矩形框标注，完美适配YOLO模型的OBB检测
- 🤖 **AI智能标注**：集成YOLOv8/YOLOv11模型，自动标注大幅提升效率
- 📦 **批量处理**：支持批量导入、处理和导出，处理大规模数据集无压力
- 🎨 **可视化界面**：现代化的PyQt5 GUI界面，直观易用
- 🔄 **实时预览**：标注实时预览，支持缩放、平移等操作
- 📊 **数据集管理**：内置数据集分割、备份、版本控制功能
- 🚀 **训练集成**：一键启动YOLO模型训练，无缝集成训练流程
- 📈 **多格式导出**：支持YOLO、COCO、Pascal VOC等多种格式导出
- 🎛️ **高度可配置**：丰富的设置选项，满足不同项目需求

## 📦 安装

### 环境要求
- Python 3.8+
- Windows/Linux/macOS

### 安装步骤

1. 克隆项目
```bash
git clone https://github.com/sycamore2323/yolo-obb-annotator.git
cd yolo-obb-annotator
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 运行程序
```bash
python main.py
```

## ▶️ 快速开始

1. **运行演示**
   ```bash
   python demo.py
   ```

2. **启动完整应用**
   ```bash
   python main.py
   ```

3. **创建项目**
   - 点击"新建项目"
   - 选择图像文件夹
   - 配置类别信息

4. **开始标注**
   - 使用鼠标绘制OBB框
   - 或启用AI自动标注
   - 保存标注结果

5. **导出数据集**
   - 选择导出格式（YOLO、COCO等）
   - 配置数据集分割比例
   - 一键导出

## 📊 示例效果

### 标注界面
![标注界面](docs/screenshots/annotation_interface.png)

### OBB标注示例
![OBB标注](docs/screenshots/obb_annotation_example.png)

### 训练结果
![训练结果](docs/screenshots/training_results.png)

## 🏗️ 架构特点

- **模块化设计**：核心功能分离，便于维护和扩展
- **多线程处理**：标注和训练异步执行，不阻塞界面
- **内存优化**：智能图像加载和缓存机制
- **错误恢复**：自动保存和备份机制，防止数据丢失

## 🤝 贡献

欢迎提交Issue和Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 License

本项目采用 MIT License - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - 强大的YOLO实现
- [PyQt5](https://pypi.org/project/PyQt5/) - 优秀的GUI框架
- [OpenCV](https://opencv.org/) - 计算机视觉库

## 📞 联系我们

- 项目主页: [https://github.com/sycamore2323/yolo-obb-annotator](https://github.com/sycamore2323/yolo-obb-annotator)
- 问题反馈: [Issues](https://github.com/sycamore2323/yolo-obb-annotator/issues)

---

⭐ 如果这个项目对你有帮助，请给我们一个Star！</content>
<parameter name="filePath">f:\yolo_obb_annotator_vs\README.md