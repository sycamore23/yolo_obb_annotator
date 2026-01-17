[English](README.md) | [中文](README_CN.md)

# YOLO OBB Annotator

A professional annotation tool for **YOLO Oriented Bounding Box (OBB)** detection, designed to make rotated object labeling **accurate, efficient, and production-ready**.

---

## 🚀 Overview

**YOLO OBB Annotator** is a dedicated annotation tool built specifically for YOLO-based models that require **Oriented Bounding Box (OBB)** labels.

Unlike traditional annotation tools that only support axis-aligned bounding boxes, this project enables **precise rotated bounding box annotation**, making it ideal for scenarios where object orientation matters.

Typical application areas include:

- Industrial visual inspection  
- Aerial and remote sensing imagery  
- Autonomous driving  
- High-precision object detection and pose-aware tasks  

---

## ✨ Key Features

- 🎯 **Native OBB Annotation**  
  Precise rotated rectangle labeling fully compatible with YOLO OBB formats.

- 🤖 **AI-Assisted Labeling**  
  Integrated YOLOv8 / YOLOv11 models for automatic annotation to significantly boost efficiency.

- 🎨 **Modern GUI**  
  PyQt5-based graphical interface with intuitive interactions and real-time visualization.

- 📦 **Batch Processing**  
  Import, annotate, and export large-scale datasets with ease.

- 🔄 **Real-Time Preview**  
  Interactive zooming, panning, and editing of annotations.

- 📊 **Dataset Management**  
  Built-in dataset splitting, backup, and basic version control support.

- 🚀 **Training Integration**  
  Seamless integration with YOLO training pipelines for end-to-end workflows.

- 📈 **Multiple Export Formats**  
  Supports YOLO, COCO, and Pascal VOC formats.

---

## 📦 Installation

### Requirements

- Python 3.8 or higher  
- Windows / Linux / macOS  

### Install from Source

```bash
git clone https://github.com/sycamore2323/yolo-obb-annotator.git
cd yolo-obb-annotator
pip install -r requirements.txt
▶️ Quick Start
Run Demo
python demo.py
Launch Full Application
python main.py
🖱️ Basic Workflow
Create a new project

Select an image directory

Configure class labels

Annotate objects:

Manually draw OBBs

Or enable AI-assisted auto-labeling

Save annotations and export the dataset

📊 Examples
It is highly recommended to add screenshots or GIFs here demonstrating:

The annotation interface

OBB labeling process

AI auto-annotation results

🏗️ Architecture
Modular design – Core logic separated from the UI for maintainability

Multi-threaded execution – Annotation and training run without blocking the interface

Memory-efficient loading – Optimized image caching and loading strategy

Auto-save & recovery – Prevents annotation data loss

🤝 Contributing
Contributions are welcome!

git checkout -b feature/YourFeature
git commit -m "Add YourFeature"
git push origin feature/YourFeature
Then open a Pull Request on GitHub.

📄 License
This project is licensed under the MIT License.
See the LICENSE file for details.

🙏 Acknowledgements
Ultralytics YOLO

PyQt5

OpenCV

⭐ If you find this project useful, please consider giving it a Star on GitHub!




