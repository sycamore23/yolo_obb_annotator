"""
AI 辅助标注工具

提供两个主要函数：
- auto_annotate_image(config, image_path) -> List[AnnotationItem]
- auto_annotate_all_images(config, image_paths, progress_callback=None) -> Dict[image_path, List[AnnotationItem]]

实现策略：优先尝试使用 `ultralytics.YOLO`（如果已安装且 `config.app_config.ai_model_path` 已配置），
否则抛出 RuntimeError，调用方应捕获并提示用户。
"""

from typing import List, Dict, Optional
import os
import cv2

from config import Config
from models.annotation_item import AnnotationItem
from models.enums import BBoxType


def _load_ultralytics_model(model_path: str):
    try:
        from ultralytics import YOLO
        return YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"无法加载 Ultralytics YOLO 模型: {e}")


def auto_annotate_image(config: Config, image_path: str) -> List[AnnotationItem]:
    """对单张图片运行 AI 模型，返回 AnnotationItem 列表。

    如果 `config.app_config.ai_model_path` 未配置或模型加载失败，将抛出 RuntimeError。
    """
    model_path = getattr(config.app_config, 'ai_model_path', '')

    # 如果未配置模型，或模型加载/推理失败，则降级为基于已有标注的迁移方法
    if not model_path:
        return auto_annotate_from_labeled_images(config, image_path)

    try:
        model = _load_ultralytics_model(model_path)

        # 读取图像以获取宽高
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"无法读取图像: {image_path}")
        h, w = img.shape[:2]

        # 预测
        res = model.predict(source=image_path, conf=getattr(config.app_config, 'ai_confidence_threshold', 0.25), max_det=getattr(config.app_config, 'ai_max_detections', 300))
    except Exception:
        # 降级：迁移已有标注
        return auto_annotate_from_labeled_images(config, image_path)

    anns: List[AnnotationItem] = []
    if not res:
        return anns

    # ultralytics 返回的 res 是 Results 对象的列表（每张图一个）
    r = res[0]
    # 可能的属性：boxes (Boxes), boxes.xyxy, boxes.conf, boxes.cls
    boxes = getattr(r, 'boxes', None)
    if boxes is None:
        return anns

    # boxes.xyxy numpy array Nx4, boxes.conf, boxes.cls
    try:
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy.numpy()
    except Exception:
        try:
            xyxy = boxes.xyxy.numpy()
        except Exception:
            xyxy = []

    try:
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf.numpy()
    except Exception:
        try:
            confs = boxes.conf.numpy()
        except Exception:
            confs = [1.0] * len(xyxy)

    try:
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls.numpy()
    except Exception:
        try:
            clss = boxes.cls.numpy()
        except Exception:
            clss = [0] * len(xyxy)

    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = map(float, box[:4])
        conf = float(confs[i]) if i < len(confs) else 1.0
        cls_id = int(clss[i]) if i < len(clss) else 0

        # 创建 AnnotationItem
        points = [
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2)
        ]
        ann = AnnotationItem(
            bbox_type=BBoxType.HORIZONTAL,
            class_id=cls_id,
            class_name=str(cls_id),
            confidence=conf,
            points=points
        )
        anns.append(ann)

    return anns


def auto_annotate_all_images(config: Config, image_paths: List[str], progress_callback: Optional[callable] = None) -> Dict[str, List[AnnotationItem]]:
    """对多张图片运行 AI 模型并返回字典 {image_path: [AnnotationItem,...]}。

    progress_callback(current_index, total) 可选。
    """
    results: Dict[str, List[AnnotationItem]] = {}
    total = len(image_paths)

    for idx, img_path in enumerate(image_paths, start=1):
        try:
            anns = auto_annotate_image(config, img_path)
            results[img_path] = anns
        except Exception:
            results[img_path] = []

        if progress_callback:
            try:
                progress_callback(idx, total)
            except Exception:
                pass

    return results


__all__ = ["auto_annotate_image", "auto_annotate_all_images"]


# -------------------- 迁移标注实现（无模型时使用） --------------------
def _parse_yolo_label_file(label_path: str, img_w: int, img_h: int) -> List[AnnotationItem]:
    anns: List[AnnotationItem] = []
    try:
        text = open(label_path, 'r', encoding='utf-8').read().strip()
    except Exception:
        return anns

    if not text:
        return anns

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            # 可能为 class + polygon pts (even count)
            if len(parts) >= 3 and (len(parts) - 1) % 2 == 0:
                try:
                    cls_id = int(parts[0])
                except Exception:
                    cls_id = 0
                pts = []
                vals = parts[1:]
                for i in range(0, len(vals), 2):
                    x = float(vals[i]) * img_w
                    y = float(vals[i+1]) * img_h
                    pts.append((x, y))
                ann = AnnotationItem(bbox_type=BBoxType.POLYGON, class_id=cls_id, class_name=str(cls_id), points=pts)
                anns.append(ann)
            continue

        # 5 values -> class x_center y_center width height (YOLO)
        try:
            cls_id = int(parts[0])
            if len(parts) == 5:
                xc = float(parts[1]) * img_w
                yc = float(parts[2]) * img_h
                w = float(parts[3]) * img_w
                h = float(parts[4]) * img_h
                x1 = xc - w / 2.0
                y1 = yc - h / 2.0
                x2 = xc + w / 2.0
                y2 = yc + h / 2.0
                pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                ann = AnnotationItem(bbox_type=BBoxType.HORIZONTAL, class_id=cls_id, class_name=str(cls_id), points=pts)
                anns.append(ann)
            else:
                # more values: treat as polygon
                vals = parts[1:]
                pts = []
                for i in range(0, len(vals), 2):
                    x = float(vals[i]) * img_w
                    y = float(vals[i+1]) * img_h
                    pts.append((x, y))
                ann = AnnotationItem(bbox_type=BBoxType.POLYGON, class_id=cls_id, class_name=str(cls_id), points=pts)
                anns.append(ann)
        except Exception:
            continue

    return anns


def _compute_homography_between_images(img1, img2):
    """使用 ORB 特征匹配尝试估计 img1 -> img2 的单应矩阵（若足够匹配），否则返回 (None, 0.0)。

    返回值: (H, score) 其中 score 是基于匹配数量和 RANSAC 内点比率的置信度 (0.0-1.0)。
    """
    try:
        orb = cv2.ORB_create(1000)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None:
            return None, 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 8:
            return None, 0.0

        pts1 = []
        pts2 = []
        # 使用前100个最优匹配进行单应估计
        for m in matches[:100]:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

        import numpy as np
        pts1 = np.array(pts1)
        pts2 = np.array(pts2)

        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            return None, 0.0

        # mask 是 Nx1 的数组，1 表示内点
        inliers = int(mask.sum()) if hasattr(mask, 'sum') else int(sum([int(x) for x in mask]))
        total_used = min(len(matches), 100)
        inlier_ratio = float(inliers) / float(total_used) if total_used > 0 else 0.0

        # 置信度综合：考虑匹配数占比与内点比率
        # 匹配数比 = min(1.0, len(matches) / 200)（200 认为是非常多的匹配）
        match_count_score = min(1.0, len(matches) / 200.0)
        score = 0.6 * inlier_ratio + 0.4 * match_count_score
        score = max(0.0, min(1.0, score))

        return H, score
    except Exception:
        return None, 0.0


def _transform_points(H, points):
    """使用单应矩阵 H 将点列表变换，points 为 [(x,y), ...]"""
    import numpy as np
    if H is None:
        return None
    src = np.array([[x, y, 1.0] for x, y in points]).T
    dst = H.dot(src)
    dst = dst / dst[2:3, :]
    pts = [(float(dst[0, i]), float(dst[1, i])) for i in range(dst.shape[1])]
    return pts


def auto_annotate_from_labeled_images(config: Config, image_path: str) -> List[AnnotationItem]:
    """在没有外部模型时，根据已有标注图片迁移标注到 `image_path`。

    策略：在 `labels` 目录中查找已有标签文件对应的图片，按 ORB 特征匹配找到最相似的源图片，
    通过单应矩阵将源图片的标注（YOLO 或多边形）映射到目标图片坐标。
    返回映射后的 `AnnotationItem` 列表（可能为空）。
    """
    labels_dir = config.get_labels_dir()
    # 读取目标图像
    img_t = cv2.imread(image_path)
    if img_t is None:
        return []
    h_t, w_t = img_t.shape[:2]

    best_H = None
    best_source = None

    # 遍历 labels 目录中的标签文件，尝试找到能匹配的源图像
    if not labels_dir.exists():
        return []

    for label_file in labels_dir.glob('*.txt'):
        try:
            stem = label_file.stem
            # 找对应的图片文件（尝试常见扩展）
            src_img_path = None
            for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'):
                p = config.get_images_dir() / f"{stem}{ext}"
                if p.exists():
                    src_img_path = str(p)
                    break
            if not src_img_path:
                continue

            img_s = cv2.imread(src_img_path)
            if img_s is None:
                continue

            H, score = _compute_homography_between_images(img_s, img_t)
            if H is not None and score > 0.0:
                # 选择得分最高的源图片
                if best_H is None or score > best_score:
                    best_H = H
                    best_score = score
                    best_source = (src_img_path, label_file, img_s)
                # 继续遍历以寻找更高分
            
        except Exception:
            continue

    if best_source is None or best_H is None:
        return [], None, 0.0

    src_img_path, label_file, img_s = best_source
    h_s, w_s = img_s.shape[:2]

    # 解析源标签
    src_anns = _parse_yolo_label_file(str(label_file), w_s, h_s)
    mapped = []
    for ann in src_anns:
        pts = ann.points
        transformed = _transform_points(best_H, pts)
        if not transformed:
            continue
        new_ann = AnnotationItem(bbox_type=ann.bbox_type, class_id=ann.class_id, class_name=ann.class_name, points=transformed, confidence=ann.confidence)
        mapped.append(new_ann)

    return mapped, src_img_path, best_score


# ---- 明确的接口：强制使用模型或强制使用已有标注迁移 ----
def auto_annotate_image_with_model(config: Config, image_path: str) -> List[AnnotationItem]:
    """仅使用外部模型进行推理；若未配置模型或加载/推理失败则抛出 RuntimeError。"""
    model_path = getattr(config.app_config, 'ai_model_path', '')
    if not model_path:
        raise RuntimeError("未配置 AI 模型路径，无法执行模型推理")

    model = _load_ultralytics_model(model_path)

    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"无法读取图像: {image_path}")
    h, w = img.shape[:2]

    res = model.predict(source=image_path, conf=getattr(config.app_config, 'ai_confidence_threshold', 0.25), max_det=getattr(config.app_config, 'ai_max_detections', 300))
    if not res:
        return []

    r = res[0]
    boxes = getattr(r, 'boxes', None)
    if boxes is None:
        return []

    try:
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy.numpy()
    except Exception:
        try:
            xyxy = boxes.xyxy.numpy()
        except Exception:
            xyxy = []

    try:
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf.numpy()
    except Exception:
        try:
            confs = boxes.conf.numpy()
        except Exception:
            confs = [1.0] * len(xyxy)

    try:
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls.numpy()
    except Exception:
        try:
            clss = boxes.cls.numpy()
        except Exception:
            clss = [0] * len(xyxy)

    anns: List[AnnotationItem] = []
    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = map(float, box[:4])
        conf = float(confs[i]) if i < len(confs) else 1.0
        cls_id = int(clss[i]) if i < len(clss) else 0

        points = [
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2)
        ]
        ann = AnnotationItem(
            bbox_type=BBoxType.HORIZONTAL,
            class_id=cls_id,
            class_name=str(cls_id),
            confidence=conf,
            points=points
        )
        anns.append(ann)

    return anns


def auto_annotate_all_images_with_model(config: Config, image_paths: List[str], progress_callback: Optional[callable] = None) -> Dict[str, List[AnnotationItem]]:
    results: Dict[str, List[AnnotationItem]] = {}
    total = len(image_paths)
    for idx, img_path in enumerate(image_paths, start=1):
        try:
            anns = auto_annotate_image_with_model(config, img_path)
            results[img_path] = anns
        except Exception:
            results[img_path] = []

        if progress_callback:
            try:
                progress_callback(idx, total)
            except Exception:
                pass

    return results


def auto_annotate_image_from_labels(config: Config, image_path: str):
    """仅使用已有标注（labels 目录）迁移到目标图像，不涉及模型推理。

    返回 (annotations_list, source_image_path_or_None, score_float)
    """
    try:
        anns, src, score = auto_annotate_from_labeled_images(config, image_path)
        return anns, src, score
    except Exception:
        return [], None, 0.0


def auto_annotate_all_images_from_labels(config: Config, image_paths: List[str], progress_callback: Optional[callable] = None) -> Dict[str, dict]:
    """对多张图片使用已有标注迁移方法。返回 {image_path: {'annotations': [...], 'source': src, 'score': score}}"""
    results: Dict[str, dict] = {}
    total = len(image_paths)
    for idx, img_path in enumerate(image_paths, start=1):
        try:
            anns, src, score = auto_annotate_from_labeled_images(config, img_path)
            results[img_path] = {'annotations': anns, 'source': src, 'score': float(score)}
        except Exception:
            results[img_path] = {'annotations': [], 'source': None, 'score': 0.0}

        if progress_callback:
            try:
                progress_callback(idx, total)
            except Exception:
                pass

    return results


