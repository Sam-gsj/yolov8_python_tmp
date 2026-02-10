import numpy as np
import cv2

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def multiclass_nms(boxes, scores, class_ids, iou_threshold):

    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices,:]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

    # Draw bounding boxes and labels of detections
    for class_id, box, score in zip(class_ids, boxes, scores):
        color = colors[class_id]

        draw_box(det_img, box, color)

        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        draw_text(det_img, caption, box, color, font_size, text_thickness)

    return det_img


def draw_box( image: np.ndarray, box: np.ndarray, color: tuple[int, int, int] = (0, 0, 255),
             thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(image: np.ndarray, text: str, box: np.ndarray, color: tuple[int, int, int] = (0, 0, 255),
              font_size: float = 0.001, text_thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=font_size, thickness=text_thickness)
    th = int(th * 1.2)

    cv2.rectangle(image, (x1, y1),
                  (x1 + tw, y1 - th), color, -1)

    return cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness, cv2.LINE_AA)

def draw_masks(image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, mask_alpha: float = 0.3) -> np.ndarray:
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for box, class_id in zip(boxes, classes):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)

def save_preprocessed_to_txt(im, save_path):
    import os
    """
    通用化保存任意维度 NumPy 数组到 TXT（适配 (1,8,8400) 推理输出）
    Args:
        im (np.ndarray): 任意维度的 NumPy 数组（如 (1,8,8400)）
        save_path (str): 保存目录/文件前缀
    """
    # 核心修复1：强制转为 NumPy 数组（兼容 Tensor）
    if not isinstance(im, np.ndarray):
        im = im.cpu().detach().numpy()
    
    # 创建保存目录
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 生成文件名（适配单批次）
    txt_path = f"{save_path}.txt" if len(im.shape) == 3 else f"{save_path}_batch_0.txt"
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        # 第一步：写入数组基本信息
        f.write(f"=== 数组基本信息 ===\n")
        f.write(f"形状: {im.shape}\n")
        f.write(f"数据类型: {im.dtype}\n")
        f.write(f"数值范围: [{np.min(im):.6f}, {np.max(im):.6f}]\n")
        f.write(f"====================\n\n")
        
        # 核心修复2：通用化遍历（按维度逐层写入，适配任意形状）
        f.write(f"=== 数组数值 ===\n")
        # 展平为一维（或按原维度分行），避免嵌套遍历报错
        im_flat = im.flatten()
        # 按每行 20 个数值排版（避免单行过长）
        line_length = 20
        for i in range(0, len(im_flat), line_length):
            chunk = im_flat[i:i+line_length]
            row_str = " ".join([f"{val:.6f}" for val in chunk])  # 此时 chunk 是一维数组，可遍历
            f.write(row_str + "\n")
    
    print(f"数组已保存到: {txt_path}")


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding: bool = True, xywh: bool = False):
    """
    纯NumPy实现：将边界框从 img1_shape 缩放到 img0_shape（对齐原逻辑）
    支持 xyxy/xywh 格式，处理 LetterBox 填充的反向缩放
    
    Args:
        img1_shape (tuple): 源图像形状 (height, width)（模型输入尺寸）
        boxes (np.ndarray): 待缩放的边界框，形状 (N, 4)
        img0_shape (tuple): 目标图像形状 (height, width)（原始图像尺寸）
        ratio_pad (tuple, optional): (ratio, pad) 缩放参数，None 则自动计算
        padding (bool): 是否考虑 LetterBox 填充的偏移
        xywh (bool): 框格式是否为 xywh（True），否则为 xyxy（False）
    
    Returns:
        np.ndarray: 缩放后的边界框，格式与输入一致
    """
    # 转换为NumPy数组（兼容其他输入类型）
    boxes = np.array(boxes, dtype=np.float32)
    
    # 计算缩放比例和填充量
    if ratio_pad is None:  # 自动计算 gain 和 pad
        # gain = 模型输入尺寸 / 原始图像缩放后的尺寸（old/new）
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        # 计算LetterBox的填充量（左右/上下）
        pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
    else:
        gain = ratio_pad[0][0]
        pad_x, pad_y = ratio_pad[1]

    # 反向抵消LetterBox的填充偏移
    if padding:
        boxes[..., 0] -= pad_x  # x1 (或xywh的x) 抵消水平填充
        boxes[..., 1] -= pad_y  # y1 (或xywh的y) 抵消垂直填充
        if not xywh:  # xyxy格式需要同时调整x2/y2
            boxes[..., 2] -= pad_x
            boxes[..., 3] -= pad_y
    
    # 按缩放比例反向缩放（恢复到原始图像尺寸）
    boxes[..., :4] /= gain
    
    # xyxy格式需要裁剪到图像边界，xywh格式直接返回
    return boxes if xywh else clip_boxes(boxes, img0_shape)

def clip_boxes(boxes, shape):
    """
    纯NumPy实现：将边界框裁剪到图像边界内（防止框超出图像）
    
    Args:
        boxes (np.ndarray): 待裁剪的边界框，形状 (N, 4)（xyxy格式）
        shape (tuple): 图像形状 (height, width) 或 (height, width, channel)
    
    Returns:
        np.ndarray: 裁剪后的边界框
    """
    # 提取图像高宽（兼容HWC/HW格式）
    h, w = shape[:2]
    
    # 批量裁剪：将x1/x2限制在[0, w]，y1/y2限制在[0, h]
    # 比逐元素裁剪更快（NumPy向量化操作）
    boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, w)  # x1, x2
    boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, h)  # y1, y2
    
    return boxes