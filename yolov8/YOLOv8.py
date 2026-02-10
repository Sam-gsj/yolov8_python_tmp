import time
import cv2
import numpy as np
import onnxruntime
import os  # 新增：用于创建保存目录
from typing import Any, List
from .nms import non_max_suppression

from .utils import xywh2xyxy, scale_boxes, draw_detections, save_preprocessed_to_txt



class LetterBox:
    """对齐 Ultralytics 的 LetterBox 实现（保持比例+填充）"""
    def __init__(
        self,
        new_shape: tuple[int, int] = (640, 640),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
        padding_value: int = 114,
        interpolation: int = cv2.INTER_LINEAR,
    ):
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center
        self.padding_value = padding_value
        self.interpolation = interpolation

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """执行 LetterBox 变换（保持比例+填充）"""
        shape = image.shape[:2]  # 原始形状 [h, w]
        new_shape = self.new_shape
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # 计算缩放比例（保持比例，取最小缩放因子）
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # 只允许缩小，不允许放大
            r = min(r, 1.0)

        # 计算缩放后的尺寸和填充量
        new_unpad = round(shape[1] * r), round(shape[0] * r)  # 缩放后的宽、高
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 宽、高方向填充量
        
        if self.auto:  # 最小矩形填充（对齐 stride）
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scale_fill:  # 拉伸填充（和你原来的逻辑一致，仅兜底）
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])

        # 均分填充（居中显示）
        if self.center:
            dw /= 2
            dh /= 2

        # 缩放图片
        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=self.interpolation)
        
        # 计算上下左右填充量
        top, bottom = round(dh - 0.1) if self.center else 0, round(dh + 0.1)
        left, right = round(dw - 0.1) if self.center else 0, round(dw + 0.1)
        
        # 执行填充
        if image.ndim == 3 and image.shape[-1] == 3:
            image = cv2.copyMakeBorder(
                image, top, bottom, left, right,
                cv2.BORDER_CONSTANT,
                value=(self.padding_value,) * 3
            )
        else:  # 单通道/多光谱图
            pad_img = np.full(
                (image.shape[0] + top + bottom, image.shape[1] + left + right, image.shape[-1]),
                fill_value=self.padding_value,
                dtype=image.dtype
            )
            pad_img[top:top+image.shape[0], left:left+image.shape[1]] = image
            image = pad_img

        return image

class YOLOv8:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.input_height = None
        self.input_width = None
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(
            path,
            providers=['CPUExecutionProvider']
        )
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        input_tensor = self.preprocess(image, save_to_txt=True)  # 开启保存 TXT
        outputs = self.inference(input_tensor)
        save_preprocessed_to_txt(outputs[0], "./infer")

        self.boxes, self.scores, self.class_ids = self.process_output(outputs)
        return self.boxes, self.scores, self.class_ids

    def pre_transform(self, im_list: List[np.ndarray]) -> List[np.ndarray]:
            """
            对齐 Ultralytics 的 pre_transform 逻辑（LetterBox 变换）
            Args:
                im_list: 图片列表 [(H, W, 3) x N]
            Returns:
                变换后的图片列表
            """
            # 1. 判断所有图片是否形状相同（决定 auto 参数）
            same_shapes = len({x.shape for x in im_list}) == 1
            new_shape=(self.input_height, self.input_width)
            
            # 2. 初始化 LetterBox（对齐 ultralytics 核心参数）
            letterbox = LetterBox(
                new_shape,  # 模型输入尺寸
                auto=False,  # 仅当所有图片形状相同时启用 auto 填充
                stride=32,         # YOLO 系列默认 stride=32
            )
            
            # 3. 对每张图片执行 LetterBox 变换
            processed = [letterbox(image=im) for im in im_list]
            return processed

    def preprocess(self, im, save_to_txt=False, save_path="./preprocessed_im"):
        """
        推理前的图片预处理（完全基于 NumPy/OpenCV，无 torch 依赖）
        Args:
            im (np.ndarray | list[np.ndarray]): 单张图片 (H, W, 3) 或多张图片列表 [(H, W, 3) x N]
            save_to_txt (bool): 是否保存预处理后的数组到 TXT 文件
            save_path (str): 保存目录/文件前缀（默认: ./preprocessed_im）
        Returns:
            np.ndarray: 预处理后的张量，形状为 (N, 3, H, W)，数据类型 float32，值范围 0.0-1.0
        """
        # 处理输入类型
        if isinstance(im, np.ndarray):
            im = [im]
        elif not isinstance(im, list):
            raise TypeError(f"输入必须是 np.ndarray 或 list[np.ndarray]，当前类型: {type(im)}")
        
        if len(im) == 1:
            self.img_height, self.img_width = im[0].shape[0], im[0].shape[1]
        
        # 调整尺寸
        im = self.pre_transform(im)
        # 堆叠为批量数组 (N, H, W, 3)
        im = np.stack(im)
        # BGR 转 RGB
        if im.shape[-1] == 3:
            im = im[..., ::-1]
        # 维度转换 (N, H, W, 3) → (N, 3, H, W)
        im = im.transpose((0, 3, 1, 2))
        # 确保内存连续
        im = np.ascontiguousarray(im)
        # 数据类型转换 + 归一化
        im = im.astype(np.float32)
        im /= 255.0
        
        # 记录原始图片尺寸

        
        # 保存为 TXT 格式
        if save_to_txt:
            save_preprocessed_to_txt(im, save_path)
        
        return im

    def save_preprocessed_to_txt(self, im, save_path):
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

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        print(f"Inference time (CPU): {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        """
        处理推理输出：执行 NMS 并将坐标还原回原始图像尺寸
        核心改动：
        1. 先处理batch维度，再转为NumPy数组（避免维度混乱）
        2. 正确判断NumPy数组是否为空（无布尔歧义）
        3. 保留强制转NumPy的需求，同时保证逻辑正确
        """
        # 1. 执行 NMS
        # NMS返回：list[np.ndarray] → 每个元素是单张图的检测结果 (N,6)
        predictions = non_max_suppression(
            output,
            self.conf_threshold,
            self.iou_threshold
        )

        # 2. 处理batch维度（假设batch_size=1，取第一张图的结果）
        if not predictions:  # 先判断列表是否为空（Python list可正常判断）
            return np.array([]), np.array([]), np.array([])
        
        # 取第一张图的结果，并强制转为NumPy数组（满足你的核心需求）
        det = np.array(predictions[0], dtype=np.float32)  # 单张图结果转NumPy

        # 3. 正确判断NumPy数组是否为空（核心修复：避免布尔歧义）
        if det.size == 0:  # 用size判断空数组，替代if not det
            return np.array([]), np.array([]), np.array([])

        # 4. 将模型输入尺寸的框缩放到原始图像尺寸
        det[:, :4] = scale_boxes(
            img1_shape=(self.input_height, self.input_width),
            boxes=det[:, :4],
            img0_shape=(self.img_height, self.img_width)
        )

        # 5. 分离数据（此时det是(N,6)的NumPy数组）
        boxes = det[:, :4]          # 原始尺寸的框 (N,4)
        scores = det[:, 4]          # 置信度 (N,)
        class_ids = det[:, 5].astype(int)  # 类别索引 (N,)

        return boxes, scores, class_ids

    def extract_boxes(self, predictions):
        boxes = predictions[:, :4]
        boxes = self.rescale_boxes(boxes)
        boxes = xywh2xyxy(boxes)
        return boxes

    def rescale_boxes(self, boxes):
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def construct_results(self, preds, img, orig_imgs):
        """Construct a list of Results objects from model predictions.

        Args:
            preds (list[torch.Tensor]): List of predicted bounding boxes and scores for each image.
            img (torch.Tensor): Batch of preprocessed images used for inference.
            orig_imgs (list[np.ndarray]): List of original images before preprocessing.

        Returns:
            (list[Results]): List of Results objects containing detection information for each image.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """Construct a single Results object from one image prediction.

        Args:
            pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
            img (torch.Tensor): Preprocessed image tensor used for inference.
            orig_img (np.ndarray): Original image before preprocessing.
            img_path (str): Path to the original image file.

        Returns:
            (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
        """
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])

if __name__ == '__main__':
    # 测试代码
    from imread_from_url import imread_from_url
    img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
    img = imread_from_url(img_url)

    if img is None:
        print("图片加载失败，请检查路径或网络")
    else:
        model_path = "../models/yolov8m.onnx"
        yolov8_detector = YOLOv8(model_path, conf_thres=0.3, iou_thres=0.5)

        # 推理（自动保存预处理结果到 TXT）
        yolov8_detector(img)

        # 手动保存示例（可选）
        # preprocessed_im = yolov8_detector.preprocess(img, save_to_txt=True, save_path="./my_preprocessed_im")

        # 绘制检测结果
        combined_img = yolov8_detector.draw_detections(img)
        cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        cv2.imshow("Output", combined_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()