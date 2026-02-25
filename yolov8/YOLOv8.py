import time
import cv2
import numpy as np
import onnxruntime
import os
from typing import Any, List

try:
    from openvino.runtime import Core
except ImportError:
    Core = None

from .nms import non_max_suppression
from .utils import xywh2xyxy, scale_boxes, draw_detections

class LetterBox:
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
        shape = image.shape[:2]
        new_shape = self.new_shape
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)

        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        
        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scale_fill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])

        if self.center:
            dw /= 2
            dh /= 2

        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=self.interpolation)
        
        top, bottom = round(dh - 0.1) if self.center else 0, round(dh + 0.1)
        left, right = round(dw - 0.1) if self.center else 0, round(dw + 0.1)
        
        if image.ndim == 3 and image.shape[-1] == 3:
            image = cv2.copyMakeBorder(
                image, top, bottom, left, right,
                cv2.BORDER_CONSTANT,
                value=(self.padding_value,) * 3
            )
        else:
            pad_img = np.full(
                (image.shape[0] + top + bottom, image.shape[1] + left + right, image.shape[-1]),
                fill_value=self.padding_value,
                dtype=image.dtype
            )
            pad_img[top:top+image.shape[0], left:left+image.shape[1]] = image
            image = pad_img

        return image

class YOLOv8:
    def __init__(self, path, conf_thres=0.7, iou_thres=0.5, engine='onnx'):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.input_height = None
        self.input_width = None
        self.engine = engine.lower()
        
        if self.engine == 'openvino' and Core is None:
            raise ImportError("OpenVINO is not installed. Please run `pip install openvino`.")
            
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        if self.engine == 'onnx':
            self.session = onnxruntime.InferenceSession(
                path,
                providers=['CPUExecutionProvider']
            )
            self.get_onnx_input_details()
            self.get_onnx_output_details()
            
        elif self.engine == 'openvino':
            print(f"Loading OpenVINO model: {path}")
            self.ov_core = Core()
            self.ov_model = self.ov_core.read_model(model=path)
            self.ov_compiled_model = self.ov_core.compile_model(self.ov_model, device_name="CPU")
            self.ov_request = self.ov_compiled_model.create_infer_request()
            
            self.get_openvino_input_details()
            self.get_openvino_output_details()
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")

    def detect_objects(self, image):
        input_tensor = self.preprocess(image, save_to_txt=True)
        outputs = self.inference(input_tensor)


        self.boxes, self.scores, self.class_ids = self.process_output(outputs)
        return self.boxes, self.scores, self.class_ids

    def pre_transform(self, im_list: List[np.ndarray]) -> List[np.ndarray]:
            same_shapes = len({x.shape for x in im_list}) == 1
            new_shape=(self.input_height, self.input_width)
            
            letterbox = LetterBox(
                new_shape,
                auto=False,
                stride=32,
            )
            
            processed = [letterbox(image=im) for im in im_list]
            return processed

    def preprocess(self, im, save_to_txt=False, save_path="./preprocessed_im"):
        if isinstance(im, np.ndarray):
            im = [im]
        elif not isinstance(im, list):
            raise TypeError(f"输入必须是 np.ndarray 或 list[np.ndarray]")
        
        if len(im) == 1:
            self.img_height, self.img_width = im[0].shape[0], im[0].shape[1]
        
        im = self.pre_transform(im)
        im = np.stack(im)
        if im.shape[-1] == 3:
            im = im[..., ::-1]
        im = im.transpose((0, 3, 1, 2))
        im = np.ascontiguousarray(im)
        im = im.astype(np.float32)
        im /= 255.0
        
        return im

    

    def inference(self, input_tensor):
        start = time.perf_counter()
        
        if self.engine == 'onnx':
            outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        elif self.engine == 'openvino':
            results = self.ov_request.infer({self.input_layer: input_tensor})
            outputs = [results[self.output_layer]]
            
        print(f"Inference time ({self.engine.upper()}): {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        keep_classes = [0, 1, 2, 3,5]
        predictions = non_max_suppression(
            output,
            self.conf_threshold,
            self.iou_threshold,
            classes = keep_classes
        )

        if not predictions:
            return np.array([]), np.array([]), np.array([])
        
        det = np.array(predictions[0], dtype=np.float32)

        if det.size == 0:
            return np.array([]), np.array([]), np.array([])

        det[:, :4] = scale_boxes(
            img1_shape=(self.input_height, self.input_width),
            boxes=det[:, :4],
            img0_shape=(self.img_height, self.img_width)
        )

        boxes = det[:, :4]
        scores = det[:, 4]
        class_ids = det[:, 5].astype(int)

        return boxes, scores, class_ids

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    def get_onnx_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_onnx_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def get_openvino_input_details(self):
        self.input_layer = self.ov_compiled_model.input(0)
        self.input_shape = list(self.input_layer.shape)
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_openvino_output_details(self):
        self.output_layer = self.ov_compiled_model.output(0)