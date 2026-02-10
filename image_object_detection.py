import cv2
from yolov8 import YOLOv8

# Initialize yolov8 object detector
model_path = "/root/ultralytics/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3,engine='openvino')

# Read image
img = cv2.imread("/root/ultralytics/383d20adc54423d5e3cc09746624c440.jpg")


# Detect Objects
boxes, scores, class_ids = yolov8_detector(img)

# Draw detections
combined_img = yolov8_detector.draw_detections(img)

cv2.imwrite("doc/img/detected_objects.jpg", combined_img)

