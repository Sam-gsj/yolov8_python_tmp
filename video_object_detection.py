import cv2
from yolov8 import YOLOv8

def yolov8_video_detection(input_video_path, output_video_path, model_path, conf_thres=0.2, iou_thres=0.3):
    """
    使用YOLOv8对视频流进行目标检测，并保存检测结果为视频
    
    Args:
        input_video_path: 输入视频路径 (如果使用摄像头，传入 0)
        output_video_path: 输出视频保存路径
        model_path: YOLOv8 ONNX模型路径
        conf_thres: 置信度阈值
        iou_thres: IOU阈值
    """
    # 初始化YOLOv8检测器
    yolov8_detector = YOLOv8(model_path, conf_thres=conf_thres, iou_thres=iou_thres, engine='openvino')
    
    # 打开视频流
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件或摄像头，请检查路径是否正确")
    
    # 获取视频基本信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 视频帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频高度
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
    
    # 创建视频写入对象
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"开始处理视频，帧率：{fps}，分辨率：{width}x{height}")
    
    # 逐帧处理视频
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        
        # 读取帧失败（到达视频末尾）
        if not ret:
            break
        
        # 目标检测
        boxes, scores, class_ids = yolov8_detector(frame)
        
        # 绘制检测框
        combined_frame = yolov8_detector.draw_detections(frame)
        
        # 写入结果视频
        out.write(combined_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count} 帧")
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"视频处理完成！共处理 {frame_count} 帧，结果已保存至：{output_video_path}")

# 主程序入口
if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = "/root/yolov8_python/models/last.onnx"  # YOLOv8模型路径
    INPUT_VIDEO = "/root/yolov8_python/QQ20260213-010128.mp4"  # 输入视频路径（摄像头用0）
    OUTPUT_VIDEO = "/root/yolov8_python/doc/output_detected10.mp4"  # 输出视频路径
    
    # 执行视频检测
    yolov8_video_detection(
        input_video_path=INPUT_VIDEO,
        output_video_path=OUTPUT_VIDEO,
        model_path=MODEL_PATH,
        conf_thres=0.4,
        iou_thres=0.3
    )