import cv2
import os
import glob
from pathlib import Path
from yolov8 import YOLOv8

def batch_detect_images(
    model_path,
    input_dir,          # 输入图片文件夹路径
    output_dir,         # 检测结果保存文件夹
    conf_thres=0.2,
    iou_thres=0.3,
    img_formats=["jpg", "jpeg", "png", "bmp"]  # 支持的图片格式
):
    """
    批量检测文件夹下的所有图片
    Args:
        model_path: ONNX模型路径
        input_dir: 输入图片文件夹
        output_dir: 输出结果文件夹（保存带检测框的图片）
        conf_thres: 置信度阈值
        iou_thres: NMS IoU阈值
        img_formats: 支持的图片格式列表
    """
    # 1. 创建输出文件夹（不存在则创建）
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 2. 初始化YOLOv8检测器
    print(f"初始化YOLOv8模型：{model_path}")
    yolov8_detector = YOLOv8(model_path, conf_thres=conf_thres, iou_thres=iou_thres)
    
    # 3. 获取文件夹下所有指定格式的图片
    img_paths = []
    for fmt in img_formats:
        img_paths.extend(glob.glob(os.path.join(input_dir, f"*.{fmt}")))
        img_paths.extend(glob.glob(os.path.join(input_dir, f"*.{fmt.upper()}")))  # 兼容大写格式（如JPG）
    
    if not img_paths:
        print(f"错误：在 {input_dir} 中未找到支持的图片格式（{img_formats}）")
        return
    
    # 4. 批量检测每张图片
    total = len(img_paths)
    print(f"开始检测：共找到 {total} 张图片")
    for idx, img_path in enumerate(img_paths, 1):
        try:
            # 读取图片
            img = cv2.imread(img_path)
            if img is None:
                print(f"[{idx}/{total}] 跳过：无法读取图片 {img_path}")
                continue
            
            # 检测物体
            boxes, scores, class_ids = yolov8_detector(img)
            
            # 绘制检测框
            combined_img = yolov8_detector.draw_detections(img)
            
            # 保存检测结果（保持原文件名）
            img_name = os.path.basename(img_path)
            save_path = os.path.join(output_dir, img_name)
            cv2.imwrite(save_path, combined_img)
            
            # 打印进度和检测结果
            det_count = len(boxes) if boxes is not None and len(boxes) > 0 else 0
            print(f"[{idx}/{total}] 完成：{img_name} → 检测到 {det_count} 个物体 → 保存至 {save_path}")
        
        except Exception as e:
            print(f"[{idx}/{total}] 错误：处理 {img_path} 时出错 → {str(e)}")
    
    print(f"\n批量检测完成！所有结果已保存至：{output_dir}")

# ====================== 配置参数（根据你的实际路径修改）======================
if __name__ == "__main__":
    # 核心配置（修改为你的实际路径）
    MODEL_PATH = "/root/ultralytics/best.onnx"          # 你的ONNX模型路径
    INPUT_DIR = "/root/ultralytics/test_images"          # 输入图片文件夹（放待检测的图片）
    OUTPUT_DIR = "/root/yolov8_python/result"    # 输出结果文件夹（自动创建）
    
    # 运行批量检测
    batch_detect_images(
        model_path=MODEL_PATH,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        conf_thres=0.2,
        iou_thres=0.3
    )