import numpy as np
import re

def read_txt_to_numpy(txt_path, has_metadata=True):
    """
    通用化读取 TXT 文件并还原为 NumPy 数组（兼容图片/推理输出格式）
    Args:
        txt_path: TXT 文件路径
        has_metadata: 是否包含元信息（形状/数据类型等）
    Returns:
        np.ndarray: 还原后的数组
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not has_metadata:
        # 纯数值格式：直接读取所有行转为数组
        data = []
        for line in lines:
            nums = [float(x) for x in line.split()]
            data.extend(nums)
        return np.array(data, dtype=np.float32)
    
    # ========== 核心修改：兼容两种元信息格式 ==========
    # 1. 提取形状信息（适配 "整体形状"/"形状 (通道数...)" 两种字符串）
    shape_str = None
    # 匹配 "整体形状: (1, 8, 8400)" 或 "形状 (通道数, 高度, 宽度): (3, 640, 640)"
    shape_pattern = re.compile(r'形状.*: \(([\d, ]+)\)')
    for line in lines:
        match = shape_pattern.search(line)
        if match:
            # 提取括号内的数字，转为元组
            shape_str = match.group(1).replace(' ', '')
            shape = tuple(map(int, shape_str.split(',')))
            break
    
    if shape_str is None:
        # 未找到形状行 → 按纯数值读取（展平数组）
        shape = None
    # ========== 提取数值部分 ==========
    # 找到数值开始的行（跳过元信息，匹配 "=== 数组数值 ===" 或 "--- R (红) 通道 ---"）
    data_start_idx = 0
    for idx, line in enumerate(lines):
        if "=== 数组数值" in line or "--- R (红) 通道" in line:
            data_start_idx = idx + 1
            break
    
    # 读取数值行
    data_lines = lines[data_start_idx:]
    data = []
    for line in data_lines:
        # 跳过通道分隔行（如 "--- G (绿) 通道 ---"）
        if line.startswith("---") and "通道" in line:
            continue
        nums = [float(x) for x in line.split()]
        data.extend(nums)
    data_np = np.array(data, dtype=np.float32)
    
    # 还原形状（如果能提取到）
    if shape is not None and len(data_np) == np.prod(shape):
        data_np = data_np.reshape(shape)
    
    return data_np

def compare_numpy_txt(txt1_path, txt2_path, rtol=1e-5, atol=1e-8, has_metadata=True):
    """
    对比两个 NumPy 数组保存的 TXT 文件（兼容任意元信息格式）
    Args:
        txt1_path: 第一个 TXT 文件路径
        txt2_path: 第二个 TXT 文件路径
        rtol: 相对误差阈值
        atol: 绝对误差阈值
        has_metadata: 是否包含元信息
    Returns:
        bool: 是否一致
    """
    print(f"===== 对比 {txt1_path} 和 {txt2_path} =====")
    
    # 1. 读取并还原数组（增加异常捕获）
    try:
        arr1 = read_txt_to_numpy(txt1_path, has_metadata)
    except Exception as e:
        print(f"❌ 读取第一个文件失败: {e}")
        return False
    try:
        arr2 = read_txt_to_numpy(txt2_path, has_metadata)
    except Exception as e:
        print(f"❌ 读取第二个文件失败: {e}")
        return False
    
    # 2. 校验形状（如果形状不一致，尝试展平后对比）
    if arr1.shape != arr2.shape:
        print(f"⚠️  数组形状不一致！")
        print(f"  - 第一个数组形状: {arr1.shape}")
        print(f"  - 第二个数组形状: {arr2.shape}")
        print(f"  → 尝试展平后对比数值...")
        arr1_flat = arr1.flatten()
        arr2_flat = arr2.flatten()
        if len(arr1_flat) != len(arr2_flat):
            print(f"❌ 展平后长度仍不一致！{len(arr1_flat)} vs {len(arr2_flat)}")
            return False
    else:
        print(f"✅ 数组形状一致: {arr1.shape}")
        arr1_flat = arr1.flatten()
        arr2_flat = arr2.flatten()
    
    # 3. 校验数据类型
    if arr1.dtype != arr2.dtype:
        print(f"⚠️  数据类型不一致，统一转为 float32 对比")
        arr1_flat = arr1_flat.astype(np.float32)
        arr2_flat = arr2_flat.astype(np.float32)
    else:
        print(f"✅ 数据类型一致: {arr1.dtype}")
    
    # 4. 校验数值（允许浮点误差）
    is_close = np.allclose(arr1_flat, arr2_flat, rtol=rtol, atol=atol)
    if is_close:
        print(f"✅ 所有数值一致（误差阈值: rtol={rtol}, atol={atol}）")
        return True
    else:
        print(f"❌ 存在数值差异！")
        # 找出前10个差异位置
        diff_mask = ~np.isclose(arr1_flat, arr2_flat, rtol=rtol, atol=atol)
        diff_indices = np.argwhere(diff_mask)[:10]
        print(f"  前10个差异位置及数值:")
        for idx in diff_indices:
            idx = idx[0]  # 展平后是一维索引
            val1 = arr1_flat[idx]
            val2 = arr2_flat[idx]
            diff = abs(val1 - val2)
            print(f"    索引 {idx}: {val1:.6f} vs {val2:.6f} (差值: {diff:.6f})")
        return False

if __name__ == "__main__":
    # 配置项
    TXT_PATH_1 = "/root/ONNX-YOLOv8-Object-Detection/nms_pred_u.txt"
    TXT_PATH_2 = "/root/ONNX-YOLOv8-Object-Detection/nms_pred.txt"
    RELATIVE_TOLERANCE = 1e-5
    ABSOLUTE_TOLERANCE = 1e-8
    HAS_METADATA = True  # 你的文件包含元信息，设为 True
    
    # 执行对比
    result = compare_numpy_txt(
        txt1_path=TXT_PATH_1,
        txt2_path=TXT_PATH_2,
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
        has_metadata=HAS_METADATA
    )
    
    print(f"\n===== 最终结果 =====")
    if result:
        print("✅ 两个 TXT 文件对应的 NumPy 数组完全一致！")
    else:
        print("❌ 两个 TXT 文件对应的 NumPy 数组不一致！")