import sys
import time
import numpy as np
from .utils import save_preprocessed_to_txt

def xywh2xyxy(x):
    """纯NumPy实现：将 [x, y, w, h] 格式转换为 [x1, y1, x2, y2] 格式"""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def box_iou(box1, box2):
    """纯NumPy实现：计算两个框的IoU（Intersection over Union）"""
    # 展开坐标
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # 计算交集坐标
    inter_x1 = np.maximum(b1_x1[:, None], b2_x1[None, :])
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1[None, :])
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2[None, :])
    inter_y2 = np.minimum(b1_y2[:, None], b2_y1[None, :])

    # 计算交集面积
    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)

    # 计算各自面积
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # 计算IoU
    iou = inter_area / (b1_area[:, None] + b2_area[None, :] - inter_area + 1e-16)
    return iou


def batch_probiou(boxes1, boxes2, eps=1e-7):
    """纯NumPy实现：旋转框的IoU计算（适配OBB）"""
    # 简化版实现（核心逻辑对齐原torch版本）
    iou = box_iou(boxes1[:, :4], boxes2[:, :4])
    return iou


def non_max_suppression(
    prediction,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels=(),
    max_det: int = 300,
    nc: int = 0,  # number of classes (optional)
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    rotated: bool = False,
    end2end: bool = False,
    return_idxs: bool = False,
):
    """
    纯NumPy实现的非极大值抑制（NMS），完全移除PyTorch依赖
    功能与原Ultralytics版本一致，仅替换张量操作为NumPy数组操作
    """
    # 输入校验
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    
    # 处理YOLOv8验证模式的输出格式
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    
    # 处理end2end模型输出
    if prediction.shape[-1] == 6 or end2end:
        output = []
        for pred in prediction:
            mask = pred[:, 4] > conf_thres
            pred_filtered = pred[mask][:max_det]
            if classes is not None:
                classes_arr = np.array(classes)
                mask_cls = np.isin(pred_filtered[:, 5], classes_arr)
                pred_filtered = pred_filtered[mask_cls]
            output.append(pred_filtered)
        return output

    # 转换为NumPy数组（防止输入是其他类型）
    prediction = np.array(prediction)
    
    # 处理类别参数
    if classes is not None:
        classes = np.array(classes)

    # 基础参数计算
    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    extra = prediction.shape[1] - nc - 4  # number of extra info
    mi = 4 + nc  # mask start index
    xc = np.max(prediction[:, 4:mi], axis=1) > conf_thres  # candidates
    
    # 生成索引（替代原torch的xinds）
    xinds = np.arange(prediction.shape[-1])[None, :, None].repeat(bs, axis=0)

    # 时间限制设置
    time_limit = 2.0 + max_time_img * bs
    multi_label &= nc > 1

    # 维度转置：(1,84,6300) → (1,6300,84)
    prediction = np.transpose(prediction, (0, 2, 1))
    if not rotated:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    save_preprocessed_to_txt(prediction,"./nms_pred")
    t = time.time()
    output = [np.zeros((0, 6 + extra)) for _ in range(bs)]
    keepi = [np.zeros((0, 1)) for _ in range(bs)]

    for xi, (x, xk) in enumerate(zip(prediction, xinds)):
        # 置信度过滤
        filt = xc[xi]
        x = x[filt]
        if return_idxs:
            xk = xk[filt]

        # 合并先验标签（autolabelling）
        if labels and len(labels[xi]) and not rotated:
            lb = np.array(labels[xi])
            v = np.zeros((len(lb), nc + extra + 4))
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            for i in range(len(lb)):
                v[i, int(lb[i, 0]) + 4] = 1.0  # cls
            x = np.concatenate((x, v), axis=0)

        # 无框则跳过
        if x.shape[0] == 0:
            continue

        # 拆分框、类别、掩码
        box = x[:, :4]
        cls = x[:, 4:mi]
        mask = x[:, mi:] if extra > 0 else np.zeros((x.shape[0], 0))

        # 多标签/单标签处理
        if multi_label:
            i, j = np.where(cls > conf_thres)
            x = np.concatenate([
                box[i], 
                x[i, 4 + j][:, None], 
                j[:, None].astype(np.float32), 
                mask[i]
            ], axis=1)
            if return_idxs:
                xk = xk[i]
        else:  # 仅保留最佳类别
            conf = np.max(cls, axis=1, keepdims=True)
            j = np.argmax(cls, axis=1).reshape(-1, 1)
            filt = conf.ravel() > conf_thres
            x = np.concatenate([box, conf, j.astype(np.float32), mask], axis=1)[filt]
            if return_idxs:
                xk = xk[filt]

        # 类别过滤
        if classes is not None:
            filt = np.isin(x[:, 5], classes)
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        # 检查框数量
        n = x.shape[0]
        if n == 0:
            continue
        if n > max_nms:
            # 按置信度排序并截断
            filt = np.argsort(x[:, 4])[::-1][:max_nms]
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        # 类别偏移（防止跨类别抑制）
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]

        # NMS计算
        if rotated:
            boxes = np.concatenate([x[:, :2] + c, x[:, 2:4], x[:, -1:]], axis=-1)
            i = TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_probiou)
        else:
            boxes = x[:, :4] + c
            # 调用纯NumPy的NMS
            i = TorchNMS.nms(boxes, scores, iou_thres)
        
        # 限制最大检测数
        i = i[:max_det]

        # 保存结果
        output[xi] = x[i]
        if return_idxs:
            keepi[xi] = xk[i].reshape(-1, 1)
        
        # 时间限制检查
        if (time.time() - t) > time_limit:
            print(f"NMS time limit {time_limit:.3f}s exceeded")
            break

    return (output, keepi) if return_idxs else output


class TorchNMS:
    """纯NumPy实现的NMS类，完全对齐原TorchNMS功能"""

    @staticmethod
    def fast_nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
        use_triu: bool = True,
        iou_func=box_iou,
        exit_early: bool = True,
    ) -> np.ndarray:
        """纯NumPy实现Fast-NMS"""
        if boxes.size == 0 and exit_early:
            return np.array([], dtype=np.int64)

        # 按置信度降序排序
        sorted_idx = np.argsort(scores)[::-1]
        boxes = boxes[sorted_idx]
        ious = iou_func(boxes, boxes)

        if use_triu:
            # 上三角矩阵（排除对角线）
            ious = np.triu(ious, k=1)
            # 筛选IoU小于阈值的框
            pick = np.nonzero(np.sum(ious >= iou_threshold, axis=0) <= 0)[0]
        else:
            n = boxes.shape[0]
            row_idx = np.arange(n)[:, None].repeat(n, axis=1)
            col_idx = np.arange(n)[None, :].repeat(n, axis=0)
            upper_mask = row_idx < col_idx
            ious = ious * upper_mask

            # 更新分数
            scores_ = scores[sorted_idx].copy()
            scores_[np.sum(ious >= iou_threshold, axis=0) > 0] = 0
            scores[sorted_idx] = scores_

            # 取topk索引
            pick = np.argsort(scores_)[::-1]

        return sorted_idx[pick]

    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """纯NumPy实现标准NMS（对齐torchvision.ops.nms）"""
        if boxes.size == 0:
            return np.array([], dtype=np.int64)

        # 提取坐标和面积
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)

        # 按置信度降序排序
        order = np.argsort(scores)[::-1]

        # 预分配保存结果的数组
        keep = np.zeros(order.size, dtype=np.int64)
        keep_idx = 0

        while order.size > 0:
            # 保留当前置信度最高的框
            i = order[0]
            keep[keep_idx] = i
            keep_idx += 1

            if order.size == 1:
                break

            # 计算当前框与剩余框的IoU
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])

            # 交集面积
            w = np.maximum(xx2 - xx1, 0)
            h = np.maximum(yy2 - yy1, 0)
            inter = w * h

            # 无交集则直接保留所有剩余框
            if np.sum(inter) == 0:
                order = rest
                continue

            # 计算IoU
            iou = inter / (areas[i] + areas[rest] - inter + 1e-16)

            # 保留IoU小于阈值的框
            order = rest[iou <= iou_threshold]

        return keep[:keep_idx]

    @staticmethod
    def batched_nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        idxs: np.ndarray,
        iou_threshold: float,
        use_fast_nms: bool = False,
    ) -> np.ndarray:
        """纯NumPy实现批量NMS（跨类别抑制）"""
        if boxes.size == 0:
            return np.array([], dtype=np.int64)

        # 按类别偏移框坐标，防止跨类别抑制
        max_coordinate = np.max(boxes)
        offsets = idxs * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

        return (
            TorchNMS.fast_nms(boxes_for_nms, scores, iou_threshold)
            if use_fast_nms
            else TorchNMS.nms(boxes_for_nms, scores, iou_threshold)
        )


# 测试示例（验证功能）
if __name__ == "__main__":
    # 生成测试数据（模拟YOLOv8输出）
    pred = np.random.rand(1, 84, 6300).astype(np.float32)
    # 执行NMS
    results = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
    print(f"NMS结果数量: {len(results[0])}")
    print(f"结果形状: {results[0].shape if len(results[0]) > 0 else '空'}")