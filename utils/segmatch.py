import numpy as np
from sklearn.cluster import DBSCAN
from scipy.ndimage import label
import cv2

#分割前景
def segment_events_by_flow(event_cnt, flow, threshold=0.01):
    # event_cnt: [H, W] 或 [C, H, W]
    # flow: [2, H, W]
    # mask: [H, W]
    event_cnt = np.abs(event_cnt).sum(0)
    valid = (event_cnt > 0)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    # 提取点光流
    event_flow = flow[:, ys, xs].T  # [N_valid, 2]
    # 估计背景光流、
    bg_flow = np.mean(event_flow, axis=0)
    # 计算欧氏距离
    flow_diff = np.linalg.norm(event_flow - bg_flow, axis=1)
    # 分割
    fg_mask = np.zeros_like(event_cnt, dtype=bool)
    fg_mask[ys, xs] = flow_diff > threshold
    #bg_idx = event_idx[flow_diff <= threshold] # 背景（相机自运动）

    # 计算连通分量并过滤闪烁点
    labeled, num = label(fg_mask)
    keep_mask = np.zeros_like(fg_mask, dtype=bool)
    for i in range(1, num+1):
        region = (labeled == i)
        if region.sum() >= 5:
            # 保留面积大于min_size 5的区域
            keep_mask |= region

    kernel = np.ones((3, 3), np.uint8)
    keep_mask_dilated = cv2.dilate(keep_mask.astype(np.uint8), kernel, iterations=1)
    keep_mask_connected = np.logical_or(keep_mask, keep_mask_dilated)

    ys_fg, xs_fg = np.where(keep_mask_connected)
    return ys_fg, xs_fg

# 前景事件聚类
def cluster_moving_objects( ys_fg, xs_fg, eps=10, min_samples=3):
    if len(xs_fg) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)  # 无前景事件
    features = np.stack([xs_fg, ys_fg], axis=1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    labels = clustering.labels_
    valid_labels = labels[labels != -1]
    if len(valid_labels) > 0:
        # 统计每个类别的数量
        unique, counts = np.unique(valid_labels, return_counts=True)
        max_label = unique[np.argmax(counts)]
        main_mask = (labels == max_label)
        # 选出最大类别的事件索引
        main_ys = ys_fg[main_mask]
        main_xs = xs_fg[main_mask]
    else:
        main_ys, main_xs = np.array([], dtype=int), np.array([], dtype=int)
    return main_ys, main_xs  # 返回每个前景事件的聚类标签
