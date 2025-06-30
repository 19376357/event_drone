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

#双目匹配
def match_stereo_events(events_left, events_right, max_dist=2, max_dt=0.001):
    # 简单最近邻匹配（可用更复杂的极线约束）
    matches = []
    for i, e_left in enumerate(events_left):
        t_l, y_l, x_l, p_l = e_left
        # 只在右目中查找极性相同、时间接近、空间距离最近的点
        candidates = events_right[
            (np.abs(events_right[:, 0] - t_l) < max_dt) &
            (events_right[:, 3] == p_l)
        ]
        if len(candidates) == 0:
            continue
        dists = np.sqrt((candidates[:, 1] - y_l) ** 2 + (candidates[:, 2] - x_l) ** 2)
        min_idx = np.argmin(dists)
        if dists[min_idx] < max_dist:
            matches.append((i, min_idx))
    return matches

#三维重建
def triangulate_stereo_points(events_left, events_right, matches, fx, fy, cx, cy, B):
    """
    events_left, events_right: [N, 4] (t, y, x, p)
    matches: [(i, j), ...]，分别为左目和右目事件索引
    f: 焦距（像素单位）
    cx, cy: 主点坐标
    B: 基线（米）
    返回: points_3d [M, 3]，每个匹配点的三维坐标（相机左目坐标系）
    """
    points_3d = []
    for idx_l, idx_r in matches:
        x_l = events_left[idx_l, 2]
        y_l = events_left[idx_l, 1]
        x_r = events_right[idx_r, 2]
        y_r = events_right[idx_r, 1]
        disparity = x_l - x_r
        if disparity == 0:
            continue  # 避免除零
        Z = fx * B / disparity
        X = (x_l - cx) * Z / fx
        Y = (y_l - cy) * Z / fy
        points_3d.append([X, Y, Z])
    return np.array(points_3d)
def estimate_object_velocity(events, flow, points_3d, matches, fx,fy, Z_mean, dt=1.0):
    """
    events: 左目事件点 [N, 4]
    flow: [2, H, W]
    points_3d: [M, 3]，三维坐标
    matches: 匹配对
    f: 焦距
    Z_mean: 物体平均深度
    dt: 时间间隔（秒）
    返回: [vx, vy, vz] 空间速度估计
    """
    us = []
    vs = []
    for idx_l, _ in matches:
        y = int(events[idx_l, 1])
        x = int(events[idx_l, 2])
        u = flow[0, y, x]
        v = flow[1, y, x]
        us.append(u)
        vs.append(v)
    if len(us) == 0:
        return np.zeros(3)
    u_mean = np.mean(us)
    v_mean = np.mean(vs)
    # 像素速度转空间速度（简化版，忽略径向畸变等）
    vx = u_mean * Z_mean / fx / dt
    vy = v_mean * Z_mean / fy / dt
    vz = 0  # 单纯用光流无法直接估计Z方向速度，需多帧或其它方法
    return np.array([vx, vy, vz])