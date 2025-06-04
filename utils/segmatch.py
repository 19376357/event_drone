def segment_events_by_flow(events, flow, mask, threshold=2.0):
    # events: [N, 4] (t, y, x, p)
    # flow: [2, H, W]
    # mask: [H, W]
    xs = events[:, 2].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    valid = (xs >= 0) & (xs < flow.shape[2]) & (ys >= 0) & (ys < flow.shape[1])
    xs, ys = xs[valid], ys[valid]
    event_idx = np.where(valid)[0]
    # 提取每个事件点的光流
    event_flow = flow[:, ys, xs].T  # [N_valid, 2]
    # 估计背景光流（如全局均值）
    bg_flow = np.mean(event_flow, axis=0)
    # 计算每个事件点与背景光流的欧氏距离
    flow_diff = np.linalg.norm(event_flow - bg_flow, axis=1)
    # 分割
    fg_idx = event_idx[flow_diff > threshold]  # 前景（自运动物体）
    bg_idx = event_idx[flow_diff <= threshold] # 背景（相机自运动）
    return fg_idx, bg_idx
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