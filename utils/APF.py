import numpy as np
def compute_apf(xs, ys, Zs, K, drone_pos, repulse_gain=1.0, min_dist=0.5):
    """
    xs, ys: 障碍物像素点
    Zs: 深度估计
    K: 相机内参
    drone_pos: 当前无人机位置 (3,)
    返回: 避障合力方向 (3,)
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    force = np.zeros(3)
    for x, y, Z in zip(xs, ys, Zs):
        # 像素转相机坐标
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy
        obstacle_pos = np.array([X, Y, Z])
        vec = drone_pos - obstacle_pos
        dist = np.linalg.norm(vec)
        if dist < min_dist:
            dist = min_dist
        repulse = repulse_gain * vec / (dist ** 3)
        force += repulse
    return force