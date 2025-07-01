import numpy as np
import cv2

def estimate_depth_single_view(xs, ys, flow, K, V, w):
    """
    xs, ys: 物体像素点坐标
    flow: [2, H, W] 光流
    K: 相机内参 (3,3)
    V: 线速度 (3,)
    w: 角速度 (3,)
    返回: 每个点的深度估计 Z
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    Zs = []
    for x, y in zip(xs, ys):
        u, v = flow[0, y, x], flow[1, y, x]
        # 像素坐标转归一化相机坐标
        xn = (x - cx) / fx
        yn = (y - cy) / fy
        # 运动视差模型
        # 这里只考虑平移分量，忽略旋转分量（可根据实际情况补充）
        # A(x, y) = [[-fx, 0, x-cx], [0, -fy, y-cy]]
        A = np.array([
            [-fx, 0, x - cx],
            [0, -fy, y - cy]
        ])
        # 光流幅值
        f_vec = np.array([u, v])
        # 只考虑平移分量
        numerator = np.linalg.norm(A @ V)
        denominator = np.linalg.norm(f_vec)
        Z = numerator / (denominator + 1e-6)
        Zs.append(Z)
    return np.array(Zs)