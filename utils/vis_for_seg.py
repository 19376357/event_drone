import cv2
import numpy as np
import matplotlib.pyplot as plt

class VisForSeg:
    def __init__(self, px=346,py=260):
        self.px = px
        self.py = py

    def show_events(self, events, title="Events", shape=(256, 256)):
        """
        events: [N, 4] (t, y, x, p)
        color: BGR tuple
        shape: (H, W)
        """
        img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        if events is not None and len(events) > 0:
            xs = np.clip(events[:, 2].astype(np.int32), 0, shape[1] - 1)
            ys = np.clip(events[:, 1].astype(np.int32), 0, shape[0] - 1)
            ps = events[:, 3]
            pos_idx = ps > 0
            neg_idx = ps < 0
            img[ys[pos_idx], xs[pos_idx]] = (0, 0, 255)   # 红色（正极性）
            img[ys[neg_idx], xs[neg_idx]] = (255, 0, 0)   # 蓝色（负极性）
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, self.px, self.py)
        cv2.imshow(title, img)

    def show_pixel_mask(self, ys, xs, title="Pixel Mask", color=(0, 0, 255), shape=(256, 256)):
        img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        if len(xs) > 0:
            img[ys, xs] = color
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, self.px, self.py)
        cv2.imshow(title, img)

    def show_flow(self, flow, title="Flow"):
        """
        flow: [2, H, W] or [1, 2, H, W]
        """
        if flow.ndim == 4:
            flow = flow[0]
        flow_x = flow[0]
        flow_y = flow[1]
        flow_img = self.flow_to_image(flow_x, flow_y)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, self.px, self.px)
        cv2.imshow(title, flow_img)

    @staticmethod
    def flow_to_image(flow_x, flow_y):
        flows = np.stack((flow_x, flow_y), axis=2)
        mag = np.linalg.norm(flows, axis=2)
        min_mag = np.min(mag)
        mag_range = np.max(mag) - min_mag

        ang = np.arctan2(flow_y, flow_x) + np.pi
        ang *= 1.0 / np.pi / 2.0

        hsv = np.zeros([flow_x.shape[0], flow_x.shape[1], 3])
        hsv[:, :, 0] = ang
        hsv[:, :, 1] = 1.0
        hsv[:, :, 2] = mag - min_mag
        if mag_range != 0.0:
            hsv[:, :, 2] /= mag_range

        flow_rgb = plt.cm.hsv(hsv[:, :, 0])[:, :, :3] * hsv[:, :, 2][..., None]
        flow_rgb = (255 * flow_rgb).astype(np.uint8)
        return flow_rgb

    def visualize_all(self, events, flow, fg_events=None, obj_events=None,fg_pixels=None,obj_pixels=None, shape=(256, 256)):
        """
        events: [N, 4] 原始事件
        flow: [2, H, W] 光流
        fg_events: [M, 4] 分割出的前景事件
        obj_events: [K, 4] 主要运动物体事件
        """
        self.show_events(events, "All Events", shape=shape)
        self.show_flow(flow, "Estimated Flow")
        if fg_events is not None and len(fg_events) > 0:
            self.show_events(fg_events, "Foreground Events", shape=shape)
        if obj_events is not None and len(obj_events) > 0:
            self.show_events(obj_events, "Main Object Events",  shape=shape)
        # 像素mask可视化
        if fg_pixels is not None and len(fg_pixels[0]) > 0:
            self.show_pixel_mask(fg_pixels[0], fg_pixels[1], title="Foreground Pixels", color=(0, 0, 255), shape=shape)
        if obj_pixels is not None and len(obj_pixels[0]) > 0:
            self.show_pixel_mask(obj_pixels[0], obj_pixels[1], title="Main Object Pixels", color=(255, 0, 0), shape=shape)
        cv2.waitKey(1)