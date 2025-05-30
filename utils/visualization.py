import os
import torch
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Visualization:
    """
    Utility class for the visualization and storage of rendered image-like representation
    of multiple elements of the optical flow estimation and image reconstruction pipeline.
    """

    def __init__(self, kwargs, eval_id=-1, path_results=None):
        self.img_idx = 0
        self.vis_delay = kwargs["vis"].get("delay", 1)
        self.px = kwargs["vis"]["px"]
        self.color_scheme = "green_red"  # gray / blue_red / green_red

        if eval_id >= 0 and path_results is not None:
            self.store_dir = path_results + "results/"
            self.store_dir = self.store_dir + "eval_" + str(eval_id) + "/"
            if not os.path.exists(self.store_dir):
                os.makedirs(self.store_dir)
            self.store_file = None

    def update(self, inputs, flow, iwe, events_window=None, masked_window_flow=None, iwe_window=None):
        """
        Live visualization.
        :param inputs: dataloader dictionary
        :param flow: [batch_size x 2 x H x W] optical flow map
        :param iwe: [batch_size x 1 x H x W] image of warped events
        """

        events = inputs["event_cnt"] if "event_cnt" in inputs.keys() else None
        frames = inputs["image"] if "image" in inputs.keys() else None
        gtflow = inputs["flow"] if "flow" in inputs.keys() else None
        gtflow_mask = gtflow * inputs["mask"].to(gtflow.device) if "flow" in inputs.keys() else None

        height = events.shape[2]
        width = events.shape[3]

        # input events
        events = events.detach()
        events_npy = events.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, -1))
        cv2.namedWindow("Input Events", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Input Events", int(self.px), int(self.px))
        cv2.imshow("Input Events", self.events_to_image(events_npy))

        '''
        # input events
        if events_window is not None:
            events_window = events_window.detach()
            events_window_npy = events_window.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, -1))
            cv2.namedWindow("Input Events - Eval window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Input Events - Eval window", int(self.px), int(self.px))
            cv2.imshow("Input Events - Eval window", self.events_to_image(events_window_npy))
        '''

        # input frames
        if frames is not None:
            frame_image = np.zeros((height, 2 * width))
            frames = frames.detach()
            frames_npy = frames.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            frame_image[:height, 0:width] = frames_npy[:, :, 0] / 255.0
            frame_image[:height, width : 2 * width] = frames_npy[:, :, 1] / 255.0
            cv2.namedWindow("Input Frames (Prev/Curr)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Input Frames (Prev/Curr)", int(2 * self.px), int(self.px))
            cv2.imshow("Input Frames (Prev/Curr)", frame_image)
        # Farneback 光流可视化
        if frames is not None:
            flow_fb_tensor = self.get_farneback_flow_img(frames, height, width)
            if "mask" in inputs.keys():
                flow_fb_tensor = flow_fb_tensor.to(inputs["mask"].device, dtype=inputs["mask"].dtype)
                flow_fb_tensor = flow_fb_tensor * inputs["mask"]
            flow_fb = flow_fb_tensor.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            flow_fb_img = self.flow_to_image(flow_fb[..., 0], flow_fb[..., 1])
            flow_fb_img = cv2.cvtColor(flow_fb_img, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Farneback Flow", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Farneback Flow", int(self.px), int(self.px))
            cv2.imshow("Farneback Flow", flow_fb_img)
        # optical flow
        if flow is not None:
            flow = flow.detach()
            flow_npy = flow.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            flow_npy = self.flow_to_image(flow_npy[:, :, 0], flow_npy[:, :, 1])
            flow_npy = cv2.cvtColor(flow_npy, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Estimated Flow", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Estimated Flow", int(self.px), int(self.px))
            cv2.imshow("Estimated Flow", flow_npy)
        
        '''
        # optical flow
        if masked_window_flow is not None:
            masked_window_flow = masked_window_flow.detach()
            masked_window_flow_npy = masked_window_flow.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            masked_window_flow_npy = self.flow_to_image(
                masked_window_flow_npy[:, :, 0], masked_window_flow_npy[:, :, 1]
            )
            masked_window_flow_npy = cv2.cvtColor(masked_window_flow_npy, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Estimated Flow - Eval window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Estimated Flow - Eval window", int(self.px), int(self.px))
            cv2.imshow("Estimated Flow - Eval window", masked_window_flow_npy)
        '''

        # ground-truth optical flow
        if gtflow is not None:
            gtflow = gtflow.detach()
            gtflow_npy = gtflow.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            gtflow_npy = self.flow_to_image(gtflow_npy[:, :, 0], gtflow_npy[:, :, 1])
            gtflow_npy = cv2.cvtColor(gtflow_npy, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Ground-truth Flow", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Ground-truth Flow", int(self.px), int(self.px))
            cv2.imshow("Ground-truth Flow", gtflow_npy)
        if gtflow_mask is not None:
            gtflow_mask = gtflow_mask.detach()
            gtflow_mask_npy = gtflow_mask.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            gtflow_mask_npy = self.flow_to_image(gtflow_mask_npy[:, :, 0], gtflow_mask_npy[:, :, 1])
            gtflow_mask_npy = cv2.cvtColor(gtflow_mask_npy, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Ground-truth Flow_mask", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Ground-truth Flow_mask", int(self.px), int(self.px))
            cv2.imshow("Ground-truth Flow_mask", gtflow_mask_npy)

        # 预测光流箭头
        if flow is not None and frames is not None:
            # 取当前帧灰度图
            frame_img = frames[0, 0].cpu().numpy() / 255.0   
            flow_vis = flow.detach().cpu().numpy()[0].transpose(1, 2, 0)  # [H, W, 2]
            img_arrow = self.get_arrow_img(frame_img, flow_vis, step=16, norm=True)
            cv2.namedWindow("Predicted Flow Arrows", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Predicted Flow Arrows", int(self.px), int(self.px))
            cv2.imshow("Predicted Flow Arrows", img_arrow) 
        
        # 真值光流箭头_默认带mask
        if gtflow_mask is not None and frames is not None:
            frame_img = frames[0, 0].cpu().numpy() / 255.0  # [H, W]
            gtflow_vis = gtflow_mask.detach().cpu().numpy()[0].transpose(1, 2, 0)  # [H, W, 2]
            img_gt_arrow = self.get_arrow_img(frame_img, gtflow_vis, step=16, norm=True)
            cv2.namedWindow("GT Flow Arrows", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("GT Flow Arrows", int(self.px), int(self.px))
            cv2.imshow("GT Flow Arrows", img_gt_arrow)

        

        # image of warped events
        if iwe is not None:
            iwe = iwe.detach()
            iwe_npy = iwe.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            iwe_npy = self.events_to_image(iwe_npy)
            cv2.namedWindow("Image of Warped Events", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image of Warped Events", int(self.px), int(self.px))
            cv2.imshow("Image of Warped Events", iwe_npy)
        '''
        # image of warped events - evaluation window
        if iwe_window is not None:
            iwe_window = iwe_window.detach()
            iwe_window_npy = iwe_window.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            iwe_window_npy = self.events_to_image(iwe_window_npy)
            cv2.namedWindow("Image of Warped Events - Eval window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image of Warped Events - Eval window", int(self.px), int(self.px))
            cv2.imshow("Image of Warped Events - Eval window", iwe_window_npy)
        '''

        cv2.waitKey(self.vis_delay)
    def update_stereo(
        self,
        events_left, events_right,
        frames_left, frames_right,
        flow_left, flow_right,
        iwe_left, iwe_right,
        gtflow_left, gtflow_right,
        inputs
    ):
        # 事件拼接
        def get_events_img(events):
            if events is None:
                return None
            events = events.detach()
            h, w = events.shape[2], events.shape[3]
            events_npy = events.cpu().numpy().transpose(0, 2, 3, 1).reshape((h, w, -1))
            return self.events_to_image(events_npy)

        img_events_left = get_events_img(events_left)
        img_events_right = get_events_img(events_right)
        if img_events_left is not None and img_events_right is not None:
            img_events = np.concatenate([img_events_left, img_events_right], axis=1)
            cv2.namedWindow("Input Events (Left | Right)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Input Events (Left | Right)", int(2 * self.px), int(self.px))
            cv2.imshow("Input Events (Left | Right)", img_events)

        # frames拼接（每个相机先拼前后帧，再横向拼接左右）
        def get_frames_img(frames):
            if frames is None:
                return None
            frames = frames.detach().cpu().numpy()
            if frames.ndim == 5 and frames.shape[0] == 1:  # [B, 2, H, W, C]
                frames_npy = frames[0]
            elif frames.ndim == 4 and frames.shape[0] == 1:
                frames_npy = frames[0]
            else:
                frames_npy = frames
            if frames_npy.shape[0] != 2:
                print(f"Warning: frames shape {frames_npy.shape}, expect first dim=2")
                return None
            # [2, H, W, C] or [2, H, W]
            if frames_npy.ndim == 4 and frames_npy.shape[-1] == 1:  # 灰度
                frame0 = np.squeeze(frames_npy[0], axis=-1)
                frame1 = np.squeeze(frames_npy[1], axis=-1)
                img = np.concatenate([frame0, frame1], axis=1)
            elif frames_npy.ndim == 4:  # 彩色
                frame0 = frames_npy[0]
                frame1 = frames_npy[1]
                img = np.concatenate([frame0, frame1], axis=1)
            elif frames_npy.ndim == 3:  # 灰度
                frame0 = frames_npy[0]
                frame1 = frames_npy[1]
                img = np.concatenate([frame0, frame1], axis=1)
            else:
                return None
            # 归一化
            if img.max() > 1.1:
                img = img / 255.0
            return img

        img_frames_left = get_frames_img(frames_left)
        img_frames_right = get_frames_img(frames_right)
        if img_frames_left is not None and img_frames_right is not None:
            img_frames = np.concatenate([img_frames_left, img_frames_right], axis=1)
            cv2.namedWindow("Input Frames (Prev/Curr) (Left | Right)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Input Frames (Prev/Curr) (Left | Right)", int(4 * self.px), int(self.px))
            cv2.imshow("Input Frames (Prev/Curr) (Left | Right)", img_frames)

        # 光流拼接
        def get_flow_img(flow):
            if flow is None:
                return None
            flow = flow.detach()
            h, w = flow.shape[2], flow.shape[3]
            flow_npy = flow.cpu().numpy().transpose(0, 2, 3, 1).reshape((h, w, 2))
            flow_npy = self.flow_to_image(flow_npy[:, :, 0], flow_npy[:, :, 1])
            flow_npy = cv2.cvtColor(flow_npy, cv2.COLOR_RGB2BGR)
            return flow_npy

        img_flow_left = get_flow_img(flow_left)
        img_flow_right = get_flow_img(flow_right)
        if img_flow_left is not None and img_flow_right is not None:
            img_flow = np.concatenate([img_flow_left, img_flow_right], axis=1)
            cv2.namedWindow("Estimated Flow (Left | Right)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Estimated Flow (Left | Right)", int(2 * self.px), int(self.px))
            cv2.imshow("Estimated Flow (Left | Right)", img_flow)
        # 光流箭头拼接
        def get_arrow_img(frames, flow):
            if frames is None or flow is None:
                return None
            # 取当前帧灰度图
            frame_img = frames[0, 0].cpu().numpy() / 255.0  # [H, W]
            flow_vis = flow.detach().cpu().numpy()[0].transpose(1, 2, 0)  # [H, W, 2]
            return self.get_arrow_img(frame_img, flow_vis, step=16, norm=True)
        img_arrow_left = get_arrow_img(frames_left, flow_left)
        img_arrow_right = get_arrow_img(frames_right, flow_right)
        if img_arrow_left is not None and img_arrow_right is not None:
            img_arrow = np.concatenate([img_arrow_left, img_arrow_right], axis=1)
            cv2.namedWindow("Estimated Flow Arrows (Left | Right)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Estimated Flow Arrows (Left | Right)", int(2 * self.px), int(self.px))
            cv2.imshow("Estimated Flow Arrows (Left | Right)", img_arrow)
        

        #光流真值箭头拼接:
        def get_gt_arrow_img(frames, gtflow):
            if frames is None or gtflow is None or gtflow.numel() == 0:
                return None
            frame_img = frames[0, 0].cpu().numpy() / 255.0
            gtflow_vis = gtflow.detach().cpu().numpy()[0].transpose(1, 2, 0)
            return self.get_arrow_img(frame_img, gtflow_vis, step=16, norm=True)
        
        gtflow_mask_left = gtflow_left * inputs["mask"].to(gtflow_left.device)
        img_gt_arrow_left = get_gt_arrow_img(frames_left, gtflow_mask_left)
        if img_gt_arrow_left is not None:
            cv2.namedWindow("GT Flow Arrows (Left)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("GT Flow Arrows (Left)", int(self.px), int(self.px))
            cv2.imshow("GT Flow Arrows (Left)", img_gt_arrow_left)


        # GT flow拼接
        def get_gtflow_img(gtflow):
            if gtflow is None or gtflow.numel() == 0:
                return None
            gtflow = gtflow.detach()
            h, w = gtflow.shape[2], gtflow.shape[3]
            gtflow_npy = gtflow.cpu().numpy().transpose(0, 2, 3, 1).reshape((h, w, 2))
            gtflow_npy = self.flow_to_image(gtflow_npy[:, :, 0], gtflow_npy[:, :, 1])
            gtflow_npy = cv2.cvtColor(gtflow_npy, cv2.COLOR_RGB2BGR)
            return gtflow_npy

        img_gtflow_left = get_gtflow_img(gtflow_left)
        gtflow_mask_left = gtflow_left * inputs["mask"].to(gtflow_left.device)
        img_gtflow_mask_left = get_gtflow_img(gtflow_mask_left)
        if img_gtflow_left is not None:
            cv2.namedWindow("Ground-truth Flow (Left)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Ground-truth Flow (Left)", int(self.px), int(self.px))
            cv2.imshow("Ground-truth Flow (Left)", img_gtflow_left)
        if img_gtflow_mask_left is not None:
            cv2.namedWindow("Ground-truth Flow_mask (Left)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Ground-truth Flow_mask (Left)", int(self.px), int(self.px))
            cv2.imshow("Ground-truth Flow_mask (Left)", img_gtflow_mask_left)

        # IWE拼接
        def get_iwe_img(iwe):
            if iwe is None:
                return None
            iwe = iwe.detach()
            h, w = iwe.shape[2], iwe.shape[3]
            iwe_npy = iwe.cpu().numpy().transpose(0, 2, 3, 1).reshape((h, w, 2))
            iwe_npy = self.events_to_image(iwe_npy)
            return iwe_npy

        img_iwe_left = get_iwe_img(iwe_left)
        img_iwe_right = get_iwe_img(iwe_right)
        if img_iwe_left is not None and img_iwe_right is not None:
            img_iwe = np.concatenate([img_iwe_left, img_iwe_right], axis=1)
            cv2.namedWindow("Image of Warped Events (Left | Right)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image of Warped Events (Left | Right)", int(2 * self.px), int(self.px))
            cv2.imshow("Image of Warped Events (Left | Right)", img_iwe)

        cv2.waitKey(self.vis_delay)


    def store(self, inputs, flow, iwe, sequence, events_window=None, masked_window_flow=None, iwe_window=None, ts=None):
        """
        Store rendered images.
        :param inputs: dataloader dictionary
        :param flow: [batch_size x 2 x H x W] optical flow map
        :param iwe: [batch_size x 1 x H x W] image of warped events
        :param sequence: filename of the event sequence under analysis
        :param ts: timestamp associated with rendered files (default = None)
        """

        events = inputs["event_cnt"] if "event_cnt" in inputs.keys() else None
        frames = inputs["frames"] if "frames" in inputs.keys() else None
        gtflow = inputs["gtflow"] if "gtflow" in inputs.keys() else None
        height = events.shape[2]
        width = events.shape[3]

        # check if new sequence
        path_to = self.store_dir + sequence + "/"
        if not os.path.exists(path_to):
            os.makedirs(path_to)
            os.makedirs(path_to + "events/")
            os.makedirs(path_to + "events_window/")
            os.makedirs(path_to + "flow/")
            os.makedirs(path_to + "flow_window/")
            os.makedirs(path_to + "gtflow/")
            os.makedirs(path_to + "frames/")
            os.makedirs(path_to + "iwe/")
            os.makedirs(path_to + "iwe_window/")
            if self.store_file is not None:
                self.store_file.close()
            self.store_file = open(path_to + "timestamps.txt", "w")
            self.img_idx = 0

        # input events
        event_image = np.zeros((height, width))
        events = events.detach()
        events_npy = events.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, -1))
        event_image = self.events_to_image(events_npy)
        filename = path_to + "events/%09d.png" % self.img_idx
        cv2.imwrite(filename, event_image * 255)

        # input events
        if events_window is not None:
            events_window = events_window.detach()
            events_window_npy = events_window.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, -1))
            events_window_npy = self.events_to_image(events_window_npy)
            filename = path_to + "events_window/%09d.png" % self.img_idx
            cv2.imwrite(filename, events_window_npy * 255)

        # input frames
        if frames is not None:
            frames = frames.detach()
            frames_npy = frames.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            filename = path_to + "frames/%09d.png" % self.img_idx
            cv2.imwrite(filename, frames_npy[:, :, 1])

        # optical flow
        if flow is not None:
            flow = flow.detach()
            flow_npy = flow.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            flow_npy = self.flow_to_image(flow_npy[:, :, 0], flow_npy[:, :, 1])
            flow_npy = cv2.cvtColor(flow_npy, cv2.COLOR_RGB2BGR)
            filename = path_to + "flow/%09d.png" % self.img_idx
            cv2.imwrite(filename, flow_npy)

        # optical flow
        if masked_window_flow is not None:
            masked_window_flow = masked_window_flow.detach()
            masked_window_flow_npy = masked_window_flow.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            masked_window_flow_npy = self.flow_to_image(
                masked_window_flow_npy[:, :, 0], masked_window_flow_npy[:, :, 1]
            )
            masked_window_flow_npy = cv2.cvtColor(masked_window_flow_npy, cv2.COLOR_RGB2BGR)
            filename = path_to + "flow_window/%09d.png" % self.img_idx
            cv2.imwrite(filename, masked_window_flow_npy)

        # ground-truth optical flow
        if gtflow is not None:
            gtflow = gtflow.detach()
            gtflow_npy = gtflow.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            gtflow_npy = self.flow_to_image(gtflow_npy[:, :, 0], gtflow_npy[:, :, 1])
            gtflow_npy = cv2.cvtColor(gtflow_npy, cv2.COLOR_RGB2BGR)
            filename = path_to + "gtflow/%09d.png" % self.img_idx
            cv2.imwrite(filename, gtflow_npy)

        # image of warped events
        if iwe is not None:
            iwe = iwe.detach()
            iwe_npy = iwe.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            iwe_npy = self.events_to_image(iwe_npy)
            filename = path_to + "iwe/%09d.png" % self.img_idx
            cv2.imwrite(filename, iwe_npy * 255)

        # image of warped events - evaluation window
        if iwe_window is not None:
            iwe_window = iwe_window.detach()
            iwe_window_npy = iwe_window.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            iwe_window_npy = self.events_to_image(iwe_window_npy)
            filename = path_to + "iwe_window/%09d.png" % self.img_idx
            cv2.imwrite(filename, iwe_window_npy * 255)

        # store timestamps
        if ts is not None:
            self.store_file.write(str(ts) + "\n")
            self.store_file.flush()

        self.img_idx += 1
        cv2.waitKey(self.vis_delay)

    @staticmethod
    def flow_to_image(flow_x, flow_y):
        """
        Use the optical flow color scheme from the supplementary materials of the paper 'Back to Event
        Basics: Self-Supervised Image Reconstruction for Event Cameras via Photometric Constancy',
        Paredes-Valles et al., CVPR'21.
        :param flow_x: [H x W x 1] horizontal optical flow component
        :param flow_y: [H x W x 1] vertical optical flow component
        :return flow_rgb: [H x W x 3] color-encoded optical flow
        """
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

        flow_rgb = matplotlib.colors.hsv_to_rgb(hsv)
        return (255 * flow_rgb).astype(np.uint8)

    @staticmethod
    def minmax_norm(x):
        """
        Robust min-max normalization.
        :param x: [H x W x 1]
        :return x: [H x W x 1] normalized x
        """
        den = np.percentile(x, 99) - np.percentile(x, 1)
        if den != 0:
            x = (x - np.percentile(x, 1)) / den
        return np.clip(x, 0, 1)

    @staticmethod
    def events_to_image(event_cnt, color_scheme="green_red"):
        """
        Visualize the input events.
        :param event_cnt: [batch_size x 2 x H x W] per-pixel and per-polarity event count
        :param color_scheme: green_red/gray
        :return event_image: [H x W x 3] color-coded event image
        """
        pos = event_cnt[:, :, 0]
        neg = event_cnt[:, :, 1]
        pos_max = np.percentile(pos, 99)
        pos_min = np.percentile(pos, 1)
        neg_max = np.percentile(neg, 99)
        neg_min = np.percentile(neg, 1)
        max = pos_max if pos_max > neg_max else neg_max

        if pos_min != max:
            pos = (pos - pos_min) / (max - pos_min)
        if neg_min != max:
            neg = (neg - neg_min) / (max - neg_min)

        pos = np.clip(pos, 0, 1)
        neg = np.clip(neg, 0, 1)

        event_image = np.ones((event_cnt.shape[0], event_cnt.shape[1]))
        if color_scheme == "gray":
            event_image *= 0.5
            pos *= 0.5
            neg *= -0.5
            event_image += pos + neg

        elif color_scheme == "green_red":
            event_image = np.repeat(event_image[:, :, np.newaxis], 3, axis=2)
            event_image *= 0
            mask_pos = pos > 0
            mask_neg = neg > 0
            mask_not_pos = pos == 0
            mask_not_neg = neg == 0

            event_image[:, :, 0][mask_pos] = 0
            event_image[:, :, 1][mask_pos] = pos[mask_pos]
            event_image[:, :, 2][mask_pos * mask_not_neg] = 0
            event_image[:, :, 2][mask_neg] = neg[mask_neg]
            event_image[:, :, 0][mask_neg] = 0
            event_image[:, :, 1][mask_neg * mask_not_pos] = 0

        return event_image
    @staticmethod
    def get_farneback_flow_img(frames,height,width):
        # frames: [1, H, W, 2]，第0通道为前帧，第1通道为当前帧
        frames_npy = frames.detach().cpu().numpy().transpose(0, 2, 3, 1).reshape((height,width, 2))
        prev_img = (frames_npy[:, :, 0] * 255).astype(np.uint8)
        curr_img = (frames_npy[:, :, 1] * 255).astype(np.uint8)
        # OpenCV 要求输入为 uint8
        flow_fb = cv2.calcOpticalFlowFarneback(
            prev_img, curr_img,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        flow_fb_tensor = torch.from_numpy(flow_fb).permute(2, 0, 1).unsqueeze(0)
        flow_fb_tensor = flow_fb_tensor.to(frames.device, frames.dtype)
        return flow_fb_tensor
    @staticmethod
    def get_arrow_img(im, flow, step=40, norm=True):
        # im: [H, W] 或 [H, W, 3]，灰度或BGR图像
        # flow: [H, W, 2]
        h, w = im.shape[:2]
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        if norm:
            mag = np.sqrt(fx**2 + fy**2)
            max_mag = np.max(mag)
            if max_mag > 1e-6:
                fx = fx / max_mag * step // 2
                fy = fy / max_mag * step // 2
        ex = x + fx
        ey = y + fy
        lines = np.vstack([x, y, ex, ey]).T.reshape(-1, 2, 2)
        lines = lines.astype(np.int32)
        if im.ndim == 2:
            vis = cv2.cvtColor((im * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            vis = (im * 255).astype(np.uint8).copy()
        for (x1, y1), (x2, y2) in lines:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        return vis


def vis_activity(activity, activity_log):
    # start of new sequence
    if activity_log is None:
        plt.close("activity")
        activity_log = []

    # update log
    activity_log.append(activity)
    df = pd.DataFrame(activity_log)

    # retrieves fig if it exists
    fig = plt.figure("activity")
    # make axis if it doesn't exist
    if not fig.axes:
        ax = fig.add_subplot()
    else:
        ax = fig.axes[0]
    lines = ax.lines

    # plot data
    if not lines:
        for name, data in df.items():
            ax.plot(data.index.to_numpy(), data.to_numpy(), label=name)
        ax.grid()
        ax.legend()
        ax.set_xlabel("step")
        ax.set_ylabel("fraction of nonzero outputs")
        plt.show(block=False)
    else:
        for line in lines:
            label = line.get_label()
            line.set_data(df[label].index.to_numpy(), df[label].to_numpy())

    # update figure
    fig.canvas.draw()
    ax.relim()
    ax.autoscale_view(True, True, True)
    fig.canvas.flush_events()

    return activity_log
