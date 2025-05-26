import os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

from .encodings import events_to_voxel, events_to_channels, get_hot_event_mask,undistort_events, resize_events, resize_image

def load_map_txt(path):
    return np.loadtxt(path, dtype=np.float32)

def find_data_triplets(data_dir):
    data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_data.hdf5')])
    gt_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_gt.hdf5')])
    flow_files = sorted([f for f in os.listdir(data_dir) if f.endswith('flow_dist.npz')])
    # 简单配对，假设文件名排序后顺序一致
    assert len(data_files) == len(gt_files) == len(flow_files), "数据文件数量不一致"
    triplets = []
    for data_f, gt_f, flow_f in zip(data_files, gt_files, flow_files):
        triplets.append((
            os.path.join(data_dir, data_f),
            os.path.join(data_dir, gt_f),
            os.path.join(data_dir, flow_f)
        ))
    return triplets

class HDF5Dataset(Dataset):
    """
    支持事件、深度、光流的Dataset,事件数据支持cnt/voxel编码,按深度时间戳切片。
    """
    def __init__(self, data_h5, gt_h5, flow_npz, voxel_bins=5, encoding='cnt', resolution=(260, 346), hot_filter=None, eye='left', undistort=False, map_dir=None):       
        self.data_h5 = data_h5
        self.gt_h5 = gt_h5
        self.flow_npz = flow_npz
        self.voxel_bins = voxel_bins
        self.encoding = encoding
        self.resolution = resolution
        self.hot_filter = hot_filter or dict(enabled=False)
        self.eye = eye
        self.undistort = undistort
        self.map_dir = map_dir
        self.maps_loaded = False  # 延迟加载
        self._init_files()
        self._init_hot_filter()

    def _load_maps(self):
        # 只在需要时加载
        if self.maps_loaded or not self.undistort:
            return
        def map_path(eye, axis):
            map_prefix="indoor_flying"
            return os.path.join(self.map_dir, f"{map_prefix}_{eye}_{axis}_map.txt")
        self.map_x_left = load_map_txt(map_path("left", "x"))
        self.map_y_left = load_map_txt(map_path("left", "y"))
        self.map_x_right = load_map_txt(map_path("right", "x"))
        self.map_y_right = load_map_txt(map_path("right", "y"))
        self.maps_loaded = True

    def _init_files(self):
        self.data_f = h5py.File(self.data_h5, 'r')
        self.gt_f = h5py.File(self.gt_h5, 'r')
        # 事件
        self.events_left = self.data_f['davis/left/events'][:]  # N,4
        self.events_right = self.data_f['davis/right/events'][:]  # N,4
        # 深度
        self.depth_left = self.gt_f['davis/left/depth_image_rect'][:]  # T, H, W
        self.depth_right = self.gt_f['davis/right/depth_image_rect'][:]  # T, H, W
        #光流
        flow_data = np.load(self.flow_npz)
        x_flow = flow_data['x_flow_dist']
        y_flow = flow_data['y_flow_dist']
        self.flow = np.stack([x_flow, y_flow], axis=1)
        # 时间戳
        self.ts_left = self.gt_f['davis/left/depth_image_rect_ts'][:]  # T, 间隔约为0.05s
        self.ts_right = self.gt_f['davis/right/depth_image_rect_ts'][:]  # T,与left相同
        self.ts_flow = flow_data['timestamps']  # T,与depth差距约为0.001ms，可以忽略不计

        # 记录每个slice的事件范围
        self.slice_indices_left = []
        start_idx = 0
        for t in self.ts_left:
            end_idx = np.searchsorted(self.events_left[:, 2], t, side='right')
            self.slice_indices_left.append((start_idx, end_idx))
            start_idx = end_idx

        self.slice_indices_right = []
        start_idx = 0
        for t in self.ts_right:
            end_idx = np.searchsorted(self.events_right[:, 2], t, side='right')
            self.slice_indices_right.append((start_idx, end_idx))
            start_idx = end_idx

    def _init_hot_filter(self):
            self.hot_events_left = torch.zeros(self.resolution)
            self.hot_events_right = torch.zeros(self.resolution)
            self.hot_idx_left = 0
            self.hot_idx_right = 0

    def __len__(self):
        # 以left为主，假设left/right长度一致
        return len(self.ts_left)

    def _process_events(self, events, encoding, hot_events, hot_idx):
        if len(events) == 0:
            xs = torch.zeros(1)
            ys = torch.zeros(1)
            ts = torch.zeros(1)
            ps = torch.zeros(1)
        else:
            xs = torch.from_numpy(events[:, 0].astype(np.float32))
            ys = torch.from_numpy(events[:, 1].astype(np.float32))
            ts_arr = events[:, 2].astype(np.float32)
            ps = torch.from_numpy(events[:, 3].astype(np.float32)) * 2 - 1
            ts = torch.from_numpy((ts_arr - ts_arr[0]) / (ts_arr[-1] - ts_arr[0]) if ts_arr[-1] > ts_arr[0] else np.zeros_like(ts_arr))
        # 编码
            event_cnt = events_to_channels(xs, ys, ps, sensor_size=self.resolution)
            event_voxel = events_to_voxel(xs, ys, ts, ps, self.voxel_bins, sensor_size=self.resolution)
        # 热像素mask
        mask = None
        if self.hot_filter.get('enabled', False) and event_cnt is not None:
            hot_update = torch.sum(event_cnt, dim=0)
            hot_update[hot_update > 0] = 1
            hot_events += hot_update
            hot_idx += 1
            event_rate = hot_events / hot_idx
            mask = get_hot_event_mask(
                event_rate,
                hot_idx,
                max_px=self.hot_filter.get('max_px', 100),
                min_obvs=self.hot_filter.get('min_obvs', 5),
                max_rate=self.hot_filter.get('max_rate', 0.8),
            )
        if mask is None:
            mask = torch.empty(0)
        return event_cnt, event_voxel, mask, hot_events, hot_idx

    def __getitem__(self, idx):
        result = {}
        if self.undistort:
            self._load_maps()
        # left
        if self.eye in ['left', 'both']:
            start, end = self.slice_indices_left[idx]
            events = self.events_left[start:end]
            # 事件校正
            if self.undistort and len(events) > 0:
                xs, ys = events[:, 0], events[:, 1]
                xs, ys = undistort_events(xs, ys, self.map_x_left, self.map_y_left)
                events[:, 1] = ys
            # 事件resize
            if len(events) > 0:
                src_shape = self.map_x_left.shape if self.undistort else (self.depth_left.shape[1], self.depth_left.shape[2])
                xs, ys = events[:, 0], events[:, 1]
                xs, ys = resize_events(xs, ys, src_shape, self.resolution)
                events[:, 0] = xs
                events[:, 1] = ys
            # 事件编码
            event_cnt, event_voxel, mask, self.hot_events_left, self.hot_idx_left = self._process_events(
                events, self.encoding,self.hot_events_left, self.hot_idx_left
            )
            # 深度resize
            depth = self.depth_left[idx]
            depth = resize_image(depth, self.resolution)
            # 光流resize
            flow = self.flow[idx] if self.flow is not None else torch.empty(0)
            if flow is not None:
                flow = resize_image(flow, self.resolution)
            result['left'] = {
                'event_cnt': event_cnt,
                'event_voxel': event_voxel,
                'mask': mask,
                'depth': torch.from_numpy(depth).float(),
                'flow': torch.from_numpy(flow).float()
            }
        # right
        if self.eye in ['right', 'both']:
            start, end = self.slice_indices_right[idx]
            events = self.events_right[start:end]
            # 事件校正
            if self.undistort and len(events) > 0:
                xs, ys = events[:, 0], events[:, 1]
                xs, ys = undistort_events(xs, ys, self.map_x_right, self.map_y_right)
                events[:, 0] = xs
                events[:, 1] = ys
            # 事件resize
            if len(events) > 0:
                src_shape = self.map_x_right.shape if self.undistort else (self.depth_right.shape[1], self.depth_right.shape[2])
                xs, ys = events[:, 0], events[:, 1]
                xs, ys = resize_events(xs, ys, src_shape, self.resolution)
                events[:, 0] = xs
                events[:, 1] = ys
            # 事件编码
            event_cnt, event_voxel, mask, self.hot_events_right, self.hot_idx_right = self._process_events(
                events, self.encoding, self.hot_events_right, self.hot_idx_right
            )
            # 深度resize
            depth = self.depth_right[idx]
            depth = resize_image(depth, self.resolution)
            # 光流resize，MVSEC右相机无光流数据
            flow = torch.empty(0)  # 用空tensor代替None
            result['right'] = {
                'event_cnt': event_cnt,
                'event_voxel': event_voxel,
                'mask': mask,
                'depth': torch.from_numpy(depth).float(),
                'flow': flow
            }
        # 只返回left或right时，直接返回dict['left']或dict['right']
        if self.eye == 'left':
            return result['left']
        elif self.eye == 'right':
            return result['right']
        else:
            return result

