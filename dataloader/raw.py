import os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import cv2
from .utils import ProgressBar  # 确保你有这个类
import hdf5plugin
from .encodings import events_to_voxel, events_to_channels, get_hot_event_mask,undistort_events, resize_events, resize_image

def simfind_data_triplets(data_dir):
    data_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.h5') or f.endswith('.hdf5')
    ])
    triplets = []
    for data_f in data_files:
        triplets.append(os.path.join(data_dir, data_f))
    return triplets

class simHDF5Dataset(Dataset):
    """
    支持事件、深度、光流的Dataset,事件数据支持cnt/voxel编码,按深度时间戳切片。
    """
    def __init__(self, data_h5, voxel_bins=5,  resolution=(960, 1240), hot_filter=None, config=None,t=0.02):       
        self.data_h5 = data_h5
        self.last_proc_timestamp = 0 
        self.voxel_bins = voxel_bins
        self.resolution = resolution
        self.hot_filter = hot_filter or dict(enabled=False)
        self.maps_loaded = False  # 延迟加载
        self.t = t
        self.config = config
        self._init_files()
        self._init_hot_filter()

        # 进度条
        if self.config and self.config.get("vis", {}).get("bars", False):
            self.open_files_bar = []
            # 这里假设每个dataset只对应一个文件
            max_iters = len(self)  # 或你自定义的 get_iters()
            filename = os.path.basename(self.data_h5)
            self.open_files_bar.append(ProgressBar(filename, max=max_iters))


    def _init_files(self):
        self.data_f = h5py.File(self.data_h5, 'r')
        # 事件
        if 'CD' in self.data_f:
            self.events_left = self.data_f['CD']['events'][:]  # N,4  # N,4
            # 如果是结构化数组，转为普通二维float64数组
            #还需要保证是x,y,p,t而不是x,y,t,p
            if self.events_left.dtype.fields is not None:
                self.events_left = np.stack([
                    self.events_left['x'].astype(np.float32),
                    self.events_left['y'].astype(np.float32),
                    self.events_left['p'].astype(np.float32),
                    (self.events_left['t']/1000000.0).astype(np.float64)  # 转为秒
                ], axis=-1)
            if self.events_left.ndim == 1 and self.events_left.shape[0] % 4 == 0:
                self.events_left = self.events_left.reshape(-1, 4)
        elif 'events' in self.data_f and isinstance(self.data_f['events'], h5py.Group):
            group = self.data_f['events']
            N = group['x'].shape[0]  # 总事件数
            half = N // 2
            if all(k in group for k in ['x', 'y', 't', 'p']):
                x = group['x'][:half]
                y = group['y'][:half]
                t = group['t'][:half]
                p = group['p'][:half]
                self.events_left = np.stack([
                    x.astype(np.float32),
                    y.astype(np.float32),
                    p.astype(np.float32),
                    (t.astype(np.float64) / 1_000_000.0)
            ], axis=-1)
        elif 'events' in self.data_f and isinstance(self.data_f['events'], h5py.Dataset):
            self.events_left = self.data_f['events'][:]  # N,4  # N,4
            # 如果是结构化数组，转为普通二维float64数组
            #还需要保证是x,y,p,t而不是x,y,t,p
            if self.events_left.dtype.fields is not None:
                self.events_left = np.stack([
                    self.events_left['x'].astype(np.float32),
                    self.events_left['y'].astype(np.float32),
                    self.events_left['p'].astype(np.float32),
                    (self.events_left['t']/1000000.0).astype(np.float64)  # 转为秒
                ], axis=-1)
            else:
                self.events_left = np.stack([
                    self.events_left[:,0].astype(np.float32),
                    self.events_left[:,1].astype(np.float32),
                    self.events_left[:,3].astype(np.float32),
                    (self.events_left[:,2]/1000000.0).astype(np.float64)  # 转为秒
                ], axis=-1)
            if self.events_left.ndim == 1 and self.events_left.shape[0] % 4 == 0:
                self.events_left = self.events_left.reshape(-1, 4)
        t_start = self.events_left[0, 3]
        t_end = self.events_left[-1, 3]
        self.ts_left = np.arange(t_start, t_end, self.t, dtype=np.float64)

        # 记录每个slice的事件范围
        self.slice_indices_left = []
        start_idx = 0
        for t in self.ts_left:
            end_idx = np.searchsorted(self.events_left[:, 3], t, side='right')
            self.slice_indices_left.append((start_idx, end_idx))
            start_idx = end_idx


    def _init_hot_filter(self):
            self.hot_events_left = torch.zeros(self.resolution)
            self.hot_idx_left = 0

    def __len__(self):
        return len(self.ts_left)

    def _process_events(self, events, hot_events, hot_idx):
        if len(events) == 0:
            xs = torch.zeros(1)
            ys = torch.zeros(1)
            ts = torch.zeros(1)
            ps = torch.zeros(1)
        else:
            xs = torch.from_numpy(events[:, 0].astype(np.float32))
            ys = torch.from_numpy(events[:, 1].astype(np.float32))
            ts_arr = events[:, 2].astype(np.float64)
            ps = torch.from_numpy(events[:, 3].astype(np.float32)) * 2 - 1
            ts = torch.from_numpy((ts_arr - ts_arr[0]) / (ts_arr[-1] - ts_arr[0]) if ts_arr[-1] > ts_arr[0] else np.zeros_like(ts_arr))
            if ts.shape[0] > 0:
                self.last_proc_timestamp = ts[-1]
        # 编码
        event_cnt = events_to_channels(xs, ys, ps, sensor_size=self.resolution)
        event_voxel = events_to_voxel(xs, ys, ts, ps, self.voxel_bins, sensor_size=self.resolution)
        event_list = torch.stack([ts, ys, xs, ps], axis=1)
        event_list_pol_mask = np.zeros((len(events), 2), dtype=np.float32)
        event_list_pol_mask[:, 0] = (ps > 0).float()
        event_list_pol_mask[:, 1] = (ps < 0).float()
        event_mask = (torch.sum(event_cnt, dim=0) > 0).float()

        # 热像素mask
        hot_mask = torch.ones(self.resolution)
        if self.hot_filter.get('enabled', False) and event_cnt is not None:
            hot_update = torch.sum(event_cnt, dim=0)
            hot_update[hot_update > 0] = 1
            hot_events += hot_update
            hot_idx += 1
            event_rate = hot_events / hot_idx
            hot_mask = get_hot_event_mask(
                event_rate,
                hot_idx,
                max_px=self.hot_filter.get('max_px', 100),
                min_obvs=self.hot_filter.get('min_obvs', 5),
                max_rate=self.hot_filter.get('max_rate', 0.8),
            )
        if hot_mask is None:
            hot_mask = torch.empty(0)
        return event_cnt, event_voxel,event_list,event_list_pol_mask, event_mask, hot_mask, hot_events, hot_idx

    def __getitem__(self, idx):
        result = {}
        # left
        start, end = self.slice_indices_left[idx]
        events = self.events_left[start:end]
        
        # 事件resize
        if len(events) > 0:
            xs, ys = events[:, 0], events[:, 1]
            ORIGINAL_EVENT_SHAPE = (720, 1280)  # 事件相机的分辨率,MVSEC(260,346),propheshe/方竹(720,1280),DESC(480,640)
            xs, ys = resize_events(xs, ys, ORIGINAL_EVENT_SHAPE, self.resolution)
            events[:, 0] = xs
            events[:, 1] = ys
        # 事件编码
        event_cnt, event_voxel, event_list, event_list_pol_mask, event_mask, hot_mask, self.hot_events_right, self.hot_idx_right = self._process_events(
            events, self.hot_events_left, self.hot_idx_left
        )
        #mask升维
        mask = (event_mask * hot_mask).unsqueeze(0)  # [1, H, W]

        # 计算事件窗口dt
        dt = float(events[-1, 2] - events[0, 2])
        result = {
            'filename': self.data_h5,#保证之后可以通过filename确定batch来自哪个文件
            'event_cnt': event_cnt,
            'event_voxel': event_voxel,
            'event_list': event_list,
            'event_list_pol_mask': event_list_pol_mask,
            'dt': dt,
            'dt_gt': dt,  # 这里假设dt_gt和dt相同
            'mask': mask,
        }

        return result


