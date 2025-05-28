
import numpy as np
import torch
import cv2

def events_to_image(xs, ys, ps, sensor_size=(260, 346), accumulate=True):
    device = xs.device
    img_size = list(sensor_size)
    img = torch.zeros(img_size, device=device)
    if xs.dtype is not torch.long:
        xs = xs.long()
    if ys.dtype is not torch.long:
        ys = ys.long()
    img.index_put_((ys, xs), ps, accumulate=accumulate)
    return img

def events_to_voxel(xs, ys, ts, ps, num_bins, sensor_size=(260, 346), round_ts=False):
    assert len(xs) == len(ys) == len(ts) == len(ps)
    voxel = []
    ts = ts * (num_bins - 1)
    if round_ts:
        ts = torch.round(ts)
    zeros = torch.zeros_like(ts)
    for b_idx in range(num_bins):
        weights = torch.max(zeros, 1.0 - torch.abs(ts - b_idx))
        voxel_bin = events_to_image(xs, ys, ps * weights, sensor_size=sensor_size)
        voxel.append(voxel_bin)
    return torch.stack(voxel)

def events_to_channels(xs, ys, ps, sensor_size=(180, 240)):
    assert len(xs) == len(ys) == len(ps)
    mask_pos = ps.clone()
    mask_neg = ps.clone()
    mask_pos[ps < 0] = 0
    mask_neg[ps > 0] = 0
    pos_cnt = events_to_image(xs, ys, ps * mask_pos, sensor_size=sensor_size)
    neg_cnt = events_to_image(xs, ys, ps * mask_neg, sensor_size=sensor_size)
    return torch.stack([pos_cnt, neg_cnt])

def get_hot_event_mask(event_rate, idx, max_px=100, min_obvs=5, max_rate=0.8):
    mask = torch.ones(event_rate.shape, device=event_rate.device)
    if idx > min_obvs:
        for _ in range(max_px):
            argmax = torch.argmax(event_rate)
            index = (argmax // event_rate.shape[1], argmax % event_rate.shape[1])
            if event_rate[index] > max_rate:
                event_rate[index] = 0
                mask[index] = 0
            else:
                break
    return mask
def undistort_events(xs, ys, map_x, map_y):
    # xs, ys: 1D numpy array, map_x/map_y: HxW
    xs_int = np.clip(xs.astype(np.int32), 0, map_x.shape[1]-1)
    ys_int = np.clip(ys.astype(np.int32), 0, map_x.shape[0]-1)
    xs_new = map_x[ys_int, xs_int]
    ys_new = map_y[ys_int, xs_int]
    return xs_new, ys_new
def resize_events(xs, ys, src_shape, dst_shape):
    scale_x = dst_shape[1] / src_shape[1]
    scale_y = dst_shape[0] / src_shape[0]
    xs_new = xs * scale_x
    ys_new = ys * scale_y
    # --- 加clip，防止越界 ---
    xs_new = np.clip(xs_new, 0, dst_shape[1] - 1)
    ys_new = np.clip(ys_new, 0, dst_shape[0] - 1)
    return xs_new, ys_new
def resize_image(img, dst_shape):
    # img: (H, W) or (C, H, W) or (H, W, C) or (N, H, W, C)
    if img.ndim == 2:
        return cv2.resize(img, (dst_shape[1], dst_shape[0]), interpolation=cv2.INTER_LINEAR)
    elif img.ndim == 3:
        if img.shape[0] == 2 and img.shape[1] != 3:  # (2, H, W) for flow
            return np.stack([cv2.resize(img[0], (dst_shape[1], dst_shape[0]), interpolation=cv2.INTER_LINEAR),
                             cv2.resize(img[1], (dst_shape[1], dst_shape[0]), interpolation=cv2.INTER_LINEAR)], axis=0)
        elif img.shape[2] == 3:  # (H, W, 3)
            return cv2.resize(img, (dst_shape[1], dst_shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError("Unsupported image shape for resize")
    elif img.ndim == 4 and img.shape[-1] == 3:  # (N, H, W, 3)
        return np.stack([cv2.resize(im, (dst_shape[1], dst_shape[0]), interpolation=cv2.INTER_LINEAR) for im in img], axis=0)
    else:
        raise ValueError("Unsupported image shape for resize")
