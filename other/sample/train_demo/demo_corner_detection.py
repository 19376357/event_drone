# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Corner Detection Demo Script
"""

"""
Corner Detection Demo Script
用于演示角点检测模型的实时推理。
"""

import numpy as np
import argparse
import torch
import csv

from metavision_core_ml.corner_detection.lightning_model import CornerDetectionLightningModel
from metavision_core_ml.preprocessing.event_to_tensor_torch import event_cd_to_torch, event_volume
from metavision_core_ml.utils.show_or_write import ShowWrite
from metavision_core_ml.corner_detection.corner_tracker import CornerTracker
from metavision_core_ml.corner_detection.utils import clean_pred, update_nn_tracker, save_nn_corners
from metavision_core.event_io import EventsIterator


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='', help='事件数据路径')
    parser.add_argument('checkpoint', type=str, default='', help='模型检查点路径')
    parser.add_argument('--video-path', type=str, default='', help='保存视频的路径')
    parser.add_argument('--show', action='store_true', help='显示检测结果')

    params, _ = parser.parse_known_args(raw_args)
    print('参数: ', params)

    # 加载事件数据
    events_iterator = EventsIterator(params.path)

    # 初始化显示或保存工具
    show_write = ShowWrite("Corner Detection", params.video_path)

    # 加载模型
    device = 'cpu' if params.cpu else 'cuda'
    model = CornerDetectionLightningModel.load_from_checkpoint(params.checkpoint)
    model.eval().to(device)

    # 初始化角点跟踪器
    tracker = CornerTracker(time_tolerance=7000)

    # 遍历事件流
    for events in events_iterator:
        if events is None or len(events) < 2:
            continue

        # 转换事件为张量
        events_th = event_cd_to_torch(events).to(device)
        tensor_th = event_volume(events_th, 1, *events_iterator.get_size(), None, None, 10, 'bilinear')

        # 模型预测
        pred = model.model(tensor_th)
        pred = torch.sigmoid(pred)
        pred = clean_pred(pred, threshold=0.1)

        # 更新角点跟踪器
        y, x = torch.where((pred[0, 0, 0, :, :] > 0).squeeze())
        tracker = update_nn_tracker(tracker, x, y, events['t'][0])

        # 显示或保存结果
        show_write(tracker.show(None))


if __name__ == '__main__':
    with torch.no_grad():
        main()