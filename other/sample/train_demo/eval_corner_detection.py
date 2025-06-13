# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
This script is a preprocessing script to be run on the Atis Corner Dataset preceding
the script compute_homography_reprojection_error. It will create csv files of corner positions
which can later be evaluated.

"""
"""
Corner Detection Evaluation Script
用于评估角点检测模型，并生成角点检测结果。
"""

import os
import argparse
import torch
import csv

from metavision_core_ml.corner_detection.lightning_model import CornerDetectionLightningModel
from metavision_core_ml.preprocessing.event_to_tensor_torch import event_cd_to_torch, event_volume
from metavision_core_ml.corner_detection.corner_tracker import CornerTracker
from metavision_core_ml.corner_detection.utils import update_nn_tracker, save_nn_corners, clean_pred
from metavision_core.event_io.py_reader import EventDatReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='', help='包含事件数据的文件夹路径')
    parser.add_argument('checkpoint', type=str, default='', help='用于评估的模型检查点路径')
    parser.add_argument('--results-path', type=str, default='', help='保存评估结果的路径')
    parser.add_argument('--cpu', action='store_true', help='使用 CPU 而非 GPU')

    params, _ = parser.parse_known_args()
    print('参数: ', params)

    # 遍历事件文件
    for events_filename in os.listdir(params.path):
        if "_td.dat" not in events_filename:
            continue
        events_path = os.path.join(params.path, events_filename)
        print("评估文件: {}".format(events_path))

        # 加载事件数据
        data_reader = EventDatReader(events_path)
        height, width = data_reader.get_size()

        # 加载模型
        device = "cuda" if not params.cpu else "cpu"
        model = CornerDetectionLightningModel.load_from_checkpoint(params.checkpoint)
        model.eval().to(device)

        # 初始化角点跟踪器
        tracker = CornerTracker(time_tolerance=7000)
        csv_path = os.path.join(params.results_path, events_filename).replace("_td.dat", ".csv")
        print("保存结果到: {}".format(csv_path))
        csv_file = open(csv_path, "w")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["x", "y", "t", "id"])

        # 遍历事件流
        while not data_reader.is_done():
            events = data_reader.load_delta_t(10000)
            if events is None or len(events) < 2:
                continue

            # 转换事件为张量
            events_th = event_cd_to_torch(events).to(device)
            tensor_th = event_volume(events_th, 1, height, width, None, None, 10, 'bilinear')

            # 模型预测
            pred = model.model(tensor_th)
            pred = torch.sigmoid(pred)
            pred = clean_pred(pred, threshold=0.3)

            # 更新角点跟踪器
            y, x = torch.where((pred[0, 0, 0, :, :] > 0).squeeze())
            tracker = update_nn_tracker(tracker, x, y, events['t'][0])

            # 保存角点结果
            save_nn_corners(tracker, csv_writer, events['t'][0])

        csv_file.close()


if __name__ == '__main__':
    with torch.no_grad():
        main()