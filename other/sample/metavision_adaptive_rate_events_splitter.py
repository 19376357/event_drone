# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
此脚本使用 Metavision SDK 的 AdaptiveRateEventsIterator 和 AdaptiveRateEventsSplitterAlgorithm，
以自适应速率将事件流分割为帧，并可视化或保存结果。
"""

from metavision_core.event_io import AdaptiveRateEventsIterator
from metavision_sdk_core import AdaptiveRateEventsSplitterAlgorithm
import numpy as np
import cv2
import os
from skvideo.io import FFmpegWriter  # 用于保存视频

# 修复 skvideo 中的 numpy 兼容性问题
np.float = np.float64
np.int = np.int_

def events_to_diff_image(events, sensor_size, strict_coord=True):
    """
    将事件转换为差分图像

    Args:
        events: 包含事件的 numpy 数组
        sensor_size: 传感器的尺寸 (height, width)
        strict_coord: 是否严格检查坐标范围

    Returns:
        差分图像，表示事件的空间分布
    """
    xs = events["x"]  # 事件的 x 坐标
    ys = events["y"]  # 事件的 y 坐标
    ps = events["p"] * 2 - 1  # 极性（+1 或 -1）

    # 检查坐标是否在传感器范围内
    mask = (xs < sensor_size[1]) * (ys < sensor_size[0]) * (xs >= 0) * (ys >= 0)
    if strict_coord:
        assert (mask == 1).all()
    coords = np.stack((ys * mask, xs * mask))
    ps *= mask

    # 将事件坐标映射到图像索引
    try:
        abs_coords = np.ravel_multi_index(coords, sensor_size)
    except ValueError:
        raise ValueError("输入数组有问题！")

    # 生成差分图像
    img = np.bincount(abs_coords, weights=ps, minlength=sensor_size[0] * sensor_size[1])
    img = img.reshape(sensor_size)
    return img


def split_into_frames(input_event_file, thr_var_per_event=5e-4, downsampling_factor=2, disable_display=False,
                      filename_output_video=None):
    """
    将事件流分割为帧并显示或保存结果

    Args:
        input_event_file (str): 输入事件文件路径（RAW 或 HDF5）
        thr_var_per_event (float): 每个事件的最小方差阈值，用于生成新帧
        downsampling_factor (int): 输入帧的降采样因子
        disable_display (bool): 是否禁用显示窗口
        filename_output_video (str): 输出视频文件路径（如果需要保存视频）
    """
    assert downsampling_factor == int(downsampling_factor), "降采样因子必须是整数"
    assert downsampling_factor >= 0, "降采样因子必须 >= 0"

    # 创建自适应速率事件迭代器
    mv_adaptive_rate_iterator = AdaptiveRateEventsIterator(input_path=input_event_file,
                                                           thr_var_per_event=thr_var_per_event,
                                                           downsampling_factor=downsampling_factor)

    # 获取传感器分辨率
    height, width = mv_adaptive_rate_iterator.get_size()

    # 初始化视频写入器
    if filename_output_video is None:
        video_process = None
    else:
        assert not os.path.exists(filename_output_video)
        video_process = FFmpegWriter(filename_output_video)

    # 初始化显示窗口
    if video_process or not disable_display:
        img_bgr = np.zeros((height, width, 3), dtype=np.uint8)

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)

    # 遍历事件流并处理
    for events in mv_adaptive_rate_iterator:
        assert events.size > 0
        start_ts = events[0]["t"]  # 当前帧的起始时间戳
        end_ts = events[-1]["t"]  # 当前帧的结束时间戳
        print("frame: {} -> {}   delta_t: {}   fps: {}   nb_ev: {}".format(
            start_ts, end_ts, end_ts - start_ts, 1e6 / (end_ts - start_ts), events.size))

        if video_process or not disable_display:
            # 将事件转换为差分图像
            img = events_to_diff_image(events, sensor_size=(height, width))
            img_bgr[...] = 0
            img_bgr[img < 0, 0] = 255  # 负极性事件显示为蓝色
            img_bgr[img > 0, 1] = 255  # 正极性事件显示为绿色

            # 在图像上叠加帧信息
            chunk_start_ts = events[0]["t"]
            chunk_end_ts = events[-1]["t"]
            delta_t_frame = chunk_end_ts - chunk_start_ts + 1
            frame_txt = "ts: {} -> {}  delta_t: {}  fps: {}  (nb_ev): {}".format(
                chunk_start_ts, chunk_end_ts, delta_t_frame, int(1.e6 / delta_t_frame), events.size)
            img_bgr[20:45, ...] = 0
            cv2.putText(img_bgr, frame_txt, (int(0.05 * width), 40),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 200, 100))

        # 保存帧到视频
        if video_process:
            video_process.writeFrame(img_bgr.astype(np.uint8)[..., ::-1])

        # 显示帧
        if not disable_display:
            cv2.imshow("img", img_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下 'q' 键退出
                break

    # 关闭视频写入器和窗口
    if video_process:
        video_process.close()
    if not disable_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import fire
    fire.Fire(split_into_frames)  # 使用 Fire 库解析命令行参数并调用函数