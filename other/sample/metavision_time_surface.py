# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Example of using Metavision SDK Core Python API for visualizing Time Surface of events
"""
"""
此示例演示如何使用 Metavision SDK Core Python API 可视化事件的时间表面 (Time Surface)。
"""

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import EventPreprocessor, MostRecentTimestampBuffer
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
import numpy as np
import cv2
import argparse


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Metavision 时间表面示例。',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input-event-file', dest='event_file_path', default="",
        help="输入事件文件的路径（RAW 或 HDF5）。如果未指定，则使用实时相机流。"
             "如果提供的是相机序列号，将尝试打开该相机。")
    args = parser.parse_args()
    return args


def main():
    """
    主函数：处理事件并生成时间表面
    """
    args = parse_args()  # 解析命令行参数

    last_processed_timestamp = 0  # 记录最后处理的时间戳

    # 创建事件迭代器，用于从相机或事件文件中读取事件
    mv_iterator = EventsIterator(input_path=args.event_file_path, delta_t=10000)
    height, width = mv_iterator.get_size()  # 获取相机的分辨率（高度和宽度）

    # 如果不是实时相机，则使用 LiveReplayEventsIterator 模拟实时事件流
    if not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator)

    # 创建窗口，用于显示时间表面
    with MTWindow(title="Metavision Events Viewer", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:
        def keyboard_cb(key, scancode, action, mods):
            """
            键盘回调函数，用于处理键盘事件

            Args:
                key: 按下的键
                scancode: 键码
                action: 键盘动作（按下或释放）
                mods: 修饰键（Shift、Ctrl、Alt）
            """
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()  # 按下 ESC 或 Q 关闭窗口

        window.set_keyboard_callback(keyboard_cb)  # 设置键盘回调

        # 创建时间表面缓冲区，用于存储最近的时间戳
        time_surface = MostRecentTimestampBuffer(rows=height, cols=width, channels=1)

        # 创建时间表面处理器，用于处理事件并生成时间表面
        ts_prod = EventPreprocessor.create_TimeSurfaceProcessor(
            input_event_width=width, input_event_height=height, split_polarity=False)

        # 创建用于显示的图像缓冲区
        img = np.empty((height, width), dtype=np.uint8)

        # 处理事件流
        for evs in mv_iterator:
            # 处理窗口事件（如键盘输入）
            EventLoop.poll_and_dispatch()

            if len(evs) == 0:  # 如果没有事件，跳过当前循环
                continue

            # 处理当前事件并更新时间表面缓冲区
            ts_prod.process_events(
                cur_frame_start_ts=evs[0][3],  # 当前帧的起始时间戳
                events_np=evs,  # 当前帧的事件
                frame_tensor_np=time_surface.numpy()  # 时间表面缓冲区
            )

            # 更新最后处理的时间戳
            last_processed_timestamp = evs[-1][3]

            # 根据时间表面缓冲区生成图像
            time_surface.generate_img_time_surface(
                last_processed_timestamp,  # 当前时间戳
                10000,  # 时间窗口（微秒）
                img  # 输出图像缓冲区
            )

            # 使用伪彩色显示时间表面
            window.show_async(cv2.applyColorMap(img, cv2.COLORMAP_JET))

            # 如果窗口关闭标志被设置，则退出循环
            if window.should_close():
                break


if __name__ == "__main__":
    main()