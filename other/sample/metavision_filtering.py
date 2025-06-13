# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Code sample showing how to create a simple application to filter and display events.
"""
"""
此代码示例展示了如何创建一个简单的应用程序来过滤和显示事件。
"""

from enum import Enum
from metavision_core.event_io import EventsIterator
from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, PolarityFilterAlgorithm, RoiFilterAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent

# 定义 ROI（感兴趣区域）的裁剪宽度
roi_crop_width = 150


class Polarity(Enum):
    """
    定义事件极性枚举类
    """
    ALL = -1  # 所有事件
    OFF = 0   # 负极性事件
    ON = 1    # 正极性事件


def parse_args():
    """
    解析命令行参数
    """
    import argparse
    parser = argparse.ArgumentParser(description='Metavision Filtering sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input-event-file', dest='event_file_path', default="",
        help="输入事件文件的路径（RAW 或 HDF5）。如果未指定，则使用实时相机流。"
             "如果提供的是相机序列号，将尝试打开该相机。")
    parser.add_argument(
        '-r', '--replay_factor', type=float, default=1,
        help="重放因子。如果大于 1.0，则以慢动作重放；否则加速重放。")
    args = parser.parse_args()
    return args


def main():
    """
    主函数
    """
    args = parse_args()  # 解析命令行参数

    print("此代码示例展示了如何创建一个简单的应用程序来过滤和显示事件。")
    print("可用的键盘选项：\n"
          "  - R: 切换 ROI 过滤算法\n"
          "  - P: 仅显示正极性事件\n"
          "  - N: 仅显示负极性事件\n"
          "  - A: 显示所有事件\n"
          "  - Q/Escape: 退出应用程序\n")

    # 创建事件迭代器，用于从相机或事件文件中读取事件
    mv_iterator = EventsIterator(input_path=args.event_file_path, delta_t=1000)
    if args.replay_factor > 0 and not is_live_camera(args.event_file_path):
        # 如果不是实时相机，则使用 LiveReplayEventsIterator 模拟实时事件流
        mv_iterator = LiveReplayEventsIterator(mv_iterator, replay_factor=args.replay_factor)
    height, width = mv_iterator.get_size()  # 获取相机的分辨率（高度和宽度）

    # 创建极性过滤器
    polarity_filters = {Polarity.OFF: PolarityFilterAlgorithm(0), Polarity.ON: PolarityFilterAlgorithm(1)}
    # 创建 ROI 过滤器
    roi_filter = RoiFilterAlgorithm(x0=roi_crop_width, y0=roi_crop_width,
                                    x1=width - roi_crop_width, y1=height - roi_crop_width)
    events_buf = RoiFilterAlgorithm.get_empty_output_buffer()  # 用于存储过滤后的事件
    use_roi_filter = False  # 是否启用 ROI 过滤
    polarity = Polarity.ALL  # 当前极性过滤模式

    # 创建事件帧生成器，用于将事件流转换为帧
    event_frame_gen = PeriodicFrameGenerationAlgorithm(width, height, accumulation_time_us=10000)

    # 创建窗口，用于显示过滤后的事件并处理键盘事件
    with MTWindow(title="Metavision Filtering", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
        def on_cd_frame_cb(ts, cd_frame):
            """
            帧生成回调函数，用于显示生成的帧

            Args:
                ts: 帧的时间戳
                cd_frame: 生成的帧
            """
            # 处理窗口事件（如键盘输入）
            EventLoop.poll_and_dispatch()
            window.show_async(cd_frame)  # 异步显示帧

        event_frame_gen.set_output_callback(on_cd_frame_cb)  # 设置帧生成回调

        def keyboard_cb(key, scancode, action, mods):
            """
            键盘回调函数，用于处理用户输入

            Args:
                key: 按下的键
                scancode: 键码
                action: 键盘动作（按下或释放）
                mods: 修饰键（Shift、Ctrl、Alt）
            """
            nonlocal use_roi_filter
            nonlocal polarity

            if action != UIAction.RELEASE:  # 仅处理按键释放事件
                return
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()  # 按下 ESC 或 Q 关闭窗口
            elif key == UIKeyEvent.KEY_A:
                # 显示所有事件
                polarity = Polarity.ALL
            elif key == UIKeyEvent.KEY_N:
                # 仅显示负极性事件
                polarity = Polarity.OFF
            elif key == UIKeyEvent.KEY_P:
                # 仅显示正极性事件
                polarity = Polarity.ON
            elif key == UIKeyEvent.KEY_R:
                # 切换 ROI 过滤
                use_roi_filter = not use_roi_filter

        window.set_keyboard_callback(keyboard_cb)  # 设置键盘回调

        # 处理事件流
        for evs in mv_iterator:
            if use_roi_filter:
                # 如果启用了 ROI 过滤，则先应用 ROI 过滤器
                roi_filter.process_events(evs, events_buf)
                if polarity in polarity_filters:
                    # 如果启用了极性过滤，则应用极性过滤器
                    polarity_filters[polarity].process_events_(events_buf)
                event_frame_gen.process_events(events_buf)  # 生成帧
            elif polarity in polarity_filters:
                # 如果仅启用了极性过滤，则直接应用极性过滤器
                polarity_filters[polarity].process_events(evs, events_buf)
                event_frame_gen.process_events(events_buf)  # 生成帧
            else:
                # 如果没有启用任何过滤器，则直接生成帧
                event_frame_gen.process_events(evs)

            if window.should_close():  # 如果窗口关闭标志被设置，则退出循环
                break


if __name__ == "__main__":
    main()