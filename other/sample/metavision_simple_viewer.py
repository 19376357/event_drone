# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Sample code that demonstrates how to use Metavision SDK to visualize events from a live camera or an event file
"""
# 导入必要的模块
from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera  # 用于处理事件流
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette  # 用于生成帧和设置颜色
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent  # 用于创建用户界面和处理事件循环
import argparse  # 用于解析命令行参数
import os  # 用于操作系统相关功能

# 定义一个函数来解析命令行参数
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision Simple Viewer sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 添加一个参数，用于指定输入事件文件路径
    parser.add_argument(
        '-i', '--input-event-file', dest='event_file_path', default="",
        help="Path to input event file (RAW, DAT or HDF5). If not specified, the camera live stream is used. "
        "If it's a camera serial number, it will try to open that camera instead.")
    args = parser.parse_args()  # 解析命令行参数
    return args  # 返回解析后的参数

# 主函数
def main():
    """ Main """
    args = parse_args()  # 获取命令行参数

    # 创建事件迭代器，用于从相机或事件文件中读取事件
    mv_iterator = EventsIterator(input_path=args.event_file_path, delta_t=1000)
    height, width = mv_iterator.get_size()  # 获取相机的分辨率（高度和宽度）

    # 如果输入不是实时相机，则使用实时回放事件迭代器
    if not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator)

    # 创建一个窗口，用于显示事件
    with MTWindow(title="Metavision Events Viewer", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:
        # 定义键盘回调函数，用于处理键盘事件
        def keyboard_cb(key, scancode, action, mods):
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:  # 按下 ESC 或 Q 键时关闭窗口
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)  # 设置键盘回调函数

        # 创建事件帧生成器，用于将事件转换为帧
        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width, sensor_height=height, fps=25,
                                                           palette=ColorPalette.Dark)

        # 定义帧生成回调函数，用于显示生成的帧
        def on_cd_frame_cb(ts, cd_frame):
            window.show_async(cd_frame)  # 异步显示帧

        event_frame_gen.set_output_callback(on_cd_frame_cb)  # 设置帧生成回调函数

        # 处理事件流
        for evs in mv_iterator:
            # 分发系统事件到窗口
            EventLoop.poll_and_dispatch()
            # 处理事件并生成帧
            event_frame_gen.process_events(evs)

            # 如果窗口需要关闭，则退出循环
            if window.should_close():
                break

# 如果脚本是直接运行的，则调用主函数
if __name__ == "__main__":
    main()