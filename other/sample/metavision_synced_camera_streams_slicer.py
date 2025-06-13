# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
This application demonstrates how to use Metavision SDK Stream module to slice events from synchronized cameras
"""
"""
此应用程序演示如何使用 Metavision SDK 的 Stream 模块从同步相机中切片事件。
"""

import numpy as np
from pathlib import Path
from typing import Optional

from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_stream import SyncedCameraSystemBuilder, SyncedCameraStreamsSlicer, FileConfigHints, \
    SliceCondition
from metavision_sdk_ui import MTWindow, BaseWindow, EventLoop, UIAction, UIKeyEvent


class CameraView:
    """
    用于显示单个相机事件切片的窗口类
    """

    def __init__(self, camera, name):
        """
        构造函数

        Args:
            camera: 要显示的相机对象
            name: 窗口名称
        """
        width = camera.width()
        height = camera.height()
        self.frame = np.zeros((height, width, 3), np.uint8)  # 初始化显示帧
        self.window = MTWindow(name, width, height, BaseWindow.RenderMode.BGR, True)  # 创建窗口

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
                self.window.set_close_flag()  # 按下 ESC 或 Q 关闭窗口

        self.window.set_keyboard_callback(keyboard_cb)  # 设置键盘回调

    def process(self, events):
        """
        从事件生成帧并显示在窗口中

        Args:
            events: 要显示的事件
        """
        BaseFrameGenerationAlgorithm.generate_frame(events, self.frame)  # 生成帧
        self.window.show_async(self.frame)  # 异步显示帧


def parse_args():
    """
    解析命令行参数
    """
    import argparse
    parser = argparse.ArgumentParser(
        description=("示例代码展示如何使用 Metavision SyncedCameraStreamsSlicer 从主从相机系统中切片事件")
    )

    # 基础选项
    parser.add_argument(
        '-i', '--input-event-files', nargs='+', default=[],
        help="输入事件文件路径（第一个为主相机）。如果未指定，则使用实时相机流。")
    parser.add_argument('-s', '--camera-serial-numbers', nargs='+', default=[],
                        help="要使用的相机序列号（第一个为主相机）")
    parser.add_argument('-r', '--real-time-playback', action='store_true',
                        help="以录制速度播放记录的标志")
    parser.add_argument('--record', type=bool, default=False,
                        help="是否记录流的标志")
    parser.add_argument('--record-path', type=str, default="",
                        help="保存记录流的路径")
    parser.add_argument('--config-path', type=str, default="",
                        help="加载每个实时相机配置文件的路径")

    # 切片选项
    parser.add_argument('-m', '--slicing-mode', type=str,
                        choices=['N_EVENTS', 'N_US', 'MIXED'],
                        default='N_US', help="切片模式（N_EVENTS, N_US, MIXED）")
    parser.add_argument('-t', '--delta-ts', type=int, default=10000,
                        help="切片持续时间（微秒，默认=10000us）")
    parser.add_argument('-n', '--delta-n-events', type=int, default=100000,
                        help="切片中的事件数量（默认=100000）")

    args = parser.parse_args()

    # 根据切片模式设置切片条件
    if args.slicing_mode == 'IDENTITY':
        args.slice_condition = SliceCondition.make_identity()
    elif args.slicing_mode == 'N_EVENTS':
        args.slice_condition = SliceCondition.make_n_events(args.delta_n_events)
    elif args.slicing_mode == 'N_US':
        args.slice_condition = SliceCondition.make_n_us(args.delta_ts)
    elif args.slicing_mode == 'MIXED':
        args.slice_condition = SliceCondition.make_mixed(args.delta_ts, args.delta_n_events)
    else:
        raise ValueError(f"无效的切片模式: {args.slicing_mode}")

    return args


def build_slicer(args):
    """
    根据命令行参数构建 SyncedCameraStreamsSlicer

    Args:
        args: 命令行参数

    Returns: SyncedCameraStreamsSlicer 实例
    """
    builder = SyncedCameraSystemBuilder()  # 创建同步相机系统构建器

    def get_settings_file_path(config_dir, serial_number) -> Optional[Path]:
        """
        获取相机配置文件路径

        Args:
            config_dir: 配置文件目录
            serial_number: 相机序列号

        Returns: 配置文件路径（如果存在）
        """
        settings_file_path = Path(config_dir) / f"{serial_number}.json"
        if not settings_file_path.exists():
            return None
        return settings_file_path

    # 添加实时相机参数
    for sn in args.camera_serial_numbers:
        print(f"添加序列号为 {sn} 的相机")
        settings_file_path = get_settings_file_path(args.config_path, sn)
        builder.add_live_camera_parameters(serial_number=sn, settings_file_path=settings_file_path)

    builder.set_record(args.record)  # 设置是否记录流
    builder.set_record_dir(args.record_path)  # 设置记录路径

    # 添加事件文件路径
    for record in args.input_event_files:
        builder.add_record_path(record)

    hints = FileConfigHints()
    hints.real_time_playback(args.real_time_playback)  # 设置实时播放标志

    builder.set_file_config_hints(hints)

    [master, slaves] = builder.build()  # 构建主从相机系统
    return SyncedCameraStreamsSlicer(master.move(), [slave.move() for slave in slaves], args.slice_condition)


def build_views(slicer):
    """
    根据 SyncedCameraStreamsSlicer 构建 CameraView 实例

    Args:
        slicer: SyncedCameraStreamsSlicer 实例

    Returns: CameraView 实例列表
    """
    views = [CameraView(slicer.master(), "Master")]  # 创建主相机视图

    for i in range(slicer.slaves_count()):
        views.append(CameraView(slicer.slave(i), f"Slave {i}"))  # 创建从相机视图

    return views


def should_exit(views):
    """
    检查程序是否应该退出（当任意窗口关闭时）

    Args:
        views: CameraView 实例列表

    Returns: 如果程序应该退出则返回 True
    """
    for view in views:
        if view.window.should_close():
            return True
    return False


def log_slice_info(slice):
    """
    记录切片信息

    Args:
        slice: 同步事件切片
    """
    print(f"===== 切片信息 =====")
    print(f"时间戳: {slice.t}")
    print(f"主相机事件数: {slice.n_events}")
    for i, slave_slice in enumerate(slice.slave_events):
        print(f"从相机 {i + 1} 事件数: {len(slave_slice)}")
    print("=================\n")


def main():
    args = parse_args()  # 解析命令行参数
    slicer = build_slicer(args)  # 构建切片器
    views = build_views(slicer)  # 构建相机视图

    # 主循环处理切片
    for slice in slicer:
        EventLoop.poll_and_dispatch()  # 处理事件循环

        log_slice_info(slice)  # 记录切片信息

        views[0].process(slice.master_events)  # 处理主相机事件

        for i in range(slicer.slaves_count()):
            views[i + 1].process(slice.slave_events[i])  # 处理从相机事件

        if should_exit(views):  # 检查是否退出
            break

    for view in views:
        view.window.destroy()  # 销毁窗口


if __name__ == "__main__":
    main()