# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
This application demonstrates how to use Metavision SDK Stream module to decode an event recording, process it and
encode it back to RAW EVT2 format.
"""
"""
此应用程序演示如何使用 Metavision SDK 的 Stream 模块解码事件记录，对其进行处理，
并将其重新编码为 RAW EVT2 格式。
"""

import os
import sys
from metavision_sdk_base import EventCDBuffer
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import FlipYAlgorithm
from metavision_sdk_stream import RAWEvt2EventFileWriter


def parse_args():
    """
    解析命令行参数
    """
    import argparse
    parser = argparse.ArgumentParser(description='Metavision RAW EVT encoder sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-event-file', type=str, dest='event_file_path',
                        required=True, help="输入事件文件的路径（RAW 或 HDF5 格式）")
    parser.add_argument('-o', '--output-file', type=str, dest='output_file', default="",
                        help="输出 RAW 文件的路径。如果未指定，将使用输入路径的修改版本。")
    parser.add_argument('--encode-triggers', action='store_true', dest='encode_triggers',
                        help="激活外部触发事件编码的标志。")
    parser.add_argument('--max-event-latency', type=int, dest='max_event_latency', default=-1,
                        help="接收事件的最大相机时间延迟（默认为无限）。")
    parser.add_argument('-s', '--start-ts', type=int,
                        default=0, help="开始时间（以微秒为单位）。")
    parser.add_argument('-d', '--max-duration', type=int,
                        default=sys.maxsize, help="最大持续时间（以微秒为单位）。")
    parser.add_argument('--delta-t', type=int, default=100000,
                        help="提供的事件切片的持续时间（以微秒为单位）。")
    args = parser.parse_args()
    return args


def main():
    """
    主函数：解码事件文件，处理事件并重新编码为 RAW EVT2 格式
    """
    args = parse_args()  # 解析命令行参数

    # 检查输入文件是否存在
    if not os.path.isfile(args.event_file_path):
        raise TypeError(f'无法访问文件: {args.event_file_path}')
    
    # 如果未指定输出文件路径，则生成默认路径
    if args.output_file == "":
        args.output_file = args.event_file_path[:-4] + "_evt_encoded.raw"

    # 创建事件迭代器，用于从输入文件中读取事件
    mv_iterator = EventsIterator(input_path=args.event_file_path, delta_t=args.delta_t, 
                                  start_ts=args.start_ts, max_duration=args.max_duration)
    stream_height, stream_width = mv_iterator.get_size()  # 获取事件流的分辨率

    # 创建 Y 轴翻转算法实例
    yflipper = FlipYAlgorithm(stream_height - 1)

    # 创建 RAW EVT2 文件写入器
    writer = RAWEvt2EventFileWriter(
        stream_width, stream_height, args.output_file, args.encode_triggers, {}, args.max_event_latency)

    print("正在处理输入文件...")
    evs_processed_buf = EventCDBuffer()  # 用于存储处理后的事件

    # 遍历事件流
    for evs in mv_iterator:
        # 使用 Y 轴翻转算法处理事件
        yflipper.process_events(evs, evs_processed_buf)

        # 将处理后的事件写入输出文件
        writer.add_cd_events(evs_processed_buf)

        # 如果启用了外部触发事件编码，则处理触发事件
        if args.encode_triggers:
            writer.add_ext_trigger_events(mv_iterator.reader.get_ext_trigger_events())
            mv_iterator.reader.clear_ext_trigger_events()

    # 刷新并关闭写入器
    writer.flush()
    writer.close()
    print("完成！")


if __name__ == "__main__":
    main()