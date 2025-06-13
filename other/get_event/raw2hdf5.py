import os
import sys
import h5py
from metavision_sdk_base import EventCDBuffer
from metavision_core.event_io import EventsIterator
from metavision_sdk_stream import RAWEvt2EventFileWriter
import metavision_sdk_stream
import numpy as np


def parse_args():
    """
    解析命令行参数
    """
    import argparse
    parser = argparse.ArgumentParser(description='Metavision RAW to HDF5 converter.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-raw-file', type=str, dest='raw_file_path',
                        required=True, help="输入 RAW 文件的路径")
    parser.add_argument('-o', '--output-hdf5-file', type=str, dest='hdf5_file_path', default="",
                        help="输出 HDF5 文件的路径。如果未指定，将使用输入路径的修改版本。")
    parser.add_argument('--delta-t', type=int, default=50000,
                        help="提供的事件切片的持续时间（以微秒为单位）。")
    args = parser.parse_args()
    return args

def raw_to_hdf5(raw_file_path, hdf5_file_path, delta_t=50000):
    """
    将 .raw 文件转换为 .hdf5 文件。

    Args:
        raw_file_path (str): 输入 RAW 文件的路径。
        hdf5_file_path (str): 输出 HDF5 文件的路径。
        delta_t (int): 每个事件切片的时间间隔（以微秒为单位）。
    """
    # 检查输入文件是否存在
    if not os.path.isfile(raw_file_path):
        raise FileNotFoundError(f"无法访问文件: {raw_file_path}")

    # 如果未指定输出文件路径，则生成默认路径
    if hdf5_file_path == "":
        hdf5_file_path = raw_file_path[:-4] + ".hdf5"

    # 创建事件迭代器，用于从输入 RAW 文件中读取事件
    mv_iterator = EventsIterator(input_path=raw_file_path, delta_t=delta_t)
    reader = mv_iterator.reader  # 获取底层事件读取器
    stream_height, stream_width = mv_iterator.get_size()  # 获取事件流的分辨率


    # 创建 HDF5 文件
    with h5py.File(hdf5_file_path, 'w') as hdf5_file:
        # 创建事件数据集
        event_dataset = hdf5_file.create_dataset(
            'events',
            shape=(0, 4),  # (x, y, t, polarity)
            maxshape=(None, 4),
            dtype='float32',
            chunks=True
        )
        # 存储元信息
        hdf5_file.attrs['height'] = stream_height
        hdf5_file.attrs['width'] = stream_width
        hdf5_file.attrs['delta_t'] = delta_t

        # 初始化事件计数
        total_events = 0

        print("正在处理输入文件...")
        # 遍历事件流
        for events in mv_iterator:
            if len(events) == 0:
                continue
            events = np.column_stack((events['x'], events['y'], events['t'], events['p']))

            # 扩展数据集大小
            event_dataset.resize((total_events + len(events), 4))
            # 写入事件数据
            event_dataset[total_events:total_events + len(events)] = events
            total_events += len(events)

            print(f"已处理事件数: {total_events}")

    print(f"转换完成！总事件数: {total_events}")
    print(f"HDF5 文件已保存到: {hdf5_file_path}")


def main():
    """
    主函数：解析命令行参数并执行转换
    """
    args = parse_args()
    raw_to_hdf5(args.raw_file_path, args.hdf5_file_path, args.delta_t)


if __name__ == "__main__":
    main()