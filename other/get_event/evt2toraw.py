import os
import sys
from metavision_sdk_base import EventCDBuffer
from metavision_core.event_io import EventsIterator
from metavision_sdk_stream import RAWEvt2EventFileWriter


def parse_args():
    """
    解析命令行参数
    """
    import argparse
    parser = argparse.ArgumentParser(description='Metavision EVT2 to RAW decoder sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-evt2-file', type=str, dest='evt2_file_path',
                        required=True, help="输入 EVT2 文件的路径")
    parser.add_argument('-o', '--output-raw-file', type=str, dest='raw_file_path', default="",
                        help="输出 RAW 文件的路径。如果未指定，将使用输入路径的修改版本。")
    parser.add_argument('--delta-t', type=int, default=100000,
                        help="提供的事件切片的持续时间（以微秒为单位）。")
    args = parser.parse_args()
    return args


def main():
    """
    主函数：解码 EVT2 文件并重新编码为 RAW 格式
    """
    args = parse_args()  # 解析命令行参数

    # 检查输入文件是否存在
    if not os.path.isfile(args.evt2_file_path):
        raise TypeError(f'无法访问文件: {args.evt2_file_path}')
    
    # 如果未指定输出文件路径，则生成默认路径
    if args.raw_file_path == "":
        args.raw_file_path = args.evt2_file_path[:-4] + "_decoded.raw"

    # 创建事件迭代器，用于从输入 EVT2 文件中读取事件
    mv_iterator = EventsIterator(input_path=args.evt2_file_path, delta_t=args.delta_t)
    stream_height, stream_width = mv_iterator.get_size()  # 获取事件流的分辨率

    # 创建 RAW 文件写入器
    writer = RAWEvt2EventFileWriter(stream_width, stream_height, args.raw_file_path)

    print("正在处理输入文件...")
    evs_processed_buf = EventCDBuffer()  # 用于存储事件7

    # 遍历事件流
    for evs in mv_iterator:
        # 将事件写入 RAW 文件
        writer.add_cd_events(evs)

    # 刷新并关闭写入器
    writer.flush()
    writer.close()
    print("完成！")


if __name__ == "__main__":
    main()