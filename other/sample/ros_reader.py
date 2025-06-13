# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
RosBag Reader

This is an example on how to read a different format.
Here we use the "rospy" library to read .bag files:
https://github.com/rospypi/simple

You can install the library like so:
python3 -m pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag
"""
"""
RosBag Reader

此代码示例展示了如何使用 `rospy` 库读取 ROS 的 .bag 文件。
它通过读取 ROS Bag 文件中的事件数据，并将其转换为 Metavision SDK 的 `EventCD` 格式的 numpy 数组。

依赖库：
- `rospy`
- `rosbag`

安装方法：
python3 -m pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag
"""


import sqlite3
import numpy as np
from metavision_sdk_base import EventCD
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

class Ros2BaseReader_eventCD:
    """
    用于读取 ROS2 .db3 格式的 Bag 文件的基础读取器。
    提供一个迭代器，返回类型为 `EventCD` 的 numpy 缓冲区。
    """

    def __init__(self, rosbag_path, event_topic='/dvs/events'):
        """
        初始化读取器。

        Args:
            rosbag_path (str): ROS2 Bag 文件的路径。
            event_topic (str): 事件数据的 ROS 话题名称（默认值为 '/dvs/events'）。
        """
        self.conn = sqlite3.connect(rosbag_path)  # 打开 ROS2 Bag 文件
        self.cursor = self.conn.cursor()
        self.event_topic = event_topic  # 事件数据的 ROS 话题名称

        # 获取所有话题及其类型
        self.cursor.execute("SELECT name, type FROM topics")
        self.topics = {name: type_ for name, type_ in self.cursor.fetchall()}

        # 检查指定的事件话题是否存在于 Bag 文件中
        assert event_topic in self.topics, '指定的事件话题不存在'

        # 获取消息类型
        self.msg_type = get_message(self.topics[event_topic])

        self.t0 = None  # 初始时间戳
        self.height, self.width = -1, -1  # 初始化传感器的分辨率

        # 从前几个消息中提取传感器的分辨率（高度和宽度）
        self.cursor.execute(f"SELECT data FROM messages WHERE topic_id = (SELECT id FROM topics WHERE name = '{event_topic}') LIMIT 6")
        for row in self.cursor.fetchall():
            msg = deserialize_message(row[0], self.msg_type)
            if hasattr(msg, "width"):
                self.height = msg.height
                self.width = msg.width

        # 如果未找到分辨率信息，则抛出异常
        if self.height < 0 or self.width < 0:
            raise BaseException("未找到包含高度或宽度字段的消息")

    def is_done(self):
        """
        检查读取器是否已完成（此处始终返回 False）。
        """
        return False

    def __del__(self):
        """
        析构函数，用于清理资源。
        """
        self.conn.close()

    def get_size(self):
        """
        获取传感器的分辨率。

        Returns:
            tuple: (height, width) 传感器的高度和宽度。
        """
        return self.height, self.width

    def seek_time(self, ts):
        """
        跳转到指定时间戳（未实现）。

        Args:
            ts (int): 时间戳。

        Raises:
            Exception: 如果尝试调用此方法，则抛出异常。
        """
        if ts != 0:
            raise Exception('时间跳转功能未在 ROS2 中实现')

    def __iter__(self):
        """
        迭代器，用于逐帧读取事件数据。

        Yields:
            numpy.ndarray: 包含事件数据的 numpy 数组，类型为 `EventCD`。
        """
        t0 = None  # 初始时间戳

        # 查询指定话题的所有消息
        self.cursor.execute(f"SELECT timestamp, data FROM messages WHERE topic_id = (SELECT id FROM topics WHERE name = '{self.event_topic}')")
        for timestamp, data in self.cursor.fetchall():
            msg = deserialize_message(data, self.msg_type)  # 反序列化消息
            evs = msg.events  # 获取事件列表
            num = len(evs)  # 事件数量

            # 初始化初始时间戳
            if t0 is None:
                t0 = evs[0].ts.to_nsec() / 1000  # 转换为微秒

            # 创建一个空的 numpy 数组，用于存储事件数据
            event_buffer = np.zeros((num,), dtype=EventCD)

            # 遍历事件并填充到 numpy 数组中
            for n, ev in enumerate(evs):
                event_buffer[n]['x'] = ev.x  # 事件的 x 坐标
                event_buffer[n]['y'] = ev.y  # 事件的 y 坐标
                event_buffer[n]['p'] = ev.polarity  # 事件的极性（正或负）
                event_buffer[n]['t'] = ev.ts.to_nsec() / 1000 - t0  # 事件的时间戳（相对初始时间）

            # 返回当前帧的事件数据
            yield event_buffer

class Ros2BaseReader:
    """
    通用 ROS2 Bag 文件读取器，支持读取任意话题的数据。
    """

    def __init__(self, rosbag_path):
        """
        初始化读取器。

        Args:
            rosbag_path (str): ROS2 Bag 文件的路径（.db3 文件）。
        """
        self.conn = sqlite3.connect(rosbag_path)  # 打开 ROS2 Bag 文件
        self.cursor = self.conn.cursor()

        # 获取所有话题及其类型
        self.cursor.execute("SELECT name, type FROM topics")
        self.topics = {name: type_ for name, type_ in self.cursor.fetchall()}

    def list_topics(self):
        """
        列出 Bag 文件中的所有话题及其类型。

        Returns:
            dict: 包含话题名称和类型的字典。
        """
        return self.topics

    def read_messages(self, topic_name):
        """
        读取指定话题的所有消息。

        Args:
            topic_name (str): 要读取的 ROS 话题名称。

        Yields:
            tuple: (timestamp, message)，分别为消息的时间戳和反序列化后的消息对象。
        """
        if topic_name not in self.topics:
            raise ValueError(f"指定的话题 '{topic_name}' 不存在于 Bag 文件中。")

        # 获取消息类型
        msg_type = get_message(self.topics[topic_name])

        # 查询指定话题的所有消息
        self.cursor.execute(
            f"SELECT timestamp, data FROM messages WHERE topic_id = (SELECT id FROM topics WHERE name = '{topic_name}')"
        )
        for timestamp, data in self.cursor.fetchall():
            # 反序列化消息
            msg = deserialize_message(data, msg_type)
            yield timestamp, msg

    def __del__(self):
        """
        析构函数，用于清理资源。
        """
        self.conn.close()