from sample.ros_reader import Ros2BaseReader_eventCD, Ros2BaseReader

'''
Available topics: {'/davis/right/imu': 'sensor_msgs/msg/Imu', 
'/davis/left/imu': 'sensor_msgs/msg/Imu', '/davis/right/events': 
'dvs_msgs/msg/EventArray', '/davis/left/events': 'dvs_msgs/msg/EventArray', 
'/davis/left/camera_info': 'sensor_msgs/msg/CameraInfo', '/davis/left/image_raw': 
'sensor_msgs/msg/Image', '/velodyne_point_cloud': 'sensor_msgs/msg/PointCloud2', 
'/davis/right/camera_info': 'sensor_msgs/msg/CameraInfo', '/davis/right/image_raw': 
'sensor_msgs/msg/Image'}
'''

def main():
    # 指定你的 .db3 文件路径和事件话题名称
    rosbag_path = "data/ros2/rosbag.db3"  # 替换为你的 .db3 文件路径
    ros1bag_path = "data/rosbag/indoor_flying1_data.bag"  # 替换为你的 .bag 文件路径

    reader = Ros2BaseReader(rosbag_path)
    topics = "/davis/right/events"
    for timestamp, msg in reader.read_messages(topics):
        print(f"Timestamp: {timestamp}, Message: {msg}")
        
if __name__ == "__main__":
    main()