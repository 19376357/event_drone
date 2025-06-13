from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent
from metavision_hal import DeviceConfig, DeviceDiscovery
import time
import os


# ↑ / ↓：增加/减少 bias_fo (-150, 200)
# → / ←：增加/减少 bias_diff_off (-150, 200)
# W / S：增加/减少 bias_diff_on (-150, 200)
# A / D：增加/减少 bias_hpf (0, 255)
# E / R：增加/减少 bias_ref (-50, 255)

def main():
    """ Main """
    # 配置设备并启用范围检查绕过
    device_config = DeviceConfig()
    device_config.enable_biases_range_check_bypass(True)
    device = DeviceDiscovery.open("", device_config)

    # 获取偏置实例
    biases = device.get_i_ll_biases()

    # 打印所有偏置
    print("Current biases:", biases.get_all_biases())

    # 获取偏置范围
    bias_ranges = {
        "bias_fo": biases.get_bias_info("bias_fo").get_bias_allowed_range(),
        "bias_diff_off": biases.get_bias_info("bias_diff_off").get_bias_allowed_range(),
        "bias_diff_on": biases.get_bias_info("bias_diff_on").get_bias_allowed_range(),
        "bias_hpf": biases.get_bias_info("bias_hpf").get_bias_allowed_range(),
        "bias_refr": biases.get_bias_info("bias_refr").get_bias_allowed_range(),
    }
    print(f"Bias ranges: {bias_ranges}")

    # 设置初始偏置值
    biases.set("bias_diff_off", 25)
    biases.set("bias_diff_on", 30)
    biases.set("bias_fo", 25)  
    biases.set("bias_hpf", 45)
    biases.set("bias_refr", 1)
    print("Updated biases:", biases.get_all_biases())

    # 初始化事件迭代器
    mv_iterator = EventsIterator.from_device(device=device)
    height, width = mv_iterator.get_size()  # 相机分辨率

    # 创建窗口
    with MTWindow(title="Metavision Bias Viewer", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:
        def keyboard_cb(key, scancode, action, mods):
            # 获取当前偏置值
            current_biases = biases.get_all_biases()

            # 调整偏置值的函数
            def adjust_bias(bias_name, delta):
                current_value = current_biases[bias_name]
                allowed_range = bias_ranges[bias_name]
                new_value = max(min(current_value + delta, allowed_range[1]), allowed_range[0])
                biases.set(bias_name, new_value)
                print(f"{bias_name} adjusted to {new_value}")

            # 键盘控制逻辑
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()
            elif key == UIKeyEvent.KEY_UP:  # 增加 bias_fo
                adjust_bias("bias_fo", 1)
            elif key == UIKeyEvent.KEY_DOWN:  # 减少 bias_fo
                adjust_bias("bias_fo", -1)
            elif key == UIKeyEvent.KEY_RIGHT:  # 增加 bias_diff_off
                adjust_bias("bias_diff_off", 1)
            elif key == UIKeyEvent.KEY_LEFT:  # 减少 bias_diff_off
                adjust_bias("bias_diff_off", -1)
            elif key == UIKeyEvent.KEY_W:  # 增加 bias_diff_on
                adjust_bias("bias_diff_on", 1)
            elif key == UIKeyEvent.KEY_S:  # 减少 bias_diff_on
                adjust_bias("bias_diff_on", -1)
            elif key == UIKeyEvent.KEY_A:  # 增加 bias_hpf
                adjust_bias("bias_hpf", 1)
            elif key == UIKeyEvent.KEY_D:  # 减少 bias_hpf
                adjust_bias("bias_hpf", -1)
            elif key == UIKeyEvent.KEY_E:  # 增加 bias_refr
                adjust_bias("bias_refr", 1)
            elif key == UIKeyEvent.KEY_R:  # 减少 bias_refr
                adjust_bias("bias_refr", -1)

        window.set_keyboard_callback(keyboard_cb)

        # 事件帧生成器
        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width, sensor_height=height, fps=25,
                                                           palette=ColorPalette.Dark)

        def on_cd_frame_cb(ts, cd_frame):
            window.show_async(cd_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        # 处理事件
        for evs in mv_iterator:
            # 分发系统事件到窗口
            EventLoop.poll_and_dispatch()
            event_frame_gen.process_events(evs)

            if window.should_close():
                break

if __name__ == "__main__":
    main()
