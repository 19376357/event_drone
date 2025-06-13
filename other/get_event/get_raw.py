from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
from metavision_hal import DeviceConfig, DeviceDiscovery
import argparse
import time
import os

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision RAW file Recorder sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-o', '--output-dir', default="", help="Directory where to create RAW file with recorded event data")
    args = parser.parse_args()
    return args


def main():
    """ Main """
    args = parse_args()

    # HAL Device on live camera
    device_config = DeviceConfig()
    device_config.enable_biases_range_check_bypass(True)
    device = DeviceDiscovery.open("", device_config)

    #device = initiate_device("")
    biases = device.get_i_ll_biases()
    biases.set("bias_diff_off", 25)
    biases.set("bias_diff_on", 30)
    biases.set("bias_fo", 25)  
    biases.set("bias_hpf", 45)
    biases.set("bias_refr", 1)
    print("set biases:", biases.get_all_biases())

    # Start the recording
    if device.get_i_events_stream():
        log_path = "recording_" + time.strftime("%y%m%d_%H%M%S", time.localtime()) + ".raw"
        if args.output_dir != "":
            log_path = os.path.join(args.output_dir, log_path)
        print(f'Recording to {log_path}')
        device.get_i_events_stream().log_raw_data(log_path)

    # Events iterator on Device
    mv_iterator = EventsIterator.from_device(device=device)
    height, width = mv_iterator.get_size()  # Camera Geometry

    # Window - Graphical User Interface
    with MTWindow(title="Metavision Events Viewer", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:
        def keyboard_cb(key, scancode, action, mods):
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        # Event Frame Generator
        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width, sensor_height=height, fps=25,
                                                           palette=ColorPalette.Dark)

        def on_cd_frame_cb(ts, cd_frame):
            window.show_async(cd_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        # Process events
        for evs in mv_iterator:
            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()
            event_frame_gen.process_events(evs)

            if window.should_close():
                # Stop the recording
                device.get_i_events_stream().stop_log_raw_data()
                break


if __name__ == "__main__":
    main()