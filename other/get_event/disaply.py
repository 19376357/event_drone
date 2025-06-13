from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_stream import Camera, CameraStreamSlicer, FileConfigHints, SliceCondition
import numpy as np
import cv2
import os
import sys
import time

import metavision_core_ml


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision SDK Get Started sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input-event-file',
        help="Path to input event file (RAW or HDF5). If not specified, the camera live stream is used.")
    parser.add_argument(
        '-t', '--delta-ts', type=int, default=10000,
        help="Slice duration in microseconds (default=10000us)")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.input_event_file:
        print(f"Opening event file: {args.input_event_file}")
        camera = Camera.from_file(args.input_event_file)
    else:
        print("Opening first available camera")
        camera = Camera.from_first_available()

    global_counter = 0  # This will track how many events we processed
    global_max_t = 0  # This will track the highest timestamp we processed

    slice_condition = SliceCondition.make_n_us(args.delta_ts)
    slicer = CameraStreamSlicer(camera.move(), slice_condition)
    width = slicer.camera().width()
    height = slicer.camera().height()
    print(f"Camera resolution: {width}x{height}")
    frame = np.zeros((height, width, 3), np.uint8)

    cv2.namedWindow("Event Display", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Event Display", width, height)

    try:
        for slice in slicer:
            #s_time = time.time()
            print("----- New event slice! -----")
            if slice.events.size == 0:
                print("The current event slice is empty.")
            else:
                print(f"ts: {slice.t}, new slice of {slice.events.size} events")
                BaseFrameGenerationAlgorithm.generate_frame(slice.events, frame)
                cv2.imshow("Event Display", frame)
                #e_time = time.time()
                #print(e_time-s_time)
                # 等待1毫秒，同时检查是否按下了'q'键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
