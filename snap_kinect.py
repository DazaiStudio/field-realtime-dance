"""Grab one color + depth + skeleton-overlay snapshot from the Azure Kinect.

Run: py -3.10 snap_kinect.py
Writes qa_screenshots/kinect_snapshot.png (color with skeleton | colorized depth).

Note: gpu_device_id=1 selects the RTX 4080 on this laptop -- DirectML adapter 0
is the integrated GPU and runs body tracking at ~6 fps instead of 30.
"""
import os

import cv2
import numpy as np
import pykinect_azure as pykinect
from pykinect_azure.k4abt import _k4abtTypes as T

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "qa_screenshots", "kinect_snapshot.png")


def main():
    pykinect.initialize_libraries(track_body=True)
    T.k4abt_tracker_default_configuration.processing_mode = \
        T.K4ABT_TRACKER_PROCESSING_MODE_GPU_DIRECTML
    T.k4abt_tracker_default_configuration.gpu_device_id = \
        int(os.getenv("FIELD_KINECT_GPU", "1"))

    config = pykinect.default_configuration
    config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
    config.synchronized_images_only = True

    device = pykinect.start_device(config=config)
    tracker = pykinect.start_body_tracker()

    color_img = depth_img = None
    n_bodies = 0
    for _ in range(40):  # let auto-exposure settle
        capture = device.update()
        body_frame = tracker.update()
        ret_c, c = capture.get_color_image()
        ret_d, d = capture.get_colored_depth_image()
        if ret_c and ret_d:
            n_bodies = body_frame.get_num_bodies()
            if n_bodies > 0:
                c = body_frame.draw_bodies(c, pykinect.K4A_CALIBRATION_TYPE_COLOR)
            color_img, depth_img = c, d
    device.close()

    if color_img is None or depth_img is None:
        raise SystemExit("no synchronized capture received")

    if color_img.shape[2] == 4:
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGRA2BGR)
    if depth_img.shape[2] == 4:
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGRA2BGR)

    scale = color_img.shape[0] / depth_img.shape[0]
    depth_img = cv2.resize(depth_img, (int(depth_img.shape[1] * scale),
                                       color_img.shape[0]))
    side = np.hstack([color_img, depth_img])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, side)
    print(f"saved {OUT}")
    print(f"bodies in frame: {n_bodies}")


if __name__ == "__main__":
    main()
