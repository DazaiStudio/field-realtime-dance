"""Azure Kinect environment check for the FIELD viewer integration.

Run with the global Python 3.10 (the same interpreter the viewer uses):

    py -3.10 check_kinect_env.py

Without a device connected it verifies the SDK DLLs load; with a device it
opens the camera, grabs captures, and runs body tracking for a few frames.

Requires (already installed 2026-07-29):
  - Azure Kinect Sensor SDK v1.4.2   -> C:\\Program Files\\Azure Kinect SDK v1.4.2
  - Azure Kinect Body Tracking 1.1.2 -> C:\\Program Files\\Azure Kinect Body Tracking SDK
  - pip install pykinect_azure       (Python 3.10)
"""
import sys
import time


def main():
    print(f"python : {sys.version.split()[0]} ({sys.executable})")

    import pykinect_azure as pykinect

    # Loads k4a.dll (sensor) and k4abt.dll (body tracking, DirectML backend).
    pykinect.initialize_libraries(track_body=True)
    print("k4a.dll + k4abt.dll loaded OK")

    # DirectML adapter selection via FIELD_KINECT_GPU (same env the viewer
    # uses). Dual-GPU laptops: adapter 0 is usually the iGPU (~6 fps), so the
    # default is 1; single-GPU machines need FIELD_KINECT_GPU=0. CUDA mode is
    # broken (SDK's ORT 1.10 provider fails to init on Ada GPUs) -- DirectML
    # on the right adapter is the way.
    import os
    from pykinect_azure.k4abt import _k4abtTypes as T
    T.k4abt_tracker_default_configuration.processing_mode = \
        T.K4ABT_TRACKER_PROCESSING_MODE_GPU_DIRECTML
    T.k4abt_tracker_default_configuration.gpu_device_id = \
        int(os.getenv("FIELD_KINECT_GPU", "1"))

    from pykinect_azure.k4a.device import Device

    count = Device.device_get_installed_count()
    print(f"devices connected: {count}")
    if count == 0:
        print("no device attached -- plug the Kinect into a USB3 port and rerun")
        print("(hardware sanity check without Python: 'C:\\Program Files\\"
              "Azure Kinect SDK v1.4.2\\tools\\k4aviewer.exe')")
        return

    # Must match the viewer's configuration (backend/pose_backends/azure_kinect.py).
    # synchronized_images_only is the one that matters here: without it a capture
    # can arrive with a colour image but no depth, and the body tracker then dies
    # on "Get depth buffer from the capture handle failed!" -- a false alarm that
    # looks exactly like broken hardware.
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
    device_config.synchronized_images_only = True

    device = pykinect.start_device(config=device_config)
    try:
        body_tracker = pykinect.start_body_tracker()

        n_bodies = 0
        t0 = time.time()
        frames = 30
        for _ in range(frames):
            device.update()
            body_frame = body_tracker.update()
            n_bodies = body_frame.get_num_bodies()
        dt = time.time() - t0
        print(f"capture + body tracking OK: {frames} frames in {dt:.1f}s "
              f"({frames / dt:.1f} fps), bodies in last frame: {n_bodies}")
    finally:
        # Leaving the device open on the way out wedges it badly enough that the
        # next run reports "devices connected: 0" until the USB cable is replugged.
        try:
            device.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
