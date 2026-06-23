import os


def make_backend(name: str = None):
    name = (name or os.getenv("FIELD_POSE_BACKEND", "yolo")).lower()
    if name == "mediapipe":
        from pose_backends.mediapipe_backend import MediaPipeBackend
        return MediaPipeBackend()
    if name == "rtmpose3d":
        from pose_backends.rtmpose3d_backend import RTMPose3DBackend
        return RTMPose3DBackend()
    from pose_backends.yolo_backend import YOLO26Backend
    return YOLO26Backend()
