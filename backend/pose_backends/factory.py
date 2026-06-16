import os


def make_backend(name: str = None):
    name = (name or os.getenv("FIELD_POSE_BACKEND", "yolo")).lower()
    if name == "mediapipe":
        from pose_backends.mediapipe_backend import MediaPipeBackend
        return MediaPipeBackend()
    from pose_backends.yolo_backend import YOLO26Backend
    return YOLO26Backend()
