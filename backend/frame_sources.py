"""Frame sources for the live stream loop.

stream_live() only needs read()/reopen()/release()/describe(). The OpenCV
implementation DELEGATES to osc_viewer's existing module-level camera
functions (global cap + camera_lock + owner token) instead of moving them —
that machinery encodes hard-won race fixes (see MAINTENANCE.md §5) and its
behavior must not change. The Kinect implementation wraps KinectRuntime.
"""
from __future__ import annotations

import time


class OpenCVFrameSource:
    """cv2.VideoCapture via osc_viewer's open/read/release functions."""

    def __init__(self, index: int, owner, open_fn, read_fn, release_fn):
        self._index = int(index)
        self._owner = owner
        self._open_fn = open_fn
        self._read_fn = read_fn
        self._release_fn = release_fn
        self._cap = None

    def open(self):
        self._cap = self._open_fn(self._index, self._owner)
        return self

    def read(self):
        if self._cap is None:
            return False, None
        return self._read_fn(self._cap, self._owner)

    def reopen(self, sleep_seconds: float = 0.35):
        self._release_fn(self._owner)
        if sleep_seconds:
            time.sleep(sleep_seconds)
        self._cap = self._open_fn(self._index, self._owner)

    def release(self):
        self._release_fn(self._owner)
        self._cap = None

    def describe(self) -> str:
        return f"Camera {self._index}"


class KinectFrameSource:
    """Azure Kinect via KinectRuntime (already acquired by the caller)."""

    def __init__(self, runtime, owner):
        self._runtime = runtime
        self._owner = owner

    def open(self):
        return self

    def read(self):
        return self._runtime.read()

    def reopen(self, sleep_seconds: float = 0.35):
        self._runtime.reopen()

    def release(self):
        self._runtime.release(owner=self._owner)

    def describe(self) -> str:
        return self._runtime.describe()
