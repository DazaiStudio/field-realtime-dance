"""Azure Kinect body-tracking backend: frame source + pose source.

Layering:
  - Pure helpers + AzureKinectPoseSource: no hardware deps, import anywhere,
    fully unit-tested against a fake runtime.
  - KinectRuntime: the only code that touches pykinect_azure (lazy imports).
    pykinect's VERIFY() calls sys.exit(1) on any sensor error, so every call
    into it goes through _guarded() which converts that into KinectError.

Verified on this machine (2026-07-30): DirectML gpu_device_id=1 = RTX 4080
(adapter 0 = iGPU at ~6 fps). Override with FIELD_KINECT_GPU on other rigs.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

# --- K4ABT joint indices used here (see keypoint_mapping for the mapping) ----
_CORE_JOINTS = (5, 12, 18, 22)          # shoulders + hips
_MAPPED_JOINTS = (18, 19, 20, 22, 23, 24, 5, 6, 7, 12, 13, 14, 27)
_CONF_QUALITY = {0: 0.0, 1: 0.4, 2: 0.8, 3: 1.0}

KINECT_VIEWS = ("color", "depth")


class KinectError(RuntimeError):
    """Sensor/tracker failure surfaced as an ordinary exception."""


@dataclass
class KinectBody:
    body_id: int
    joints: np.ndarray      # (32, 4): x_mm, y_mm, z_mm, confidence(0-3)
    joints2d: np.ndarray    # (32, 2): pixels in the native view image


def body_quality(confidences: np.ndarray) -> tuple[float, bool]:
    """K4ABT confidence levels (0..3) -> (quality 0..1, valid).
    Quality = mean over the 13 joints the H36M mapping uses; a body is invalid
    when any core joint (hips/shoulders) has NONE confidence."""
    conf = np.asarray(confidences, dtype=int)
    levels = [_CONF_QUALITY.get(int(conf[i]), 0.0) for i in _MAPPED_JOINTS]
    quality = float(np.mean(levels)) if levels else 0.0
    valid = all(int(conf[i]) > 0 for i in _CORE_JOINTS)
    return quality, bool(valid)


def transform_points_2d(points: np.ndarray, native_size: tuple[int, int],
                        frame_size: tuple[int, int], mirrored: bool) -> np.ndarray:
    """Native view pixels -> displayed frame pixels (mirror happens in native
    space first, matching apply_live_mirror flipping the raw view image)."""
    pts = np.asarray(points, dtype=float).copy()
    nw, nh = native_size
    fw, fh = frame_size
    if mirrored:
        pts[:, 0] = float(nw) - pts[:, 0]
    pts[:, 0] *= float(fw) / max(float(nw), 1.0)
    pts[:, 1] *= float(fh) / max(float(nh), 1.0)
    return pts


def bbox_from_points(points: np.ndarray, frame_size: tuple[int, int],
                     pad_frac: float = 0.08) -> tuple[float, float, float, float]:
    """Padded, frame-clamped bbox around 2D joints (for the track registry)."""
    pts = np.asarray(points, dtype=float)
    fw, fh = frame_size
    x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
    pad_x, pad_y = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
    return (max(0.0, x1 - pad_x), max(0.0, y1 - pad_y),
            min(float(fw), x2 + pad_x), min(float(fh), y2 + pad_y))


def pad_to_aspect(image: np.ndarray, aspect_w: int, aspect_h: int):
    """Pad an image with black bars to at least the given aspect ratio.
    Returns (padded, x_offset, y_offset) so joint pixels can be shifted.
    Used for the NFOV depth view (640x576) so the 16:9 stream resize doesn't
    stretch it."""
    h, w = image.shape[:2]
    target_w = int(round(h * aspect_w / aspect_h))
    if target_w <= w:
        return image, 0, 0
    left = (target_w - w) // 2
    right = target_w - w - left
    padded = cv2.copyMakeBorder(image, 0, 0, left, right,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded, left, 0
