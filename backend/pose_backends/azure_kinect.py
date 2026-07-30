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


# --- Pose source -------------------------------------------------------------

import sys as _sys  # noqa: E402

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in _sys.path:
    _sys.path.insert(0, _BACKEND_DIR)

from keypoint_mapping import (  # noqa: E402
    _K4_L_SH, _K4_L_EL, _K4_L_WR, _K4_R_SH, _K4_R_EL, _K4_R_WR,
    _K4_L_HIP, _K4_L_KNEE, _K4_L_ANK, _K4_R_HIP, _K4_R_KNEE, _K4_R_ANK,
    _K4_NOSE, k4abt32_to_h36m17, mirror_h36m17,
)
from person_tracker import MultiPersonTrackRegistry, PersonTrack, bbox_area  # noqa: E402

# H36M-17 skeleton edges for the overlay.
_H36M_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),
]
# Overlay draws the real joints + pelvis/thorax/head (derived spine/neck too,
# via the connection midpoints they sit on) — indices actually rendered:
_H36M_DRAWN = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)


class AzureKinectPoseSource:
    """PoseSource protocol backed by K4ABT bodies cached on the KinectRuntime.

    The runtime's read() (frame source) refreshes last_bodies once per frame;
    estimate() only consumes that cache, so pose data and the displayed frame
    always come from the same capture."""

    def __init__(self, runtime, tracking_enabled: bool = False,
                 tracker_hold_seconds: float = 0.8,
                 tracking_selection: str = "auto_largest", **_ignored):
        self.runtime = runtime
        self.tracking_enabled = bool(tracking_enabled)
        self.tracking_selection = str(tracking_selection or "auto_largest")
        self.track_registry = MultiPersonTrackRegistry(hold_seconds=float(tracker_hold_seconds))
        self.last_pose_quality = 0.0
        self.last_pose_valid = False
        self.last_h36m_by_id = {} if self.tracking_enabled else None
        self.last_tracking = self._tracking_status("enabled" if self.tracking_enabled else "disabled")
        self._last_stable_tracks = []
        self._last_active_track = None
        self._overlay_points_by_id = {}

    # --- PoseSource protocol -------------------------------------------------
    def configure_tracking(self, enabled=None, selection=None, **_ignored):
        if enabled is not None and bool(enabled) != self.tracking_enabled:
            self.tracking_enabled = bool(enabled)
            self.reset_tracking()
        if selection is not None:
            self.tracking_selection = str(selection or "auto_largest")

    def reset_tracking(self):
        self.track_registry.reset()
        self._last_stable_tracks = []
        self._last_active_track = None
        self._overlay_points_by_id = {}
        self.last_h36m_by_id = {} if self.tracking_enabled else None
        self.last_tracking = self._tracking_status("enabled" if self.tracking_enabled else "disabled")

    def estimate(self, frame, timestamp_ms: float, draw: bool = True):
        self.last_pose_quality = 0.0
        self.last_pose_valid = False

        error = getattr(self.runtime, "last_error", None)
        if error:
            self._last_stable_tracks = []
            self._last_active_track = None
            self.last_h36m_by_id = {} if self.tracking_enabled else None
            self.last_tracking = self._tracking_status("error", error=str(error))
            return frame, None

        frame_h, frame_w = frame.shape[:2]
        native_size = tuple(getattr(self.runtime, "native_view_size", (frame_w, frame_h)))
        mirrored = bool(getattr(self.runtime, "mirrored", False))

        # Per-body: H36M skeleton (mirror-aware) + 2D points + registry track.
        raw_tracks, data_by_raw = [], {}
        for body in getattr(self.runtime, "last_bodies", []) or []:
            quality, valid = body_quality(body.joints[:, 3])
            h36m = k4abt32_to_h36m17(body.joints)
            if mirrored:
                h36m = mirror_h36m17(h36m)
            pts2d = transform_points_2d(body.joints2d, native_size,
                                        (frame_w, frame_h), mirrored)
            bbox = bbox_from_points(pts2d, (frame_w, frame_h))
            conf = float(np.clip(quality, 0.0, 1.0))
            raw_tracks.append(PersonTrack(track_id=int(body.body_id), bbox=bbox,
                                          confidence=conf))
            data_by_raw[int(body.body_id)] = (h36m, pts2d, quality, valid)

        now = float(timestamp_ms) / 1000.0
        stable_tracks = self.track_registry.update(raw_tracks, now)
        active, state = self.track_registry.choose_active(self.tracking_selection, frame.shape)
        self._last_stable_tracks = stable_tracks
        self._last_active_track = active

        h36m_by_id, overlay_by_id = {}, {}
        for track in stable_tracks:
            if track.state != "tracking" or track.raw_id is None:
                continue
            data = data_by_raw.get(int(track.raw_id))
            if data is None:
                continue
            h36m, pts2d, quality, valid = data
            if not valid:
                continue
            h36m_by_id[int(track.stable_id)] = h36m
            overlay_by_id[int(track.stable_id)] = self._points_dict(pts2d)
        self._overlay_points_by_id = overlay_by_id

        active_id = int(active.stable_id) if active is not None else None
        active_h36m = h36m_by_id.get(active_id) if active_id is not None else None
        if active is not None and active.raw_id is not None and int(active.raw_id) in data_by_raw:
            _, _, quality, valid = data_by_raw[int(active.raw_id)]
            self.last_pose_quality = quality
            self.last_pose_valid = bool(valid and active_h36m is not None)

        self.last_tracking = self._tracking_status(state, tracks=stable_tracks, active=active)
        self.last_h36m_by_id = h36m_by_id if self.tracking_enabled else None

        if draw:
            self.draw_cached_overlay(frame)
        return frame, active_h36m

    def draw_cached_overlay(self, frame):
        active_id = self._last_active_track.stable_id if self._last_active_track is not None else None
        tracks = [t for t in self._last_stable_tracks if t.state != "lost"]
        for track in sorted(tracks, key=lambda item: bbox_area(item.bbox)):
            x1, y1, x2, y2 = [int(round(v)) for v in track.bbox]
            is_active = active_id is not None and int(track.stable_id) == int(active_id)
            color = (192, 211, 52) if is_active else (150, 142, 132)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if is_active else 1)
            label = f"id {track.stable_id}"
            if track.raw_id is not None and int(track.raw_id) != int(track.stable_id):
                label += f" raw {track.raw_id}"
            if is_active or track.state != "tracking":
                label += f" {track.state}"
            cv2.putText(frame, label, (max(0, x1), max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        for stable_id, points in sorted(self._overlay_points_by_id.items()):
            if stable_id == active_id:
                self._draw_points(frame, points, (0, 255, 255), (0, 255, 0))
            else:
                self._draw_points(frame, points, (192, 211, 52), (192, 211, 52))
        return frame

    def close(self):
        # The runtime/device lifecycle belongs to the frame source; nothing to
        # release here beyond caches.
        self._overlay_points_by_id = {}
        self._last_stable_tracks = []
        self._last_active_track = None

    # --- internals -----------------------------------------------------------
    @staticmethod
    def _points_dict(pts2d: np.ndarray) -> dict:
        """H36M-17 overlay points from K4ABT 2D joints (same joints as the 3D
        mapping so the drawn skeleton matches the data)."""
        h36m_pts = np.zeros((17, 2))
        h36m_pts[0] = (pts2d[_K4_L_HIP] + pts2d[_K4_R_HIP]) / 2.0
        h36m_pts[1], h36m_pts[2], h36m_pts[3] = pts2d[_K4_R_HIP], pts2d[_K4_R_KNEE], pts2d[_K4_R_ANK]
        h36m_pts[4], h36m_pts[5], h36m_pts[6] = pts2d[_K4_L_HIP], pts2d[_K4_L_KNEE], pts2d[_K4_L_ANK]
        sh_mid = (pts2d[_K4_L_SH] + pts2d[_K4_R_SH]) / 2.0
        h36m_pts[8], h36m_pts[10] = sh_mid, pts2d[_K4_NOSE]
        h36m_pts[7] = (h36m_pts[0] + sh_mid) / 2.0
        h36m_pts[9] = (sh_mid + pts2d[_K4_NOSE]) / 2.0
        h36m_pts[11], h36m_pts[12], h36m_pts[13] = pts2d[_K4_L_SH], pts2d[_K4_L_EL], pts2d[_K4_L_WR]
        h36m_pts[14], h36m_pts[15], h36m_pts[16] = pts2d[_K4_R_SH], pts2d[_K4_R_EL], pts2d[_K4_R_WR]
        return {i: (int(round(h36m_pts[i, 0])), int(round(h36m_pts[i, 1])))
                for i in _H36M_DRAWN}

    @staticmethod
    def _draw_points(frame, points, joint_color, line_color):
        h, w = frame.shape[:2]
        for cx, cy in points.values():
            if -20 <= cx <= w + 20 and -20 <= cy <= h + 20:
                cv2.circle(frame, (cx, cy), 4, joint_color, -1)
        for a, b in _H36M_CONNECTIONS:
            if a in points and b in points:
                cv2.line(frame, points[a], points[b], line_color, 2)

    def _tracking_status(self, state, tracks=None, active=None, error=None):
        tracks = tracks or []
        stable_id = int(active.stable_id) if active is not None else None
        raw_id = int(active.raw_id) if active is not None and active.raw_id is not None else None
        return {
            "enabled": bool(self.tracking_enabled),
            "state": state,
            "count": len([t for t in tracks if getattr(t, "state", "") == "tracking"]),
            "locked_id": stable_id,
            "stable_id": stable_id,
            "raw_id": raw_id,
            "active_id": stable_id if active is not None else None,
            "selection": self.tracking_selection,
            "bbox": [float(v) for v in active.bbox] if active is not None else None,
            "tracks": [
                {
                    "stable_id": int(t.stable_id),
                    "raw_id": int(t.raw_id) if t.raw_id is not None else None,
                    "state": t.state,
                    "confidence": float(t.confidence),
                    "bbox": [float(v) for v in t.bbox],
                    "active": bool(active is not None and t.stable_id == active.stable_id),
                }
                for t in tracks
            ],
            "error": error,
        }
