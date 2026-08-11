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
# How many of those four must not be NONE. Requiring all four threw the whole
# frame away whenever one shoulder or hip was occluded -- turning sideways,
# another dancer passing in front, floor work -- while K4ABT was still tracking
# the body perfectly well. Three tolerates one-sided occlusion; it still
# rejects "both hips gone", where the H36M pelvis every metric is centred on
# would be pure guesswork.
_MIN_CORE_JOINTS = 3
_MAPPED_JOINTS = (18, 19, 20, 22, 23, 24, 5, 6, 7, 12, 13, 14, 27)
_CONF_QUALITY = {0: 0.0, 1: 0.4, 2: 0.8, 3: 1.0}

KINECT_VIEWS = ("color", "depth")

# Depth mode drives the working range, which is stage-dependent: NFOV unbinned
# reaches ~3.9 m, binning it quadruples the IR signal per pixel for ~5.5 m at
# the same 30 fps (half the depth resolution), and the WFOV modes trade reach
# for a 120 deg spread. Selected with FIELD_KINECT_DEPTH_MODE.
# WFOV unbinned is deliberately absent: it is the only mode capped at 15 fps,
# and wfov_binned beats it on reach as well, so it would be a footgun next to
# the hardcoded 30 fps camera setting.
_DEPTH_MODES = {
    "nfov": "K4A_DEPTH_MODE_NFOV_UNBINNED",             # ~0.5-3.9 m, 75 deg, 30 fps
    "nfov_binned": "K4A_DEPTH_MODE_NFOV_2X2BINNED",     # ~0.5-5.5 m, 75 deg, 30 fps
    "wfov_binned": "K4A_DEPTH_MODE_WFOV_2X2BINNED",     # ~0.25-2.9 m, 120 deg, 30 fps
}
DEFAULT_DEPTH_MODE = "nfov"


def resolve_depth_mode(name=None) -> str:
    """Depth mode key -> pykinect constant name.

    Falls back to the default rather than raising: a typo in an environment
    variable must not take the camera down mid-rehearsal.
    """
    key = str(name or "").strip().lower()
    return _DEPTH_MODES.get(key, _DEPTH_MODES[DEFAULT_DEPTH_MODE])


class KinectError(RuntimeError):
    """Sensor/tracker failure surfaced as an ordinary exception."""


@dataclass
class KinectBody:
    body_id: int
    joints: np.ndarray      # (32, 4): x_mm, y_mm, z_mm, confidence(0-3)
    joints2d: np.ndarray    # (32, 2): pixels in the native view image


def body_quality(confidences: np.ndarray) -> tuple[float, bool]:
    """K4ABT confidence levels (0..3) -> (quality 0..1, valid).

    Quality = mean over the 13 joints the H36M mapping uses. A body is valid
    when at least _MIN_CORE_JOINTS of the four shoulders/hips beat NONE, so at
    most one may be missing -- which also means neither girdle is ever lost
    whole, and the H36M pelvis and thorax every metric is measured against are
    never pure guesswork.
    """
    conf = np.asarray(confidences, dtype=int)
    levels = [_CONF_QUALITY.get(int(conf[i]), 0.0) for i in _MAPPED_JOINTS]
    quality = float(np.mean(levels)) if levels else 0.0
    present = sum(1 for i in _CORE_JOINTS if int(conf[i]) > 0)
    valid = present >= _MIN_CORE_JOINTS
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


def sanitize_joints2d(points: np.ndarray, confidences: np.ndarray) -> np.ndarray:
    """Mark unusable 2D joints as NaN: k4a returns (0, 0) when the 3d->2d
    projection fails (joint outside the view frustum), and NONE-confidence
    joints are position guesses for fully occluded limbs. Left unfiltered they
    draw skeleton lines into the frame corner and blow the bbox up to the
    whole frame (seen on hardware 2026-07-30 with legs occluded by a desk)."""
    pts = np.asarray(points, dtype=float)[:, :2].copy()
    conf = np.asarray(confidences, dtype=float)
    bad = ~np.isfinite(pts).all(axis=1)
    bad |= (np.abs(pts[:, 0]) < 1e-6) & (np.abs(pts[:, 1]) < 1e-6)
    bad |= conf <= 0
    pts[bad] = np.nan
    return pts


def bbox_from_points(points: np.ndarray, frame_size: tuple[int, int],
                     pad_frac: float = 0.08):
    """Padded, frame-clamped bbox around the finite 2D joints (for the track
    registry). Returns None when no joint projects into the view."""
    pts = np.asarray(points, dtype=float)
    valid = np.isfinite(pts).all(axis=1)
    if not valid.any():
        return None
    pts = pts[valid]
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
from group_extent import GroupExtentTracker, hip_floor_position, union_bbox  # noqa: E402
from group_overlay import draw_group_box  # noqa: E402
from group_smoothing import GroupSmoother, smoothed_group_outputs  # noqa: E402
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
                 tracking_selection: str = "auto_largest",
                 group_extent_enabled: bool = False,
                 group_max_people: int = 4,
                 group_smooth_cutoff: float = 0.0, **_ignored):
        self.runtime = runtime
        self.tracking_enabled = bool(tracking_enabled)
        self.tracking_selection = str(tracking_selection or "auto_largest")
        self.track_registry = MultiPersonTrackRegistry(hold_seconds=float(tracker_hold_seconds))
        self.group_extent_enabled = bool(group_extent_enabled)
        self.group_tracker = GroupExtentTracker(max_people=int(group_max_people))
        self.group_smoother = GroupSmoother(group_smooth_cutoff)
        self.last_group_extent = None
        self._last_group_box = None
        self.last_group_box_norm = None
        # K4ABT gives absolute mm, so the floor-plane metre values are real.
        self.group_extent_is_metric = True
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

    def configure_group_extent(self, enabled=None, max_people=None,
                               smooth_cutoff=None, **_ignored):
        """Group floor bbox is independent of stable id: it reads the raw
        bodies, so it can be on with tracking off (and vice versa)."""
        if enabled is not None and bool(enabled) != self.group_extent_enabled:
            self.group_extent_enabled = bool(enabled)
            self.group_tracker.reset()
            self.group_smoother.reset()
            self.last_group_extent = None
            self._last_group_box = None
        if max_people is not None:
            self.group_tracker.max_people = max(1, int(max_people))
        if smooth_cutoff is not None:
            self.group_smoother.configure(smooth_cutoff)

    def reset_tracking(self):
        self.track_registry.reset()
        self._last_stable_tracks = []
        self._last_active_track = None
        self._overlay_points_by_id = {}
        self.last_h36m_by_id = {} if self.tracking_enabled else None
        self.last_tracking = self._tracking_status("enabled" if self.tracking_enabled else "disabled")
        self.group_tracker.reset()
        self.group_smoother.reset()
        self.last_group_extent = None
        self._last_group_box = None
        self.last_group_box_norm = None

    def estimate(self, frame, timestamp_ms: float, draw: bool = True):
        self.last_pose_quality = 0.0
        self.last_pose_valid = False

        error = getattr(self.runtime, "last_error", None)
        if error:
            self._last_stable_tracks = []
            self._last_active_track = None
            self.last_h36m_by_id = {} if self.tracking_enabled else None
            self.last_group_extent = None
            self._last_group_box = None
            self.last_group_box_norm = None
            self.group_smoother.reset()
            self.last_tracking = self._tracking_status("error", error=str(error))
            return frame, None

        frame_h, frame_w = frame.shape[:2]
        native_size = tuple(getattr(self.runtime, "native_view_size", (frame_w, frame_h)))
        mirrored = bool(getattr(self.runtime, "mirrored", False))

        bodies = getattr(self.runtime, "last_bodies", []) or []
        # Group inputs are gathered in the same pass as everything else so the
        # drawn box and the metre values always describe the same set of bodies.
        group_positions, group_boxes = [], []

        # Per-body: H36M skeleton (mirror-aware) + 2D points + registry track.
        raw_tracks, data_by_raw = [], {}
        for body in bodies:
            quality, valid = body_quality(body.joints[:, 3])
            h36m = k4abt32_to_h36m17(body.joints)
            if mirrored:
                h36m = mirror_h36m17(h36m)
            pts2d = transform_points_2d(body.joints2d, native_size,
                                        (frame_w, frame_h), mirrored)
            bbox = bbox_from_points(pts2d, (frame_w, frame_h))
            if bbox is None:
                # No joint projects into the view: nothing to draw or box, and
                # the registry cannot geometry-match it. Skip this body.
                continue
            conf = float(np.clip(quality, 0.0, 1.0))
            raw_tracks.append(PersonTrack(track_id=int(body.body_id), bbox=bbox,
                                          confidence=conf))
            data_by_raw[int(body.body_id)] = (h36m, pts2d, quality, valid)

            if self.group_extent_enabled:
                # Floor position comes off the RAW joints, never the H36M array
                # -- that one is root-centered per dancer, so a bbox over it
                # would always measure zero.
                floor = hip_floor_position(body, mirrored=mirrored)
                if floor is not None:
                    group_positions.append(floor)
                    group_boxes.append(bbox)

        if self.group_extent_enabled:
            group_now = float(timestamp_ms) / 1000.0
            extent = self.group_tracker.update(group_positions, group_now)
            (self.last_group_extent,
             self._last_group_box,
             self.last_group_box_norm) = smoothed_group_outputs(
                self.group_smoother, extent, union_bbox(group_boxes),
                frame.shape, group_now,
            )
        else:
            self.last_group_extent = None
            self._last_group_box = None
            self.last_group_box_norm = None
            self.group_smoother.reset()

        now = float(timestamp_ms) / 1000.0
        stable_tracks = self.track_registry.update(raw_tracks, now)
        # The target dropdown is a multi-person control; single mode always
        # follows the largest dancer.
        selection = self.tracking_selection if self.tracking_enabled else "auto_largest"
        active, state = self.track_registry.choose_active(selection, frame.shape)
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

        active_id = int(active.stable_id) if active is not None else None
        if not self.tracking_enabled:
            # K4ABT always reports every body it sees, but single mode shows the
            # subject only, matching the MediaPipe single-person view.
            overlay_by_id = {pid: pts for pid, pts in overlay_by_id.items() if pid == active_id}
        self._overlay_points_by_id = overlay_by_id

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

    def _refresh_overlay_points(self, frame):
        """Re-project the drawn skeletons from the runtime's newest bodies.

        K4ABT already ran for this frame -- the frame source calls it on every
        read -- but estimate() is gated to analysis_fps, so without this the
        skeleton would be drawn at a position up to one analysis interval old.

        Only tracks that are already on screen are refreshed: a newly visible
        body has no stable id until the next analysis, and a body that stopped
        being reported keeps its last position rather than blinking out.
        """
        if not self._overlay_points_by_id:
            return
        bodies = {int(body.body_id): body
                  for body in (getattr(self.runtime, "last_bodies", None) or [])}
        if not bodies:
            return
        frame_h, frame_w = frame.shape[:2]
        native_size = tuple(getattr(self.runtime, "native_view_size", (frame_w, frame_h)))
        mirrored = bool(getattr(self.runtime, "mirrored", False))
        raw_by_stable = {int(track.stable_id): int(track.raw_id)
                         for track in self._last_stable_tracks if track.raw_id is not None}
        for stable_id in self._overlay_points_by_id:
            raw_id = raw_by_stable.get(int(stable_id))
            body = bodies.get(raw_id) if raw_id is not None else None
            if body is None:
                continue
            pts2d = transform_points_2d(body.joints2d, native_size,
                                        (frame_w, frame_h), mirrored)
            self._overlay_points_by_id[stable_id] = self._points_dict(pts2d)

    def draw_cached_overlay(self, frame):
        self._refresh_overlay_points(frame)
        active_id = self._last_active_track.stable_id if self._last_active_track is not None else None
        # Boxes and id labels are multi-person affordances: single mode keeps the
        # view clean, like MediaPipe's _draw_tracking_overlay early-return.
        tracks = [t for t in self._last_stable_tracks if t.state != "lost"] if self.tracking_enabled else []
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
        if self.group_extent_enabled and self._last_group_box is not None:
            draw_group_box(frame, self._last_group_box, self.last_group_extent)
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
        # NaN source joints (occluded/unprojectable) drop out here; derived
        # joints computed from them are NaN too and drop with them.
        return {i: (int(round(h36m_pts[i, 0])), int(round(h36m_pts[i, 1])))
                for i in _H36M_DRAWN if np.isfinite(h36m_pts[i]).all()}

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


# --- Hardware runtime --------------------------------------------------------

_SDK_DIRS = (
    r"C:\Program Files\Azure Kinect SDK v1.4.2",
    r"C:\Program Files\Azure Kinect SDK v1.4.1",
)
_BT_SDK_DIR = r"C:\Program Files\Azure Kinect Body Tracking SDK"


def azure_kinect_available() -> bool:
    """Cheap probe for the backend dropdown: Windows + SDKs + python binding.
    Does NOT import pykinect (import loads DLLs; keep the probe instant)."""
    if os.name != "nt":
        return False
    import importlib.util
    if importlib.util.find_spec("pykinect_azure") is None:
        return False
    return any(os.path.isdir(d) for d in _SDK_DIRS) and os.path.isdir(_BT_SDK_DIR)


def _guarded(what: str, fn, *args, **kwargs):
    """Run a pykinect call, converting BOTH exceptions and sys.exit into
    KinectError. pykinect's VERIFY() calls sys.exit(1) on sensor errors —
    uncaught, an unplugged cable would kill the whole viewer process."""
    try:
        return fn(*args, **kwargs)
    except SystemExit as exc:
        raise KinectError(f"Kinect {what} failed (SDK aborted)") from exc
    except KinectError:
        raise
    except Exception as exc:
        raise KinectError(f"Kinect {what} failed: {exc}") from exc


class KinectRuntime:
    """Owns the k4a device + body tracker. read() = capture + track + cache.

    One module-level instance (get_runtime()); the device is opened by the
    frame source (acquire) per stream session and closed on release, with the
    same owner-token semantics as osc_viewer's camera globals."""

    POP_TIMEOUT_MS = 350
    # Never wait forever for a capture: read() holds _lock for the whole
    # capture+track cycle, and release() (called from the stream loop's
    # finally, on the event loop) needs the same lock. A stalled device with
    # the pykinect default of K4A_WAIT_INFINITE therefore freezes the entire
    # server, not just the stream. Timing out drops to the existing
    # missed_frames -> reopen recovery path instead.
    CAPTURE_TIMEOUT_MS = 1000

    def __init__(self):
        self._lock = threading.Lock()
        self._opened = False
        self._owner = None
        self._device = None
        self._tracker = None
        self.view = "color"
        self.mirrored = False
        self.native_view_size = (1280, 720)
        self.depth_mode_name = None
        self.last_bodies: list[KinectBody] = []
        self.last_error = None
        self._calibration_type_color = None
        self._calibration_type_depth = None

    # --- lifecycle -----------------------------------------------------------
    def acquire(self, owner, view: str = "color", mirrored: bool = False):
        """(Re)open the device for a stream session (takes over like
        open_camera does)."""
        with self._lock:
            self._close_device()
            self.view = view if view in KINECT_VIEWS else "color"
            self.mirrored = bool(mirrored)
            self._owner = owner
            self._open_device()
            self._opened = True
            self.last_error = None
            return self

    def release(self, owner=None, force: bool = False):
        with self._lock:
            if not self._opened:
                return
            if not force and owner is not None and self._owner != owner:
                return
            self._close_device()
            self._opened = False
            self._owner = None

    def reopen(self):
        """Close + reopen after read failures (mirrors reopen_live_camera)."""
        with self._lock:
            owner, view, mirrored = self._owner, self.view, self.mirrored
            self._close_device()
            time.sleep(0.35)
            self.view = view
            self.mirrored = mirrored
            self._owner = owner
            self._open_device()
            self._opened = True
            self.last_error = None

    def _open_device(self):
        import pykinect_azure as pykinect
        from pykinect_azure.k4abt import _k4abtTypes as _bt

        _guarded("library init", pykinect.initialize_libraries, track_body=True)
        # initialize_libraries resets processing_mode, so set these AFTER it.
        _bt.k4abt_tracker_default_configuration.processing_mode = \
            _bt.K4ABT_TRACKER_PROCESSING_MODE_GPU_DIRECTML
        _bt.k4abt_tracker_default_configuration.gpu_device_id = \
            int(os.getenv("FIELD_KINECT_GPU", "1"))

        config = pykinect.default_configuration
        config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        depth_mode = resolve_depth_mode(os.getenv("FIELD_KINECT_DEPTH_MODE"))
        config.depth_mode = getattr(pykinect, depth_mode)
        # Kept so the floor map can draw this mode's far range limit.
        self.depth_mode_name = depth_mode
        config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
        config.synchronized_images_only = True

        self._device = _guarded("device open", pykinect.start_device, config=config)
        model = _bt.K4ABT_LITE_MODEL if os.getenv("FIELD_KINECT_MODEL", "full") == "lite" \
            else _bt.K4ABT_DEFAULT_MODEL
        self._tracker = _guarded("tracker create", pykinect.start_body_tracker, model)
        self._calibration_type_color = pykinect.K4A_CALIBRATION_TYPE_COLOR
        self._calibration_type_depth = pykinect.K4A_CALIBRATION_TYPE_DEPTH

    def _close_device(self):
        device, self._device = self._device, None
        self._tracker = None
        if device is not None:
            try:
                device.close()
            except (Exception, SystemExit):
                pass
        self.last_bodies = []

    # --- frame source protocol ----------------------------------------------
    def read(self):
        """Capture one frame, run body tracking, cache bodies, return the view
        image (BGR). On failure returns (False, None) and sets last_error."""
        with self._lock:
            if not self._opened or self._device is None:
                self.last_error = "Kinect not open"
                return False, None
            try:
                capture = _guarded("capture", self._device.update,
                                   timeout_in_ms=self.CAPTURE_TIMEOUT_MS)
                body_frame = _guarded("body tracking", self._tracker.update,
                                      timeout_in_ms=self.POP_TIMEOUT_MS)
                image = self._render_view(capture, body_frame)
                if image is None:
                    return False, None
                self.last_error = None
                return True, image
            except KinectError as exc:
                self.last_error = str(exc)
                self.last_bodies = []
                return False, None

    def describe(self) -> str:
        return "Azure Kinect"

    # --- internals -----------------------------------------------------------
    def _render_view(self, capture, body_frame):
        if self.view == "depth":
            ret, image = _guarded("depth image", capture.get_colored_depth_image)
            dest_camera = self._calibration_type_depth
        else:
            ret, image = _guarded("color image", capture.get_color_image)
            dest_camera = self._calibration_type_color
        if not ret or image is None:
            return None
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        x_off = y_off = 0
        if self.view == "depth":
            image, x_off, y_off = pad_to_aspect(image, 16, 9)
        self.native_view_size = (image.shape[1], image.shape[0])
        self._extract_bodies(body_frame, dest_camera, x_off, y_off)
        return image

    def _extract_bodies(self, body_frame, dest_camera, x_off, y_off):
        bodies = []
        n = _guarded("body count", body_frame.get_num_bodies)
        for i in range(int(n)):
            body_id = int(_guarded("body id", body_frame.get_body_id, i))
            raw = np.asarray(
                _guarded("body joints", lambda idx=i: body_frame.get_body(idx).numpy()),
                dtype=float)
            joints = raw[:, [0, 1, 2, 7]]          # x, y, z, confidence
            raw2d = np.asarray(
                _guarded("body 2d", lambda idx=i: body_frame.get_body2d(idx, dest_camera).numpy()),
                dtype=float)
            joints2d = sanitize_joints2d(raw2d, joints[:, 3])
            joints2d += np.array([x_off, y_off], dtype=float)
            bodies.append(KinectBody(body_id=body_id, joints=joints, joints2d=joints2d))
        self.last_bodies = bodies


_runtime = None
_runtime_lock = threading.Lock()


def get_runtime() -> KinectRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = KinectRuntime()
        return _runtime
