"""Pluggable single-person pose sources for PoseEngine.

A pose source produces, per frame, an annotated frame + a single H36M-17
joint array (17, 3) in ~mm (or None when no person is found), ready for the
DanceMetricsEngine. Two backends:

  - MediaPipePoseSource  (default): MediaPipe Pose world landmarks. Pseudo-3D,
    cross-platform (CPU/GPU, runs on M1). The original viewer's backend.
  - RTMPose3DPoseSource: rtmlib RTMW3D-x monocular 3D (top-down, NVIDIA GPU via
    onnxruntime). Cleaner 3D; single dancer = the largest detected person.

Heavy deps (mediapipe / rtmlib / torch) are imported lazily inside each class's
__init__ so this module (and its pure helpers) import without them.
"""
import os
import cv2
import numpy as np

from keypoint_mapping import mp33_to_h36m17, coco17_to_h36m17_3d
from person_tracker import (
    MultiPersonTrackRegistry,
    UltralyticsPersonTracker,
    bbox_area,
    crop_frame,
    expand_bbox_rect,
)

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---------------------------------------------------------------------------
# MediaPipe drawing (33-landmark skeleton), kept identical to the original UI.
# ---------------------------------------------------------------------------
_MP_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (31, 27),
    (24, 26), (26, 28), (28, 30), (30, 32), (32, 28),
]
_MP_QUALITY_WEIGHTS = {
    0: 0.5,    # nose/head matters less for these metrics
    11: 1.3, 12: 1.3, 23: 1.3, 24: 1.3,  # shoulders + hips
    13: 1.1, 14: 1.1, 25: 1.1, 26: 1.1,  # elbows + knees
    15: 1.4, 16: 1.4, 27: 1.4, 28: 1.4,  # wrists + ankles
}
_MP_CORE_LANDMARKS = (11, 12, 23, 24)
_MP_END_EFFECTORS = (15, 16, 27, 28)
POSE_QUALITY_MIN = 0.75
POSE_CORE_MIN = 0.45
POSE_EFFECTOR_MIN = 0.35
POSE_MIN_EFFECTORS = 3

POSE_MODEL_URLS = {
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

# COCO-17 skeleton edges (for the RTMPose overlay).
_COCO_CONNECTIONS = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
]

# RTMPose3D calibration (see feat/multi-person Task 5): bring the dimensionless
# z into the same unit as x,y (x bbox height) then a uniform scale so the 9
# metrics land in MediaPipe's magnitude band.
RTM_POSE_SCALE = 3.0


def _register_torch_cuda_dlls() -> None:
    """Put PyTorch's bundled CUDA/cuDNN DLLs on the search path so onnxruntime's
    CUDAExecutionProvider can load (needs cudnn64_9.dll etc.). PyTorch ships them
    under torch/lib (pulled in by ultralytics). No-op off Windows or without torch."""
    if not hasattr(os, "add_dll_directory"):
        return
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            os.add_dll_directory(torch_lib)
    except Exception:
        pass


def _quiet_absl_logs() -> None:
    try:
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:
        pass


def _largest_person(kp: np.ndarray, kp2d) -> int:
    """Index of the person with the largest body bounding box (single-person =
    pick the dominant dancer). Uses pixel keypoints_2d when available."""
    src = kp2d if (kp2d is not None and kp2d.shape[0] == kp.shape[0]) else kp[:, :, :2]
    best_i, best_area = 0, -1.0
    for i in range(src.shape[0]):
        b = src[i, :17, :2]
        area = (b[:, 0].max() - b[:, 0].min()) * (b[:, 1].max() - b[:, 1].min())
        if area > best_area:
            best_area, best_i = area, i
    return best_i


def _fit_affine_xy(crop, full):
    """Per-axis (scale, offset) mapping crop-space x,y -> full-frame x,y,
    fit from the two point sets' bbox extents (robust to a few bad joints)."""
    out = []
    for ax in range(2):
        c0, c1 = float(crop[:, ax].min()), float(crop[:, ax].max())
        f0, f1 = float(full[:, ax].min()), float(full[:, ax].max())
        cs = (c1 - c0) or 1.0
        s = (f1 - f0) / cs
        out.append((s, f0 - s * c0))
    return out


def _apply_affine_xy(crop, xform):
    import numpy as _np
    pts = _np.empty((crop.shape[0], 2), dtype=float)
    for ax in range(2):
        s, o = xform[ax]
        pts[:, ax] = crop[:, ax] * s + o
    return pts


def _landmark_confidence(landmark) -> float:
    for attr in ("visibility", "presence"):
        if hasattr(landmark, attr):
            try:
                value = float(getattr(landmark, attr))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return float(np.clip(value, 0.0, 1.0))
    return 1.0


def mediapipe_pose_quality(landmarks) -> tuple[float, bool]:
    """Weighted calibration-quality gate from MediaPipe landmark confidence."""
    if landmarks is None or len(landmarks) <= max(_MP_QUALITY_WEIGHTS):
        return 0.0, False

    confidences = {i: _landmark_confidence(landmarks[i]) for i in _MP_QUALITY_WEIGHTS}
    total_weight = sum(_MP_QUALITY_WEIGHTS.values())
    quality = sum(confidences[i] * weight for i, weight in _MP_QUALITY_WEIGHTS.items()) / total_weight
    core_ok = all(confidences[i] >= POSE_CORE_MIN for i in _MP_CORE_LANDMARKS)
    end_count = sum(1 for i in _MP_END_EFFECTORS if confidences[i] >= POSE_EFFECTOR_MIN)
    valid = core_ok and end_count >= POSE_MIN_EFFECTORS and quality >= POSE_QUALITY_MIN
    return float(quality), bool(valid)


def rtmw3d_primary_h36m(keypoints, keypoints_2d=None, scale: float = RTM_POSE_SCALE):
    """Pure helper: from RTMW3D output pick the primary (largest) person and
    return (h36m17 (17,3) | None, overlay_pts2d (17,2) | None).

    Steps: take body-17 (COCO order), make z share x,y units (x bbox height),
    uniform scale, root-center on the pelvis (match MediaPipe's per-frame hip
    origin), map to standard H36M-17."""
    kp = np.asarray(keypoints, dtype=float)
    if kp.ndim != 3 or kp.shape[0] == 0:
        return None, None
    kp2 = np.asarray(keypoints_2d, dtype=float) if keypoints_2d is not None else None

    i = _largest_person(kp, kp2)
    body3d = kp[i, :17, :].copy()          # (17,3): x_px, y_px, z_norm

    bbox_h = float(body3d[:, 1].max() - body3d[:, 1].min())
    if bbox_h < 1.0:
        return None, None
    body3d[:, 2] *= bbox_h                  # z -> same unit as x,y
    body3d *= scale                         # overall magnitude ~ MediaPipe mm
    pelvis = (body3d[11] + body3d[12]) / 2.0  # COCO hips
    body3d = body3d - pelvis                # root-center (match MediaPipe)

    h36m = coco17_to_h36m17_3d(body3d)
    pts2d = kp2[i, :17, :] if (kp2 is not None and i < kp2.shape[0]) else None
    return h36m, pts2d


class MediaPipePoseSource:
    """Single-person MediaPipe Pose (world landmarks -> H36M-17). Default backend."""

    def __init__(
        self,
        model_path: str = "pose_landmarker_full.task",
        tracking_enabled: bool = False,
        tracker_model: str = "yolov8n.pt",
        tracker_yaml: str = "bytetrack.yaml",
        tracker_hold_seconds: float = 0.8,
        tracking_selection: str = "auto_largest",
    ):
        _quiet_absl_logs()
        import mediapipe as mp
        self._mp = mp
        self.model_path = model_path
        self.tracking_enabled = bool(tracking_enabled)
        self.tracker_model = tracker_model
        self.tracker_yaml = tracker_yaml
        self.tracker_hold_seconds = float(tracker_hold_seconds)
        self.tracking_selection = str(tracking_selection or "auto_largest")
        self.tracker = None
        self.track_registry = MultiPersonTrackRegistry(hold_seconds=self.tracker_hold_seconds)
        if self.tracking_enabled:
            # Load YOLO (and its first-use weight download) off the frame loop.
            self._ensure_tracker().start_preload()
        self._ensure_model()
        self.landmarker = self._create_landmarker(mp.tasks.vision.RunningMode.VIDEO)
        self.image_landmarker = None
        self.last_pose_landmarks = None
        self.last_pose_quality = 0.0
        self.last_pose_valid = False
        self.last_tracking = self._tracking_status("disabled")
        self.last_h36m_by_id = {} if self.tracking_enabled else None
        self._last_overlay_points = None
        self._last_overlay_points_by_id = {}
        self._last_stable_tracks = []
        self._last_active_track = None

    def _ensure_model(self):
        if not os.path.exists(self.model_path):
            import urllib.request
            url = POSE_MODEL_URLS.get(os.path.basename(self.model_path))
            if url is None:
                raise FileNotFoundError(
                    f"Model file not found and no download URL configured: {self.model_path}")
            print(f"Downloading {self.model_path}...")
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete.")

    def _create_landmarker(self, running_mode):
        BaseOptions = self._mp.tasks.BaseOptions
        PoseLandmarker = self._mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = self._mp.tasks.vision.PoseLandmarkerOptions
        return PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=running_mode,
            )
        )

    def _ensure_image_landmarker(self):
        if self.image_landmarker is None:
            self.image_landmarker = self._create_landmarker(self._mp.tasks.vision.RunningMode.IMAGE)
        return self.image_landmarker

    def configure_tracking(
        self,
        enabled: bool = None,
        tracker_model: str = None,
        tracker_yaml: str = None,
        hold_seconds: float = None,
        selection: str = None,
    ) -> None:
        reset_needed = False
        if enabled is not None and bool(enabled) != self.tracking_enabled:
            self.tracking_enabled = bool(enabled)
            reset_needed = True
        if tracker_model and tracker_model != self.tracker_model:
            self.tracker_model = tracker_model
            self.tracker = None
            reset_needed = True
        if tracker_yaml and tracker_yaml != self.tracker_yaml:
            self.tracker_yaml = tracker_yaml
            self.tracker = None
            reset_needed = True
        if hold_seconds is not None and float(hold_seconds) != self.tracker_hold_seconds:
            self.tracker_hold_seconds = float(hold_seconds)
            self.track_registry.hold_seconds = self.tracker_hold_seconds
            reset_needed = True
        if selection is not None:
            self.tracking_selection = str(selection or "auto_largest")
        if reset_needed:
            self.reset_tracking()
        if self.tracking_enabled:
            # Explicit user action: (re)start the model load, retrying a
            # previously latched failure.
            self._ensure_tracker().start_preload(retry_error=True)

    def reset_tracking(self) -> None:
        self.track_registry.reset()
        self._last_stable_tracks = []
        self._last_active_track = None
        self._last_overlay_points_by_id = {}
        self.last_h36m_by_id = {} if self.tracking_enabled else None
        self.last_tracking = self._tracking_status("enabled" if self.tracking_enabled else "disabled")

    def _ensure_tracker(self):
        if self.tracker is None:
            self.tracker = UltralyticsPersonTracker(
                model_name=self.tracker_model,
                tracker_yaml=self.tracker_yaml,
            )
        return self.tracker

    def _tracking_status(self, state: str, tracks=None, active=None, error: str = None) -> dict:
        tracks = tracks or []
        stable_id = int(active.stable_id) if active is not None else None
        raw_id = int(active.raw_id) if active is not None and active.raw_id is not None else None
        return {
            "enabled": bool(self.tracking_enabled),
            "state": state,
            "count": len([track for track in tracks if getattr(track, "state", "") == "tracking"]),
            "locked_id": stable_id,
            "stable_id": stable_id,
            "raw_id": raw_id,
            "active_id": stable_id if active is not None else None,
            "selection": self.tracking_selection,
            "bbox": [float(v) for v in active.bbox] if active is not None else None,
            "tracks": [self._stable_track_payload(track, active) for track in tracks],
            "error": error,
        }

    @staticmethod
    def _stable_track_payload(track, active=None) -> dict:
        return {
            "stable_id": int(track.stable_id),
            "raw_id": int(track.raw_id) if track.raw_id is not None else None,
            "state": track.state,
            "confidence": float(track.confidence),
            "bbox": [float(v) for v in track.bbox],
            "active": bool(active is not None and track.stable_id == active.stable_id),
        }

    def _select_tracking_crop(self, frame, timestamp_ms: float):
        if not self.tracking_enabled:
            self.last_tracking = self._tracking_status("disabled")
            return
        tracker = self._ensure_tracker()
        if not tracker.is_ready():
            error = tracker.load_error
            if error:
                # Latched load failure: report it, never retry per frame.
                self.last_tracking = self._tracking_status("error", error=error)
            else:
                tracker.start_preload()  # idempotent no-op while loading
                self.last_tracking = self._tracking_status("loading")
            return
        try:
            tracks = tracker.track(frame)
        except Exception as exc:
            self._last_stable_tracks = []
            self._last_active_track = None
            self.last_tracking = self._tracking_status("error", error=str(exc))
            return

        now = float(timestamp_ms) / 1000.0
        stable_tracks = self.track_registry.update(tracks, now)
        active, state = self.track_registry.choose_active(self.tracking_selection, frame.shape)
        self._last_stable_tracks = stable_tracks
        self._last_active_track = active
        if active is None:
            self.last_tracking = self._tracking_status(state, tracks=stable_tracks)
            return
        self.last_tracking = self._tracking_status(state, tracks=stable_tracks, active=active)

    def _landmark_points(self, image, landmarks, crop_rect=None):
        if crop_rect is None:
            x0, y0 = 0, 0
            h, w = image.shape[:2]
        else:
            x0, y0, x1, y1 = crop_rect
            w, h = max(1, x1 - x0), max(1, y1 - y0)
        points = {}
        for i, lm in enumerate(landmarks):
            if i < 11:
                continue
            cx, cy = int(x0 + lm.x * w), int(y0 + lm.y * h)
            points[i] = (cx, cy)
        return points

    def _draw_points(self, image, points, joint_color=(0, 255, 255), line_color=(0, 255, 0)):
        if not points:
            return
        h, w = image.shape[:2]
        for cx, cy in points.values():
            if -20 <= cx <= w + 20 and -20 <= cy <= h + 20:
                cv2.circle(image, (cx, cy), 4, joint_color, -1)
        for a, b in _MP_CONNECTIONS:
            if a in points and b in points:
                cv2.line(image, points[a], points[b], line_color, 2)

    def _draw_tracking_overlay(self, image):
        if not self.tracking_enabled:
            return
        tracks = [track for track in self._last_stable_tracks if track.state != "lost"]
        active_stable_id = self._last_active_track.stable_id if self._last_active_track is not None else None
        for track in sorted(tracks, key=lambda item: bbox_area(item.bbox)):
            x1, y1, x2, y2 = [int(round(v)) for v in track.bbox]
            is_active = active_stable_id is not None and int(track.stable_id) == int(active_stable_id)
            color = (192, 211, 52) if is_active else (150, 142, 132)
            thickness = 2 if is_active else 1
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            label = f"id {track.stable_id}"
            if track.raw_id is not None and int(track.raw_id) != int(track.stable_id):
                label += f" raw {track.raw_id}"
            if is_active or track.state != "tracking":
                label += f" {track.state}"
            cv2.putText(
                image,
                label,
                (max(0, x1), max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

    def draw_cached_overlay(self, frame):
        self._draw_tracking_overlay(frame)
        if self.tracking_enabled:
            active_id = self._last_active_track.stable_id if self._last_active_track is not None else None
            for stable_id, points in sorted(self._last_overlay_points_by_id.items()):
                if stable_id == active_id:
                    self._draw_points(frame, points, joint_color=(0, 255, 255), line_color=(0, 255, 0))
                else:
                    self._draw_points(frame, points, joint_color=(192, 211, 52), line_color=(192, 211, 52))
        elif self._last_overlay_points:
            self._draw_points(frame, self._last_overlay_points)
        return frame

    def _detect_frame_pose(self, input_frame, timestamp_ms: float, stateless: bool = False):
        rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if stateless:
            return self._ensure_image_landmarker().detect(mp_image)
        return self.landmarker.detect_for_video(mp_image, int(timestamp_ms))

    def _pose_from_result(self, frame, crop_rect, result):
        h36m = None
        overlay_points = None
        quality = 0.0
        valid = False
        landmarks = None
        if result.pose_landmarks:
            landmarks = result.pose_landmarks
            quality, valid = mediapipe_pose_quality(result.pose_landmarks[0])
            overlay_points = self._landmark_points(frame, result.pose_landmarks[0], crop_rect)
        if result.pose_world_landmarks:
            h36m = mp33_to_h36m17(result.pose_world_landmarks[0])
            if not result.pose_landmarks:
                quality = 1.0
                valid = True
        return h36m, overlay_points, quality, valid, landmarks

    def _estimate_tracked_people(self, frame, timestamp_ms: float, draw: bool = True):
        self._select_tracking_crop(frame, timestamp_ms)
        if draw:
            self._draw_tracking_overlay(frame)

        self.last_pose_landmarks = None
        self.last_pose_quality = 0.0
        self.last_pose_valid = False
        self._last_overlay_points = None

        active_id = self._last_active_track.stable_id if self._last_active_track is not None else None
        non_lost = [track for track in self._last_stable_tracks if track.state != "lost"]

        active_h36m = None
        fresh_points = {}
        h36m_by_id = {}
        for stable_track in non_lost:
            if stable_track.state != "tracking":
                # 'holding' bboxes are frozen at the last-seen position; running
                # pose there would report whoever walks through under this ID.
                continue
            rect = expand_bbox_rect(stable_track.bbox, frame.shape, padding_x=0.06, padding_y=0.08)
            if rect is None:
                continue
            input_frame = crop_frame(frame, rect)
            result = self._detect_frame_pose(input_frame, timestamp_ms, stateless=True)
            h36m, points, quality, valid, landmarks = self._pose_from_result(frame, rect, result)
            if points:
                fresh_points[stable_track.stable_id] = points
            if h36m is not None:
                h36m_by_id[stable_track.stable_id] = h36m
            if stable_track.stable_id == active_id:
                if landmarks is not None:
                    self.last_pose_landmarks = landmarks
                self.last_pose_quality = quality
                self.last_pose_valid = valid
                active_h36m = h36m

        # Keep the previous overlay for non-lost tracks that missed a
        # detection this frame so the skeleton doesn't strobe.
        merged_points = {}
        for track in non_lost:
            sid = track.stable_id
            if sid in fresh_points:
                merged_points[sid] = fresh_points[sid]
            elif sid in self._last_overlay_points_by_id:
                merged_points[sid] = self._last_overlay_points_by_id[sid]
        self._last_overlay_points_by_id = merged_points
        self._last_overlay_points = merged_points.get(active_id)
        self.last_h36m_by_id = h36m_by_id

        if draw:
            for sid, points in sorted(merged_points.items()):
                if sid == active_id:
                    self._draw_points(frame, points, joint_color=(0, 255, 255), line_color=(0, 255, 0))
                else:
                    self._draw_points(frame, points, joint_color=(192, 211, 52), line_color=(192, 211, 52))
        return frame, active_h36m

    def estimate(self, frame, timestamp_ms: float, draw: bool = True):
        if self.tracking_enabled:
            return self._estimate_tracked_people(frame, timestamp_ms, draw)

        result = self._detect_frame_pose(frame, timestamp_ms, stateless=False)
        h36m, points, quality, valid, landmarks = self._pose_from_result(frame, None, result)
        self.last_pose_quality = quality
        self.last_pose_valid = valid
        if landmarks is not None:
            # Only overwrite the cached overlay on a successful detection so a
            # single missed frame doesn't strobe the skeleton.
            self.last_pose_landmarks = landmarks
            self._last_overlay_points = points
        self._last_overlay_points_by_id = {}
        self.last_h36m_by_id = None
        if draw and self._last_overlay_points:
            self._draw_points(frame, self._last_overlay_points)
        return frame, h36m

    def close(self):
        self.landmarker.close()
        if self.image_landmarker is not None:
            self.image_landmarker.close()


class RTMPose3DPoseSource:
    """Single-person RTMPose3D (rtmlib RTMW3D-x) via onnxruntime/CUDA."""

    def __init__(self, device: str = "cuda", det_frequency: int = 10, mode: str = "balanced"):
        _register_torch_cuda_dlls()
        from rtmlib import PoseTracker, Wholebody3d
        self.tracker = PoseTracker(
            Wholebody3d, det_frequency=det_frequency, tracking=True,
            mode=mode, to_openpose=False, backend="onnxruntime", device=device,
        )
        self._last_kpts2d = None
        self._last_overlay_pts = None
        self._xform = None  # cached crop->full-frame transform for the overlay
        self.last_pose_quality = 0.0
        self.last_pose_valid = False

    def _draw(self, frame, pts2d):
        pts = {i: (int(pts2d[i][0]), int(pts2d[i][1])) for i in range(len(pts2d))}
        for i, (cx, cy) in pts.items():
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
        for a, b in _COCO_CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)

    def draw_cached_overlay(self, frame):
        if self._last_overlay_pts is not None:
            self._draw(frame, self._last_overlay_pts)
        return frame

    def estimate(self, frame, timestamp_ms: float, draw: bool = True):
        result = self.tracker(frame)
        self.last_pose_quality = 0.0
        self.last_pose_valid = False
        if result is None or len(result) < 2:
            return frame, None
        keypoints = np.asarray(result[0], dtype=float)
        kpts2d_full = np.asarray(result[3], dtype=float) if len(result) >= 4 else None

        # Metrics: tested helper (picks the largest person, builds H36M-17).
        h36m, _ = rtmw3d_primary_h36m(keypoints, kpts2d_full, RTM_POSE_SCALE)
        if h36m is not None:
            self.last_pose_quality = 1.0
            self.last_pose_valid = True

        # Overlay in full-frame pixels that tracks EVERY frame. Detection frames
        # give full-frame keypoints_2d directly and refresh the crop->full
        # transform; on tracking frames (no 2D) the fresh 3D crop x,y are mapped
        # through the cached transform, so the skeleton follows the dancer
        # instead of freezing until the next detection (~det_frequency frames).
        if keypoints.ndim == 3 and keypoints.shape[0] > 0:
            i = _largest_person(keypoints, kpts2d_full)
            crop17 = keypoints[i, :17, :2]
            overlay = None
            if kpts2d_full is not None and i < kpts2d_full.shape[0]:
                overlay = kpts2d_full[i, :17, :]
                self._xform = _fit_affine_xy(crop17, overlay)
            elif self._xform is not None:
                overlay = _apply_affine_xy(crop17, self._xform)
            if overlay is not None:
                self._last_overlay_pts = overlay
                if draw:
                    self._draw(frame, overlay)
        return frame, h36m

    def close(self):
        pass
