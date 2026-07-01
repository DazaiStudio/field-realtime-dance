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

    def __init__(self, model_path: str = "pose_landmarker_full.task"):
        import mediapipe as mp
        self._mp = mp
        self.model_path = model_path
        self._ensure_model()
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        self.landmarker = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=VisionRunningMode.VIDEO,
            )
        )
        self.last_pose_landmarks = None
        self.last_pose_quality = 0.0
        self.last_pose_valid = False

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

    def _draw_landmarks(self, image, landmarks):
        h, w, _ = image.shape
        points = {}
        for i, lm in enumerate(landmarks):
            if i < 11:
                continue
            cx, cy = int(lm.x * w), int(lm.y * h)
            points[i] = (cx, cy)
            cv2.circle(image, (cx, cy), 4, (0, 255, 255), -1)
        for a, b in _MP_CONNECTIONS:
            if a in points and b in points:
                cv2.line(image, points[a], points[b], (0, 255, 0), 2)

    def draw_cached_overlay(self, frame):
        if self.last_pose_landmarks:
            for lms in self.last_pose_landmarks:
                self._draw_landmarks(frame, lms)
        return frame

    def estimate(self, frame, timestamp_ms: float, draw: bool = True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, int(timestamp_ms))
        h36m = None
        self.last_pose_quality = 0.0
        self.last_pose_valid = False
        if result.pose_landmarks:
            self.last_pose_landmarks = result.pose_landmarks
            self.last_pose_quality, self.last_pose_valid = mediapipe_pose_quality(result.pose_landmarks[0])
            if draw:
                self.draw_cached_overlay(frame)
        if result.pose_world_landmarks:
            h36m = mp33_to_h36m17(result.pose_world_landmarks[0])
            if not result.pose_landmarks:
                self.last_pose_quality = 1.0
                self.last_pose_valid = True
        return frame, h36m

    def close(self):
        self.landmarker.close()


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
