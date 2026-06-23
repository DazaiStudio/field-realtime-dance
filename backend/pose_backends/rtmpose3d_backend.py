"""RTMPose3D backend: monocular 3D pose via rtmlib's RTMW3D-x.

Architecture mirrors yolo_backend.py:
  - Pure helper `personposes_from_rtmw3d(...)` is unit-testable without rtmlib.
  - `RTMPose3DBackend` owns the heavy model; it does a lazy `from rtmlib import`
    inside __init__ so the module can be imported without rtmlib installed.

PoseTracker return signature (rtmlib 0.0.15, tracking=True, Wholebody3d):
    On detection frames (every det_frequency frames):
        keypoints, scores, keypoints_simcc, keypoints_2d = tracker(frame)
        # 4-tuple

    On tracking frames (between detections):
        keypoints, scores = tracker(frame)
        # 2-tuple — keypoints_2d is NOT returned; the class caches the last value

    - keypoints:      (N, 133, 3) float32 — 3D coords in model-input pixel space
                      The first 17 joints are COCO-17 body order WITH z.
                      x, y: pixel coords in the resized model-input crop (~288×384).
                      z:    root-relative normalised depth, scaled by bbox-height
                            in model-input space, range approximately [-2.17, 2.17].
    - scores:         (N, 133)    float32 — confidence per keypoint
    - keypoints_simcc:(N, 133, 3) float32 — SimCC intermediate (not used here)
    - keypoints_2d:   (N, 133, 2) float32 — pixel coords in the original video frame
                      (back-projected from model-input space to full resolution).

  Track IDs: `tracker.track_ids_last_frame` after each call holds integer IDs.

Coordinate system (empirically measured, Task 5 calibration):
    3D x range (body 0-16): ~[100, 190]  — model-input crop pixel columns
    3D y range (body 0-16): ~[ 90, 280]  — model-input crop pixel rows
    3D z range (body 0-16): ~[-0.7, -0.3]  — normalised depth (this clip)
    2D x range:             ~[830, 1060] — full video frame pixels (1920-wide)
    2D y range:             ~[335, 880]  — full video frame pixels (1080-tall)

Units transform (Task 5 calibration):
    Problem: x,y are in model-input-pixel space (~0–288/384) while z is a
    dimensionless ratio normalised by bbox_height in MODEL-INPUT SPACE. This
    means the z axis is ~200× smaller in scale than x,y, skewing all 3D limb
    directions and corrupting angular metrics (energy, torque, jerk).

    Two-step fix applied inside personposes_from_rtmw3d():
      Step 1 — Make z share the same unit as x,y:
          z_consistent = z_normalised × bbox_height_3d
          where bbox_height_3d = y_max − y_min of the 17 body keypoints
          in model-input space (~190 px for a full-body clip).
          Result: x, y, z are all in model-input-pixel units.

      Step 2 — Scale uniformly so positional metrics are in a comparable
          numeric range to MediaPipe:
          coords_out = coords_model_px × POSE_SCALE
          POSE_SCALE = 3.0  (calibrated against MediaPipe on a 300-frame
          dance clip — see Rationale below)

    Rationale for POSE_SCALE = 3.0:
        Expansion (convex hull volume of 17 joints) scales as SCALE³.
        At scale=3.0, RTMPose3D expansion mean≈0.97 ≈ MediaPipe 0.94.
        Sway (linear in scale) lands at 0.13 vs MediaPipe 0.34 — same
        order of magnitude (within 3×).
        Energy is angular/scale-invariant; its residual gap (363 vs 53)
        reflects intrinsic jitter in the RTMPose3D tracker vs MediaPipe's
        temporally-smoothed world landmarks — not fixable with scale.
        Height has a sign/origin mismatch: RTMPose3D uses absolute model-
        input pixel y (top→bottom), while MediaPipe world landmarks are
        root-relative (pelvis=origin). The engine's -com[1]/1000 formula
        yields negative height for RTMPose3D. Scale cannot fix this.

    Before/After calibration table (300-frame clip, single dancer):
        Metric          MediaPipe (mean/med/p95)  RTMPose3D-before   RTMPose3D-after (mean/med/p95)
        energy          53.0 / 19.9 / 228.6       2922 (1 frame)     363.2 / 199.1 / 1269.3
        expansion        0.94/  0.90/   1.34        0.0003 (bad)       0.97/  1.22/   1.57
        height           0.10/  0.12/   0.17       -0.19 (bad)        -0.29/ -0.21/  -0.11
        sway             0.34/  0.21/   0.83        0.005 (bad)        0.13/  0.06/   0.38
        torque          80.8 / 60.9 / 195.2        (no data)         206.4 /135.7 / 668.4
        sync_velocity    0.59/  0.59/   0.97        0.65 (1 frame)     0.36/  0.21/   0.98
        curvature        0.07/  0.03/   0.27        (no data)          0.55/  0.09/   2.50
        (Note: "before" had only 1 frame of data due to 4-tuple vs 2-tuple bug)

bbox and kpts_2d:
    The visual overlay (bbox, kpts_2d) must use PIXEL coords from
    keypoints_2d (the 2D back-projection), NOT the 3D model-input coords.
    The helper takes keypoints_2d separately for this purpose.
    On tracking frames (2-tuple), the class caches the last keypoints_2d.
"""

import numpy as np

from pose_backend import PersonPose
from keypoint_mapping import coco17_to_h36m17_3d

# Uniform scale applied to all 3 axes AFTER the z-normalisation step.
# See module docstring for full rationale and calibration table.
# TASK 5 result: calibrated to 3.0 (down from placeholder 1000.0).
POSE_SCALE: float = 3.0


def personposes_from_rtmw3d(
    keypoints: np.ndarray,
    scores: np.ndarray,
    track_ids: "list[int]",
    keypoints_2d: "np.ndarray | None" = None,
    scale: float = POSE_SCALE,
) -> "list[PersonPose]":
    """Convert RTMW3D PoseTracker outputs into a list of PersonPose.

    Parameters
    ----------
    keypoints:
        (N, 133, 3) float array — 3D coordinates from rtmlib in model-input
        pixel space.  x, y in crop-pixel units; z normalised by bbox height.
        The first 17 joints (indices 0-16) are COCO-17 body order with z.
    scores:
        (N, 133) float array — confidence scores (not used to filter here;
        caller may add a threshold later).
    track_ids:
        list of N integers.  Pass tracker.track_ids_last_frame after calling
        the PoseTracker.  A value of -1 means no valid id; that person is
        skipped (mirrors YOLO backend's "no ids yet" behaviour).
    keypoints_2d:
        (N, 133, 2) float array — pixel coords in the original video frame
        (back-projected from model-input space).  Used ONLY for bbox and
        kpts_2d (visual overlay), NOT for the 3D metric computation.
        If None (e.g., cached from the previous detection frame), 3D x,y
        are used as a fallback for bbox/kpts_2d.
    scale:
        Uniform multiplier applied to all three axes after z-normalisation.
        Defaults to POSE_SCALE (module-level constant, 5.5).

    Returns
    -------
    list[PersonPose] — one entry per person with a valid track id.

    Units transform applied here (see module docstring):
        z_consistent = z_raw × bbox_height_3d    (makes z same unit as x,y)
        h36m17 = coco17_to_h36m17_3d(body3d × scale)
    """
    if len(track_ids) == 0:
        return []

    people: list[PersonPose] = []
    kpts = np.asarray(keypoints, dtype=float)   # (N, 133, 3)

    # keypoints_2d for pixel-space bbox/overlay
    kpts2d = None
    if keypoints_2d is not None:
        kpts2d = np.asarray(keypoints_2d, dtype=float)  # (N, 133, 2)

    for i, tid in enumerate(track_ids):
        if tid < 0:
            # Tracker has not assigned a valid id (person too small / occluded).
            continue

        # --- Extract COCO-17 body keypoints (first 17) with z -----------------
        body_3d = kpts[i, :17, :].copy()    # (17, 3): x_px, y_px, z_norm

        # Step 1: Convert z from normalised depth → model-input-pixel units.
        #   z_norm is scaled by the body's bbox height in model-input space,
        #   so multiplying by that height restores z to the same px unit as x,y.
        y_vals = body_3d[:, 1]
        bbox_h_3d = float(y_vals.max() - y_vals.min())
        if bbox_h_3d < 1.0:
            # Degenerate frame (all joints collapsed to a point) — skip.
            continue
        body_3d[:, 2] *= bbox_h_3d   # z now in model-input-pixel units

        # Step 2: Uniform scale → mm-equivalent range (matches MediaPipe ×1000).
        body_3d *= scale

        h36m = coco17_to_h36m17_3d(body_3d)   # (17, 3)

        # --- bbox and kpts_2d from pixel-space 2D keypoints ------------------
        # Prefer back-projected pixel coords for visual overlay accuracy.
        if kpts2d is not None and i < kpts2d.shape[0]:
            body_2d = kpts2d[i, :17, :]    # (17, 2) — full-frame pixels
        else:
            # Fallback: use 3D x,y (model-input px, less accurate for overlay)
            body_2d = body_3d[:, :2] / scale   # undo scale to get crop px

        x1 = float(body_2d[:, 0].min())
        y1 = float(body_2d[:, 1].min())
        x2 = float(body_2d[:, 0].max())
        y2 = float(body_2d[:, 1].max())

        people.append(PersonPose(
            track_id=int(tid),
            h36m17=h36m,
            bbox=(x1, y1, x2, y2),
            kpts_2d=body_2d.copy(),   # (17, 2) for skeleton overlay
            is_3d=True,
        ))

    return people


class RTMPose3DBackend:
    """Pose backend using rtmlib's RTMW3D-x via PoseTracker.

    The heavy rtmlib import is deferred to __init__ so that importing this
    module does not require rtmlib, ONNX weights, or a GPU.

    rtmlib return-value behaviour:
        Detection frames (every det_frequency):  4-tuple (kpts, scores, simcc, kpts2d)
        Tracking frames (between detections):    2-tuple (kpts, scores)
    This class caches keypoints_2d from each detection frame and reuses it
    on the following tracking frames so every frame produces a PersonPose.

    Parameters
    ----------
    device:
        'cuda' (default) or 'cpu'.
    det_frequency:
        Run person detector every N frames; pose model runs every frame.
        Default 10 matches the spike configuration.
    mode:
        rtmlib model variant; 'balanced' is the only Wholebody3d mode.
    """

    def __init__(
        self,
        device: str = "cuda",
        det_frequency: int = 10,
        mode: str = "balanced",
    ):
        # Lazy import keeps the module importable without rtmlib / GPU.
        from rtmlib import PoseTracker, Wholebody3d  # noqa: PLC0415

        self._tracker = PoseTracker(
            Wholebody3d,
            det_frequency=det_frequency,
            tracking=True,          # enables track_ids_last_frame
            mode=mode,
            to_openpose=False,      # mmpose/COCO keypoint order (133 kpts)
            backend="onnxruntime",
            device=device,
        )
        # Cache keypoints_2d from detection frames to reuse on tracking frames.
        self._last_kpts2d: "np.ndarray | None" = None

    def estimate(self, frame, timestamp_ms: float) -> "list[PersonPose]":
        """Run pose estimation on one frame and return PersonPose list.

        Handles both 4-tuple (detection frame) and 2-tuple (tracking frame)
        return values from rtmlib's PoseTracker.
        """
        result = self._tracker(frame)

        if result is None or len(result) < 2:
            return []

        keypoints = result[0]
        scores = result[1]

        # On detection frames rtmlib returns a 4-tuple; update the 2D cache.
        if len(result) >= 4:
            self._last_kpts2d = result[3]   # (N, 133, 2) full-frame pixels

        # Track IDs are stored on the tracker after each call.
        track_ids: list[int] = list(self._tracker.track_ids_last_frame)

        return personposes_from_rtmw3d(
            keypoints,
            scores,
            track_ids,
            keypoints_2d=self._last_kpts2d,
        )

    def close(self) -> None:
        """Release resources.  rtmlib has no explicit teardown; this is a no-op."""
        pass
