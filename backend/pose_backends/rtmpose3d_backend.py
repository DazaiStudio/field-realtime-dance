"""RTMPose3D backend: monocular 3D pose via rtmlib's RTMW3D-x.

Architecture mirrors yolo_backend.py:
  - Pure helper `personposes_from_rtmw3d(...)` is unit-testable without rtmlib.
  - `RTMPose3DBackend` owns the heavy model; it does a lazy `from rtmlib import`
    inside __init__ so the module can be imported without rtmlib installed.

PoseTracker return signature (rtmlib 0.0.15, tracking=True, RTMPose3d model):
    keypoints, scores, keypoints_simcc, keypoints_2d = tracker(frame)

    - keypoints:      (N, 133, 3) float32 -- 3D normalised coords
                      first 17 joints are COCO-17 body order WITH z
                      z is root-relative normalised depth, range ~[-2.17, 2.17]
    - scores:         (N, 133)    float32 -- confidence per keypoint
    - keypoints_simcc:(N, 133, 3) float32 -- SimCC intermediate (not used here)
    - keypoints_2d:   (N, 133, 2) float32 -- pixel-space 2D coordinates

  Track IDs are NOT returned in the tuple. After the call,
  `tracker.track_ids_last_frame` holds the integer IDs in the same order as
  the returned `keypoints` rows.  A value of -1 means the person was too small
  to be assigned a valid id (mirrors YOLO backend's "no ids yet" skip logic).

POSE_SCALE constant
-------------------
The DanceMetricsEngine's energy/symmetry constants were originally tuned for
MediaPipe's coordinate range (roughly millimetres, i.e. ×1000 of normalised).
RTMPose3D z is normalised to ~[-2.17, 2.17].  Multiplying by 1000 brings it
into a similar numeric range without changing the metric calculations.
Task 5 will calibrate this empirically against MediaPipe output — change only
this constant.
"""

import numpy as np

from pose_backend import PersonPose
from keypoint_mapping import coco17_to_h36m17_3d

# Scale applied to the normalised 3D z coordinate before passing it to the
# DanceMetricsEngine.  Start at 1000.0 to match MediaPipe's ~mm range.
# TASK 5: calibrate empirically by comparing energy magnitudes with MediaPipe.
POSE_SCALE: float = 1000.0


def personposes_from_rtmw3d(
    keypoints: np.ndarray,
    scores: np.ndarray,
    track_ids: "list[int]",
    scale: float = POSE_SCALE,
) -> "list[PersonPose]":
    """Convert RTMW3D PoseTracker outputs into a list of PersonPose.

    Parameters
    ----------
    keypoints:
        (N, 133, 3) float array — 3D normalised coordinates from rtmlib.
        The first 17 joints (indices 0-16) are COCO-17 body order with z.
    scores:
        (N, 133) float array — confidence scores (not used to filter here;
        caller may add a threshold later).
    track_ids:
        list of N integers.  Pass tracker.track_ids_last_frame after calling
        the PoseTracker.  A value of -1 means no valid id; that person is
        skipped (mirrors YOLO backend's "no ids yet" behaviour).
    scale:
        Multiplier applied to z before building h36m17.  Defaults to
        POSE_SCALE (module-level constant).

    Returns
    -------
    list[PersonPose] — one entry per person with a valid track id.
    """
    if len(track_ids) == 0:
        return []

    people: list[PersonPose] = []
    kpts = np.asarray(keypoints, dtype=float)  # (N, 133, 3)

    for i, tid in enumerate(track_ids):
        if tid < 0:
            # Tracker has not assigned a valid id (person too small / occluded).
            continue

        # --- Extract COCO-17 body keypoints (first 17) with z -----------------
        body_kpts = kpts[i, :17, :]          # (17, 3): x, y, z_normalised

        # Scale z before mapping so h36m17 has the engine-expected magnitude.
        body_kpts_scaled = body_kpts.copy()
        body_kpts_scaled[:, 2] *= scale

        h36m = coco17_to_h36m17_3d(body_kpts_scaled)  # (17, 3)

        # --- Derive bbox from 2D body keypoint extent -------------------------
        body_xy = body_kpts[:, :2]           # (17, 2) pixel x,y (un-scaled)
        x1 = float(body_xy[:, 0].min())
        y1 = float(body_xy[:, 1].min())
        x2 = float(body_xy[:, 0].max())
        y2 = float(body_xy[:, 1].max())

        people.append(PersonPose(
            track_id=int(tid),
            h36m17=h36m,
            bbox=(x1, y1, x2, y2),
            kpts_2d=body_xy.copy(),           # (17, 2) for skeleton overlay
            is_3d=True,
        ))

    return people


class RTMPose3DBackend:
    """Pose backend using rtmlib's RTMW3D-x via PoseTracker.

    The heavy rtmlib import is deferred to __init__ so that importing this
    module does not require rtmlib, ONNX weights, or a GPU.

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

    def estimate(self, frame, timestamp_ms: float) -> "list[PersonPose]":
        """Run pose estimation on one frame and return PersonPose list."""
        # PoseTracker(RTMPose3d) returns a 4-tuple when tracking=True.
        result = self._tracker(frame)

        if result is None or len(result) < 4:
            return []

        keypoints, scores, _keypoints_simcc, _keypoints_2d = result

        # Track IDs are stored on the tracker after each call.
        track_ids: list[int] = list(self._tracker.track_ids_last_frame)

        return personposes_from_rtmw3d(keypoints, scores, track_ids)

    def close(self) -> None:
        """Release resources.  rtmlib has no explicit teardown; this is a no-op."""
        pass
