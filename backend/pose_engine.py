"""PoseEngine: orchestrates a selectable single-person pose source, optional
One-Euro joint smoothing, and the DanceMetricsEngine.

Public interface is unchanged from the original (process_frame / set_metrics_fps
/ draw_cached_overlay / close) so the viewer is agnostic to the backend. The
backend ("mediapipe" default, or "rtmpose3d") is chosen at construction and can
be swapped live with set_backend().
"""
import numpy as np

from dance_metrics import DanceMetricsEngine
from one_euro import JointSmoother

VALID_BACKENDS = ("mediapipe", "rtmpose3d")


class PoseEngine:
    def __init__(self, model_path: str = "pose_landmarker_full.task",
                 backend: str = "mediapipe", smoothing_enabled: bool = False,
                 smooth_min_cutoff: float = 1.5, smooth_beta: float = 0.0008,
                 tracking_enabled: bool = False, tracker_model: str = "yolov8n.pt",
                 tracker_yaml: str = "bytetrack.yaml", tracking_selection: str = "auto_largest"):
        self.model_path = model_path
        self.backend_name = backend if backend in VALID_BACKENDS else "mediapipe"
        self.metrics_engine = DanceMetricsEngine(fps=30)
        self.smoother = JointSmoother(min_cutoff=smooth_min_cutoff, beta=smooth_beta)
        self.smoothing_enabled = bool(smoothing_enabled)
        self.tracking_enabled = bool(tracking_enabled)
        self.tracker_model = tracker_model
        self.tracker_yaml = tracker_yaml
        self.tracking_selection = str(tracking_selection or "auto_largest")
        self.last_pose_valid = False
        self.last_pose_quality = 0.0
        self.last_skeleton_h36m = None
        self.last_tracking = {"enabled": False, "state": "disabled", "count": 0}
        self._metrics_person_id = None
        self.source = self._make_source(self.backend_name)

    # --- source / backend management ---------------------------------------
    def _make_source(self, backend: str):
        if backend == "rtmpose3d":
            try:
                from pose_sources import RTMPose3DPoseSource
                return RTMPose3DPoseSource()
            except Exception as exc:
                # rtmlib/onnxruntime missing or no CUDA (e.g. on a Mac) -> degrade
                # to MediaPipe instead of crashing the stream.
                print(f"[PoseEngine] RTMPose3D unavailable ({exc}); falling back to MediaPipe.")
                self.backend_name = "mediapipe"
        from pose_sources import MediaPipePoseSource
        return MediaPipePoseSource(
            self.model_path,
            tracking_enabled=self.tracking_enabled,
            tracker_model=self.tracker_model,
            tracker_yaml=self.tracker_yaml,
            tracking_selection=self.tracking_selection,
        )

    def set_backend(self, backend: str) -> None:
        """Swap the pose backend live. Rebuilds the source and clears all
        history so the previous backend's frames don't pollute the metrics."""
        if backend not in VALID_BACKENDS or backend == self.backend_name:
            return
        try:
            self.source.close()
        except Exception:
            pass
        self.backend_name = backend
        self.source = self._make_source(backend)
        self.smoother.reset()
        fps = self.metrics_engine.fps
        self.metrics_engine = DanceMetricsEngine(fps=fps)

    def configure_smoothing(self, enabled: bool = None, min_cutoff: float = None,
                            beta: float = None) -> None:
        if enabled is not None:
            self.smoothing_enabled = bool(enabled)
        self.smoother.configure(min_cutoff=min_cutoff, beta=beta)

    def configure_tracking(self, enabled: bool = None, tracker_model: str = None,
                           tracker_yaml: str = None, selection: str = None) -> None:
        if enabled is not None:
            self.tracking_enabled = bool(enabled)
        if tracker_model:
            self.tracker_model = tracker_model
        if tracker_yaml:
            self.tracker_yaml = tracker_yaml
        if selection is not None:
            self.tracking_selection = str(selection or "auto_largest")
        if hasattr(self.source, "configure_tracking"):
            self.source.configure_tracking(
                enabled=self.tracking_enabled,
                tracker_model=self.tracker_model,
                tracker_yaml=self.tracker_yaml,
                selection=self.tracking_selection,
            )

    def set_metrics_fps(self, fps):
        self.metrics_engine.set_fps(fps)

    def reset_metrics(self):
        self.metrics_engine.reset()
        self.smoother.reset()

    def reset_tracking(self):
        if hasattr(self.source, "reset_tracking"):
            self.source.reset_tracking()

    def draw_cached_overlay(self, frame):
        return self.source.draw_cached_overlay(frame)

    # --- per-frame ----------------------------------------------------------
    def process_frame(self, frame, timestamp_ms, draw_overlay: bool = True):
        frame, h36m = self.source.estimate(frame, timestamp_ms, draw_overlay)
        metrics = self.metrics_engine.get_empty_metrics()
        pose_complete = self._pose_is_complete(h36m)
        self.last_pose_quality = 0.0
        self.last_pose_valid = False
        self.last_skeleton_h36m = None
        self.last_tracking = getattr(self.source, "last_tracking", {"enabled": False, "state": "disabled", "count": 0})
        if pose_complete:
            self.last_pose_quality = self._clamp_quality(getattr(self.source, "last_pose_quality", 1.0))
            self.last_pose_valid = bool(getattr(self.source, "last_pose_valid", True))
            active_id = self.last_tracking.get("active_id") if isinstance(self.last_tracking, dict) else None
            if active_id != self._metrics_person_id:
                # A different dancer now feeds the metrics: restart history so
                # velocity/jerk are never computed across two bodies.
                self.metrics_engine.reset()
                self.smoother.reset()
                self._metrics_person_id = active_id
            if self.smoothing_enabled:
                h36m = self.smoother.filter(h36m, timestamp_ms)
            self.last_skeleton_h36m = np.asarray(h36m, dtype=float).copy()
            metrics = self.metrics_engine.update(h36m)
        return frame, metrics

    @staticmethod
    def _pose_is_complete(h36m) -> bool:
        if h36m is None:
            return False
        arr = np.asarray(h36m, dtype=float)
        return arr.shape == (17, 3) and bool(np.isfinite(arr).all())

    @staticmethod
    def _clamp_quality(value) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(numeric):
            return 0.0
        return float(np.clip(numeric, 0.0, 1.0))

    def close(self):
        try:
            self.source.close()
        except Exception:
            pass
