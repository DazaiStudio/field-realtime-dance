"""PoseEngine: orchestrates a selectable single-person pose source, One-Euro
joint smoothing, and the DanceMetricsEngine.

Public interface is unchanged from the original (process_frame / set_metrics_fps
/ draw_cached_overlay / close) so the viewer is agnostic to the backend. The
backend ("mediapipe" default, or "rtmpose3d") is chosen at construction and can
be swapped live with set_backend().
"""
from dance_metrics import DanceMetricsEngine
from one_euro import JointSmoother

VALID_BACKENDS = ("mediapipe", "rtmpose3d")


class PoseEngine:
    def __init__(self, model_path: str = "pose_landmarker_full.task",
                 backend: str = "mediapipe", smoothing_enabled: bool = True,
                 smooth_min_cutoff: float = 1.5, smooth_beta: float = 0.0008):
        self.model_path = model_path
        self.backend_name = backend if backend in VALID_BACKENDS else "mediapipe"
        self.metrics_engine = DanceMetricsEngine(fps=30)
        self.smoother = JointSmoother(min_cutoff=smooth_min_cutoff, beta=smooth_beta)
        self.smoothing_enabled = bool(smoothing_enabled)
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
        return MediaPipePoseSource(self.model_path)

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

    def set_metrics_fps(self, fps):
        self.metrics_engine.set_fps(fps)

    def draw_cached_overlay(self, frame):
        return self.source.draw_cached_overlay(frame)

    # --- per-frame ----------------------------------------------------------
    def process_frame(self, frame, timestamp_ms, draw_overlay: bool = True):
        frame, h36m = self.source.estimate(frame, timestamp_ms, draw_overlay)
        metrics = self.metrics_engine.get_empty_metrics()
        if h36m is not None:
            if self.smoothing_enabled:
                h36m = self.smoother.filter(h36m, timestamp_ms)
            metrics = self.metrics_engine.update(h36m)
        return frame, metrics

    def close(self):
        try:
            self.source.close()
        except Exception:
            pass
