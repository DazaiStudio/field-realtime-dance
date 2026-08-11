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

VALID_BACKENDS = ("mediapipe", "rtmpose3d", "azure_kinect")


class PoseEngine:
    def __init__(self, model_path: str = "pose_landmarker_full.task",
                 backend: str = "mediapipe", smoothing_enabled: bool = False,
                 smooth_min_cutoff: float = 1.5, smooth_beta: float = 0.0008,
                 tracking_enabled: bool = False, tracker_model: str = "yolov8n.pt",
                 tracker_yaml: str = "bytetrack.yaml", tracking_selection: str = "auto_largest",
                 group_extent_enabled: bool = False, group_max_people: int = 4,
                 group_smooth_cutoff: float = 0.0):
        self.model_path = model_path
        self.backend_name = backend if backend in VALID_BACKENDS else "mediapipe"
        self.metrics_engine = DanceMetricsEngine(fps=30)
        self.smoother = JointSmoother(min_cutoff=smooth_min_cutoff, beta=smooth_beta)
        self.smoothing_enabled = bool(smoothing_enabled)
        self.tracking_enabled = bool(tracking_enabled)
        self.tracker_model = tracker_model
        self.tracker_yaml = tracker_yaml
        self.tracking_selection = str(tracking_selection or "auto_largest")
        self.group_extent_enabled = bool(group_extent_enabled)
        self.group_max_people = int(group_max_people)
        self.group_smooth_cutoff = float(group_smooth_cutoff)
        self.last_group_extent = None
        self.last_group_box_norm = None
        self.group_extent_is_metric = False
        self.last_pose_valid = False
        self.last_pose_quality = 0.0
        self.last_skeleton_h36m = None
        self.last_tracking = {"enabled": False, "state": "disabled", "count": 0}
        self._metrics_person_id = None
        # Per-dancer pipelines (stable id -> engine/smoother): each tracked
        # person gets an independent metrics stream for /field/<id>/<metric>.
        self._metric_engines = {}
        self._person_smoothers = {}
        self.last_metrics_by_id = {}
        self.last_skeletons_by_id = {}
        self.source = self._make_source(self.backend_name)

    # --- source / backend management ---------------------------------------
    def _make_source(self, backend: str):
        if backend == "azure_kinect":
            try:
                from pose_backends.azure_kinect import AzureKinectPoseSource, get_runtime
                return AzureKinectPoseSource(
                    get_runtime(),
                    tracking_enabled=self.tracking_enabled,
                    tracking_selection=self.tracking_selection,
                    group_extent_enabled=self.group_extent_enabled,
                    group_max_people=self.group_max_people,
                    group_smooth_cutoff=self.group_smooth_cutoff,
                )
            except Exception as exc:
                # pykinect/SDK missing -> degrade to MediaPipe, same pattern as
                # the rtmpose3d fallback below.
                print(f"[PoseEngine] Azure Kinect unavailable ({exc}); falling back to MediaPipe.")
                self.backend_name = "mediapipe"
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
            group_extent_enabled=self.group_extent_enabled,
            group_max_people=self.group_max_people,
            group_smooth_cutoff=self.group_smooth_cutoff,
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

    def configure_group_extent(self, enabled: bool = None, max_people: int = None,
                               smooth_cutoff: float = None) -> None:
        """Both backends draw the group box; only Kinect can put real metres
        behind it (MediaPipe has no depth, so it reports screen-space only).
        Sources without the hook silently ignore this."""
        if enabled is not None:
            self.group_extent_enabled = bool(enabled)
        if max_people is not None:
            self.group_max_people = max(1, int(max_people))
        if smooth_cutoff is not None:
            self.group_smooth_cutoff = max(0.0, float(smooth_cutoff))
        if hasattr(self.source, "configure_group_extent"):
            self.source.configure_group_extent(
                enabled=self.group_extent_enabled,
                max_people=self.group_max_people,
                smooth_cutoff=self.group_smooth_cutoff,
            )
        else:
            self.last_group_extent = None

    def group_extent_supported(self) -> bool:
        return hasattr(self.source, "configure_group_extent")

    def set_metrics_fps(self, fps):
        self.metrics_engine.set_fps(fps)
        for engine in self._metric_engines.values():
            engine.set_fps(fps)

    def reset_metrics(self):
        self.metrics_engine.reset()
        self.smoother.reset()
        self._metric_engines.clear()
        self._person_smoothers.clear()
        self.last_metrics_by_id = {}
        self.last_skeletons_by_id = {}

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
        # Group extent rides beside the per-dancer path, not inside it: it is
        # measured from raw bodies and stays valid whether or not stable id and
        # its metrics pipeline are running.
        self.last_group_extent = getattr(self.source, "last_group_extent", None)
        self.last_group_box_norm = getattr(self.source, "last_group_box_norm", None)
        self.group_extent_is_metric = bool(
            getattr(self.source, "group_extent_is_metric", False)
        )

        poses_by_id = getattr(self.source, "last_h36m_by_id", None)
        if self.tracking_enabled and isinstance(poses_by_id, dict):
            return frame, self._process_tracked(poses_by_id, timestamp_ms)

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
            # Single-person output rides the same per-id OSC schema as id 1.
            self.last_metrics_by_id = {1: metrics}
            self.last_skeletons_by_id = {1: self.last_skeleton_h36m}
        else:
            self.last_metrics_by_id = {}
            self.last_skeletons_by_id = {}
        return frame, metrics

    def _process_tracked(self, poses_by_id, timestamp_ms):
        """Per-dancer pipelines: each stable id feeds its own metrics engine,
        so no stream ever mixes two bodies' motion."""
        active_id = self.last_tracking.get("active_id") if isinstance(self.last_tracking, dict) else None
        keep_ids = self._registry_person_ids()
        if keep_ids is not None:
            for pid in [pid for pid in self._metric_engines if pid not in keep_ids]:
                self._metric_engines.pop(pid, None)
                self._person_smoothers.pop(pid, None)

        metrics_by_id = {}
        skeletons_by_id = {}
        for pid in sorted(poses_by_id):
            pose = poses_by_id[pid]
            if not self._pose_is_complete(pose):
                continue
            engine = self._metric_engines.get(pid)
            if engine is None:
                engine = DanceMetricsEngine(fps=30)
                engine.set_fps(self.metrics_engine.fps)
                self._metric_engines[pid] = engine
            arr = np.asarray(pose, dtype=float)
            if self.smoothing_enabled:
                smoother = self._person_smoothers.get(pid)
                if smoother is None:
                    smoother = JointSmoother(
                        min_cutoff=self.smoother.min_cutoff, beta=self.smoother.beta
                    )
                    self._person_smoothers[pid] = smoother
                arr = smoother.filter(arr, timestamp_ms)
            skeletons_by_id[pid] = np.asarray(arr, dtype=float).copy()
            metrics_by_id[pid] = engine.update(arr)

        self.last_metrics_by_id = metrics_by_id
        self.last_skeletons_by_id = skeletons_by_id
        self.last_pose_quality = self._clamp_quality(getattr(self.source, "last_pose_quality", 0.0))
        self.last_pose_valid = bool(getattr(self.source, "last_pose_valid", False))
        display = metrics_by_id.get(active_id) if active_id is not None else None
        self.last_skeleton_h36m = skeletons_by_id.get(active_id) if active_id is not None else None
        return display if display is not None else self.metrics_engine.get_empty_metrics()

    def _registry_person_ids(self):
        tracks = self.last_tracking.get("tracks") if isinstance(self.last_tracking, dict) else None
        if not isinstance(tracks, list):
            return None
        ids = set()
        for track in tracks:
            if not isinstance(track, dict) or track.get("state") == "lost":
                continue
            stable_id = track.get("stable_id")
            if stable_id is not None:
                ids.add(int(stable_id))
        return ids

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
