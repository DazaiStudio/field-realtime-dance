from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import numpy as np


@dataclass
class PersonPose:
    track_id: int            # volatile id from the backend's own tracker
    h36m17: np.ndarray       # (17, 3) unified joints for DanceMetricsEngine
    bbox: tuple              # (x1, y1, x2, y2) image coords
    kpts_2d: np.ndarray      # (K, 2) image coords for skeleton overlay
    is_3d: bool              # whether h36m17 z is meaningful


@runtime_checkable
class PoseBackend(Protocol):
    def estimate(self, frame, timestamp_ms: float) -> "list[PersonPose]": ...
    def close(self) -> None: ...
