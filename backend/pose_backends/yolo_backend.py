import numpy as np

from pose_backend import PersonPose
from keypoint_mapping import coco17_to_h36m17


def _to_numpy(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def results_to_personposes(result) -> "list[PersonPose]":
    """Convert one Ultralytics pose result into PersonPose items.
    Returns [] if the tracker has not assigned ids yet."""
    if result.boxes is None or result.boxes.id is None:
        return []
    ids = _to_numpy(result.boxes.id).astype(int)
    xyxy = _to_numpy(result.boxes.xyxy)
    kxy = _to_numpy(result.keypoints.xy)   # (N,17,2)
    people = []
    for i in range(len(ids)):
        coco = kxy[i]
        people.append(PersonPose(
            track_id=int(ids[i]),
            h36m17=coco17_to_h36m17(coco),
            bbox=tuple(float(v) for v in xyxy[i]),
            kpts_2d=coco.astype(float),
            is_3d=False))
    return people


class YOLO26Backend:
    def __init__(self, weights="yolo26m-pose.pt", tracker="botsort.yaml", device=None):
        from ultralytics import YOLO   # lazy import so tests don't need the dep
        # NOTE: verify the exact YOLO26 pose weight name is available in the
        # installed ultralytics version at deploy time; fall back to newest *-pose.pt.
        self.model = YOLO(weights)
        self.tracker = tracker
        self.device = device   # None -> Ultralytics auto-selects CUDA/MPS/CPU

    def estimate(self, frame, timestamp_ms: float):
        results = self.model.track(
            frame, persist=True, tracker=self.tracker,
            verbose=False, device=self.device)
        if not results:
            return []
        return results_to_personposes(results[0])

    def close(self):
        pass
