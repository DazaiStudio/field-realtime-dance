import math
from typing import Dict, List, Tuple


class CentroidTracker:
    """Greedy nearest-centroid tracker: assigns stable ids to (x, y) points
    across frames. Used to give MediaPipe (which has no ids) per-person ids."""

    def __init__(self, max_distance: float = 120.0, evict_after: int = 15):
        self.max_distance = max_distance
        self.evict_after = evict_after
        self._next_id = 1
        self._objects: Dict[int, Tuple[float, float]] = {}
        self._missing: Dict[int, int] = {}

    def update(self, centroids: List[Tuple[float, float]]) -> List[int]:
        assigned: List[int] = [None] * len(centroids)
        used_ids = set()

        for idx, c in enumerate(centroids):
            best_id, best_d = None, self.max_distance
            for oid, oc in self._objects.items():
                if oid in used_ids:
                    continue
                d = math.dist(c, oc)
                if d < best_d:
                    best_id, best_d = oid, d
            if best_id is not None:
                assigned[idx] = best_id
                used_ids.add(best_id)
                self._objects[best_id] = c
                self._missing[best_id] = 0

        for idx, c in enumerate(centroids):
            if assigned[idx] is None:
                oid = self._next_id
                self._next_id += 1
                self._objects[oid] = c
                self._missing[oid] = 0
                assigned[idx] = oid
                used_ids.add(oid)

        for oid in list(self._objects.keys()):
            if oid not in used_ids:
                self._missing[oid] += 1
                if self._missing[oid] > self.evict_after:
                    del self._objects[oid]
                    del self._missing[oid]
        return assigned
