# Multi-Person Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the realtime dance-analysis viewer to track up to 4 dancers at once, each with isolated metrics and stable per-slot OSC output, with a swappable pose backend (YOLO26 default, MediaPipe optional).

**Architecture:** A `PoseBackend` interface hides detect+track+keypoint-mapping and returns `PersonPose` items in unified H36M-17 form. A `SlotBinder` maps volatile tracker ids onto 4 fixed slots (auto + manual). Per-slot `DanceMetricsEngine` and per-slot `OSCSender` produce `/field/{slot}/...`. The viewer renders all skeletons + 4 metric panels + manual-reassign controls.

**Tech Stack:** Python 3.10, FastAPI, OpenCV, NumPy, SciPy, MediaPipe, Ultralytics (new), python-osc. Tests use stdlib **unittest** (not pytest).

**Test commands (this repo):**
- All backend tests: `python -m unittest discover -s backend/tests -t backend`
- Single: `python -m unittest tests.test_<name> -v` (run with CWD = `backend/`)
- Test files add `backend/` to `sys.path` via `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`.

**Reference layouts:**
- **H36M-17** (consumed by `DanceMetricsEngine`, see `backend/pose_engine.py:79-96`): `0 pelvis, 1 r_hip, 2 r_knee, 3 r_ankle, 4 l_hip, 5 l_knee, 6 l_ankle, 7 spine, 8 neck, 9 head, 10 l_shoulder, 11 l_elbow, 12 l_wrist, 13 r_shoulder, 14 r_elbow, 15 r_wrist, 16 neck`.
- **COCO-17** (YOLO pose output): `0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear, 5 l_shoulder, 6 r_shoulder, 7 l_elbow, 8 r_elbow, 9 l_wrist, 10 r_wrist, 11 l_hip, 12 r_hip, 13 l_knee, 14 r_knee, 15 l_ankle, 16 r_ankle`.

---

## Task 1: PoseBackend interface + PersonPose

**Files:**
- Create: `backend/pose_backend.py`
- Test: `backend/tests/test_pose_backend.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_pose_backend.py
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_backend import PersonPose, PoseBackend


class _Dummy:
    def estimate(self, frame, timestamp_ms): return []
    def close(self): pass


class TestPoseBackend(unittest.TestCase):
    def test_personpose_holds_fields(self):
        p = PersonPose(track_id=3, h36m17=np.zeros((17, 3)),
                       bbox=(0, 0, 10, 20), kpts_2d=np.zeros((17, 2)), is_3d=False)
        self.assertEqual(p.track_id, 3)
        self.assertEqual(p.h36m17.shape, (17, 3))
        self.assertFalse(p.is_3d)

    def test_dummy_satisfies_protocol(self):
        self.assertIsInstance(_Dummy(), PoseBackend)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run (CWD `backend/`): `python -m unittest tests.test_pose_backend -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pose_backend'`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/pose_backend.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_pose_backend -v` → Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/pose_backend.py backend/tests/test_pose_backend.py
git commit -m "feat: add PoseBackend interface and PersonPose"
```

---

## Task 2: COCO-17 → H36M-17 mapping

**Files:**
- Create: `backend/keypoint_mapping.py`
- Test: `backend/tests/test_keypoint_mapping.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_keypoint_mapping.py
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from keypoint_mapping import coco17_to_h36m17


def _coco():
    # distinct values per joint so derived joints are checkable
    c = np.zeros((17, 2))
    for i in range(17):
        c[i] = (i * 10, i * 10 + 1)
    return c


class TestCocoToH36m(unittest.TestCase):
    def test_shape_and_zero_z(self):
        out = coco17_to_h36m17(_coco())
        self.assertEqual(out.shape, (17, 3))
        self.assertTrue(np.allclose(out[:, 2], 0.0))

    def test_derived_joints(self):
        c = _coco()
        out = coco17_to_h36m17(c)
        l_hip, r_hip = c[11], c[12]
        l_sh, r_sh = c[5], c[6]
        np.testing.assert_allclose(out[0][:2], (l_hip + r_hip) / 2)   # pelvis
        np.testing.assert_allclose(out[8][:2], (l_sh + r_sh) / 2)     # neck
        np.testing.assert_allclose(out[9][:2], c[0])                  # head=nose
        np.testing.assert_allclose(out[7][:2], (out[0][:2] + out[8][:2]) / 2)  # spine

    def test_direct_limb_joints(self):
        c = _coco()
        out = coco17_to_h36m17(c)
        np.testing.assert_allclose(out[1][:2], c[12])   # r_hip
        np.testing.assert_allclose(out[3][:2], c[16])   # r_ankle
        np.testing.assert_allclose(out[12][:2], c[9])   # l_wrist
        np.testing.assert_allclose(out[15][:2], c[10])  # r_wrist


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_keypoint_mapping -v` → Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/keypoint_mapping.py
import numpy as np

# COCO-17 indices
_NOSE = 0
_L_SH, _R_SH = 5, 6
_L_EL, _R_EL = 7, 8
_L_WR, _R_WR = 9, 10
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_ANK, _R_ANK = 15, 16


def coco17_to_h36m17(coco: np.ndarray) -> np.ndarray:
    """Map COCO-17 keypoints (image coords, 2D) to the H36M-17 layout
    used by DanceMetricsEngine. z is set to 0 (2D backend)."""
    def p(i):
        return np.array([coco[i][0], coco[i][1], 0.0])

    l_hip, r_hip = p(_L_HIP), p(_R_HIP)
    pelvis = (l_hip + r_hip) / 2.0
    l_sh, r_sh = p(_L_SH), p(_R_SH)
    neck = (l_sh + r_sh) / 2.0
    spine = (pelvis + neck) / 2.0
    head = p(_NOSE)

    j = np.zeros((17, 3))
    j[0] = pelvis
    j[1] = r_hip;  j[2] = p(_R_KNEE);  j[3] = p(_R_ANK)
    j[4] = l_hip;  j[5] = p(_L_KNEE);  j[6] = p(_L_ANK)
    j[7] = spine;  j[8] = neck;        j[9] = head
    j[10] = l_sh;  j[11] = p(_L_EL);   j[12] = p(_L_WR)
    j[13] = r_sh;  j[14] = p(_R_EL);   j[15] = p(_R_WR)
    j[16] = neck
    return j
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_keypoint_mapping -v` → Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/keypoint_mapping.py backend/tests/test_keypoint_mapping.py
git commit -m "feat: add COCO-17 to H36M-17 keypoint mapping"
```

---

## Task 3: 2D adaptation in DanceMetricsEngine

The metrics engine consumes H36M-17. For a 2D backend z=0, so:
- `energy / sync / curvature / torque / jerk / height` already work (planar angles, z-component of cross product, image-y height).
- **`sway` already degenerates correctly**: `sqrt((com_x-bos_x)^2 + (com_z-bos_z)^2)` with z=0 == `|com_x-bos_x|`. **No change needed.**
- **`expansion` must change**: a 3D ConvexHull of coplanar (z=0) points is degenerate → ~0. Use a 2D hull whose `.volume` is the enclosed **area** (SciPy: in 2D, `ConvexHull.volume` = area, `.area` = perimeter).

**Files:**
- Modify: `backend/dance_metrics.py` (`__init__` ~line 6-19, `_calculate_expansion` ~line 154-163)
- Test: `backend/tests/test_metrics_2d.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_metrics_2d.py
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dance_metrics import DanceMetricsEngine


def _square_positions():
    # 17 joints, all z=0, spread over an XY square so a 2D hull has area
    pos = np.zeros((17, 3))
    pos[:, 0] = np.linspace(0, 100, 17)
    pos[:, 1] = np.linspace(0, 100, 17)
    pos[5, :2] = (0, 100)      # push corners out so the hull is 2D, not a line
    pos[6, :2] = (100, 0)
    return pos


class TestMetrics2D(unittest.TestCase):
    def test_expansion_2d_is_positive(self):
        eng = DanceMetricsEngine(fps=30, is_3d=False)
        area = eng._calculate_expansion(_square_positions())
        self.assertGreater(area, 0.0)

    def test_expansion_defaults_to_3d(self):
        eng = DanceMetricsEngine(fps=30)   # is_3d defaults True
        self.assertTrue(eng.is_3d)

    def test_sway_is_horizontal_when_z_zero(self):
        eng = DanceMetricsEngine(fps=30, is_3d=False)
        pos = np.zeros((17, 3))
        # CoM offset purely in X from base-of-support (ankles at idx 3,6)
        pos[:, 0] = 5.0
        _h, sway = eng._calculate_stability(pos)
        self.assertAlmostEqual(sway, abs(5.0) / 1000.0, places=6)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_metrics_2d -v`
Expected: FAIL — `__init__() got an unexpected keyword argument 'is_3d'`.

- [ ] **Step 3: Write minimal implementation**

In `backend/dance_metrics.py`, add `is_3d` to `__init__`:

```python
    def __init__(self, fps=30, is_3d=True):
        self.fps = fps
        self.dt = 1.0 / fps
        self.is_3d = is_3d
        # ... existing history buffers unchanged ...
```

Replace `_calculate_expansion`:

```python
    def _calculate_expansion(self, positions):
        """4. Body Geometry - Expansion (3D volume, or 2D area for 2D backends)."""
        try:
            pts = positions if self.is_3d else positions[:, :2]
            noise = np.random.normal(0, 1e-4, pts.shape)
            hull = ConvexHull(pts + noise)
            if self.is_3d:
                return hull.volume / 100000000.0   # mm^3 -> readable
            return hull.volume / 10000.0           # 2D: .volume == area (px^2)
        except Exception:
            return 0.0
```

(Leave `_calculate_stability` unchanged — sway is correct for z=0.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_metrics_2d -v` → Expected: PASS (3 tests).
Also run the full suite to confirm no regression: `python -m unittest discover -s backend/tests -t backend`.

- [ ] **Step 5: Commit**

```bash
git add backend/dance_metrics.py backend/tests/test_metrics_2d.py
git commit -m "feat: 2D expansion (area) path in DanceMetricsEngine via is_3d"
```

---

## Task 4: SlotBinder (4 fixed slots, auto + manual)

**Files:**
- Create: `backend/slot_binder.py`
- Test: `backend/tests/test_slot_binder.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_slot_binder.py
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slot_binder import SlotBinder


class TestSlotBinder(unittest.TestCase):
    def test_auto_assigns_lowest_free_slot(self):
        b = SlotBinder(num_slots=4, evict_after=2)
        mapping = b.update([101, 102])
        self.assertEqual(mapping, {101: 1, 102: 2})
        self.assertEqual(b.active_slots(), [1, 2])

    def test_existing_track_keeps_its_slot(self):
        b = SlotBinder(num_slots=4, evict_after=2)
        b.update([101, 102])
        mapping = b.update([102, 101])   # order changes, slots stable
        self.assertEqual(mapping, {102: 2, 101: 1})

    def test_slot_evicted_after_threshold(self):
        b = SlotBinder(num_slots=4, evict_after=2)
        b.update([101])
        b.update([])      # missing 1
        self.assertEqual(b.active_slots(), [1])   # not yet evicted
        b.update([])      # missing 2 -> evicted
        self.assertEqual(b.active_slots(), [])
        # slot 1 now free, new track reuses it
        self.assertEqual(b.update([200]), {200: 1})

    def test_manual_bind_moves_track(self):
        b = SlotBinder(num_slots=4, evict_after=2)
        b.update([101, 102])           # 101->1, 102->2
        b.manual_bind(101, 3)          # force 101 into slot 3
        mapping = b.update([101, 102])
        self.assertEqual(mapping[101], 3)
        self.assertEqual(mapping[102], 2)

    def test_swap_exchanges_two_slots(self):
        b = SlotBinder(num_slots=4, evict_after=2)
        b.update([101, 102])           # 101->1, 102->2
        b.swap(1, 2)
        mapping = b.update([101, 102])
        self.assertEqual(mapping[101], 2)
        self.assertEqual(mapping[102], 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_slot_binder -v` → Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/slot_binder.py
from typing import Dict, List, Optional


class SlotBinder:
    """Maps volatile tracker ids onto a fixed set of slots (1..num_slots).

    - Auto-assigns a new track to the lowest free slot.
    - Frees a slot whose bound track has been absent for > evict_after updates.
    - manual_bind / swap let an operator override the mapping.
    """

    def __init__(self, num_slots: int = 4, evict_after: int = 15):
        self.num_slots = num_slots
        self.evict_after = evict_after
        self.slot_to_track: Dict[int, Optional[int]] = {s: None for s in range(1, num_slots + 1)}
        self.missing: Dict[int, int] = {s: 0 for s in range(1, num_slots + 1)}

    def _track_to_slot(self) -> Dict[int, int]:
        return {t: s for s, t in self.slot_to_track.items() if t is not None}

    def _free_slots(self) -> List[int]:
        return [s for s in range(1, self.num_slots + 1) if self.slot_to_track[s] is None]

    def update(self, present_track_ids: List[int]) -> Dict[int, int]:
        present = list(dict.fromkeys(present_track_ids))  # de-dupe, keep order
        t2s = self._track_to_slot()

        # 1) assign new tracks to free slots
        for t in present:
            if t not in t2s:
                free = self._free_slots()
                if not free:
                    continue  # no room; track is dropped this frame
                slot = free[0]
                self.slot_to_track[slot] = t
        # rebuild after assignment
        t2s = self._track_to_slot()

        # 2) age/evict slots whose track is absent
        present_set = set(present)
        for slot, track in list(self.slot_to_track.items()):
            if track is None:
                continue
            if track in present_set:
                self.missing[slot] = 0
            else:
                self.missing[slot] += 1
                if self.missing[slot] > self.evict_after:
                    self.slot_to_track[slot] = None
                    self.missing[slot] = 0

        # 3) return mapping for tracks present AND still bound
        final = self._track_to_slot()
        return {t: final[t] for t in present if t in final}

    def manual_bind(self, track_id: int, slot: int) -> None:
        if slot not in self.slot_to_track:
            return
        # remove this track from any other slot first
        for s, t in self.slot_to_track.items():
            if t == track_id:
                self.slot_to_track[s] = None
                self.missing[s] = 0
        self.slot_to_track[slot] = track_id
        self.missing[slot] = 0

    def swap(self, slot_a: int, slot_b: int) -> None:
        if slot_a in self.slot_to_track and slot_b in self.slot_to_track:
            self.slot_to_track[slot_a], self.slot_to_track[slot_b] = (
                self.slot_to_track[slot_b], self.slot_to_track[slot_a])
            self.missing[slot_a] = self.missing[slot_b] = 0

    def active_slots(self) -> List[int]:
        return [s for s in range(1, self.num_slots + 1) if self.slot_to_track[s] is not None]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_slot_binder -v` → Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/slot_binder.py backend/tests/test_slot_binder.py
git commit -m "feat: add SlotBinder for stable per-slot identity"
```

---

## Task 5: Per-slot OSC manager

**Files:**
- Create: `backend/osc_manager.py`
- Test: `backend/tests/test_osc_manager.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_osc_manager.py
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osc_manager import MultiSlotOSC


class TestMultiSlotOSC(unittest.TestCase):
    def test_per_slot_addresses(self):
        m = MultiSlotOSC(num_slots=4, enabled=False, alpha=1.0)
        self.assertEqual(m.sender(1).metric_address("energy"), "/field/1/energy")
        self.assertEqual(m.sender(3).metric_address("sync_velocity"), "/field/3/sync_vel")

    def test_state_isolated_between_slots(self):
        m = MultiSlotOSC(num_slots=4, enabled=False, alpha=1.0, mode="normalize")
        m.sender(1)._prepare_value("energy", 10.0)   # slot1 peak -> 10
        v2 = m.sender(2)._prepare_value("energy", 5.0)  # slot2 has its own peak
        self.assertAlmostEqual(v2, 1.0)               # 5/5, not 5/10

    def test_configure_broadcasts(self):
        m = MultiSlotOSC(num_slots=4, enabled=False, alpha=1.0)
        m.configure(host="10.0.0.9", port=9001)
        for s in (1, 2, 3, 4):
            self.assertEqual(m.sender(s).host, "10.0.0.9")
            self.assertEqual(m.sender(s).port, 9001)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_osc_manager -v` → Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/osc_manager.py
from typing import Dict, List, Mapping, Optional

from pythonosc.udp_client import SimpleUDPClient

from osc_sender import OSCSender


class MultiSlotOSC:
    """Owns one OSCSender per slot (namespace /field/{slot}) plus a base
    client for meta messages (/field/active_slots, /field/count)."""

    def __init__(self, num_slots: int = 4, base_namespace: str = "/field",
                 host: str = "127.0.0.1", port: int = 9000, enabled: bool = True,
                 mode: str = "raw", alpha: float = 0.25):
        self.num_slots = num_slots
        self.base_namespace = base_namespace.rstrip("/") or "/field"
        self.host = host
        self.port = int(port)
        self.enabled = bool(enabled)
        self._senders: Dict[int, OSCSender] = {
            s: OSCSender(host=host, port=port, namespace=f"{self.base_namespace}/{s}",
                         enabled=enabled, mode=mode, alpha=alpha)
            for s in range(1, num_slots + 1)
        }
        self._meta_client = SimpleUDPClient(host, self.port)

    def sender(self, slot: int) -> OSCSender:
        return self._senders[slot]

    def send_slot(self, slot: int, metrics: Mapping[str, float]) -> None:
        self._senders[slot].send_metrics(metrics)

    def send_named_slot(self, slot: int, name: str, value: float) -> None:
        self._senders[slot].send_named(name, value)

    def send_meta(self, active_slots: List[int]) -> None:
        if not self.enabled:
            return
        try:
            self._meta_client.send_message(f"{self.base_namespace}/active_slots", active_slots)
            self._meta_client.send_message(f"{self.base_namespace}/count", len(active_slots))
        except Exception as exc:
            print(f"OSC meta send failed: {exc}")

    def configure(self, host: Optional[str] = None, port: Optional[int] = None,
                  enabled: Optional[bool] = None, mode: Optional[str] = None,
                  alpha: Optional[float] = None) -> None:
        if host is not None:
            self.host = host
        if port is not None:
            self.port = int(port)
        if enabled is not None:
            self.enabled = bool(enabled)
        for s in self._senders.values():
            s.configure(host=host, port=port, enabled=enabled, mode=mode, alpha=alpha)
        if host is not None or port is not None:
            self._meta_client = SimpleUDPClient(self.host, self.port)

    def prepared_for(self, slot: int) -> Dict[str, float]:
        return dict(self._senders[slot].last_prepared_metrics)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_osc_manager -v` → Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/osc_manager.py backend/tests/test_osc_manager.py
git commit -m "feat: add per-slot OSC manager (MultiSlotOSC)"
```

---

## Task 6: MediaPipeBackend (wrap existing path behind the interface)

Refactor the existing MediaPipe code into a `PoseBackend`. MediaPipe Tasks gives no track ids, so add a small centroid tracker. Keep the existing MP-33→H36M-17 mapping.

**Files:**
- Create: `backend/pose_backends/__init__.py` (empty)
- Create: `backend/pose_backends/mediapipe_backend.py`
- Create: `backend/centroid_tracker.py`
- Test: `backend/tests/test_centroid_tracker.py`

- [ ] **Step 1: Write the failing test (centroid tracker only — the MediaPipe model itself isn't unit-tested)**

```python
# backend/tests/test_centroid_tracker.py
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from centroid_tracker import CentroidTracker


class TestCentroidTracker(unittest.TestCase):
    def test_assigns_ids_and_keeps_them_stable(self):
        t = CentroidTracker(max_distance=50, evict_after=2)
        ids1 = t.update([(10, 10), (200, 200)])
        self.assertEqual(len(set(ids1)), 2)
        # small movement -> same ids, order preserved
        ids2 = t.update([(12, 11), (205, 198)])
        self.assertEqual(ids1, ids2)

    def test_new_centroid_gets_new_id(self):
        t = CentroidTracker(max_distance=50, evict_after=2)
        t.update([(10, 10)])
        ids = t.update([(10, 10), (400, 400)])
        self.assertEqual(len(set(ids)), 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_centroid_tracker -v` → Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/centroid_tracker.py
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

        # match each centroid to nearest existing object within max_distance
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

        # unmatched centroids -> new ids
        for idx, c in enumerate(centroids):
            if assigned[idx] is None:
                oid = self._next_id
                self._next_id += 1
                self._objects[oid] = c
                self._missing[oid] = 0
                assigned[idx] = oid
                used_ids.add(oid)

        # age + evict objects not seen this frame
        for oid in list(self._objects.keys()):
            if oid not in used_ids:
                self._missing[oid] += 1
                if self._missing[oid] > self.evict_after:
                    del self._objects[oid]
                    del self._missing[oid]
        return assigned
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_centroid_tracker -v` → Expected: PASS (2 tests).

- [ ] **Step 5: Write the MediaPipe backend (no new unit test; exercised by the smoke test in Task 9)**

```python
# backend/pose_backends/__init__.py
# (empty)
```

```python
# backend/pose_backends/mediapipe_backend.py
import cv2
import numpy as np
import mediapipe as mp

from pose_backend import PersonPose
from centroid_tracker import CentroidTracker

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def _mp33_to_h36m17(lms) -> np.ndarray:
    def g(i):
        return np.array([lms[i].x, lms[i].y, lms[i].z]) * 1000.0
    l_hip, r_hip = g(23), g(24)
    pelvis = (l_hip + r_hip) / 2
    l_sh, r_sh = g(11), g(12)
    neck = (l_sh + r_sh) / 2
    spine = (pelvis + neck) / 2
    head = g(0)
    j = np.zeros((17, 3))
    j[0] = pelvis
    j[1] = r_hip;  j[2] = g(26); j[3] = g(28)
    j[4] = l_hip;  j[5] = g(25); j[6] = g(27)
    j[7] = spine;  j[8] = neck;  j[9] = head
    j[10] = l_sh;  j[11] = g(13); j[12] = g(15)
    j[13] = r_sh;  j[14] = g(14); j[15] = g(16)
    j[16] = neck
    return j


class MediaPipeBackend:
    def __init__(self, model_path="pose_landmarker_full.task", num_poses=4):
        opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=num_poses)
        self.landmarker = PoseLandmarker.create_from_options(opts)
        self.tracker = CentroidTracker()

    def estimate(self, frame, timestamp_ms: float):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, int(timestamp_ms))
        people = []
        if not result.pose_world_landmarks:
            return people
        h, w, _ = frame.shape
        # centroid (pelvis) per person, in image space, for tracking
        centroids = []
        for img_lms in result.pose_landmarks:
            cx = (img_lms[23].x + img_lms[24].x) / 2 * w
            cy = (img_lms[23].y + img_lms[24].y) / 2 * h
            centroids.append((cx, cy))
        ids = self.tracker.update(centroids)
        for i, world_lms in enumerate(result.pose_world_landmarks):
            img_lms = result.pose_landmarks[i]
            kpts = np.array([[lm.x * w, lm.y * h] for lm in img_lms])
            xs, ys = kpts[:, 0], kpts[:, 1]
            bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            people.append(PersonPose(
                track_id=ids[i], h36m17=_mp33_to_h36m17(world_lms),
                bbox=bbox, kpts_2d=kpts, is_3d=True))
        return people

    def close(self):
        self.landmarker.close()
```

- [ ] **Step 6: Commit**

```bash
git add backend/pose_backends backend/centroid_tracker.py backend/tests/test_centroid_tracker.py
git commit -m "feat: MediaPipe pose backend + centroid tracker"
```

---

## Task 7: YOLO26Backend

Add Ultralytics. **At implementation time, verify** the installed `ultralytics` version exposes a YOLO26 pose weight (`yolo26m-pose.pt`); if not yet pinnable, fall back to the newest available `*-pose.pt` and leave a TODO to bump. The `model.track(..., persist=True)` API and `results[0].keypoints` / `results[0].boxes.id` shape are stable across recent versions.

**Files:**
- Modify: `requirements.txt` (add `ultralytics`)
- Create: `backend/pose_backends/yolo_backend.py`
- Test: `backend/tests/test_yolo_backend.py` (mapping/adaptation logic with a faked result; the model is not loaded in tests)

- [ ] **Step 1: Add dependency**

Append to `requirements.txt`:
```
# Multi-person detection + tracking (YOLO26 pose, BoT-SORT)
ultralytics
```
Run: `pip install ultralytics`

- [ ] **Step 2: Write the failing test (result-adaptation helper, no model load)**

```python
# backend/tests/test_yolo_backend.py
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_backends.yolo_backend import results_to_personposes


class _FakeKpts:
    def __init__(self, xy): self.xy = xy            # (N,17,2) tensor-like (np ok)
class _FakeBoxes:
    def __init__(self, xyxy, ids):
        self.xyxy = xyxy
        self.id = ids                                # (N,) or None
class _FakeResult:
    def __init__(self, kpts, boxes):
        self.keypoints = kpts
        self.boxes = boxes


class TestYoloAdaptation(unittest.TestCase):
    def test_builds_personposes_with_ids(self):
        kxy = np.zeros((2, 17, 2)); kxy[0] += 1.0; kxy[1] += 2.0
        boxes = _FakeBoxes(np.array([[0, 0, 10, 20], [5, 5, 30, 40]]),
                           np.array([7, 9]))
        out = results_to_personposes(_FakeResult(_FakeKpts(kxy), boxes))
        self.assertEqual([p.track_id for p in out], [7, 9])
        self.assertTrue(all(p.is_3d is False for p in out))
        self.assertEqual(out[0].h36m17.shape, (17, 3))

    def test_skips_when_no_ids_yet(self):
        kxy = np.zeros((1, 17, 2))
        boxes = _FakeBoxes(np.array([[0, 0, 10, 20]]), None)
        out = results_to_personposes(_FakeResult(_FakeKpts(kxy), boxes))
        self.assertEqual(out, [])   # no track ids -> nothing to bind yet


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m unittest tests.test_yolo_backend -v` → Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 4: Write minimal implementation**

```python
# backend/pose_backends/yolo_backend.py
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
        from ultralytics import YOLO   # imported lazily so tests don't need the dep
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m unittest tests.test_yolo_backend -v` → Expected: PASS (2 tests).

- [ ] **Step 6: Manual smoke (one-time, not automated)**

Run from `backend/`:
```bash
python -c "from pose_backends.yolo_backend import YOLO26Backend; b=YOLO26Backend(); import cv2,numpy as np; print(b.estimate(np.zeros((480,640,3),np.uint8),0))"
```
Expected: prints `[]` (no people in a blank frame) without error; first run downloads the weight.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt backend/pose_backends/yolo_backend.py backend/tests/test_yolo_backend.py
git commit -m "feat: YOLO26 pose backend with BoT-SORT tracking"
```

---

## Task 8: Backend factory + wire the stream loop to slots

**Files:**
- Create: `backend/pose_backends/factory.py`
- Modify: `backend/osc_viewer.py` — replace single-engine pipeline; add slot endpoints; extend `state_payload`.

- [ ] **Step 1: Backend factory**

```python
# backend/pose_backends/factory.py
import os


def make_backend(name: str = None):
    name = (name or os.getenv("FIELD_POSE_BACKEND", "yolo")).lower()
    if name == "mediapipe":
        from pose_backends.mediapipe_backend import MediaPipeBackend
        return MediaPipeBackend()
    from pose_backends.yolo_backend import YOLO26Backend
    return YOLO26Backend()
```

- [ ] **Step 2: Replace the per-frame pipeline in `osc_viewer.py`**

Introduce module-level multi-person state near the existing `osc_sender` setup (`osc_viewer.py:34`). Replace the single `OSCSender` with `MultiSlotOSC`, add `SlotBinder`, a `{slot: DanceMetricsEngine}` dict, and the backend:

```python
from osc_manager import MultiSlotOSC
from slot_binder import SlotBinder
from dance_metrics import DanceMetricsEngine
from pose_backends.factory import make_backend

osc = MultiSlotOSC(
    host=os.getenv("FIELD_OSC_HOST", "127.0.0.1"),
    port=int(os.getenv("FIELD_OSC_PORT", "9000")),
    enabled=os.getenv("FIELD_OSC_ENABLED", "1") == "1",
    mode=os.getenv("FIELD_OSC_MODE", "raw"),
    alpha=float(os.getenv("FIELD_OSC_ALPHA", "0.25")),
)
binder = SlotBinder(num_slots=4)
slot_engines: dict[int, DanceMetricsEngine] = {}
_backend = None  # lazily created in the worker

def get_backend():
    global _backend
    if _backend is None:
        _backend = make_backend()
    return _backend
```

Add a function that processes one frame for all people (runs in the worker thread):

```python
def process_multi(frame, timestamp_ms, measured_fps, draw=True):
    backend = get_backend()
    people = backend.estimate(frame, timestamp_ms)
    mapping = binder.update([p.track_id for p in people])  # track_id -> slot
    by_track = {p.track_id: p for p in people}

    latest = {}
    for track_id, slot in mapping.items():
        person = by_track[track_id]
        eng = slot_engines.get(slot)
        if eng is None or eng.is_3d != person.is_3d:
            eng = DanceMetricsEngine(fps=max(measured_fps, 1), is_3d=person.is_3d)
            slot_engines[slot] = eng
        eng.set_fps(max(measured_fps, 1))
        metrics = eng.update(person.h36m17)
        osc.send_slot(slot, metrics)
        latest[slot] = dict(osc.prepared_for(slot))
        if draw:
            draw_person(frame, person, slot)   # see Task 9

    # drop engines whose slot was evicted
    for slot in list(slot_engines.keys()):
        if slot not in binder.active_slots():
            slot_engines.pop(slot, None)

    osc.send_meta(binder.active_slots())
    return frame, latest
```

In `stream_live` / `stream_video`, replace the `engine.process_frame(...)` call (`osc_viewer.py:408-414` and `:516-522`) with `await asyncio.to_thread(process_multi, frame, ts_ms, measured_fps)` and store `processing_state["slots"] = latest`.

> **Threading note:** because YOLO `track(persist=True)` is stateful, `process_multi` must be the *only* caller of the backend and must not run concurrently with itself. The single live/video stream loop already serializes frames, so one `asyncio.to_thread` call at a time is sufficient.

- [ ] **Step 3: Slot reassignment endpoints**

```python
@app.post("/api/slots/bind")
async def api_slot_bind(track_id: int = Form(...), slot: int = Form(...)):
    binder.manual_bind(track_id, slot)
    return {"status": "ok", "active_slots": binder.active_slots()}

@app.post("/api/slots/swap")
async def api_slot_swap(slot_a: int = Form(...), slot_b: int = Form(...)):
    binder.swap(slot_a, slot_b)
    return {"status": "ok", "active_slots": binder.active_slots()}
```

- [ ] **Step 4: Extend `state_payload`** to include `"slots": processing_state.get("slots", {})` and `"active_slots": binder.active_slots()` so the websocket pushes per-slot metrics.

- [ ] **Step 5: Update OSC config endpoints** (`apply_osc_config`, `osc_viewer.py:596`) to call `osc.configure(...)` instead of the old single `osc_sender.configure(...)`.

- [ ] **Step 6: Run the full unit suite (no regressions)**

Run: `python -m unittest discover -s backend/tests -t backend` → Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/pose_backends/factory.py backend/osc_viewer.py
git commit -m "feat: wire multi-person slot pipeline into the stream loop"
```

---

## Task 9: Viewer UI — 4 panels, slot skeletons, manual reassign

**Files:**
- Modify: `backend/osc_viewer.py` — `draw_person` helper + `VIEWER_HTML` (inline template).

- [ ] **Step 1: Per-person draw helper (Python side)**

```python
SLOT_COLORS = {1: (0, 255, 255), 2: (255, 128, 0), 3: (0, 255, 0), 4: (255, 0, 255)}

def draw_person(frame, person, slot):
    import cv2
    color = SLOT_COLORS.get(slot, (200, 200, 200))
    x1, y1, x2, y2 = [int(v) for v in person.bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"#{slot}", (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    for (px, py) in person.kpts_2d.astype(int):
        cv2.circle(frame, (px, py), 3, color, -1)
```

- [ ] **Step 2: VIEWER_HTML — render 4 panels from `state.slots`**

In the metrics aside, replace the single panel with a 4-slot grid. Each panel reads `state.slots[slot]` (the per-slot prepared metrics) and shows the 9 metric bars; empty slots render greyed. Slot color matches `SLOT_COLORS`. Add a small reassign control per panel:

```html
<div id="slotPanels" class="slot-grid">
  <!-- one .slot-panel[data-slot] per slot, built by renderSlots() -->
</div>
```

```javascript
const SLOT_COLORS = {1:'#00ffff',2:'#ff8000',3:'#00ff00',4:'#ff00ff'};
const METRIC_KEYS = ['energy','sync_velocity','sync_correlation','expansion',
                     'curvature','height','sway','torque','jerk'];

function renderSlots(state) {
  const slots = state.slots || {};
  const active = state.active_slots || [];
  const grid = document.getElementById('slotPanels');
  grid.innerHTML = '';
  for (let s = 1; s <= 4; s++) {
    const panel = document.createElement('div');
    panel.className = 'slot-panel' + (active.includes(s) ? '' : ' empty');
    panel.style.borderColor = SLOT_COLORS[s];
    const metrics = slots[s] || {};
    const rows = METRIC_KEYS.map(k =>
      `<div class="m-row"><span>${k}</span><b>${(metrics[k] ?? 0).toFixed(2)}</b></div>`).join('');
    panel.innerHTML = `<header style="color:${SLOT_COLORS[s]}">Dancer ${s}</header>${rows}
      <div class="slot-actions">
        <button onclick="swapSlot(${s})">swap…</button>
      </div>`;
    grid.appendChild(panel);
  }
}

let swapFrom = null;
function swapSlot(s) {
  if (swapFrom === null) { swapFrom = s; return; }
  const body = new FormData(); body.append('slot_a', swapFrom); body.append('slot_b', s);
  fetch('/api/slots/swap', {method:'POST', body}); swapFrom = null;
}
```

Wire `renderSlots(state)` into the existing websocket `onmessage` handler that already receives `state`. Keep the existing fullscreen overlay but point it at the 4-panel grid.

> Manual bind-by-click (click a skeleton on the video → click a slot) can reuse `/api/slots/bind`; for the first pass the **swap** control above satisfies "manual reassignment". Click-to-bind on the canvas is a follow-up refinement.

- [ ] **Step 3: Manual verification (the app)**

Launch the viewer (per repo env: global Python 3.10, `Start-Process`), open `http://localhost:9100`, run a clip/camera with 2+ people. Confirm: skeletons get slot colors + `#n` labels; 4 panels update; swap button exchanges two dancers; OSC monitor shows `/field/1/..`–`/field/4/..` and `/field/active_slots`.

- [ ] **Step 4: Commit**

```bash
git add backend/osc_viewer.py
git commit -m "feat: 4-panel multi-person viewer UI with slot reassign"
```

---

## Task 10: Integration smoke test + README/docs

**Files:**
- Create: `backend/tests/test_pipeline_smoke.py`
- Modify: `README.md` (document `FIELD_POSE_BACKEND`, per-slot OSC schema)

- [ ] **Step 1: Smoke test wiring the logic units together (no model load)**

```python
# backend/tests/test_pipeline_smoke.py
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slot_binder import SlotBinder
from osc_manager import MultiSlotOSC
from dance_metrics import DanceMetricsEngine
from keypoint_mapping import coco17_to_h36m17


class TestPipelineSmoke(unittest.TestCase):
    def test_two_people_get_isolated_slots_and_metrics(self):
        binder = SlotBinder(num_slots=4)
        osc = MultiSlotOSC(num_slots=4, enabled=False, alpha=1.0)
        engines = {}
        # two fake COCO people, a few frames of motion
        rng = np.random.default_rng(0)
        for _frame in range(5):
            people = {7: rng.random((17, 2)) * 100, 12: rng.random((17, 2)) * 100}
            mapping = binder.update(list(people))
            for tid, slot in mapping.items():
                eng = engines.setdefault(slot, DanceMetricsEngine(fps=30, is_3d=False))
                m = eng.update(coco17_to_h36m17(people[tid]))
                osc.send_slot(slot, m)
        self.assertEqual(sorted(binder.active_slots()), [1, 2])
        self.assertIn("energy", osc.prepared_for(1))
        self.assertIn("energy", osc.prepared_for(2))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it** → `python -m unittest tests.test_pipeline_smoke -v` → Expected: PASS.

- [ ] **Step 3: Run the whole suite** → `python -m unittest discover -s backend/tests -t backend` → Expected: all PASS.

- [ ] **Step 4: Document** in `README.md`: backend selection env var `FIELD_POSE_BACKEND=yolo|mediapipe` (default `yolo`), per-slot OSC addresses `/field/{1..4}/<metric>`, meta `/field/active_slots` + `/field/count`, manual reassign via the panel swap button.

- [ ] **Step 5: Commit**

```bash
git add backend/tests/test_pipeline_smoke.py README.md
git commit -m "test: multi-person pipeline smoke + docs"
```

---

## Self-Review Notes (author)

- **Spec coverage:** PoseBackend (T1), YOLO26 default + COCO map (T2,T7), MediaPipe optional (T6), 2D adaptation expansion/sway (T3), slot identity + manual reassign (T4,T8,T9), per-slot metrics lifecycle (T8), per-slot OSC + meta (T5,T8), 4-panel UI + skeleton labels (T9), single-worker threading note (T8), out-of-scope items untouched. ✓
- **Sway:** spec said "→ horizontal X"; implementation note clarifies it already degenerates to `|dx|` at z=0, so no code change — documented in T3 to avoid a phantom task.
- **Types consistent:** `PersonPose(track_id, h36m17, bbox, kpts_2d, is_3d)` used identically in T1/T6/T7/T8/T9; `SlotBinder.update→dict[track_id,slot]`, `.manual_bind`, `.swap`, `.active_slots`; `MultiSlotOSC.sender/send_slot/send_meta/configure/prepared_for`; `DanceMetricsEngine(fps, is_3d)`.
- **Known unknown (flagged, not a placeholder):** exact YOLO26 pose weight name / `ultralytics` pin — verify at T7 install; API calls used are version-stable.
