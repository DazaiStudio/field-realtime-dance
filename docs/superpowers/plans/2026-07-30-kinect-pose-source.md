# Azure Kinect Pose Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Azure Kinect body tracking as a selectable pose backend (frame source + 32-joint skeletons → H36M-17 → existing per-person metrics/OSC pipeline), per spec `docs/superpowers/specs/2026-07-30-kinect-pose-source-design.md`.

**Architecture:** A `KinectRuntime` singleton owns the k4a device + body tracker and doubles as the frame provider (color or colorized-depth view). `AzureKinectPoseSource` implements the existing duck-typed PoseSource protocol by consuming the runtime's cached bodies, feeding `MultiPersonTrackRegistry` with `PersonTrack(track_id=body_id, ...)`. A minimal `FrameSource` protocol decouples `stream_live()` from `cv2.VideoCapture` via delegation (no behavior change for the OpenCV path).

**Tech Stack:** Python 3.10, pykinect_azure 0.0.4 (lazy import), numpy, OpenCV, unittest. Test command (always from `backend/`): `python -m unittest discover -s tests`.

**Branch:** `feat/kinect-pose-source`. Run baseline first: full suite must be green before Task 1.

**Machine facts (verified 2026-07-30):** DirectML `gpu_device_id=1` = RTX 4080 (adapter 0 = iGPU, 6 fps — unusable). CUDA mode broken on this machine (SDK's ORT 1.10 provider fails init). K4ABT `Body.numpy()` → `(32, 8)` = `[x_mm, y_mm, z_mm, qw, qx, qy, qz, confidence(0-3)]`.

---

### Task 0: Baseline

- [ ] **Step 0.1:** Run: `cd D:\Github\field-realtime-dance\backend; python -m unittest discover -s tests` (use global Python 3.10: `C:\Users\tommy\AppData\Local\Programs\Python\Python310\python.exe`). Record the pass count (MAINTENANCE.md is self-contradictory: 89 vs 105). Expected: all green, 1 skip. Do not proceed on red.

---

### Task 1: K4ABT-32 → H36M-17 mapping + mirror helper

**Files:**
- Modify: `backend/keypoint_mapping.py` (append)
- Test: `backend/tests/test_keypoint_mapping.py` (append)

- [ ] **Step 1.1: Write failing tests** — append to `backend/tests/test_keypoint_mapping.py`:

```python
import numpy as np

from keypoint_mapping import k4abt32_to_h36m17, mirror_h36m17


def _k4abt_body():
    """Synthetic 32-joint body, mm, depth-camera coords (y down)."""
    j = np.zeros((32, 3))
    j[18] = [-100, 0, 1000]   # HIP_LEFT
    j[22] = [100, 0, 1000]    # HIP_RIGHT
    j[19] = [-110, 400, 1000]  # KNEE_LEFT
    j[23] = [110, 400, 1000]   # KNEE_RIGHT
    j[20] = [-120, 800, 1000]  # ANKLE_LEFT
    j[24] = [120, 800, 1000]   # ANKLE_RIGHT
    j[5] = [-180, -500, 1000]  # SHOULDER_LEFT
    j[12] = [180, -500, 1000]  # SHOULDER_RIGHT
    j[6] = [-200, -250, 1000]  # ELBOW_LEFT
    j[13] = [200, -250, 1000]  # ELBOW_RIGHT
    j[7] = [-210, 0, 1000]     # WRIST_LEFT
    j[14] = [210, 0, 1000]     # WRIST_RIGHT
    j[27] = [0, -700, 1000]    # NOSE
    return j


class K4abtMappingTests(unittest.TestCase):
    def test_shape_and_root_centering(self):
        h = k4abt32_to_h36m17(_k4abt_body())
        self.assertEqual(h.shape, (17, 3))
        # pelvis = hip midpoint, root-centered at origin
        np.testing.assert_allclose(h[0], [0, 0, 0], atol=1e-9)

    def test_joint_assignment(self):
        h = k4abt32_to_h36m17(_k4abt_body())
        np.testing.assert_allclose(h[3], [120, 800, 0])    # r_ankle (z centered)
        np.testing.assert_allclose(h[6], [-120, 800, 0])   # l_ankle
        np.testing.assert_allclose(h[13], [-210, 0, 0])    # l_wrist
        np.testing.assert_allclose(h[16], [210, 0, 0])     # r_wrist
        np.testing.assert_allclose(h[10], [0, -700, 0])    # head=nose

    def test_derived_joints(self):
        h = k4abt32_to_h36m17(_k4abt_body())
        np.testing.assert_allclose(h[8], [0, -500, 0])     # thorax = mid-shoulders
        np.testing.assert_allclose(h[7], [0, -250, 0])     # spine = mid(pelvis, thorax)
        np.testing.assert_allclose(h[9], [0, -600, 0])     # neck = mid(thorax, nose)

    def test_accepts_extra_columns(self):
        j = np.zeros((32, 8))
        j[:, :3] = _k4abt_body()
        h = k4abt32_to_h36m17(j)
        np.testing.assert_allclose(h[3], [120, 800, 0])


class MirrorH36MTests(unittest.TestCase):
    def test_involution(self):
        h = k4abt32_to_h36m17(_k4abt_body())
        np.testing.assert_allclose(mirror_h36m17(mirror_h36m17(h)), h, atol=1e-9)

    def test_swaps_sides_and_negates_x(self):
        h = k4abt32_to_h36m17(_k4abt_body())
        m = mirror_h36m17(h)
        # right ankle (index 3) becomes the mirrored left ankle
        np.testing.assert_allclose(m[3], [120, 800, 0])   # was l_ankle (-120) -> x negated
        np.testing.assert_allclose(m[6], [-120, 800, 0])
        np.testing.assert_allclose(m[13], [-210, 0, 0])
        # midline joints: x negated only
        np.testing.assert_allclose(m[10], [0, -700, 0])
```

(`import unittest` and existing imports are already at the top of the file — only add what's missing.)

- [ ] **Step 1.2:** Run: `python -m unittest tests.test_keypoint_mapping -v`. Expected: FAIL (ImportError: cannot import name 'k4abt32_to_h36m17').

- [ ] **Step 1.3: Implement** — append to `backend/keypoint_mapping.py`:

```python
# --- Azure Kinect Body Tracking (K4ABT) 32-joint indices ---
_K4_L_SH, _K4_L_EL, _K4_L_WR = 5, 6, 7
_K4_R_SH, _K4_R_EL, _K4_R_WR = 12, 13, 14
_K4_L_HIP, _K4_L_KNEE, _K4_L_ANK = 18, 19, 20
_K4_R_HIP, _K4_R_KNEE, _K4_R_ANK = 22, 23, 24
_K4_NOSE = 27

# H36M index pairs to swap when mirroring (right limb <-> left limb).
_H36M_MIRROR_SWAP = [(1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16)]


def k4abt32_to_h36m17(joints: np.ndarray) -> np.ndarray:
    """K4ABT 32-joint skeleton (mm, depth-camera coords) -> standard H36M-17,
    root-centered on the hip midpoint. Accepts (32, 3+) arrays (extra columns
    such as orientation/confidence are ignored). Same conventions as the other
    mappings: pelvis = hip midpoint, head = nose, spine/thorax/neck derived."""
    j = np.asarray(joints, dtype=float)[:, :3]

    def p(i):
        return j[i]

    h36m = _assemble_h36m17(
        pelvis=(p(_K4_L_HIP) + p(_K4_R_HIP)) / 2.0,
        r_hip=p(_K4_R_HIP), r_knee=p(_K4_R_KNEE), r_ank=p(_K4_R_ANK),
        l_hip=p(_K4_L_HIP), l_knee=p(_K4_L_KNEE), l_ank=p(_K4_L_ANK),
        l_sh=p(_K4_L_SH), l_el=p(_K4_L_EL), l_wr=p(_K4_L_WR),
        r_sh=p(_K4_R_SH), r_el=p(_K4_R_EL), r_wr=p(_K4_R_WR),
        nose=p(_K4_NOSE),
    )
    return h36m - h36m[0]


def mirror_h36m17(h36m: np.ndarray) -> np.ndarray:
    """Mirror a standard H36M-17 skeleton: negate x and swap left/right limbs.
    Matches what the camera mirror does to the displayed image, so mirrored
    frames and skeleton data stay consistent."""
    out = np.asarray(h36m, dtype=float).copy()
    out[:, 0] = -out[:, 0]
    for a, b in _H36M_MIRROR_SWAP:
        out[[a, b]] = out[[b, a]]
    return out
```

- [ ] **Step 1.4:** Run: `python -m unittest tests.test_keypoint_mapping -v`. Expected: PASS (all, including the 2 pre-existing tests).

- [ ] **Step 1.5:** Commit: `git add backend/keypoint_mapping.py backend/tests/test_keypoint_mapping.py; git commit -m "Add K4ABT-32 -> H36M-17 mapping + H36M mirror helper"`

---

### Task 2: Kinect pure helpers (quality gate, 2D transforms, aspect pad)

**Files:**
- Create: `backend/pose_backends/__init__.py` (empty file)
- Create: `backend/pose_backends/azure_kinect.py`
- Test: `backend/tests/test_pose_backends_kinect.py` (new)

No pykinect import at module top — hardware bindings are imported lazily inside `KinectRuntime` methods only (Task 4), so this module imports everywhere (CI/macOS).

- [ ] **Step 2.1: Write failing tests** — create `backend/tests/test_pose_backends_kinect.py`:

```python
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_backends.azure_kinect import (
    KinectBody,
    body_quality,
    bbox_from_points,
    transform_points_2d,
    pad_to_aspect,
)


def _conf(fill=2):
    return np.full(32, fill, dtype=float)


class BodyQualityTests(unittest.TestCase):
    def test_all_medium_is_valid(self):
        quality, valid = body_quality(_conf(2))
        self.assertAlmostEqual(quality, 0.8)
        self.assertTrue(valid)

    def test_none_core_joint_invalidates(self):
        conf = _conf(3)
        conf[18] = 0  # HIP_LEFT = NONE
        quality, valid = body_quality(conf)
        self.assertFalse(valid)

    def test_none_peripheral_joint_keeps_valid(self):
        conf = _conf(2)
        conf[7] = 0  # WRIST_LEFT
        quality, valid = body_quality(conf)
        self.assertTrue(valid)
        self.assertLess(quality, 0.8)


class Transform2DTests(unittest.TestCase):
    def test_scale_only(self):
        pts = np.array([[640.0, 360.0]])
        out = transform_points_2d(pts, native_size=(1280, 720),
                                  frame_size=(1920, 1080), mirrored=False)
        np.testing.assert_allclose(out, [[960.0, 540.0]])

    def test_mirror_flips_x_in_native_space(self):
        pts = np.array([[100.0, 50.0]])
        out = transform_points_2d(pts, native_size=(1280, 720),
                                  frame_size=(1280, 720), mirrored=True)
        np.testing.assert_allclose(out, [[1180.0, 50.0]])


class BboxTests(unittest.TestCase):
    def test_bbox_padded_and_clamped(self):
        pts = np.array([[10.0, 10.0], [110.0, 210.0]])
        x1, y1, x2, y2 = bbox_from_points(pts, frame_size=(200, 220), pad_frac=0.1)
        self.assertAlmostEqual(x1, 0.0)     # 10 - 10% of 100 = 0
        self.assertAlmostEqual(y1, 0.0)     # 10 - 10% of 200 = -10 -> clamp 0
        self.assertAlmostEqual(x2, 120.0)
        self.assertAlmostEqual(y2, 220.0)   # 230 -> clamp to frame


class PadToAspectTests(unittest.TestCase):
    def test_nfov_depth_to_16_9(self):
        img = np.zeros((576, 640, 3), dtype=np.uint8)
        out, x_off, y_off = pad_to_aspect(img, 16, 9)
        self.assertEqual(out.shape[0], 576)
        self.assertEqual(out.shape[1], 1024)  # 576 * 16/9
        self.assertEqual(x_off, (1024 - 640) // 2)
        self.assertEqual(y_off, 0)

    def test_already_wide_enough(self):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        out, x_off, y_off = pad_to_aspect(img, 16, 9)
        self.assertEqual(out.shape, (720, 1280, 3))
        self.assertEqual((x_off, y_off), (0, 0))
```

- [ ] **Step 2.2:** Run: `python -m unittest tests.test_pose_backends_kinect -v`. Expected: FAIL (ModuleNotFoundError: pose_backends.azure_kinect).

- [ ] **Step 2.3: Implement** — create empty `backend/pose_backends/__init__.py`, then create `backend/pose_backends/azure_kinect.py`:

```python
"""Azure Kinect body-tracking backend: frame source + pose source.

Layering:
  - Pure helpers + AzureKinectPoseSource: no hardware deps, import anywhere,
    fully unit-tested against a fake runtime.
  - KinectRuntime: the only code that touches pykinect_azure (lazy imports).
    pykinect's VERIFY() calls sys.exit(1) on any sensor error, so every call
    into it goes through _guarded() which converts that into KinectError.

Verified on this machine (2026-07-30): DirectML gpu_device_id=1 = RTX 4080
(adapter 0 = iGPU at ~6 fps). Override with FIELD_KINECT_GPU on other rigs.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

# --- K4ABT joint indices used here (see keypoint_mapping for the mapping) ----
_CORE_JOINTS = (5, 12, 18, 22)          # shoulders + hips
_MAPPED_JOINTS = (18, 19, 20, 22, 23, 24, 5, 6, 7, 12, 13, 14, 27)
_CONF_QUALITY = {0: 0.0, 1: 0.4, 2: 0.8, 3: 1.0}

KINECT_VIEWS = ("color", "depth")


class KinectError(RuntimeError):
    """Sensor/tracker failure surfaced as an ordinary exception."""


@dataclass
class KinectBody:
    body_id: int
    joints: np.ndarray      # (32, 4): x_mm, y_mm, z_mm, confidence(0-3)
    joints2d: np.ndarray    # (32, 2): pixels in the native view image


def body_quality(confidences: np.ndarray) -> tuple[float, bool]:
    """K4ABT confidence levels (0..3) -> (quality 0..1, valid).
    Quality = mean over the 13 joints the H36M mapping uses; a body is invalid
    when any core joint (hips/shoulders) has NONE confidence."""
    conf = np.asarray(confidences, dtype=int)
    levels = [_CONF_QUALITY.get(int(conf[i]), 0.0) for i in _MAPPED_JOINTS]
    quality = float(np.mean(levels)) if levels else 0.0
    valid = all(int(conf[i]) > 0 for i in _CORE_JOINTS)
    return quality, bool(valid)


def transform_points_2d(points: np.ndarray, native_size: tuple[int, int],
                        frame_size: tuple[int, int], mirrored: bool) -> np.ndarray:
    """Native view pixels -> displayed frame pixels (mirror happens in native
    space first, matching apply_live_mirror flipping the raw view image)."""
    pts = np.asarray(points, dtype=float).copy()
    nw, nh = native_size
    fw, fh = frame_size
    if mirrored:
        pts[:, 0] = float(nw) - pts[:, 0]
    pts[:, 0] *= float(fw) / max(float(nw), 1.0)
    pts[:, 1] *= float(fh) / max(float(nh), 1.0)
    return pts


def bbox_from_points(points: np.ndarray, frame_size: tuple[int, int],
                     pad_frac: float = 0.08) -> tuple[float, float, float, float]:
    """Padded, frame-clamped bbox around 2D joints (for the track registry)."""
    pts = np.asarray(points, dtype=float)
    fw, fh = frame_size
    x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
    pad_x, pad_y = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
    return (max(0.0, x1 - pad_x), max(0.0, y1 - pad_y),
            min(float(fw), x2 + pad_x), min(float(fh), y2 + pad_y))


def pad_to_aspect(image: np.ndarray, aspect_w: int, aspect_h: int):
    """Pad an image with black bars to at least the given aspect ratio.
    Returns (padded, x_offset, y_offset) so joint pixels can be shifted.
    Used for the NFOV depth view (640x576) so the 16:9 stream resize doesn't
    stretch it."""
    h, w = image.shape[:2]
    target_w = int(round(h * aspect_w / aspect_h))
    if target_w <= w:
        return image, 0, 0
    left = (target_w - w) // 2
    right = target_w - w - left
    padded = cv2.copyMakeBorder(image, 0, 0, left, right,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded, left, 0
```

- [ ] **Step 2.4:** Run: `python -m unittest tests.test_pose_backends_kinect -v`. Expected: PASS.

- [ ] **Step 2.5:** Commit: `git add backend/pose_backends; git add backend/tests/test_pose_backends_kinect.py; git commit -m "Add Kinect backend pure helpers (quality gate, 2D transform, aspect pad)"`

---

### Task 3: AzureKinectPoseSource (against a fake runtime)

**Files:**
- Modify: `backend/pose_backends/azure_kinect.py` (append)
- Test: `backend/tests/test_pose_backends_kinect.py` (append)

- [ ] **Step 3.1: Write failing tests** — append to `backend/tests/test_pose_backends_kinect.py`:

```python
from pose_backends.azure_kinect import AzureKinectPoseSource


class FakeRuntime:
    def __init__(self, bodies=None, view="color", mirrored=False,
                 native_size=(1280, 720)):
        self.last_bodies = bodies or []
        self.view = view
        self.mirrored = mirrored
        self.native_view_size = native_size
        self.last_error = None


def _fake_body(body_id, x_offset=0.0, conf=2):
    joints = np.zeros((32, 4))
    joints[:, 3] = conf
    base = {18: [-100, 0, 1000], 22: [100, 0, 1000],
            19: [-110, 400, 1000], 23: [110, 400, 1000],
            20: [-120, 800, 1000], 24: [120, 800, 1000],
            5: [-180, -500, 1000], 12: [180, -500, 1000],
            6: [-200, -250, 1000], 13: [200, -250, 1000],
            7: [-210, 0, 1000], 14: [210, 0, 1000],
            27: [0, -700, 1000]}
    for i, xyz in base.items():
        joints[i, :3] = xyz
        joints[i, 0] += x_offset
    joints2d = np.zeros((32, 2))
    joints2d[:, 0] = 300.0 + x_offset / 5.0
    joints2d[:, 1] = 300.0
    joints2d[20, 1] = joints2d[24, 1] = 600.0   # ankles lower for a real bbox
    return KinectBody(body_id=body_id, joints=joints, joints2d=joints2d)


def _frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


class PoseSourceSingleTests(unittest.TestCase):
    def test_no_bodies(self):
        source = AzureKinectPoseSource(FakeRuntime())
        frame, h36m = source.estimate(_frame(), 1000.0)
        self.assertIsNone(h36m)
        self.assertFalse(source.last_pose_valid)
        self.assertIsNone(source.last_h36m_by_id)

    def test_single_body_disabled_tracking(self):
        source = AzureKinectPoseSource(FakeRuntime(bodies=[_fake_body(7)]))
        frame, h36m = source.estimate(_frame(), 1000.0)
        self.assertEqual(h36m.shape, (17, 3))
        self.assertTrue(source.last_pose_valid)
        self.assertAlmostEqual(source.last_pose_quality, 0.8)
        self.assertIsNone(source.last_h36m_by_id)      # single-person contract
        # registry still runs: active id present so PoseEngine resets on change
        self.assertEqual(source.last_tracking["active_id"], 1)
        self.assertFalse(source.last_tracking["enabled"])

    def test_invalid_body_returns_none(self):
        body = _fake_body(7)
        body.joints[18, 3] = 0   # HIP_LEFT NONE
        source = AzureKinectPoseSource(FakeRuntime(bodies=[body]))
        frame, h36m = source.estimate(_frame(), 1000.0)
        self.assertIsNone(h36m)
        self.assertFalse(source.last_pose_valid)


class PoseSourceTrackedTests(unittest.TestCase):
    def _source(self, runtime):
        return AzureKinectPoseSource(runtime, tracking_enabled=True)

    def test_two_bodies_get_stable_ids(self):
        runtime = FakeRuntime(bodies=[_fake_body(11), _fake_body(23, x_offset=500)])
        source = self._source(runtime)
        source.estimate(_frame(), 1000.0)
        self.assertEqual(set(source.last_h36m_by_id), {1, 2})
        self.assertEqual(source.last_tracking["count"], 2)
        states = {t["stable_id"]: t["state"] for t in source.last_tracking["tracks"]}
        self.assertEqual(states, {1: "tracking", 2: "tracking"})

    def test_body_id_change_keeps_stable_id(self):
        runtime = FakeRuntime(bodies=[_fake_body(11)])
        source = self._source(runtime)
        source.estimate(_frame(), 1000.0)
        runtime.last_bodies = [_fake_body(99)]   # K4ABT re-assigned the raw id
        source.estimate(_frame(), 1100.0)
        self.assertEqual(list(source.last_h36m_by_id), [1])   # registry re-id

    def test_mirror_flips_skeleton(self):
        runtime = FakeRuntime(bodies=[_fake_body(3)], mirrored=True)
        source = self._source(runtime)
        source.estimate(_frame(), 1000.0)
        h = source.last_h36m_by_id[1]
        # r_ankle should carry the mirrored left ankle x (+120)
        np.testing.assert_allclose(h[3], [120, 800, 0], atol=1e-6)

    def test_configure_tracking_toggles(self):
        runtime = FakeRuntime(bodies=[_fake_body(4)])
        source = self._source(runtime)
        source.configure_tracking(enabled=False)
        source.estimate(_frame(), 1000.0)
        self.assertIsNone(source.last_h36m_by_id)
        source.configure_tracking(enabled=True)
        source.estimate(_frame(), 1100.0)
        self.assertIsInstance(source.last_h36m_by_id, dict)

    def test_runtime_error_reported(self):
        runtime = FakeRuntime()
        runtime.last_error = "device unplugged"
        source = self._source(runtime)
        source.estimate(_frame(), 1000.0)
        self.assertEqual(source.last_tracking["state"], "error")
        self.assertEqual(source.last_tracking["error"], "device unplugged")
```

- [ ] **Step 3.2:** Run: `python -m unittest tests.test_pose_backends_kinect -v`. Expected: FAIL (ImportError: AzureKinectPoseSource).

- [ ] **Step 3.3: Implement** — append to `backend/pose_backends/azure_kinect.py`:

```python
import sys as _sys

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in _sys.path:
    _sys.path.insert(0, _BACKEND_DIR)

from keypoint_mapping import k4abt32_to_h36m17, mirror_h36m17  # noqa: E402
from person_tracker import MultiPersonTrackRegistry, PersonTrack, bbox_area  # noqa: E402

# H36M-17 skeleton edges for the overlay.
_H36M_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),
]
# Overlay uses the 13 real joints (derived spine/neck omitted to reduce noise).
_H36M_DRAWN = (0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)


class AzureKinectPoseSource:
    """PoseSource protocol backed by K4ABT bodies cached on the KinectRuntime.

    The runtime's read() (frame source) refreshes last_bodies once per frame;
    estimate() only consumes that cache, so pose data and the displayed frame
    always come from the same capture."""

    def __init__(self, runtime, tracking_enabled: bool = False,
                 tracker_hold_seconds: float = 0.8,
                 tracking_selection: str = "auto_largest", **_ignored):
        self.runtime = runtime
        self.tracking_enabled = bool(tracking_enabled)
        self.tracking_selection = str(tracking_selection or "auto_largest")
        self.track_registry = MultiPersonTrackRegistry(hold_seconds=float(tracker_hold_seconds))
        self.last_pose_quality = 0.0
        self.last_pose_valid = False
        self.last_h36m_by_id = {} if self.tracking_enabled else None
        self.last_tracking = self._tracking_status("enabled" if self.tracking_enabled else "disabled")
        self._last_stable_tracks = []
        self._last_active_track = None
        self._overlay_points_by_id = {}

    # --- PoseSource protocol -------------------------------------------------
    def configure_tracking(self, enabled=None, selection=None, **_ignored):
        if enabled is not None and bool(enabled) != self.tracking_enabled:
            self.tracking_enabled = bool(enabled)
            self.reset_tracking()
        if selection is not None:
            self.tracking_selection = str(selection or "auto_largest")

    def reset_tracking(self):
        self.track_registry.reset()
        self._last_stable_tracks = []
        self._last_active_track = None
        self._overlay_points_by_id = {}
        self.last_h36m_by_id = {} if self.tracking_enabled else None
        self.last_tracking = self._tracking_status("enabled" if self.tracking_enabled else "disabled")

    def estimate(self, frame, timestamp_ms: float, draw: bool = True):
        self.last_pose_quality = 0.0
        self.last_pose_valid = False

        error = getattr(self.runtime, "last_error", None)
        if error:
            self._last_stable_tracks = []
            self._last_active_track = None
            self.last_h36m_by_id = {} if self.tracking_enabled else None
            self.last_tracking = self._tracking_status("error", error=str(error))
            return frame, None

        frame_h, frame_w = frame.shape[:2]
        native_size = tuple(getattr(self.runtime, "native_view_size", (frame_w, frame_h)))
        mirrored = bool(getattr(self.runtime, "mirrored", False))

        # Per-body: H36M skeleton (mirror-aware) + 2D points + registry track.
        raw_tracks, data_by_raw = [], {}
        for body in getattr(self.runtime, "last_bodies", []) or []:
            quality, valid = body_quality(body.joints[:, 3])
            h36m = k4abt32_to_h36m17(body.joints)
            if mirrored:
                h36m = mirror_h36m17(h36m)
            pts2d = transform_points_2d(body.joints2d, native_size,
                                        (frame_w, frame_h), mirrored)
            bbox = bbox_from_points(pts2d, (frame_w, frame_h))
            conf = float(np.clip(quality, 0.0, 1.0))
            raw_tracks.append(PersonTrack(track_id=int(body.body_id), bbox=bbox,
                                          confidence=conf))
            data_by_raw[int(body.body_id)] = (h36m, pts2d, quality, valid)

        now = float(timestamp_ms) / 1000.0
        stable_tracks = self.track_registry.update(raw_tracks, now)
        active, state = self.track_registry.choose_active(self.tracking_selection, frame.shape)
        self._last_stable_tracks = stable_tracks
        self._last_active_track = active

        h36m_by_id, overlay_by_id = {}, {}
        for track in stable_tracks:
            if track.state != "tracking" or track.raw_id is None:
                continue
            data = data_by_raw.get(int(track.raw_id))
            if data is None:
                continue
            h36m, pts2d, quality, valid = data
            if not valid:
                continue
            h36m_by_id[int(track.stable_id)] = h36m
            overlay_by_id[int(track.stable_id)] = self._points_dict(pts2d)
        self._overlay_points_by_id = overlay_by_id

        active_id = int(active.stable_id) if active is not None else None
        active_h36m = h36m_by_id.get(active_id) if active_id is not None else None
        if active_id is not None and int(active.raw_id or -1) in data_by_raw:
            _, _, quality, valid = data_by_raw[int(active.raw_id)]
            self.last_pose_quality = quality
            self.last_pose_valid = bool(valid and active_h36m is not None)

        self.last_tracking = self._tracking_status(state, tracks=stable_tracks, active=active)
        self.last_h36m_by_id = h36m_by_id if self.tracking_enabled else None

        if draw:
            self.draw_cached_overlay(frame)
        return frame, active_h36m

    def draw_cached_overlay(self, frame):
        active_id = self._last_active_track.stable_id if self._last_active_track is not None else None
        tracks = [t for t in self._last_stable_tracks if t.state != "lost"]
        for track in sorted(tracks, key=lambda item: bbox_area(item.bbox)):
            x1, y1, x2, y2 = [int(round(v)) for v in track.bbox]
            is_active = active_id is not None and int(track.stable_id) == int(active_id)
            color = (192, 211, 52) if is_active else (150, 142, 132)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if is_active else 1)
            label = f"id {track.stable_id}"
            if track.raw_id is not None and int(track.raw_id) != int(track.stable_id):
                label += f" raw {track.raw_id}"
            if is_active or track.state != "tracking":
                label += f" {track.state}"
            cv2.putText(frame, label, (max(0, x1), max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        for stable_id, points in sorted(self._overlay_points_by_id.items()):
            if stable_id == active_id:
                self._draw_points(frame, points, (0, 255, 255), (0, 255, 0))
            else:
                self._draw_points(frame, points, (192, 211, 52), (192, 211, 52))
        return frame

    def close(self):
        # The runtime/device lifecycle belongs to the frame source; nothing to
        # release here beyond caches.
        self._overlay_points_by_id = {}
        self._last_stable_tracks = []
        self._last_active_track = None

    # --- internals -----------------------------------------------------------
    @staticmethod
    def _points_dict(pts2d: np.ndarray) -> dict:
        h36m_pts = np.zeros((17, 2))
        # Project the same joints the 3D mapping uses so overlay matches data.
        from keypoint_mapping import (_K4_L_SH, _K4_L_EL, _K4_L_WR, _K4_R_SH,
                                      _K4_R_EL, _K4_R_WR, _K4_L_HIP, _K4_L_KNEE,
                                      _K4_L_ANK, _K4_R_HIP, _K4_R_KNEE,
                                      _K4_R_ANK, _K4_NOSE)
        h36m_pts[0] = (pts2d[_K4_L_HIP] + pts2d[_K4_R_HIP]) / 2.0
        h36m_pts[1], h36m_pts[2], h36m_pts[3] = pts2d[_K4_R_HIP], pts2d[_K4_R_KNEE], pts2d[_K4_R_ANK]
        h36m_pts[4], h36m_pts[5], h36m_pts[6] = pts2d[_K4_L_HIP], pts2d[_K4_L_KNEE], pts2d[_K4_L_ANK]
        sh_mid = (pts2d[_K4_L_SH] + pts2d[_K4_R_SH]) / 2.0
        h36m_pts[8], h36m_pts[10] = sh_mid, pts2d[_K4_NOSE]
        h36m_pts[7] = (h36m_pts[0] + sh_mid) / 2.0
        h36m_pts[9] = (sh_mid + pts2d[_K4_NOSE]) / 2.0
        h36m_pts[11], h36m_pts[12], h36m_pts[13] = pts2d[_K4_L_SH], pts2d[_K4_L_EL], pts2d[_K4_L_WR]
        h36m_pts[14], h36m_pts[15], h36m_pts[16] = pts2d[_K4_R_SH], pts2d[_K4_R_EL], pts2d[_K4_R_WR]
        return {i: (int(round(h36m_pts[i, 0])), int(round(h36m_pts[i, 1])))
                for i in _H36M_DRAWN}

    @staticmethod
    def _draw_points(frame, points, joint_color, line_color):
        h, w = frame.shape[:2]
        for cx, cy in points.values():
            if -20 <= cx <= w + 20 and -20 <= cy <= h + 20:
                cv2.circle(frame, (cx, cy), 4, joint_color, -1)
        for a, b in _H36M_CONNECTIONS:
            if a in points and b in points:
                cv2.line(frame, points[a], points[b], line_color, 2)

    def _tracking_status(self, state, tracks=None, active=None, error=None):
        tracks = tracks or []
        stable_id = int(active.stable_id) if active is not None else None
        raw_id = int(active.raw_id) if active is not None and active.raw_id is not None else None
        return {
            "enabled": bool(self.tracking_enabled),
            "state": state,
            "count": len([t for t in tracks if getattr(t, "state", "") == "tracking"]),
            "locked_id": stable_id,
            "stable_id": stable_id,
            "raw_id": raw_id,
            "active_id": stable_id if active is not None else None,
            "selection": self.tracking_selection,
            "bbox": [float(v) for v in active.bbox] if active is not None else None,
            "tracks": [
                {
                    "stable_id": int(t.stable_id),
                    "raw_id": int(t.raw_id) if t.raw_id is not None else None,
                    "state": t.state,
                    "confidence": float(t.confidence),
                    "bbox": [float(v) for v in t.bbox],
                    "active": bool(active is not None and t.stable_id == active.stable_id),
                }
                for t in tracks
            ],
            "error": error,
        }
```

Note: `_points_dict` imports the `_K4_*` index constants from `keypoint_mapping` — they were defined there in Task 1. Keep that single source of truth (do not re-declare indices here).

- [ ] **Step 3.4:** Run: `python -m unittest tests.test_pose_backends_kinect -v`. Expected: PASS. Note: `test_single_body_disabled_tracking` intentionally asserts `active_id` is set while `enabled` is False — that is how PoseEngine's change-of-dancer metrics reset works for Kinect single mode (documented deviation from the MediaPipe disabled-status shape).

- [ ] **Step 3.5:** Run full suite: `python -m unittest discover -s tests`. Expected: baseline + new, all green.

- [ ] **Step 3.6:** Commit: `git add backend/pose_backends/azure_kinect.py backend/tests/test_pose_backends_kinect.py; git commit -m "Add AzureKinectPoseSource: K4ABT bodies -> registry -> per-person H36M"`

---

### Task 4: KinectRuntime (hardware wrapper, guarded)

**Files:**
- Modify: `backend/pose_backends/azure_kinect.py` (append)
- Test: `backend/tests/test_pose_backends_kinect.py` (append — guard + singleton semantics only; capture paths are hardware-verified in Task 8)

- [ ] **Step 4.1: Write failing tests** — append:

```python
from pose_backends.azure_kinect import KinectError, KinectRuntime, _guarded


class GuardedCallTests(unittest.TestCase):
    def test_converts_system_exit(self):
        def boom():
            raise SystemExit(1)   # pykinect VERIFY() does this
        with self.assertRaises(KinectError):
            _guarded("enqueue", boom)

    def test_converts_exception(self):
        def boom():
            raise RuntimeError("usb reset")
        with self.assertRaises(KinectError):
            _guarded("capture", boom)

    def test_passes_result(self):
        self.assertEqual(_guarded("ok", lambda: 42), 42)


class RuntimeOwnershipTests(unittest.TestCase):
    def test_release_requires_owner(self):
        runtime = KinectRuntime()
        runtime._opened = True
        runtime._owner = 7
        closed = []
        runtime._close_device = lambda: closed.append(True)
        runtime.release(owner=3)      # wrong owner: no-op
        self.assertTrue(runtime._opened)
        runtime.release(owner=7)
        self.assertFalse(runtime._opened)
        self.assertEqual(closed, [True])

    def test_read_when_closed_reports_error(self):
        runtime = KinectRuntime()
        ok, frame = runtime.read()
        self.assertFalse(ok)
        self.assertIsNone(frame)
        self.assertTrue(runtime.last_error)
```

- [ ] **Step 4.2:** Run: `python -m unittest tests.test_pose_backends_kinect -v`. Expected: FAIL (ImportError: KinectRuntime).

- [ ] **Step 4.3: Implement** — append to `backend/pose_backends/azure_kinect.py`:

```python
# --- Hardware runtime --------------------------------------------------------

_SDK_DIRS = (
    r"C:\Program Files\Azure Kinect SDK v1.4.2",
    r"C:\Program Files\Azure Kinect SDK v1.4.1",
)
_BT_SDK_DIR = r"C:\Program Files\Azure Kinect Body Tracking SDK"


def azure_kinect_available() -> bool:
    """Cheap probe for the backend dropdown: Windows + SDKs + python binding.
    Does NOT import pykinect (import loads DLLs; keep the probe instant)."""
    if os.name != "nt":
        return False
    import importlib.util
    if importlib.util.find_spec("pykinect_azure") is None:
        return False
    return any(os.path.isdir(d) for d in _SDK_DIRS) and os.path.isdir(_BT_SDK_DIR)


def _guarded(what: str, fn, *args, **kwargs):
    """Run a pykinect call, converting BOTH exceptions and sys.exit into
    KinectError. pykinect's VERIFY() calls sys.exit(1) on sensor errors —
    uncaught, an unplugged cable would kill the whole viewer process."""
    try:
        return fn(*args, **kwargs)
    except SystemExit as exc:
        raise KinectError(f"Kinect {what} failed (SDK aborted)") from exc
    except KinectError:
        raise
    except Exception as exc:
        raise KinectError(f"Kinect {what} failed: {exc}") from exc


class KinectRuntime:
    """Owns the k4a device + body tracker. read() = capture + track + cache.

    One module-level instance (get_runtime()); the device is opened by the
    frame source (acquire) per stream session and closed on release, with the
    same owner-token semantics as osc_viewer's camera globals."""

    POP_TIMEOUT_MS = 350

    def __init__(self):
        self._lock = threading.Lock()
        self._opened = False
        self._owner = None
        self._device = None
        self._tracker = None
        self.view = "color"
        self.mirrored = False
        self.native_view_size = (1280, 720)
        self.last_bodies: list[KinectBody] = []
        self.last_error = None
        self._calibration_type_color = None
        self._calibration_type_depth = None

    # --- lifecycle -----------------------------------------------------------
    def acquire(self, owner, view: str = "color", mirrored: bool = False):
        """(Re)open the device for a stream session (takes over like
        open_camera does)."""
        with self._lock:
            self._close_device()
            self.view = view if view in KINECT_VIEWS else "color"
            self.mirrored = bool(mirrored)
            self._owner = owner
            self._open_device()
            self._opened = True
            self.last_error = None
            return self

    def release(self, owner=None, force: bool = False):
        with self._lock:
            if not self._opened:
                return
            if not force and owner is not None and self._owner != owner:
                return
            self._close_device()
            self._opened = False
            self._owner = None

    def reopen(self):
        """Close + reopen after read failures (mirrors reopen_live_camera)."""
        with self._lock:
            owner, view, mirrored = self._owner, self.view, self.mirrored
            self._close_device()
            time.sleep(0.35)
            self.view = view
            self.mirrored = mirrored
            self._owner = owner
            self._open_device()
            self._opened = True
            self.last_error = None

    def _open_device(self):
        import pykinect_azure as pykinect
        from pykinect_azure.k4abt import _k4abtTypes as _bt

        _guarded("library init", pykinect.initialize_libraries, track_body=True)
        # initialize_libraries resets processing_mode, so set these AFTER it.
        _bt.k4abt_tracker_default_configuration.processing_mode = \
            _bt.K4ABT_TRACKER_PROCESSING_MODE_GPU_DIRECTML
        _bt.k4abt_tracker_default_configuration.gpu_device_id = \
            int(os.getenv("FIELD_KINECT_GPU", "1"))

        config = pykinect.default_configuration
        config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
        config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
        config.synchronized_images_only = True

        self._device = _guarded("device open", pykinect.start_device, config=config)
        model = _bt.K4ABT_LITE_MODEL if os.getenv("FIELD_KINECT_MODEL", "full") == "lite" \
            else _bt.K4ABT_DEFAULT_MODEL
        self._tracker = _guarded("tracker create", pykinect.start_body_tracker, model)
        self._calibration_type_color = pykinect.K4A_CALIBRATION_TYPE_COLOR
        self._calibration_type_depth = pykinect.K4A_CALIBRATION_TYPE_DEPTH

    def _close_device(self):
        device, self._device = self._device, None
        self._tracker = None
        if device is not None:
            try:
                device.close()
            except (Exception, SystemExit):
                pass
        self.last_bodies = []

    # --- frame source protocol ----------------------------------------------
    def read(self):
        """Capture one frame, run body tracking, cache bodies, return the view
        image (BGR). On failure returns (False, None) and sets last_error."""
        with self._lock:
            if not self._opened or self._device is None:
                self.last_error = "Kinect not open"
                return False, None
            try:
                capture = _guarded("capture", self._device.update)
                body_frame = _guarded("body tracking", self._tracker.update,
                                      timeout_in_ms=self.POP_TIMEOUT_MS)
                image = self._render_view(capture, body_frame)
                if image is None:
                    return False, None
                self.last_error = None
                return True, image
            except KinectError as exc:
                self.last_error = str(exc)
                self.last_bodies = []
                return False, None

    def describe(self) -> str:
        return "Azure Kinect"

    # --- internals -----------------------------------------------------------
    def _render_view(self, capture, body_frame):
        if self.view == "depth":
            ret, image = capture.get_colored_depth_image()
            dest_camera = self._calibration_type_depth
        else:
            ret, image = capture.get_color_image()
            dest_camera = self._calibration_type_color
        if not ret or image is None:
            return None
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        x_off = y_off = 0
        if self.view == "depth":
            image, x_off, y_off = pad_to_aspect(image, 16, 9)
        self.native_view_size = (image.shape[1], image.shape[0])
        self._extract_bodies(body_frame, dest_camera, x_off, y_off)
        return image

    def _extract_bodies(self, body_frame, dest_camera, x_off, y_off):
        bodies = []
        n = _guarded("body count", body_frame.get_num_bodies)
        for i in range(int(n)):
            body_id = int(_guarded("body id", body_frame.get_body_id, i))
            raw = np.asarray(_guarded("body joints", lambda: body_frame.get_body(i).numpy()),
                             dtype=float)
            joints = raw[:, [0, 1, 2, 7]]          # x, y, z, confidence
            raw2d = np.asarray(
                _guarded("body 2d", lambda: body_frame.get_body2d(i, dest_camera).numpy()),
                dtype=float)
            joints2d = raw2d[:, :2] + np.array([x_off, y_off], dtype=float)
            bodies.append(KinectBody(body_id=body_id, joints=joints, joints2d=joints2d))
        self.last_bodies = bodies


_runtime: KinectRuntime | None = None
_runtime_lock = threading.Lock()


def get_runtime() -> KinectRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = KinectRuntime()
        return _runtime
```

- [ ] **Step 4.4:** Run: `python -m unittest tests.test_pose_backends_kinect -v`. Expected: PASS (guard + ownership tests; nothing here touches hardware).

- [ ] **Step 4.5:** Commit: `git add backend/pose_backends/azure_kinect.py backend/tests/test_pose_backends_kinect.py; git commit -m "Add KinectRuntime: guarded k4a device/tracker wrapper with owner semantics"`

---

### Task 5: FrameSource protocol + OpenCV delegation

**Files:**
- Create: `backend/frame_sources.py`
- Test: `backend/tests/test_frame_sources.py` (new)

- [ ] **Step 5.1: Write failing tests** — create `backend/tests/test_frame_sources.py`:

```python
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frame_sources import KinectFrameSource, OpenCVFrameSource


class FakeCap:
    def __init__(self):
        self.frames = [np.zeros((4, 4, 3), dtype=np.uint8)]

    def read(self):
        return True, self.frames[0]


class OpenCVDelegationTests(unittest.TestCase):
    def _source(self, log):
        cap = FakeCap()
        return OpenCVFrameSource(
            index=2, owner=9,
            open_fn=lambda index, owner: log.append(("open", index, owner)) or cap,
            read_fn=lambda c, owner: log.append(("read", owner)) or c.read(),
            release_fn=lambda owner=None, force=False: log.append(("release", owner)),
        )

    def test_open_read_release(self):
        log = []
        source = self._source(log)
        source.open()
        ok, frame = source.read()
        self.assertTrue(ok)
        source.release()
        self.assertEqual([entry[0] for entry in log], ["open", "read", "release"])
        self.assertEqual(log[0], ("open", 2, 9))
        self.assertEqual(log[1], ("read", 9))
        self.assertEqual(log[2], ("release", 9))

    def test_reopen_releases_then_opens(self):
        log = []
        source = self._source(log)
        source.open()
        source.reopen(sleep_seconds=0.0)
        self.assertEqual([entry[0] for entry in log], ["open", "release", "open"])

    def test_describe(self):
        self.assertEqual(self._source([]).describe(), "Camera 2")


class KinectFrameSourceTests(unittest.TestCase):
    def test_delegates_to_runtime(self):
        class FakeRuntime:
            def __init__(self):
                self.calls = []
            def read(self):
                self.calls.append("read")
                return True, np.zeros((2, 2, 3), dtype=np.uint8)
            def reopen(self):
                self.calls.append("reopen")
            def release(self, owner=None, force=False):
                self.calls.append(("release", owner))
            def describe(self):
                return "Azure Kinect"

        runtime = FakeRuntime()
        source = KinectFrameSource(runtime, owner=5)
        ok, _ = source.read()
        self.assertTrue(ok)
        source.reopen()
        source.release()
        self.assertEqual(runtime.calls, ["read", "reopen", ("release", 5)])
        self.assertEqual(source.describe(), "Azure Kinect")
```

- [ ] **Step 5.2:** Run: `python -m unittest tests.test_frame_sources -v`. Expected: FAIL (ModuleNotFoundError: frame_sources).

- [ ] **Step 5.3: Implement** — create `backend/frame_sources.py`:

```python
"""Frame sources for the live stream loop.

stream_live() only needs read()/reopen()/release()/describe(). The OpenCV
implementation DELEGATES to osc_viewer's existing module-level camera
functions (global cap + camera_lock + owner token) instead of moving them —
that machinery encodes hard-won race fixes (see MAINTENANCE.md §5) and its
behavior must not change. The Kinect implementation wraps KinectRuntime.
"""
from __future__ import annotations

import time


class OpenCVFrameSource:
    """cv2.VideoCapture via osc_viewer's open/read/release functions."""

    def __init__(self, index: int, owner, open_fn, read_fn, release_fn):
        self._index = int(index)
        self._owner = owner
        self._open_fn = open_fn
        self._read_fn = read_fn
        self._release_fn = release_fn
        self._cap = None

    def open(self):
        self._cap = self._open_fn(self._index, self._owner)
        return self

    def read(self):
        if self._cap is None:
            return False, None
        return self._read_fn(self._cap, self._owner)

    def reopen(self, sleep_seconds: float = 0.35):
        self._release_fn(self._owner)
        if sleep_seconds:
            time.sleep(sleep_seconds)
        self._cap = self._open_fn(self._index, self._owner)

    def release(self):
        self._release_fn(self._owner)
        self._cap = None

    def describe(self) -> str:
        return f"Camera {self._index}"


class KinectFrameSource:
    """Azure Kinect via KinectRuntime (already acquired by the caller)."""

    def __init__(self, runtime, owner):
        self._runtime = runtime
        self._owner = owner

    def open(self):
        return self

    def read(self):
        return self._runtime.read()

    def reopen(self, sleep_seconds: float = 0.35):
        self._runtime.reopen()

    def release(self):
        self._runtime.release(owner=self._owner)

    def describe(self) -> str:
        return self._runtime.describe()
```

- [ ] **Step 5.4:** Run: `python -m unittest tests.test_frame_sources -v`. Expected: PASS.

- [ ] **Step 5.5:** Commit: `git add backend/frame_sources.py backend/tests/test_frame_sources.py; git commit -m "Add FrameSource protocol: OpenCV delegation + Kinect wrapper"`

---

### Task 6: PoseEngine backend registration

**Files:**
- Modify: `backend/pose_engine.py:14` (VALID_BACKENDS) and `:46-63` (_make_source)
- Test: `backend/tests/test_pose_engine.py` (append)

- [ ] **Step 6.1: Write failing test** — append to `backend/tests/test_pose_engine.py`:

```python
from pose_engine import VALID_BACKENDS


class KinectBackendRegistrationTests(unittest.TestCase):
    def test_azure_kinect_is_a_valid_backend(self):
        self.assertIn("azure_kinect", VALID_BACKENDS)
```

- [ ] **Step 6.2:** Run: `python -m unittest tests.test_pose_engine -v`. Expected: FAIL (azure_kinect not in tuple).

- [ ] **Step 6.3: Implement** — in `backend/pose_engine.py` change line 14:

```python
VALID_BACKENDS = ("mediapipe", "rtmpose3d", "azure_kinect")
```

and in `_make_source` insert BEFORE the `if backend == "rtmpose3d":` block:

```python
        if backend == "azure_kinect":
            try:
                from pose_backends.azure_kinect import AzureKinectPoseSource, get_runtime
                return AzureKinectPoseSource(
                    get_runtime(),
                    tracking_enabled=self.tracking_enabled,
                    tracking_selection=self.tracking_selection,
                )
            except Exception as exc:
                # pykinect/SDK missing -> degrade to MediaPipe, same pattern as
                # the rtmpose3d fallback below.
                print(f"[PoseEngine] Azure Kinect unavailable ({exc}); falling back to MediaPipe.")
                self.backend_name = "mediapipe"
```

(`backend/pose_backends/azure_kinect.py` self-inserts the backend dir into `sys.path`, and `pose_engine.py` runs from `backend/`, so the import resolves in both the server and the tests.)

- [ ] **Step 6.4:** Run: `python -m unittest tests.test_pose_engine -v` then the full suite `python -m unittest discover -s tests`. Expected: PASS / all green.

- [ ] **Step 6.5:** Commit: `git add backend/pose_engine.py backend/tests/test_pose_engine.py; git commit -m "Register azure_kinect backend in PoseEngine with MediaPipe fallback"`

---

### Task 7: osc_viewer wiring (frame-source swap, availability, kinect_view UI)

**Files:**
- Modify: `backend/osc_viewer.py` — availability (`~451`), source_state (`~91-107`), `/api/apply` (`~1279+`), `stream_live()` (`755-852`), delete `reopen_live_camera` (`269-273`), HTML (`~2380`), JS (`~3563`)

This is the minefield file (MAINTENANCE.md §5): no logic moves, only the listed insertions/replacements. Viewer has no automated endpoint tests — verification is the full suite (import-level) + Task 8 hardware pass.

- [ ] **Step 7.1: Availability probe + backend list** — after `rtmpose3d_selectable()` (line ~448) add:

```python
def azure_kinect_selectable() -> bool:
    try:
        from pose_backends.azure_kinect import azure_kinect_available
        return azure_kinect_available()
    except Exception:
        return False
```

and in `available_pose_backends()` append before `return backends`:

```python
    if azure_kinect_selectable():
        backends.append({
            "id": "azure_kinect",
            "label": "Azure Kinect",
            "description": "depth 3D, multi-person",
        })
```

- [ ] **Step 7.2: State + apply param** — in the `source_state = {` dict (around line 91, next to `"camera_index": 0,`) add:

```python
    "kinect_view": "color",
```

In the `/api/apply` signature (after `pose_backend: str = Form("mediapipe"),` line ~1292) add:

```python
    kinect_view: str = Form("color"),
```

and after the `source_state["pose_backend"] = ...` line (~1321) add:

```python
    source_state["kinect_view"] = kinect_view if kinect_view in ("color", "depth") else "color"
```

- [ ] **Step 7.3: Frame source factory** — near `open_camera` (after `reopen_live_camera`, line ~274) add, and delete the now-unused `reopen_live_camera` (its 0.35 s backoff moved into `OpenCVFrameSource.reopen`):

```python
def make_live_frame_source(session_id: int):
    """Build the frame source for a live stream session. Kinect brings its own
    capture (depth + body tracking ride the same device); everything else is
    the existing cv2 path via delegation."""
    if source_state.get("pose_backend") == "azure_kinect":
        from pose_backends.azure_kinect import get_runtime
        runtime = get_runtime().acquire(
            owner=session_id,
            view=str(source_state.get("kinect_view", "color")),
            mirrored=bool(source_state.get("mirror_live")),
        )
        return KinectFrameSource(runtime, owner=session_id)
    return OpenCVFrameSource(
        index=int(source_state["camera_index"]),
        owner=session_id,
        open_fn=open_camera,
        read_fn=read_camera_frame,
        release_fn=release_camera,
    ).open()
```

Add to the imports at the top of the file: `from frame_sources import KinectFrameSource, OpenCVFrameSource`.

- [ ] **Step 7.4: stream_live swap** — replace the open block (lines 758-763):

```python
    try:
        frame_source = await asyncio.to_thread(make_live_frame_source, session_id)
    except Exception as exc:
        processing_state["error"] = f"Camera unavailable: {exc}"
        processing_state["running"] = False
        return
```

Replace the read call (line 778): `ok, frame = await asyncio.to_thread(frame_source.read)`

Replace the reconnect block (lines 780-791):

```python
            if not ok:
                missed_frames += 1
                if missed_frames >= 5:
                    try:
                        processing_state["error"] = f"{frame_source.describe()} frame dropped; reconnecting"
                        await asyncio.to_thread(frame_source.reopen)
                        missed_frames = 0
                    except Exception as exc:
                        processing_state["error"] = f"{frame_source.describe()} reconnect failed: {exc}"
                        await asyncio.sleep(0.75)
                else:
                    processing_state["error"] = f"{frame_source.describe()} frame not available"
                    await asyncio.sleep(0.08)
                continue
```

Replace `release_camera(session_id)` in the `finally` (line 851) with `frame_source.release()`.

Everything else in the loop (mirror, resize, analysis cadence, encode) stays byte-identical. `stream_live_preview` (line 855+) keeps using `open_camera` — untouched.

- [ ] **Step 7.5: HTML** — after the `poseBackend` label block (line ~2384, right after its closing `</label>`) insert:

```html
              <label id="kinectViewRow" class="model-row hidden">Kinect view
                <select id="kinectView" name="kinect_view">
                  <option value="color">Color</option>
                  <option value="depth">Depth</option>
                </select>
              </label>
```

- [ ] **Step 7.6: JS** — extend `syncPoseBackends(payload)` (line ~3563). After `select.value = ...` add:

```javascript
      const kinectRow = document.getElementById('kinectViewRow');
      const kinectSelect = document.getElementById('kinectView');
      kinectRow.classList.toggle('hidden', select.value !== 'azure_kinect');
      if (document.activeElement !== kinectSelect) {
        kinectSelect.value = payload?.source?.kinect_view || 'color';
      }
```

and register (next to the other listeners, where `poseBackend` change handling lives — search for `poseBackend` listeners; if none exists, add after the `syncPoseBackends` definition):

```javascript
    document.getElementById('poseBackend').addEventListener('change', () => {
      const isKinect = document.getElementById('poseBackend').value === 'azure_kinect';
      document.getElementById('kinectViewRow').classList.toggle('hidden', !isKinect);
    });
```

The apply fetch uses `new FormData(form)` (line 2937) so the named select submits automatically; `state_payload()` returns `dict(source_state)` so `kinect_view` syncs automatically. The `activeElement` guard satisfies minefield #4.

- [ ] **Step 7.7:** Sanity: `python -c "import sys; sys.path.insert(0, 'D:/Github/field-realtime-dance/backend'); import osc_viewer"` — expect clean import (run from anywhere; module-level code must not touch hardware). Then full suite: `python -m unittest discover -s tests` — all green.

- [ ] **Step 7.8:** Commit: `git add backend/osc_viewer.py; git commit -m "Wire Azure Kinect backend: frame-source swap, availability probe, kinect_view UI"`

---

### Task 8: Full verification (suite + hardware) and docs

**Files:**
- Modify: `MAINTENANCE.md` (§7.5 status note), `HANDOFF.md` (usage snippet)

- [ ] **Step 8.1:** Full suite one more time: `python -m unittest discover -s tests`. Expected: baseline + ~20 new, all green, 1 skip.

- [ ] **Step 8.2: Hardware smoke (device plugged in):** run `python backend/osc_viewer.py`, open http://127.0.0.1:9100 and verify each:
  1. Backend dropdown shows "Azure Kinect (depth 3D, multi-person)"; select + Enter → stream starts, skeleton overlay on color view, fps overlay ≥ 25
  2. Kinect view dropdown appears only for the Kinect backend; switch to Depth → colorized depth (16:9 letterboxed, not stretched), overlay aligned on the person
  3. Mirror checkbox on: image flips AND overlay stays glued to the body; `python backend/osc_monitor.py` shows plausible `/field/1/*` values
  4. Stable ID on + two people: `/field/1/*` and `/field/2/*` both stream; one person ducks behind the other and returns → same ids (registry re-id)
  5. Pull the Kinect USB mid-stream → UI shows the reconnect error, process stays alive; replug → stream recovers within a few seconds
  6. Lights off (or cover the RGB lens): depth view + skeleton + OSC keep working
- [ ] **Step 8.3:** Fix anything found (each fix: failing test where representable → fix → suite green → commit).
- [ ] **Step 8.4: Docs** — MAINTENANCE.md §7.5: prepend `**狀態:已實作於 feat/kinect-pose-source(2026-07-30)**,用法見 HANDOFF.md;` to the section. HANDOFF.md: add a short "Azure Kinect backend" subsection: requirements (Sensor SDK 1.4.2 + Body Tracking 1.1.2 + `pip install pykinect_azure`, Windows only), env vars (`FIELD_KINECT_GPU` default 1 — set 0 on single-GPU machines; `FIELD_KINECT_MODEL` full|lite), NFOV range 0.5–3.9 m, note that camera dropdown/preview are ignored for this backend and that the color camera must not be simultaneously opened as a UVC webcam.
- [ ] **Step 8.5:** Commit: `git add MAINTENANCE.md HANDOFF.md; git commit -m "Document Azure Kinect backend status + usage"`. Push branch: `git push -u origin feat/kinect-pose-source`.

---

## Self-review notes (done at plan time)

- Spec §5a-§5d, §6, §7 all map to Tasks 1-8; spec §2 user decisions land in Task 4 (view render), Task 3 (mirror), Task 7 (UI).
- Type check: `PersonTrack(track_id=...)` matches `person_tracker.py:16` (field is `track_id`, not `raw_id`); registry `update(raw_tracks, now)` + `choose_active(selection, frame_shape)` match `person_tracker.py:297+`; `Body.numpy()` column layout verified on hardware 2026-07-30.
- Known intentional deviation: Kinect single-person `last_tracking.active_id` is set while `enabled=False` (drives PoseEngine's change-of-dancer reset); MediaPipe leaves it None there.
