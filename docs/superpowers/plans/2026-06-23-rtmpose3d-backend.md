# RTMPose3D (RTMW3D) Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Steps use checkbox (`- [ ]`) syntax. Run tests from `backend/` with
> `python -m unittest discover -s tests` (running elsewhere picks up
> ultralytics' own tests in site-packages).

**Goal:** Add a selectable `rtmpose3d` pose backend (rtmlib RTMW3D-x, ONNX) that
feeds the existing NCKU 9-metric engine, single-person first, without disturbing
the MediaPipe/YOLO backends.

**Architecture:** YOLO+ByteTrack (crops + track_id) → per-crop RTMW3D-x (133×3)
→ body-17 → standard H36M-17 (with z) → `PersonPose(is_3d=True)` → existing
SlotBinder / DanceMetricsEngine / MultiSlotOSC.

**Tech stack:** rtmlib (RTMW3D-x ONNX) + onnxruntime(-gpu). No MMPose/nvcc.

Spec: `docs/superpowers/specs/2026-06-23-rtmpose3d-backend-design.md`.

---

### Task 1: Gate 1 feasibility spike — RTMW3D-x real-time on 4080 (KILL-SWITCH)

**Not TDD — this is an empirical spike. It gates everything below.**

**Environment notes (the spike must first resolve these):**
- The field repo currently has NO `rtmlib`/`onnxruntime` in `requirements.txt`.
  Determine which Python interpreter runs the viewer (`osc_viewer.py`) and
  install into THAT environment.
- onnxruntime-gpu needs CUDA runtime DLLs on PATH. In the sibling ai-motion
  project this is solved by prepending PyTorch's CUDA DLL dir to PATH at runtime;
  ultralytics (already installed here) ships PyTorch+CUDA, so reuse that DLL dir
  if the CUDA EP fails to load.
- A test clip with a visible dancer exists under `backend/recordings/` or the
  ai-motion `data/test/` videos; fall back to webcam or a synthetic person crop
  if none is usable.

- [ ] **Step 1:** `pip install rtmlib onnxruntime-gpu` into the viewer's env.
- [ ] **Step 2:** Write a throwaway script `backend/_spike_rtmw3d.py` that loads
  RTMW3D-x via rtmlib (download URL or HF `Soykaf/RTMW3D-x`) with the onnxruntime
  CUDA execution provider, runs it on ~300 frames of a test clip, and prints:
  mean FPS, whether CUDA EP is active (not CPU fallback), and output shape.
- [ ] **Step 3:** Run it. Record single-person FPS on the 4080.
- [ ] **Step 4 (decision):** If FPS ≥ the viewer's single-dancer analysis rate
  (~12–30 fps target) on GPU → **PASS**, proceed to Task 2. If it cannot reach
  real-time even for one person, or the CUDA EP won't load → **STOP and escalate**
  (pivot to ROMP per the spec fallback). Report the number either way.
- [ ] **Step 5:** Delete `_spike_rtmw3d.py` (or keep under a clearly-temp name);
  add `rtmlib` + `onnxruntime-gpu` to `requirements.txt` only if PASS.

### Task 2: 3D keypoint → H36M-17 adapter

**Files:**
- Modify: `backend/keypoint_mapping.py`
- Test: `backend/tests/test_keypoint_mapping_3d.py`

- [ ] **Step 1: Write the failing test.** Synthetic 17×3 COCO-body input where
  each joint's coords encode its index; assert standard H36M-17 placement AND
  that z is preserved (not zeroed):

```python
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keypoint_mapping import coco17_to_h36m17_3d

NOSE=0; L_SH,R_SH=5,6; L_EL,R_EL=7,8; L_WR,R_WR=9,10
L_HIP,R_HIP=11,12; L_KNEE,R_KNEE=13,14; L_ANK,R_ANK=15,16

def _kpts3d():
    c = np.zeros((17, 3))
    for i in range(17):
        c[i] = (i*10, i*10+1, i*10+2)   # distinct z per joint
    return c

class TestCoco3D(unittest.TestCase):
    def test_arms_and_z_preserved(self):
        c = _kpts3d(); out = coco17_to_h36m17_3d(c)
        self.assertEqual(out.shape, (17, 3))
        np.testing.assert_allclose(out[13], c[L_WR])   # left wrist -> 13, z kept
        np.testing.assert_allclose(out[16], c[R_WR])   # right wrist -> 16
        np.testing.assert_allclose(out[11], c[L_SH])
        np.testing.assert_allclose(out[14], c[R_SH])
        np.testing.assert_allclose(out[10], c[NOSE])   # head
        self.assertNotEqual(out[13][2], 0.0)           # z not zeroed

    def test_spine_chain(self):
        c = _kpts3d(); out = coco17_to_h36m17_3d(c)
        pelvis = (c[L_HIP]+c[R_HIP])/2; thorax = (c[L_SH]+c[R_SH])/2
        np.testing.assert_allclose(out[0], pelvis)
        np.testing.assert_allclose(out[8], thorax)
        np.testing.assert_allclose(out[7], (pelvis+thorax)/2)
        np.testing.assert_allclose(out[9], (thorax+c[NOSE])/2)
```

- [ ] **Step 2:** Run it, verify it fails (`coco17_to_h36m17_3d` undefined).
- [ ] **Step 3: Implement.** Add to `keypoint_mapping.py` — same layout as the
  fixed `coco17_to_h36m17` but keep z from input:

```python
def coco17_to_h36m17_3d(kpts: np.ndarray) -> np.ndarray:
    """Map COCO-17 body keypoints WITH z (3D) to standard H36M-17.
    Identical layout to coco17_to_h36m17 but z is preserved, not zeroed."""
    def p(i):
        return np.asarray(kpts[i], dtype=float)[:3]
    l_hip, r_hip = p(_L_HIP), p(_R_HIP)
    pelvis = (l_hip + r_hip) / 2.0
    l_sh, r_sh = p(_L_SH), p(_R_SH)
    thorax = (l_sh + r_sh) / 2.0
    head = p(_NOSE)
    spine = (pelvis + thorax) / 2.0
    neck = (thorax + head) / 2.0
    j = np.zeros((17, 3))
    j[0] = pelvis
    j[1] = r_hip;  j[2] = p(_R_KNEE);  j[3] = p(_R_ANK)
    j[4] = l_hip;  j[5] = p(_L_KNEE);  j[6] = p(_L_ANK)
    j[7] = spine;  j[8] = thorax;      j[9] = neck;       j[10] = head
    j[11] = l_sh;  j[12] = p(_L_EL);   j[13] = p(_L_WR)
    j[14] = r_sh;  j[15] = p(_R_EL);   j[16] = p(_R_WR)
    return j
```

- [ ] **Step 4:** Run the new test + full suite from `backend/`. All green.
- [ ] **Step 5:** Commit.

### Task 3: RTMPose3DBackend class

**Files:**
- Create: `backend/pose_backends/rtmpose3d_backend.py`
- Test: `backend/tests/test_rtmpose3d_backend.py`

- [ ] **Step 1: Write the failing test** for a pure adaptation helper that does
  not require rtmlib loaded — factor the per-detection assembly into
  `personposes_from_rtmw3d(kpts133_list, bboxes, track_ids)` and test it with
  synthetic data (mirrors `test_yolo_backend.py`): asserts 17×3 h36m17,
  `is_3d is True`, track_ids passed through, body-17 taken from `[:17]`.
- [ ] **Step 2:** Run, verify fail.
- [ ] **Step 3: Implement.** `RTMPose3DBackend` implementing `PoseBackend`:
  - Lazy `from rtmlib import RTMW3D` (and the detector/tracker) inside `__init__`.
  - `estimate(frame, ts)`: get person boxes + track_ids from the existing
    YOLO+ByteTrack path (reuse `YOLO26Backend`'s tracker or rtmlib's bundled
    detector for the single-person spike), run RTMW3D per crop, build
    PersonPose via `personposes_from_rtmw3d` + `coco17_to_h36m17_3d` (apply the
    scale factor from Task 5).
  - `close()` releases the model.
  - Keep the live-model code path out of the unit-tested helper.
- [ ] **Step 4:** Run helper test + full suite. Green.
- [ ] **Step 5:** Commit.

### Task 4: Factory wiring

**Files:**
- Modify: `backend/pose_backends/factory.py`
- Test: `backend/tests/test_factory_rtmpose3d.py` (monkeypatch the lazy import so
  the test does not need rtmlib/weights).

- [ ] **Step 1:** Failing test: `make_backend("rtmpose3d")` constructs
  `RTMPose3DBackend` (patch its heavy `__init__`/import).
- [ ] **Step 2:** Run, verify fail.
- [ ] **Step 3:** Add the `rtmpose3d` branch to `make_backend`.
- [ ] **Step 4:** Run + full suite. Green.
- [ ] **Step 5:** Commit.

### Task 5: Gate 2 integration + scale calibration (INTEGRATION — full review)

**Empirical. Run the backend single-person through the real pipeline.**

- [ ] **Step 1:** Launch the viewer with `FIELD_POSE_BACKEND=rtmpose3d` on a
  single-dancer clip/webcam.
- [ ] **Step 2:** Confirm the 9 metrics are finite, non-degenerate, and the OSC
  stream flows for one slot.
- [ ] **Step 3:** Compare metric magnitudes against the MediaPipe backend on the
  same clip; pick a scale constant in the adapter so expansion/height/sway land
  in the same order of magnitude. Document the chosen scale.
- [ ] **Step 4:** Note FPS in the live pipeline (single person). Record results
  in this plan file.
- [ ] **Step 5:** Commit (code + notes).
