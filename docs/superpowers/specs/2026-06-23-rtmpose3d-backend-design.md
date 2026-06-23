# RTMPose3D (RTMW3D) Pose Backend — Design

**Status:** Draft for review · 2026-06-23 · branch `feat/multi-person`

**Goal:** Add a selectable monocular-3D pose backend (`rtmpose3d`) that feeds
the existing NCKU 9-metric engine, without disturbing the current MediaPipe
and YOLO backends. MediaPipe stays the cross-platform default; `rtmpose3d` is
an opt-in choice for machines with an NVIDIA GPU.

**Architecture (one line):** YOLO+ByteTrack (crops + stable track_id) →
per-crop RTMW3D-x (133×3) → take body-17 → map to standard H36M-17 (with z) →
`PersonPose(is_3d=True)` → existing SlotBinder / per-slot DanceMetricsEngine /
MultiSlotOSC.

**Tech stack:** `rtmlib` (RTMW3D-x ONNX) + `onnxruntime`(-gpu). No MMPose, no
nvcc — confirmed: rtmlib's only deps are numpy / opencv / onnxruntime.

---

## 1. Motivation

The engine's most derivative-heavy metrics (torque = 2nd, jerk = 3rd order)
are corrupted by jitter; a cleaner, anatomically-plausible 3D source helps.
MediaPipe's monocular z is the weakest axis. RTMW3D is trained specifically
for monocular 3D (redefined z-axis) and lives in the rtmlib/ONNX ecosystem the
team already knows. Quantitative depth accuracy is **not** a goal (the signal
drives expressive OSC control); stability and "good enough" 3D are.

## 2. Feasibility — Gate 0 (RESOLVED)

- rtmlib ships **RTMW3D-x**: 133-keypoint whole-body **3D**, ONNX, runs purely
  on onnxruntime. No MMPose, no CUDA Toolkit/nvcc. ✅
- ONNX weights available (e.g. HuggingFace `Soykaf/RTMW3D-x`); rtmlib loads via
  URL or local path.
- This removes the original deployment risk. **No pivot to ROMP needed** unless
  Gate 1 (real-time) fails.

## 3. Architecture

New module `backend/pose_backends/rtmpose3d_backend.py` implementing the
existing `PoseBackend` protocol (`estimate(frame, timestamp_ms) -> [PersonPose]`,
`close()`). Top-down:

1. **Detection + tracking:** reuse the existing YOLO+ByteTrack path for person
   boxes + stable `track_id`. (Spike may start with rtmlib's bundled detector
   for a single person, then switch to YOLO+ByteTrack for multi-person parity.)
2. **Per-crop 3D pose:** run RTMW3D-x on each person crop → 133×3 keypoints.
3. **Body-17 extract:** take keypoints `[0:17]` (these are exactly COCO-17 body
   order within COCO-WholeBody-133).
4. **Map to H36M-17:** a 3D variant of the now-fixed `coco17_to_h36m17` that
   keeps the real z instead of zeroing it (see §4). Output is the **standard
   H36M-17 layout** (post-fix: 8 thorax, 9 neck, 10 head, 11-13 left arm,
   14-16 right arm).
5. **Emit** `PersonPose(track_id, h36m17, bbox, kpts_2d, is_3d=True)`.

Lazy-import `rtmlib` inside `__init__` (like the YOLO backend) so importing the
module never forces the dependency on MediaPipe-only machines.

## 4. Integration contract & coordinate alignment

- **Joint layout:** must match the other backends' (now-corrected) standard
  H36M-17 — add a `coco17_to_h36m17_3d(kpts3d)` helper (or parametrize the
  existing one) that fills z from the input rather than 0. Keep the spine-chain
  derivation identical (pelvis, thorax=mid-shoulders, spine, neck, head).
- **Coordinate convention:** RTMW3D native axes (x→right, y→down, z→depth)
  already align with the engine (height = −Y, ground = X–Z). No axis flip
  expected; **verify** the z sign empirically.
- **Scale:** RTMW3D x/y are pixel-scale and z is its own unit. The engine has
  scale constants tuned for MediaPipe's ×1000 "mm" (`expansion /1e8`,
  stability `/1000`). The adapter must bring RTMW3D output to a comparable
  scale so the 9 metrics land in the same magnitude band as MediaPipe.
  Acceptance = metric magnitudes within the same order as MediaPipe on the same
  clip.

## 5. Factory wiring

`make_backend` gains an `rtmpose3d` branch:
```python
if name == "rtmpose3d":
    from pose_backends.rtmpose3d_backend import RTMPose3DBackend
    return RTMPose3DBackend()
```
Selected via `FIELD_POSE_BACKEND=rtmpose3d`. Default stays `yolo`; MediaPipe via
`mediapipe`.

## 6. Gates / acceptance

- **Gate 0 — deployment:** ✅ resolved (rtmlib RTMW3D-x ONNX, no nvcc).
- **Gate 1 — real-time (single person, 4080):** install rtmlib, load RTMW3D-x
  with the onnxruntime CUDA EP, run on a test clip, measure FPS. Target ≥ the
  viewer's analysis rate for one dancer. If it cannot hit real-time even for one
  person → **fall back to ROMP** (simple-romp, already evaluated).
- **Gate 2 — metrics sane:** feed RTMW3D output through the adapter into the
  engine; the 9 metrics are finite, non-degenerate, and same-order-of-magnitude
  as MediaPipe on the same input.

## 7. Scope

**In:** single-person `rtmpose3d` backend end-to-end (detect → RTMW3D → H36M-17
→ metrics), factory wiring, unit tests for the 3D mapping adapter, Gate 1/2
verification notes.

**Out (separate phases, per agreed order 3D → smoothing → multi-person):**
- One-Euro smoothing (phase 2) — applies to all backends, not RTMPose-specific.
- 4-person real-time validation (phase 3) — needs 4 crops/forward passes; FPS
  headroom TBD; may motivate a lighter RTMW3D variant or ROMP.
- Re-export of the ai-motion CultureScore culture map after the joint-mapping
  fix (tracked separately; affects morrisness, not this backend).

## 8. Risks

- **RTMW3D-x is the large variant** → single-person real-time on 4080 likely OK,
  4-person uncertain (phase 3 problem).
- **Scale alignment** is the main integration effort and is empirical
  (tune-and-verify against MediaPipe magnitudes).
- **Depth remains monocular** — output is plausible/stable 3D, not metric truth.
  Accepted (quantitative accuracy is out of scope).

## 9. Testing approach

- Unit-test the 3D mapping adapter with synthetic 133×3 input: assert body-17
  extraction + standard H36M-17 placement (wrists on 13/16, head on 10) and that
  z is preserved (not zeroed). Mirror the `test_keypoint_mapping` style.
- Gate 1/2 are empirical (run on a clip); record FPS + metric-magnitude notes in
  the plan, not as asserted unit tests.
