# Multi-Person Detection — Design Spec

- **Date**: 2026-06-16
- **Branch**: `feat/multi-person`
- **Component**: `field-realtime-dance` viewer (`backend/`)
- **Status**: approved for planning

## 1. Goal & Constraints

Extend the realtime dance-analysis viewer from single-person to **up to 4 simultaneous dancers**, each with their own metrics and OSC output, with **stable identity** as the top priority.

Hard requirements (from Tommy):
- **One unified system**, portable across any hardware — must run on RTX 4080 (CUDA) *and* M1 Pro (MPS), like MediaPipe does today. No GPU-specific second system.
- **Stable per-dancer identity** that survives tracker churn.
- **Quantitative comparability with the offline RTMPose pipeline is NOT a goal** — metrics are expressive control signals for OSC, not research-grade measurements. 2D is acceptable.
- **4 simultaneous metric panels** in the UI.
- **Manual ID reassignment** included in the initial scope.

Non-goals (initial): ReID hyper-parameter tuning, aggregate whole-stage metrics, depth-camera/true-3D, IR-camera handling.

## 2. Architecture Overview

```
camera ─▶ PoseBackend.estimate() ─▶ [PersonPose...]  (detect + track + map, per backend)
                 │
                 ▼
         Slot Binding layer (track_id ─▶ fixed slot 1..4, auto + manual)
                 │
                 ▼
     per-slot DanceMetricsEngine  ─▶  per-slot OSCSender  ─▶  /field/{slot}/...
                 │
                 ▼
         viewer UI: N skeletons (slot color + label) + 4 compact metric panels + reassign controls
```

The pose model is a **swappable module** behind one interface. Everything downstream (binding, metrics, OSC, UI) is backend-agnostic. This is one system with a configurable pose backend, not two systems.

## 3. PoseBackend Interface

New module `backend/pose_backends/` (or `backend/pose_backend.py`):

```python
@dataclass
class PersonPose:
    track_id: int            # volatile id from the backend's tracker
    h36m17: np.ndarray       # (17, 3) — unified format the metrics engine consumes
    bbox: tuple              # (x1, y1, x2, y2) image coords — drawing / labels / click-to-select
    kpts_2d: np.ndarray      # (K, 2) image coords — skeleton overlay
    is_3d: bool              # whether h36m17 z is meaningful (False for 2D backends)

class PoseBackend(Protocol):
    def estimate(self, frame, timestamp_ms: float) -> list[PersonPose]: ...
    def close(self) -> None: ...
```

Each backend internally owns detection + tracking + keypoint mapping and returns a list of tracked people already mapped to H36M-17.

### 3.1 YOLO26Backend (default)
- Ultralytics `model.track(frame, persist=True, tracker="botsort.yaml")` → boxes (+ `boxes.id` track ids) + COCO-17 keypoints (2D image coords).
- BoT-SORT tracker (ReID left at sensible defaults initially; tuning is out of scope).
- Map **COCO-17 → H36M-17** (see §3.3), `z = 0`, `is_3d = False`.
- Device auto-select: CUDA on the 4080, MPS on Apple Silicon, CPU fallback. One codebase, any hardware.
- Weight: `yolo26{n,s,m}-pose.pt`, default `m`; size is a tunable chosen against real fps. (Confirm exact weight name / `ultralytics` version at implementation time.)
- Stateful tracker ⇒ must be driven from a **single dedicated worker thread** (no arbitrary thread-pool concurrency).

### 3.2 MediaPipeBackend (optional, selectable)
- `PoseLandmarker` with `num_poses=4`.
- MediaPipe gives no track ids → a lightweight **centroid/IoU tracker** assigns volatile ids.
- Reuse existing MP-33 → H36M-17 mapping (`pose_engine._get_h36m_compatible_landmarks`), pseudo-z present, `is_3d = True`.
- Selected via `FIELD_POSE_BACKEND=mediapipe`. Kept as fallback (e.g., a machine where YOLO underperforms, or quick comparison).

Backend chosen by `FIELD_POSE_BACKEND=yolo|mediapipe` (default `yolo`), overridable from the UI.

### 3.3 COCO-17 → H36M-17 mapping
COCO-17 lacks pelvis/neck/spine; derive them exactly as the existing MediaPipe mapping does:
- `pelvis = mid(l_hip, r_hip)`, `neck = mid(l_shoulder, r_shoulder)`, `spine = mid(pelvis, neck)`, `head = nose`.
- Limb joints map directly (shoulders/elbows/wrists, hips/knees/ankles). `z = 0`.
- Unit covered by a focused unit test against the H36M-17 index layout used in `dance_metrics`.

## 4. Identity — Slot-Based (4 fixed slots)

Decouple volatile tracker ids from the stable identities the show cares about.

- **4 fixed slots** (Dancer 1–4). OSC and UI are always keyed by slot, never by raw track_id.
- **Binding layer** maps `track_id → slot` each frame:
  - **Auto-assign**: a new track with no slot gets the lowest free slot.
  - **Eviction**: a slot whose bound track has been missing for `K` frames is freed (and stops emitting / shows empty).
  - **Manual reassignment** (initial scope): operator can bind a given track to a chosen slot, and swap two slots. Entry points in the UI (§7).
- **State follows the slot**: the per-slot `DanceMetricsEngine` and OSC smoothing state persist across track_id changes, so a dancer who is lost and re-acquired into the same slot keeps continuity.

## 5. Per-Slot Metrics Engine

- `{slot: DanceMetricsEngine}`, created when a slot becomes occupied, retained while occupied (history is per-slot, never shared).
- Freed on slot eviction.
- The engine consumes H36M-17 regardless of backend.

### 5.1 2D adaptation (when `is_3d == False`)
- `energy`, `sync_velocity`, `sync_correlation`, `curvature`, `torque`, `jerk`, `height` — work unchanged in 2D (planar angles, z-component of cross product, image-y height).
- `expansion`: 3D ConvexHull **volume** → **2D ConvexHull area** (volume degenerates on coplanar z=0 points).
- `sway`: 3D XZ-plane offset → **horizontal (X) offset** of CoM from base of support.
- Implemented as a 2D branch in `DanceMetricsEngine` selected by an `is_3d` flag/dim. Goal is stable, responsive signals — not cross-backend numeric equality.

## 6. OSC Schema

- Per-slot addresses: `/field/{slot}/energy`, `/field/{slot}/sync_vel`, `/field/{slot}/sync_corr`, `/field/{slot}/expansion`, `/field/{slot}/curvature`, `/field/{slot}/height`, `/field/{slot}/sway`, `/field/{slot}/torque`, `/field/{slot}/jerk`, `/field/{slot}/morrisness`.
- Meta so the receiver knows who is live: `/field/active_slots [1,3,4]`, `/field/count 3`.
- `OSCSender` currently holds smoothing/normalize state **globally per-metric** → multi-person crosstalk. Fix: **one OSCSender per slot** (shared host/port/mode/alpha; config changes broadcast to all), keyed by slot. Internal logic unchanged.
- Per-slot state isolation covered by a unit test.

## 7. UI

- Live feed draws **all skeletons**; each bbox labelled with its slot number and a **slot-determined color** (stable per slot).
- Right side: **4 fixed compact panels** (small multiples), one per slot, each showing that slot's 9 metrics. Unoccupied slots render as "empty".
- **Manual reassignment controls**: click a skeleton then click a slot panel to bind it; per-panel "reassign / swap" control. Backed by a small set of `POST /api/slots/...` endpoints.
- Fullscreen overlay extends to show the 4 panels.

## 8. Concurrency

- YOLO `track(persist=True)` is stateful → run inference in a **single dedicated worker thread** feeding the async stream loop (current code already offloads `process_frame` via `asyncio.to_thread`; tighten to a fixed worker so tracker state stays consistent).

## 9. Testing

- Unit: COCO-17→H36M-17 mapping; slot binding (auto-assign, eviction, manual bind, swap); per-slot OSCSender state isolation; 2D metric variants (expansion area, sway horizontal) on synthetic poses.
- Smoke: run a short multi-person clip through the pipeline, assert 4 slots populate, OSC addresses emit per slot, no crosstalk.
- Existing `tests/test_osc_sender.py` is the pattern to follow.

## 10. Out of Scope (initial)
- ReID hyper-parameter tuning.
- Aggregate / whole-stage composite metrics.
- True 3D (depth camera, multi-view) and 2D→3D lifting.
- IR-camera adaptation.

## 11. Files (anticipated)
- **New**: `backend/pose_backend.py` (interface + `PersonPose`), `backend/pose_backends/yolo26.py`, `backend/pose_backends/mediapipe.py`, `backend/slot_binder.py`.
- **Changed**: `backend/dance_metrics.py` (2D variants + `is_3d`), `backend/osc_sender.py` (per-slot manager), `backend/osc_viewer.py` (multi-slot stream loop, slot endpoints, VIEWER_HTML 4-panel UI), `backend/pose_engine.py` (refactor existing MediaPipe path behind the backend interface).
- **Tests**: extend `backend/tests/`.
