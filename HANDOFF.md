# Handoff — field-realtime-dance (2026-06-24)

Continuation notes for picking up development on another machine (e.g. macOS).

## TL;DR — where things are

- **`feat/rtmpose-backend`** ← **current deliverable.** Clean single-person
  viewer: original MediaPipe by default, **a "Detection model" dropdown to use
  RTMPose3D instead**, and **One-Euro joint smoothing**. Branched off `master`,
  so it does NOT carry the multi-person machinery. **Develop on this branch.**
- **`feat/multi-person`** ← shelved exploration (4-dancer slots, per-slot OSC).
  Pushed as a backup. Its `docs/superpowers/specs|plans/` hold the original
  RTMPose3D design/plan and the multi-person spec if you ever want them.
- `master` ← the original single-person MediaPipe viewer (unchanged).

Both feature branches are pushed to `origin`
(https://github.com/DazaiStudio/field-realtime-dance).

## Run it

Python 3.10. From the repo root:

```bash
pip install -r requirements.txt
python backend/osc_viewer.py        # web UI on http://127.0.0.1:9100, OSC out on :9000
```

In the browser: pick a **Camera** (or upload a video) → **Enter** → toggle the
person-shaped **detection** button. Metrics stream to the right panel and out
over OSC at `/field/<metric>` (energy, sync_velocity, sync_correlation,
expansion, curvature, height, sway, torque, jerk).

## Backends & cross-platform (important for the Mac)

| Backend | Where it runs | Notes |
|---|---|---|
| **MediaPipe** (default) | Win / **macOS** / Linux, CPU or any GPU | Pseudo-3D. The cross-platform path — **use this on the Mac.** No extra deps. |
| **RTMPose3D** (dropdown) | NVIDIA GPU for real-time | Cleaner monocular 3D (rtmlib RTMW3D-x, ONNX). Needs `pip install rtmlib onnxruntime-gpu` + CUDA 12/cuDNN 9. **On a Mac there's no CUDA → the viewer auto-falls back to MediaPipe** (it won't crash; it just won't use RTMPose3D). |

So on the Mac you can develop the whole pipeline with MediaPipe + One-Euro.
RTMPose3D is the Windows/NVIDIA path and is fully optional (lazy-imported).

## Architecture / file map (`backend/`)

- `osc_viewer.py` — FastAPI app, camera/video streaming, the web UI, OSC config,
  `/api/apply` (carries `pose_backend`, `smooth_enabled`, `smooth_min_cutoff`).
  `get_pose_engine()` (re)builds the engine when the backend changes.
- `pose_engine.py` — thin orchestrator: picks a pose source by backend name,
  runs One-Euro on the joints, feeds `DanceMetricsEngine`. Public interface
  unchanged from the original. `set_backend()` swaps live; graceful fallback to
  MediaPipe if RTMPose3D can't load.
- `pose_sources.py` — `MediaPipePoseSource` + `RTMPose3DPoseSource`
  (single-person = the largest detected person; no tracker needed). RTMPose3D
  calibration + the CUDA-DLL registration live here.
- `one_euro.py` — `JointSmoother`: speed-adaptive One-Euro filter over (17,3),
  applied **before** the metrics engine. Tunable `min_cutoff` (UI slider) /
  `beta`. Measured ~82–88% jerk-jitter reduction on a test clip.
- `keypoint_mapping.py` — `mp33_to_h36m17` (MediaPipe) and `coco17_to_h36m17_3d`
  (RTMW3D body-17) → **standard** H36M-17.
- `dance_metrics.py` / `constants.py` — the NCKU 9-metric engine (unchanged).
- `osc_sender.py` — OSC output (`/field/<metric>`, optional EMA + normalize).

## Key technical notes

- **H36M mapping bug (fixed here).** Upstream NCKU realtime-dance-analysis
  placed the wrists on H36M indices 12/15 instead of the standard 13/16 (and
  shifted the head/shoulders), so `curvature` actually tracked the shoulders +
  neck and each arm's energy included a cross-body segment. `keypoint_mapping.py`
  uses the correct standard layout. **Consequence:** metric magnitudes now
  differ from the old/NCKU behaviour (intended; we don't do quantitative
  comparison). If you use the **CultureScore / morrisness** feature, the
  culture-map in the sibling `ai-motion` repo was exported with the OLD mapping
  and **must be re-exported** to stay consistent.
- **RTMPose3D calibration** (see `pose_sources.py` docstrings): RTMW3D x,y are
  model-input pixels and z is a dimensionless depth; we make z share x,y units
  (× bbox height), apply a uniform `RTM_POSE_SCALE = 3.0`, then root-center on
  the pelvis to match MediaPipe's per-frame hip origin. Tuned so the 9 metrics
  land in MediaPipe's magnitude band.
- **Two smoothing stages (different jobs).** (1) **Joint layer** — One-Euro on
  the skeleton before metrics (`one_euro.py`, global min_cutoff/beta, the
  "Smooth joints" slider). This is the primary jitter reducer; cleans positions
  before the derivative-heavy torque/jerk. (2) **Output layer** — per-metric EMA
  on each of the 9 OSC channels (`osc_sender.py` `metric_alphas`, a "smooth"
  slider on every metric card + the global alpha as fallback; live via
  `/api/metric_smoothing`). Stage 1 = clean source; stage 2 = per-channel feel
  for the OSC consumers (visuals/sound) since metrics differ wildly (jerk noisy
  vs height slow). **Output EMA defaults OFF (alpha=1)** so One-Euro is the sole
  default smoother (no double-smoothing); pull a slider down only where wanted.
- **Live charts page** `GET /charts` (link on the viewer): canvas waveforms of
  all 9 metrics, polling `/api/metrics` (~10 Hz). Bold = OSC value, faint =
  pre-output. Use it to watch the smoothing while tuning the sliders.
- **OSC modes & calibration.** Output modes: `raw` (physical units, comparable
  across time, unbounded), `normalize` (adaptive 0..1 — auto-ranges to the
  dancer but NOT comparable across time, the reference peak drifts), `fixed`
  (0..1 from calibrated per-metric ranges — bounded AND comparable AND
  personalised). The **Calibrate ("試音")** button records a short routine
  (rest / fast-big / extend↔curl / big circles / jump↔crouch / lean) and stores
  2nd/98th-percentile ranges per metric to `calibration_profile.json`
  (`calibration.py`), then switches to `fixed`. Endpoints
  `/api/calibrate/start` and `/api/calibrate/stop`. 7 unbounded metrics get
  ranges; sync_velocity/sync_correlation are already bounded. **Named presets**
  (per dancer/venue) live in `calibration_presets.json` via
  `/api/calibrate/{presets,save_preset,load_preset,delete_preset}` + a preset
  dropdown in the viewer — calibrate once, recall later (no re-calibration). The
  last unnamed calibration also auto-saves to `calibration_profile.json` and
  auto-loads on startup.

## Pending / TODO

- [ ] **Live camera check** — eyeball the viewer with a real camera: skeleton
      tracks correctly, metrics move, switch MediaPipe ↔ RTMPose3D, tune the
      smoothing slider for feel. (Verified end-to-end on a recorded clip; not
      yet eyeballed live.)
- [ ] **Re-export the ai-motion culture map** before relying on morrisness
      (because of the H36M mapping fix above).
- [ ] Multi-person: parked on `feat/multi-person` if/when needed.

## Verification done

End-to-end on a test clip, both backends produce metrics; One-Euro cuts jerk
std ~82–88%. Unit tests: `cd backend && python -m unittest discover -s tests`
(34 green). Server starts and the UI (model dropdown + smoothing controls)
renders.
