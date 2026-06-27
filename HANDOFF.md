# Handoff — field-realtime-dance (2026-06-24)

Continuation notes for picking up development on another machine (e.g. macOS).

## TL;DR — where things are

- **`feat/rtmpose-backend`** ← **current deliverable.** Clean single-person
  viewer: original MediaPipe path in the rehearsal UI, RTMPose3D code retained
  but hidden from the dropdown, and **One-Euro joint smoothing**. Branched off
  `master`,
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

In the browser: pick a **Camera** → **Enter**. The viewer opens the camera and
starts detection immediately. Metrics stream to the right panel and out over
OSC at `/field/<metric>` (energy, sync_velocity, sync_correlation, expansion,
curvature, height, sway, torque, jerk).

## Backends & cross-platform (important for the Mac)

| Backend | Where it runs | Notes |
|---|---|---|
| **MediaPipe** (default) | Win / **macOS** / Linux, CPU or any GPU | Pseudo-3D. The cross-platform path — **use this on the Mac.** No extra deps. |
| **RTMPose3D** (hidden from rehearsal UI) | NVIDIA GPU for real-time | Cleaner monocular 3D (rtmlib RTMW3D-x, ONNX). Needs `pip install rtmlib onnxruntime-gpu` + CUDA 12/cuDNN 9. Code remains available, but the viewer currently exposes MediaPipe only so Mac/field use stays simple. |

So on the Mac you can develop the whole pipeline with MediaPipe + One-Euro.
RTMPose3D is the Windows/NVIDIA path and is fully optional (lazy-imported).

## Architecture / file map (`backend/`)

- `osc_viewer.py` — FastAPI app, camera streaming, the web UI, OSC config,
  `/api/apply` (carries `pose_backend`, `smooth_enabled`, `smooth_min_cutoff`).
  `get_pose_engine()` (re)builds the engine when the backend changes.
- `pose_engine.py` — thin orchestrator: picks a pose source by backend name,
  runs One-Euro on the joints, feeds `DanceMetricsEngine`. Public interface
  unchanged from the original. `set_backend()` swaps live; graceful fallback to
  MediaPipe if RTMPose3D can't load.
- `pose_sources.py` — `MediaPipePoseSource` + `RTMPose3DPoseSource`
  (single-person = the largest detected person; no tracker needed). RTMPose3D
  coordinate scaling + the CUDA-DLL registration live here.
- `one_euro.py` — `JointSmoother`: speed-adaptive One-Euro filter over (17,3),
  applied **before** the metrics engine. Tunable `min_cutoff` (UI slider) /
  `beta`. Measured ~82–88% jerk-jitter reduction on a test clip.
- `keypoint_mapping.py` — `mp33_to_h36m17` (MediaPipe) and `coco17_to_h36m17_3d`
  (RTMW3D body-17) → **standard** H36M-17.
- `dance_metrics.py` / `constants.py` — the NCKU 9-metric engine (unchanged).
- `osc_sender.py` — OSC output (`/field/<metric>`, per-metric EMA + normalize).

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
- **RTMPose3D coordinate scaling** (see `pose_sources.py` docstrings): RTMW3D x,y are
  model-input pixels and z is a dimensionless depth; we make z share x,y units
  (× bbox height), apply a uniform `RTM_POSE_SCALE = 3.0`, then root-center on
  the pelvis to match MediaPipe's per-frame hip origin. Tuned so the 9 metrics
  land in MediaPipe's magnitude band.
- **Two smoothing stages (different jobs).** (1) **Joint layer** — One-Euro on
  the skeleton before metrics (`one_euro.py`, global min_cutoff/beta). It is
  controlled by the `Joint smoothness` switch in the rehearsal UI and defaults
  on. This is the primary jitter reducer; cleans positions before the
  derivative-heavy torque/jerk. (2) **Output layer** — per-metric EMA
  on each of the enabled OSC channels (`osc_sender.py` `metric_alphas`, a "smoothness"
  slider on every metric card; live via
  `/api/metric_smoothing`). Stage 1 = clean source; stage 2 = per-channel feel
  for the OSC consumers (visuals/sound) since metrics differ wildly (jerk noisy
  vs height slow). The old global OSC alpha is kept internally at 1.0 for
  compatibility, but it is no longer shown in the viewer. **Output EMA defaults
  OFF (alpha=1)** so One-Euro is the sole default smoother (no double-smoothing).
  Smoothness sliders snap in 5% steps for repeatable values. Each
  per-metric Smoothness control sits at the top-right of its card, with the
  value/bar readout on a full row below it.
- **Metric output toggles.** Each of the 9 metric cards has a checkbox. Checked
  metrics are shown and sent over OSC; unchecked metrics remain visible as
  disabled cards but are omitted from `/field/<metric>`, Output values, fullscreen
  overlay, and `/charts` output snapshots. Metric cards keep the visible label
  compact; hover the card/name for a fuller explanation. The metrics engine
  still computes all 9 internally.
- **Live charts page** `GET /charts` (link on the viewer): canvas waveforms of
  enabled metrics, polling `/api/metrics` (~10 Hz). Bold = OSC value, faint =
  pre-output. Use it to watch the smoothing while tuning the sliders.
- **OSC modes.** Viewer currently exposes `raw` (physical units, comparable
  across time, unbounded) and `normalize` (adaptive 0..1 — auto-ranges to the
  dancer but NOT comparable across time, the reference peak drifts). Sync
  Correlation stays bipolar (`-1..1`) even when Normalize is on. The right metric
  column uses a compact `Normalize` switch: off = raw, on = normalize. It
  defaults on. The former Calibrate / fixed-range preset UI has been removed for
  now to keep rehearsal controls simpler.
- **OSC prefix.** The Prefix field can be blank. Blank sends root-level OSC
  addresses like `/energy`; non-empty values still normalize to `/prefix/...`.
- **Input UI.** The rehearsal UI is camera-only, fixed to the highest quality
  preset (1920x1080 capture, 20fps target, 10fps analysis). The old video upload
  and quality selector are hidden for now. Live detection now streams raw camera
  frames immediately while the pose engine loads in the background, so the first
  Enter should not sit on a black frame while MediaPipe warms up. Live/preview
  stream exits release the shared camera in `finally` so reconnects do not leave
  the webcam stuck.

## Pending / TODO

- [ ] **Live camera check** — eyeball the viewer with a real camera: skeleton
      tracks correctly, metrics move, and tune the Smoothness sliders for feel.
      (Verified end-to-end on a recorded clip; not yet eyeballed live.)
- [ ] **Re-export the ai-motion culture map** before relying on morrisness
      (because of the H36M mapping fix above).
- [ ] Multi-person: parked on `feat/multi-person` if/when needed.

## Verification done

End-to-end on a test clip, both backends produce metrics; One-Euro cuts jerk
std ~82–88%. Unit tests: `cd backend && python -m unittest discover -s tests`
(50 green, 1 skipped). Server starts and the UI renders.
