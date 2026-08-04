# Handoff - field-realtime-dance (2026-06-27)

> **Maintainers / AI assistants: read [MAINTENANCE.md](MAINTENANCE.md) first** - current branch states, open issues from the 2026-07-07 review, deliberate design decisions, and code minefields. Parts of this file predate the 06/30-07/01 rehearsal tuning (e.g. RTMPose3D is no longer reachable from the viewer).

Continuation notes for picking up development on another machine, including macOS.

## TL;DR

- `master` is the current deliverable branch. It contains the clean single-person rehearsal viewer: MediaPipe-only in the UI, camera-only input, Normalize default-on, joint smoothing disabled in the viewer path, and per-metric Smoothness(EMA) defaulting to 3f except Sync Correlation at 0f.
- `feat/rtmpose-backend` was fast-forward merged into `master` at `d8526ef`. The branch remains on origin as a backup/history branch, but new work should continue from `master`.
- `feat/multi-person` is a shelved exploration for 4-dancer slots and per-slot OSC. Its `docs/superpowers/specs|plans/` keep the original RTMPose3D design/plan and multi-person spec.
- The remote default branch is `master`: https://github.com/DazaiStudio/field-realtime-dance.

## Run It

> Setting up a fresh Windows machine (including the Azure Kinect backend)?
> Follow **`INSTALL_WINDOWS.md`** — it covers Python, SDKs, env vars, and the
> verification chain step by step.

Python 3.10+ from the repo root:

```bash
pip install -r requirements.txt
python backend/osc_viewer.py
```

Open `http://127.0.0.1:9100`. OSC defaults to `udp://127.0.0.1:9000` with prefix `/field`.

In the browser: pick a **Camera** -> **Enter**. The viewer opens the camera and starts detection immediately. Metrics stream to the right panel and out over OSC at `/field/<metric>`:

```text
energy, sync_velocity, sync_correlation, expansion, curvature,
height, sway, torque, jerk
```

## Backends And Mac Notes

| Backend | Where it runs | Notes |
|---|---|---|
| MediaPipe (default) | Windows / macOS / Linux, CPU or any GPU | Cross-platform pseudo-3D path. Use this on the Mac. No extra deps. |
| RTMPose3D (hidden from rehearsal UI) | NVIDIA GPU for real-time | Cleaner monocular 3D via rtmlib RTMW3D-x/ONNX. Needs `rtmlib`, `onnxruntime-gpu`, CUDA 12, cuDNN 9. Code remains available, but the viewer exposes MediaPipe only for field use. |
| Azure Kinect (auto-shown when available) | Windows only, GPU (DirectML) | True depth 3D in mm + native multi-person body tracking; works in the dark (ToF/IR). Appears in the backend dropdown only when the SDKs + `pykinect_azure` are installed. |

RTMPose3D is fully optional and lazy-imported. The rehearsal UI intentionally hides it.

### Azure Kinect backend (2026-07-30, branch feat/kinect-pose-source)

- Requirements (Windows only): Azure Kinect Sensor SDK v1.4.2 + Body Tracking SDK 1.1.2 installed to their default `C:\Program Files` locations, plus `pip install pykinect_azure`.
- Env vars: `FIELD_KINECT_GPU` = DirectML adapter index (default `1` for the dual-GPU dev laptop where 0 is the iGPU at ~6 fps; set `0` on single-GPU machines), `FIELD_KINECT_MODEL` = `full` (default) | `lite`, `FIELD_KINECT_DEPTH_MODE` = `nfov` (default, ~0.5-3.9 m) | `nfov_binned` (~0.5-5.5 m, half the depth resolution, still 30 fps) | `wfov_binned` (120 deg but only ~0.25-2.9 m).
- The Kinect brings its own capture: the camera dropdown and `/preview_stream` are ignored for this backend, and the Kinect's RGB camera must not be opened as a UVC webcam at the same time.
- "Kinect view" dropdown (visible only for this backend): Color, or colorized Depth (16:9 letterboxed) for dark-stage work.
- Stable ID uses the same `MultiPersonTrackRegistry`; K4ABT body ids are raw ids only (they change after occlusion). OSC contract v2 (`/field/<id>/<metric>`) is unchanged.
- Depth range NFOV ~0.5-3.9 m; measure the stage before relying on it.
- **The depth camera sees a narrower field than the colour view** (NFOV ~75 deg horizontal vs ~90 deg for 720P colour). A dancer visible at the left or right edge of the colour preview can be outside the tracked volume entirely. Switch "Kinect view" to Depth to see the real coverage before setting marks.
- **Never SIGKILL the viewer** (`Stop-Process -Force`, killing the terminal). That skips `finally: frame_source.release()` and the lifespan hook, and leaves the k4a device wedged: the next run reports `devices connected: 0` until the USB cable is physically unplugged and replugged. Stop it with the UI Quit button, or `POST /api/camera/release` then `POST /api/shutdown`.
- Range is *radial* distance from the sensor, so mounting high and angling down costs reach: a camera 3 m up looking at a dancer 4 m away horizontally is measuring 5 m. Chest height and level maximises horizontal coverage.
- Costumes matter: the depth camera is an IR time-of-flight sensor, and dark IR-absorbing fabric can return almost nothing, so a dancer inside the range can still be missing from the depth map. Test the actual costumes before the get-in.

## File Map

- `backend/osc_viewer.py` - FastAPI app, camera streaming, web UI, OSC config, metric toggles, smoothing controls, `/charts`, `/api/apply`.
- `backend/pose_engine.py` - selects the pose source, optionally runs One-Euro over joints for legacy/manual testing, feeds `DanceMetricsEngine`.
- `backend/pose_sources.py` - `MediaPipePoseSource` and `RTMPose3DPoseSource`; single-person = largest detected person.
- `backend/one_euro.py` - speed-adaptive One-Euro joint smoother over `(17, 3)`.
- `backend/keypoint_mapping.py` - MediaPipe 33 and COCO/RTM 17 into standard H36M-17.
- `backend/dance_metrics.py` / `backend/constants.py` - the NCKU 9-metric engine.
- `backend/osc_sender.py` - OSC output, normalization, per-metric EMA smoothing.

## Current Viewer Behavior

- **Input UI:** camera-only. The old video upload and quality selector are hidden.
- **Quality:** fixed to the highest preset: 1920x1080 capture, 20fps stream target, 10fps analysis target.
- **Enter flow:** first screen is a pure camera input menu. Pressing Enter opens the camera and detection together.
- **Warmup behavior:** live detection streams raw camera frames while MediaPipe loads, so the first Enter should not sit on a black frame.
- **Camera recovery:** live/preview stream cleanup releases the shared camera. If OpenCV drops frames repeatedly, the stream attempts to release/reopen the camera instead of freezing on the last frame.
- **Change Input:** returns to the pure input menu state.
- **RTMPose:** hidden from the dropdown. The UI exposes only MediaPipe.
- **Calibration:** calibration/fixed-range preset UI is active. Calibration records Smoothness(EMA) output samples for the seven range metrics; `sync_velocity` and `sync_correlation` are not calibrated.

## Smoothing Model

The viewer's main path uses output smoothing only.

1. **Metric source:** joint smoothing is disabled by default, so metrics are calculated from the detected H36M skeleton directly. The One-Euro code and `/api/smoothing` endpoint remain only for compatibility/manual testing.
2. **Output layer:** per-metric EMA smoothing on each enabled OSC channel. Every metric card has its own `Smoothness(EMA)` slider in pose-analysis frames. The range is 0f-10f, where 0f is off; the default is 3f except Sync Correlation at 0f.

The old global OSC alpha is kept internally at `1.0` for compatibility, but it is no longer shown in the viewer.

## Metric Output

- Each of the 9 metric cards has a checkbox. Checked metrics are shown and sent over OSC. Unchecked metrics remain visible as disabled cards but are omitted from OSC, Output values, fullscreen overlay, and `/charts`.
- Metric cards keep labels compact; hover the card/name for a fuller explanation.
- `Open live charts` is below the metric cards. `/charts` shows enabled metrics, polling `/api/metrics`; bold = OSC value after output smoothing, faint = pre-output value.

## Normalize / OSC

- Normalize defaults on.
- Raw mode sends physical metric units.
- The viewer exposes `None - raw` and saved calibration profiles. The old adaptive normalize mode remains in `OSCSender` for compatibility but is no longer offered in the UI.
- `Sync Correlation` stays bipolar (`-1..1`) even when Normalize is on.
- `Sync Correlation` means left-right timing match, not movement size: `+1` same rhythm, `0` no clear timing link, `-1` alternating/opposite timing.
- Prefix defaults to `/field`, but can be blank. Blank sends root-level addresses such as `/energy`; non-empty values normalize to `/prefix/...`.

Default OSC addresses:

```text
/field/energy      /field/sync_vel    /field/sync_corr
/field/expansion   /field/curvature   /field/height
/field/sway        /field/torque      /field/jerk
```

## Technical Notes

- **H36M mapping fix:** upstream NCKU realtime-dance-analysis placed wrists on H36M indices 12/15 instead of standard 13/16, shifting head/shoulders and affecting energy/curvature. `keypoint_mapping.py` uses the standard layout. Metric magnitudes now differ from the old/NCKU behavior, intentionally.
- **CultureScore / morrisness:** the culture map in the sibling `ai-motion` repo was exported with the old mapping and must be re-exported before relying on morrisness.
- **RTMPose3D scaling:** RTMW3D x/y are model-input pixels and z is dimensionless depth. `pose_sources.py` scales z into x/y units, applies `RTM_POSE_SCALE = 3.0`, then root-centers on the pelvis to land in MediaPipe's magnitude band.

## Pending

- [ ] Visual live-camera check in the rehearsal space: skeleton tracks correctly, metrics move, and Smoothness(EMA) feel is appropriate.
- [ ] Re-export the `ai-motion` culture map before relying on morrisness.
- [ ] Multi-person remains parked on `feat/multi-person` until needed.

## Verification Done

- `cd backend && python -m unittest discover -s tests` -> 50 passed, 1 skipped.
- Latest live-stream probe saw 355 frames over 20 seconds with no backend error, around 20fps stream and 7-8fps analysis.
- Server starts and the UI renders at `http://127.0.0.1:9100`.
