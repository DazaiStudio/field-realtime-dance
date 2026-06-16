# FIELD Realtime Dance

Browser viewer for live camera / video dance analysis: skeleton overlay, 9 realtime metrics, OSC output.
Fork of [`YukiHataRin/realtime-dance-analysis`](https://github.com/YukiHataRin/realtime-dance-analysis) for FIELD rehearsals and performance.

## Requirements

- Python 3.11 (plain install, venv, conda, or uv all fine)
- A webcam, or video files to upload

## Install & Run

Video walkthrough (macOS): https://www.youtube.com/watch?v=mqPr_ezzFXo

Windows (PowerShell):

```powershell
git clone https://github.com/DazaiStudio/field-realtime-dance.git
cd field-realtime-dance
pip install -r requirements.txt
python backend\osc_viewer.py
```

macOS / Linux:

```bash
git clone https://github.com/DazaiStudio/field-realtime-dance.git
cd field-realtime-dance
python3 -m pip install -r requirements.txt
python3 backend/osc_viewer.py
```

Open `http://127.0.0.1:9100` and leave the terminal running.
The MediaPipe pose model downloads automatically on first run.

To update an existing clone:

```bash
git pull
pip install -r requirements.txt
python backend/osc_viewer.py
```

## Usage

1. Pick a camera (mirrored by default) **or** click to upload a video, then press `Enter`.
2. Use the buttons at the bottom of the video: the **camera** icon turns the stream on/off, the **skeleton** icon turns pose detection (overlay + metrics + OSC) on/off.
3. `Change Input` reopens the input card.

Video files loop by default. Only one app can own a camera at a time — close other camera apps if the feed is black.

## OSC

Default target `udp://127.0.0.1:9000`, prefix `/field`, one float per metric:

```text
/field/energy      /field/sync_vel    /field/sync_corr
/field/expansion   /field/curvature   /field/height
/field/sway        /field/torque      /field/jerk
```

Host, port, prefix, mode, smoothing, and enable are all changeable at runtime in the viewer — no restart needed.

- **Mode** `raw`: original metric values. `normalize`: bounded 0–1 output (adaptive peaks/ranges; `sync_corr` maps −1..1 → 0..1).
- **Alpha** smoothing: `1.0` = none, lower = smoother (default `0.25`).
- The on-screen values always match what is sent over OSC.

## Multi-Person

Up to 4 dancers are tracked at once, each pinned to a fixed slot (1–4) so downstream OSC
receivers always know which address is which dancer, even as people enter/leave frame.

**Pose backend** — set via env var `FIELD_POSE_BACKEND=yolo|mediapipe` (default `yolo`).
YOLO26 (BoT-SORT tracking, 2D) is the default and runs on CUDA/MPS/CPU; `mediapipe` is the
alternative (pseudo-3D). `ultralytics` is now a dependency.

**Per-slot OSC** — each of the 9 metrics plus `morrisness` is sent per slot:

```text
/field/{slot}/energy      /field/{slot}/sync_vel    /field/{slot}/sync_corr
/field/{slot}/expansion   /field/{slot}/curvature   /field/{slot}/height
/field/{slot}/sway        /field/{slot}/torque      /field/{slot}/jerk
/field/{slot}/morrisness
```

Meta addresses report which slots are currently occupied: `/field/active_slots` (list of
slot numbers) and `/field/count` (int).

**Manual reassignment** — the viewer shows 4 per-dancer metric panels. If the tracker
misassigns a dancer to the wrong slot, use a panel's **swap…** button (two clicks: pick the
panel to swap with) to exchange which dancer occupies which slot.
(Endpoints: `POST /api/slots/swap`, `POST /api/slots/bind`.)
