# FIELD Realtime Dance

Browser viewer for live camera dance analysis: skeleton overlay, 9 realtime metrics, OSC output.
Fork of [`YukiHataRin/realtime-dance-analysis`](https://github.com/YukiHataRin/realtime-dance-analysis) for FIELD rehearsals and performance.

## Requirements

- Python 3.11 (plain install, venv, conda, or uv all fine)
- A webcam

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

1. Pick a camera (mirrored by default), then press `Enter`.
2. Use the buttons at the bottom of the video: the **camera** icon turns the stream on/off, the **skeleton** icon turns pose detection (overlay + metrics + OSC) on/off.
3. `Change Input` reopens the input card.

Only one app can own a camera at a time - close other camera apps if the feed is black.

## OSC

Default target `udp://127.0.0.1:9000`, prefix `/field`, one float per metric:

```text
/field/energy      /field/sync_vel    /field/sync_corr
/field/expansion   /field/curvature   /field/height
/field/sway        /field/torque      /field/jerk
```

Host, port, prefix, normalize mode, metric toggles, per-metric smoothing, and enable are all changeable at runtime in the viewer - no restart needed.

- **Normalize off:** raw metric values.
- **Normalize on:** most metrics map into adaptive 0..1 ranges; `sync_corr` stays `-1..1`.
- **Joint smoothness:** One-Euro smoothing before metrics, default on.
- **Per-metric Smoothness:** each output channel defaults to 30%, snaps in 5% steps, and can be tuned independently.
- **Prefix:** defaults to `/field`; blank prefix sends root-level addresses such as `/energy`.
- The on-screen values always match what is sent over OSC.
