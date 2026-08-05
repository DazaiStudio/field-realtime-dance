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

1. Pick a camera (tick **Mirror camera** if you want a mirrored view), then press `Enter`.
2. Use the buttons at the bottom of the video: the **camera** icon turns the stream on/off, the **skeleton** icon turns pose detection (overlay + metrics + OSC) on/off.
3. `Change Input` reopens the input card.

Only one app can own a camera at a time - close other camera apps if the feed is black.
Camera choices are shown as stable OpenCV indices (`Camera 0`, `Camera 1`, ...). On macOS and Windows, system camera-name order can differ from the index order that OpenCV actually opens, so the preview is the source of truth.

## OSC

Default output `udp://127.0.0.1:9000`, prefix `/field`, one float per metric:

```text
/field/energy      /field/sync_vel    /field/sync_corr
/field/expansion   /field/curvature   /field/height
/field/sway        /field/torque      /field/jerk
```

Outputs, prefix, normalize profile, metric toggles, per-metric smoothing, and enable are all changeable at runtime in the viewer - no restart needed.

- **Normalize profile:** defaults to `None - raw` for raw metric values. Choose a saved calibration profile for fixed profile ranges. The selected option becomes active immediately.
- **Calibration presets:** use the input-screen `Calibrate` button, or `Calibration -> Start`, run a short compact/open movement range, then press `Stop` to save the profile. Profiles are built from Smoothness(EMA) output samples and stored locally in `backend/calibration_presets.json`.
- **Outputs:** use `Add Output` for extra OSC receivers. Each row has a name, target host/IP, and port. Broadcast is automatic for `.255` broadcast addresses.
- **Joint smoothness:** disabled in the viewer path so metrics are calculated from the detected skeleton directly.
- **Per-metric Smoothness(EMA):** each output channel uses a 0f-10f EMA frame slider. 0f is off; the default is 3f, except `Sync Correlation`, which defaults to 0f.
- **Prefix:** defaults to `/field`; blank prefix sends root-level addresses such as `/energy`.
- The on-screen values always match what is sent over OSC.
