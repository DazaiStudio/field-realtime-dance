# FIELD Realtime Dance

Local viewer for live camera / video dance analysis with skeleton overlay, realtime metrics, and OSC output.

This repository is a field-use fork of [`YukiHataRin/realtime-dance-analysis`](https://github.com/YukiHataRin/realtime-dance-analysis). The original project provides the MediaPipe pose pipeline and dance metric engine; this fork focuses on making a simple local viewer that the team can run for testing, rehearsals, and OSC integration.

## What This Runs

The main tool is a browser viewer at `http://127.0.0.1:9100`.

Use it to:

- choose a live camera or click to upload a video file
- mirror live camera input when needed
- preview the input
- toggle pose detection on/off
- switch performance presets for live/video processing
- display 9 realtime dance metrics
- send the 9 metrics over OSC

## Quick Start

Recommended Python:

```text
Python 3.11
```

The project does not require conda specifically. A normal Python install, conda env, venv, or uv-managed Python is fine as long as `python --version` / `python3 --version` reports Python 3.11.

First clone the repo, or open your existing local copy:

```bash
git clone https://github.com/DazaiStudio/field-realtime-dance.git
cd field-realtime-dance
```

Install dependencies and start the viewer:

Windows PowerShell:

```powershell
pip install -r requirements.txt
python backend\osc_viewer.py --osc-port 9000 --web-port 9100 --pose-model lite
```

macOS / Linux:

```bash
python3 -m pip install -r requirements.txt
python3 backend/osc_viewer.py --osc-port 9000 --web-port 9100 --pose-model lite
```

Open:

```text
http://127.0.0.1:9100
```

Leave the terminal window running while using the viewer.

Use `backend/osc_viewer.py` for the local browser viewer. Do not start the viewer with `backend/osc_video_test.py`; that script is only for sending OSC from one video file without the web UI.

## Update Existing Clone

If the repo is already cloned on this machine, do not clone it again. Stop the viewer if it is running, then update from GitHub:

Windows PowerShell:

```powershell
cd path\to\field-realtime-dance
git pull
pip install -r requirements.txt
python backend\osc_viewer.py --osc-port 9000 --web-port 9100 --pose-model lite
```

macOS / Linux:

```bash
cd /path/to/field-realtime-dance
git pull
python3 -m pip install -r requirements.txt
python3 backend/osc_viewer.py --osc-port 9000 --web-port 9100 --pose-model lite
```

Open `http://127.0.0.1:9100` after the viewer starts.

## Viewer Workflow

1. Open `http://127.0.0.1:9100`.
2. Choose a camera, or click to upload a video file.
3. `Mirror camera` is enabled by default; turn it off if you want unmirrored live camera input.
4. Press `Enter` in the viewer.
5. Confirm the raw preview appears.
6. Toggle `Detect On` to start pose overlay, metrics, and OSC output.
7. Toggle `Detect Off` to return to preview without pose detection.

For video files, playback loops by default. For live camera, the viewer shows the selected camera stream and reports FPS/status below the video.

## OSC

Default OSC target:

```text
udp://127.0.0.1:9000
```

Default prefix:

```text
/field
```

Metric addresses:

```text
/field/energy
/field/sync_vel
/field/sync_corr
/field/expansion
/field/curvature
/field/height
/field/sway
/field/torque
/field/jerk
```

All 9 metric messages send one float value. There are no integer metric messages.

The viewer lets you change:

- OSC host
- OSC port
- address prefix
- raw / normalize mode
- Alpha (smooth) slider
- whether OSC output is enabled

## Performance

The viewer has runtime performance presets:

```text
fast      640x360, 24fps display, 10fps pose
balanced  720x405, 24fps display, 12fps pose  (default)
quality   960x540, 20fps display, 15fps pose
```

Changing `Performance` while the viewer is running restarts the local stream with the new processing size and pose-analysis rate. The page shows both `fps` (MJPEG display stream) and `pose` (actual MediaPipe analysis rate). The skeleton overlay stays enabled during detection.

The viewer defaults to the MediaPipe `lite` pose model for smoother live use. Start with `--pose-model full` if you need the previous higher-accuracy model and can accept lower FPS.

`Enable OSC` turns metric output on or off for all 9 metrics.

The viewer's metric values follow the selected OSC mode. In `raw` mode, the page shows raw output values. In `normalize` mode, the page shows the same normalized values that are sent over OSC.

OSC settings apply at runtime. Changing mode, prefix, alpha, target, or enabled state does not require restarting detection.

Metric bars are neutral readouts, not good/bad scores. Each metric has its own scale hint in the viewer. In `raw` mode, `sync_correlation` uses the center as neutral because its range is `-1..1`; in `normalize` mode, it displays left-to-right like the `0..1` OSC value.

## Camera Notes

The camera dropdown uses the device names reported by the operating system when available. On macOS, the viewer reads names from `system_profiler`; if macOS/OpenCV cannot map a device name to an index, it may still show a generic name such as `Camera 0`.

If the viewer does not show camera video:

- close other apps that may be using the camera
- check camera privacy settings
- on Windows, test the camera in the Windows Camera app
- on macOS, allow camera access for Terminal / iTerm / the Python app in System Settings
- restart the viewer after changing camera or virtual-camera sources

Only one app should own the same camera source at a time.

`Mirror camera` only affects live camera input. Video files are not mirrored.

## Test Video

You can test with any local `.mp4` file. Example:

```text
path/to/test-video.mp4
```

Click `Click to upload video` in the viewer, select the file, press `Enter`, then toggle `Detect On`.

## CLI Options

Local browser viewer:

```bash
python backend/osc_viewer.py \
  --web-host 127.0.0.1 \
  --web-port 9100 \
  --osc-host 127.0.0.1 \
  --osc-port 9000 \
  --osc-mode raw \
  --osc-alpha 0.25 \
  --osc-namespace /field \
  --pose-model lite
```

`--osc-mode` can be:

- `raw`: send metric values directly
- `normalize`: map values into a more bounded range

`--osc-alpha` controls smoothing. The default is `0.25`; `1.0` means no smoothing, and lower values are smoother.

`--pose-model` can be `lite`, `full`, or `heavy`. Use `lite` for live rehearsals and performances; use `full` or `heavy` only when pose accuracy matters more than FPS.

Standalone video-to-OSC test, without the browser viewer:

```bash
python backend/osc_video_test.py path/to/video.mp4 --osc-port 9000 --loop
```

`osc_video_test.py` needs a video path and does not accept `--web-port`.

## Project Structure

```text
backend/
  osc_viewer.py      # Local browser viewer
  osc_sender.py      # OSC output, normalization, smoothing
  osc_video_test.py  # Standalone video-to-OSC test, no web UI
  osc_monitor.py     # Standalone OSC monitor
  pose_engine.py     # MediaPipe pose detection and overlay
  dance_metrics.py   # 9 dance metric calculations
frontend/            # Original React dashboard
requirements.txt     # Python dependencies
```

## Development Checks

Useful commands before committing changes:

```bash
python -m compileall backend
git diff --check
git status --short
```
