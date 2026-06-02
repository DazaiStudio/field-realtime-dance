# Real-time Dance Aesthetics Analysis

A sophisticated real-time motion analysis dashboard designed for dance and movement aesthetics. This project leverages MediaPipe's Task API for pose estimation and calculates nine distinct metrics based on biomechanical principles to provide live feedback on a dancer's performance.

![Architecture](https://img.shields.io/badge/Architecture-FastAPI%20%2B%20React%20%2B%20MediaPipe-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Key Features

- **Real-time Pose Estimation**: High-fidelity 33-point body tracking using MediaPipe's Pose Landmarker.
- **Biomechanical Metrics Engine**: Calculates 9 distinct aesthetic indicators based on H36M compatible skeleton data.
- **WebSocket Synchronization**: Low-latency data transmission between the Python backend and React frontend.
- **Interactive Dashboard**: 3x3 grid visualization with real-time charts and value tracking.
- **Dynamic Skeleton Overlay**: Real-time visualization of the tracking skeleton (focusing on torso and limbs).

## 📊 The 9 Aesthetic Metrics

Our analysis engine decomposes movement into nine key indicators as defined in our system's pseudocode:

1.  **Intensity (Energy)**: Sum of limb angular velocities ($rad^2/s$). Reflects the overall physical output.
2.  **Sync - Balance**: The magnitude ratio between left and right limb velocities ([0, 1]). Measures spatial symmetry.
3.  **Sync - Correlation**: Rolling Pearson correlation between left and right side movements ([-1, 1]). Measures temporal synchronicity.
4.  **Volume (Expansion)**: The 3D convex hull volume occupied by the 17 key joints. Reflects spatial extension.
5.  **Roundness (Curvature)**: Geometric curvature ($\kappa$) of the extremities' (wrists/ankles) trajectories.
6.  **Stability - Height**: Vertical level of the body's Center of Mass (CoM).
7.  **Stability - Sway**: Horizontal deviation of the CoM from the Base of Support (mid-point of ankles).
8.  **Effort (Torque)**: Sum of limb angular accelerations ($rad/s^2$). Measures the force required for transitions.
9.  **Smoothness (Jerk)**: Time derivative of acceleration. Higher values indicate more abrupt, less fluid movements.

## 🛠️ Tech Stack

- **Backend**: Python 3.9+, FastAPI, MediaPipe Tasks API, OpenCV, SciPy, NumPy.
- **Frontend**: React (Vite), Tailwind CSS, Recharts (for live data visualization), Lucide React.
- **Communication**: WebSockets (Metrics), MJPEG (Video Stream), and OSC over UDP.

## 📡 OSC Output

The backend sends the 9 dance metrics over OSC at the same cadence as the live metrics broadcast.

- Default target: `udp://127.0.0.1:9000`
- Namespace: `/field`
- Addresses: `/field/energy`, `/field/sync_velocity`, `/field/sync_correlation`, `/field/expansion`, `/field/curvature`, `/field/height`, `/field/sway`, `/field/torque`, `/field/jerk`
- Heartbeat: `/field/heartbeat <timestamp_ms>`
- Modes: `raw` or `normalize`
- Smoothing: `alpha` uses EMA smoothing; `1.0` means no smoothing

Environment variables:

```bash
FIELD_OSC_HOST=127.0.0.1
FIELD_OSC_PORT=9000
FIELD_OSC_ENABLED=1
FIELD_OSC_MODE=raw
FIELD_OSC_ALPHA=1.0
FIELD_OSC_NAMESPACE=/field
```

Runtime API:

```bash
curl http://127.0.0.1:8000/api/osc/status

curl -X POST http://127.0.0.1:8000/api/osc/config \
  -H "Content-Type: application/json" \
  -d "{\"host\":\"127.0.0.1\",\"port\":9000,\"enabled\":true,\"mode\":\"normalize\",\"alpha\":0.35}"
```

Normalization keeps bounded metrics in their natural range, maps `sync_correlation` from `[-1, 1]` to `[0, 1]`, and uses adaptive peak normalization for `energy`, `expansion`, `curvature`, `torque`, and `jerk`.

### Local Input Viewer

Run a local browser-based processor for live camera or video-file tests:

```bash
python backend/osc_viewer.py --osc-port 9000 --web-port 9100
```

Open `http://127.0.0.1:9100`. FIELD Realtime Dance has an input section for `Live Cam` or `Video File`; press **Apply** to process the selected input, show the skeleton overlay, display the 9 metrics, and send OSC at the same time.

The viewer can also change OSC target, prefix, mode, smoothing alpha, and which metrics are sent directly from the page. Uploaded test videos are stored under `backend/viewer_uploads/` and are ignored by git.

Terminal OSC monitor:

```bash
python backend/osc_monitor.py --host 127.0.0.1 --port 9000 --prefix /field
```

The monitor prints each received OSC address and formats float values to two decimals.

### Video-to-OSC Test

Process any input video and send the generated metrics to OSC without using the main frontend:

```bash
python backend/osc_video_test.py path/to/input.mp4 --host 127.0.0.1 --port 9000 --mode normalize --alpha 0.35
```

Recorded clips can also replay their saved metrics through OSC from the app's library using the radio button, or through the API:

```bash
curl -X POST http://127.0.0.1:8000/recordings/dance_YYYYMMDD-HHMMSS.mp4/osc/replay \
  -H "Content-Type: application/json" \
  -d "{\"speed\":1.0,\"loop\":false}"
```

## 🚀 Getting Started

### Quick Start (One-Click Launcher)

The easiest way to run the application is using the provided `start_app.py` script, which automatically handles dependency checks, model downloads, and environment setup:

```bash
python start_app.py
```

### Manual Setup (Development)

#### Backend Setup
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the FastAPI server:
    ```bash
    cd backend
    python app.py
    ```
    The server will start on `http://localhost:8000`.

#### Frontend Setup
1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```
2.  Install packages:
    ```bash
    npm install
    ```
3.  Build or Run:
    - **Development**: `npm run dev`
    - **Production**: `npm run build`

## 📂 Project Structure

```text
├── backend/
│   ├── app.py              # FastAPI & WebSocket server
│   ├── dance_metrics.py    # Metric calculation engine
│   ├── pose_engine.py      # MediaPipe integration & drawing
│   └── constants.py        # H36M joint mappings & weights
├── frontend/
│   ├── src/
│   │   ├── components/     # VideoFeed & MetricsDashboard
│   │   └── AppContent.jsx  # Main application logic
│   └── dist/               # Compiled static assets
├── pseudo_code.md          # Theoretical basis for metrics
└── requirements.txt        # Backend dependencies
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
