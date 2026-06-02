import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Set
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, StreamingResponse
import uvicorn

from osc_sender import METRIC_NAMES, OSC_ADDRESS_NAMES, OSCSender
from pose_engine import PoseEngine


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
UPLOAD_DIR = BASE_DIR / "viewer_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

clients: Set[WebSocket] = set()
pose_engine: Optional[PoseEngine] = None
camera = None
camera_cache = {"updated_at": 0.0, "cameras": []}
camera_signal_cache = {"updated_at": 0.0, "signals": {}}
osc_sender = OSCSender(
    host=os.getenv("FIELD_OSC_HOST", "127.0.0.1"),
    port=int(os.getenv("FIELD_OSC_PORT", "9000")),
    enabled=os.getenv("FIELD_OSC_ENABLED", "1") == "1",
    mode=os.getenv("FIELD_OSC_MODE", "raw"),
    alpha=float(os.getenv("FIELD_OSC_ALPHA", "0.25")),
    namespace=os.getenv("FIELD_OSC_NAMESPACE", "/field"),
)

PERFORMANCE_PRESETS = {
    "fast": {"width": 640, "height": 360, "target_fps": 20.0, "jpeg_quality": 55},
    "balanced": {"width": 720, "height": 405, "target_fps": 24.0, "jpeg_quality": 60},
    "quality": {"width": 960, "height": 540, "target_fps": 24.0, "jpeg_quality": 65},
}
DEFAULT_PERFORMANCE = "balanced"

source_state = {
    "source": "live",
    "camera_index": 0,
    "mirror_live": True,
    "video_path": None,
    "video_name": None,
    "loop": True,
    "performance": DEFAULT_PERFORMANCE,
    "overlay_enabled": True,
    "target_fps": PERFORMANCE_PRESETS[DEFAULT_PERFORMANCE]["target_fps"],
    "jpeg_quality": PERFORMANCE_PRESETS[DEFAULT_PERFORMANCE]["jpeg_quality"],
    "width": PERFORMANCE_PRESETS[DEFAULT_PERFORMANCE]["width"],
    "height": PERFORMANCE_PRESETS[DEFAULT_PERFORMANCE]["height"],
    "session_id": 0,
    "osc_metrics": list(METRIC_NAMES),
    "detect_enabled": False,
    "applied_at": time.time(),
}

processing_state = {
    "running": False,
    "frame_count": 0,
    "started_at": None,
    "elapsed_seconds": 0.0,
    "fps": 0.0,
    "latest_metrics": {},
    "latest_raw_metrics": {},
    "latest_timestamp_ms": None,
    "last_frame_at": None,
    "signal_mean": None,
    "error": None,
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    release_camera()
    if pose_engine is not None:
        pose_engine.close()


app = FastAPI(lifespan=lifespan)


def get_pose_engine() -> PoseEngine:
    global pose_engine
    if pose_engine is None:
        pose_engine = PoseEngine(model_path=str(REPO_ROOT / "pose_landmarker_full.task"))
    return pose_engine


def release_camera() -> None:
    global camera
    if camera is not None:
        camera.release()
        camera = None


def open_camera(index: int):
    global camera
    if camera is None:
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        camera = cv2.VideoCapture(index, backend)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(source_state["width"]))
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(source_state["height"]))
    return camera


def state_payload() -> dict:
    payload = {
        "source": dict(source_state),
        "processing": dict(processing_state),
        "osc": osc_sender.get_status(),
        "addresses": [osc_sender.metric_address(name) for name in METRIC_NAMES],
    }
    payload["source"]["video_path"] = None
    return payload


def get_windows_camera_names() -> list[str]:
    if os.name != "nt":
        return []
    command = (
        "Get-CimInstance Win32_PnPEntity | "
        "Where-Object { $_.PNPClass -in @('Camera','Image') } | "
        "Select-Object -ExpandProperty Name"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
    except Exception:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def get_directshow_camera_names() -> list[str]:
    if os.name != "nt":
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph
    except Exception:
        return []
    try:
        return [name.strip() for name in FilterGraph().get_input_devices() if name.strip()]
    except Exception:
        return []


def get_macos_camera_names() -> list[str]:
    if sys.platform != "darwin":
        return []
    try:
        completed = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return []

    names = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line.endswith(":"):
            continue
        name = line[:-1].strip()
        if not name or name in {"Camera", "Cameras"}:
            continue
        if name not in names:
            names.append(name)
    return names


def list_cameras(max_index: int = 4) -> list[dict]:
    if time.time() - camera_cache["updated_at"] < 10 and camera_cache["cameras"]:
        return camera_cache["cameras"]

    directshow_names = get_directshow_camera_names()
    if directshow_names:
        cameras = [
            {
                "index": index,
                "name": name,
                "label": f"{index} - {name}",
                "source": "DirectShow",
            }
            for index, name in enumerate(directshow_names)
        ]
        camera_cache["updated_at"] = time.time()
        camera_cache["cameras"] = cameras
        return cameras

    names = get_macos_camera_names() or get_windows_camera_names()
    cameras = []
    for index in range(max_index):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        available = cap.isOpened()
        cap.release()
        if not available:
            continue
        name = names[len(cameras)] if len(cameras) < len(names) else f"Camera {index}"
        cameras.append({"index": index, "name": name, "label": f"{index} - {name}", "source": "OpenCV"})
    if not cameras:
        cameras.append({"index": 0, "name": "Camera 0", "label": "0 - Camera 0", "source": "Fallback"})
    camera_cache["updated_at"] = time.time()
    camera_cache["cameras"] = cameras
    return cameras


def test_camera_signal(index: int) -> dict:
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    opened = cap.isOpened()
    ok = False
    mean = None
    shape = None
    if opened:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(source_state["width"]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(source_state["height"]))
        for _ in range(4):
            ok, frame = cap.read()
        if ok:
            mean = float(frame.mean())
            shape = [int(frame.shape[1]), int(frame.shape[0])]
    cap.release()

    if not opened:
        status = "unavailable"
    elif not ok:
        status = "no frame"
    elif mean is not None and mean < 3:
        status = "dark"
    else:
        status = "ok"

    return {"index": index, "opened": opened, "read": ok, "mean": mean, "shape": shape, "status": status}


def scan_camera_signals(max_cameras: int = 10) -> dict:
    if time.time() - camera_signal_cache["updated_at"] < 20 and camera_signal_cache["signals"]:
        return camera_signal_cache["signals"]

    release_camera()
    cameras = list_cameras()
    signals = {}
    for camera_info in cameras[:max_cameras]:
        index = int(camera_info["index"])
        signals[str(index)] = test_camera_signal(index)

    camera_signal_cache["updated_at"] = time.time()
    camera_signal_cache["signals"] = signals
    return signals


def set_latest(metrics: dict, timestamp_ms: int, frame=None) -> None:
    if processing_state["started_at"] is None:
        processing_state["started_at"] = time.time()
    processing_state["latest_raw_metrics"] = metrics
    processing_state["latest_timestamp_ms"] = timestamp_ms
    processing_state["last_frame_at"] = time.time()
    processing_state["elapsed_seconds"] = processing_state["last_frame_at"] - processing_state["started_at"]
    processing_state["frame_count"] += 1
    if frame is not None:
        processing_state["signal_mean"] = float(frame.mean())
    if processing_state["elapsed_seconds"] > 0.25:
        processing_state["fps"] = processing_state["frame_count"] / processing_state["elapsed_seconds"]
    if (
        source_state["source"] == "live"
        and processing_state["signal_mean"] is not None
        and processing_state["signal_mean"] < 3
    ):
        processing_state["error"] = "Camera signal is very dark"
    else:
        processing_state["error"] = None
    osc_sender.send_metrics(metrics)
    processing_state["latest_metrics"] = dict(osc_sender.last_prepared_metrics)


def set_preview_frame(frame=None) -> None:
    if processing_state["started_at"] is None:
        processing_state["started_at"] = time.time()
    processing_state["last_frame_at"] = time.time()
    processing_state["elapsed_seconds"] = processing_state["last_frame_at"] - processing_state["started_at"]
    processing_state["frame_count"] += 1
    if frame is not None:
        processing_state["signal_mean"] = float(frame.mean())
    if processing_state["elapsed_seconds"] > 0.25:
        processing_state["fps"] = processing_state["frame_count"] / processing_state["elapsed_seconds"]
    if processing_state["signal_mean"] is not None and processing_state["signal_mean"] < 3:
        processing_state["error"] = "Camera signal is very dark"
    else:
        processing_state["error"] = None


def encode_frame(frame):
    quality = int(source_state["jpeg_quality"])
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
    )


def resize_frame(frame):
    target_width = int(source_state["width"])
    target_height = int(source_state["height"])
    if frame.shape[1] == target_width and frame.shape[0] == target_height:
        return frame
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def apply_live_mirror(frame):
    if source_state.get("mirror_live"):
        return cv2.flip(frame, 1)
    return frame


def apply_performance_preset(performance: str) -> None:
    selected = performance if performance in PERFORMANCE_PRESETS else DEFAULT_PERFORMANCE
    preset = PERFORMANCE_PRESETS[selected]
    source_state["performance"] = selected
    source_state["target_fps"] = preset["target_fps"]
    source_state["jpeg_quality"] = preset["jpeg_quality"]
    source_state["width"] = preset["width"]
    source_state["height"] = preset["height"]


async def stream_live():
    processing_state["running"] = True
    session_id = source_state["session_id"]
    cap = open_camera(int(source_state["camera_index"]))
    engine = get_pose_engine()

    while (
        source_state["source"] == "live"
        and source_state["session_id"] == session_id
        and source_state["detect_enabled"]
    ):
        started = time.time()
        ok, frame = cap.read()
        if not ok:
            processing_state["error"] = "Camera frame not available"
            await asyncio.sleep(0.2)
            continue
        frame = apply_live_mirror(frame)
        frame = resize_frame(frame)

        timestamp_ms = int(time.time() * 1000)
        processed, metrics = engine.process_frame(
            frame,
            timestamp_ms,
            draw_overlay=bool(source_state.get("overlay_enabled", True)),
        )
        set_latest(metrics, timestamp_ms, frame)
        encoded = encode_frame(processed)
        if encoded:
            yield encoded
        frame_interval = 1.0 / max(float(source_state["target_fps"]), 1.0)
        elapsed = time.time() - started
        await asyncio.sleep(max(0.0, frame_interval - elapsed))

    processing_state["running"] = False


async def stream_live_preview():
    session_id = source_state["session_id"]
    cap = open_camera(int(source_state["camera_index"]))

    while (
        source_state["source"] == "live"
        and source_state["session_id"] == session_id
        and not source_state["detect_enabled"]
    ):
        started = time.time()
        ok, frame = cap.read()
        if not ok:
            processing_state["error"] = "Camera frame not available"
            await asyncio.sleep(0.2)
            continue
        frame = apply_live_mirror(frame)
        frame = resize_frame(frame)

        set_preview_frame(frame)
        encoded = encode_frame(frame)
        if encoded:
            yield encoded

        frame_interval = 1.0 / max(float(source_state["target_fps"]), 1.0)
        elapsed = time.time() - started
        await asyncio.sleep(max(0.0, frame_interval - elapsed))


async def stream_video():
    processing_state["running"] = True
    session_id = source_state["session_id"]
    release_camera()
    engine = get_pose_engine()

    while (
        source_state["source"] == "video"
        and source_state["session_id"] == session_id
        and source_state["detect_enabled"]
    ):
        video_path = source_state.get("video_path")
        if not video_path:
            processing_state["error"] = "No video selected"
            await asyncio.sleep(0.5)
            continue

        cap = cv2.VideoCapture(str(video_path))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        target_fps = max(float(source_state["target_fps"]), 1.0)
        frame_skip = max(1, round(source_fps / target_fps))
        frame_interval = frame_skip / max(source_fps, 1.0)
        frame_index = 0

        while (
            source_state["source"] == "video"
            and source_state["session_id"] == session_id
            and source_state["detect_enabled"]
        ):
            started = time.time()
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame_skip > 1 and frame_index % frame_skip != 1:
                continue
            frame = resize_frame(frame)

            timestamp_ms = int(time.time() * 1000)
            processed, metrics = engine.process_frame(
                frame,
                timestamp_ms,
                draw_overlay=bool(source_state.get("overlay_enabled", True)),
            )
            set_latest(metrics, timestamp_ms, frame)
            encoded = encode_frame(processed)
            if encoded:
                yield encoded

            elapsed = time.time() - started
            await asyncio.sleep(max(0.0, frame_interval - elapsed))

        cap.release()
        if not source_state.get("loop"):
            break

    processing_state["running"] = False


@app.get("/")
async def index():
    return HTMLResponse(VIEWER_HTML)


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/api/state")
async def api_state():
    return state_payload()


@app.get("/api/cameras")
async def api_cameras():
    return {"cameras": list_cameras()}


@app.get("/api/cameras/scan")
async def api_camera_scan():
    return {"cameras": list_cameras(), "signals": scan_camera_signals()}


@app.post("/api/osc/config")
async def apply_osc_config(
    osc_host: str = Form("127.0.0.1"),
    osc_port: int = Form(9000),
    osc_enabled: bool = Form(True),
    osc_mode: str = Form("raw"),
    osc_alpha: float = Form(0.25),
    osc_namespace: str = Form("/field"),
):
    osc_sender.configure(
        host=osc_host,
        port=osc_port,
        enabled=osc_enabled,
        mode=osc_mode,
        alpha=osc_alpha,
        namespace=osc_namespace,
    )
    raw_metrics = processing_state.get("latest_raw_metrics") or {}
    if raw_metrics:
        osc_sender.send_metrics(raw_metrics, send_keys=set())
        processing_state["latest_metrics"] = dict(osc_sender.last_prepared_metrics)

    return {"status": "applied", **state_payload()}


@app.post("/api/apply")
async def apply_input(
    source: str = Form("live"),
    camera_index: int = Form(0),
    mirror_live: bool = Form(False),
    loop: bool = Form(True),
    detect_enabled: bool = Form(False),
    osc_host: str = Form("127.0.0.1"),
    osc_port: int = Form(9000),
    osc_enabled: bool = Form(True),
    osc_mode: str = Form("raw"),
    osc_alpha: float = Form(0.25),
    osc_namespace: str = Form("/field"),
    performance: str = Form(DEFAULT_PERFORMANCE),
    overlay_enabled: bool = Form(True),
    video: Optional[UploadFile] = File(None),
):
    if source not in {"live", "video"}:
        return {"status": "error", "error": "source must be live or video"}

    osc_sender.configure(
        host=osc_host,
        port=osc_port,
        enabled=osc_enabled,
        mode=osc_mode,
        alpha=osc_alpha,
        namespace=osc_namespace,
    )

    source_state["session_id"] += 1
    source_state["detect_enabled"] = bool(detect_enabled)
    apply_performance_preset(performance)
    source_state["overlay_enabled"] = bool(overlay_enabled)
    source_state["osc_metrics"] = list(METRIC_NAMES)

    if source == "live":
        release_camera()
        source_state.update(
            {
                "source": "live",
                "camera_index": int(camera_index),
                "mirror_live": bool(mirror_live),
                "video_path": None,
                "video_name": None,
                "loop": bool(loop),
                "applied_at": time.time(),
            }
        )
    else:
        if video is not None and video.filename:
            suffix = Path(video.filename).suffix or ".mp4"
            safe_name = f"{uuid4().hex}{suffix}"
            upload_path = UPLOAD_DIR / safe_name
            with upload_path.open("wb") as f:
                f.write(await video.read())
            source_state["video_path"] = str(upload_path)
            source_state["video_name"] = video.filename
        elif not source_state.get("video_path"):
            return {"status": "error", "error": "video source requires a file"}

        source_state.update(
            {
                "source": "video",
                "camera_index": int(camera_index),
                "mirror_live": bool(mirror_live),
                "loop": bool(loop),
                "applied_at": time.time(),
            }
        )

    processing_state["frame_count"] = 0
    processing_state["started_at"] = None
    processing_state["elapsed_seconds"] = 0.0
    processing_state["fps"] = 0.0
    processing_state["latest_metrics"] = {}
    processing_state["latest_raw_metrics"] = {}
    processing_state["latest_timestamp_ms"] = None
    processing_state["last_frame_at"] = None
    processing_state["signal_mean"] = None
    processing_state["error"] = None
    osc_sender.reset_state()
    return {"status": "applied", **state_payload()}


@app.get("/stream")
async def stream():
    if not source_state["detect_enabled"]:
        return StreamingResponse(iter(()), media_type="multipart/x-mixed-replace; boundary=frame")
    if source_state["source"] == "video":
        generator = stream_video()
    else:
        generator = stream_live()
    return StreamingResponse(generator, media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/preview_stream")
async def preview_stream():
    return StreamingResponse(stream_live_preview(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            await websocket.send_text(json.dumps(state_payload()))
            await asyncio.sleep(1 / 15)
    except WebSocketDisconnect:
        clients.discard(websocket)


VIEWER_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FIELD Realtime Dance</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #141210;
      color: #f4efe6;
      --bg: #141210;
      --surface: #1d1a17;
      --surface-soft: #27221d;
      --line: #3a3129;
      --muted: #a99d91;
      --text: #f4efe6;
      --amber: #d98b5f;
      --teal: #54b3a8;
      --red: #d45d5d;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); }
    main { max-width: 1420px; margin: 0 auto; padding: 18px; }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 14px;
    }
    h1 { margin: 0; font-size: 28px; font-weight: 720; letter-spacing: 0; }
    .status { display: flex; align-items: center; gap: 8px; color: var(--muted); font: 13px ui-monospace, monospace; }
    .dot { width: 10px; height: 10px; border-radius: 999px; background: var(--red); }
    .dot.live { background: var(--teal); box-shadow: 0 0 0 5px rgba(84,179,168,.14); }
    .layout { display: grid; grid-template-columns: minmax(360px, 1fr) 430px; gap: 14px; align-items: start; }
    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .controls { padding: 12px; }
    .section-title {
      margin: 0 0 10px;
      color: var(--amber);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .08em;
    }
    .controls-grid { display: grid; grid-template-columns: 1fr 92px 1fr 110px 120px; gap: 10px; align-items: end; }
    .hint { color: var(--muted); font: 12px ui-monospace, monospace; align-self: center; }
    .hidden { display: none !important; }
    label { display: grid; gap: 6px; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }
    input, select, button {
      width: 100%;
      border: 1px solid var(--line);
      background: #141210;
      color: var(--text);
      border-radius: 8px;
      padding: 10px 11px;
      font-size: 14px;
      min-height: 42px;
    }
    input[type="checkbox"] { width: 18px; min-height: 18px; accent-color: var(--teal); }
    input[type="range"] {
      min-height: 24px;
      padding: 0;
      border: 0;
      accent-color: var(--teal);
      background: transparent;
    }
    button {
      cursor: pointer;
      background: var(--amber);
      border-color: #f0a878;
      color: #17120e;
      font-weight: 750;
      align-self: end;
    }
    .video-wrap { position: relative; background: #090806; aspect-ratio: 16 / 9; }
    #stream, #previewVideo { width: 100%; height: 100%; object-fit: contain; display: block; }
    #previewVideo.hidden, #stream.hidden { display: none; }
    .empty {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      color: var(--muted);
      font: 14px ui-monospace, monospace;
      pointer-events: none;
    }
    .input-overlay {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      padding: 18px;
      background: rgba(9, 8, 6, .76);
      backdrop-filter: blur(4px);
      transition: opacity .18s ease;
    }
    .input-overlay.compact {
      pointer-events: none;
      opacity: 0;
    }
    .input-card {
      width: min(620px, 100%);
      display: grid;
      gap: 14px;
      padding: 18px;
      background: rgba(29, 26, 23, .92);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .input-row { display: grid; grid-template-columns: 1fr auto 1fr; gap: 12px; align-items: start; }
    .camera-row { display: grid; grid-template-columns: 1fr; gap: 8px; align-items: end; }
    .mirror-row {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 22px;
      color: var(--muted);
      font-size: 12px;
      text-transform: none;
      letter-spacing: 0;
    }
    .mirror-row input { width: 15px; min-height: 15px; }
    .or { color: var(--muted); align-self: start; padding-top: 33px; font: 12px ui-monospace, monospace; }
    .drop-zone {
      min-height: 42px;
      height: 42px;
      display: grid;
      place-items: center;
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--muted);
      padding: 10px 11px;
      text-align: center;
      cursor: pointer;
      background: #211f1c;
    }
    .drop-zone.has-file { border-color: var(--amber); color: var(--text); }
    .detect-button {
      position: absolute;
      right: 14px;
      top: 14px;
      width: auto;
      min-width: 112px;
      z-index: 2;
      box-shadow: 0 10px 30px rgba(0,0,0,.22);
    }
    .detect-button.off {
      background: rgba(29, 26, 23, .84);
      color: var(--text);
      border-color: var(--line);
    }
    .enter-button {
      width: auto;
      min-width: 96px;
      justify-self: end;
    }
    .change-input {
      position: absolute;
      left: 14px;
      top: 14px;
      width: auto;
      min-width: 118px;
      z-index: 2;
      background: rgba(29, 26, 23, .84);
      color: var(--text);
      border-color: var(--line);
    }
    .meta {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      border-top: 1px solid var(--line);
      padding: 10px 14px;
      color: var(--muted);
      font: 12px ui-monospace, monospace;
    }
    .meta div { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .metric-osc-controls {
      display: grid;
      grid-template-columns: 126px minmax(150px, 1fr);
      gap: 10px;
      align-items: end;
      padding: 12px;
      border-bottom: 1px solid var(--line);
      background: #171411;
    }
    .metric-toggle-row {
      display: flex;
      align-items: end;
      gap: 8px;
      min-height: 42px;
      padding-bottom: 2px;
      color: var(--text);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: .06em;
    }
    .osc-toggle-row {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 42px;
      color: var(--text);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: .06em;
    }
    .label-row { display: flex; align-items: center; gap: 6px; }
    .info-dot {
      display: inline-grid;
      place-items: center;
      width: 16px;
      height: 16px;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: var(--muted);
      font: 11px ui-monospace, monospace;
      text-transform: none;
      letter-spacing: 0;
      cursor: help;
    }
    .range-label-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .range-value { color: var(--text); font: 12px ui-monospace, monospace; }
    .range-hint { color: var(--muted); font-size: 11px; text-transform: none; letter-spacing: 0; }
    .metric-grid { display: grid; grid-template-columns: 1fr; gap: 9px; padding: 12px; }
    .metric {
      display: grid;
      grid-template-columns: 132px 102px 1fr;
      align-items: center;
      gap: 10px;
      min-height: 52px;
      padding: 9px;
      background: var(--surface-soft);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .metric-label { display: grid; gap: 3px; min-width: 0; }
    .name { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; overflow-wrap: anywhere; }
    .metric-hint { color: #7f756b; font-size: 10px; line-height: 1.15; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .value {
      font: 16px ui-monospace, SFMono-Regular, Menlo, monospace;
      text-align: right;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .bar { position: relative; height: 7px; background: #3a3129; border-radius: 999px; overflow: hidden; }
    .bar.centered::before {
      content: "";
      position: absolute;
      left: 50%;
      top: 0;
      bottom: 0;
      width: 1px;
      background: #7f756b;
      opacity: .75;
      z-index: 1;
    }
    .fill {
      position: absolute;
      left: 0;
      top: 0;
      height: 100%;
      width: 0%;
      background: var(--teal);
      opacity: .78;
      transition: width .08s linear, left .08s linear;
    }
    .address-panel { border-top: 1px solid var(--line); padding: 14px; background: #171411; }
    .osc-settings {
      display: grid;
      grid-template-columns: 150px 88px 160px 130px;
      gap: 10px;
      align-items: end;
    }
    .metric-address-panel { border-top: 1px solid var(--line); padding: 12px; background: #171411; }
    .address-list { display: grid; gap: 6px; color: var(--muted); font: 12px ui-monospace, monospace; }
    .address-list div { overflow-wrap: anywhere; }
    @media (max-width: 1080px) {
      .layout { grid-template-columns: 1fr; }
      .controls-grid, .osc-settings, .metric-osc-controls { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .apply-row { grid-template-columns: 1fr; }
    }
    @media (max-width: 680px) {
      header { align-items: flex-start; flex-direction: column; }
      .controls-grid, .osc-settings, .metric-osc-controls, .meta { grid-template-columns: 1fr; }
      .metric { grid-template-columns: minmax(0, 1fr) auto; }
      .metric .value { text-align: right; }
      .metric .bar { grid-column: 1 / -1; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>FIELD Realtime Dance</h1>
        <div class="status"><span id="dot" class="dot"></span><span id="status">idle</span></div>
      </div>
    </header>

    <form id="form">
    <section class="layout">
      <section class="panel">
        <div class="video-wrap">
          <video id="previewVideo" class="hidden" controls loop muted playsinline></video>
          <img id="stream" class="hidden" alt="Processed pose stream" />
          <div id="inputOverlay" class="input-overlay">
            <div class="input-card">
              <p class="section-title">Input</p>
              <div class="input-row">
                <label>Camera
                  <div class="camera-row">
                    <select id="camera" name="camera_index">
                      <option value="0">0 - Camera 0</option>
                    </select>
                    <span class="mirror-row"><input id="mirrorLive" name="mirror_live" type="checkbox" checked /> Mirror camera</span>
                  </div>
                </label>
                <div class="or">or</div>
                <label>Video
                  <input id="video" name="video" type="file" accept="video/*" class="hidden" />
                  <div id="dropZone" class="drop-zone">Click to upload video</div>
                </label>
              </div>
              <button id="enterInputButton" class="enter-button" type="button">Enter</button>
            </div>
          </div>
          <button id="changeInputButton" class="change-input" type="button">Change Input</button>
          <button id="detectButton" class="detect-button off" type="button">Detect Off</button>
        </div>
        <div class="meta">
          <div id="metaA">source: -</div>
          <div id="metaB">fps: 0.0</div>
          <div id="metaC">input: -</div>
          <div id="metaD">osc: -</div>
        </div>
        <div class="address-panel">
          <p class="section-title">OSC</p>
          <div class="osc-settings">
            <label>Host
              <input id="oscHost" name="osc_host" value="127.0.0.1" />
            </label>
            <label>Port
              <input id="oscPort" name="osc_port" type="number" min="1" max="65535" value="9000" />
            </label>
            <label>Prefix
              <input id="oscNamespace" name="osc_namespace" value="/field" />
            </label>
            <label class="osc-toggle-row"><input id="oscEnabled" name="osc_enabled" type="checkbox" checked /> Enable OSC</label>
          </div>
        </div>
      </section>

      <aside class="panel">
        <div class="metric-osc-controls">
          <label><span class="label-row">Performance <span class="info-dot" title="Fast lowers processing resolution for smoother live detection. Quality keeps a larger processing size.">?</span></span>
            <select id="performance" name="performance">
              <option value="fast">fast</option>
              <option value="balanced" selected>balanced</option>
              <option value="quality">quality</option>
            </select>
          </label>
          <label class="metric-toggle-row" title="Show or hide the skeleton drawing. Detection, metrics, and OSC continue either way.">
            <input id="overlayEnabled" name="overlay_enabled" type="checkbox" checked /> Overlay
          </label>
          <label><span class="label-row">Mode <span class="info-dot" title="raw: send original metric values. normalize: map output toward a bounded 0-1 range using adaptive peaks; sync_correlation maps -1..1 to 0..1.">?</span></span>
            <select id="oscMode" name="osc_mode" title="raw: original metric values. normalize: adaptive bounded output for OSC and display.">
              <option value="raw">raw</option>
              <option value="normalize">normalize</option>
            </select>
          </label>
          <label><span class="range-label-row"><span>Alpha (smooth)</span><span id="oscAlphaValue" class="range-value">0.25</span></span>
            <input id="oscAlpha" name="osc_alpha" type="range" min="0.01" max="1" step="0.01" value="0.25" />
            <span class="range-hint">Lower = smoother</span>
          </label>
        </div>
        <div id="metrics" class="metric-grid"></div>
        <div class="metric-address-panel">
          <p class="section-title">OSC output values</p>
          <div id="addresses" class="address-list"></div>
        </div>
      </aside>
    </section>
    <input id="source" name="source" type="hidden" value="live" />
    <input id="loop" name="loop" type="hidden" value="true" />
    <input id="detectEnabled" name="detect_enabled" type="hidden" value="false" />
    </form>
  </main>
  <script>
    const metricNames = %METRICS%;
    const oscAddressNames = %OSC_ADDRESS_NAMES%;
    const metricsEl = document.getElementById('metrics');
    const maxSeen = {};
    const metricHints = {
      energy: 'more motion',
      sync_velocity: '1 = balanced',
      sync_correlation: 'neutral: 0 raw / .5 norm',
      expansion: 'larger shape',
      curvature: 'more curved',
      height: 'body level',
      sway: 'lower = stable',
      torque: 'more force',
      jerk: 'lower = smoother',
    };
    const metricDescriptions = {
      energy: 'Overall movement intensity from weighted limb angular velocity. Higher means more active motion.',
      sync_velocity: 'Left/right movement magnitude balance. 1 means both sides move with similar velocity; 0 means strongly uneven.',
      sync_correlation: 'Temporal correlation between left and right side movement. Raw range is -1 to 1, with 0 as neutral. Normalize maps it to 0 to 1, where 0.5 is neutral.',
      expansion: 'Body volume / spatial reach from the convex hull of the tracked joints. Higher means a larger body shape.',
      curvature: 'Curvature of wrist and ankle trajectories. Higher means more curved or circular movement paths.',
      height: 'Estimated body center height from the center of mass. Higher/lower depends on body level in the frame.',
      sway: 'Horizontal center-of-mass deviation from the base of support. Lower usually means more stable.',
      torque: 'Movement transition effort from angular acceleration. Higher means sharper forceful transitions.',
      jerk: 'Abruptness cost from changes in acceleration. Lower generally means smoother motion; raw values can be very large.',
    };
    const centeredMetrics = new Set(['sync_correlation']);

    for (const name of metricNames) {
      maxSeen[name] = 1;
      const row = document.createElement('div');
      row.className = 'metric';
      row.title = metricDescriptions[name] || name;
      row.innerHTML = `
        <div class="metric-label" title="${metricDescriptions[name] || name}">
          <div class="name">${name}</div>
          <div class="metric-hint">${metricHints[name] || ''}</div>
        </div>
        <div class="value" id="v-${name}">0.00</div>
        <div class="bar" id="bar-${name}"><div class="fill" id="b-${name}"></div></div>
      `;
      metricsEl.appendChild(row);
    }

    async function applySettings(detectEnabled) {
      const form = document.getElementById('form');
      const data = new FormData(form);
      data.set('loop', 'true');
      data.set('detect_enabled', detectEnabled ? 'true' : 'false');
      data.set('osc_enabled', document.getElementById('oscEnabled').checked ? 'true' : 'false');
      data.set('performance', document.getElementById('performance').value);
      data.set('overlay_enabled', document.getElementById('overlayEnabled').checked ? 'true' : 'false');
      if (document.getElementById('source').value === 'live') {
        data.delete('video');
      }

      const res = await fetch('/api/apply', { method: 'POST', body: data });
      const payload = await res.json();
      if (payload.status !== 'applied') {
        alert(payload.error || 'Apply failed');
        return payload;
      }
      inputDirty = false;
      oscDirty = false;
      runtimeDirty = false;

      return payload;
    }

    function scheduleRuntimeApply() {
      runtimeDirty = true;
      window.clearTimeout(runtimeApplyTimer);
      runtimeApplyTimer = window.setTimeout(async () => {
        const payload = await applySettings(isDetecting);
        if (!payload || payload.status !== 'applied') return;
        if (isDetecting) {
          showDetectionStream();
        } else {
          showPreview();
        }
      }, 80);
    }

    function buildOscFormData() {
      const data = new FormData();
      data.set('osc_host', document.getElementById('oscHost').value);
      data.set('osc_port', document.getElementById('oscPort').value);
      data.set('osc_enabled', document.getElementById('oscEnabled').checked ? 'true' : 'false');
      data.set('osc_mode', document.getElementById('oscMode').value);
      data.set('osc_alpha', document.getElementById('oscAlpha').value);
      data.set('osc_namespace', document.getElementById('oscNamespace').value);
      return data;
    }

    async function applyOscSettings() {
      const res = await fetch('/api/osc/config', { method: 'POST', body: buildOscFormData() });
      const payload = await res.json();
      if (payload.status !== 'applied') {
        console.warn('OSC config failed', payload.error || payload);
        return payload;
      }
      oscDirty = false;
      update(payload);
      return payload;
    }

    document.getElementById('form').addEventListener('submit', async (event) => {
      event.preventDefault();
    });

    async function loadCameras() {
      try {
        const res = await fetch('/api/cameras');
        const payload = await res.json();
        const select = document.getElementById('camera');
        select.innerHTML = '';
        for (const camera of payload.cameras || []) {
          const option = document.createElement('option');
          option.value = camera.index;
          option.textContent = camera.label;
          select.appendChild(option);
        }
      } catch (error) {
        console.warn('Camera list unavailable', error);
      }
    }

    const videoInput = document.getElementById('video');
    const dropZone = document.getElementById('dropZone');
    const sourceInput = document.getElementById('source');
    const detectInput = document.getElementById('detectEnabled');
    const streamImage = document.getElementById('stream');
    const previewVideo = document.getElementById('previewVideo');
    const detectButton = document.getElementById('detectButton');

    let selectedVideoUrl = null;
    let isDetecting = false;
    let lastPayload = null;
    let inputDirty = false;
    let oscDirty = false;
    let runtimeDirty = false;
    let oscApplyTimer = null;
    let runtimeApplyTimer = null;

    function showPreview() {
      streamImage.classList.add('hidden');
      if (sourceInput.value === 'video' && selectedVideoUrl) {
        previewVideo.src = selectedVideoUrl;
        previewVideo.classList.remove('hidden');
        previewVideo.play().catch(() => {});
      } else {
        previewVideo.classList.add('hidden');
        streamImage.classList.remove('hidden');
        streamImage.src = `/preview_stream?t=${Date.now()}`;
      }
    }

    function showDetectionStream() {
      previewVideo.pause();
      previewVideo.classList.add('hidden');
      streamImage.classList.remove('hidden');
      streamImage.src = `/stream?t=${Date.now()}`;
    }

    function setDetectButton(enabled) {
      isDetecting = enabled;
      detectInput.value = enabled ? 'true' : 'false';
      detectButton.textContent = enabled ? 'Detect On' : 'Detect Off';
      detectButton.classList.toggle('off', !enabled);
    }

    document.getElementById('camera').addEventListener('change', () => {
      inputDirty = true;
      sourceInput.value = 'live';
      videoInput.value = '';
      selectedVideoUrl = null;
      previewVideo.removeAttribute('src');
      dropZone.classList.remove('has-file');
      dropZone.textContent = 'Click to upload video';
    });
    document.getElementById('mirrorLive').addEventListener('change', () => {
      inputDirty = true;
      sourceInput.value = 'live';
    });
    document.getElementById('performance').addEventListener('change', scheduleRuntimeApply);
    document.getElementById('overlayEnabled').addEventListener('change', scheduleRuntimeApply);
    dropZone.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      videoInput.click();
    });
    videoInput.addEventListener('change', () => {
      if (videoInput.files.length > 0) {
        inputDirty = true;
        sourceInput.value = 'video';
        if (selectedVideoUrl) URL.revokeObjectURL(selectedVideoUrl);
        selectedVideoUrl = URL.createObjectURL(videoInput.files[0]);
        dropZone.classList.add('has-file');
        dropZone.textContent = videoInput.files[0].name;
        showPreview();
      }
    });
    document.getElementById('changeInputButton').addEventListener('click', () => {
      document.getElementById('inputOverlay').classList.remove('compact');
    });
    document.getElementById('enterInputButton').addEventListener('click', async () => {
      setDetectButton(false);
      const payload = await applySettings(false);
      if (!payload || payload.status !== 'applied') return;
      document.getElementById('inputOverlay').classList.add('compact');
      showPreview();
    });
    detectButton.addEventListener('click', async () => {
      const nextState = !isDetecting;
      setDetectButton(nextState);
      const payload = await applySettings(nextState);
      if (!payload || payload.status !== 'applied') {
        setDetectButton(!nextState);
        return;
      }
      document.getElementById('inputOverlay').classList.add('compact');
      if (nextState) {
        showDetectionStream();
      } else {
        streamImage.removeAttribute('src');
        showPreview();
      }
    });

    function normalizePrefix(prefix) {
      let value = (prefix || '/field').trim().replace(/\\/+$/, '');
      if (!value) value = '/field';
      if (!value.startsWith('/')) value = `/${value}`;
      return value;
    }

    function updateAddresses(payload = lastPayload) {
      const metrics = payload?.processing?.latest_metrics || {};
      const prefix = normalizePrefix(document.getElementById('oscNamespace').value);
      const container = document.getElementById('addresses');
      container.innerHTML = '';
      for (const name of metricNames) {
        const row = document.createElement('div');
        const value = Number(metrics[name] ?? 0);
        row.textContent = `${prefix}/${oscAddressNames[name] || name}  ${formatMetric(value)}`;
        container.appendChild(row);
      }
    }

    function syncAlphaLabel() {
      const alpha = Number(document.getElementById('oscAlpha').value || 0.25);
      document.getElementById('oscAlphaValue').textContent = alpha.toFixed(2);
    }

    function formatMetric(value) {
      if (!Number.isFinite(value)) return String(value);
      const abs = Math.abs(value);
      if (abs >= 1000000) return value.toExponential(2);
      return value.toFixed(2);
    }

    function updateMetricBar(name, value, mode) {
      const bar = document.getElementById(`bar-${name}`);
      const fill = document.getElementById(`b-${name}`);
      if (!bar || !fill) return;

      if (centeredMetrics.has(name) && mode !== 'normalize') {
        bar.classList.add('centered');
        const delta = Math.max(-1, Math.min(1, value));
        const width = Math.abs(delta) * 50;
        fill.style.left = delta < 0 ? `${50 - width}%` : '50%';
        fill.style.width = `${width}%`;
        return;
      }

      bar.classList.remove('centered');
      maxSeen[name] = Math.max(maxSeen[name] * 0.995, Math.abs(value), 1);
      const width = Math.max(0, Math.min(100, Math.abs(value) / maxSeen[name] * 100));
      fill.style.left = '0%';
      fill.style.width = `${width}%`;
    }

    function update(payload) {
      lastPayload = payload;
      const source = payload.source || {};
      const processing = payload.processing || {};
      const osc = payload.osc || {};
      const metrics = processing.latest_metrics || {};
      const age = processing.last_frame_at ? (Date.now() / 1000) - processing.last_frame_at : Infinity;
      if (!inputDirty) {
        document.getElementById('mirrorLive').checked = Boolean(source.mirror_live);
      }
      if (!runtimeDirty) {
        document.getElementById('performance').value = source.performance || 'balanced';
        document.getElementById('overlayEnabled').checked = source.overlay_enabled !== false;
      }
      if (!oscDirty) {
        document.getElementById('oscHost').value = osc.host || '127.0.0.1';
        document.getElementById('oscPort').value = osc.port || 9000;
        document.getElementById('oscNamespace').value = osc.namespace || '/field';
        document.getElementById('oscAlpha').value = osc.alpha ?? 0.25;
        syncAlphaLabel();
        document.getElementById('oscMode').value = osc.mode || 'raw';
        document.getElementById('oscEnabled').checked = Boolean(osc.enabled);
      }

      document.getElementById('dot').className = age < 2 ? 'dot live' : 'dot';
      document.getElementById('status').textContent =
        processing.error || (source.detect_enabled ? (age < 2 ? 'detecting' : 'waiting') : 'detect off');
      document.getElementById('metaA').textContent = `source: ${source.source || '-'}`;
      document.getElementById('metaB').textContent = `fps: ${Number(processing.fps || 0).toFixed(1)}`;
      if (source.source === 'video') {
        document.getElementById('metaC').textContent = `time: ${Number(processing.elapsed_seconds || 0).toFixed(1)}s`;
        document.getElementById('metaD').textContent = `file: ${source.video_name || '-'} / loop ${source.loop ? 'on' : 'off'}`;
      } else {
        const cameraSelect = document.getElementById('camera');
        const cameraLabel = cameraSelect.options[cameraSelect.selectedIndex]?.textContent || source.camera_index || '-';
        document.getElementById('metaC').textContent = `camera: ${cameraLabel}`;
        document.getElementById('metaD').textContent = `mirror: ${source.mirror_live ? 'on' : 'off'} / osc: ${osc.enabled ? 'on' : 'off'} ${osc.host || '-'}:${osc.port || '-'}`;
      }
      updateAddresses(payload);

      for (const name of metricNames) {
        const value = Number(metrics[name] ?? 0);
        const valueEl = document.getElementById(`v-${name}`);
        valueEl.textContent = formatMetric(value);
        valueEl.title = Number.isFinite(value) ? value.toFixed(2) : String(value);
        updateMetricBar(name, value, osc.mode || 'raw');
      }
    }

    function markOscDirty() {
      oscDirty = true;
    }

    function scheduleOscApply(delay = 300) {
      markOscDirty();
      window.clearTimeout(oscApplyTimer);
      oscApplyTimer = window.setTimeout(() => {
        applyOscSettings().catch(error => console.warn('OSC config failed', error));
      }, delay);
    }

    ['oscHost', 'oscPort', 'oscAlpha', 'oscMode', 'oscEnabled'].forEach(id => {
      const input = document.getElementById(id);
      input.addEventListener('input', () => {
        if (id === 'oscAlpha') syncAlphaLabel();
        scheduleOscApply();
      });
      input.addEventListener('change', () => scheduleOscApply(0));
    });
    document.getElementById('oscNamespace').addEventListener('input', () => {
      scheduleOscApply();
      updateAddresses();
    });
    loadCameras();

    const ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onmessage = event => update(JSON.parse(event.data));
  </script>
</body>
</html>
""".replace("%METRICS%", json.dumps(list(METRIC_NAMES))).replace(
    "%OSC_ADDRESS_NAMES%", json.dumps(OSC_ADDRESS_NAMES)
)


def main():
    parser = argparse.ArgumentParser(description="Local FIELD input viewer with pose overlay, metrics, and OSC output.")
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=9100)
    parser.add_argument("--osc-host", default=os.getenv("FIELD_OSC_HOST", "127.0.0.1"))
    parser.add_argument("--osc-port", type=int, default=int(os.getenv("FIELD_OSC_PORT", "9000")))
    parser.add_argument("--osc-mode", choices=["raw", "normalize"], default=os.getenv("FIELD_OSC_MODE", "raw"))
    parser.add_argument("--osc-alpha", type=float, default=float(os.getenv("FIELD_OSC_ALPHA", "0.25")))
    parser.add_argument("--osc-namespace", default=os.getenv("FIELD_OSC_NAMESPACE", "/field"))
    args = parser.parse_args()

    osc_sender.configure(
        host=args.osc_host,
        port=args.osc_port,
        mode=args.osc_mode,
        alpha=args.osc_alpha,
        namespace=args.osc_namespace,
    )
    print(f"FIELD input viewer: http://{args.web_host}:{args.web_port}")
    print(
        f"OSC output: udp://{args.osc_host}:{args.osc_port} "
        f"prefix={args.osc_namespace} mode={args.osc_mode} alpha={args.osc_alpha}"
    )
    uvicorn.run(app, host=args.web_host, port=args.web_port)


if __name__ == "__main__":
    main()
