import argparse
import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Set
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, StreamingResponse
import uvicorn

from osc_sender import METRIC_NAMES, OSCSender
from pose_engine import PoseEngine


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
UPLOAD_DIR = BASE_DIR / "viewer_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI()
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
    alpha=float(os.getenv("FIELD_OSC_ALPHA", "1.0")),
    namespace=os.getenv("FIELD_OSC_NAMESPACE", "/field"),
)

source_state = {
    "source": "live",
    "camera_index": 0,
    "mirror_live": False,
    "video_path": None,
    "video_name": None,
    "loop": True,
    "target_fps": 24.0,
    "jpeg_quality": 60,
    "width": 960,
    "height": 540,
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
osc_terminal_log = []


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
        "addresses": [f"{osc_sender.namespace}/{name}" for name in source_state["osc_metrics"]],
        "osc_terminal": list(osc_terminal_log),
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

    names = get_windows_camera_names()
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
    sent_messages = osc_sender.send_metrics(metrics, send_keys=set(source_state["osc_metrics"]))
    processing_state["latest_metrics"] = dict(osc_sender.last_prepared_metrics)
    log_osc_messages(sent_messages)


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


def log_osc_messages(messages: list[dict]) -> None:
    if not messages:
        return
    timestamp = time.strftime("%H:%M:%S")
    for message in messages:
        value = message["value"]
        if isinstance(value, float):
            value = f"{value:.2f}"
        osc_terminal_log.append(
            {
                "time": timestamp,
                "address": message["address"],
                "value": value,
            }
        )
    del osc_terminal_log[:-40]


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
        processed, metrics = engine.process_frame(frame, timestamp_ms)
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
        await asyncio.sleep(frame_interval)


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
            processed, metrics = engine.process_frame(frame, timestamp_ms)
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


@app.on_event("shutdown")
def shutdown_event():
    release_camera()
    if pose_engine is not None:
        pose_engine.close()


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
    osc_alpha: float = Form(1.0),
    osc_namespace: str = Form("/field"),
    osc_metrics_selected: bool = Form(False),
    osc_metrics: Optional[list[str]] = Form(None),
):
    osc_sender.configure(
        host=osc_host,
        port=osc_port,
        enabled=osc_enabled,
        mode=osc_mode,
        alpha=osc_alpha,
        namespace=osc_namespace,
    )
    if osc_metrics_selected:
        source_state["osc_metrics"] = [name for name in (osc_metrics or []) if name in METRIC_NAMES]

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
    osc_alpha: float = Form(1.0),
    osc_namespace: str = Form("/field"),
    target_fps: float = Form(24.0),
    jpeg_quality: int = Form(60),
    width: int = Form(960),
    height: int = Form(540),
    osc_metrics_selected: bool = Form(False),
    osc_metrics: Optional[list[str]] = Form(None),
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
    source_state["target_fps"] = max(1.0, min(float(target_fps), 30.0))
    source_state["jpeg_quality"] = max(35, min(int(jpeg_quality), 90))
    source_state["width"] = max(320, min(int(width), 1280))
    source_state["height"] = max(180, min(int(height), 720))
    if osc_metrics_selected:
        selected_metrics = [name for name in (osc_metrics or []) if name in METRIC_NAMES]
    else:
        selected_metrics = list(METRIC_NAMES)
    source_state["osc_metrics"] = selected_metrics

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
    osc_terminal_log.clear()
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
    button {
      cursor: pointer;
      background: var(--amber);
      border-color: #f0a878;
      color: #17120e;
      font-weight: 750;
      align-self: end;
    }
    .check-row { display: flex; align-items: center; gap: 8px; padding-top: 22px; color: var(--text); font-size: 14px; }
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
    .input-row { display: grid; grid-template-columns: 1fr auto 1fr; gap: 12px; align-items: end; }
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
    .or { color: var(--muted); align-self: center; padding-bottom: 11px; font: 12px ui-monospace, monospace; }
    .drop-zone {
      min-height: 66px;
      display: grid;
      place-items: center;
      border: 1px dashed #6b5b4d;
      border-radius: 8px;
      color: var(--muted);
      padding: 10px;
      text-align: center;
      cursor: pointer;
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
    .metric-grid { display: grid; grid-template-columns: 1fr; gap: 9px; padding: 12px; }
    .metric {
      display: grid;
      grid-template-columns: 22px 122px 130px 1fr;
      align-items: center;
      gap: 10px;
      min-height: 46px;
      padding: 9px;
      background: var(--surface-soft);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .name { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; overflow-wrap: anywhere; }
    .value {
      font: 16px ui-monospace, SFMono-Regular, Menlo, monospace;
      text-align: right;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .bar { height: 7px; background: #3a3129; border-radius: 999px; overflow: hidden; }
    .fill { height: 100%; width: 0%; background: linear-gradient(90deg, var(--amber), var(--teal)); transition: width .08s linear; }
    .address-panel { border-top: 1px solid var(--line); padding: 14px; background: #171411; }
    .osc-settings {
      display: grid;
      grid-template-columns: minmax(160px, 1fr) 88px minmax(160px, 1fr) 96px 120px 92px;
      gap: 10px;
      align-items: end;
    }
    .osc-output-grid {
      display: grid;
      grid-template-columns: minmax(220px, .8fr) minmax(320px, 1.2fr);
      gap: 12px;
      margin-top: 12px;
    }
    .address-title-row { display: flex; align-items: end; justify-content: space-between; gap: 12px; margin-top: 12px; }
    .address-title-row .section-title { margin-bottom: 0; }
    .mode-inline { width: 150px; }
    .address-list { display: grid; gap: 6px; color: var(--muted); font: 12px ui-monospace, monospace; }
    .address-list div { overflow-wrap: anywhere; }
    .terminal-details {
      margin-top: 12px;
    }
    .terminal-details summary {
      cursor: pointer;
      color: var(--amber);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .08em;
      list-style-position: inside;
      margin-bottom: 10px;
    }
    .osc-terminal {
      min-height: 168px;
      max-height: 180px;
      overflow-y: auto;
      display: grid;
      align-content: start;
      gap: 4px;
      padding: 10px;
      background: #0d0b09;
      border: 1px solid var(--line);
      border-radius: 8px;
      color: #d8d0c6;
      font: 12px ui-monospace, monospace;
    }
    .osc-terminal div { white-space: pre; overflow: hidden; text-overflow: ellipsis; }
    .metric input { width: 16px; min-height: 16px; }
    @media (max-width: 1080px) {
      .layout { grid-template-columns: 1fr; }
      .controls-grid, .osc-settings, .osc-output-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .apply-row { grid-template-columns: 1fr; }
    }
    @media (max-width: 680px) {
      header { align-items: flex-start; flex-direction: column; }
      .controls-grid, .osc-settings, .osc-output-grid, .meta { grid-template-columns: 1fr; }
      .metric { grid-template-columns: 1fr; }
      .value { text-align: left; }
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
      <div class="status">pose overlay + metrics + OSC</div>
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
                    <span class="mirror-row"><input id="mirrorLive" name="mirror_live" type="checkbox" /> Mirror camera</span>
                  </div>
                </label>
                <div class="or">or</div>
                <label>Video
                  <input id="video" name="video" type="file" accept="video/*" class="hidden" />
                  <div id="dropZone" class="drop-zone">Click to choose video</div>
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
            <label>Alpha
              <input id="oscAlpha" name="osc_alpha" type="number" min="0.01" max="1" step="0.01" value="1" />
            </label>
            <label>Mode
              <select id="oscMode" name="osc_mode">
                <option value="raw">raw</option>
                <option value="normalize">normalize</option>
              </select>
            </label>
            <label class="check-row"><input id="oscEnabled" name="osc_enabled" type="checkbox" checked /> Enabled</label>
          </div>
          <p class="section-title">OSC addresses</p>
          <div id="addresses" class="address-list"></div>
          <details class="terminal-details">
            <summary>OSC terminal</summary>
              <div id="oscTerminal" class="osc-terminal"></div>
          </details>
        </div>
      </section>

      <aside class="panel">
        <div id="metrics" class="metric-grid"></div>
      </aside>
    </section>
    <input id="source" name="source" type="hidden" value="live" />
    <input id="loop" name="loop" type="hidden" value="true" />
    <input id="detectEnabled" name="detect_enabled" type="hidden" value="false" />
    </form>
  </main>
  <script>
    const metricNames = %METRICS%;
    const metricsEl = document.getElementById('metrics');
    const maxSeen = {};

    for (const name of metricNames) {
      maxSeen[name] = 1;
      const row = document.createElement('div');
      row.className = 'metric';
      row.innerHTML = `
        <input class="metric-send" type="checkbox" value="${name}" checked title="Send OSC" />
        <div class="name">${name}</div>
        <div class="value" id="v-${name}">0.00</div>
        <div class="bar"><div class="fill" id="b-${name}"></div></div>
      `;
      metricsEl.appendChild(row);
    }

    async function applySettings(detectEnabled) {
      const form = document.getElementById('form');
      const data = new FormData(form);
      data.set('loop', 'true');
      data.set('detect_enabled', detectEnabled ? 'true' : 'false');
      data.set('osc_enabled', document.getElementById('oscEnabled').checked ? 'true' : 'false');
      data.set('osc_metrics_selected', 'true');
      data.delete('osc_metrics');
      document.querySelectorAll('.metric-send:checked').forEach(input => {
        data.append('osc_metrics', input.value);
      });
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

      return payload;
    }

    function buildOscFormData() {
      const data = new FormData();
      data.set('osc_host', document.getElementById('oscHost').value);
      data.set('osc_port', document.getElementById('oscPort').value);
      data.set('osc_enabled', document.getElementById('oscEnabled').checked ? 'true' : 'false');
      data.set('osc_mode', document.getElementById('oscMode').value);
      data.set('osc_alpha', document.getElementById('oscAlpha').value);
      data.set('osc_namespace', document.getElementById('oscNamespace').value);
      data.set('osc_metrics_selected', 'true');
      document.querySelectorAll('.metric-send:checked').forEach(input => {
        data.append('osc_metrics', input.value);
      });
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
    let oscApplyTimer = null;

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
      dropZone.textContent = 'Click to choose video';
    });
    document.getElementById('mirrorLive').addEventListener('change', () => {
      inputDirty = true;
      sourceInput.value = 'live';
    });
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
      let value = (prefix || '/field').trim().replace(/\/+$/, '');
      if (!value) value = '/field';
      if (!value.startsWith('/')) value = `/${value}`;
      return value;
    }

    function selectedMetricNames() {
      return Array.from(document.querySelectorAll('.metric-send:checked')).map(input => input.value);
    }

    function updateAddresses(payload = lastPayload) {
      const metrics = payload?.processing?.latest_metrics || {};
      const prefix = normalizePrefix(document.getElementById('oscNamespace').value);
      const selected = selectedMetricNames();
      const container = document.getElementById('addresses');
      container.innerHTML = '';
      for (const name of selected) {
        const row = document.createElement('div');
        const value = Number(metrics[name] ?? 0);
        row.textContent = `${prefix}/${name}  ${formatMetric(value)}`;
        container.appendChild(row);
      }
      if (selected.length === 0) {
        const row = document.createElement('div');
        row.textContent = 'No metrics selected';
        container.appendChild(row);
      }
    }

    function updateTerminal(payload) {
      const rows = payload.osc_terminal || [];
      const terminal = document.getElementById('oscTerminal');
      terminal.innerHTML = '';
      if (rows.length === 0) {
        const row = document.createElement('div');
        row.textContent = 'waiting for OSC output...';
        terminal.appendChild(row);
        return;
      }
      for (const item of rows.slice(-18)) {
        const row = document.createElement('div');
        row.textContent = `${item.time}  ${item.address}  ${item.value}`;
        terminal.appendChild(row);
      }
      terminal.scrollTop = terminal.scrollHeight;
    }

    function formatMetric(value) {
      if (!Number.isFinite(value)) return String(value);
      const abs = Math.abs(value);
      if (abs >= 1000000) return value.toExponential(2);
      return value.toFixed(2);
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
      if (!oscDirty) {
        document.getElementById('oscHost').value = osc.host || '127.0.0.1';
        document.getElementById('oscPort').value = osc.port || 9000;
        document.getElementById('oscNamespace').value = osc.namespace || '/field';
        document.getElementById('oscAlpha').value = osc.alpha ?? 1;
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
      updateTerminal(payload);

      for (const name of metricNames) {
        const value = Number(metrics[name] ?? 0);
        maxSeen[name] = Math.max(maxSeen[name] * 0.995, Math.abs(value), 1);
        const valueEl = document.getElementById(`v-${name}`);
        valueEl.textContent = formatMetric(value);
        valueEl.title = Number.isFinite(value) ? value.toFixed(2) : String(value);
        const width = Math.max(0, Math.min(100, Math.abs(value) / maxSeen[name] * 100));
        document.getElementById(`b-${name}`).style.width = `${width}%`;
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
      input.addEventListener('input', () => scheduleOscApply());
      input.addEventListener('change', () => scheduleOscApply(0));
    });
    document.getElementById('oscNamespace').addEventListener('input', () => {
      scheduleOscApply();
      updateAddresses();
    });
    document.querySelectorAll('.metric-send').forEach(input => {
      input.addEventListener('change', () => {
        updateAddresses();
        scheduleOscApply(0);
      });
    });

    loadCameras();

    const ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onmessage = event => update(JSON.parse(event.data));
  </script>
</body>
</html>
""".replace("%METRICS%", json.dumps(list(METRIC_NAMES)))


def main():
    parser = argparse.ArgumentParser(description="Local FIELD input viewer with pose overlay, metrics, and OSC output.")
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=9100)
    parser.add_argument("--osc-host", default=os.getenv("FIELD_OSC_HOST", "127.0.0.1"))
    parser.add_argument("--osc-port", type=int, default=int(os.getenv("FIELD_OSC_PORT", "9000")))
    parser.add_argument("--osc-mode", choices=["raw", "normalize"], default=os.getenv("FIELD_OSC_MODE", "raw"))
    parser.add_argument("--osc-alpha", type=float, default=float(os.getenv("FIELD_OSC_ALPHA", "1.0")))
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
