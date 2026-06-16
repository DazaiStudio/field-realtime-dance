import argparse
import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, StreamingResponse
import uvicorn

from culture_score import CultureScore
from osc_sender import METRIC_NAMES, OSC_ADDRESS_NAMES, OSCSender
from pose_engine import PoseEngine


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
UPLOAD_DIR = BASE_DIR / "viewer_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

pose_engine: Optional[PoseEngine] = None
pose_model_path = REPO_ROOT / "pose_landmarker_lite.task"
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
# Optional: offline culture centroids enable the /field/morrisness output.
culture_score = CultureScore.try_load(BASE_DIR / "culture_map.json")

PERFORMANCE_PRESETS = {
    "fast": {"width": 640, "height": 360, "target_fps": 24.0, "analysis_fps": 10.0, "jpeg_quality": 55},
    "balanced": {"width": 720, "height": 405, "target_fps": 24.0, "analysis_fps": 12.0, "jpeg_quality": 60},
    "quality": {"width": 960, "height": 540, "target_fps": 20.0, "analysis_fps": 15.0, "jpeg_quality": 65},
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
    "analysis_fps": PERFORMANCE_PRESETS[DEFAULT_PERFORMANCE]["analysis_fps"],
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
    "analysis_count": 0,
    "analysis_fps": 0.0,
    "pose_ms": 0.0,
    "encode_ms": 0.0,
    "latest_metrics": {},
    "latest_raw_metrics": {},
    "latest_timestamp_ms": None,
    "last_frame_at": None,
    "signal_mean": None,
    "morrisness": None,
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
        pose_engine = PoseEngine(model_path=str(pose_model_path))
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

    # Never probe the camera a stream is currently using: opening it a second
    # time fails on most backends and releasing it would kill the live feed.
    active_index = int(source_state["camera_index"]) if camera is not None else None
    cameras = list_cameras()
    signals = {}
    for camera_info in cameras[:max_cameras]:
        index = int(camera_info["index"])
        if index == active_index:
            signals[str(index)] = {
                "index": index,
                "opened": True,
                "read": True,
                "mean": processing_state.get("signal_mean"),
                "shape": None,
                "status": "in use",
            }
            continue
        signals[str(index)] = test_camera_signal(index)

    camera_signal_cache["updated_at"] = time.time()
    camera_signal_cache["signals"] = signals
    return signals


def set_stream_frame(frame=None, encode_ms: float = 0.0) -> None:
    if processing_state["started_at"] is None:
        processing_state["started_at"] = time.time()
    processing_state["last_frame_at"] = time.time()
    processing_state["elapsed_seconds"] = processing_state["last_frame_at"] - processing_state["started_at"]
    processing_state["frame_count"] += 1
    processing_state["encode_ms"] = float(encode_ms)
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


def set_analysis_result(metrics: dict, timestamp_ms: int, pose_ms: float = 0.0) -> None:
    if processing_state["started_at"] is None:
        processing_state["started_at"] = time.time()
    processing_state["latest_raw_metrics"] = metrics
    processing_state["latest_timestamp_ms"] = timestamp_ms
    processing_state["analysis_count"] += 1
    processing_state["pose_ms"] = float(pose_ms)
    elapsed = time.time() - processing_state["started_at"]
    if elapsed > 0.25:
        processing_state["analysis_fps"] = processing_state["analysis_count"] / elapsed
    osc_sender.send_metrics(metrics)
    processing_state["latest_metrics"] = dict(osc_sender.last_prepared_metrics)
    if culture_score is not None:
        # All-zero metrics mean "no pose detected"; keep the previous score
        # instead of letting zeros drag the rolling average around.
        if metrics.get("energy", 0.0) != 0.0 or metrics.get("expansion", 0.0) != 0.0:
            morrisness = culture_score.update(metrics)
            if morrisness is not None:
                processing_state["morrisness"] = morrisness
                osc_sender.send_named("morrisness", morrisness)


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
    source_state["analysis_fps"] = preset["analysis_fps"]
    source_state["jpeg_quality"] = preset["jpeg_quality"]
    source_state["width"] = preset["width"]
    source_state["height"] = preset["height"]


async def stream_live():
    processing_state["running"] = True
    session_id = source_state["session_id"]
    cap = await asyncio.to_thread(open_camera, int(source_state["camera_index"]))
    engine = get_pose_engine()
    next_analysis_at = 0.0
    last_analysis_at = None
    measured_fps = max(float(source_state["analysis_fps"]), 1.0)

    while (
        source_state["source"] == "live"
        and source_state["session_id"] == session_id
        and source_state["detect_enabled"]
    ):
        started = time.time()
        ok, frame = await asyncio.to_thread(cap.read)
        if not ok:
            processing_state["error"] = "Camera frame not available"
            await asyncio.sleep(0.2)
            continue
        frame = apply_live_mirror(frame)
        frame = resize_frame(frame)

        timestamp_ms = int(time.time() * 1000)
        processed = frame
        now = time.time()
        analysis_interval = 1.0 / max(float(source_state["analysis_fps"]), 1.0)
        if now >= next_analysis_at:
            # The metrics engine divides by dt, so feed it the measured
            # analysis rate (EMA-smoothed) instead of the scheduled one;
            # otherwise energy/torque/jerk scale with machine load.
            if last_analysis_at is not None and now > last_analysis_at:
                measured_fps = 0.8 * measured_fps + 0.2 * (1.0 / (now - last_analysis_at))
            last_analysis_at = now
            engine.set_metrics_fps(measured_fps)
            pose_started = time.perf_counter()
            processed, metrics = await asyncio.to_thread(
                engine.process_frame,
                frame,
                timestamp_ms,
                draw_overlay=bool(source_state.get("overlay_enabled", True)),
            )
            pose_ms = (time.perf_counter() - pose_started) * 1000.0
            set_analysis_result(metrics, timestamp_ms, pose_ms)
            next_analysis_at = now + analysis_interval
        elif source_state.get("overlay_enabled", True):
            processed = engine.draw_cached_overlay(frame)

        encode_started = time.perf_counter()
        encoded = await asyncio.to_thread(encode_frame, processed)
        encode_ms = (time.perf_counter() - encode_started) * 1000.0
        set_stream_frame(frame, encode_ms)
        if encoded:
            yield encoded
        frame_interval = 1.0 / max(float(source_state["target_fps"]), 1.0)
        elapsed = time.time() - started
        await asyncio.sleep(max(0.0, frame_interval - elapsed))

    processing_state["running"] = False


async def stream_live_preview():
    session_id = source_state["session_id"]
    cap = await asyncio.to_thread(open_camera, int(source_state["camera_index"]))

    while (
        source_state["source"] == "live"
        and source_state["session_id"] == session_id
        and not source_state["detect_enabled"]
    ):
        started = time.time()
        ok, frame = await asyncio.to_thread(cap.read)
        if not ok:
            processing_state["error"] = "Camera frame not available"
            await asyncio.sleep(0.2)
            continue
        frame = apply_live_mirror(frame)
        frame = resize_frame(frame)

        encode_started = time.perf_counter()
        encoded = await asyncio.to_thread(encode_frame, frame)
        encode_ms = (time.perf_counter() - encode_started) * 1000.0
        set_stream_frame(frame, encode_ms)
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

        cap = await asyncio.to_thread(cv2.VideoCapture, str(video_path))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        target_fps = max(float(source_state["target_fps"]), 1.0)
        frame_skip = max(1, round(source_fps / target_fps))
        frame_interval = frame_skip / max(source_fps, 1.0)
        frame_index = 0
        next_analysis_at = 0.0
        last_analysis_at = None
        measured_fps = max(float(source_state["analysis_fps"]), 1.0)

        while (
            source_state["source"] == "video"
            and source_state["session_id"] == session_id
            and source_state["detect_enabled"]
        ):
            started = time.time()
            ok, frame = await asyncio.to_thread(cap.read)
            if not ok:
                break
            frame_index += 1
            if frame_skip > 1 and frame_index % frame_skip != 1:
                continue
            frame = resize_frame(frame)

            timestamp_ms = int(time.time() * 1000)
            processed = frame
            now = time.time()
            analysis_interval = 1.0 / max(float(source_state["analysis_fps"]), 1.0)
            if now >= next_analysis_at:
                # Same as stream_live: keep the metric time base honest by
                # feeding the measured analysis rate, not the scheduled one.
                if last_analysis_at is not None and now > last_analysis_at:
                    measured_fps = 0.8 * measured_fps + 0.2 * (1.0 / (now - last_analysis_at))
                last_analysis_at = now
                engine.set_metrics_fps(measured_fps)
                pose_started = time.perf_counter()
                processed, metrics = await asyncio.to_thread(
                    engine.process_frame,
                    frame,
                    timestamp_ms,
                    draw_overlay=bool(source_state.get("overlay_enabled", True)),
                )
                pose_ms = (time.perf_counter() - pose_started) * 1000.0
                set_analysis_result(metrics, timestamp_ms, pose_ms)
                next_analysis_at = now + analysis_interval
            elif source_state.get("overlay_enabled", True):
                processed = engine.draw_cached_overlay(frame)

            encode_started = time.perf_counter()
            encoded = await asyncio.to_thread(encode_frame, processed)
            encode_ms = (time.perf_counter() - encode_started) * 1000.0
            set_stream_frame(frame, encode_ms)
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


@app.post("/api/camera/release")
async def api_camera_release():
    """Stop any running stream and release the camera (UI camera-off)."""
    source_state["session_id"] += 1
    source_state["detect_enabled"] = False
    await asyncio.to_thread(release_camera)
    processing_state["error"] = None
    return {"status": "released", **state_payload()}


@app.post("/api/shutdown")
async def api_shutdown():
    """Stop streams, release the camera, then exit the process (UI Quit button)."""
    source_state["session_id"] += 1
    source_state["detect_enabled"] = False
    try:
        await asyncio.to_thread(release_camera)
    except Exception:
        pass
    # Delay the hard exit so the HTTP response reaches the browser first.
    threading.Timer(0.5, lambda: os._exit(0)).start()
    return {"status": "stopping"}


@app.get("/api/cameras")
async def api_cameras():
    return {"cameras": await asyncio.to_thread(list_cameras)}


@app.get("/api/cameras/scan")
async def api_camera_scan():
    cameras = await asyncio.to_thread(list_cameras)
    signals = await asyncio.to_thread(scan_camera_signals)
    return {"cameras": cameras, "signals": signals}


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
    source_state["overlay_enabled"] = True
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
    processing_state["analysis_count"] = 0
    processing_state["analysis_fps"] = 0.0
    processing_state["pose_ms"] = 0.0
    processing_state["encode_ms"] = 0.0
    processing_state["latest_metrics"] = {}
    processing_state["latest_raw_metrics"] = {}
    processing_state["latest_timestamp_ms"] = None
    processing_state["last_frame_at"] = None
    processing_state["signal_mean"] = None
    processing_state["morrisness"] = None
    processing_state["error"] = None
    osc_sender.reset_state()
    if culture_score is not None:
        culture_score.reset()
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
    try:
        while True:
            await websocket.send_text(json.dumps(state_payload()))
            await asyncio.sleep(1 / 15)
    except WebSocketDisconnect:
        pass


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
      font-family: "Bahnschrift", "Avenir Next", "Segoe UI", ui-sans-serif, system-ui, sans-serif;
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
    body {
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(1100px 500px at 18% -10%, rgba(217,139,95,.07), transparent 60%),
        radial-gradient(900px 480px at 100% 0%, rgba(84,179,168,.05), transparent 55%),
        var(--bg);
      min-height: 100vh;
    }
    main { max-width: 1420px; margin: 0 auto; padding: 18px; }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 14px;
    }
    .quit-btn {
      align-self: center;
      width: auto;
      flex: none;
      min-height: 0;
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 7px 14px;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: .06em;
      background: rgba(29, 26, 23, .84);
      color: var(--muted);
      border: 1px solid var(--line);
      border-radius: 7px;
    }
    .quit-btn:hover { background: rgba(54, 47, 40, .92); color: var(--text); }
    .quit-btn svg { width: 16px; height: 16px; display: block; }
    .quit-btn.confirm { color: var(--red); border-color: var(--red); background: rgba(60, 20, 16, .5); }
    .quit-btn.confirm:hover { background: rgba(80, 26, 20, .65); }
    .stopped-overlay {
      position: fixed;
      inset: 0;
      z-index: 50;
      display: grid;
      place-items: center;
      background: rgba(9, 8, 6, .92);
      backdrop-filter: blur(6px);
    }
    .stopped-overlay.hidden { display: none; }
    .stopped-title { font-size: 22px; color: var(--text); margin: 0 0 8px; letter-spacing: .04em; text-align: center; }
    .stopped-sub { color: var(--muted); margin: 0; font: 14px ui-monospace, monospace; text-align: center; }
    h1 {
      margin: 0;
      font-size: 30px;
      font-weight: 700;
      letter-spacing: .16em;
      text-transform: uppercase;
    }
    h1 .h1-sub {
      font-size: 13px;
      font-weight: 400;
      letter-spacing: .22em;
      color: var(--muted);
      margin-left: 10px;
      vertical-align: 4px;
    }
    .status { display: flex; align-items: center; gap: 8px; color: var(--muted); font: 14px ui-monospace, monospace; }
    .dot { width: 10px; height: 10px; border-radius: 999px; background: var(--red); }
    .dot.live { background: var(--teal); animation: pulse 2.2s ease-in-out infinite; }
    @keyframes pulse {
      0%, 100% { box-shadow: 0 0 0 4px rgba(84,179,168,.16); }
      50% { box-shadow: 0 0 0 8px rgba(84,179,168,.05); }
    }
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
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: .08em;
    }
    .controls-grid { display: grid; grid-template-columns: 1fr 92px 1fr 110px 120px; gap: 10px; align-items: end; }
    .hint { color: var(--muted); font: 13px ui-monospace, monospace; align-self: center; }
    .hidden { display: none !important; }
    label { display: grid; gap: 6px; color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: .06em; }
    input, select, button {
      width: 100%;
      border: 1px solid var(--line);
      background: #141210;
      color: var(--text);
      border-radius: 8px;
      padding: 10px 11px;
      font-size: 14px;
      min-height: 42px;
      font-family: inherit;
      transition: border-color .15s ease, background-color .15s ease;
    }
    input:focus-visible, select:focus-visible, button:focus-visible {
      outline: 2px solid var(--teal);
      outline-offset: 1px;
    }
    input:hover, select:hover { border-color: #50453a; }
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
      font-weight: 700;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: .07em;
      align-self: end;
    }
    button:hover { background: #e29a6f; }
    button:active { background: #c97e54; }
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
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: .06em;
    }
    .mirror-row input { width: 15px; min-height: 15px; }
    .or { color: var(--muted); align-self: start; padding-top: 33px; font: 13px ui-monospace, monospace; }
    .drop-zone {
      min-height: 42px;
      height: 42px;
      display: grid;
      place-items: center;
      border: 1px dashed var(--line);
      border-radius: 8px;
      color: var(--muted);
      padding: 10px 11px;
      text-align: center;
      cursor: pointer;
      background: #211f1c;
      text-transform: none;
      letter-spacing: 0;
      font-size: 13px;
      transition: border-color .15s ease, color .15s ease;
    }
    .drop-zone:hover { border-color: var(--amber); color: var(--text); }
    .drop-zone.has-file { border: 1px solid var(--amber); color: var(--text); }
    .control-bar {
      position: absolute;
      left: 50%;
      bottom: 16px;
      transform: translateX(-50%);
      display: flex;
      gap: 14px;
      z-index: 2;
    }
    .ctrl-btn {
      width: 54px;
      height: 54px;
      min-height: 54px;
      padding: 0;
      border-radius: 999px;
      display: grid;
      place-items: center;
      background: rgba(29, 26, 23, .88);
      border: 1px solid #4a4036;
      color: var(--teal);
      box-shadow: 0 10px 30px rgba(0,0,0,.4);
    }
    .ctrl-btn:hover { background: rgba(54, 47, 40, .92); }
    .ctrl-btn:active { background: rgba(29, 26, 23, .88); }
    .ctrl-btn svg { width: 26px; height: 26px; display: block; }
    .ctrl-btn .slash { display: none; stroke: var(--red); }
    .ctrl-btn.off { color: var(--muted); }
    .ctrl-btn.off .slash { display: block; }
    #detectButton:not(.off) { color: var(--amber); border-color: #6b543f; }
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
    .fs-btn {
      position: absolute;
      right: 14px;
      bottom: 16px;
      width: 40px;
      height: 40px;
      min-height: 40px;
      padding: 0;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: rgba(29, 26, 23, .78);
      border: 1px solid #4a4036;
      color: var(--text);
      box-shadow: 0 8px 24px rgba(0,0,0,.4);
      opacity: .85;
      z-index: 3;
    }
    .fs-btn:hover { background: rgba(54, 47, 40, .92); opacity: 1; }
    .fs-btn svg { width: 22px; height: 22px; display: block; }
    .fs-btn .fs-exit { display: none; }
    .fs-btn.fs-active .fs-enter { display: none; }
    .fs-btn.fs-active .fs-exit { display: block; }
    .video-wrap:fullscreen,
    .video-wrap:-webkit-full-screen {
      aspect-ratio: auto;
      width: 100vw;
      height: 100vh;
      background: #000;
    }
    .metric-overlay {
      position: absolute;
      top: 16px;
      right: 16px;
      display: none;
      flex-direction: column;
      gap: 5px;
      min-width: 230px;
      padding: 14px 18px;
      background: rgba(9, 8, 6, .58);
      border: 1px solid rgba(74, 64, 54, .6);
      border-radius: 9px;
      font: 15px ui-monospace, monospace;
      backdrop-filter: blur(3px);
      pointer-events: none;
      z-index: 3;
    }
    .metric-overlay .ov-row { display: flex; justify-content: space-between; gap: 22px; }
    .metric-overlay .ov-name { color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
    .metric-overlay .ov-val { color: var(--teal); font-variant-numeric: tabular-nums; }
    .video-wrap:fullscreen .metric-overlay,
    .video-wrap:-webkit-full-screen .metric-overlay { display: flex; }
    .meta {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      border-top: 1px solid var(--line);
      padding: 10px 14px;
      color: var(--muted);
      font: 13px ui-monospace, monospace;
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
    .osc-toggle-row {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 42px;
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: .06em;
    }
    .label-row { display: flex; align-items: center; gap: 6px; }
    .info-dot {
      display: inline-grid;
      place-items: center;
      width: 18px;
      height: 18px;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: var(--muted);
      font: 12px ui-monospace, monospace;
      text-transform: none;
      letter-spacing: 0;
      cursor: help;
    }
    .range-label-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .range-value { color: var(--text); font: 12px ui-monospace, monospace; }
    .range-hint { color: var(--muted); font-size: 12px; text-transform: none; letter-spacing: 0; }
    .metric-grid { display: grid; grid-template-columns: 1fr; gap: 9px; padding: 12px; }
    .metric {
      display: grid;
      grid-template-columns: 162px 96px 1fr;
      align-items: center;
      gap: 10px;
      min-height: 52px;
      padding: 9px;
      background: var(--surface-soft);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .metric-label { display: grid; gap: 3px; min-width: 0; }
    .name { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; overflow-wrap: anywhere; }
    .metric-hint { color: #93887c; font-size: 12px; line-height: 1.15; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .value {
      font: 16px ui-monospace, SFMono-Regular, Menlo, monospace;
      font-variant-numeric: tabular-nums;
      text-align: right;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .bar { position: relative; height: 8px; background: #2e2822; border-radius: 999px; overflow: hidden; box-shadow: inset 0 1px 2px rgba(0,0,0,.35); }
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
    .culture-panel { border-top: 1px solid var(--line); padding: 14px 12px 16px; background: #171411; }
    .culture-row { display: flex; align-items: center; gap: 12px; margin-top: 4px; }
    .culture-end {
      font: 11px ui-monospace, monospace;
      letter-spacing: .12em;
      flex-shrink: 0;
    }
    .culture-end.baye { color: var(--red); }
    .culture-end.morris { color: var(--teal); }
    .culture-track {
      position: relative;
      flex: 1;
      height: 10px;
      border-radius: 999px;
      background: linear-gradient(90deg, rgba(212,93,93,.55), rgba(58,49,41,.9) 50%, rgba(84,179,168,.55));
      box-shadow: inset 0 1px 2px rgba(0,0,0,.4);
    }
    .culture-track::before {
      content: "";
      position: absolute;
      left: 50%;
      top: -3px;
      bottom: -3px;
      width: 1px;
      background: #7f756b;
      opacity: .6;
    }
    .culture-marker {
      position: absolute;
      top: 50%;
      left: 50%;
      width: 16px;
      height: 16px;
      border-radius: 999px;
      background: var(--text);
      border: 3px solid var(--bg);
      box-shadow: 0 0 10px rgba(244,239,230,.45);
      transform: translate(-50%, -50%);
      transition: left .25s ease;
    }
    .culture-value {
      margin-top: 8px;
      text-align: center;
      color: var(--muted);
      font: 12px ui-monospace, monospace;
      font-variant-numeric: tabular-nums;
    }
    .address-panel { border-top: 1px solid var(--line); padding: 14px; background: #171411; }
    .osc-settings {
      display: grid;
      grid-template-columns: 150px 88px 160px 130px;
      gap: 10px;
      align-items: end;
    }
    .metric-address-panel { border-top: 1px solid var(--line); padding: 12px; background: #171411; }
    .address-list { display: grid; gap: 7px; color: var(--muted); font: 13px ui-monospace, monospace; }
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
        <h1>FIELD<span class="h1-sub">Realtime Dance</span></h1>
        <div class="status"><span id="dot" class="dot"></span><span id="status">idle</span></div>
      </div>
      <button id="quitButton" class="quit-btn" type="button" title="Stop the viewer and shut down the server">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 3.5v8"/>
          <path d="M6.8 6.8a8 8 0 1 0 10.4 0"/>
        </svg>
        <span id="quitLabel">Quit</span>
      </button>
    </header>
    <div id="stoppedOverlay" class="stopped-overlay hidden">
      <div>
        <p class="stopped-title">Viewer stopped</p>
        <p class="stopped-sub">You can close this tab.</p>
      </div>
    </div>

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
          <div id="emptyState" class="empty hidden">camera off</div>
          <div id="metricOverlay" class="metric-overlay"></div>
          <button id="changeInputButton" class="change-input" type="button">Change Input</button>
          <div class="control-bar">
            <button id="cameraButton" class="ctrl-btn off" type="button" title="Turn camera on">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <rect x="2.5" y="6.5" width="13" height="11" rx="2.5"/>
                <path d="M15.5 10.5 21 7.5v9l-5.5-3"/>
                <line class="slash" x1="3.5" y1="20.5" x2="20.5" y2="3.5"/>
              </svg>
            </button>
            <button id="detectButton" class="ctrl-btn off" type="button" title="Turn detection on">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="4.4" r="2.1"/>
                <line x1="12" y1="6.8" x2="12" y2="13"/>
                <path d="M6 10.2 12 8.2l6 2"/>
                <path d="M8.2 20.5 12 13l3.8 7.5"/>
                <line class="slash" x1="3.5" y1="20.5" x2="20.5" y2="3.5"/>
              </svg>
            </button>
          </div>
          <button id="fullscreenButton" class="fs-btn" type="button" title="Fullscreen">
            <svg class="fs-enter" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
              <path d="M8 4H5a1 1 0 0 0-1 1v3"/>
              <path d="M16 4h3a1 1 0 0 1 1 1v3"/>
              <path d="M16 20h3a1 1 0 0 0 1-1v-3"/>
              <path d="M8 20H5a1 1 0 0 1-1-1v-3"/>
            </svg>
            <svg class="fs-exit" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
              <path d="M8 4v3a1 1 0 0 1-1 1H4"/>
              <path d="M16 4v3a1 1 0 0 0 1 1h3"/>
              <path d="M16 20v-3a1 1 0 0 1 1-1h3"/>
              <path d="M8 20v-3a1 1 0 0 0-1-1H4"/>
            </svg>
          </button>
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
        <div class="metric-address-panel">
          <p class="section-title">Output values</p>
          <div id="addresses" class="address-list"></div>
        </div>
      </section>

      <aside class="panel">
        <div class="metric-osc-controls">
          <label><span class="label-row">Mode <span class="info-dot" title="raw: values in their original units. normalize: every metric mapped into 0-1 with an adaptive range (sync_corr -1..1 becomes 0..1, 0.5 neutral).">?</span></span>
            <select id="oscMode" name="osc_mode">
              <option value="raw">raw</option>
              <option value="normalize">normalize</option>
            </select>
          </label>
          <label><span class="range-label-row"><span class="label-row">Smoothing <span class="info-dot" title="Exponential smoothing applied to every OSC output. 1 = no smoothing; lower values respond more slowly but read steadier.">?</span></span><span id="oscAlphaValue" class="range-value">0.25</span></span>
            <input id="oscAlpha" name="osc_alpha" type="range" min="0.01" max="1" step="0.01" value="0.25" />
            <span class="range-hint">lower = smoother &middot; 1 = off</span>
          </label>
        </div>
        <div id="metrics" class="metric-grid"></div>
        <div id="culturePanel" class="culture-panel hidden">
          <p class="section-title">Culture Axis <span class="info-dot" title="A rolling ~2s average of the 9 metrics, compared against the Morris and BaYe training clips. 1.0 = movement statistics match Morris, 0.0 = match BaYe, 0.5 = between the two.">?</span></p>
          <div class="culture-row">
            <span class="culture-end baye">BAYE</span>
            <div class="culture-track"><div id="cultureMarker" class="culture-marker"></div></div>
            <span class="culture-end morris">MORRIS</span>
          </div>
          <div id="cultureValue" class="culture-value">/field/morrisness –</div>
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
      energy: 'high = more motion',
      sync_velocity: '1 = both sides active',
      sync_correlation: 'high = left/right in phase',
      expansion: 'high = open posture',
      curvature: 'high = rounder paths',
      height: 'body centre level',
      sway: 'low = steadier balance',
      torque: 'high = forceful accents',
      jerk: 'low = smoother motion',
    };
    const metricDescriptions = {
      energy: 'Overall movement intensity, from how fast all limbs rotate. Big, fast motion pushes it up; standing still reads near 0.',
      sync_velocity: 'How evenly left and right move. 1 = both sides equally active, 0 = one side does all the work.',
      sync_correlation: 'Whether left and right move at the same time. +1 = together, 0 = independent, -1 = alternating. Normalize shows it as 0-1 with 0.5 neutral.',
      expansion: 'How much space the body occupies. High = open, spread-out postures; low = compact, closed shapes.',
      curvature: 'How curved the hand and foot paths are. High = round, circular movement; low = straight lines.',
      height: 'Body centre height relative to the hips. Drops when crouching or folding the torso.',
      sway: 'Horizontal drift of the body centre away from the feet. High = leaning or off balance.',
      torque: 'How hard the movement changes speed. High = sharp, forceful accents; low = even pacing.',
      jerk: 'How abrupt those speed changes are. Low = flowing; high = jagged, twitchy. Raw values get very large - use normalize for a readable range.',
    };
    const centeredMetrics = new Set(['sync_correlation']);
    const metricOverlayEl = document.getElementById('metricOverlay');

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

      const ovRow = document.createElement('div');
      ovRow.className = 'ov-row';
      ovRow.innerHTML = `<span class="ov-name">${name}</span><span class="ov-val" id="ov-${name}">0.00</span>`;
      metricOverlayEl.appendChild(ovRow);
    }

    async function applySettings(detectEnabled) {
      const form = document.getElementById('form');
      const data = new FormData(form);
      data.set('loop', 'true');
      data.set('detect_enabled', detectEnabled ? 'true' : 'false');
      data.set('osc_enabled', document.getElementById('oscEnabled').checked ? 'true' : 'false');
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
    const cameraButton = document.getElementById('cameraButton');

    let selectedVideoUrl = null;
    let isDetecting = false;
    let cameraOn = false;
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
      detectButton.classList.toggle('off', !enabled);
      detectButton.title = enabled ? 'Turn detection off' : 'Turn detection on';
    }

    function setCameraButton(on) {
      cameraOn = on;
      cameraButton.classList.toggle('off', !on);
      cameraButton.title = on ? 'Turn camera off' : 'Turn camera on';
      document.getElementById('emptyState').classList.toggle('hidden', on);
    }

    async function stopStream() {
      streamImage.removeAttribute('src');
      streamImage.classList.add('hidden');
      previewVideo.pause();
      previewVideo.classList.add('hidden');
      try { await fetch('/api/camera/release', { method: 'POST' }); } catch (e) {}
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
    const videoWrap = document.querySelector('.video-wrap');
    const fullscreenButton = document.getElementById('fullscreenButton');
    fullscreenButton.addEventListener('click', () => {
      const fsEl = document.fullscreenElement || document.webkitFullscreenElement;
      if (fsEl) {
        (document.exitFullscreen || document.webkitExitFullscreen).call(document);
      } else {
        (videoWrap.requestFullscreen || videoWrap.webkitRequestFullscreen).call(videoWrap);
      }
    });
    function syncFullscreenButton() {
      const active = !!(document.fullscreenElement || document.webkitFullscreenElement);
      fullscreenButton.classList.toggle('fs-active', active);
      fullscreenButton.title = active ? 'Exit fullscreen' : 'Fullscreen';
    }
    document.addEventListener('fullscreenchange', syncFullscreenButton);
    document.addEventListener('webkitfullscreenchange', syncFullscreenButton);

    const quitButton = document.getElementById('quitButton');
    const quitLabel = document.getElementById('quitLabel');
    let quitArmed = false;
    let quitTimer = null;
    quitButton.addEventListener('click', async () => {
      if (!quitArmed) {
        quitArmed = true;
        quitButton.classList.add('confirm');
        quitLabel.textContent = 'Confirm quit';
        quitTimer = window.setTimeout(() => {
          quitArmed = false;
          quitButton.classList.remove('confirm');
          quitLabel.textContent = 'Quit';
        }, 3000);
        return;
      }
      window.clearTimeout(quitTimer);
      quitArmed = false;
      quitLabel.textContent = 'Stopping...';
      try { await fetch('/api/shutdown', { method: 'POST' }); } catch (e) {}
      document.getElementById('stoppedOverlay').classList.remove('hidden');
    });
    document.getElementById('enterInputButton').addEventListener('click', async () => {
      setDetectButton(false);
      const payload = await applySettings(false);
      if (!payload || payload.status !== 'applied') return;
      document.getElementById('inputOverlay').classList.add('compact');
      setCameraButton(true);
      showPreview();
    });
    cameraButton.addEventListener('click', async () => {
      if (cameraOn) {
        setDetectButton(false);
        setCameraButton(false);
        await stopStream();
      } else {
        const payload = await applySettings(false);
        if (!payload || payload.status !== 'applied') return;
        document.getElementById('inputOverlay').classList.add('compact');
        setCameraButton(true);
        showPreview();
      }
    });
    detectButton.addEventListener('click', async () => {
      const nextState = !isDetecting;
      const payload = await applySettings(nextState);
      if (!payload || payload.status !== 'applied') return;
      setDetectButton(nextState);
      setCameraButton(true);
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
      document.getElementById('metaB').textContent =
        `fps: ${Number(processing.fps || 0).toFixed(1)} / pose: ${Number(processing.analysis_fps || 0).toFixed(1)}`;
      if (source.source === 'video') {
        document.getElementById('metaC').textContent = `time: ${Number(processing.elapsed_seconds || 0).toFixed(1)}s`;
        document.getElementById('metaD').textContent = `file: ${source.video_name || '-'} / loop ${source.loop ? 'on' : 'off'}`;
      } else {
        const cameraSelect = document.getElementById('camera');
        const cameraLabel = cameraSelect.options[cameraSelect.selectedIndex]?.textContent || source.camera_index || '-';
        document.getElementById('metaC').textContent = `camera: ${cameraLabel}`;
        document.getElementById('metaD').textContent =
          `pose ${Number(processing.pose_ms || 0).toFixed(0)}ms / jpeg ${Number(processing.encode_ms || 0).toFixed(0)}ms / osc: ${osc.enabled ? 'on' : 'off'} ${osc.host || '-'}:${osc.port || '-'}`;
      }
      updateAddresses(payload);

      for (const name of metricNames) {
        const value = Number(metrics[name] ?? 0);
        const valueEl = document.getElementById(`v-${name}`);
        valueEl.textContent = formatMetric(value);
        valueEl.title = Number.isFinite(value) ? value.toFixed(2) : String(value);
        updateMetricBar(name, value, osc.mode || 'raw');
        const ovEl = document.getElementById(`ov-${name}`);
        if (ovEl) ovEl.textContent = formatMetric(value);
      }

      const morrisness = processing.morrisness;
      if (morrisness !== null && morrisness !== undefined) {
        document.getElementById('culturePanel').classList.remove('hidden');
        const pct = Math.max(0, Math.min(100, Number(morrisness) * 100));
        document.getElementById('cultureMarker').style.left = `${pct.toFixed(1)}%`;
        document.getElementById('cultureValue').textContent =
          `/field/morrisness  ${Number(morrisness).toFixed(2)}`;
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
    parser.add_argument(
        "--pose-model",
        choices=["lite", "full", "heavy"],
        default=os.getenv("FIELD_POSE_MODEL", "lite"),
        help="MediaPipe pose model. lite is fastest; full/heavy are more accurate but slower.",
    )
    args = parser.parse_args()

    global pose_model_path
    pose_model_path = REPO_ROOT / f"pose_landmarker_{args.pose_model}.task"

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
    print(f"Pose model: {args.pose_model} ({pose_model_path})")
    uvicorn.run(app, host=args.web_host, port=args.web_port)


if __name__ == "__main__":
    main()
