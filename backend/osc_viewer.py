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
from calibration import (
    CalibrationCollector, RANGE_METRICS, load_profile, save_profile,
    load_presets, save_presets,
)
from pose_engine import PoseEngine, VALID_BACKENDS


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
    alpha=float(os.getenv("FIELD_OSC_ALPHA", "1.0")),
    namespace=os.getenv("FIELD_OSC_NAMESPACE", "/field"),
)

# Calibration ("sound-check") -> fixed-range normalize.
CALIB_PROFILE_PATH = REPO_ROOT / "calibration_profile.json"
calibration = CalibrationCollector()
calibrating = False
_calib_profile = load_profile(CALIB_PROFILE_PATH)
if _calib_profile:
    osc_sender.set_metric_ranges(_calib_profile)
    print(f"Loaded calibration profile: {sorted(_calib_profile)}")

# Named calibration presets (multiple saved calibrations, e.g. per dancer/venue).
PRESETS_PATH = REPO_ROOT / "calibration_presets.json"
calib_presets = load_presets(PRESETS_PATH)
active_preset = None
# Optional: offline culture centroids enable the /field/morrisness output.
culture_score = CultureScore.try_load(BASE_DIR / "culture_map.json")

PERFORMANCE_PRESETS = {
    "fast": {"width": 854, "height": 480, "target_fps": 24.0, "analysis_fps": 12.0, "jpeg_quality": 60},
    "balanced": {"width": 1280, "height": 720, "target_fps": 24.0, "analysis_fps": 12.0, "jpeg_quality": 65},
    "quality": {"width": 1920, "height": 1080, "target_fps": 20.0, "analysis_fps": 10.0, "jpeg_quality": 72},
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
    "pose_backend": os.getenv("FIELD_POSE_BACKEND", "mediapipe"),
    "smooth_enabled": True,
    "smooth_min_cutoff": 1.5,
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
    """Return the singleton PoseEngine, (re)building it when the selected
    backend changes. Backend swaps happen here (on stream (re)start) rather
    than inside /api/apply, so the previous stream loop has already exited and
    there is no mid-frame race. Smoothing config is cheap and applied live."""
    global pose_engine
    desired = source_state.get("pose_backend", "mediapipe")
    if desired not in VALID_BACKENDS:
        desired = "mediapipe"
    smooth_enabled = bool(source_state.get("smooth_enabled", True))
    min_cutoff = float(source_state.get("smooth_min_cutoff", 1.5))

    if pose_engine is None:
        pose_engine = PoseEngine(model_path=str(pose_model_path), backend=desired,
                                 smoothing_enabled=smooth_enabled,
                                 smooth_min_cutoff=min_cutoff)
    elif pose_engine.backend_name != desired:
        old = pose_engine
        pose_engine = PoseEngine(model_path=str(pose_model_path), backend=desired,
                                 smoothing_enabled=smooth_enabled,
                                 smooth_min_cutoff=min_cutoff)
        try:
            old.close()
        except Exception:
            pass
    else:
        pose_engine.configure_smoothing(enabled=smooth_enabled, min_cutoff=min_cutoff)
    # Reflect the backend that actually loaded (it may have fallen back, e.g. to
    # MediaPipe on a machine without CUDA), so we don't retry-build every frame.
    source_state["pose_backend"] = pose_engine.backend_name
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
    if calibrating:
        calibration.add(metrics)
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


@app.get("/charts")
async def charts():
    return HTMLResponse(CHARTS_HTML)


@app.get("/api/metrics")
async def api_metrics():
    """Lightweight metrics snapshot for the live charts page."""
    return {
        "raw": processing_state.get("latest_raw_metrics", {}),
        "smoothed": processing_state.get("latest_metrics", {}),
        "running": bool(processing_state.get("running", False)),
        "t": processing_state.get("latest_timestamp_ms"),
    }


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
    osc_alpha: float = Form(1.0),
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


@app.post("/api/smoothing")
async def apply_smoothing(
    smooth_enabled: bool = Form(True),
    smooth_min_cutoff: float = Form(1.5),
):
    """Live-tune the One-Euro joint smoothing WITHOUT restarting the stream.
    Unlike /api/apply this does NOT bump session_id or reset state, so dragging
    the slider does not kill the video feed."""
    source_state["smooth_enabled"] = bool(smooth_enabled)
    source_state["smooth_min_cutoff"] = float(smooth_min_cutoff)
    if pose_engine is not None:
        pose_engine.configure_smoothing(enabled=source_state["smooth_enabled"],
                                        min_cutoff=source_state["smooth_min_cutoff"])
    return {"status": "applied"}


@app.post("/api/metric_smoothing")
async def apply_metric_smoothing(metric: str = Form(...), alpha: float = Form(...)):
    """Live per-metric OSC output smoothing (no stream restart). alpha in (0,1],
    1 = off; falls back to the global smoothing for metrics left untouched."""
    try:
        osc_sender.set_metric_alpha(metric, alpha)
    except ValueError:
        return {"status": "error", "error": "alpha must be > 0 and <= 1"}
    return {"status": "applied", "metric": metric, "alpha": float(alpha)}


@app.post("/api/calibrate/start")
async def calibrate_start():
    """Begin a calibration ('sound-check') recording. Requires detection running
    so metrics flow into the collector."""
    global calibrating
    calibration.reset()
    calibrating = True
    return {"status": "calibrating"}


@app.post("/api/calibrate/stop")
async def calibrate_stop():
    """Finish calibration: derive per-metric ranges (percentiles), install them,
    save the profile, and switch OSC to 'fixed' mode."""
    global calibrating
    calibrating = False
    ranges = calibration.ranges()
    if ranges:
        osc_sender.set_metric_ranges(ranges)
        try:
            save_profile(CALIB_PROFILE_PATH, osc_sender.metric_ranges)
        except Exception as exc:
            print(f"calibration profile save failed: {exc}")
        osc_sender.configure(mode="fixed")
    return {
        "status": "applied",
        "mode": osc_sender.mode,
        "ranges": {k: list(v) for k, v in osc_sender.metric_ranges.items()},
        "counts": {m: calibration.count(m) for m in RANGE_METRICS},
    }


@app.get("/api/calibrate/presets")
async def calibrate_presets():
    return {"presets": sorted(calib_presets.keys()), "active": active_preset}


@app.post("/api/calibrate/save_preset")
async def calibrate_save_preset(name: str = Form(...)):
    global active_preset
    name = name.strip()
    if not name:
        return {"status": "error", "error": "name required"}
    if not osc_sender.metric_ranges:
        return {"status": "error", "error": "calibrate first (no ranges yet)"}
    calib_presets[name] = dict(osc_sender.metric_ranges)
    try:
        save_presets(PRESETS_PATH, calib_presets)
    except Exception as exc:
        print(f"preset save failed: {exc}")
    active_preset = name
    return {"status": "saved", "name": name, "presets": sorted(calib_presets.keys())}


@app.post("/api/calibrate/load_preset")
async def calibrate_load_preset(name: str = Form(...)):
    global active_preset
    rng = calib_presets.get(name)
    if not rng:
        return {"status": "error", "error": "unknown preset"}
    osc_sender.clear_metric_ranges()
    osc_sender.set_metric_ranges(rng)
    osc_sender.configure(mode="fixed")
    active_preset = name
    return {"status": "applied", "name": name, "mode": osc_sender.mode,
            "ranges": {k: list(v) for k, v in osc_sender.metric_ranges.items()}}


@app.post("/api/calibrate/delete_preset")
async def calibrate_delete_preset(name: str = Form(...)):
    global active_preset
    calib_presets.pop(name, None)
    if active_preset == name:
        active_preset = None
    try:
        save_presets(PRESETS_PATH, calib_presets)
    except Exception as exc:
        print(f"preset delete-save failed: {exc}")
    return {"status": "deleted", "presets": sorted(calib_presets.keys())}


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
    performance: str = Form(DEFAULT_PERFORMANCE),
    pose_backend: str = Form("mediapipe"),
    smooth_enabled: bool = Form(True),
    smooth_min_cutoff: float = Form(1.5),
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

    source_state["pose_backend"] = pose_backend if pose_backend in VALID_BACKENDS else "mediapipe"
    source_state["smooth_enabled"] = bool(smooth_enabled)
    source_state["smooth_min_cutoff"] = float(smooth_min_cutoff)
    # Live-apply smoothing to the running engine (cheap). A backend SWAP is
    # deferred to get_pose_engine() on the next stream start, so it never races
    # the still-running stream loop.
    if pose_engine is not None and pose_engine.backend_name == source_state["pose_backend"]:
        pose_engine.configure_smoothing(enabled=source_state["smooth_enabled"],
                                        min_cutoff=source_state["smooth_min_cutoff"])

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
    .metric-smooth-row { grid-column: 1 / -1; display: flex; align-items: center; gap: 8px; margin-top: 4px; }
    .metric-smooth-row .ms-label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }
    .metric-smooth-row input[type=range] { flex: 1; min-width: 0; }
    .metric-smooth-row .ms-val { color: var(--teal); font: 12px ui-monospace, monospace; min-width: 32px; text-align: right; }
    .calib-btn { width: 100%; padding: 9px; border-radius: 8px; border: 1px solid var(--line); background: var(--surface-soft); color: inherit; cursor: pointer; font-size: 13px; }
    .calib-btn.recording { border-color: #c0392b; color: #f1948a; }
    .calib-info { margin-top: 6px; color: var(--muted); font-size: 12px; line-height: 1.4; }
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
              <label class="model-row">Detection model
                <select id="poseBackend" name="pose_backend">
                  <option value="mediapipe">MediaPipe (default &middot; any GPU)</option>
                  <option value="rtmpose3d">RTMPose3D (NVIDIA GPU)</option>
                </select>
              </label>
              <label class="model-row">Quality
                <select id="quality" name="performance">
                  <option value="fast">Fast (854x480)</option>
                  <option value="balanced" selected>Balanced (1280x720)</option>
                  <option value="quality">Quality (1920x1080)</option>
                </select>
              </label>
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
              <option value="normalize">normalize (adaptive)</option>
              <option value="fixed">fixed (calibrated)</option>
            </select>
          </label>
          <label><span class="range-label-row"><span class="label-row">Smoothing <span class="info-dot" title="Global EMA on the OSC output (fallback for metrics without their own slider). 1 = off; lower = steadier but slower. Default off -- One-Euro on the joints is the primary smoother.">?</span></span><span id="oscAlphaValue" class="range-value">1.00</span></span>
            <input id="oscAlpha" name="osc_alpha" type="range" min="0.01" max="1" step="0.01" value="1" />
            <span class="range-hint">lower = smoother &middot; 1 = off</span>
          </label>
          <label class="osc-toggle-row"><input id="smoothEnabled" name="smooth_enabled" type="checkbox" checked /> Smooth joints (One-Euro)</label>
          <label><span class="range-label-row"><span class="label-row">Joint smoothing <span class="info-dot" title="One-Euro filter on the skeleton before metrics. Lower = smoother (more lag); higher = more responsive. Cuts jitter in torque/jerk at the source.">?</span></span><span id="smoothCutoffValue" class="range-value">1.5</span></span>
            <input id="smoothCutoff" name="smooth_min_cutoff" type="range" min="0.3" max="6" step="0.1" value="1.5" />
            <span class="range-hint">lower = smoother</span>
          </label>
        </div>
        <div class="calib-panel" style="padding:6px 14px;">
          <button id="calibBtn" type="button" class="calib-btn">Calibrate (&#35430;&#38899;)</button>
          <div id="calibInfo" class="calib-info hidden">Recording &mdash; hold each ~4s while detecting: <b>1) small &amp; still</b> (curl up, crouch low, freeze) &middot; <b>2) big &amp; round</b> (reach/hop up tall, spread wide, big circles with hands &amp; feet, lean far each way) &middot; <b>3) fast &amp; sharp</b> (explosive bursts, sudden stops). Then press stop.</div>
          <div class="preset-row" style="display:flex; gap:6px; margin-top:6px; align-items:center;">
            <select id="presetSelect" style="flex:1; min-width:0;"><option value="">&mdash; preset &mdash;</option></select>
            <button id="presetSaveBtn" type="button" class="calib-btn" style="width:auto; padding:6px 10px;">Save as&hellip;</button>
            <button id="presetDelBtn" type="button" class="calib-btn" style="width:auto; padding:6px 10px;">Del</button>
          </div>
        </div>
        <div class="charts-link" style="padding:2px 14px 8px;"><a href="/charts" target="_blank" style="color:var(--teal);text-decoration:none;font-size:13px;">&#128200; Open live charts &#8599;</a></div>
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
        <div class="metric-smooth-row">
          <span class="ms-label">smooth</span>
          <input class="metric-smooth" type="range" min="0.02" max="1" step="0.01" value="1" title="Per-metric output smoothing (lower = smoother, 1 = off). On top of One-Euro." />
          <span class="ms-val">1.00</span>
        </div>
      `;
      metricsEl.appendChild(row);
      const msInput = row.querySelector('.metric-smooth');
      const msVal = row.querySelector('.ms-val');
      msInput.addEventListener('input', () => { msVal.textContent = msInput.value; });
      msInput.addEventListener('change', async () => {
        const data = new FormData();
        data.set('metric', name);
        data.set('alpha', msInput.value);
        try { await fetch('/api/metric_smoothing', { method: 'POST', body: data }); } catch (e) {}
      });

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
      data.set('smooth_enabled', document.getElementById('smoothEnabled').checked ? 'true' : 'false');
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

    document.getElementById('poseBackend').addEventListener('change', async () => {
      inputDirty = true;
      // Backend swap rebuilds the engine on stream restart; re-apply + restart.
      if (isDetecting) {
        const payload = await applySettings(true);
        if (payload && payload.status === 'applied') showDetectionStream();
      }
    });
    document.getElementById('quality').addEventListener('change', async () => {
      inputDirty = true;
      // Resolution change reopens the camera (apply releases it), so re-apply.
      if (!cameraOn && !isDetecting) return;
      const payload = await applySettings(isDetecting);
      if (!payload || payload.status !== 'applied') return;
      if (isDetecting) showDetectionStream(); else showPreview();
    });
    const smoothCutoffEl = document.getElementById('smoothCutoff');
    const smoothCutoffValueEl = document.getElementById('smoothCutoffValue');
    async function applySmoothing() {
      // Lightweight live-tune: does NOT restart the stream (no session bump).
      const data = new FormData();
      data.set('smooth_enabled', document.getElementById('smoothEnabled').checked ? 'true' : 'false');
      data.set('smooth_min_cutoff', smoothCutoffEl.value);
      try { await fetch('/api/smoothing', { method: 'POST', body: data }); } catch (e) {}
    }
    smoothCutoffEl.addEventListener('input', () => {
      smoothCutoffValueEl.textContent = smoothCutoffEl.value;
    });
    smoothCutoffEl.addEventListener('change', applySmoothing);
    document.getElementById('smoothEnabled').addEventListener('change', applySmoothing);

    let calibrating = false;
    const calibBtn = document.getElementById('calibBtn');
    const calibInfo = document.getElementById('calibInfo');
    calibBtn.addEventListener('click', async () => {
      if (!calibrating) {
        if (!isDetecting) { alert('Start detection first, then Calibrate.'); return; }
        try { await fetch('/api/calibrate/start', { method: 'POST' }); } catch (e) { return; }
        calibrating = true;
        calibBtn.textContent = 'Stop & save (recording...)';
        calibBtn.classList.add('recording');
        calibInfo.classList.remove('hidden');
      } else {
        let d = {};
        try { const r = await fetch('/api/calibrate/stop', { method: 'POST' }); d = await r.json(); } catch (e) {}
        calibrating = false;
        calibBtn.textContent = 'Calibrate (試音)';
        calibBtn.classList.remove('recording');
        calibInfo.classList.add('hidden');
        const n = Object.keys(d.ranges || {}).length;
        if (n > 0) {
          document.getElementById('oscMode').value = 'fixed';
          alert('Calibrated ' + n + ' metrics. OSC mode set to fixed (calibrated ranges).');
        } else {
          alert('Not enough data - keep detection running and keep moving for a few seconds, then Calibrate again.');
        }
      }
    });

    const presetSelect = document.getElementById('presetSelect');
    async function refreshPresets(active) {
      try {
        const r = await fetch('/api/calibrate/presets');
        const d = await r.json();
        const sel = (active !== undefined) ? active : d.active;
        presetSelect.innerHTML = '<option value="">— preset —</option>';
        for (const n of (d.presets || [])) {
          const o = document.createElement('option');
          o.value = n; o.textContent = n;
          if (n === sel) o.selected = true;
          presetSelect.appendChild(o);
        }
      } catch (e) {}
    }
    presetSelect.addEventListener('change', async () => {
      const name = presetSelect.value;
      if (!name) return;
      const data = new FormData(); data.set('name', name);
      const r = await fetch('/api/calibrate/load_preset', { method: 'POST', body: data });
      const d = await r.json();
      if (d.status === 'applied') document.getElementById('oscMode').value = 'fixed';
      else alert(d.error || 'load failed');
    });
    document.getElementById('presetSaveBtn').addEventListener('click', async () => {
      const name = prompt('Save current calibration as preset:');
      if (!name) return;
      const data = new FormData(); data.set('name', name);
      const r = await fetch('/api/calibrate/save_preset', { method: 'POST', body: data });
      const d = await r.json();
      if (d.status === 'saved') refreshPresets(name);
      else alert(d.error || 'save failed - calibrate first');
    });
    document.getElementById('presetDelBtn').addEventListener('click', async () => {
      const name = presetSelect.value;
      if (!name) { alert('Pick a preset to delete first.'); return; }
      if (!confirm('Delete preset "' + name + '"?')) return;
      const data = new FormData(); data.set('name', name);
      await fetch('/api/calibrate/delete_preset', { method: 'POST', body: data });
      refreshPresets('');
    });
    refreshPresets();

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


CHARTS_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>FIELD - Live Charts</title>
<style>
  :root { --bg:#0c0b09; --panel:#141210; --line:#2a251f; --muted:#8a7f72; --teal:#34d3c0; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:#e8e1d6; font:14px ui-monospace, SFMono-Regular, Menlo, monospace; }
  header { display:flex; align-items:center; gap:16px; padding:12px 18px; border-bottom:1px solid var(--line); }
  header h1 { font-size:15px; margin:0; letter-spacing:.04em; }
  header .status { color:var(--muted); }
  header .dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:#a33; margin-right:6px; }
  header .dot.on { background:#3a3; }
  header a { color:var(--teal); text-decoration:none; margin-left:auto; }
  .grid { display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; padding:16px; }
  @media (max-width:900px){ .grid{ grid-template-columns:repeat(2,1fr);} }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:9px; padding:10px 12px; }
  .panel .head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:6px; }
  .panel .name { color:var(--muted); text-transform:uppercase; font-size:12px; letter-spacing:.06em; }
  .panel .val { color:var(--teal); font-variant-numeric:tabular-nums; }
  canvas { width:100%; height:120px; display:block; }
  .legend { color:var(--muted); font-size:11px; padding:0 18px 18px; }
  .legend b { color:var(--teal); } .legend i { color:#8c8070; font-style:normal; }
</style>
</head>
<body>
<header>
  <h1>FIELD - live metric charts</h1>
  <span class="status"><span id="dot" class="dot"></span><span id="statusText">waiting...</span></span>
  <a href="/">back to viewer</a>
</header>
<div class="grid" id="grid"></div>
<div class="legend">Per chart: <b>bold = value sent to OSC (after per-metric output smoothing)</b>, <i>faint = pre-output value (already One-Euro smoothed at the joints)</i>. Window approx last 30s. Start detection in the viewer, then watch here. Toggle/drag the smoothing sliders in the viewer and watch the curves change.</div>
<script>
const METRICS = [
  {key:'energy', name:'Energy'},
  {key:'sync_velocity', name:'Sync velocity'},
  {key:'sync_correlation', name:'Sync correlation'},
  {key:'expansion', name:'Expansion'},
  {key:'curvature', name:'Curvature'},
  {key:'height', name:'CoM height'},
  {key:'sway', name:'Sway'},
  {key:'torque', name:'Torque'},
  {key:'jerk', name:'Jerk'}
];
const WINDOW = 300, POLL_MS = 100;
const buffers = {}, canvases = {}, valEls = {};
const grid = document.getElementById('grid');
for (const m of METRICS) {
  buffers[m.key] = { raw: [], smooth: [] };
  const panel = document.createElement('div');
  panel.className = 'panel';
  panel.innerHTML = '<div class="head"><span class="name">' + m.name + '</span><span class="val" id="val-' + m.key + '">-</span></div><canvas></canvas>';
  grid.appendChild(panel);
  canvases[m.key] = panel.querySelector('canvas');
  valEls[m.key] = panel.querySelector('#val-' + m.key);
}
function pushVal(arr, v) {
  arr.push((typeof v === 'number' && isFinite(v)) ? v : null);
  if (arr.length > WINDOW) arr.shift();
}
function fit(c) {
  const r = c.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(r.width * dpr), h = Math.round(r.height * dpr);
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  return dpr;
}
function series(g, w, h, arr, lo, hi, color, lw) {
  const span = (hi - lo) || 1;
  g.strokeStyle = color; g.lineWidth = lw; g.beginPath();
  let started = false;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (v == null) { started = false; continue; }
    const x = (i / (WINDOW - 1)) * w;
    const y = h - (((v - lo) / span) * h * 0.9) - h * 0.05;
    if (!started) { g.moveTo(x, y); started = true; } else g.lineTo(x, y);
  }
  g.stroke();
}
function draw() {
  for (const m of METRICS) {
    const c = canvases[m.key], dpr = fit(c), g = c.getContext('2d');
    const w = c.width, h = c.height;
    g.clearRect(0, 0, w, h);
    const b = buffers[m.key];
    const all = b.raw.concat(b.smooth).filter(v => v != null);
    g.strokeStyle = 'rgba(255,255,255,0.05)'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(0, h / 2); g.lineTo(w, h / 2); g.stroke();
    if (all.length === 0) continue;
    let lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
    if (lo === hi) { lo -= 1; hi += 1; }
    series(g, w, h, b.raw, lo, hi, 'rgba(140,128,112,0.5)', 1 * dpr);
    series(g, w, h, b.smooth, lo, hi, '#34d3c0', 1.8 * dpr);
    const last = b.smooth.filter(v => v != null).slice(-1)[0];
    if (last != null) valEls[m.key].textContent = last.toFixed(3);
  }
}
let running = false;
async function poll() {
  try {
    const res = await fetch('/api/metrics', { cache: 'no-store' });
    const d = await res.json();
    running = !!d.running;
    const raw = d.raw || {}, sm = d.smoothed || {};
    for (const m of METRICS) { pushVal(buffers[m.key].raw, raw[m.key]); pushVal(buffers[m.key].smooth, sm[m.key]); }
  } catch (e) { running = false; }
  document.getElementById('dot').className = 'dot' + (running ? ' on' : '');
  document.getElementById('statusText').textContent = running ? 'streaming' : 'no stream - start detection in the viewer';
  draw();
}
setInterval(poll, POLL_MS);
poll();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Local FIELD input viewer with pose overlay, metrics, and OSC output.")
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=9100)
    parser.add_argument("--osc-host", default=os.getenv("FIELD_OSC_HOST", "127.0.0.1"))
    parser.add_argument("--osc-port", type=int, default=int(os.getenv("FIELD_OSC_PORT", "9000")))
    parser.add_argument("--osc-mode", choices=["raw", "normalize"], default=os.getenv("FIELD_OSC_MODE", "raw"))
    parser.add_argument("--osc-alpha", type=float, default=float(os.getenv("FIELD_OSC_ALPHA", "1.0")))
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
