import argparse
import asyncio
import json
import os
import re
import signal
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

from calibration import CalibrationCollector, load_presets, normalize_presets, save_presets
from culture_score import CultureScore
from osc_sender import METRIC_NAMES, OSC_ADDRESS_NAMES, OSCSender
from pose_engine import PoseEngine, VALID_BACKENDS


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
UPLOAD_DIR = BASE_DIR / "viewer_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
CALIBRATION_PRESETS_PATH = BASE_DIR / "calibration_presets.json"

pose_engine: Optional[PoseEngine] = None
pose_model_path = REPO_ROOT / "pose_landmarker_lite.task"
camera = None
camera_owner: Optional[int] = None
camera_index_opened: Optional[int] = None
camera_lock = threading.Lock()
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

# Optional: offline culture centroids enable the /field/morrisness output.
culture_score = CultureScore.try_load(BASE_DIR / "culture_map.json")
calibration_collector = CalibrationCollector()
calibration_presets = load_presets(CALIBRATION_PRESETS_PATH)
calibration_state = {
    "active": False,
    "started_at": None,
    "countdown_until": None,
    "sample_count": 0,
    "skipped_frames": 0,
    "applied_preset": None,
}

PERFORMANCE_PRESETS = {
    "fast": {"width": 854, "height": 480, "target_fps": 24.0, "analysis_fps": 12.0, "jpeg_quality": 60},
    "balanced": {"width": 1280, "height": 720, "target_fps": 24.0, "analysis_fps": 12.0, "jpeg_quality": 65},
    "quality": {"width": 1920, "height": 1080, "target_fps": 20.0, "analysis_fps": 10.0, "jpeg_quality": 72},
}
DEFAULT_PERFORMANCE = "quality"
DEFAULT_METRIC_EMA_FRAMES = 3.0
DEFAULT_METRIC_ALPHA = 2.0 / (DEFAULT_METRIC_EMA_FRAMES + 1.0)
NO_SMOOTHING_ALPHA = 1.0
CALIBRATION_COUNTDOWN_SECONDS = 3.0
CAMERA_SCAN_MAX_INDEX = int(os.getenv("FIELD_CAMERA_SCAN_MAX_INDEX", "8"))

for metric_name in METRIC_NAMES:
    alpha = NO_SMOOTHING_ALPHA if metric_name == "sync_correlation" else DEFAULT_METRIC_ALPHA
    osc_sender.set_metric_alpha(metric_name, alpha)

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
    "smooth_enabled": False,
    "smooth_min_cutoff": 0.3,
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
    "pose_valid": False,
    "pose_quality": 0.0,
    "last_frame_at": None,
    "signal_mean": None,
    "morrisness": None,
    "error": None,
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    release_camera(force=True)
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
    smooth_enabled = bool(source_state.get("smooth_enabled", False))
    min_cutoff = float(source_state.get("smooth_min_cutoff", 0.3))

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


def release_camera(owner: Optional[int] = None, force: bool = False) -> None:
    """Release the active camera.

    Stream cleanup passes its session id as owner, so an old stream cannot
    release a newer stream's camera after /api/apply has already restarted it.
    Explicit server shutdown/restart paths use force=True; regular UI camera-off
    invalidates the stream first, then releases only that old session owner.
    """
    global camera, camera_owner, camera_index_opened
    with camera_lock:
        if camera is None:
            return
        if not force and owner is not None and camera_owner != owner:
            return
        cap = camera
        camera = None
        camera_owner = None
        camera_index_opened = None
        cap.release()


def read_camera_frame(cap, owner: Optional[int]):
    """Read a frame while holding the same lock used for release/open.

    OpenCV's AVFoundation backend can segfault if VideoCapture.release() races
    with VideoCapture.read() on another thread.
    """
    with camera_lock:
        if camera is not cap or camera_owner != owner:
            return False, None
        return cap.read()


def camera_backend():
    if os.name == "nt":
        return cv2.CAP_DSHOW
    if sys.platform == "darwin":
        return getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    return cv2.CAP_ANY


def open_camera(index: int, owner: Optional[int] = None):
    global camera, camera_owner, camera_index_opened
    with camera_lock:
        if camera is not None and camera_owner == owner and camera_index_opened == index:
            return camera
        if camera is not None:
            camera.release()
            camera = None
            camera_owner = None
            camera_index_opened = None

        cap = cv2.VideoCapture(index, camera_backend())
        if os.name == "nt":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(source_state["width"]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(source_state["height"]))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Camera {index} could not be opened")
        camera = cap
        camera_owner = owner
        camera_index_opened = index
        return camera


async def reopen_live_camera(session_id: int, reason: str):
    processing_state["error"] = reason
    await asyncio.to_thread(release_camera, session_id)
    await asyncio.sleep(0.35)
    return await asyncio.to_thread(open_camera, int(source_state["camera_index"]), session_id)


def state_payload() -> dict:
    active_metrics = set(active_metric_names())
    backends = available_pose_backends()
    backend_ids = {item["id"] for item in backends}
    if source_state.get("pose_backend") not in backend_ids:
        source_state["pose_backend"] = "mediapipe"
    payload = {
        "source": dict(source_state),
        "processing": dict(processing_state),
        "osc": osc_sender.get_status(),
        "calibration": calibration_payload(),
        "addresses": [osc_sender.metric_address(name) for name in METRIC_NAMES if name in active_metrics],
        "pose_backends": backends,
    }
    payload["source"]["video_path"] = None
    return payload


def calibration_payload() -> dict:
    remaining = calibration_countdown_remaining()
    return {
        **calibration_state,
        "countdown_remaining": remaining,
        "collecting": bool(calibration_state.get("active")) and remaining <= 0.0,
        "presets": sorted(calibration_presets.keys()),
        "ranges": serialize_calibration_presets(),
    }


def calibration_countdown_remaining(now: Optional[float] = None) -> float:
    countdown_until = calibration_state.get("countdown_until")
    if not calibration_state.get("active") or countdown_until is None:
        return 0.0
    current = time.time() if now is None else float(now)
    return max(0.0, float(countdown_until) - current)


def has_usable_calibration_metrics(metrics: dict) -> bool:
    return bool(metrics) and (
        float(metrics.get("energy", 0.0) or 0.0) != 0.0
        or float(metrics.get("expansion", 0.0) or 0.0) != 0.0
    )


def serialize_calibration_presets() -> dict:
    return {
        name: {metric: [float(lo), float(hi)] for metric, (lo, hi) in ranges.items()}
        for name, ranges in calibration_presets.items()
    }


def safe_download_stem(name: str, fallback: str = "profile") -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name or "").strip()).strip(".-")
    return (stem or fallback)[:120]


def refresh_prepared_metrics() -> None:
    raw_metrics = processing_state.get("latest_raw_metrics") or {}
    if raw_metrics:
        osc_sender.send_metrics(raw_metrics, send_keys=set())
        processing_state["latest_metrics"] = filter_active_metrics(osc_sender.last_prepared_metrics)


def reset_pose_metric_history() -> None:
    if pose_engine is not None:
        pose_engine.reset_metrics()


def active_metric_names() -> list[str]:
    selected = source_state.get("osc_metrics") or []
    return [name for name in METRIC_NAMES if name in selected]


def active_metric_set() -> set[str]:
    return set(active_metric_names())


def filter_active_metrics(metrics: dict) -> dict:
    active = active_metric_set()
    return {k: v for k, v in (metrics or {}).items() if k in active}


def is_broadcast_host(host: str) -> bool:
    value = str(host or "").strip()
    return value == "255.255.255.255" or value.endswith(".255")


def parse_osc_targets(raw: str, fallback_host: str, fallback_port: int) -> list[dict]:
    if not raw:
        return [{
            "id": "default",
            "name": "Output 1",
            "host": fallback_host,
            "port": int(fallback_port),
            "enabled": True,
            "broadcast": is_broadcast_host(fallback_host),
        }]
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid osc_targets JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("osc_targets must be a list")

    targets = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError("each OSC target must be an object")
        host = str(item.get("host") or "").strip()
        if not host:
            raise ValueError("OSC target host is required")
        try:
            port = int(item.get("port"))
        except (TypeError, ValueError) as exc:
            raise ValueError("OSC target port must be a number") from exc
        if port < 1 or port > 65535:
            raise ValueError("OSC target port must be 1-65535")
        targets.append({
            "id": str(item.get("id") or f"output-{index + 1}").strip(),
            "name": str(item.get("name") or f"Output {index + 1}").strip(),
            "host": host,
            "port": port,
            "enabled": bool(item.get("enabled", True)),
            "broadcast": bool(item.get("broadcast", False)) or is_broadcast_host(host),
        })
    return targets


def rtmpose3d_selectable() -> bool:
    """RTMPose3D remains available in code, but is hidden from the rehearsal UI."""
    return False


def available_pose_backends() -> list[dict]:
    backends = [{
        "id": "mediapipe",
        "label": "MediaPipe",
        "description": "default, cross-platform",
    }]
    if rtmpose3d_selectable():
        backends.append({
            "id": "rtmpose3d",
            "label": "RTMPose3D",
            "description": "NVIDIA GPU",
        })
    return backends


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
            timeout=10,
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


def list_cameras(max_index: int = CAMERA_SCAN_MAX_INDEX) -> list[dict]:
    if time.time() - camera_cache["updated_at"] < 10 and camera_cache["cameras"]:
        return camera_cache["cameras"]

    # OpenCV camera indices are the only reliable values the stream can open.
    # OS/DirectShow/macOS device-name order is not guaranteed to match OpenCV
    # indices, especially with multiple USB or virtual cameras, so the UI uses
    # stable index labels instead of potentially wrong device names.
    with camera_lock:
        active_index = camera_index_opened if camera is not None else None
    selected_index = int(source_state.get("camera_index", 0))
    mac_names = get_macos_camera_names() if sys.platform == "darwin" else []
    if sys.platform == "darwin":
        scan_limit = max(1, len(mac_names), selected_index + 1, (active_index + 1) if active_index is not None else 0)
    else:
        scan_limit = max(max_index, selected_index + 1, (active_index + 1) if active_index is not None else 0)

    cameras = []
    for index in range(scan_limit):
        if active_index == index:
            available = True
        else:
            cap = cv2.VideoCapture(index, camera_backend())
            available = cap.isOpened()
            cap.release()
        if not available:
            continue
        display_name = mac_names[index] if index < len(mac_names) else f"Camera {index}"
        cameras.append({
            "index": index,
            "name": display_name,
            "label": f"{index} - {display_name}",
            "source": "OpenCV",
        })
    if not cameras:
        cameras.append({"index": 0, "name": "Camera 0", "label": "0 - Camera 0", "source": "Fallback"})
    camera_cache["updated_at"] = time.time()
    camera_cache["cameras"] = cameras
    return cameras


def test_camera_signal(index: int) -> dict:
    cap = cv2.VideoCapture(index, camera_backend())
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


def set_analysis_result(
    metrics: dict,
    timestamp_ms: int,
    pose_ms: float = 0.0,
    pose_valid: bool = True,
    pose_quality: Optional[float] = None,
) -> None:
    if processing_state["started_at"] is None:
        processing_state["started_at"] = time.time()
    processing_state["latest_raw_metrics"] = metrics
    processing_state["latest_timestamp_ms"] = timestamp_ms
    processing_state["pose_valid"] = bool(pose_valid)
    processing_state["pose_quality"] = float(pose_quality) if pose_quality is not None else (1.0 if pose_valid else 0.0)
    processing_state["analysis_count"] += 1
    processing_state["pose_ms"] = float(pose_ms)
    elapsed = time.time() - processing_state["started_at"]
    if elapsed > 0.25:
        processing_state["analysis_fps"] = processing_state["analysis_count"] / elapsed
    active = active_metric_set()
    osc_sender.send_metrics(metrics, send_keys=active)
    prepared_metrics = osc_sender.last_prepared_metrics
    if calibration_state["active"] and calibration_countdown_remaining() <= 0.0:
        if pose_valid and has_usable_calibration_metrics(prepared_metrics):
            calibration_collector.add(prepared_metrics)
            calibration_state["sample_count"] = calibration_collector.count()
        elif not pose_valid:
            calibration_state["skipped_frames"] = int(calibration_state.get("skipped_frames") or 0) + 1
    processing_state["latest_metrics"] = filter_active_metrics(osc_sender.last_prepared_metrics)
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
    try:
        cap = await asyncio.to_thread(open_camera, int(source_state["camera_index"]), session_id)
    except Exception as exc:
        processing_state["error"] = f"Camera unavailable: {exc}"
        processing_state["running"] = False
        return
    engine_task = asyncio.create_task(asyncio.to_thread(get_pose_engine))
    engine = None
    next_analysis_at = 0.0
    last_analysis_at = None
    measured_fps = max(float(source_state["analysis_fps"]), 1.0)
    missed_frames = 0

    try:
        while (
            source_state["source"] == "live"
            and source_state["session_id"] == session_id
            and source_state["detect_enabled"]
        ):
            started = time.time()
            ok, frame = await asyncio.to_thread(read_camera_frame, cap, session_id)
            if not ok:
                missed_frames += 1
                if missed_frames >= 5:
                    try:
                        cap = await reopen_live_camera(session_id, "Camera frame dropped; reconnecting")
                        missed_frames = 0
                    except Exception as exc:
                        processing_state["error"] = f"Camera reconnect failed: {exc}"
                        await asyncio.sleep(0.75)
                else:
                    processing_state["error"] = "Camera frame not available"
                    await asyncio.sleep(0.08)
                continue
            missed_frames = 0
            frame = apply_live_mirror(frame)
            frame = resize_frame(frame)

            timestamp_ms = int(time.time() * 1000)
            processed = frame
            now = time.time()

            if engine is None and engine_task.done():
                try:
                    engine = engine_task.result()
                except Exception as exc:
                    processing_state["error"] = f"Pose engine unavailable: {exc}"

            if engine is not None:
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
                    set_analysis_result(
                        metrics,
                        timestamp_ms,
                        pose_ms,
                        pose_valid=bool(getattr(engine, "last_pose_valid", False)),
                        pose_quality=getattr(engine, "last_pose_quality", 0.0),
                    )
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
    finally:
        if not engine_task.done():
            engine_task.cancel()
        release_camera(session_id)
        processing_state["running"] = False


async def stream_live_preview():
    session_id = source_state["session_id"]
    try:
        cap = await asyncio.to_thread(open_camera, int(source_state["camera_index"]), session_id)
    except Exception as exc:
        processing_state["error"] = f"Camera unavailable: {exc}"
        return
    missed_frames = 0

    try:
        while (
            source_state["source"] == "live"
            and source_state["session_id"] == session_id
            and not source_state["detect_enabled"]
        ):
            started = time.time()
            ok, frame = await asyncio.to_thread(read_camera_frame, cap, session_id)
            if not ok:
                missed_frames += 1
                if missed_frames >= 5:
                    try:
                        cap = await reopen_live_camera(session_id, "Camera frame dropped; reconnecting")
                        missed_frames = 0
                    except Exception as exc:
                        processing_state["error"] = f"Camera reconnect failed: {exc}"
                        await asyncio.sleep(0.75)
                else:
                    processing_state["error"] = "Camera frame not available"
                    await asyncio.sleep(0.08)
                continue
            missed_frames = 0
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
    finally:
        release_camera(session_id)


async def stream_video():
    processing_state["running"] = True
    session_id = source_state["session_id"]
    release_camera(force=True)
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
                set_analysis_result(
                    metrics,
                    timestamp_ms,
                    pose_ms,
                    pose_valid=bool(getattr(engine, "last_pose_valid", False)),
                    pose_quality=getattr(engine, "last_pose_quality", 0.0),
                )
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
        "raw": filter_active_metrics(processing_state.get("latest_raw_metrics", {})),
        "smoothed": filter_active_metrics(processing_state.get("latest_metrics", {})),
        "enabled": active_metric_names(),
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
    old_session_id = source_state["session_id"]
    source_state["session_id"] += 1
    source_state["detect_enabled"] = False
    await asyncio.sleep(0.25)
    await asyncio.to_thread(release_camera, old_session_id)
    processing_state["error"] = None
    return {"status": "released", **state_payload()}


def request_process_shutdown(delay: float = 0.25) -> None:
    def _shutdown():
        time.sleep(delay)
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.75)
        except Exception:
            pass
        os._exit(0)

    threading.Thread(target=_shutdown, daemon=True).start()


@app.post("/api/shutdown")
async def api_shutdown():
    """Stop streams, release the camera, then exit the process (UI Quit button)."""
    source_state["session_id"] += 1
    source_state["detect_enabled"] = False
    try:
        await asyncio.to_thread(release_camera, None, True)
    except Exception:
        pass
    request_process_shutdown()
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
    osc_targets: str = Form(""),
    osc_enabled: bool = Form(True),
    osc_mode: str = Form("raw"),
    osc_alpha: float = Form(1.0),
    osc_namespace: str = Form(""),
):
    try:
        targets = parse_osc_targets(osc_targets, osc_host, osc_port)
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}
    osc_sender.configure(
        enabled=osc_enabled,
        mode=osc_mode,
        alpha=osc_alpha,
        namespace=osc_namespace,
    )
    if osc_sender.mode != "fixed":
        osc_sender.clear_metric_ranges()
        calibration_state["applied_preset"] = None
    try:
        osc_sender.configure_targets(targets)
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}
    refresh_prepared_metrics()

    return {"status": "applied", **state_payload()}


@app.post("/api/smoothing")
async def apply_smoothing(
    smooth_enabled: bool = Form(False),
    smooth_min_cutoff: float = Form(0.3),
):
    """Legacy endpoint kept for compatibility; viewer joint smoothing stays off."""
    source_state["smooth_enabled"] = False
    source_state["smooth_min_cutoff"] = float(smooth_min_cutoff)
    if pose_engine is not None:
        pose_engine.configure_smoothing(enabled=False,
                                        min_cutoff=source_state["smooth_min_cutoff"])
    return {"status": "disabled"}


@app.post("/api/metric_smoothing")
async def apply_metric_smoothing(metric: str = Form(...), alpha: float = Form(...)):
    """Live per-metric OSC output smoothing (no stream restart).
    alpha in (0,1], 1 = off."""
    try:
        osc_sender.set_metric_alpha(metric, alpha)
    except ValueError:
        return {"status": "error", "error": "alpha must be > 0 and <= 1"}
    return {"status": "applied", "metric": metric, "alpha": float(alpha)}


@app.post("/api/metrics/enabled")
async def apply_metric_enabled(metric: str = Form(...), enabled: bool = Form(True)):
    if metric not in METRIC_NAMES:
        return {"status": "error", "error": "unknown metric"}
    current = active_metric_names()
    if enabled and metric not in current:
        current.append(metric)
    elif not enabled:
        current = [name for name in current if name != metric]
    source_state["osc_metrics"] = [name for name in METRIC_NAMES if name in current]
    processing_state["latest_metrics"] = filter_active_metrics(processing_state.get("latest_metrics", {}))
    return {"status": "applied", **state_payload()}


@app.post("/api/calibration/start")
async def calibration_start():
    calibration_collector.reset()
    osc_sender.clear_metric_ranges()
    if osc_sender.mode != "raw":
        osc_sender.configure(mode="raw")
    osc_sender.reset_state()
    reset_pose_metric_history()
    now = time.time()
    calibration_state["active"] = True
    calibration_state["started_at"] = now
    calibration_state["countdown_until"] = now + CALIBRATION_COUNTDOWN_SECONDS
    calibration_state["sample_count"] = 0
    calibration_state["skipped_frames"] = 0
    calibration_state["applied_preset"] = None
    processing_state["latest_metrics"] = {}
    return {"status": "started", **state_payload()}


@app.post("/api/calibration/stop")
async def calibration_stop(name: str = Form(...)):
    clean_name = str(name or "").strip()
    if not clean_name:
        return {"status": "error", "error": "profile name is required"}
    ranges = calibration_collector.ranges(min_samples=10)
    calibration_state["active"] = False
    calibration_state["started_at"] = None
    calibration_state["countdown_until"] = None
    calibration_state["sample_count"] = calibration_collector.count()
    if not ranges:
        return {"status": "error", "error": "not enough valid calibration samples"}
    calibration_presets[clean_name] = ranges
    save_presets(CALIBRATION_PRESETS_PATH, calibration_presets)
    osc_sender.reset_state()
    reset_pose_metric_history()
    processing_state["latest_metrics"] = {}
    return {"status": "saved", "profile": clean_name, **state_payload()}


@app.post("/api/calibration/apply")
async def calibration_apply(name: str = Form(...)):
    clean_name = str(name or "").strip()
    ranges = calibration_presets.get(clean_name)
    if not ranges:
        return {"status": "error", "error": "unknown calibration profile"}
    osc_sender.clear_metric_ranges()
    osc_sender.set_metric_ranges(ranges)
    osc_sender.configure(mode="fixed")
    osc_sender.reset_state()
    reset_pose_metric_history()
    calibration_state["applied_preset"] = clean_name
    refresh_prepared_metrics()
    return {"status": "applied", "profile": clean_name, **state_payload()}


@app.post("/api/calibration/clear")
async def calibration_clear():
    osc_sender.clear_metric_ranges()
    if osc_sender.mode == "fixed":
        osc_sender.configure(mode="raw")
    osc_sender.reset_state()
    reset_pose_metric_history()
    calibration_state["applied_preset"] = None
    refresh_prepared_metrics()
    return {"status": "cleared", **state_payload()}


@app.get("/api/calibration/export")
async def calibration_export(name: str = ""):
    clean_name = str(name or "").strip()
    presets = serialize_calibration_presets()
    if clean_name:
        ranges = presets.get(clean_name)
        if not ranges:
            body = json.dumps({"status": "error", "error": "unknown calibration profile"}, ensure_ascii=False)
            return Response(content=body, media_type="application/json", status_code=404)
        payload = {clean_name: ranges}
        filename = f"field-calibration-{safe_download_stem(clean_name)}.json"
    else:
        payload = presets
        filename = f"field-calibration-all-{time.strftime('%Y-%m-%d-%H-%M')}.json"

    body = json.dumps(payload, indent=2, ensure_ascii=False)
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/calibration/import")
async def calibration_import(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > 2 * 1024 * 1024:
        return {"status": "error", "error": "preset file is too large"}
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except Exception:
        return {"status": "error", "error": "invalid preset JSON"}

    fallback_name = Path(file.filename or "imported").stem or "imported"
    imported = normalize_presets(data, default_name=fallback_name)
    if not imported:
        return {"status": "error", "error": "no valid calibration presets found"}

    calibration_presets.update(imported)
    save_presets(CALIBRATION_PRESETS_PATH, calibration_presets)

    applied = calibration_state.get("applied_preset")
    if applied in imported and osc_sender.mode == "fixed":
        osc_sender.clear_metric_ranges()
        osc_sender.set_metric_ranges(calibration_presets[applied])
        osc_sender.reset_state()
        refresh_prepared_metrics()

    return {
        "status": "imported",
        "profiles": sorted(imported.keys()),
        "count": len(imported),
        **state_payload(),
    }


@app.post("/api/apply")
async def apply_input(
    source: str = Form("live"),
    camera_index: int = Form(0),
    mirror_live: bool = Form(False),
    loop: bool = Form(True),
    detect_enabled: bool = Form(False),
    osc_host: str = Form("127.0.0.1"),
    osc_port: int = Form(9000),
    osc_targets: str = Form(""),
    osc_enabled: bool = Form(True),
    osc_mode: str = Form("raw"),
    osc_alpha: float = Form(1.0),
    osc_namespace: str = Form(""),
    performance: str = Form(DEFAULT_PERFORMANCE),
    pose_backend: str = Form("mediapipe"),
    smooth_enabled: bool = Form(False),
    smooth_min_cutoff: float = Form(0.3),
    video: Optional[UploadFile] = File(None),
):
    if source not in {"live", "video"}:
        return {"status": "error", "error": "source must be live or video"}

    try:
        targets = parse_osc_targets(osc_targets, osc_host, osc_port)
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}
    osc_sender.configure(
        enabled=osc_enabled,
        mode=osc_mode,
        alpha=osc_alpha,
        namespace=osc_namespace,
    )
    if osc_sender.mode != "fixed":
        osc_sender.clear_metric_ranges()
        calibration_state["applied_preset"] = None
    try:
        osc_sender.configure_targets(targets)
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}

    available_backend_ids = {item["id"] for item in available_pose_backends()}
    source_state["pose_backend"] = pose_backend if pose_backend in available_backend_ids else "mediapipe"
    source_state["smooth_enabled"] = False
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

    if source == "live":
        release_camera(force=True)
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
    processing_state["pose_valid"] = False
    processing_state["pose_quality"] = 0.0
    processing_state["last_frame_at"] = None
    processing_state["signal_mean"] = None
    processing_state["morrisness"] = None
    processing_state["error"] = None
    osc_sender.reset_state()
    reset_pose_metric_history()
    if calibration_state.get("applied_preset") in calibration_presets:
        osc_sender.set_metric_ranges(calibration_presets[calibration_state["applied_preset"]])
    if culture_score is not None:
        culture_score.reset()
    return {"status": "applied", **state_payload()}


@app.get("/stream")
async def stream():
    if not source_state["detect_enabled"]:
        return StreamingResponse(iter(()), media_type="multipart/x-mixed-replace; boundary=frame")
    source_state["session_id"] += 1
    if source_state["source"] == "video":
        generator = stream_video()
    else:
        generator = stream_live()
    return StreamingResponse(generator, media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/preview_stream")
async def preview_stream():
    if source_state["detect_enabled"]:
        return StreamingResponse(iter(()), media_type="multipart/x-mixed-replace; boundary=frame")
    source_state["session_id"] += 1
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
    .input-row { display: grid; grid-template-columns: 1fr; gap: 12px; align-items: start; }
    .input-source-toggle {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .source-choice {
      min-height: 36px;
      background: rgba(12, 11, 9, .72);
      border-color: var(--line);
      color: var(--muted);
    }
    .source-choice.active {
      background: rgba(52, 211, 192, .16);
      border-color: rgba(52, 211, 192, .65);
      color: var(--teal);
    }
    .input-source-panel.hidden { display: none; }
    .camera-row { display: grid; grid-template-columns: 1fr; gap: 8px; align-items: end; }
    .video-row { display: grid; grid-template-columns: 1fr; gap: 8px; align-items: end; }
    .file-name {
      min-height: 18px;
      color: var(--muted);
      font: 12px ui-monospace, monospace;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
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
    .input-actions {
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      flex-wrap: wrap;
    }
    .enter-button,
    .calibrate-enter-button {
      width: auto;
      min-width: 96px;
    }
    .calibrate-enter-button {
      background: rgba(84, 179, 168, .18);
      border-color: rgba(84, 179, 168, .68);
      color: var(--teal);
    }
    .calibrate-enter-button:hover { background: rgba(84, 179, 168, .28); }
    .calibration-overlay {
      position: absolute;
      left: 50%;
      top: 16px;
      transform: translateX(-50%);
      z-index: 4;
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 9px 14px;
      border: 1px solid rgba(217, 139, 95, .72);
      border-radius: 999px;
      background: rgba(20, 17, 14, .82);
      color: var(--amber);
      font: 13px ui-monospace, monospace;
      letter-spacing: .08em;
      text-transform: uppercase;
      pointer-events: none;
      box-shadow: 0 10px 30px rgba(0,0,0,.36);
      backdrop-filter: blur(4px);
    }
    .calibration-overlay::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--amber);
      box-shadow: 0 0 0 5px rgba(217, 139, 95, .14);
      animation: pulse 1.4s ease-in-out infinite;
    }
    .calibration-overlay.countdown {
      top: 50%;
      transform: translate(-50%, -50%);
      min-width: 170px;
      padding: 18px 24px 20px;
      flex-direction: column;
      gap: 8px;
      border-radius: 8px;
      background: rgba(20, 17, 14, .74);
      color: var(--text);
      letter-spacing: 0;
      text-transform: none;
    }
    .calibration-overlay.countdown::before { display: none; }
    .countdown-label {
      color: var(--amber);
      font-size: 12px;
      line-height: 1;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .countdown-number {
      color: var(--text);
      font-size: 92px;
      font-weight: 800;
      line-height: .86;
      letter-spacing: 0;
      text-shadow: 0 8px 30px rgba(0,0,0,.56);
    }
    @media (max-width: 640px) {
      .countdown-number { font-size: 68px; }
      .calibration-overlay.countdown { min-width: 142px; }
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
      grid-template-columns: 1fr;
      gap: 12px;
      align-items: center;
      padding: 12px;
      border-bottom: 1px solid var(--line);
      background: #171411;
    }
    .switch-row {
      position: relative;
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 9px;
      color: var(--muted);
      font-size: 12px;
      letter-spacing: .04em;
      min-height: 34px;
      white-space: nowrap;
    }
    .calibration-panel {
      display: grid;
      gap: 10px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
      background: #171411;
    }
    .calibration-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .calibration-head .section-title { margin: 0; }
    .calibration-status {
      color: var(--muted);
      font: 12px ui-monospace, monospace;
      text-align: right;
      min-width: 92px;
    }
    .calibration-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 88px;
      gap: 8px;
      align-items: end;
    }
    .calibration-row button {
      min-height: 38px;
      padding: 7px 9px;
      font-size: 11px;
    }
    .calibration-row button.secondary {
      background: rgba(29, 26, 23, .84);
      color: var(--text);
      border-color: var(--line);
    }
    .calibration-row button.secondary:hover { background: rgba(54, 47, 40, .92); }
    .calibration-row button.recording {
      background: rgba(217, 139, 95, .22);
      border-color: rgba(217, 139, 95, .72);
      color: var(--amber);
    }
    .calibration-file-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .calibration-file-row button {
      min-height: 34px;
      padding: 7px 9px;
      font-size: 11px;
    }
    .calibration-file-row button.secondary {
      background: rgba(29, 26, 23, .84);
      color: var(--text);
      border-color: var(--line);
    }
    .calibration-file-row button.secondary:hover { background: rgba(54, 47, 40, .92); }
    .calibration-row input,
    .calibration-row select {
      min-height: 38px;
      padding: 8px 9px;
    }
    .normalize-profile-row {
      display: grid;
      grid-column: 1 / -1;
      grid-template-columns: 1fr;
      gap: 6px;
      padding-top: 2px;
    }
    .switch-row input {
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }
    .switch-track {
      position: relative;
      width: 38px;
      height: 18px;
      border-radius: 999px;
      background: #453d35;
      border: 1px solid #5a5046;
      transition: background .12s ease, border-color .12s ease;
    }
    .switch-track::after {
      content: "";
      position: absolute;
      top: 3px;
      left: 3px;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: #a69a8b;
      transition: transform .12s ease, background .12s ease;
    }
    .switch-row input:checked + .switch-track {
      background: rgba(79, 189, 176, .28);
      border-color: rgba(79, 189, 176, .65);
    }
    .switch-row input:checked + .switch-track::after {
      transform: translateX(20px);
      background: var(--teal);
    }
    .label-row { display: flex; align-items: center; gap: 6px; }
    [data-tooltip-title], [data-tooltip-body] { cursor: help; }
    .range-label-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .range-value { color: var(--text); font: 12px ui-monospace, monospace; }
    .custom-tooltip {
      position: fixed;
      z-index: 30;
      max-width: 340px;
      padding: 12px 14px;
      border: 1px solid rgba(79, 189, 176, .45);
      border-radius: 8px;
      background: rgba(20, 17, 14, .97);
      box-shadow: 0 14px 38px rgba(0, 0, 0, .42);
      color: var(--text);
      font: 14px/1.4 Inter, system-ui, sans-serif;
      pointer-events: none;
      opacity: 0;
      visibility: hidden;
      transition: opacity .08s ease;
    }
    .custom-tooltip.visible { opacity: 1; visibility: visible; }
    .tooltip-title {
      margin-bottom: 6px;
      color: var(--teal);
      font: 13px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
      letter-spacing: .04em;
    }
    .tooltip-body { color: #e2d8cb; font-size: 15px; }
    .thin-range {
      -webkit-appearance: none;
      appearance: none;
      width: 100%;
      height: 10px;
      margin: 0;
      padding: 0;
      background: transparent;
      --range-fill: 0%;
      cursor: pointer;
    }
    .thin-range::-webkit-slider-runnable-track {
      height: 3px;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--teal) 0 var(--range-fill), #4a4239 var(--range-fill) 100%);
    }
    .thin-range::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 9px;
      height: 9px;
      margin-top: -3px;
      border-radius: 50%;
      background: var(--teal);
      border: 0;
    }
    .thin-range::-moz-range-track {
      height: 3px;
      border-radius: 999px;
      background: #4a4239;
    }
    .thin-range::-moz-range-progress {
      height: 3px;
      border-radius: 999px;
      background: var(--teal);
    }
    .thin-range::-moz-range-thumb {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--teal);
      border: 0;
    }
    .thin-range:disabled { opacity: .35; cursor: default; }
    .thin-range:focus-visible { outline: 1px solid rgba(79, 189, 176, .55); outline-offset: 3px; }
    .metric-grid { display: grid; grid-template-columns: 1fr; gap: 9px; padding: 12px; }
    .metric {
      display: grid;
      grid-template-columns: 22px 28px minmax(0, 1fr) minmax(230px, 250px);
      align-items: center;
      gap: 10px;
      min-height: 52px;
      padding: 9px;
      background: var(--surface-soft);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .metric.disabled { opacity: .45; }
    .metric.dragging { opacity: .28; }
    .metric.drag-over { border-color: rgba(84,179,168,.78); }
    .metric-drag {
      width: 22px;
      min-height: 32px;
      padding: 0;
      display: grid;
      place-items: center;
      background: transparent;
      border: 0;
      border-radius: 6px;
      cursor: grab;
    }
    .metric-drag:active { cursor: grabbing; }
    .metric-drag::before {
      content: "";
      width: 12px;
      height: 16px;
      opacity: .62;
      background:
        radial-gradient(circle, var(--muted) 1.2px, transparent 1.8px) 0 0 / 6px 6px,
        radial-gradient(circle, var(--muted) 1.2px, transparent 1.8px) 6px 0 / 6px 6px;
    }
    .metric-drag:hover::before { opacity: .9; }
    .metric-enable { display: grid; place-items: center; }
    .metric-enable input { width: 16px; height: 16px; accent-color: var(--teal); }
    .metric-label { display: grid; gap: 3px; min-width: 0; }
    .name { color: var(--muted); font-size: 13px; letter-spacing: .02em; overflow-wrap: anywhere; }
    .metric-smooth-row { display: flex; align-items: center; gap: 7px; justify-self: end; width: 250px; }
    .metric-smooth-row .ms-label { color: #756b60; font-size: 10px; letter-spacing: 0; white-space: nowrap; }
    .metric-smooth-row input[type=range] {
      flex: 1;
      min-width: 0;
    }
    .metric-smooth-row .ms-val { color: var(--teal); font: 12px ui-monospace, monospace; min-width: 28px; text-align: right; white-space: nowrap; }
    .metric-readout {
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: max-content minmax(0, 1fr);
      align-items: center;
      gap: 10px;
      width: 100%;
    }
    .value {
      font: 16px ui-monospace, SFMono-Regular, Menlo, monospace;
      font-variant-numeric: tabular-nums;
      min-width: 4.5ch;
      text-align: left;
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
    .osc-toolbar {
      display: grid;
      grid-template-columns: max-content minmax(130px, 220px) max-content;
      gap: 10px;
      align-items: center;
      justify-content: start;
    }
    .osc-toolbar .section-title { margin: 0; }
    .osc-prefix {
      display: grid;
      grid-template-columns: max-content minmax(0, 1fr);
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .osc-prefix input { min-height: 34px; padding: 6px 10px; }
    .add-osc-target {
      width: auto;
      min-height: 32px;
      padding: 6px 10px;
      font-size: 11px;
    }
    .osc-targets { display: grid; gap: 8px; margin-top: 10px; }
    .osc-empty {
      margin-top: 9px;
      color: var(--muted);
      font: 12px ui-monospace, monospace;
    }
    .osc-target-row {
      display: grid;
      grid-template-columns: minmax(88px, .8fr) minmax(118px, 1fr) 82px 42px;
      gap: 8px;
      align-items: end;
    }
    .osc-target-row label { min-width: 0; }
    .osc-target-row input { min-width: 0; }
    .osc-remove {
      width: 42px;
      min-height: 42px;
      padding: 0;
      align-self: end;
      background: rgba(29, 26, 23, .84);
      color: var(--muted);
      border-color: var(--line);
      font-size: 18px;
      line-height: 1;
    }
    .osc-remove:hover { color: var(--red); border-color: rgba(232, 112, 91, .7); background: rgba(60, 20, 16, .35); }
    .metric-address-panel { border-top: 1px solid var(--line); padding: 12px; background: #171411; }
    .output-header {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 190px;
      gap: 12px;
      align-items: end;
      margin-bottom: 10px;
    }
    .output-header .section-title { margin: 0 0 8px; }
    .output-mode select { min-height: 34px; padding: 6px 10px; }
    .address-list { display: grid; gap: 7px; color: var(--muted); font: 13px ui-monospace, monospace; }
    .address-list div { overflow-wrap: anywhere; }
    @media (max-width: 1080px) {
      .layout { grid-template-columns: 1fr; }
      .controls-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .apply-row { grid-template-columns: 1fr; }
    }
    @media (max-width: 680px) {
      header { align-items: flex-start; flex-direction: column; }
      .controls-grid, .osc-toolbar, .osc-prefix, .osc-target-row, .input-source-toggle, .metric-osc-controls, .calibration-row, .meta, .output-header { grid-template-columns: 1fr; }
      .osc-remove { width: 100%; }
      .switch-row { justify-content: flex-start; }
      .metric { grid-template-columns: 22px 28px minmax(0, 1fr); align-items: start; }
      .metric-smooth-row { grid-column: 1 / -1; justify-self: stretch; width: 100%; }
      .metric-readout { grid-column: 1 / -1; }
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
                <div class="input-source-toggle" role="group" aria-label="Input source">
                  <button id="cameraSourceButton" class="source-choice active" type="button">Camera</button>
                  <button id="videoSourceButton" class="source-choice" type="button">Video</button>
                </div>
                <label id="cameraInputPanel" class="input-source-panel">Camera
                  <div class="camera-row">
                    <select id="camera" name="camera_index">
                      <option value="0">0 - Camera 0</option>
                    </select>
                    <span class="mirror-row"><input id="mirrorLive" name="mirror_live" type="checkbox" checked /> Mirror camera</span>
                  </div>
                </label>
                <label id="videoInputPanel" class="input-source-panel hidden">Video
                  <div class="video-row">
                    <input id="videoFile" name="video" type="file" accept="video/*" />
                    <span id="videoFileName" class="file-name">No video selected</span>
                    <span class="mirror-row"><input id="loopVideo" type="checkbox" checked /> Loop video</span>
                  </div>
                </label>
              </div>
              <label class="model-row">Detection model
                <select id="poseBackend" name="pose_backend">
                  <option value="mediapipe">MediaPipe</option>
                </select>
              </label>
              <div class="input-actions">
                <button id="enterInputButton" class="enter-button" type="button">Enter</button>
                <button id="enterCalibrationButton" class="calibrate-enter-button" type="button">Calibrate</button>
              </div>
            </div>
          </div>
          <div id="emptyState" class="empty hidden"></div>
          <div id="metricOverlay" class="metric-overlay"></div>
          <div id="calibrationOverlay" class="calibration-overlay hidden">Calibrating 0</div>
          <button id="changeInputButton" class="change-input hidden" type="button">Change Input</button>
          <div id="controlBar" class="control-bar hidden">
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
          <div class="osc-toolbar">
            <p class="section-title">OSC</p>
            <label class="osc-prefix">Prefix
              <input id="oscNamespace" name="osc_namespace" value="/field" />
            </label>
            <button id="addOscTarget" class="add-osc-target" type="button">Add Output</button>
          </div>
          <div id="oscTargets" class="osc-targets"></div>
          <div id="oscTargetsEmpty" class="osc-empty hidden">No OSC outputs</div>
          <input id="oscTargetsInput" name="osc_targets" type="hidden" value="" />
        </div>
        <div class="metric-address-panel">
          <div class="output-header">
            <p class="section-title">Output values</p>
          </div>
          <div id="addresses" class="address-list"></div>
        </div>
      </section>

      <aside class="panel">
        <div class="calibration-panel">
          <div class="calibration-head">
            <p class="section-title">Calibration</p>
            <span id="calibrationStatus" class="calibration-status">idle</span>
          </div>
          <div class="calibration-row">
            <label>Profile
              <input id="calibrationName" type="text" placeholder="stage-a" autocomplete="off" />
            </label>
            <button id="calibrationStart" type="button">Start</button>
          </div>
          <div class="calibration-row">
            <label class="normalize-profile-row" data-tooltip-title="Normalize Profile" data-tooltip-body="None sends raw metric values. Saved calibration profiles use fixed ranges from Smoothness(EMA) output samples.">
              Normalize profile
              <select id="normalizeProfile"></select>
            </label>
          </div>
          <div class="calibration-file-row">
            <button id="calibrationExport" class="secondary" type="button">Export</button>
            <button id="calibrationLoad" class="secondary" type="button">Load</button>
            <input id="calibrationPresetFile" class="hidden" type="file" accept=".json,application/json" />
          </div>
        </div>
        <div id="metrics" class="metric-grid"></div>
        <div class="charts-link" style="padding:0 14px 12px;"><a href="/charts" target="_blank" style="color:var(--teal);text-decoration:none;font-size:13px;">Open live charts &#8599;</a></div>
        <div id="culturePanel" class="culture-panel hidden">
          <p class="section-title" data-tooltip-title="Culture Axis" data-tooltip-body="A rolling average of the 9 metrics compared against the Morris and BaYe training clips. 1.0 matches Morris, 0.0 matches BaYe, 0.5 sits between the two.">Culture Axis</p>
          <div class="culture-row">
            <span class="culture-end baye">BAYE</span>
            <div class="culture-track"><div id="cultureMarker" class="culture-marker"></div></div>
            <span class="culture-end morris">MORRIS</span>
          </div>
          <div id="cultureValue" class="culture-value">/field/morrisness -</div>
        </div>
      </aside>
    </section>
    <div id="customTooltip" class="custom-tooltip" role="tooltip"></div>
    <input id="source" name="source" type="hidden" value="live" />
    <input id="loop" name="loop" type="hidden" value="true" />
    <input id="detectEnabled" name="detect_enabled" type="hidden" value="false" />
    </form>
  </main>
  <script>
    const metricNames = %METRICS%;
    const oscAddressNames = %OSC_ADDRESS_NAMES%;
    const defaultMetricEmaFrames = %DEFAULT_METRIC_EMA_FRAMES%;
    const metricsEl = document.getElementById('metrics');
    const metricOrderStorageKey = 'field.metricOrder.v1';
    const oscTargetsEl = document.getElementById('oscTargets');
    const tooltipEl = document.getElementById('customTooltip');
    const maxSeen = {};
    let lastMetricBarScaleMode = null;
    let metricEmaInitialized = false;
    const metricLabels = {
      energy: 'Energy',
      sync_velocity: 'Sync Velocity',
      sync_correlation: 'Sync Correlation',
      expansion: 'Expansion',
      curvature: 'Curvature',
      height: 'Height',
      sway: 'Sway',
      torque: 'Torque',
      jerk: 'Jerk',
    };
    const metricDescriptions = {
      energy: 'Overall movement intensity from limb rotation speed. Higher means bigger or faster full-body motion; near 0 means stillness.',
      sync_velocity: 'Left-right activity balance. 1 means both sides are similarly active; 0 means one side is doing most of the movement.',
      sync_correlation: 'Left-right timing match, not movement size. +1 means both sides move in the same rhythm, 0 means no clear timing link, -1 means they alternate or move opposite each other.',
      expansion: 'How much space the body occupies. Higher means open or spread-out shapes; lower means compact or closed shapes.',
      curvature: 'How rounded the hand and foot paths are. Higher means curved/circular paths; lower means straighter paths.',
      height: 'Body center height above the foot base. Lower values usually mean crouching, folding, or dropping the torso.',
      sway: 'Horizontal drift of the body center away from the feet. Higher means leaning, traveling, or less steady balance.',
      torque: 'How strongly movement changes speed around the joints. Higher means sharper accents or more forceful changes.',
      jerk: 'Abruptness of acceleration changes. Higher means jagged or twitchy motion; lower means smoother, more continuous flow.',
    };
    function labelForMetric(name) {
      return metricLabels[name] || name;
    }
    function setTooltipTarget(el, title, body) {
      el.dataset.tooltipTitle = title || '';
      el.dataset.tooltipBody = body || '';
    }
    function tooltipTargetFrom(eventTarget) {
      if (eventTarget?.closest?.('.metric-drag')) return null;
      return eventTarget?.closest?.('[data-tooltip-title], [data-tooltip-body]');
    }
    function setTooltipPosition(event) {
      const margin = 14;
      let left = event.clientX + 16;
      let top = event.clientY + 16;
      tooltipEl.style.left = `${left}px`;
      tooltipEl.style.top = `${top}px`;
      const rect = tooltipEl.getBoundingClientRect();
      if (rect.right > window.innerWidth - margin) left = event.clientX - rect.width - 16;
      if (rect.bottom > window.innerHeight - margin) top = event.clientY - rect.height - 16;
      tooltipEl.style.left = `${Math.max(margin, left)}px`;
      tooltipEl.style.top = `${Math.max(margin, top)}px`;
    }
    function showTooltip(target, event) {
      const title = target.dataset.tooltipTitle || '';
      const body = target.dataset.tooltipBody || '';
      if (!title && !body) return;
      tooltipEl.innerHTML = '';
      if (title) {
        const titleEl = document.createElement('div');
        titleEl.className = 'tooltip-title';
        titleEl.textContent = title;
        tooltipEl.appendChild(titleEl);
      }
      if (body) {
        const bodyEl = document.createElement('div');
        bodyEl.className = 'tooltip-body';
        bodyEl.textContent = body;
        tooltipEl.appendChild(bodyEl);
      }
      tooltipEl.classList.add('visible');
      setTooltipPosition(event);
    }
    function hideTooltip() {
      tooltipEl.classList.remove('visible');
    }
    document.addEventListener('pointermove', event => {
      const target = tooltipTargetFrom(event.target);
      if (!target) {
        hideTooltip();
        return;
      }
      showTooltip(target, event);
    });
    document.addEventListener('pointerleave', hideTooltip);
    document.addEventListener('scroll', hideTooltip, true);
    const centeredMetrics = new Set(['sync_correlation']);
    const metricOverlayEl = document.getElementById('metricOverlay');
    function snapEmaFrames(value) {
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) return defaultMetricEmaFrames;
      return Math.max(0, Math.min(10, Math.round(numeric)));
    }
    function setRangeFill(input, value = input.value) {
      const pct = snapEmaFrames(value) * 10;
      input.style.setProperty('--range-fill', `${pct}%`);
    }
    function emaFramesToAlpha(frames) {
      const count = snapEmaFrames(frames);
      if (count <= 0) return 1;
      return 2 / (count + 1);
    }
    function alphaToEmaFrames(alpha) {
      const numeric = Number(alpha);
      if (!Number.isFinite(numeric)) return defaultMetricEmaFrames;
      if (numeric >= 1) return 0;
      return snapEmaFrames((2 / numeric) - 1);
    }
    function formatEmaFrames(frames) {
      return `${snapEmaFrames(frames)}f`;
    }

    function normalizeMetricOrder(order = metricNames) {
      const seen = new Set();
      const clean = [];
      for (const name of order || []) {
        if (metricNames.includes(name) && !seen.has(name)) {
          clean.push(name);
          seen.add(name);
        }
      }
      for (const name of metricNames) {
        if (!seen.has(name)) clean.push(name);
      }
      return clean;
    }

    function loadMetricOrder() {
      try {
        return normalizeMetricOrder(JSON.parse(localStorage.getItem(metricOrderStorageKey) || '[]'));
      } catch (error) {
        return normalizeMetricOrder();
      }
    }

    function saveMetricOrder() {
      try {
        localStorage.setItem(metricOrderStorageKey, JSON.stringify(metricOrder));
      } catch (error) {}
    }

    function orderedMetricNames(activeSet = null) {
      const order = normalizeMetricOrder(metricOrder);
      metricOrder = order;
      if (!activeSet) return order;
      return [
        ...order.filter(name => activeSet.has(name)),
        ...order.filter(name => !activeSet.has(name)),
      ];
    }

    function applyMetricOrder(activeSet = null) {
      const ordered = orderedMetricNames(activeSet);
      const current = Array.from(metricsEl.children).map(row => row.id.replace(/^metric-/, ''));
      if (current.join('|') !== ordered.join('|')) {
        for (const name of ordered) {
          const row = document.getElementById(`metric-${name}`);
          if (row) metricsEl.appendChild(row);
        }
      }

      const currentOverlay = Array.from(metricOverlayEl.children).map(row => row.id.replace(/^ov-row-/, ''));
      if (currentOverlay.join('|') !== ordered.join('|')) {
        for (const name of ordered) {
          const overlayRow = document.getElementById(`ov-row-${name}`);
          if (overlayRow) metricOverlayEl.appendChild(overlayRow);
        }
      }
    }

    function moveMetricBeforeOrAfter(dragName, targetName, afterTarget) {
      if (!dragName || !targetName || dragName === targetName) return;
      const next = normalizeMetricOrder(metricOrder).filter(name => name !== dragName);
      const targetIndex = next.indexOf(targetName);
      if (targetIndex < 0) return;
      next.splice(targetIndex + (afterTarget ? 1 : 0), 0, dragName);
      metricOrder = normalizeMetricOrder(next);
      saveMetricOrder();
      const activeSet = new Set(lastPayload?.source?.osc_metrics || metricNames);
      applyMetricOrder(activeSet);
    }

    let metricOrder = loadMetricOrder();

    function refreshEmaFrameLabels() {
      for (const row of document.querySelectorAll('.metric')) {
        const smooth = row.querySelector('.metric-smooth');
        const smoothValue = row.querySelector('.ms-val');
        if (!smooth || !smoothValue) continue;
        smoothValue.textContent = formatEmaFrames(smooth.value);
      }
    }

    function makeOscTargetId() {
      if (window.crypto?.randomUUID) return window.crypto.randomUUID();
      return `target-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }

    function defaultOscTarget(index = 0) {
      return {
        id: makeOscTargetId(),
        name: `Output ${index + 1}`,
        host: '127.0.0.1',
        port: 9000,
        enabled: true,
        broadcast: false,
      };
    }

    function isBroadcastHost(host) {
      const value = String(host || '').trim();
      return value === '255.255.255.255' || value.endsWith('.255');
    }

    function readOscTargets() {
      return Array.from(oscTargetsEl.querySelectorAll('.osc-target-row')).map((row, index) => ({
        id: row.dataset.id || makeOscTargetId(),
        name: row.querySelector('.osc-target-name').value.trim() || `Output ${index + 1}`,
        host: row.querySelector('.osc-target-host').value.trim(),
        port: Number(row.querySelector('.osc-target-port').value || 9000),
        enabled: true,
        broadcast: isBroadcastHost(row.querySelector('.osc-target-host').value),
      }));
    }

    function syncOscTargetsInput() {
      document.getElementById('oscTargetsInput').value = JSON.stringify(readOscTargets());
    }

    function updateOscEmptyState() {
      document.getElementById('oscTargetsEmpty').classList.toggle('hidden', oscTargetsEl.children.length > 0);
    }

    function oscControlsHaveFocus() {
      const active = document.activeElement;
      if (!active) return false;
      return oscTargetsEl.contains(active) || active.id === 'oscNamespace';
    }

    function addOscTargetRow(target = null) {
      const index = oscTargetsEl.children.length;
      const config = target || defaultOscTarget(index);
      const row = document.createElement('div');
      row.className = 'osc-target-row';
      row.dataset.id = config.id || makeOscTargetId();
      row.innerHTML = `
        <label>Name
          <input class="osc-target-name" value="" placeholder="Sound" />
        </label>
        <label>Target
          <input class="osc-target-host" value="" placeholder="192.168.1.21" />
        </label>
        <label>Port
          <input class="osc-target-port" type="number" min="1" max="65535" value="9000" />
        </label>
        <button class="osc-remove" type="button" title="Remove output">&times;</button>
      `;
      row.querySelector('.osc-target-name').value = config.name || `Output ${index + 1}`;
      row.querySelector('.osc-target-host').value = config.host || '';
      row.querySelector('.osc-target-port').value = Number(config.port || 9000);
      row.querySelectorAll('input').forEach(input => {
        input.addEventListener('input', () => {
          syncOscTargetsInput();
          scheduleOscApply();
        });
        input.addEventListener('change', () => {
          syncOscTargetsInput();
          scheduleOscApply(0);
        });
      });
      row.querySelector('.osc-remove').addEventListener('pointerdown', () => {
        markOscDirty();
      });
      row.querySelector('.osc-remove').addEventListener('click', () => {
        markOscDirty();
        row.remove();
        syncOscTargetsInput();
        updateOscEmptyState();
        scheduleOscApply(0);
      });
      oscTargetsEl.appendChild(row);
      syncOscTargetsInput();
      updateOscEmptyState();
    }

    function renderOscTargets(targets) {
      oscTargetsEl.innerHTML = '';
      const list = Array.isArray(targets) ? targets : [defaultOscTarget(0)];
      for (const target of list) addOscTargetRow(target);
      syncOscTargetsInput();
      updateOscEmptyState();
    }

    for (const name of metricNames) {
      maxSeen[name] = 1;
      const row = document.createElement('div');
      row.className = 'metric';
      row.id = `metric-${name}`;
      setTooltipTarget(row, labelForMetric(name), metricDescriptions[name] || '');
      row.innerHTML = `
        <span class="metric-drag" draggable="true" aria-label="Drag ${labelForMetric(name)}"></span>
        <label class="metric-enable" data-tooltip-title="Output ${labelForMetric(name)}" data-tooltip-body="Checked metrics are included in OSC output, Output values, overlays, and live charts."><input class="metric-toggle" type="checkbox" checked /></label>
        <div class="metric-label" data-tooltip-title="${labelForMetric(name)}" data-tooltip-body="${metricDescriptions[name] || ''}">
          <div class="name">${labelForMetric(name)}</div>
        </div>
        <div class="metric-smooth-row">
          <span class="ms-label">Smoothness(EMA)</span>
          <input class="metric-smooth thin-range" type="range" min="0" max="10" step="1" value="${defaultMetricEmaFrames}" data-tooltip-title="${labelForMetric(name)} Smoothness(EMA)" data-tooltip-body="Per-metric output EMA after metrics are calculated. Value is pose analysis frames, not rendered camera frames. 0f is off; 10f is the strongest setting." />
          <span class="ms-val">${formatEmaFrames(defaultMetricEmaFrames)}</span>
        </div>
        <div class="metric-readout">
          <div class="value" id="v-${name}">0.00</div>
          <div class="bar" id="bar-${name}"><div class="fill" id="b-${name}"></div></div>
        </div>
      `;
      metricsEl.appendChild(row);
      const toggle = row.querySelector('.metric-toggle');
      const dragHandle = row.querySelector('.metric-drag');
      const msInput = row.querySelector('.metric-smooth');
      const msVal = row.querySelector('.ms-val');
      setRangeFill(msInput, defaultMetricEmaFrames);
      dragHandle.addEventListener('dragstart', event => {
        row.classList.add('dragging');
        event.dataTransfer.effectAllowed = 'move';
        event.dataTransfer.setData('text/plain', name);
      });
      dragHandle.addEventListener('dragend', () => {
        row.classList.remove('dragging');
        document.querySelectorAll('.metric.drag-over').forEach(item => item.classList.remove('drag-over'));
      });
      toggle.addEventListener('change', async () => {
        const data = new FormData();
        data.set('metric', name);
        data.set('enabled', toggle.checked ? 'true' : 'false');
        try {
          const res = await fetch('/api/metrics/enabled', { method: 'POST', body: data });
          update(await res.json());
        } catch (e) {}
      });
      msInput.addEventListener('input', () => {
        const snapped = snapEmaFrames(msInput.value);
        setRangeFill(msInput, snapped);
        msVal.textContent = formatEmaFrames(snapped);
      });
      msInput.addEventListener('change', async () => {
        const snapped = snapEmaFrames(msInput.value);
        msInput.value = snapped;
        setRangeFill(msInput, snapped);
        msVal.textContent = formatEmaFrames(snapped);
        const data = new FormData();
        data.set('metric', name);
        data.set('alpha', String(emaFramesToAlpha(snapped)));
        try { await fetch('/api/metric_smoothing', { method: 'POST', body: data }); } catch (e) {}
      });

      const ovRow = document.createElement('div');
      ovRow.className = 'ov-row';
      ovRow.id = `ov-row-${name}`;
      ovRow.innerHTML = `<span class="ov-name">${labelForMetric(name)}</span><span class="ov-val" id="ov-${name}">0.00</span>`;
      metricOverlayEl.appendChild(ovRow);
    }
    applyMetricOrder(new Set(metricNames));

    metricsEl.addEventListener('dragover', event => {
      const dragging = document.querySelector('.metric.dragging');
      const target = event.target.closest('.metric');
      if (!dragging || !target || !metricsEl.contains(target) || target === dragging) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = 'move';
      document.querySelectorAll('.metric.drag-over').forEach(item => {
        if (item !== target) item.classList.remove('drag-over');
      });
      target.classList.add('drag-over');
    });

    metricsEl.addEventListener('dragleave', event => {
      const target = event.target.closest('.metric');
      if (target && !target.contains(event.relatedTarget)) target.classList.remove('drag-over');
    });

    metricsEl.addEventListener('drop', event => {
      const target = event.target.closest('.metric');
      if (!target || !metricsEl.contains(target)) return;
      event.preventDefault();
      const dragging = document.querySelector('.metric.dragging');
      const dragName = event.dataTransfer.getData('text/plain') || dragging?.id?.replace(/^metric-/, '');
      const targetName = target.id.replace(/^metric-/, '');
      const rect = target.getBoundingClientRect();
      const afterTarget = event.clientY > rect.top + (rect.height / 2);
      target.classList.remove('drag-over');
      moveMetricBeforeOrAfter(dragName, targetName, afterTarget);
    });

    async function applySettings(detectEnabled) {
      const form = document.getElementById('form');
      syncOscTargetsInput();
      const data = new FormData(form);
      data.set('loop', document.getElementById('loopVideo').checked ? 'true' : 'false');
      data.set('detect_enabled', detectEnabled ? 'true' : 'false');
      data.set('osc_enabled', 'true');
      data.set('osc_mode', selectedOscMode());
      data.set('smooth_enabled', 'false');
      data.set('smooth_min_cutoff', '0.3');
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
      resetMetricBarRanges();

      return payload;
    }

    function selectedOscMode() {
      const selected = normalizeProfileSelect.value || 'none';
      if (selected === 'none') return 'raw';
      if (selected.startsWith('profile:')) return 'fixed';
      return 'raw';
    }

    function buildOscFormData() {
      const data = new FormData();
      const targets = readOscTargets();
      const primary = targets[0] || defaultOscTarget(0);
      syncOscTargetsInput();
      data.set('osc_host', primary.host || '127.0.0.1');
      data.set('osc_port', String(primary.port || 9000));
      data.set('osc_targets', JSON.stringify(targets));
      data.set('osc_enabled', 'true');
      data.set('osc_mode', selectedOscMode());
      data.set('osc_alpha', '1');
      data.set('osc_namespace', document.getElementById('oscNamespace').value);
      return data;
    }

    async function applyOscSettings() {
      oscApplyInFlight = true;
      try {
        const res = await fetch('/api/osc/config', { method: 'POST', body: buildOscFormData() });
        const payload = await res.json();
        if (payload.status !== 'applied') {
          console.warn('OSC config failed', payload.error || payload);
          return payload;
        }
        oscDirty = false;
        update(payload);
        return payload;
      } finally {
        oscApplyInFlight = false;
      }
    }

    document.getElementById('form').addEventListener('submit', async (event) => {
      event.preventDefault();
    });

    async function loadCameras() {
      try {
        const res = await fetch('/api/cameras');
        const payload = await res.json();
        const select = document.getElementById('camera');
        const currentValue = select.value;
        select.innerHTML = '';
        for (const camera of payload.cameras || []) {
          const option = document.createElement('option');
          option.value = camera.index;
          option.textContent = camera.label;
          select.appendChild(option);
        }
        const preferredValue = String(lastPayload?.source?.camera_index ?? currentValue ?? '0');
        if (Array.from(select.options).some(option => option.value === preferredValue)) {
          select.value = preferredValue;
        }
      } catch (error) {
        console.warn('Camera list unavailable', error);
      }
    }

    const sourceInput = document.getElementById('source');
    const detectInput = document.getElementById('detectEnabled');
    const streamImage = document.getElementById('stream');
    const previewVideo = document.getElementById('previewVideo');
    const cameraSourceButton = document.getElementById('cameraSourceButton');
    const videoSourceButton = document.getElementById('videoSourceButton');
    const cameraInputPanel = document.getElementById('cameraInputPanel');
    const videoInputPanel = document.getElementById('videoInputPanel');
    const videoFileInput = document.getElementById('videoFile');
    const loopVideoInput = document.getElementById('loopVideo');
    const videoFileNameEl = document.getElementById('videoFileName');
    const detectButton = document.getElementById('detectButton');
    const cameraButton = document.getElementById('cameraButton');
    const inputOverlay = document.getElementById('inputOverlay');
    const changeInputButton = document.getElementById('changeInputButton');
    const controlBar = document.getElementById('controlBar');
    const calibrationOverlayEl = document.getElementById('calibrationOverlay');
    const calibrationNameInput = document.getElementById('calibrationName');
    const normalizeProfileSelect = document.getElementById('normalizeProfile');
    const calibrationStatusEl = document.getElementById('calibrationStatus');
    const calibrationStartButton = document.getElementById('calibrationStart');
    const calibrationExportButton = document.getElementById('calibrationExport');
    const calibrationLoadButton = document.getElementById('calibrationLoad');
    const calibrationPresetFileInput = document.getElementById('calibrationPresetFile');

    let isDetecting = false;
    let cameraOn = false;
    let lastPayload = null;
    let inputDirty = false;
    let oscDirty = false;
    let oscApplyTimer = null;
    let oscApplyInFlight = false;
    let videoObjectUrl = null;
    let activeStreamKind = null;

    function currentSource() {
      return sourceInput.value === 'video' ? 'video' : 'live';
    }

    function setSourceMode(mode) {
      const isVideo = mode === 'video';
      sourceInput.value = isVideo ? 'video' : 'live';
      cameraSourceButton.classList.toggle('active', !isVideo);
      videoSourceButton.classList.toggle('active', isVideo);
      cameraInputPanel.classList.toggle('hidden', isVideo);
      videoInputPanel.classList.toggle('hidden', !isVideo);
      if (!isVideo) {
        previewVideo.pause();
        previewVideo.classList.add('hidden');
      }
      setCameraButton(cameraOn);
    }

    function setVideoPreview(file) {
      if (videoObjectUrl) {
        URL.revokeObjectURL(videoObjectUrl);
        videoObjectUrl = null;
      }
      if (!file) {
        previewVideo.removeAttribute('src');
        videoFileNameEl.textContent = 'No video selected';
        return;
      }
      videoObjectUrl = URL.createObjectURL(file);
      previewVideo.src = videoObjectUrl;
      previewVideo.loop = loopVideoInput.checked;
      previewVideo.muted = true;
      videoFileNameEl.textContent = file.name;
    }

    function setInputControlsVisible(visible) {
      changeInputButton.classList.toggle('hidden', !visible);
      controlBar.classList.toggle('hidden', !visible);
    }

    function showInputMenu() {
      inputOverlay.classList.remove('compact');
      setInputControlsVisible(false);
      document.getElementById('emptyState').classList.add('hidden');
    }

    function showActiveInput() {
      inputOverlay.classList.add('compact');
      setInputControlsVisible(true);
    }

    function showPreview() {
      if (currentSource() === 'video') {
        streamImage.removeAttribute('src');
        activeStreamKind = null;
        streamImage.classList.add('hidden');
        if (previewVideo.getAttribute('src')) {
          previewVideo.loop = loopVideoInput.checked;
          previewVideo.classList.remove('hidden');
          previewVideo.play().catch(() => {});
        }
        return;
      }
      streamImage.classList.add('hidden');
      previewVideo.classList.add('hidden');
      streamImage.classList.remove('hidden');
      if (activeStreamKind !== 'preview' || !streamImage.getAttribute('src')) {
        streamImage.src = `/preview_stream?t=${Date.now()}`;
        activeStreamKind = 'preview';
      }
    }

    function showDetectionStream() {
      previewVideo.pause();
      previewVideo.classList.add('hidden');
      streamImage.classList.remove('hidden');
      if (activeStreamKind === 'detect' && streamImage.getAttribute('src')) return;
      streamImage.src = `/stream?t=${Date.now()}`;
      activeStreamKind = 'detect';
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
      const noun = currentSource() === 'video' ? 'video' : 'camera';
      cameraButton.title = on ? `Turn ${noun} off` : `Turn ${noun} on`;
      document.getElementById('emptyState').classList.add('hidden');
    }

    async function stopStream() {
      streamImage.removeAttribute('src');
      activeStreamKind = null;
      streamImage.classList.add('hidden');
      previewVideo.pause();
      previewVideo.classList.add('hidden');
      try { await fetch('/api/camera/release', { method: 'POST' }); } catch (e) {}
    }

    async function ensureDetectionStream() {
      if (isDetecting) {
        showActiveInput();
        setCameraButton(true);
        showDetectionStream();
        return true;
      }
      const payload = await applySettings(true);
      if (!payload || payload.status !== 'applied') return false;
      setDetectButton(true);
      showActiveInput();
      setCameraButton(true);
      showDetectionStream();
      return true;
    }

    cameraSourceButton.addEventListener('click', () => {
      inputDirty = true;
      setSourceMode('live');
    });
    videoSourceButton.addEventListener('click', () => {
      inputDirty = true;
      setSourceMode('video');
      showPreview();
    });
    document.getElementById('camera').addEventListener('change', () => {
      inputDirty = true;
      setSourceMode('live');
    });
    document.getElementById('mirrorLive').addEventListener('change', () => {
      inputDirty = true;
      setSourceMode('live');
    });
    videoFileInput.addEventListener('change', () => {
      inputDirty = true;
      setSourceMode('video');
      setVideoPreview(videoFileInput.files?.[0]);
      showPreview();
    });
    loopVideoInput.addEventListener('change', () => {
      inputDirty = true;
      setSourceMode('video');
      previewVideo.loop = loopVideoInput.checked;
    });
    changeInputButton.addEventListener('click', async () => {
      setDetectButton(false);
      setCameraButton(false);
      await applySettings(false).catch(() => null);
      await stopStream();
      showInputMenu();
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
      document.getElementById('stoppedOverlay').classList.remove('hidden');
      try { fetch('/api/shutdown', { method: 'POST', keepalive: true }); } catch (e) {}
    });
    document.getElementById('enterInputButton').addEventListener('click', async () => {
      await ensureDetectionStream();
    });
    document.getElementById('enterCalibrationButton').addEventListener('click', async () => {
      await beginCalibration();
    });
    cameraButton.addEventListener('click', async () => {
      if (cameraOn) {
        setDetectButton(false);
        setCameraButton(false);
        await stopStream();
      } else {
        const payload = await applySettings(false);
        if (!payload || payload.status !== 'applied') return;
        showActiveInput();
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
      showActiveInput();
      if (nextState) {
        showDetectionStream();
      } else {
        streamImage.removeAttribute('src');
        activeStreamKind = null;
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
    async function startCalibration() {
      try {
        const res = await fetch('/api/calibration/start', { method: 'POST' });
        resetMetricBarRanges();
        update(await res.json());
      } catch (error) {
        console.warn('Calibration start failed', error);
      }
    }

    async function beginCalibration() {
      if (!calibrationNameInput.value.trim()) calibrationNameInput.value = defaultCalibrationName();
      normalizeProfileSelect.value = 'none';
      if (!await ensureDetectionStream()) return;
      await startCalibration();
    }

    async function stopCalibration() {
      const name = calibrationNameInput.value.trim() || defaultCalibrationName();
      calibrationNameInput.value = name;
      const data = new FormData();
      data.set('name', name);
      try {
        calibrationStartButton.disabled = true;
        const res = await fetch('/api/calibration/stop', { method: 'POST', body: data });
        const payload = await res.json();
        if (payload.status === 'error') alert(payload.error || 'Calibration save failed');
        if (payload.status === 'saved') resetMetricBarRanges();
        update(payload);
      } catch (error) {
        console.warn('Calibration save failed', error);
      } finally {
        calibrationStartButton.disabled = false;
      }
    }

    async function toggleCalibration() {
      if (lastPayload?.calibration?.active) {
        await stopCalibration();
      } else {
        await beginCalibration();
      }
    }

    calibrationStartButton.addEventListener('click', toggleCalibration);

    calibrationExportButton.addEventListener('click', () => {
      const selected = normalizeProfileSelect.value || 'none';
      if (selected.startsWith('profile:')) {
        const name = selected.slice('profile:'.length);
        window.location.href = `/api/calibration/export?name=${encodeURIComponent(name)}`;
      } else {
        window.location.href = '/api/calibration/export';
      }
    });

    calibrationLoadButton.addEventListener('click', () => {
      calibrationPresetFileInput.click();
    });

    calibrationPresetFileInput.addEventListener('change', async () => {
      const file = calibrationPresetFileInput.files?.[0];
      if (!file) return;
      const data = new FormData();
      data.set('file', file);
      try {
        calibrationLoadButton.disabled = true;
        calibrationStatusEl.textContent = 'loading';
        const res = await fetch('/api/calibration/import', { method: 'POST', body: data });
        const payload = await res.json();
        if (payload.status === 'error') {
          alert(payload.error || 'Calibration load failed');
          return;
        }
        update(payload);
      } catch (error) {
        console.warn('Calibration load failed', error);
      } finally {
        calibrationPresetFileInput.value = '';
        calibrationLoadButton.disabled = false;
      }
    });

    async function applyNormalizeProfile() {
      const selected = normalizeProfileSelect.value || 'none';
      if (!selected.startsWith('profile:')) {
        resetMetricBarRanges();
        await applyOscSettings();
        return;
      }
      const name = selected.slice('profile:'.length);
      const data = new FormData();
      data.set('name', name);
      try {
        const res = await fetch('/api/calibration/apply', { method: 'POST', body: data });
        const payload = await res.json();
        if (payload.status === 'error') alert(payload.error || 'Calibration apply failed');
        if (payload.status === 'applied') resetMetricBarRanges();
        update(payload);
      } catch (error) {
        console.warn('Calibration apply failed', error);
      }
    }

    normalizeProfileSelect.addEventListener('change', applyNormalizeProfile);

    function normalizePrefix(prefix) {
      let value = (prefix || '').trim().replace(/\\/+$/, '');
      if (!value) return '';
      if (!value.startsWith('/')) value = `/${value}`;
      return value;
    }

    function metricAddress(prefix, name) {
      const leaf = oscAddressNames[name] || name;
      return prefix ? `${prefix}/${leaf}` : `/${leaf}`;
    }

    function updateAddresses(payload = lastPayload) {
      const metrics = payload?.processing?.latest_metrics || {};
      const active = new Set(payload?.source?.osc_metrics || metricNames);
      const prefix = normalizePrefix(document.getElementById('oscNamespace').value);
      const container = document.getElementById('addresses');
      container.innerHTML = '';
      for (const name of orderedMetricNames(active)) {
        if (!active.has(name)) continue;
        const row = document.createElement('div');
        const value = Number(metrics[name] ?? 0);
        row.textContent = `${metricAddress(prefix, name)}  ${formatMetric(value)}`;
        container.appendChild(row);
      }
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
      const normalizedMode = mode === 'fixed' || mode === 'normalize';

      if (centeredMetrics.has(name)) {
        bar.classList.add('centered');
        const delta = Math.max(-1, Math.min(1, value));
        const width = Math.abs(delta) * 50;
        fill.style.left = delta < 0 ? `${50 - width}%` : '50%';
        fill.style.width = `${width}%`;
        return;
      }

      bar.classList.remove('centered');
      if (normalizedMode) {
        maxSeen[name] = 1;
        fill.style.left = '0%';
        fill.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
        return;
      }

      maxSeen[name] = Math.max(maxSeen[name] * 0.995, Math.abs(value), 1);
      const width = Math.max(0, Math.min(100, Math.abs(value) / maxSeen[name] * 100));
      fill.style.left = '0%';
      fill.style.width = `${width}%`;
    }

    function resetMetricBarRanges() {
      for (const name of metricNames) {
        maxSeen[name] = 1;
        const fill = document.getElementById(`b-${name}`);
        if (fill) {
          fill.style.left = '0%';
          fill.style.width = '0%';
        }
      }
    }

    function syncInitialMetricSmoothness(osc) {
      if (metricEmaInitialized) return;
      const metricAlphas = osc?.metric_alphas || {};
      for (const name of metricNames) {
        const row = document.getElementById(`metric-${name}`);
        const smooth = row?.querySelector('.metric-smooth');
        const smoothValue = row?.querySelector('.ms-val');
        if (!smooth || !smoothValue) continue;
        const frames = alphaToEmaFrames(metricAlphas[name]);
        smooth.value = frames;
        smoothValue.textContent = formatEmaFrames(frames);
        setRangeFill(smooth, frames);
      }
      metricEmaInitialized = true;
    }

    function syncPoseBackends(payload) {
      const select = document.getElementById('poseBackend');
      const backends = payload?.pose_backends || [{ id: 'mediapipe', label: 'MediaPipe', description: 'cross-platform' }];
      const signature = backends.map(b => `${b.id}:${b.label}:${b.description || ''}`).join('|');
      if (select.dataset.signature !== signature) {
        select.innerHTML = '';
        for (const backend of backends) {
          const option = document.createElement('option');
          option.value = backend.id;
          option.textContent = backend.description ? `${backend.label} (${backend.description})` : backend.label;
          select.appendChild(option);
        }
        select.dataset.signature = signature;
      }
      select.value = payload?.source?.pose_backend || 'mediapipe';
    }

    function defaultCalibrationName() {
      const d = new Date();
      const pad = value => String(value).padStart(2, '0');
      return `${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}`;
    }

    function syncCalibration(payload) {
      const calibration = payload?.calibration || {};
      const presets = Array.isArray(calibration.presets) ? calibration.presets : [];
      const signature = presets.join('|');
      const previous = normalizeProfileSelect.value || 'none';
      if (normalizeProfileSelect.dataset.signature !== signature) {
        normalizeProfileSelect.innerHTML = '';
        const noneOption = document.createElement('option');
        noneOption.value = 'none';
        noneOption.textContent = 'None - raw';
        normalizeProfileSelect.appendChild(noneOption);
        for (const name of presets) {
          const option = document.createElement('option');
          option.value = `profile:${name}`;
          option.textContent = name;
          normalizeProfileSelect.appendChild(option);
        }
        normalizeProfileSelect.dataset.signature = signature;
      }

      const applied = calibration.applied_preset || '';
      const oscMode = payload?.osc?.mode || 'raw';
      let desired = 'none';
      if (oscMode === 'raw') desired = 'none';
      if (oscMode === 'fixed' && applied) desired = `profile:${applied}`;
      if (previous && previous !== desired && document.activeElement === normalizeProfileSelect) desired = previous;
      if (Array.from(normalizeProfileSelect.options).some(option => option.value === desired)) {
        normalizeProfileSelect.value = desired;
      }

      const sampleCount = Number(calibration.sample_count || 0);
      const countdown = Math.max(0, Math.ceil(Number(calibration.countdown_remaining || 0)));
      const poseValid = Boolean(payload?.processing?.pose_valid);
      if (calibration.active && countdown > 0) {
        calibrationStatusEl.textContent = `starting ${countdown}`;
        calibrationOverlayEl.classList.add('countdown');
        calibrationOverlayEl.innerHTML = `<span class="countdown-label">Calibration starts</span><span class="countdown-number">${countdown}</span>`;
        calibrationOverlayEl.classList.remove('hidden');
      } else if (calibration.active) {
        calibrationStatusEl.textContent = poseValid || sampleCount > 0 ? `recording ${sampleCount}` : 'waiting clean pose';
        calibrationOverlayEl.classList.remove('countdown');
        calibrationOverlayEl.textContent = poseValid || sampleCount > 0 ? `Calibrating ${sampleCount}` : 'Waiting clean pose';
        calibrationOverlayEl.classList.remove('hidden');
      } else if (applied) {
        calibrationStatusEl.textContent = `fixed ${applied}`;
        calibrationOverlayEl.classList.remove('countdown');
        calibrationOverlayEl.classList.add('hidden');
      } else {
        calibrationStatusEl.textContent = sampleCount ? `${sampleCount} samples` : 'idle';
        calibrationOverlayEl.classList.remove('countdown');
        calibrationOverlayEl.classList.add('hidden');
      }
      calibrationStartButton.disabled = false;
      calibrationStartButton.textContent = calibration.active ? 'Stop' : 'Start';
      calibrationStartButton.classList.toggle('recording', Boolean(calibration.active));
    }

    function update(payload) {
      lastPayload = payload;
      const source = payload.source || {};
      const processing = payload.processing || {};
      const osc = payload.osc || {};
      const metrics = processing.latest_metrics || {};
      const active = new Set(source.osc_metrics || metricNames);
      const barScaleMode = (osc.mode === 'fixed' || osc.mode === 'normalize') ? 'normalized' : 'raw';
      if (barScaleMode !== lastMetricBarScaleMode) {
        resetMetricBarRanges();
        lastMetricBarScaleMode = barScaleMode;
      }
      const age = processing.last_frame_at ? (Date.now() / 1000) - processing.last_frame_at : Infinity;
      syncPoseBackends(payload);
      syncInitialMetricSmoothness(osc);
      refreshEmaFrameLabels();
      syncCalibration(payload);
      applyMetricOrder(active);
      if (!inputDirty) {
        setSourceMode(source.source === 'video' ? 'video' : 'live');
        document.getElementById('mirrorLive').checked = Boolean(source.mirror_live);
        loopVideoInput.checked = Boolean(source.loop ?? true);
        previewVideo.loop = loopVideoInput.checked;
        const cameraSelect = document.getElementById('camera');
        const cameraValue = String(source.camera_index ?? '0');
        if (Array.from(cameraSelect.options).some(option => option.value === cameraValue)) {
          cameraSelect.value = cameraValue;
        }
      }
      if (!oscDirty && !oscControlsHaveFocus()) {
        document.getElementById('oscNamespace').value =
          (osc.namespace === undefined || osc.namespace === null) ? '/field' : osc.namespace;
        renderOscTargets(Array.isArray(osc.targets) ? osc.targets : [{
          id: 'default',
          name: 'Output 1',
          host: osc.host || '127.0.0.1',
          port: osc.port || 9000,
          enabled: true,
          broadcast: false,
        }]);
      }

      document.getElementById('dot').className = age < 2 ? 'dot live' : 'dot';
      const calibrationActive = Boolean(payload?.calibration?.active);
      const calibrationCountdown = Math.max(0, Math.ceil(Number(payload?.calibration?.countdown_remaining || 0)));
      document.getElementById('status').textContent =
        processing.error || (calibrationActive ? (calibrationCountdown > 0 ? `calibrating in ${calibrationCountdown}` : 'calibrating') : (source.detect_enabled ? (age < 2 ? 'detecting' : 'waiting') : 'detect off'));
      document.getElementById('metaA').textContent = `source: ${source.source || '-'}`;
      document.getElementById('metaB').textContent =
        `fps: ${Number(processing.fps || 0).toFixed(1)} / pose: ${Number(processing.analysis_fps || 0).toFixed(1)}`;
      if (source.source === 'video') {
        document.getElementById('metaC').textContent = `time: ${Number(processing.elapsed_seconds || 0).toFixed(1)}s`;
        document.getElementById('metaD').textContent = `file: ${source.video_name || '-'} / loop ${source.loop ? 'on' : 'off'}`;
      } else {
        const cameraSelect = document.getElementById('camera');
        const cameraValue = String(source.camera_index ?? '');
        const cameraOption = Array.from(cameraSelect.options).find(option => option.value === cameraValue);
        const cameraLabel = cameraOption?.textContent || (cameraValue ? `Camera ${cameraValue}` : '-');
        const targets = Array.isArray(osc.targets) ? osc.targets : [];
        const oscTargetText = `${targets.length} output${targets.length === 1 ? '' : 's'}`;
        document.getElementById('metaC').textContent = `camera: ${cameraLabel}`;
        document.getElementById('metaD').textContent =
          `pose ${Number(processing.pose_ms || 0).toFixed(0)}ms / jpeg ${Number(processing.encode_ms || 0).toFixed(0)}ms / osc: ${oscTargetText}`;
      }
      updateAddresses(payload);

      for (const name of orderedMetricNames(active)) {
        const enabled = active.has(name);
        const row = document.getElementById(`metric-${name}`);
        const toggle = row?.querySelector('.metric-toggle');
        const smooth = row?.querySelector('.metric-smooth');
        const valueEl = document.getElementById(`v-${name}`);
        if (row) row.classList.toggle('disabled', !enabled);
        if (toggle) toggle.checked = enabled;
        if (smooth) smooth.disabled = !enabled;
        const ovRow = document.getElementById(`ov-row-${name}`);
        if (ovRow) ovRow.style.display = enabled ? '' : 'none';
        if (!enabled) {
          valueEl.textContent = 'off';
          valueEl.title = 'Metric disabled';
          const fill = document.getElementById(`b-${name}`);
          if (fill) {
            fill.style.left = '0%';
            fill.style.width = '0%';
          }
          continue;
        }
        const value = Number(metrics[name] ?? 0);
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
        const culturePrefix = normalizePrefix(document.getElementById('oscNamespace').value);
        const cultureAddress = culturePrefix ? `${culturePrefix}/morrisness` : '/morrisness';
        document.getElementById('cultureValue').textContent =
          `${cultureAddress}  ${Number(morrisness).toFixed(2)}`;
      }
    }

    function markOscDirty() {
      oscDirty = true;
    }

    function scheduleOscApply(delay = 300) {
      markOscDirty();
      window.clearTimeout(oscApplyTimer);
      oscApplyTimer = window.setTimeout(() => {
        oscApplyTimer = null;
        applyOscSettings().catch(error => console.warn('OSC config failed', error));
      }, delay);
    }

    oscTargetsEl.addEventListener('focusout', () => {
      window.setTimeout(() => {
        if (!oscControlsHaveFocus() && !oscApplyTimer && !oscApplyInFlight && !oscDirty) {
          oscDirty = false;
          if (lastPayload) update(lastPayload);
        }
      }, 0);
    });
    document.getElementById('addOscTarget').addEventListener('click', () => {
      addOscTargetRow(defaultOscTarget(oscTargetsEl.children.length));
      scheduleOscApply(0);
    });
    document.getElementById('oscNamespace').addEventListener('input', () => {
      scheduleOscApply();
      updateAddresses();
    });
    renderOscTargets([{ id: 'default', name: 'Output 1', host: '127.0.0.1', port: 9000, enabled: true, broadcast: false }]);
    loadCameras();

    const ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onmessage = event => update(JSON.parse(event.data));
  </script>
</body>
</html>
""".replace("%METRICS%", json.dumps(list(METRIC_NAMES))).replace(
    "%OSC_ADDRESS_NAMES%", json.dumps(OSC_ADDRESS_NAMES)
).replace(
    "%DEFAULT_METRIC_EMA_FRAMES%", str(int(DEFAULT_METRIC_EMA_FRAMES))
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
  .legend b { color:var(--teal); }
</style>
</head>
<body>
<header>
  <h1>FIELD - live metric charts</h1>
  <span class="status"><span id="dot" class="dot"></span><span id="statusText">waiting...</span></span>
  <a href="/">back to viewer</a>
</header>
<div class="grid" id="grid"></div>
<div class="legend">Per chart: <b>value sent to OSC after Smoothness(EMA) and normalize profile</b>. Window approx last 30s.</div>
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
  buffers[m.key] = [];
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
    const all = b.filter(v => v != null);
    g.strokeStyle = 'rgba(255,255,255,0.05)'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(0, h / 2); g.lineTo(w, h / 2); g.stroke();
    if (all.length === 0) continue;
    let lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
    if (lo === hi) { lo -= 1; hi += 1; }
    series(g, w, h, b, lo, hi, '#34d3c0', 1.8 * dpr);
    const last = b.filter(v => v != null).slice(-1)[0];
    if (last != null) valEls[m.key].textContent = last.toFixed(3);
  }
}
let running = false;
async function poll() {
  try {
    const res = await fetch('/api/metrics', { cache: 'no-store' });
    const d = await res.json();
    running = !!d.running;
    const values = d.smoothed || {};
    for (const m of METRICS) pushVal(buffers[m.key], values[m.key]);
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
    parser.add_argument("--osc-mode", choices=["raw", "normalize", "fixed"], default=os.getenv("FIELD_OSC_MODE", "raw"))
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
