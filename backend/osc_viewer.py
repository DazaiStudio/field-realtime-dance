import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional, Set
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
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
osc_sender = OSCSender(
    host=os.getenv("FIELD_OSC_HOST", "127.0.0.1"),
    port=int(os.getenv("FIELD_OSC_PORT", "9000")),
    enabled=os.getenv("FIELD_OSC_ENABLED", "1") == "1",
    mode=os.getenv("FIELD_OSC_MODE", "raw"),
    alpha=float(os.getenv("FIELD_OSC_ALPHA", "1.0")),
)

source_state = {
    "source": "live",
    "camera_index": 0,
    "video_path": None,
    "video_name": None,
    "loop": True,
    "applied_at": time.time(),
}

processing_state = {
    "running": False,
    "frame_count": 0,
    "latest_metrics": {},
    "latest_timestamp_ms": None,
    "last_frame_at": None,
    "error": None,
}


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
        camera = cv2.VideoCapture(index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    return camera


def state_payload() -> dict:
    payload = {
        "source": dict(source_state),
        "processing": dict(processing_state),
        "osc": osc_sender.get_status(),
    }
    payload["source"]["video_path"] = None
    return payload


def set_latest(metrics: dict, timestamp_ms: int) -> None:
    processing_state["latest_metrics"] = metrics
    processing_state["latest_timestamp_ms"] = timestamp_ms
    processing_state["last_frame_at"] = time.time()
    processing_state["frame_count"] += 1
    processing_state["error"] = None
    osc_sender.send_metrics(metrics)
    osc_sender.send_heartbeat(timestamp_ms)


def encode_frame(frame):
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
    )


async def stream_live():
    processing_state["running"] = True
    cap = open_camera(int(source_state["camera_index"]))
    engine = get_pose_engine()

    while source_state["source"] == "live":
        ok, frame = cap.read()
        if not ok:
            processing_state["error"] = "Camera frame not available"
            await asyncio.sleep(0.2)
            continue

        timestamp_ms = int(time.time() * 1000)
        processed, metrics = engine.process_frame(frame, timestamp_ms)
        set_latest(metrics, timestamp_ms)
        encoded = encode_frame(processed)
        if encoded:
            yield encoded
        await asyncio.sleep(1 / 30)

    processing_state["running"] = False


async def stream_video():
    processing_state["running"] = True
    release_camera()
    engine = get_pose_engine()

    while source_state["source"] == "video":
        video_path = source_state.get("video_path")
        if not video_path:
            processing_state["error"] = "No video selected"
            await asyncio.sleep(0.5)
            continue

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = 1.0 / max(fps, 1.0)

        while source_state["source"] == "video":
            started = time.time()
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_ms = int(time.time() * 1000)
            processed, metrics = engine.process_frame(frame, timestamp_ms)
            set_latest(metrics, timestamp_ms)
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


@app.get("/api/state")
async def api_state():
    return state_payload()


@app.post("/api/apply")
async def apply_input(
    source: str = Form("live"),
    camera_index: int = Form(0),
    loop: bool = Form(True),
    osc_host: str = Form("127.0.0.1"),
    osc_port: int = Form(9000),
    osc_enabled: bool = Form(True),
    osc_mode: str = Form("raw"),
    osc_alpha: float = Form(1.0),
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
    )

    if source == "live":
        release_camera()
        source_state.update(
            {
                "source": "live",
                "camera_index": int(camera_index),
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
                "loop": bool(loop),
                "applied_at": time.time(),
            }
        )

    processing_state["frame_count"] = 0
    processing_state["latest_metrics"] = {}
    processing_state["latest_timestamp_ms"] = None
    processing_state["last_frame_at"] = None
    processing_state["error"] = None
    osc_sender.reset_state()
    return {"status": "applied", **state_payload()}


@app.get("/stream")
async def stream():
    if source_state["source"] == "video":
        generator = stream_video()
    else:
        generator = stream_live()
    return StreamingResponse(generator, media_type="multipart/x-mixed-replace; boundary=frame")


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
  <title>FIELD Input Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0a0f1e;
      color: #e5e7eb;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: #0a0f1e; color: #e5e7eb; }
    main { max-width: 1480px; margin: 0 auto; padding: 18px; }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 14px;
    }
    h1 { margin: 0; font-size: 24px; letter-spacing: 0; }
    .status { display: flex; align-items: center; gap: 8px; color: #94a3b8; font: 13px ui-monospace, monospace; }
    .dot { width: 10px; height: 10px; border-radius: 999px; background: #ef4444; }
    .dot.live { background: #22c55e; box-shadow: 0 0 0 5px rgba(34,197,94,.14); }
    .layout { display: grid; grid-template-columns: minmax(360px, 1fr) 430px; gap: 14px; align-items: start; }
    .panel {
      background: #111827;
      border: 1px solid #243044;
      border-radius: 8px;
      overflow: hidden;
    }
    .controls { padding: 14px; display: grid; gap: 12px; }
    .controls-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
    label { display: grid; gap: 6px; color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }
    input, select, button {
      width: 100%;
      border: 1px solid #334155;
      background: #0f172a;
      color: #e5e7eb;
      border-radius: 8px;
      padding: 10px 11px;
      font-size: 14px;
      min-height: 42px;
    }
    input[type="checkbox"] { width: 18px; min-height: 18px; accent-color: #22c55e; }
    button {
      cursor: pointer;
      background: #2563eb;
      border-color: #3b82f6;
      font-weight: 750;
      align-self: end;
    }
    .check-row { display: flex; align-items: center; gap: 8px; padding-top: 22px; color: #cbd5e1; font-size: 14px; }
    .video-wrap { position: relative; background: #020617; aspect-ratio: 16 / 9; }
    #stream { width: 100%; height: 100%; object-fit: contain; display: block; }
    .empty {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      color: #64748b;
      font: 14px ui-monospace, monospace;
      pointer-events: none;
    }
    .meta {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      border-top: 1px solid #243044;
      padding: 10px 14px;
      color: #94a3b8;
      font: 12px ui-monospace, monospace;
    }
    .metric-grid { display: grid; grid-template-columns: 1fr; gap: 9px; padding: 12px; }
    .metric {
      display: grid;
      grid-template-columns: 150px 92px 1fr;
      align-items: center;
      gap: 10px;
      min-height: 46px;
      padding: 9px;
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 8px;
    }
    .name { color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; overflow-wrap: anywhere; }
    .value { font: 18px ui-monospace, SFMono-Regular, Menlo, monospace; text-align: right; }
    .bar { height: 7px; background: #1f2937; border-radius: 999px; overflow: hidden; }
    .fill { height: 100%; width: 0%; background: linear-gradient(90deg, #22c55e, #38bdf8); transition: width .08s linear; }
    @media (max-width: 1080px) {
      .layout { grid-template-columns: 1fr; }
      .controls-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 680px) {
      header { align-items: flex-start; flex-direction: column; }
      .controls-grid, .meta { grid-template-columns: 1fr; }
      .metric { grid-template-columns: 1fr; }
      .value { text-align: left; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>FIELD Input Viewer</h1>
        <div class="status"><span id="dot" class="dot"></span><span id="status">idle</span></div>
      </div>
      <div class="status">pose overlay + metrics + OSC</div>
    </header>

    <section class="panel controls">
      <form id="form" class="controls-grid">
        <label>Input
          <select id="source" name="source">
            <option value="live">Live Cam</option>
            <option value="video">Video File</option>
          </select>
        </label>
        <label>Video
          <input id="video" name="video" type="file" accept="video/*" />
        </label>
        <label>Camera
          <input id="camera" name="camera_index" type="number" min="0" step="1" value="0" />
        </label>
        <label class="check-row"><input id="loop" name="loop" type="checkbox" checked /> Loop video</label>
        <label>OSC host
          <input id="oscHost" name="osc_host" value="127.0.0.1" />
        </label>
        <label>OSC port
          <input id="oscPort" name="osc_port" type="number" min="1" max="65535" value="9000" />
        </label>
        <label>Mode
          <select id="oscMode" name="osc_mode">
            <option value="raw">raw</option>
            <option value="normalize">normalize</option>
          </select>
        </label>
        <label>Alpha
          <input id="oscAlpha" name="osc_alpha" type="number" min="0.01" max="1" step="0.01" value="1" />
        </label>
        <label class="check-row"><input id="oscEnabled" name="osc_enabled" type="checkbox" checked /> OSC enabled</label>
        <button type="submit">Apply</button>
      </form>
    </section>

    <section class="layout" style="margin-top:14px">
      <section class="panel">
        <div class="video-wrap">
          <img id="stream" alt="Processed pose stream" />
          <div id="empty" class="empty">Choose input and press Apply</div>
        </div>
        <div class="meta">
          <div>source: <span id="mSource">-</span></div>
          <div>frames: <span id="mFrames">0</span></div>
          <div>osc: <span id="mOsc">-</span></div>
          <div>file: <span id="mFile">-</span></div>
        </div>
      </section>

      <aside class="panel">
        <div id="metrics" class="metric-grid"></div>
      </aside>
    </section>
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
        <div class="name">${name}</div>
        <div class="value" id="v-${name}">0.000</div>
        <div class="bar"><div class="fill" id="b-${name}"></div></div>
      `;
      metricsEl.appendChild(row);
    }

    document.getElementById('form').addEventListener('submit', async (event) => {
      event.preventDefault();
      const form = event.currentTarget;
      const data = new FormData(form);
      data.set('loop', document.getElementById('loop').checked ? 'true' : 'false');
      data.set('osc_enabled', document.getElementById('oscEnabled').checked ? 'true' : 'false');
      if (document.getElementById('source').value === 'live') {
        data.delete('video');
      }

      const res = await fetch('/api/apply', { method: 'POST', body: data });
      const payload = await res.json();
      if (payload.status !== 'applied') {
        alert(payload.error || 'Apply failed');
        return;
      }

      document.getElementById('empty').style.display = 'none';
      document.getElementById('stream').src = `/stream?t=${Date.now()}`;
    });

    function update(payload) {
      const source = payload.source || {};
      const processing = payload.processing || {};
      const osc = payload.osc || {};
      const metrics = processing.latest_metrics || {};
      const age = processing.last_frame_at ? (Date.now() / 1000) - processing.last_frame_at : Infinity;

      document.getElementById('dot').className = age < 2 ? 'dot live' : 'dot';
      document.getElementById('status').textContent = processing.error || (age < 2 ? 'processing' : 'waiting');
      document.getElementById('mSource').textContent = source.source || '-';
      document.getElementById('mFrames').textContent = processing.frame_count || 0;
      document.getElementById('mOsc').textContent = `${osc.enabled ? 'on' : 'off'} ${osc.host || '-'}:${osc.port || '-'}`;
      document.getElementById('mFile').textContent = source.video_name || '-';

      for (const name of metricNames) {
        const value = Number(metrics[name] ?? 0);
        maxSeen[name] = Math.max(maxSeen[name] * 0.995, Math.abs(value), 1);
        document.getElementById(`v-${name}`).textContent = Number.isFinite(value) ? value.toFixed(3) : String(value);
        const width = Math.max(0, Math.min(100, Math.abs(value) / maxSeen[name] * 100));
        document.getElementById(`b-${name}`).style.width = `${width}%`;
      }
    }

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
    args = parser.parse_args()

    osc_sender.configure(
        host=args.osc_host,
        port=args.osc_port,
        mode=args.osc_mode,
        alpha=args.osc_alpha,
    )
    print(f"FIELD input viewer: http://{args.web_host}:{args.web_port}")
    print(f"OSC output: udp://{args.osc_host}:{args.osc_port} mode={args.osc_mode} alpha={args.osc_alpha}")
    uvicorn.run(app, host=args.web_host, port=args.web_port)


if __name__ == "__main__":
    main()
