import argparse
import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
import uvicorn


METRIC_NAMES = [
    "energy",
    "sync_velocity",
    "sync_correlation",
    "expansion",
    "curvature",
    "height",
    "sway",
    "torque",
    "jerk",
]

app = FastAPI()
state = {
    "metrics": {},
    "heartbeat": None,
    "last_address": None,
    "received_count": 0,
    "last_received_at": None,
}
clients: Set[WebSocket] = set()
message_queue = None


def handle_osc(address, *args):
    global message_queue
    value = args[0] if args else None
    now = time.time()
    name = address.rstrip("/").split("/")[-1]

    state["last_address"] = address
    state["received_count"] += 1
    state["last_received_at"] = now

    if name == "heartbeat":
        state["heartbeat"] = value
    else:
        state["metrics"][name] = value

    if message_queue is not None:
        asyncio.run_coroutine_threadsafe(message_queue.put(dict(state)), app.state.loop)


def start_osc_server(host: str, port: int):
    dispatcher = Dispatcher()
    dispatcher.map("/field/*", handle_osc)
    server = ThreadingOSCUDPServer((host, port), dispatcher)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


@app.on_event("startup")
async def startup():
    global message_queue
    app.state.loop = asyncio.get_running_loop()
    message_queue = asyncio.Queue()


@app.get("/")
async def index():
    return HTMLResponse(VIEWER_HTML)


@app.get("/api/state")
async def api_state():
    return state


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        await websocket.send_text(json.dumps(state))
        while True:
            payload = await message_queue.get()
            disconnected = []
            for client in clients:
                try:
                    await client.send_text(json.dumps(payload))
                except Exception:
                    disconnected.append(client)
            for client in disconnected:
                clients.discard(client)
    except WebSocketDisconnect:
        clients.discard(websocket)


VIEWER_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FIELD OSC Viewer</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    body { margin: 0; background: #0b1020; color: #e5e7eb; }
    main { max-width: 1180px; margin: 0 auto; padding: 24px; }
    header { display: flex; justify-content: space-between; gap: 16px; align-items: center; margin-bottom: 20px; }
    h1 { margin: 0; font-size: 24px; letter-spacing: 0; }
    .status { display: flex; gap: 10px; align-items: center; color: #94a3b8; font: 13px ui-monospace, monospace; }
    .dot { width: 10px; height: 10px; border-radius: 999px; background: #ef4444; }
    .dot.live { background: #22c55e; box-shadow: 0 0 0 5px rgba(34,197,94,.12); }
    .toolbar { display: grid; grid-template-columns: 1fr 110px 150px 110px; gap: 10px; margin-bottom: 18px; }
    input, select, button { border: 1px solid #334155; background: #111827; color: #e5e7eb; border-radius: 8px; padding: 10px 12px; font-size: 14px; }
    button { cursor: pointer; background: #2563eb; border-color: #3b82f6; font-weight: 700; }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
    .card { border: 1px solid #1f2937; background: #111827; border-radius: 8px; padding: 14px; min-height: 92px; }
    .name { color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
    .value { margin-top: 8px; font: 26px ui-monospace, SFMono-Regular, Menlo, monospace; }
    .bar { height: 8px; background: #1f2937; border-radius: 999px; overflow: hidden; margin-top: 12px; }
    .fill { height: 100%; width: 0%; background: linear-gradient(90deg, #22c55e, #3b82f6); transition: width .08s linear; }
    .meta { margin-top: 18px; color: #94a3b8; font: 13px ui-monospace, monospace; display: grid; gap: 6px; }
    @media (max-width: 760px) { .grid, .toolbar { grid-template-columns: 1fr; } header { align-items: flex-start; flex-direction: column; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>FIELD OSC Viewer</h1>
        <div class="status"><span id="dot" class="dot"></span><span id="status">waiting for OSC</span></div>
      </div>
      <div class="status">listening on /field/*</div>
    </header>

    <section class="toolbar">
      <input id="backend" value="http://127.0.0.1:8000" aria-label="Backend URL" />
      <select id="mode" aria-label="OSC mode">
        <option value="raw">raw</option>
        <option value="normalize">normalize</option>
      </select>
      <input id="alpha" type="number" min="0.01" max="1" step="0.01" value="1" aria-label="Smoothing alpha" />
      <button id="apply">Apply</button>
    </section>

    <section id="grid" class="grid"></section>
    <section class="meta">
      <div>last address: <span id="lastAddress">-</span></div>
      <div>heartbeat: <span id="heartbeat">-</span></div>
      <div>received: <span id="count">0</span></div>
    </section>
  </main>
  <script>
    const metrics = %METRICS%;
    const grid = document.getElementById('grid');
    const cards = {};
    const maxSeen = {};

    for (const name of metrics) {
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `<div class="name">${name}</div><div class="value" id="v-${name}">0.000</div><div class="bar"><div class="fill" id="b-${name}"></div></div>`;
      grid.appendChild(card);
      cards[name] = card;
      maxSeen[name] = 1;
    }

    function update(payload) {
      const now = Date.now() / 1000;
      const age = payload.last_received_at ? now - payload.last_received_at : Infinity;
      document.getElementById('dot').className = age < 2 ? 'dot live' : 'dot';
      document.getElementById('status').textContent = age < 2 ? 'receiving OSC' : 'waiting for OSC';
      document.getElementById('lastAddress').textContent = payload.last_address || '-';
      document.getElementById('heartbeat').textContent = payload.heartbeat ?? '-';
      document.getElementById('count').textContent = payload.received_count || 0;

      for (const name of metrics) {
        const value = Number(payload.metrics?.[name] ?? 0);
        maxSeen[name] = Math.max(maxSeen[name] * 0.995, Math.abs(value), 1);
        document.getElementById(`v-${name}`).textContent = Number.isFinite(value) ? value.toFixed(3) : String(value);
        const width = Math.max(0, Math.min(100, Math.abs(value) / maxSeen[name] * 100));
        document.getElementById(`b-${name}`).style.width = `${width}%`;
      }
    }

    const ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onmessage = event => update(JSON.parse(event.data));

    document.getElementById('apply').addEventListener('click', async () => {
      const backend = document.getElementById('backend').value.replace(/\\/$/, '');
      const mode = document.getElementById('mode').value;
      const alpha = Number(document.getElementById('alpha').value);
      await fetch(`${backend}/api/osc/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode, alpha, enabled: true })
      });
    });
  </script>
</body>
</html>
""".replace("%METRICS%", json.dumps(METRIC_NAMES))


def main():
    parser = argparse.ArgumentParser(description="Local web viewer for FIELD OSC output.")
    parser.add_argument("--osc-host", default="127.0.0.1")
    parser.add_argument("--osc-port", type=int, default=9000)
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=9100)
    args = parser.parse_args()

    start_osc_server(args.osc_host, args.osc_port)
    print(f"OSC viewer listening on udp://{args.osc_host}:{args.osc_port}")
    print(f"Open http://{args.web_host}:{args.web_port}")
    uvicorn.run(app, host=args.web_host, port=args.web_port)


if __name__ == "__main__":
    main()
