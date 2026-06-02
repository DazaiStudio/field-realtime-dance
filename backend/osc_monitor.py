import argparse
import time

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer


def handle_message(address, *args):
    timestamp = time.strftime("%H:%M:%S")
    values = " ".join(format_value(value) for value in args)
    print(f"{timestamp}  {address:<32} {values}", flush=True)


def format_value(value):
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Terminal OSC monitor for FIELD Realtime Dance.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--prefix", default="/field")
    args = parser.parse_args()

    prefix = args.prefix.rstrip("/") or "/field"
    if not prefix.startswith("/"):
        prefix = f"/{prefix}"

    dispatcher = Dispatcher()
    dispatcher.map(f"{prefix}/*", handle_message)

    server = BlockingOSCUDPServer((args.host, args.port), dispatcher)
    print(f"OSC monitor listening on udp://{args.host}:{args.port}{prefix}/*")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
