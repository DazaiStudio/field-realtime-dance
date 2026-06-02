import argparse
import os
import sys
import time
from pathlib import Path

import cv2

from osc_sender import OSCSender
from pose_engine import PoseEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Process an input video and send FIELD metrics over OSC.")
    if any(arg == "--web-port" or arg.startswith("--web-port=") for arg in sys.argv[1:]):
        parser.error("--web-port is only for backend/osc_viewer.py. Use osc_viewer.py to open the local browser viewer.")
    parser.add_argument("video", help="Path to an input video file.")
    parser.add_argument("--host", "--osc-host", dest="host", default=os.getenv("FIELD_OSC_HOST", "127.0.0.1"))
    parser.add_argument("--port", "--osc-port", dest="port", type=int, default=int(os.getenv("FIELD_OSC_PORT", "9000")))
    parser.add_argument("--mode", "--osc-mode", dest="mode", choices=["raw", "normalize"], default=os.getenv("FIELD_OSC_MODE", "raw"))
    parser.add_argument("--alpha", "--osc-alpha", dest="alpha", type=float, default=float(os.getenv("FIELD_OSC_ALPHA", "0.25")))
    parser.add_argument("--web-port", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "pose_landmarker_full.task"
    pose_engine = PoseEngine(model_path=str(model_path))
    sender = OSCSender(args.host, args.port, mode=args.mode, alpha=args.alpha)

    print(f"Sending OSC to udp://{args.host}:{args.port} mode={args.mode} alpha={args.alpha}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval = 1.0 / fps
            frame_index = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                started = time.time()
                timestamp_ms = int(frame_index * frame_interval * 1000)
                processed, metrics = pose_engine.process_frame(frame, timestamp_ms)
                sender.send_metrics(metrics)

                if not args.no_preview:
                    cv2.imshow("FIELD OSC video test", processed)
                    if cv2.waitKey(1) & 0xFF == 27:
                        return 0

                elapsed = time.time() - started
                time.sleep(max(0.0, frame_interval - elapsed))
                frame_index += 1

            cap.release()
            if not args.loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        pose_engine.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
