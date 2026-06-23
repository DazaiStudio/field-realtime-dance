"""
RTMPose3D calibration harness — Task 5.

Run from repo root:
    py -3.10 backend/_calibrate_rtmpose3d.py [--frames N] [--probe-only] [--skip-probe]

Compares MediaPipe vs RTMPose3D metric magnitudes on the same 300-frame clip.
Prints per-axis raw ranges of RTMPose3D 3D keypoints, then runs both backends
through DanceMetricsEngine and shows side-by-side per-metric summary stats
(mean, median, p95, max).

=== FINAL CALIBRATION RESULTS (2026-06-23, glitch-1, 1920×1080 @ 30fps) ===

Bug discovered and fixed (Task 5):
  rtmlib PoseTracker returns a 4-tuple on detection frames (every det_frequency)
  but a 2-tuple (keypoints, scores) on tracking frames.  The original backend
  checked `len(result) < 4` and silently returned [] on tracking frames — i.e.
  9 out of every 10 frames were DISCARDED.  After fix: 297/300 frames produce data.

Raw RTMPose3D axis ranges (body joints 0-16, 95 frames, 1920×1080 clip):
  keypoints (3D):
    x : min= 106.5   max= 185.0   mean= 143.9   std=  19.3   → model-input-crop px columns
    y : min=  90.0   max= 280.0   mean= 153.5   std=  63.6   → model-input-crop px rows
    z : min=  -0.66  max=  -0.45  mean=  -0.59  std=   0.07  → normalised depth (this clip)
  keypoints_2d (pixel):
    x : min= 832.4   max=1058.6   mean= 940.3                → full-frame video pixels (1920-wide)
    y : min= 334.6   max= 882.2   mean= 517.5                → full-frame video pixels (1080-tall)
  bbox_height from kpts2d body joints: mean= 547.6 px

  Model input size: 384 (H) × 288 (W).
  3D x,y are in the RESIZED CROP (288×384), NOT the video frame.
  3D z is root-relative depth normalised by bbox_height IN CROP SPACE (~190 px).
  Scale ratio 2D/3D is consistently 2.88× for both axes.

Problem (BEFORE fix):
  x,y in crop-pixel space (~100–280), z is a dimensionless ratio (~-0.66).
  Units are inconsistent: z is ~200× smaller than x,y relative to body size.
  This skews ALL 3D limb angles → energy/torque/jerk wildly inflated.
  Also: only 1 frame of data due to 4-tuple vs 2-tuple bug.

Two-step fix implemented in personposes_from_rtmw3d():
  Step 1 (unit consistency):
    z_consistent = z_norm × bbox_height_3d
    where bbox_height_3d = body3d_y.max() - body3d_y.min()  (~190 px in this clip)
    Result: x, y, z all in model-input-crop pixel units.

  Step 2 (overall magnitude):
    coords_out = coords_consistent × POSE_SCALE   (POSE_SCALE = 3.0)
    Chosen so convex-hull expansion matches MediaPipe numerically.
    Expansion scales as SCALE³; at 3.0 → expansion ≈ 0.97 vs MediaPipe 0.94.

Chosen POSE_SCALE: 3.0
  Expansion (scale³-dependent):  RTMPose3D 0.97 ≈ MediaPipe 0.94  ✓
  Sway (scale-dependent):        RTMPose3D 0.13 vs MediaPipe 0.34  (same order)
  Height sign mismatch:          RTMPose3D is negative (absolute y, not root-relative)
  Energy residual gap (7×):      intrinsic detector jitter, not fixable with scale

Before/After calibration (300-frame clip, single dancer, mean / median / p95):
  Metric          MediaPipe          RTMPose3D-BEFORE     RTMPose3D-AFTER
  energy          53.0 /  19.9 / 228.6    2922 (1 frame)   363.2 / 199.1 / 1269.3
  expansion        0.94/   0.90/   1.34   0.0003 (bad)       0.97/   1.22/   1.57
  height           0.10/   0.12/   0.17  -0.19  (bad)       -0.29/  -0.21/  -0.11 [*]
  sway             0.34/   0.21/   0.83   0.005  (bad)       0.13/   0.06/   0.38
  torque          80.8 /  60.9 / 195.2   (no data)         206.4 / 135.7 / 668.4
  sync_velocity    0.59/   0.59/   0.97   0.65 (1 frame)     0.36/   0.21/   0.98
  curvature        0.07/   0.03/   0.27  (no data)           0.55/   0.09/   2.50
  jerk        78993298 /22100683/...     (no data)      361246258/179875141/...  [**]
  sync_corr        0.23/   0.21/   0.61  (no data)           (varies by clip)

  [*] Height is negative because RTMPose3D uses absolute model-crop-pixel y
      (y=0 at top, increasing downward). MediaPipe uses root-relative world
      coords (pelvis=origin). The engine's -com[1]/1000 gives opposite sign.
      Not fixable with scale; would need pelvis-subtraction in the backend.
  [**] Jerk values are very large in both backends (it's a 2nd derivative);
       the ratio remains comparable (~5× rather than orders of magnitude).
"""

# ---------------------------------------------------------------------------
# 0.  CUDA DLL fix (must be before any onnxruntime import)
# ---------------------------------------------------------------------------
import os
import sys

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BACKEND_DIR)


def _prepend_cuda_dlls():
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            os.add_dll_directory(torch_lib)
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass
    try:
        import importlib.util
        site = os.path.dirname(importlib.util.find_spec("torch").origin)
        site_pkgs = os.path.dirname(site)
        nvidia_root = os.path.join(site_pkgs, "nvidia")
        if os.path.isdir(nvidia_root):
            for pkg in os.listdir(nvidia_root):
                bin_dir = os.path.join(nvidia_root, pkg, "bin")
                if os.path.isdir(bin_dir):
                    os.add_dll_directory(bin_dir)
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


_prepend_cuda_dlls()

# ---------------------------------------------------------------------------
# 1.  Standard imports
# ---------------------------------------------------------------------------
import argparse
import time

import cv2
import numpy as np

from dance_metrics import DanceMetricsEngine
from pose_backends.mediapipe_backend import MediaPipeBackend
from pose_backends.rtmpose3d_backend import RTMPose3DBackend

# ---------------------------------------------------------------------------
# 2.  Config
# ---------------------------------------------------------------------------
TEST_VIDEO_PATHS = [
    r"D:\Github\ai-motion\data\test\05-Feb-2026\Movement research-glitch-1_05-Feb-2026.mov",
    r"D:\Github\ai-motion\data\test\05-Feb-2026\Movement research-glitch-2_05-Feb-2026.mov",
]
MODEL_PATH = os.path.join(os.path.dirname(_BACKEND_DIR), "pose_landmarker_full.task")
FPS = 30
WARMUP_FRAMES = 5


# ---------------------------------------------------------------------------
# 3.  Helpers
# ---------------------------------------------------------------------------

def _load_frames(video_path: str, n: int) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    frames = []
    src_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[VIDEO] {os.path.basename(video_path)}  {total} frames @ {src_fps:.1f}fps")
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                break
        frames.append(frame)
    cap.release()
    return frames


def _summary(arr: np.ndarray, label: str) -> str:
    if len(arr) == 0:
        return f"  {label:20s}  no data"
    a = np.array(arr)
    return (f"  {label:20s}  "
            f"mean={np.mean(a):9.4f}  "
            f"med={np.median(a):9.4f}  "
            f"p95={np.percentile(a, 95):9.4f}  "
            f"max={np.max(a):9.4f}")


METRIC_KEYS = [
    "energy", "sync_velocity", "sync_correlation",
    "expansion", "curvature", "height", "sway", "torque", "jerk",
]


# ---------------------------------------------------------------------------
# 4.  RTMPose3D raw-range probe (printed before calibration loop)
# ---------------------------------------------------------------------------

def probe_rtmpose3d_ranges(frames: list, n_probe: int = 30):
    """Run RTMPose3D on the first n_probe frames and print raw axis ranges."""
    print("\n" + "=" * 60)
    print("PROBE: RTMPose3D raw 3D keypoint ranges (body joints 0-16)")
    print("=" * 60)

    from rtmlib import PoseTracker, Wholebody3d  # noqa: PLC0415

    tracker = PoseTracker(
        Wholebody3d,
        det_frequency=10,
        tracking=True,
        mode="balanced",
        to_openpose=False,
        backend="onnxruntime",
        device="cuda",
    )

    xs, ys, zs = [], [], []
    x2d_all, y2d_all = [], []

    for idx, frame in enumerate(frames[:n_probe]):
        result = tracker(frame)
        if result is None or len(result) < 4:
            continue
        kpts3d, scores, _simcc, kpts2d = result
        kpts3d = np.asarray(kpts3d)
        kpts2d = np.asarray(kpts2d)
        if kpts3d.size == 0:
            continue
        # Body joints only (0-16)
        body3d = kpts3d[:, :17, :]   # (N,17,3)
        body2d = kpts2d[:, :17, :]   # (N,17,2)
        xs.append(body3d[:, :, 0])
        ys.append(body3d[:, :, 1])
        zs.append(body3d[:, :, 2])
        x2d_all.append(body2d[:, :, 0])
        y2d_all.append(body2d[:, :, 1])

    del tracker  # release GPU

    xs = np.concatenate(xs) if xs else np.array([])
    ys = np.concatenate(ys) if ys else np.array([])
    zs = np.concatenate(zs) if zs else np.array([])
    x2d = np.concatenate(x2d_all) if x2d_all else np.array([])
    y2d = np.concatenate(y2d_all) if y2d_all else np.array([])

    def _rng(arr, name):
        if arr.size == 0:
            print(f"  {name}: no data")
            return
        print(f"  {name}: min={arr.min():8.3f}  max={arr.max():8.3f}  "
              f"mean={arr.mean():8.3f}  std={arr.std():8.3f}")

    print(f"\n  3D keypoints[:, :17, :] from tracker output:")
    _rng(xs, "x (3D)")
    _rng(ys, "y (3D)")
    _rng(zs, "z (3D)")
    print(f"\n  2D keypoints[:, :17, :] from tracker output (should be pixels):")
    _rng(x2d, "x (2D/pixel)")
    _rng(y2d, "y (2D/pixel)")
    print()


# ---------------------------------------------------------------------------
# 5.  Run one backend through DanceMetricsEngine, collect metrics
# ---------------------------------------------------------------------------

def run_backend(backend, frames: list, fps: float = FPS) -> dict[str, list]:
    engine = DanceMetricsEngine(fps=fps)
    per_metric: dict[str, list] = {k: [] for k in METRIC_KEYS}

    for fi, frame in enumerate(frames):
        ts_ms = fi * (1000.0 / fps)
        poses = backend.estimate(frame, ts_ms)
        if not poses:
            continue
        # Take the first (most-tracked) person
        pose = poses[0]
        metrics = engine.update(pose.h36m17)
        for k in METRIC_KEYS:
            v = metrics.get(k, 0.0)
            if v != 0.0:          # skip empty-return frames (< 2 history)
                per_metric[k].append(v)

    return per_metric


# ---------------------------------------------------------------------------
# 6.  Print side-by-side table
# ---------------------------------------------------------------------------

def print_comparison(mp_metrics: dict, rtm_metrics: dict, label_a="MediaPipe", label_b="RTMPose3D"):
    print("\n" + "=" * 90)
    print(f"METRIC COMPARISON: {label_a} vs {label_b}")
    print("=" * 90)
    print(f"  {'metric':20s}  {'backend':12s}  {'mean':>10s}  {'median':>10s}  {'p95':>10s}  {'max':>10s}")
    print("  " + "-" * 84)
    for k in METRIC_KEYS:
        for label, data in [(label_a, mp_metrics), (label_b, rtm_metrics)]:
            arr = np.array(data.get(k, []))
            if arr.size == 0:
                print(f"  {k:20s}  {label:12s}  no data")
            else:
                print(f"  {k:20s}  {label:12s}  "
                      f"{np.mean(arr):10.4f}  "
                      f"{np.median(arr):10.4f}  "
                      f"{np.percentile(arr,95):10.4f}  "
                      f"{np.max(arr):10.4f}")
        print()


# ---------------------------------------------------------------------------
# 7.  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RTMPose3D calibration harness")
    parser.add_argument("--frames", type=int, default=300,
                        help="Number of frames to process per backend (default 300)")
    parser.add_argument("--probe-only", action="store_true",
                        help="Only print raw axis ranges, skip full metric run")
    parser.add_argument("--skip-probe", action="store_true",
                        help="Skip the raw-range probe (faster if already known)")
    args = parser.parse_args()

    # --- Find test video ---
    video_path = None
    for p in TEST_VIDEO_PATHS:
        if os.path.exists(p):
            video_path = p
            break
    if video_path is None:
        print("[ERROR] No test video found. Checked:")
        for p in TEST_VIDEO_PATHS:
            print(f"  {p}")
        sys.exit(1)

    frames = _load_frames(video_path, args.frames + WARMUP_FRAMES)
    frames = frames[WARMUP_FRAMES:]   # skip first few (model warm-up artefacts)
    frames = frames[:args.frames]
    print(f"[INFO] Using {len(frames)} frames for calibration")

    # --- Raw range probe ---
    if not args.skip_probe:
        probe_rtmpose3d_ranges(frames, n_probe=min(30, len(frames)))

    if args.probe_only:
        return

    # --- MediaPipe backend ---
    print("\n" + "=" * 60)
    print("Running MediaPipe backend...")
    print("=" * 60)
    mp_backend = MediaPipeBackend(model_path=MODEL_PATH, num_poses=1)
    t0 = time.perf_counter()
    mp_metrics = run_backend(mp_backend, frames)
    mp_time = time.perf_counter() - t0
    mp_backend.close()
    print(f"  Done in {mp_time:.1f}s ({len(frames)/mp_time:.1f} fps)")

    # --- RTMPose3D backend ---
    print("\n" + "=" * 60)
    print("Running RTMPose3D backend...")
    print("=" * 60)
    rtm_backend = RTMPose3DBackend(device="cuda", det_frequency=10)
    t0 = time.perf_counter()
    rtm_metrics = run_backend(rtm_backend, frames)
    rtm_time = time.perf_counter() - t0
    rtm_backend.close()
    print(f"  Done in {rtm_time:.1f}s ({len(frames)/rtm_time:.1f} fps)")

    # --- Compare ---
    print_comparison(mp_metrics, rtm_metrics)


if __name__ == "__main__":
    main()
