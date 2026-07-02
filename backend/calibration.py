"""Calibration ("sound-check"): collect per-metric output-EMA samples during a
short guided routine, then derive fixed [lo, hi] ranges (percentiles) for the
OSC sender's "fixed" normalize mode.

Why percentiles instead of min/max: a single bad detection frame can spike a
metric, and min/max would bake that outlier into the range. The 1st/99th
percentiles give a robust working range fitted to the actual dancer + camera
and the Smoothness settings used during calibration.

Only the 7 unbounded metrics need a calibrated range. sync_velocity and
sync_correlation are already bounded (0..1 / -1..1), so they are excluded.
"""
import json
import os

import numpy as np

RANGE_METRICS = ("energy", "torque", "jerk", "expansion", "curvature", "height", "sway")


def normalize_presets(data, default_name="imported") -> dict:
    """Return {name: {metric: (lo, hi)}} from supported preset JSON shapes."""
    if not isinstance(data, dict):
        return {}

    if isinstance(data.get("presets"), dict):
        source = data["presets"]
    elif isinstance(data.get("ranges"), dict):
        source = {str(default_name or "imported"): data["ranges"]}
    else:
        source = data

    out = {}
    for name, ranges in source.items():
        clean_name = str(name or "").strip()
        if not clean_name or not isinstance(ranges, dict):
            continue
        clean_ranges = {}
        for metric, value in ranges.items():
            if metric not in RANGE_METRICS:
                continue
            try:
                lo = float(value[0])
                hi = float(value[1])
            except (TypeError, ValueError, IndexError, KeyError):
                continue
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                clean_ranges[metric] = (lo, hi)
        if clean_ranges:
            out[clean_name] = clean_ranges
    return out


class CalibrationCollector:
    def __init__(self, metrics=RANGE_METRICS):
        self.metrics = tuple(metrics)
        self._samples = {m: [] for m in self.metrics}

    def reset(self) -> None:
        for m in self.metrics:
            self._samples[m].clear()

    def add(self, metrics: dict) -> None:
        """Accumulate one analysis frame's (raw) metrics."""
        if not metrics:
            return
        for m in self.metrics:
            v = metrics.get(m)
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                self._samples[m].append(v)

    def count(self, metric: str = None) -> int:
        if metric is not None:
            return len(self._samples.get(metric, []))
        return min((len(s) for s in self._samples.values()), default=0)

    def ranges(self, lo_pct: float = 1.0, hi_pct: float = 99.0,
               min_samples: int = 10) -> dict:
        """Per-metric (lo, hi) from the lo/hi percentiles. Metrics with fewer
        than min_samples are skipped (not enough data to trust)."""
        out = {}
        for m, s in self._samples.items():
            if len(s) < min_samples:
                continue
            arr = np.asarray(s, dtype=float)
            lo = float(np.percentile(arr, lo_pct))
            hi = float(np.percentile(arr, hi_pct))
            if hi <= lo:
                hi = lo + 1e-6
            out[m] = (lo, hi)
        return out


def save_profile(path, ranges: dict) -> None:
    data = {"ranges": {k: [float(v[0]), float(v[1])] for k, v in ranges.items()}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_profile(path) -> dict:
    """Return {metric: (lo, hi)} from a saved profile, or {} if missing/invalid."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        presets = normalize_presets(data, default_name="profile")
        return next(iter(presets.values()), {})
    except Exception:
        return {}


# --- Named preset library (multiple saved calibrations) --------------------

def load_presets(path) -> dict:
    """Return {name: {metric: (lo, hi)}} from the presets file, or {}."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return normalize_presets(data)
    except Exception:
        return {}


def save_presets(path, presets: dict) -> None:
    data = {name: {k: [float(v[0]), float(v[1])] for k, v in rng.items()}
            for name, rng in presets.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
