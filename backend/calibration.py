"""Calibration ("sound-check"): collect per-metric samples during a short
guided routine, then derive fixed [lo, hi] ranges (percentiles) for the OSC
sender's "fixed" normalize mode.

Why percentiles instead of raw min/max: a single bad detection frame can spike
a metric, and raw min/max would bake that outlier into the range. The 2nd/98th
percentiles give a robust working range fitted to the actual dancer + camera.

Only the 7 unbounded metrics need a calibrated range. sync_velocity and
sync_correlation are already bounded (0..1 / -1..1), so they are excluded.
"""
import json
import os

import numpy as np

RANGE_METRICS = ("energy", "torque", "jerk", "expansion", "curvature", "height", "sway")


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

    def ranges(self, lo_pct: float = 2.0, hi_pct: float = 98.0,
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
        return {k: (float(v[0]), float(v[1]))
                for k, v in data.get("ranges", {}).items()
                if v and float(v[1]) > float(v[0])}
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
        out = {}
        for name, rng in data.items():
            out[name] = {k: (float(v[0]), float(v[1]))
                         for k, v in rng.items()
                         if v and float(v[1]) > float(v[0])}
        return out
    except Exception:
        return {}


def save_presets(path, presets: dict) -> None:
    data = {name: {k: [float(v[0]), float(v[1])] for k, v in rng.items()}
            for name, rng in presets.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
