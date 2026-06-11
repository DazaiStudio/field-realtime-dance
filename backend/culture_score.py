"""Live culture score: how Morris-like vs BaYe-like the current movement is.

Loads centroids exported by the offline analysis (ai-motion
analysis/field/export_culture_map.py) and scores a rolling average of the
9 raw metrics:

    morrisness = d_baye / (d_morris + d_baye)   in 0..1
    1.0 = at the Morris centroid, 0.0 = at the BaYe centroid

The offline centroids are built from per-clip metric means, so the live
input is EMA-smoothed (~seconds) before scoring to stay comparable.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Optional

EMA_ALPHA = 0.05  # per-analysis-frame smoothing; ~2s window at 12 fps


class CultureScore:
    def __init__(self, map_path: Path) -> None:
        data = json.loads(Path(map_path).read_text(encoding="utf-8"))
        self.features = list(data["features"])
        self.log_features = set(data["log_features"])
        self.mean = [float(v) for v in data["mean"]]
        self.std = [float(v) if v else 1.0 for v in data["std"]]
        self.centroid_morris = [float(v) for v in data["centroid_morris"]]
        self.centroid_baye = [float(v) for v in data["centroid_baye"]]
        self._ema: dict = {}

    @classmethod
    def try_load(cls, map_path: Path) -> Optional["CultureScore"]:
        try:
            return cls(map_path)
        except FileNotFoundError:
            return None
        except Exception as exc:
            print(f"culture_map load failed ({map_path}): {exc}")
            return None

    def reset(self) -> None:
        self._ema.clear()

    def update(self, metrics: Mapping[str, float]) -> Optional[float]:
        """Feed one analysis frame of raw metrics; returns morrisness or None."""
        for name in self.features:
            value = metrics.get(name)
            if value is None or not math.isfinite(float(value)):
                return None if not self._ema else self._score()
            value = float(value)
            previous = self._ema.get(name)
            self._ema[name] = value if previous is None else (
                EMA_ALPHA * value + (1.0 - EMA_ALPHA) * previous
            )
        return self._score()

    def _score(self) -> Optional[float]:
        if len(self._ema) < len(self.features):
            return None
        d_morris = 0.0
        d_baye = 0.0
        for i, name in enumerate(self.features):
            value = self._ema[name]
            if name in self.log_features:
                value = math.log10(1.0 + max(value, 0.0))
            z = (value - self.mean[i]) / self.std[i]
            d_morris += (z - self.centroid_morris[i]) ** 2
            d_baye += (z - self.centroid_baye[i]) ** 2
        d_morris = math.sqrt(d_morris)
        d_baye = math.sqrt(d_baye)
        total = d_morris + d_baye
        if total <= 0:
            return 0.5
        return d_baye / total
