import math
import time
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

from pythonosc.udp_client import SimpleUDPClient


METRIC_NAMES = (
    "energy",
    "sync_velocity",
    "sync_correlation",
    "expansion",
    "curvature",
    "height",
    "sway",
    "torque",
    "jerk",
)

OSC_ADDRESS_NAMES = {
    "sync_velocity": "sync_vel",
    "sync_correlation": "sync_corr",
}

BOUNDED_METRICS = {"sync_velocity"}
# height is CoM above the foot base in metres and sway is a small metre-scale
# offset; clamping them to 0..1 flattens both to ~0.
# Track an adaptive min/max range instead so normalize mode stays expressive.
ADAPTIVE_RANGE_METRICS = {"height", "sway"}
UNBOUNDED_METRICS = {"energy", "expansion", "curvature", "torque", "jerk"}

# Per-message decay applied to the adaptive range so stale extremes fade.
RANGE_DECAY = 0.001
DEFAULT_TARGET_ID = "default"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class OSCTarget:
    id: str
    name: str
    host: str
    port: int
    enabled: bool = True
    broadcast: bool = False
    client: SimpleUDPClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.id = str(self.id or DEFAULT_TARGET_ID).strip() or DEFAULT_TARGET_ID
        self.name = str(self.name or self.id).strip() or self.id
        self.host = str(self.host or "127.0.0.1").strip() or "127.0.0.1"
        self.port = int(self.port)
        if self.port < 1 or self.port > 65535:
            raise ValueError("OSC target port must be 1-65535")
        self.enabled = bool(self.enabled)
        self.broadcast = bool(self.broadcast)
        self.client = SimpleUDPClient(self.host, self.port, allow_broadcast=self.broadcast)

    def to_status(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "enabled": self.enabled,
            "broadcast": self.broadcast,
        }

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close is not None:
            close()


class OSCSender:
    """Send dance metrics over OSC with optional normalization and smoothing."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        namespace: str = "/field",
        enabled: bool = True,
        mode: str = "raw",
        alpha: float = 0.25,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.namespace = self._normalize_namespace(namespace)
        self.enabled = bool(enabled)
        self.mode = self._validate_mode(mode)
        self.alpha = self._validate_alpha(alpha)
        self.targets = [
            OSCTarget(
                id=DEFAULT_TARGET_ID,
                name="Output 1",
                host=self.host,
                port=self.port,
                enabled=True,
            )
        ]
        self._smoothed: Dict[str, float] = {}
        # Per-metric output smoothing override; falls back to self.alpha (the
        # global slider) for any metric without its own value. 1.0 = off.
        self.metric_alphas: Dict[str, float] = {}
        # Calibrated fixed [lo, hi] ranges per metric, used by mode "fixed"
        # (comparable across time + bounded + personalised). Empty = none yet.
        self.metric_ranges: Dict[str, tuple] = {}
        self._peaks: Dict[str, float] = {}
        self._ranges: Dict[str, tuple] = {}
        self.last_prepared_metrics: Dict[str, float] = {}
        self.last_sent_at: Optional[float] = None

    def configure(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        enabled: Optional[bool] = None,
        mode: Optional[str] = None,
        alpha: Optional[float] = None,
        namespace: Optional[str] = None,
    ) -> None:
        if host is not None or port is not None:
            primary = self.targets[0] if self.targets else OSCTarget(
                id=DEFAULT_TARGET_ID,
                name="Output 1",
                host=self.host,
                port=self.port,
                enabled=True,
            )
            self.targets = [
                OSCTarget(
                    id=primary.id,
                    name=primary.name,
                    host=host if host is not None else primary.host,
                    port=port if port is not None else primary.port,
                    enabled=primary.enabled,
                    broadcast=primary.broadcast,
                ),
                *self.targets[1:],
            ]
            primary.close()
            self._sync_primary_target()
        if namespace is not None:
            self.namespace = self._normalize_namespace(namespace)
        if enabled is not None:
            self.enabled = bool(enabled)
        if mode is not None:
            new_mode = self._validate_mode(mode)
            if new_mode != self.mode:
                self.mode = new_mode
                self.reset_state()
        if alpha is not None:
            self.alpha = self._validate_alpha(alpha)

    def configure_targets(self, targets: list[dict]) -> None:
        parsed = []
        used_ids = set()
        for index, target in enumerate(targets or []):
            target_id = str(target.get("id") or f"output-{index + 1}").strip()
            if not target_id:
                target_id = f"output-{index + 1}"
            base_id = target_id
            suffix = 2
            while target_id in used_ids:
                target_id = f"{base_id}-{suffix}"
                suffix += 1
            used_ids.add(target_id)
            parsed.append(OSCTarget(
                id=target_id,
                name=target.get("name") or f"Output {index + 1}",
                host=target.get("host") or "127.0.0.1",
                port=int(target.get("port") or 9000),
                enabled=target.get("enabled", True),
                broadcast=target.get("broadcast", False),
            ))

        old_targets = self.targets
        self.targets = parsed
        for target in old_targets:
            target.close()
        self._sync_primary_target()

    def _sync_primary_target(self) -> None:
        if not self.targets:
            return
        primary = self.targets[0]
        self.host = primary.host
        self.port = primary.port

    def reset_state(self) -> None:
        self._smoothed.clear()
        self._peaks.clear()
        self._ranges.clear()
        self.last_prepared_metrics.clear()

    def close(self) -> None:
        for target in self.targets:
            target.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def get_status(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "enabled": self.enabled,
            "namespace": self.namespace,
            "mode": self.mode,
            "alpha": self.alpha,
            "targets": [target.to_status() for target in self.targets],
            "metric_alphas": dict(self.metric_alphas),
            "metric_ranges": {k: list(v) for k, v in self.metric_ranges.items()},
            "last_sent_at": self.last_sent_at,
        }

    def send_metrics(self, metrics: Mapping[str, float], send_keys: Optional[set[str]] = None) -> list[dict]:
        sent = []
        self.last_prepared_metrics = {}

        for key in METRIC_NAMES:
            if key not in metrics:
                continue
            value = self._prepare_value(key, metrics[key])
            if value is None:
                continue
            self.last_prepared_metrics[key] = value
            if send_keys is not None and key not in send_keys:
                continue
            if not self.enabled:
                continue
            address = self.metric_address(key)
            for target_id in self._send_message(address, value):
                sent.append({"address": address, "value": value, "target": target_id})

        if sent:
            self.last_sent_at = time.time()
        return sent

    def _prepare_value(self, key: str, value: object) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(numeric):
            return None

        # Keep calibration and runtime on the same signal path: Smoothness(EMA)
        # first, then apply fixed/profile normalization to that smoothed value.
        numeric = self._smooth(key, numeric)

        if self.mode in ("normalize", "fixed"):
            numeric = self._normalize(key, numeric)

        return numeric

    def _normalize(self, key: str, value: float) -> float:
        if key == "sync_correlation":
            return _clamp(value, -1.0, 1.0)
        if key in BOUNDED_METRICS:
            return _clamp(value)
        # Calibrated fixed range takes priority in "fixed" mode (comparable
        # across time). Metrics without a calibrated range fall through to the
        # adaptive logic below, so "fixed" degrades gracefully.
        if self.mode == "fixed" and key in self.metric_ranges:
            lo, hi = self.metric_ranges[key]
            span = hi - lo
            if span < 1e-9:
                return 0.5
            return _clamp((value - lo) / span)
        if key in ADAPTIVE_RANGE_METRICS:
            return self._normalize_range(key, value)
        if key in UNBOUNDED_METRICS:
            current_peak = self._peaks.get(key, 1e-6)
            decayed_peak = current_peak * 0.995
            peak = max(decayed_peak, abs(value), 1e-6)
            self._peaks[key] = peak
            return _clamp(value / peak)
        return value

    def _normalize_range(self, key: str, value: float) -> float:
        lo, hi = self._ranges.get(key, (value, value))
        span = hi - lo
        lo = min(lo + span * RANGE_DECAY, value)
        hi = max(hi - span * RANGE_DECAY, value)
        self._ranges[key] = (lo, hi)
        span = hi - lo
        if span < 1e-9:
            return 0.5
        return _clamp((value - lo) / span)

    def _smooth(self, key: str, value: float) -> float:
        alpha = self.metric_alphas.get(key, self.alpha)
        if alpha >= 1.0:
            self._smoothed[key] = value
            return value

        previous = self._smoothed.get(key)
        if previous is None:
            smoothed = value
        else:
            smoothed = (alpha * value) + ((1.0 - alpha) * previous)

        self._smoothed[key] = smoothed
        return smoothed

    def set_metric_alpha(self, name: str, alpha: float) -> None:
        """Override the output smoothing for a single metric (per-channel).
        Ignores unknown metric names; raises ValueError on an out-of-range alpha."""
        if name not in METRIC_NAMES:
            return
        self.metric_alphas[name] = self._validate_alpha(alpha)

    def set_metric_ranges(self, ranges: Mapping[str, tuple]) -> None:
        """Install calibrated fixed [lo, hi] ranges (used by mode 'fixed').
        Ignores unknown metrics and invalid ranges (hi must exceed lo)."""
        for name, rng in (ranges or {}).items():
            if name not in METRIC_NAMES or rng is None:
                continue
            lo, hi = float(rng[0]), float(rng[1])
            if hi > lo:
                self.metric_ranges[name] = (lo, hi)

    def clear_metric_ranges(self) -> None:
        self.metric_ranges.clear()

    def _send_message(self, address: str, value: object) -> None:
        sent_targets = []
        if not self.enabled:
            return sent_targets
        for target in self.targets:
            if not target.enabled:
                continue
            try:
                target.client.send_message(address, value)
                sent_targets.append(target.id)
            except Exception as exc:
                print(f"OSC send failed for {target.name} {address}: {exc}")
        return sent_targets

    def send_named(self, name: str, value: float) -> None:
        """Send one extra value under the namespace (derived metrics etc.)."""
        if not self.enabled:
            return
        if self._send_message(self._address(name), float(value)):
            self.last_sent_at = time.time()

    def metric_address(self, key: str) -> str:
        return self._address(OSC_ADDRESS_NAMES.get(key, key))

    def _address(self, name: str) -> str:
        if not self.namespace:
            return f"/{name}"
        return f"{self.namespace}/{name}"

    @staticmethod
    def _validate_mode(mode: str) -> str:
        normalized = str(mode).lower()
        if normalized not in {"raw", "normalize", "fixed"}:
            raise ValueError("OSC mode must be 'raw', 'normalize' or 'fixed'")
        return normalized

    @staticmethod
    def _validate_alpha(alpha: float) -> float:
        value = float(alpha)
        if not 0.0 < value <= 1.0:
            raise ValueError("OSC alpha must be > 0 and <= 1")
        return value

    @staticmethod
    def _normalize_namespace(namespace: str) -> str:
        value = str(namespace if namespace is not None else "/field").strip().rstrip("/")
        if not value:
            return ""
        if not value.startswith("/"):
            value = f"/{value}"
        return value
