import math
import time
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
# height is -com_y in metres around the hip origin (often negative) and sway
# is a small metre-scale offset; clamping them to 0..1 flattens both to ~0.
# Track an adaptive min/max range instead so normalize mode stays expressive.
ADAPTIVE_RANGE_METRICS = {"height", "sway"}
UNBOUNDED_METRICS = {"energy", "expansion", "curvature", "torque", "jerk"}

# Per-message decay applied to the adaptive range so stale extremes fade.
RANGE_DECAY = 0.001


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


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
        self.client = SimpleUDPClient(self.host, self.port)
        self._smoothed: Dict[str, float] = {}
        # Per-metric output smoothing override; falls back to self.alpha (the
        # global slider) for any metric without its own value. 1.0 = off.
        self.metric_alphas: Dict[str, float] = {}
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
        client_changed = False

        if host is not None and host != self.host:
            self.host = host
            client_changed = True
        if port is not None and int(port) != self.port:
            self.port = int(port)
            client_changed = True
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

        if client_changed:
            self.client = SimpleUDPClient(self.host, self.port)

    def reset_state(self) -> None:
        self._smoothed.clear()
        self._peaks.clear()
        self._ranges.clear()
        self.last_prepared_metrics.clear()

    def get_status(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "enabled": self.enabled,
            "namespace": self.namespace,
            "mode": self.mode,
            "alpha": self.alpha,
            "metric_alphas": dict(self.metric_alphas),
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
            self._send_message(address, value)
            sent.append({"address": address, "value": value})

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

        if self.mode == "normalize":
            numeric = self._normalize(key, numeric)

        return self._smooth(key, numeric)

    def _normalize(self, key: str, value: float) -> float:
        if key == "sync_correlation":
            return _clamp((value + 1.0) / 2.0)
        if key in BOUNDED_METRICS:
            return _clamp(value)
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

    def _send_message(self, address: str, value: object) -> None:
        try:
            self.client.send_message(address, value)
        except Exception as exc:
            print(f"OSC send failed for {address}: {exc}")

    def send_named(self, name: str, value: float) -> None:
        """Send one extra value under the namespace (derived metrics etc.)."""
        if not self.enabled:
            return
        self._send_message(f"{self.namespace}/{name}", float(value))

    def metric_address(self, key: str) -> str:
        return f"{self.namespace}/{OSC_ADDRESS_NAMES.get(key, key)}"

    @staticmethod
    def _validate_mode(mode: str) -> str:
        normalized = str(mode).lower()
        if normalized not in {"raw", "normalize"}:
            raise ValueError("OSC mode must be 'raw' or 'normalize'")
        return normalized

    @staticmethod
    def _validate_alpha(alpha: float) -> float:
        value = float(alpha)
        if not 0.0 < value <= 1.0:
            raise ValueError("OSC alpha must be > 0 and <= 1")
        return value

    @staticmethod
    def _normalize_namespace(namespace: str) -> str:
        value = str(namespace or "/field").strip().rstrip("/")
        if not value:
            return "/field"
        if not value.startswith("/"):
            value = f"/{value}"
        return value
