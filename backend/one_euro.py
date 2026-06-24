"""One-Euro filter for real-time jitter reduction on pose joints.

The One-Euro filter (Casiez et al., 2012) is a speed-adaptive low-pass:
it filters hard when the signal moves slowly (kills jitter) and lightly
when it moves fast (keeps responsiveness / low lag). This is the realtime
counterpart to the offline centered moving average used in the dashboard;
unlike a fixed-alpha EMA it does not force a single lag-vs-smoothness tradeoff.

Applied at the SKELETON layer (joint coordinates) BEFORE the metrics engine,
so it benefits every backend and cleans the positions that the high-order
derivatives (torque = 2nd, jerk = 3rd) amplify.

Timestamps are passed in milliseconds (matching the viewer); internally
converted to seconds. Cutoff frequencies are in Hz.
"""
import math
import numpy as np

_MIN_DT = 1e-3  # clamp to avoid div-by-zero / spikes on duplicate timestamps


def _alpha(cutoff, dt):
    # Smoothing factor of a first-order low-pass with the given cutoff (Hz).
    # Works elementwise when `cutoff` is an array.
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class JointSmoother:
    """Vectorised One-Euro filter over an array of joint coordinates.

    Each coordinate of each joint is filtered independently; all share the
    frame timestamp. Defaults are tuned for millimetre-scale H36M coordinates
    (the engine works in ~mm)."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.0008,
                 d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def configure(self, min_cutoff: float = None, beta: float = None,
                  d_cutoff: float = None) -> None:
        if min_cutoff is not None:
            self.min_cutoff = float(min_cutoff)
        if beta is not None:
            self.beta = float(beta)
        if d_cutoff is not None:
            self.d_cutoff = float(d_cutoff)

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def filter(self, x: np.ndarray, t_ms: float) -> np.ndarray:
        """Return the smoothed copy of `x` (same shape) at time `t_ms` (ms)."""
        x = np.asarray(x, dtype=float)
        t = float(t_ms) / 1000.0

        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            return x.copy()

        dt = t - self._t_prev
        if dt <= 0.0:
            dt = _MIN_DT

        # Filtered derivative.
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Speed-adaptive cutoff (per element), then low-pass the signal.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat
