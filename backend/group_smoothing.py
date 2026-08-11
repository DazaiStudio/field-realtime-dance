"""Output smoothing for the group box.

Both group quantities are measured fresh from whatever the detector returned
this frame, and nothing downstream filters them: ``OSCSender.send_named`` is a
raw passthrough, and the per-metric EMA in OSCSender only covers the nine dance
metrics. So the detector's frame-to-frame jitter -- an edge that wobbles
because YOLO's box moved a few pixels, a width that breathes because one hip
flicked between MEDIUM and LOW confidence -- lands directly on a patch driving
a light off ``group/box_w``.

What is and is not filtered:

- **Smoothed**: ``box_x1/y1/x2/y2`` and the floor primitives width, depth, cx,
  cz. ``area``, ``diagonal`` and ``aspect`` are GroupExtent properties derived
  from width and depth, so they follow the smoothed primitives and stay
  mutually consistent; filtering them as separate signals would let ``area``
  drift away from ``width * depth``. Likewise ``box_w/h/cx/cy`` are rebuilt
  from the smoothed corners rather than filtered on their own.
- **Never smoothed**: ``count`` and ``held``. The receiving end is told to
  branch on both (SESSION_NOTES_20260810 §3). A count easing from 2 to 3 passes
  through 2.4 dancers, and a fractional ``held`` is not a gate at all.

One-Euro rather than the fixed-alpha EMA used for the metrics: a group box is
mostly still and occasionally fast, which is precisely the case where a single
alpha has to trade jitter at rest against lag on a fast crossing.

The screen box is filtered in normalised 0-1 units, not pixels, for the same
reason it is *sent* normalised -- so a performance preset that changes capture
resolution does not quietly change what the tuning means.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from group_overlay import denormalized_box, normalized_box
from one_euro import JointSmoother

# Speed, in each signal's own units per second, at which the cutoff doubles
# (beta = cutoff / speed). One-Euro's beta is not scale-free and these two
# signals are on different scales -- a box edge crossing half the frame in a
# second is 0.5 units/s, a walking dancer about 1 m/s -- so they cannot share
# one value. Note the JointSmoother defaults are tuned for millimetre joint
# coordinates and would leave the adaptive term inert on both of these.
_BOX_REF_SPEED = 0.5      # frame fractions per second
_EXTENT_REF_SPEED = 1.0   # metres per second

# The UI talks in analysis frames, like the per-metric smoothness sliders, and
# the conversion lives here so the filter itself keeps its honest unit (Hz).
#
# An EMA of alpha = 2/(N+1) -- the N-frame average those sliders use -- is a
# first-order low-pass with alpha = 1/(1 + tau/dt) and tau = 1/(2*pi*fc),
# so fc = 1 / (pi * dt * (N - 1)).
#
# dt is nominal, not measured: the frame count describes what the filter does
# at the configured analysis rate, which is the setting an operator is picking
# against. It is only an equivalence at rest anyway -- One-Euro widens its own
# cutoff as the box speeds up, which is the whole reason it is here.
NOMINAL_ANALYSIS_DT = 0.1   # seconds; the viewer's default 10 Hz analysis rate
DEFAULT_FRAMES = 3          # ~1.6 Hz at 10 Hz, same default as the metric rows
MAX_FRAMES = 10


def cutoff_for_frames(frames, dt: float = NOMINAL_ANALYSIS_DT) -> float:
    """Analysis frames -> One-Euro cutoff in Hz. 0 (and 1) mean off.

    1 is off for the same reason it is on the metric sliders: a one-frame
    average is the identity, and 2/(1+1) = 1 is a no-op alpha.
    """
    try:
        count = float(frames)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(count) or count <= 1.0:
        return 0.0
    count = min(count, float(MAX_FRAMES))
    step = float(dt) if float(dt) > 0 else NOMINAL_ANALYSIS_DT
    return 1.0 / (np.pi * step * (count - 1.0))


class GroupSmoother:
    """One-Euro filter over the group box outputs. ``cutoff <= 0`` is off.

    "0 = off" is a UI convention rather than the maths -- a literal 0 Hz cutoff
    would freeze the output forever. When off, values are passed through
    untouched and the filter state is dropped, so switching smoothing off
    mid-show snaps back to the live measurement instead of drifting there.
    """

    def __init__(self, cutoff: float = 0.0):
        self._box = JointSmoother()
        self._extent = JointSmoother()
        self.cutoff = 0.0
        self.configure(cutoff)

    @property
    def enabled(self) -> bool:
        return self.cutoff > 0.0

    def configure(self, cutoff: Optional[float] = None) -> None:
        if cutoff is None:
            return
        value = max(0.0, float(cutoff))
        if value == self.cutoff:
            return
        self.cutoff = value
        if value > 0.0:
            self._box.configure(min_cutoff=value, beta=value / _BOX_REF_SPEED)
            self._extent.configure(min_cutoff=value, beta=value / _EXTENT_REF_SPEED)
        # Retune and restart together: the old state was filtered at the old
        # cutoff, and blending it into the new one is neither setting.
        self.reset()

    def reset(self) -> None:
        self._box.reset()
        self._extent.reset()

    def apply(self, extent, box_norm: Optional[dict], now: float):
        """Filter one frame's outputs -> ``(extent, box_norm)``.

        ``now`` is in **seconds**, matching ``GroupExtentTracker.update`` at the
        call site (JointSmoother takes milliseconds; the conversion is here so
        the two neighbouring calls cannot be given different clocks).

        Held frames are filtered like any other: the repeated value is what the
        tracker is asserting, so letting the filter converge on it means that
        when the hold ends the output eases once, from the held value to the
        new measurement, instead of stepping twice.
        """
        if not self.enabled:
            return extent, box_norm

        count = int(getattr(extent, "count", 0) or 0)
        if count <= 0 and box_norm is None:
            # Empty stage. Drop the state so the next entrance starts at the
            # dancer's real position instead of sliding in from wherever the
            # cast happened to leave the frame.
            self.reset()
            return extent, box_norm

        t_ms = float(now) * 1000.0

        if box_norm is not None:
            corners = np.array(
                [box_norm["x1"], box_norm["y1"], box_norm["x2"], box_norm["y2"]],
                dtype=float,
            )
            x1, y1, x2, y2 = (float(v) for v in self._box.filter(corners, t_ms))
            # Each corner is low-passed with its own speed-adaptive cutoff, so
            # unlike a shared-alpha filter this one can cross x2 under x1 on a
            # fast shrink. Order them before deriving w/h, or a negative width
            # goes out on the wire looking like a real measurement.
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            box_norm = {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "w": x2 - x1, "h": y2 - y1,
                "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
            }

        if extent is not None and count > 0:
            primitives = np.array(
                [extent.width, extent.depth, extent.cx, extent.cz], dtype=float
            )
            width, depth, cx, cz = (float(v) for v in self._extent.filter(primitives, t_ms))
            # replace() names only the four primitives on purpose: count and
            # held travel through untouched and cannot be smoothed by accident.
            extent = replace(extent, width=width, depth=depth, cx=cx, cz=cz)

        return extent, box_norm


def smoothed_group_outputs(smoother: GroupSmoother, extent, raw_box, frame_shape, now: float):
    """The single path from a raw union box to the three published forms.

    Returns ``(extent, box_px, box_norm)``. The pixel box is re-derived from the
    smoothed normalised box so the rectangle drawn on the preview is the one
    that went out on OSC -- if those two ever diverged, watching the preview
    would stop being a way to check the wire.

    With smoothing off the caller's own ``raw_box`` is handed straight back
    rather than round-tripped through normalise/denormalise, so "off" is
    exactly the behaviour that existed before this module.
    """
    box_norm = normalized_box(raw_box, frame_shape)
    if not smoother.enabled:
        return extent, raw_box, box_norm
    extent, box_norm = smoother.apply(extent, box_norm, now)
    return extent, denormalized_box(box_norm, frame_shape), box_norm
