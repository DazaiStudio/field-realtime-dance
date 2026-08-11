"""Group floor-plane extent: how much floor the whole cast is covering.

Deliberately independent of the stable-id registry. The bbox is a property of
the bodies visible this frame, not of who they are, so this path never touches
StableIdTracker and works with Stable ID switched off.

Two things this module exists to get right:

1. **Absolute coordinates.** ``k4abt32_to_h36m17`` root-centers every skeleton
   on its own pelvis, so the H36M arrays the rest of the pipeline passes around
   all sit at the origin -- a group bbox built from those is identically zero.
   The input here is the *raw* K4ABT joint array (mm, depth-camera space).

2. **Dropout is not togetherness.** A dancer the tracker loses looks exactly
   like a dancer who stepped next to their partner: the bbox shrinks. With the
   rehearsal position at 4.85 m already dropping frames (see
   SESSION_NOTES_20260805 §4a), a raw per-frame bbox would report a confident,
   plausible, wrong number. GroupExtentTracker holds the last good value while
   the count is depressed and flags it, so the receiving end can tell a real
   huddle from a lost body.

Coordinates follow the rest of the codebase: X/Z is the floor plane, Y is
vertical (cf. dance_metrics sway), and values are converted mm -> m on output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from keypoint_mapping import _K4_L_HIP, _K4_R_HIP

# K4ABT confidence levels: 0 = NONE (position is a guess for an occluded joint),
# 1 = LOW, 2 = MEDIUM. Require both hips to beat NONE -- a guessed hip can sit
# metres from the body and would blow the bbox open on its own.
_MIN_HIP_CONFIDENCE = 1

MM_PER_M = 1000.0

# Output unit -> multiplier applied to the metre values on the way out.
LENGTH_SCALE = {"m": 1.0, "cm": 100.0}
DEFAULT_UNITS = "m"


@dataclass(frozen=True)
class GroupExtent:
    """Floor-plane bounding box of every body present, in metres."""

    count: int = 0
    width: float = 0.0      # X extent
    depth: float = 0.0      # Z extent
    cx: float = 0.0         # bbox center X
    cz: float = 0.0         # bbox center Z
    held: bool = False      # value repeated from an earlier frame, not measured

    @property
    def area(self) -> float:
        """Floor covered, m^2.

        Beware with two dancers: the box collapses whenever they line up with
        an axis, so area drops to ~0 while they are still far apart. It is a
        trustworthy 'how much floor' signal from three dancers up; for two,
        drive things off diagonal or width/depth instead.
        """
        return float(self.width) * float(self.depth)

    @property
    def diagonal(self) -> float:
        """Corner-to-corner span, m. Unlike area this never collapses on a
        line formation, so it is the safer size scalar for two dancers."""
        return float(np.hypot(self.width, self.depth))

    @property
    def aspect(self) -> float:
        """width / depth. >1 spread across the stage, <1 strung up-and-down it,
        1 square. Clamped rather than infinite when depth is 0."""
        if self.depth <= 1e-9:
            return 999.0 if self.width > 1e-9 else 1.0
        return float(self.width) / float(self.depth)

    def as_osc(self, units: str = "m") -> dict:
        """Flat name -> float mapping for OSCSender.send_named.

        ``units`` scales the length values ("m" or "cm"). ``group/units`` is
        always sent alongside as the scale factor actually applied (1 or 100),
        so a patch can tell which it is receiving instead of inferring it from
        magnitude -- 2.4 m and 2.4 cm are both plausible-looking numbers.
        """
        scale = LENGTH_SCALE.get(str(units).lower(), 1.0)
        # Area is two lengths, so it scales by the square: m^2 -> cm^2 is 10000.
        area_scale = scale * scale
        return {
            "group/count": float(self.count),
            "group/width": float(self.width) * scale,
            "group/depth": float(self.depth) * scale,
            "group/area": self.area * area_scale,
            "group/diagonal": self.diagonal * scale,
            "group/aspect": self.aspect,          # a ratio: unitless either way
            "group/cx": float(self.cx) * scale,
            "group/cz": float(self.cz) * scale,
            "group/units": float(scale),
            "group/held": 1.0 if self.held else 0.0,
        }


def hip_floor_position(
    body,
    mirrored: bool = False,
    min_confidence: int = _MIN_HIP_CONFIDENCE,
) -> Optional[tuple[float, float]]:
    """One raw K4ABT body -> its (x, z) floor position in metres, or None.

    Uses the hip midpoint: it is the same point the H36M pelvis is built from,
    it is the steadiest joint on a moving dancer, and it is what the §4a
    coverage measurements were taken against.

    Bodies whose hips are missing or NONE-confidence return None rather than a
    guess -- they would otherwise drag the bbox to wherever K4ABT parked the
    occluded joint.
    """
    joints = np.asarray(getattr(body, "joints", body), dtype=float)
    if joints.ndim != 2 or joints.shape[0] <= max(_K4_L_HIP, _K4_R_HIP):
        return None
    left, right = joints[_K4_L_HIP], joints[_K4_R_HIP]
    if joints.shape[1] > 3:
        if int(left[3]) < min_confidence or int(right[3]) < min_confidence:
            return None
    mid = (left[:3] + right[:3]) / 2.0
    if not np.isfinite(mid).all():
        return None
    x, z = float(mid[0]), float(mid[2])
    if mirrored:
        # Mirror the floor plane the same way the preview image is flipped, so
        # "stage left" in the OSC matches what the operator is watching.
        x = -x
    return (x / MM_PER_M, z / MM_PER_M)


def hip_floor_positions(
    bodies: Iterable,
    mirrored: bool = False,
    min_confidence: int = _MIN_HIP_CONFIDENCE,
) -> list[tuple[float, float]]:
    """Raw K4ABT bodies -> one (x, z) floor position per usable body."""
    out = []
    for body in bodies or []:
        pos = hip_floor_position(body, mirrored, min_confidence)
        if pos is not None:
            out.append(pos)
    return out


def union_bbox(boxes: Sequence[Sequence[float]]):
    """Smallest screen-space rect containing every per-person box.

    This is the group box the operator sees on the preview. It is a *different*
    quantity from the floor extent: perspective means two dancers the same
    distance apart read as a wider box downstage than upstage, and raised arms
    push the edges out. Useful to look at, not a substitute for the metre
    values -- which is why both go out under their own OSC names.
    """
    usable = [b for b in (boxes or []) if b is not None and len(b) >= 4
              and all(np.isfinite(float(v)) for v in b[:4])]
    if not usable:
        return None
    x1 = min(float(b[0]) for b in usable)
    y1 = min(float(b[1]) for b in usable)
    x2 = max(float(b[2]) for b in usable)
    y2 = max(float(b[3]) for b in usable)
    return (x1, y1, x2, y2)


def measure_extent(positions: Sequence[tuple[float, float]]) -> GroupExtent:
    """Floor bbox over already-converted (x, z) metre positions.

    A single body is a legitimate group of one: width/depth are 0 and the
    center is that dancer. Callers that need to distinguish "one dancer" from
    "we lost the others" should read ``count`` (or let GroupExtentTracker do it).
    """
    if not positions:
        return GroupExtent()
    xs = [float(p[0]) for p in positions]
    zs = [float(p[1]) for p in positions]
    x_min, x_max = min(xs), max(xs)
    z_min, z_max = min(zs), max(zs)
    return GroupExtent(
        count=len(positions),
        width=x_max - x_min,
        depth=z_max - z_min,
        cx=(x_min + x_max) / 2.0,
        cz=(z_min + z_max) / 2.0,
    )


class GroupExtentTracker:
    """Per-frame extent with dropout hold.

    While fewer bodies are visible than were recently present, the previous
    extent is repeated with ``held=True`` instead of emitting a collapsed bbox.
    If the lower count persists for ``hold_seconds`` it is accepted as real --
    someone actually left -- and measurement resumes.

    ``hold_seconds`` is deliberately short (default 1.0 s ~= 30 analysis
    frames). The 12 s used for stable-id re-identification would be far too
    long here: a group bbox frozen for twelve seconds is worse than a noisy one,
    because the number keeps looking live while the cast walks out from under it.
    """

    def __init__(self, hold_seconds: float = 1.0, max_people: int = 4):
        self.hold_seconds = max(0.0, float(hold_seconds))
        self.max_people = max(1, int(max_people))
        self._last: Optional[GroupExtent] = None
        self._expected_count = 0
        self._short_since: Optional[float] = None

    def reset(self) -> None:
        self._last = None
        self._expected_count = 0
        self._short_since = None

    def update(self, positions: Sequence[tuple[float, float]], now: float) -> GroupExtent:
        if len(positions) > self.max_people:
            # Keep the people nearest the group's own center: a spurious extra
            # body (a reflection, someone in the wings) is usually the outlier,
            # and letting it through would inflate the bbox without bound.
            positions = _nearest_to_center(positions, self.max_people)

        measured = measure_extent(positions)
        count = measured.count

        if count == 0:
            # Nothing to measure. Hold briefly, then report the empty stage.
            if self._last is not None and self._within_hold(now):
                return _as_held(self._last)
            self.reset()
            return GroupExtent()

        if count >= self._expected_count:
            self._expected_count = count
            self._short_since = None
            self._last = measured
            return measured

        # Fewer bodies than we had. Assume dropout until it lasts long enough
        # to be a real exit.
        if self._short_since is None:
            self._short_since = now
        if (now - self._short_since) <= self.hold_seconds and self._last is not None:
            return _as_held(self._last)

        self._expected_count = count
        self._short_since = None
        self._last = measured
        return measured

    def _within_hold(self, now: float) -> bool:
        if self._short_since is None:
            self._short_since = now
        return (now - self._short_since) <= self.hold_seconds


def _as_held(extent: GroupExtent) -> GroupExtent:
    return GroupExtent(
        count=extent.count,
        width=extent.width,
        depth=extent.depth,
        cx=extent.cx,
        cz=extent.cz,
        held=True,
    )


def _nearest_to_center(
    positions: Sequence[tuple[float, float]], keep: int
) -> list[tuple[float, float]]:
    pts = [(float(p[0]), float(p[1])) for p in positions]
    cx = sum(p[0] for p in pts) / len(pts)
    cz = sum(p[1] for p in pts) / len(pts)
    ranked = sorted(pts, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cz) ** 2)
    return ranked[:keep]
