"""The group box drawn on the preview: one rectangle around everyone on stage.

This is a screen-space box -- the union of the per-person boxes, in frame
pixels. Worth knowing what that does and does not mean:

- It is what you see. Two dancers, one box, drawn where they actually are.
- It is *not* the floor extent. Perspective means the same physical gap reads
  wider downstage than upstage, and raised arms push the edges out. So the box
  growing does not always mean the cast spread out.

Both quantities go out on their own OSC names (``/field/group/width`` in metres
from the floor plane, ``/field/group/box_*`` normalised from this rectangle), so
whichever one a patch listens to, it gets the thing it thinks it is getting.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

_BOX = (192, 211, 52)          # live, same family as the per-dancer boxes
_BOX_HELD = (120, 160, 200)    # frozen while a body is missing
_LABEL = (232, 228, 222)


def _dashed_rect(frame, pt1, pt2, color, thickness=2, dash=10, gap=7) -> None:
    """cv2 has no dashed primitive, and a held box must not be mistaken for a
    live measurement, so we draw one."""
    x1, y1 = pt1
    x2, y2 = pt2
    edges = [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
             ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]
    for (ax, ay), (bx, by) in edges:
        length = int(round(np.hypot(bx - ax, by - ay)))
        if length <= 0:
            continue
        for start in range(0, length, dash + gap):
            end = min(start + dash, length)
            t0, t1 = start / length, end / length
            p0 = (int(round(ax + (bx - ax) * t0)), int(round(ay + (by - ay) * t0)))
            p1 = (int(round(ax + (bx - ax) * t1)), int(round(ay + (by - ay) * t1)))
            cv2.line(frame, p0, p1, color, thickness, cv2.LINE_AA)


def draw_group_box(frame, bbox, extent=None, thickness: int = 3):
    """Draw the group rectangle: a full, closed outline around everyone.

    ``extent`` (a GroupExtent) only supplies the label text; the rectangle
    itself is ``bbox``, in frame pixels. A single dancer still gets a box --
    their own -- because on screen that is a real, visible rectangle, unlike
    the floor extent which would collapse to a point.
    """
    if frame is None or bbox is None:
        return frame
    h, w = frame.shape[:2]
    x1 = int(round(max(0.0, min(float(bbox[0]), w - 1))))
    y1 = int(round(max(0.0, min(float(bbox[1]), h - 1))))
    x2 = int(round(max(0.0, min(float(bbox[2]), w - 1))))
    y2 = int(round(max(0.0, min(float(bbox[3]), h - 1))))
    if x2 <= x1 or y2 <= y1:
        return frame

    held = bool(getattr(extent, "held", False))
    color = _BOX_HELD if held else _BOX

    if held:
        _dashed_rect(frame, (x1, y1), (x2, y2), color, int(thickness))
    else:
        # A dark backing line first: the outline has to stay readable over a
        # bright floor or a white cyc, where the box colour alone washes out.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (24, 22, 20),
                      int(thickness) + 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, int(thickness), cv2.LINE_AA)

    label = "GROUP"
    if extent is not None:
        count = int(getattr(extent, "count", 0))
        label = f"GROUP {count}"
        width = float(getattr(extent, "width", 0.0))
        depth = float(getattr(extent, "depth", 0.0))
        if count >= 2 and (width > 0.0 or depth > 0.0):
            label += f"  {width:.2f} x {depth:.2f} m"
        if held:
            label += "  HELD"
    label_pos = (x1, max(18, y1 - 10))
    cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (24, 22, 20), 4, cv2.LINE_AA)
    cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                color if held else _LABEL, 2, cv2.LINE_AA)
    return frame


def normalized_box(bbox, frame_shape) -> Optional[dict]:
    """Screen box -> 0..1 fractions of the frame, for the OSC side.

    Normalised rather than pixels so a patch does not silently break when the
    performance preset changes the capture resolution.
    """
    if bbox is None or frame_shape is None:
        return None
    h, w = float(frame_shape[0]), float(frame_shape[1])
    if h <= 0 or w <= 0:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return {
        "x1": x1 / w, "y1": y1 / h,
        "x2": x2 / w, "y2": y2 / h,
        "w": (x2 - x1) / w, "h": (y2 - y1) / h,
        "cx": ((x1 + x2) / 2.0) / w, "cy": ((y1 + y2) / 2.0) / h,
    }


def denormalized_box(norm, frame_shape):
    """0..1 fractions -> frame pixels; the inverse of ``normalized_box``.

    Used to put the *smoothed* box back into pixels for drawing, so the preview
    rectangle and the OSC values stay the same rectangle.
    """
    if norm is None or frame_shape is None:
        return None
    h, w = float(frame_shape[0]), float(frame_shape[1])
    if h <= 0 or w <= 0:
        return None
    return (float(norm["x1"]) * w, float(norm["y1"]) * h,
            float(norm["x2"]) * w, float(norm["y2"]) * h)
