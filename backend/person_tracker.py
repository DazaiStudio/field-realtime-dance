"""Optional person detection + stable track selection helpers.

The heavy Ultralytics dependency is imported lazily so the default MediaPipe
viewer keeps working on machines that have not installed YOLO tracking extras.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class PersonTrack:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float = 0.0


@dataclass
class StablePersonTrack:
    stable_id: int
    raw_id: Optional[int]
    bbox: tuple[float, float, float, float]
    confidence: float = 0.0
    state: str = "tracking"
    last_seen_at: float = 0.0


def bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    return bbox_area((max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)))


def bbox_containment(inner: Sequence[float], outer: Sequence[float]) -> float:
    area = bbox_area(inner)
    if area <= 0:
        return 0.0
    return bbox_intersection_area(inner, outer) / area


def suppress_duplicate_person_tracks(
    tracks: Sequence[PersonTrack],
    containment_threshold: float = 0.82,
    max_area_ratio: float = 0.72,
    high_iou_threshold: float = 0.82,
) -> list[PersonTrack]:
    """Drop partial-body duplicate boxes before assigning stable IDs."""
    clean = [track for track in tracks if track.track_id is not None and bbox_area(track.bbox) > 0]
    keep = [True] * len(clean)
    for i, a in enumerate(clean):
        if not keep[i]:
            continue
        area_a = bbox_area(a.bbox)
        for j, b in enumerate(clean):
            if i == j or not keep[j]:
                continue
            area_b = bbox_area(b.bbox)
            if area_a <= area_b:
                smaller_i, larger_i = i, j
                smaller, larger = a, b
                smaller_area, larger_area = area_a, area_b
            else:
                smaller_i, larger_i = j, i
                smaller, larger = b, a
                smaller_area, larger_area = area_b, area_a
            if smaller_area <= 0 or larger_area <= 0:
                continue
            area_ratio = smaller_area / larger_area
            contained = bbox_containment(smaller.bbox, larger.bbox)
            if contained >= containment_threshold and area_ratio <= max_area_ratio:
                keep[smaller_i] = False
                continue
            if bbox_iou(smaller.bbox, larger.bbox) >= high_iou_threshold:
                drop_i = smaller_i if smaller.confidence <= larger.confidence else larger_i
                keep[drop_i] = False
    return [track for track, should_keep in zip(clean, keep) if should_keep]


def expand_bbox(
    bbox: Sequence[float],
    frame_shape,
    padding: float = 0.18,
    min_size: int = 32,
) -> Optional[tuple[int, int, int, int]]:
    """Return a clamped crop rect around a person bbox."""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad = max(bw, bh) * float(padding)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    size = max(bw + pad * 2.0, bh + pad * 2.0, float(min_size))

    left = int(round(cx - size * 0.5))
    top = int(round(cy - size * 0.5))
    right = int(round(cx + size * 0.5))
    bottom = int(round(cy + size * 0.5))

    left = max(0, min(w - 1, left))
    top = max(0, min(h - 1, top))
    right = max(left + 1, min(w, right))
    bottom = max(top + 1, min(h, bottom))
    if right - left < min_size or bottom - top < min_size:
        return None
    return left, top, right, bottom


def expand_bbox_rect(
    bbox: Sequence[float],
    frame_shape,
    padding_x: float = 0.06,
    padding_y: float = 0.08,
    min_size: int = 32,
) -> Optional[tuple[int, int, int, int]]:
    """Return a tight rectangular crop around a person bbox.

    Square crops are useful for some top-down models, but with two dancers close
    together they can include the neighbor and make single-person MediaPipe lock
    onto the same person twice.
    """
    h, w = int(frame_shape[0]), int(frame_shape[1])
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * float(padding_x)
    pad_y = bh * float(padding_y)

    left = int(round(x1 - pad_x))
    top = int(round(y1 - pad_y))
    right = int(round(x2 + pad_x))
    bottom = int(round(y2 + pad_y))

    left = max(0, min(w - 1, left))
    top = max(0, min(h - 1, top))
    right = max(left + 1, min(w, right))
    bottom = max(top + 1, min(h, bottom))
    if right - left < min_size or bottom - top < min_size:
        return None
    return left, top, right, bottom


def crop_frame(frame, rect: tuple[int, int, int, int]):
    x1, y1, x2, y2 = rect
    return frame[y1:y2, x1:x2]


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


class StableTrackSelector:
    """Keep one active track id stable across short occlusions."""

    def __init__(
        self,
        hold_seconds: float = 0.8,
        reidentify_seconds: float = 12.0,
        reidentify_min_iou: float = 0.12,
        reidentify_max_center_distance: float = 1.35,
    ):
        self.hold_seconds = float(hold_seconds)
        self.reidentify_seconds = float(reidentify_seconds)
        self.reidentify_min_iou = float(reidentify_min_iou)
        self.reidentify_max_center_distance = float(reidentify_max_center_distance)
        self.locked_id: Optional[int] = None
        self.stable_id: Optional[int] = None
        self._next_stable_id = 1
        self.last_track: Optional[PersonTrack] = None
        self.last_seen_at: Optional[float] = None

    def reset(self) -> None:
        self.locked_id = None
        self.stable_id = None
        self._next_stable_id = 1
        self.last_track = None
        self.last_seen_at = None

    def choose(self, tracks: Sequence[PersonTrack], now: float):
        clean_tracks = [track for track in tracks if track.track_id is not None]
        by_id = {int(track.track_id): track for track in clean_tracks}

        if self.locked_id is not None and self.locked_id in by_id:
            track = by_id[self.locked_id]
            self._ensure_stable_id()
            self.last_track = track
            self.last_seen_at = now
            return track, "tracking"

        if self._can_hold(now):
            return self.last_track, "holding"

        reidentified = self._reidentify(clean_tracks, now)
        if reidentified is not None:
            self.locked_id = int(reidentified.track_id)
            self._ensure_stable_id()
            self.last_track = reidentified
            self.last_seen_at = now
            return reidentified, "reidentified"

        if clean_tracks:
            track = max(clean_tracks, key=lambda item: (bbox_area(item.bbox), item.confidence))
            self._assign_new_stable_id()
            self.locked_id = int(track.track_id)
            self.last_track = track
            self.last_seen_at = now
            return track, "tracking"

        if not self._can_reidentify(now):
            self.locked_id = None
            self.stable_id = None
            self.last_track = None
            self.last_seen_at = None
        return None, "lost"

    def _can_hold(self, now: float) -> bool:
        if self.last_track is None or self.last_seen_at is None:
            return False
        return (now - self.last_seen_at) <= self.hold_seconds

    def _can_reidentify(self, now: float) -> bool:
        if self.last_track is None or self.last_seen_at is None or self.stable_id is None:
            return False
        return (now - self.last_seen_at) <= self.reidentify_seconds

    def _ensure_stable_id(self) -> None:
        if self.stable_id is None:
            self._assign_new_stable_id()

    def _assign_new_stable_id(self) -> None:
        self.stable_id = self._next_stable_id
        self._next_stable_id += 1

    def _reidentify(self, tracks: Sequence[PersonTrack], now: float) -> Optional[PersonTrack]:
        if not tracks or not self._can_reidentify(now):
            return None
        scored = [(self._reidentify_score(track), track) for track in tracks]
        scored.sort(key=lambda item: item[0], reverse=True)
        score, track = scored[0]
        if len(tracks) == 1 and score > 0.0:
            return track
        return track if score >= 1.0 else None

    def _reidentify_score(self, track: PersonTrack) -> float:
        if self.last_track is None:
            return 0.0
        iou = bbox_iou(self.last_track.bbox, track.bbox)
        center_score = 1.0 - min(1.0, normalized_center_distance(self.last_track.bbox, track.bbox))
        if iou >= self.reidentify_min_iou:
            return 1.0 + iou + center_score
        if normalized_center_distance(self.last_track.bbox, track.bbox) <= self.reidentify_max_center_distance:
            return center_score
        return 0.0


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = bbox_area((ix1, iy1, ix2, iy2))
    union = bbox_area(a) + bbox_area(b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def normalized_center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    acx, acy = (ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5
    bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
    aw, ah = max(1.0, ax2 - ax1), max(1.0, ay2 - ay1)
    bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
    scale = max(1.0, ((aw * aw + ah * ah) ** 0.5 + (bw * bw + bh * bh) ** 0.5) * 0.5)
    distance = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
    return distance / scale


class MultiPersonTrackRegistry:
    """Maintain stable IDs for all visible people, then pick one active target."""

    def __init__(
        self,
        hold_seconds: float = 0.8,
        reidentify_seconds: float = 12.0,
        reidentify_min_iou: float = 0.12,
        reidentify_max_center_distance: float = 0.85,
        auto_switch_area_ratio: float = 1.3,
    ):
        self.hold_seconds = float(hold_seconds)
        self.reidentify_seconds = float(reidentify_seconds)
        self.reidentify_min_iou = float(reidentify_min_iou)
        self.reidentify_max_center_distance = float(reidentify_max_center_distance)
        self.auto_switch_area_ratio = float(auto_switch_area_ratio)
        self._next_stable_id = 1
        self._tracks: dict[int, StablePersonTrack] = {}
        self._raw_to_stable: dict[int, int] = {}
        self._last_active_id: Optional[int] = None

    def reset(self) -> None:
        self._next_stable_id = 1
        self._tracks = {}
        self._raw_to_stable = {}
        self._last_active_id = None

    def update(self, raw_tracks: Sequence[PersonTrack], now: float) -> list[StablePersonTrack]:
        clean_tracks = suppress_duplicate_person_tracks(raw_tracks)
        matched_stable_ids: set[int] = set()
        unmatched_raw_tracks: list[PersonTrack] = []

        for raw in clean_tracks:
            stable_id = self._raw_to_stable.get(int(raw.track_id))
            if stable_id is not None and stable_id in self._tracks and stable_id not in matched_stable_ids:
                self._update_track(stable_id, raw, now)
                matched_stable_ids.add(stable_id)
            else:
                unmatched_raw_tracks.append(raw)

        for raw in unmatched_raw_tracks:
            stable_id = self._best_reidentify_match(raw, now, matched_stable_ids)
            if stable_id is None:
                stable_id = self._new_track(raw, now)
            else:
                self._update_track(stable_id, raw, now)
            matched_stable_ids.add(stable_id)

        for stable_id in list(self._tracks):
            if stable_id in matched_stable_ids:
                continue
            track = self._tracks[stable_id]
            age = now - track.last_seen_at
            if age <= self.hold_seconds:
                track.raw_id = None
                track.state = "holding"
            elif age <= self.reidentify_seconds:
                track.raw_id = None
                track.state = "lost"
            else:
                del self._tracks[stable_id]

        self._rebuild_raw_index()
        return self.tracks()

    def tracks(self, include_lost: bool = True) -> list[StablePersonTrack]:
        tracks = list(self._tracks.values())
        if not include_lost:
            tracks = [track for track in tracks if track.state != "lost"]
        return sorted(tracks, key=lambda track: track.stable_id)

    def choose_active(
        self,
        selection: str = "auto_largest",
        frame_shape=None,
    ) -> tuple[Optional[StablePersonTrack], str]:
        selected = str(selection or "auto_largest").strip()
        tracks = self.tracks(include_lost=True)
        if selected.startswith("id:"):
            try:
                stable_id = int(selected.split(":", 1)[1])
            except ValueError:
                stable_id = None
            track = self._tracks.get(stable_id) if stable_id is not None else None
            if track is None or track.state == "lost":
                return None, "lost"
            self._last_active_id = track.stable_id
            return track, track.state

        visible = [track for track in tracks if track.state == "tracking"]
        if not visible:
            visible = [track for track in tracks if track.state == "holding"]
        if not visible:
            return None, "lost"

        if selected == "auto_center":
            frame_center = self._frame_center(frame_shape)
            if frame_center is not None:
                track = min(visible, key=lambda item: center_distance(item.bbox, frame_center))
                self._last_active_id = track.stable_id
                return track, track.state
        track = max(visible, key=lambda item: (bbox_area(item.bbox), item.confidence))
        # Stickiness: keep the incumbent active dancer unless the challenger
        # is clearly larger, so similar-sized dancers don't swap the metrics
        # subject frame to frame.
        incumbent = self._tracks.get(self._last_active_id) if self._last_active_id is not None else None
        if (
            incumbent is not None
            and incumbent.stable_id != track.stable_id
            and any(candidate is incumbent for candidate in visible)
            and bbox_area(track.bbox) < self.auto_switch_area_ratio * bbox_area(incumbent.bbox)
        ):
            track = incumbent
        self._last_active_id = track.stable_id
        return track, track.state

    def _new_track(self, raw: PersonTrack, now: float) -> int:
        stable_id = self._next_stable_id
        self._next_stable_id += 1
        self._tracks[stable_id] = StablePersonTrack(
            stable_id=stable_id,
            raw_id=int(raw.track_id),
            bbox=tuple(float(v) for v in raw.bbox),
            confidence=float(raw.confidence),
            state="tracking",
            last_seen_at=now,
        )
        return stable_id

    def _update_track(self, stable_id: int, raw: PersonTrack, now: float) -> None:
        track = self._tracks[stable_id]
        track.raw_id = int(raw.track_id)
        track.bbox = tuple(float(v) for v in raw.bbox)
        track.confidence = float(raw.confidence)
        track.state = "tracking"
        track.last_seen_at = now

    def _best_reidentify_match(
        self,
        raw: PersonTrack,
        now: float,
        excluded_ids: set[int],
    ) -> Optional[int]:
        best_id = None
        best_score = 0.0
        for stable_id, track in self._tracks.items():
            if stable_id in excluded_ids or not self._can_reidentify(track, now):
                continue
            score = self._reidentify_score(track, raw)
            if score > best_score:
                best_id = stable_id
                best_score = score
        if best_score >= 1.0:
            return best_id
        if best_id is not None and self._tracks[best_id].state == "lost" and best_score >= 0.25:
            return best_id
        return None

    def _can_reidentify(self, track: StablePersonTrack, now: float) -> bool:
        return (now - track.last_seen_at) <= self.reidentify_seconds

    def _reidentify_score(self, stable: StablePersonTrack, raw: PersonTrack) -> float:
        iou = bbox_iou(stable.bbox, raw.bbox)
        distance = normalized_center_distance(stable.bbox, raw.bbox)
        center_score = 1.0 - min(1.0, distance)
        if iou >= self.reidentify_min_iou:
            return 1.0 + iou + center_score
        if distance <= self.reidentify_max_center_distance:
            return center_score
        return 0.0

    def _rebuild_raw_index(self) -> None:
        self._raw_to_stable = {
            int(track.raw_id): track.stable_id
            for track in self._tracks.values()
            if track.raw_id is not None and track.state == "tracking"
        }

    @staticmethod
    def _frame_center(frame_shape):
        if not frame_shape:
            return None
        h, w = float(frame_shape[0]), float(frame_shape[1])
        return w * 0.5, h * 0.5


def center_distance(bbox: Sequence[float], point: tuple[float, float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    px, py = point
    return ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5


class UltralyticsPersonTracker:
    """YOLO person detector + built-in MOT tracker wrapper."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        tracker_yaml: str = "bytetrack.yaml",
        confidence: float = 0.25,
        iou: float = 0.45,
    ):
        self.model_name = model_name
        self.tracker_yaml = tracker_yaml
        self.confidence = float(confidence)
        self.iou = float(iou)
        self._model = None

    def track(self, frame) -> list[PersonTrack]:
        self._ensure_model()
        results = self._model.track(
            source=frame,
            persist=True,
            classes=[0],
            tracker=self.tracker_yaml,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or getattr(boxes, "id", None) is None:
            return []

        xyxy = _to_numpy(getattr(boxes, "xyxy", None))
        ids = _to_numpy(getattr(boxes, "id", None))
        confs = _to_numpy(getattr(boxes, "conf", None))
        if xyxy is None or ids is None:
            return []

        tracks = []
        for index, raw_id in enumerate(np.asarray(ids).reshape(-1)):
            try:
                track_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            bbox = tuple(float(v) for v in np.asarray(xyxy[index]).reshape(-1)[:4])
            confidence = float(np.asarray(confs).reshape(-1)[index]) if confs is not None else 0.0
            tracks.append(PersonTrack(track_id=track_id, bbox=bbox, confidence=confidence))
        return tracks

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("Stable ID tracking needs `pip install ultralytics lap`") from exc
        self._model = YOLO(self.model_name)
