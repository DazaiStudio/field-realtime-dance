import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from group_extent import GroupExtentTracker
from group_smoothing import GroupSmoother
from person_tracker import PersonTrack


class _FakeTracker:
    """Stands in for UltralyticsPersonTracker: detections, no identity."""

    def __init__(self, detections):
        self.detections = detections
        self.load_error = None
        self.calls = 0

    def is_ready(self):
        return True

    def start_preload(self, retry_error=False):
        pass

    def track(self, frame):
        self.calls += 1
        return self.detections


def make_source(detections, group_enabled=True):
    """A bare object carrying just the group-box collaborators, so the test
    does not need MediaPipe's model files or a camera."""
    from pose_sources import MediaPipePoseSource

    src = MediaPipePoseSource.__new__(MediaPipePoseSource)
    src.tracking_enabled = False
    src.group_extent_enabled = group_enabled
    src.group_tracker = GroupExtentTracker(hold_seconds=1.0, max_people=4)
    # Off: these tests assert on the measured box, not on a filtered one.
    src.group_smoother = GroupSmoother(0.0)
    src.group_extent_is_metric = False
    src.last_group_extent = None
    src._last_group_box = None
    src.last_group_box_norm = None
    src.tracker = _FakeTracker(detections)
    src.tracking_selection = "auto_largest"
    src._last_stable_tracks = []
    src._last_active_track = None
    return src


class TestGroupBoxWithoutStableId(unittest.TestCase):
    """The whole point: 'where is the group' must not require per-dancer ids."""

    def setUp(self):
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.detections = [
            PersonTrack(track_id=1, bbox=(100.0, 200.0, 300.0, 600.0), confidence=0.9),
            PersonTrack(track_id=2, bbox=(700.0, 180.0, 900.0, 620.0), confidence=0.9),
        ]

    def test_group_box_is_produced_with_stable_id_off(self):
        src = make_source(self.detections)
        src._detect_group_only(self.frame, timestamp_ms=0.0)
        self.assertIsNotNone(src._last_group_box)
        self.assertEqual(src._last_group_box, (100.0, 180.0, 900.0, 620.0))
        self.assertEqual(src.last_group_extent.count, 2)

    def test_registry_is_never_touched(self):
        src = make_source(self.detections)
        src._detect_group_only(self.frame, timestamp_ms=0.0)
        # No stable ids were assigned: identity was never computed.
        self.assertEqual(src._last_stable_tracks, [])
        self.assertIsNone(src._last_active_track)

    def test_normalised_box_is_frame_relative(self):
        src = make_source(self.detections)
        src._detect_group_only(self.frame, timestamp_ms=0.0)
        norm = src.last_group_box_norm
        self.assertAlmostEqual(norm["x1"], 100.0 / 1280.0, places=6)
        self.assertAlmostEqual(norm["y2"], 620.0 / 720.0, places=6)

    def test_dropout_still_holds_without_stable_id(self):
        src = make_source(self.detections)
        src._detect_group_only(self.frame, timestamp_ms=0.0)
        src.tracker.detections = [self.detections[0]]
        src._detect_group_only(self.frame, timestamp_ms=200.0)
        self.assertTrue(src.last_group_extent.held)
        self.assertEqual(src.last_group_extent.count, 2)

    def test_disabled_does_not_run_the_detector(self):
        src = make_source(self.detections, group_enabled=False)
        src._detect_group_only(self.frame, timestamp_ms=0.0)
        self.assertEqual(src.tracker.calls, 0)
        self.assertIsNone(src._last_group_box)

    def test_no_detections_clears_the_box(self):
        src = make_source([])
        src._detect_group_only(self.frame, timestamp_ms=0.0)
        self.assertIsNone(src._last_group_box)
        self.assertEqual(src.last_group_extent.count, 0)


if __name__ == "__main__":
    unittest.main()
