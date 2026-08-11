import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pose_sources
from pose_sources import MediaPipePoseSource


class TestMediaPipeSourceConstruction(unittest.TestCase):
    """Constructs the real __init__ with only the heavy I/O stubbed.

    The group-box tests build their source with __new__, which skips __init__
    entirely -- so they cannot catch an attribute being read before it is
    assigned. That bug shipped once: the YOLO preload check consulted
    group_extent_enabled fifteen lines before it existed, __init__ raised
    AttributeError, the engine was never built, and the stream ran with zero
    analysis frames -- no skeleton, no box, and no error anywhere in the UI.
    """

    def _construct(self, **kwargs):
        with mock.patch.object(MediaPipePoseSource, "_ensure_model", lambda self: None), \
             mock.patch.object(MediaPipePoseSource, "_create_landmarker",
                               lambda self, mode: object()), \
             mock.patch.object(MediaPipePoseSource, "_ensure_tracker",
                               lambda self: mock.MagicMock()):
            return MediaPipePoseSource("pose_landmarker_lite.task", **kwargs)

    def test_constructs_with_group_box_on_and_stable_id_off(self):
        src = self._construct(tracking_enabled=False, group_extent_enabled=True)
        self.assertTrue(src.group_extent_enabled)
        self.assertFalse(src.tracking_enabled)

    def test_constructs_with_everything_off(self):
        src = self._construct()
        self.assertFalse(src.group_extent_enabled)

    def test_constructs_with_both_on(self):
        src = self._construct(tracking_enabled=True, group_extent_enabled=True)
        self.assertTrue(src.group_extent_enabled)
        self.assertTrue(src.tracking_enabled)

    def test_group_attributes_all_exist_after_init(self):
        src = self._construct(group_extent_enabled=True)
        for attr in ("group_extent_enabled", "group_tracker", "group_smoother",
                     "group_extent_is_metric", "last_group_extent",
                     "_last_group_box", "last_group_box_norm"):
            self.assertTrue(hasattr(src, attr), f"missing {attr}")

    def test_mediapipe_never_claims_metric_units(self):
        """Guards the OSC split: /field/group/width must never carry screen
        fractions on an address a patch reads as metres."""
        src = self._construct(group_extent_enabled=True)
        self.assertFalse(src.group_extent_is_metric)

    def test_preload_runs_for_group_box_without_stable_id(self):
        with mock.patch.object(MediaPipePoseSource, "_ensure_model", lambda self: None), \
             mock.patch.object(MediaPipePoseSource, "_create_landmarker",
                               lambda self, mode: object()):
            tracker = mock.MagicMock()
            with mock.patch.object(MediaPipePoseSource, "_ensure_tracker",
                                   lambda self: tracker):
                MediaPipePoseSource("pose_landmarker_lite.task",
                                    tracking_enabled=False, group_extent_enabled=True)
            tracker.start_preload.assert_called()



class TestGroupBoxReachesTheSinglePersonPath(unittest.TestCase):
    """estimate() branches to the single-person path when Stable ID is off, and
    that branch never touches _select_tracking_crop. The group detector was
    first wired into _select_tracking_crop, so with Stable ID off it sat in
    unreachable code: skeleton drew fine, box never appeared, no error anywhere.
    These call the real estimate() so the wiring itself is under test.
    """

    def _source(self, group_enabled=True):
        with mock.patch.object(MediaPipePoseSource, "_ensure_model", lambda self: None), \
             mock.patch.object(MediaPipePoseSource, "_create_landmarker",
                               lambda self, mode: object()), \
             mock.patch.object(MediaPipePoseSource, "_ensure_tracker",
                               lambda self: mock.MagicMock()):
            src = MediaPipePoseSource("pose_landmarker_lite.task",
                                      tracking_enabled=False,
                                      group_extent_enabled=group_enabled)
        # Pose detection is irrelevant here: stub it to "no person found" so the
        # test isolates the group-box wiring.
        src._detect_frame_pose = lambda *a, **k: object()
        src._pose_from_result = lambda *a, **k: (None, None, 0.0, False, None)
        return src

    def _ready_tracker(self, detections):
        tracker = mock.MagicMock()
        tracker.is_ready.return_value = True
        tracker.load_error = None
        tracker.track.return_value = detections
        return tracker

    def test_estimate_produces_a_group_box_with_stable_id_off(self):
        from person_tracker import PersonTrack
        src = self._source()
        src.tracker = self._ready_tracker([
            PersonTrack(track_id=1, bbox=(100.0, 200.0, 300.0, 600.0), confidence=0.9),
            PersonTrack(track_id=2, bbox=(700.0, 180.0, 900.0, 620.0), confidence=0.9),
        ])
        src._ensure_tracker = lambda: src.tracker
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        src.estimate(frame, timestamp_ms=0.0, draw=True)
        self.assertIsNotNone(src._last_group_box)
        self.assertEqual(src._last_group_box, (100.0, 180.0, 900.0, 620.0))
        self.assertEqual(src.last_group_extent.count, 2)

    def test_estimate_actually_draws_the_box(self):
        from person_tracker import PersonTrack
        src = self._source()
        src.tracker = self._ready_tracker(
            [PersonTrack(track_id=1, bbox=(100.0, 200.0, 300.0, 600.0), confidence=0.9)])
        src._ensure_tracker = lambda: src.tracker
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        src.estimate(frame, timestamp_ms=0.0, draw=True)
        self.assertGreater(int(np.count_nonzero(frame.any(axis=2))), 0)

    def test_group_box_disabled_leaves_the_frame_clean(self):
        src = self._source(group_enabled=False)
        src.tracker = self._ready_tracker([])
        src._ensure_tracker = lambda: src.tracker
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        src.estimate(frame, timestamp_ms=0.0, draw=True)
        self.assertEqual(int(np.count_nonzero(frame.any(axis=2))), 0)
        self.assertIsNone(src._last_group_box)


if __name__ == "__main__":
    unittest.main()
