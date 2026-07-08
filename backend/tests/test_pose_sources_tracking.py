import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from person_tracker import StablePersonTrack  # noqa: E402
from pose_sources import MediaPipePoseSource  # noqa: E402


def _bare_source(**overrides):
    """MediaPipePoseSource without __init__ (no mediapipe / model load)."""
    src = MediaPipePoseSource.__new__(MediaPipePoseSource)
    src.tracking_enabled = True
    src.tracking_selection = "auto_largest"
    src.last_pose_landmarks = None
    src.last_pose_quality = 0.0
    src.last_pose_valid = False
    src.last_tracking = {"enabled": True, "state": "tracking", "count": 0}
    src._last_overlay_points = None
    src._last_overlay_points_by_id = {}
    src._last_stable_tracks = []
    src._last_active_track = None
    for key, value in overrides.items():
        setattr(src, key, value)
    return src


def _track(stable_id, state, bbox=(10, 10, 110, 210)):
    return StablePersonTrack(
        stable_id=stable_id,
        raw_id=stable_id if state == "tracking" else None,
        bbox=bbox,
        confidence=0.9,
        state=state,
        last_seen_at=0.0,
    )


class TestHoldingTracksSkipPose(unittest.TestCase):
    """A 'holding' track's bbox is frozen at the last-seen position; running
    pose on that stale crop reports whoever walks through it under the held
    ID. Pose must only run for tracks with a fresh detection ('tracking')."""

    def _run_estimate(self, tracks, active):
        src = _bare_source()
        detected_ids = []

        def fake_select_crop(frame, timestamp_ms):
            src._last_stable_tracks = tracks
            src._last_active_track = active
            return None

        def fake_detect(input_frame, timestamp_ms, stateless=False):
            return SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])

        def fake_pose_from_result(frame, rect, result):
            return np.zeros((17, 3)), {11: (5, 5)}, 1.0, True, None

        src._select_tracking_crop = fake_select_crop
        src._detect_frame_pose = fake_detect
        src._pose_from_result = fake_pose_from_result

        original_detect = src._detect_frame_pose

        def recording_detect(input_frame, timestamp_ms, stateless=False):
            detected_ids.append(True)
            return original_detect(input_frame, timestamp_ms, stateless)

        src._detect_frame_pose = recording_detect

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        src._estimate_tracked_people(frame, 1000.0, draw=False)
        return src, len(detected_ids)

    def test_pose_runs_only_for_fresh_tracking_state(self):
        tracking = _track(1, "tracking", bbox=(10, 10, 110, 210))
        holding = _track(2, "holding", bbox=(300, 10, 400, 210))
        src, detect_calls = self._run_estimate([tracking, holding], tracking)
        self.assertEqual(detect_calls, 1)  # holding track must not run pose

    def test_holding_active_track_produces_no_pose(self):
        holding = _track(3, "holding")
        src, detect_calls = self._run_estimate([holding], holding)
        self.assertEqual(detect_calls, 0)
        self.assertFalse(src.last_pose_valid)


class TestSinglePersonOverlayGuard(unittest.TestCase):
    """One missed detection must not clear the cached overlay skeleton —
    master held the last pose until the next successful detection."""

    def test_missed_detection_keeps_cached_overlay(self):
        cached_points = {11: (100, 100), 12: (140, 100)}
        src = _bare_source(
            tracking_enabled=False,
            _last_overlay_points=cached_points,
            last_pose_landmarks=["previous"],
        )
        src._detect_frame_pose = lambda frame, ts, stateless=False: SimpleNamespace(
            pose_landmarks=[], pose_world_landmarks=[]
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _frame, h36m = src.estimate(frame, 1000.0, draw=False)

        self.assertIsNone(h36m)
        self.assertEqual(src._last_overlay_points, cached_points)
        self.assertEqual(src.last_pose_landmarks, ["previous"])


if __name__ == "__main__":
    unittest.main()
