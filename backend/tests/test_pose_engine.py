import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_engine import PoseEngine  # noqa: E402


class FakeSource:
    """Minimal pose source: returns a preset skeleton and tracking status."""

    def __init__(self):
        self.last_tracking = {"enabled": True, "state": "tracking", "count": 1, "active_id": 1}
        self.last_pose_quality = 1.0
        self.last_pose_valid = True
        self.h36m = np.random.RandomState(0).rand(17, 3) * 100.0

    def estimate(self, frame, timestamp_ms, draw_overlay=True):
        return frame, self.h36m

    def close(self):
        pass


def _make_engine(source):
    with mock.patch.object(PoseEngine, "_make_source", return_value=source):
        return PoseEngine()


class TestActiveSwitchResetsMetrics(unittest.TestCase):
    """When the tracked (active) dancer changes, the shared metrics engine
    must restart: velocity/jerk across the A->B position jump would
    otherwise be sent over OSC as a huge fake spike."""

    def test_metrics_history_resets_when_active_person_changes(self):
        source = FakeSource()
        engine = _make_engine(source)
        for ts in (0, 33, 66):
            engine.process_frame(None, ts)
        self.assertEqual(len(engine.metrics_engine.positions_history), 3)

        # a different dancer becomes active, standing 2m away
        source.last_tracking = {"enabled": True, "state": "tracking", "count": 2, "active_id": 2}
        source.h36m = source.h36m + 2000.0
        _frame, metrics = engine.process_frame(None, 99)

        self.assertEqual(len(engine.metrics_engine.positions_history), 1)
        self.assertEqual(metrics["energy"], 0.0)
        self.assertEqual(metrics["jerk"], 0.0)

    def test_single_person_mode_keeps_history(self):
        source = FakeSource()
        source.last_tracking = {"enabled": False, "state": "disabled", "count": 0}
        engine = _make_engine(source)
        for ts in (0, 33, 66):
            engine.process_frame(None, ts)
        self.assertEqual(len(engine.metrics_engine.positions_history), 3)


if __name__ == "__main__":
    unittest.main()
