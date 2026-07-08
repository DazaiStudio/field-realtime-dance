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


def _tracking_payload(active_id, ids):
    return {
        "enabled": True,
        "state": "tracking",
        "count": len(ids),
        "active_id": active_id,
        "tracks": [{"stable_id": pid, "state": "tracking"} for pid in ids],
    }


def _make_tracked_engine(source):
    with mock.patch.object(PoseEngine, "_make_source", return_value=source):
        return PoseEngine(tracking_enabled=True)


class TestPerPersonMetrics(unittest.TestCase):
    """With Stable ID tracking, every tracked dancer feeds their own
    DanceMetricsEngine and appears in last_metrics_by_id."""

    def _multi_source(self):
        source = FakeSource()
        pose_a = np.random.RandomState(1).rand(17, 3) * 100.0
        pose_b = pose_a + 1500.0  # a different dancer across the stage
        source.last_h36m_by_id = {1: pose_a, 2: pose_b}
        source.last_tracking = _tracking_payload(1, [1, 2])
        source.h36m = pose_a
        return source

    def test_each_person_gets_their_own_metrics(self):
        source = self._multi_source()
        engine = _make_tracked_engine(source)
        for ts in (0, 33, 66):
            _frame, display = engine.process_frame(None, ts)
        self.assertEqual(set(engine.last_metrics_by_id), {1, 2})
        self.assertEqual(set(engine.last_skeletons_by_id), {1, 2})
        # both dancers hold still -> neither stream shows fake motion
        self.assertAlmostEqual(engine.last_metrics_by_id[1]["energy"], 0.0, places=6)
        self.assertAlmostEqual(engine.last_metrics_by_id[2]["energy"], 0.0, places=6)
        # display metrics follow the active person
        self.assertEqual(display, engine.last_metrics_by_id[1])

    def test_departed_person_engine_is_pruned(self):
        source = self._multi_source()
        engine = _make_tracked_engine(source)
        engine.process_frame(None, 0)
        self.assertEqual(set(engine._metric_engines), {1, 2})

        source.last_h36m_by_id = {1: source.last_h36m_by_id[1]}
        source.last_tracking = _tracking_payload(1, [1])
        engine.process_frame(None, 33)
        self.assertEqual(set(engine._metric_engines), {1})
        self.assertEqual(set(engine.last_metrics_by_id), {1})

    def test_single_person_fallback_reports_id_1(self):
        source = FakeSource()
        source.last_tracking = {"enabled": False, "state": "disabled", "count": 0}
        engine = _make_engine(source)
        engine.process_frame(None, 0)
        self.assertEqual(set(engine.last_metrics_by_id), {1})


if __name__ == "__main__":
    unittest.main()
