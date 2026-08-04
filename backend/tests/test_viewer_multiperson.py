import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import osc_viewer  # noqa: E402
from osc_sender import METRIC_NAMES  # noqa: E402


def _metrics(energy):
    values = {name: 0.0 for name in METRIC_NAMES}
    values["energy"] = energy
    return values


class TestViewerMultiPersonOutput(unittest.TestCase):
    def setUp(self):
        sender = osc_viewer.osc_sender
        self._saved = (sender.enabled, sender.mode, dict(sender.metric_alphas))
        sender.enabled = False  # never hit the network from tests
        sender.configure(mode="raw")
        sender.metric_alphas.clear()
        sender.reset_state()
        self._saved_tracking = osc_viewer.processing_state.get("tracking")

    def tearDown(self):
        sender = osc_viewer.osc_sender
        enabled, mode, alphas = self._saved
        sender.enabled = enabled
        sender.configure(mode=mode)
        sender.metric_alphas.clear()
        sender.metric_alphas.update(alphas)
        sender.reset_state()
        osc_viewer.processing_state["tracking"] = self._saved_tracking
        osc_viewer.processing_state["latest_raw_metrics_by_id"] = {}

    def test_all_persons_prepared_and_display_follows_active(self):
        tracking = {
            "enabled": True,
            "state": "tracking",
            "count": 2,
            "active_id": 2,
            "tracks": [
                {"stable_id": 1, "state": "tracking"},
                {"stable_id": 2, "state": "tracking"},
            ],
        }
        osc_viewer.set_analysis_result(
            _metrics(2.0),
            timestamp_ms=1000,
            pose_valid=True,
            tracking=tracking,
            metrics_by_id={1: _metrics(1.0), 2: _metrics(2.0)},
        )
        sender = osc_viewer.osc_sender
        self.assertEqual(set(sender.last_prepared_by_id), {1, 2})
        self.assertAlmostEqual(sender.last_prepared_by_id[1]["energy"], 1.0)
        # UI display follows the active dancer (id 2)
        self.assertAlmostEqual(
            osc_viewer.processing_state["latest_metrics"]["energy"], 2.0
        )

    def test_every_dancer_is_published_for_the_ui(self):
        # The panels showed only the active dancer while OSC was already
        # sending all of them, so a second dancer's stream could die unnoticed.
        tracking = {
            "enabled": True, "state": "tracking", "count": 2, "active_id": 5,
            "tracks": [{"stable_id": 4, "state": "tracking"},
                       {"stable_id": 5, "state": "tracking"}],
        }
        osc_viewer.set_analysis_result(
            _metrics(2.0),
            timestamp_ms=4000,
            pose_valid=True,
            tracking=tracking,
            metrics_by_id={4: _metrics(1.0), 5: _metrics(2.0)},
        )
        by_id = osc_viewer.processing_state["latest_metrics_by_id"]
        self.assertEqual(set(by_id), {4, 5})
        self.assertAlmostEqual(by_id[4]["energy"], 1.0)
        self.assertAlmostEqual(by_id[5]["energy"], 2.0)

    def test_ui_metrics_by_id_reach_the_state_payload(self):
        osc_viewer.set_analysis_result(
            _metrics(7.0), timestamp_ms=5000, pose_valid=True,
            metrics_by_id={3: _metrics(7.0)},
        )
        processing = osc_viewer.state_payload()["processing"]
        self.assertAlmostEqual(processing["latest_metrics_by_id"][3]["energy"], 7.0)

    def test_stale_dancers_drop_out_of_the_ui_list(self):
        osc_viewer.set_analysis_result(
            _metrics(1.0), timestamp_ms=6000, pose_valid=True,
            tracking={"enabled": True, "state": "tracking", "count": 2, "active_id": 1,
                      "tracks": [{"stable_id": 1, "state": "tracking"},
                                 {"stable_id": 2, "state": "tracking"}]},
            metrics_by_id={1: _metrics(1.0), 2: _metrics(2.0)},
        )
        self.assertEqual(set(osc_viewer.processing_state["latest_metrics_by_id"]), {1, 2})
        # Dancer 2 leaves for good: their panel row must not linger forever.
        osc_viewer.set_analysis_result(
            _metrics(1.0), timestamp_ms=7000, pose_valid=True,
            tracking={"enabled": True, "state": "tracking", "count": 1, "active_id": 1,
                      "tracks": [{"stable_id": 1, "state": "tracking"}]},
            metrics_by_id={1: _metrics(1.0)},
        )
        self.assertEqual(set(osc_viewer.processing_state["latest_metrics_by_id"]), {1})

    def test_single_person_display_ignores_registry_active_id(self):
        # Kinect keeps assigning stable ids in single-person mode, but the
        # single-person stream is always published as id 1 -- the UI panels and
        # calibration must not follow an active_id that carries no metrics.
        tracking = {
            "enabled": False,
            "state": "tracking",
            "count": 1,
            "active_id": 2,
            "tracks": [{"stable_id": 2, "state": "tracking"}],
        }
        osc_viewer.set_analysis_result(
            _metrics(4.0),
            timestamp_ms=3000,
            pose_valid=True,
            tracking=tracking,
        )
        self.assertAlmostEqual(
            osc_viewer.processing_state["latest_metrics"]["energy"], 4.0
        )

    def test_single_person_fallback_prepares_id_1(self):
        osc_viewer.set_analysis_result(
            _metrics(3.0),
            timestamp_ms=2000,
            pose_valid=True,
        )
        sender = osc_viewer.osc_sender
        self.assertEqual(set(sender.last_prepared_by_id), {1})
        self.assertAlmostEqual(
            osc_viewer.processing_state["latest_metrics"]["energy"], 3.0
        )


if __name__ == "__main__":
    unittest.main()
