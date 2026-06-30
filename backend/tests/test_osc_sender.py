"""Unit tests for the OSC output contract (addresses, normalize, smoothing).

Run from the repo root:
    python -m unittest discover -s backend/tests -t backend
or from backend/:
    python -m unittest tests.test_osc_sender
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osc_sender import METRIC_NAMES, OSCSender


def make_sender(**kwargs) -> OSCSender:
    kwargs.setdefault("enabled", False)  # never hit the network in tests
    kwargs.setdefault("alpha", 1.0)      # no smoothing unless a test wants it
    return OSCSender(**kwargs)


class TestAddresses(unittest.TestCase):
    def test_frozen_addresses(self):
        sender = make_sender()
        expected = {
            "energy": "/field/energy",
            "sync_velocity": "/field/sync_vel",
            "sync_correlation": "/field/sync_corr",
            "expansion": "/field/expansion",
            "curvature": "/field/curvature",
            "height": "/field/height",
            "sway": "/field/sway",
            "torque": "/field/torque",
            "jerk": "/field/jerk",
        }
        for name in METRIC_NAMES:
            self.assertEqual(sender.metric_address(name), expected[name])

    def test_namespace_normalization(self):
        self.assertEqual(make_sender(namespace="field").namespace, "/field")
        self.assertEqual(make_sender(namespace="/custom/").namespace, "/custom")
        empty = make_sender(namespace="")
        self.assertEqual(empty.namespace, "")
        self.assertEqual(empty.metric_address("energy"), "/energy")


class TestNormalize(unittest.TestCase):
    def test_sync_correlation_stays_bipolar_in_normalize_mode(self):
        sender = make_sender(mode="normalize")
        self.assertAlmostEqual(sender._prepare_value("sync_correlation", -1.0), -1.0)
        self.assertAlmostEqual(sender._prepare_value("sync_correlation", 0.0), 0.0)
        self.assertAlmostEqual(sender._prepare_value("sync_correlation", 1.0), 1.0)

    def test_unbounded_metric_tracks_peak(self):
        sender = make_sender(mode="normalize")
        self.assertAlmostEqual(sender._prepare_value("energy", 10.0), 1.0)
        # peak decays slightly (0.995), so 5.0 lands just above 0.5
        second = sender._prepare_value("energy", 5.0)
        self.assertAlmostEqual(second, 5.0 / 9.95, places=6)

    def test_height_adaptive_range_handles_negative_values(self):
        sender = make_sender(mode="normalize")
        # first sample: degenerate range -> neutral midpoint
        self.assertAlmostEqual(sender._prepare_value("height", -0.2), 0.5)
        high = sender._prepare_value("height", 0.3)   # establishes the range
        low = sender._prepare_value("height", -0.2)
        mid = sender._prepare_value("height", 0.05)
        self.assertAlmostEqual(high, 1.0, places=3)
        self.assertAlmostEqual(low, 0.0, places=3)
        self.assertTrue(0.0 < mid < 1.0)
        self.assertLess(low, mid)
        self.assertLess(mid, high)

    def test_sway_adaptive_range_stays_in_0_1(self):
        sender = make_sender(mode="normalize")
        for value in (0.0, 0.05, 0.17, 0.4, 0.02):
            out = sender._prepare_value("sway", value)
            self.assertGreaterEqual(out, 0.0)
            self.assertLessEqual(out, 1.0)

    def test_raw_mode_passes_values_through(self):
        sender = make_sender(mode="raw")
        self.assertAlmostEqual(sender._prepare_value("height", -0.2), -0.2)
        self.assertAlmostEqual(sender._prepare_value("jerk", 1.5e9), 1.5e9)

    def test_non_finite_values_are_dropped(self):
        sender = make_sender(mode="raw")
        self.assertIsNone(sender._prepare_value("energy", float("nan")))
        self.assertIsNone(sender._prepare_value("energy", math.inf))
        self.assertIsNone(sender._prepare_value("energy", "not a number"))


class TestSmoothing(unittest.TestCase):
    def test_alpha_below_one_smooths(self):
        sender = make_sender(mode="raw", alpha=0.5)
        self.assertAlmostEqual(sender._prepare_value("energy", 0.0), 0.0)
        self.assertAlmostEqual(sender._prepare_value("energy", 1.0), 0.5)
        self.assertAlmostEqual(sender._prepare_value("energy", 1.0), 0.75)

    def test_alpha_one_disables_smoothing(self):
        sender = make_sender(mode="raw", alpha=1.0)
        self.assertAlmostEqual(sender._prepare_value("energy", 0.0), 0.0)
        self.assertAlmostEqual(sender._prepare_value("energy", 1.0), 1.0)

    def test_reset_state_clears_history(self):
        sender = make_sender(mode="normalize", alpha=0.5)
        sender._prepare_value("energy", 10.0)
        sender._prepare_value("height", -0.2)
        sender.reset_state()
        self.assertEqual(sender._smoothed, {})
        self.assertEqual(sender._peaks, {})
        self.assertEqual(sender._ranges, {})


class TestSendMetrics(unittest.TestCase):
    def test_disabled_sender_prepares_but_does_not_send(self):
        sender = make_sender(mode="raw")
        sent = sender.send_metrics({name: 1.0 for name in METRIC_NAMES})
        self.assertEqual(sent, [])
        self.assertEqual(set(sender.last_prepared_metrics), set(METRIC_NAMES))

    def test_unknown_keys_are_ignored(self):
        sender = make_sender(mode="raw")
        sender.send_metrics({"energy": 1.0, "bogus": 2.0})
        self.assertEqual(set(sender.last_prepared_metrics), {"energy"})

    def test_multiple_targets_receive_same_metric_value(self):
        sender = make_sender(enabled=True, mode="raw")
        sender.configure_targets([
            {"id": "sound", "name": "Sound", "host": "127.0.0.1", "port": 9000, "enabled": True},
            {"id": "visuals", "name": "Visuals", "host": "127.0.0.1", "port": 9001, "enabled": True},
            {"id": "off", "name": "Off", "host": "127.0.0.1", "port": 9002, "enabled": False},
        ])

        sent_by_target = {}

        class FakeClient:
            def __init__(self, target_id):
                self.target_id = target_id

            def send_message(self, address, value):
                sent_by_target.setdefault(self.target_id, []).append((address, value))

        for target in sender.targets:
            target.client.close()
            target.client = FakeClient(target.id)

        sent = sender.send_metrics({"energy": 2.5}, send_keys={"energy"})
        self.assertEqual({item["target"] for item in sent}, {"sound", "visuals"})
        self.assertEqual(sent_by_target["sound"], [("/field/energy", 2.5)])
        self.assertEqual(sent_by_target["visuals"], [("/field/energy", 2.5)])
        self.assertNotIn("off", sent_by_target)

    def test_status_exposes_osc_targets(self):
        sender = make_sender()
        sender.configure_targets([
            {"id": "sound", "name": "Sound", "host": "192.168.1.21", "port": 9000, "enabled": True},
            {"id": "broadcast", "name": "Broadcast", "host": "192.168.1.255", "port": 9000, "enabled": False, "broadcast": True},
        ])
        status = sender.get_status()
        self.assertEqual(status["host"], "192.168.1.21")
        self.assertEqual(status["port"], 9000)
        self.assertEqual([target["id"] for target in status["targets"]], ["sound", "broadcast"])
        self.assertTrue(status["targets"][1]["broadcast"])


if __name__ == "__main__":
    unittest.main()
