import sys, os, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration import (
    CalibrationCollector, save_profile, load_profile, save_presets, load_presets,
)
from osc_sender import OSCSender


class TestCalibrationCollector(unittest.TestCase):
    def test_ranges_use_percentiles(self):
        c = CalibrationCollector()
        for v in range(0, 101):  # 0..100 inclusive
            c.add({"energy": float(v)})
        rng = c.ranges(lo_pct=2.0, hi_pct=98.0)
        self.assertIn("energy", rng)
        lo, hi = rng["energy"]
        self.assertAlmostEqual(lo, 2.0, delta=1.0)   # ~2nd percentile, not 0
        self.assertAlmostEqual(hi, 98.0, delta=1.0)  # ~98th percentile, not 100

    def test_min_samples_gate(self):
        c = CalibrationCollector()
        for _ in range(5):
            c.add({"energy": 1.0})  # too few
        self.assertNotIn("energy", c.ranges(min_samples=10))

    def test_ignores_non_finite_and_missing(self):
        c = CalibrationCollector()
        for _ in range(20):
            c.add({"energy": float("nan"), "torque": 3.0})
        self.assertEqual(c.count("energy"), 0)
        self.assertEqual(c.count("torque"), 20)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "profile.json")
            save_profile(p, {"energy": (1.0, 9.0), "sway": (0.1, 0.4)})
            loaded = load_profile(p)
            self.assertEqual(loaded["energy"], (1.0, 9.0))
            self.assertEqual(loaded["sway"], (0.1, 0.4))

    def test_load_missing_returns_empty(self):
        self.assertEqual(load_profile("/nope/does/not/exist.json"), {})

    def test_presets_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "presets.json")
            presets = {"Dancer A": {"energy": (1.0, 9.0)},
                       "Dancer B": {"sway": (0.0, 0.5)}}
            save_presets(p, presets)
            loaded = load_presets(p)
            self.assertEqual(loaded["Dancer A"]["energy"], (1.0, 9.0))
            self.assertEqual(loaded["Dancer B"]["sway"], (0.0, 0.5))
        self.assertEqual(load_presets("/no/such.json"), {})


class TestFixedNormalize(unittest.TestCase):
    def _sender(self):
        return OSCSender(enabled=False, mode="fixed", alpha=1.0)

    def test_fixed_range_maps_to_0_1(self):
        osc = self._sender()
        osc.set_metric_ranges({"energy": (0.0, 100.0)})
        osc.send_metrics({"energy": 50.0}, send_keys=set())
        self.assertAlmostEqual(osc.last_prepared_metrics["energy"], 0.5)

    def test_fixed_clamps_out_of_range(self):
        osc = self._sender()
        osc.set_metric_ranges({"energy": (0.0, 10.0)})
        osc.send_metrics({"energy": 999.0}, send_keys=set())
        self.assertAlmostEqual(osc.last_prepared_metrics["energy"], 1.0)

    def test_metric_without_range_falls_back_to_adaptive(self):
        osc = self._sender()  # no ranges set
        osc.send_metrics({"expansion": 5.0}, send_keys=set())
        v = osc.last_prepared_metrics["expansion"]
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)

    def test_set_metric_ranges_ignores_invalid(self):
        osc = self._sender()
        osc.set_metric_ranges({"energy": (5.0, 5.0), "bogus": (0.0, 1.0)})
        self.assertNotIn("energy", osc.metric_ranges)  # hi !> lo
        self.assertNotIn("bogus", osc.metric_ranges)

    def test_fixed_is_valid_mode(self):
        osc = OSCSender(enabled=False, mode="fixed")
        self.assertEqual(osc.mode, "fixed")


if __name__ == "__main__":
    unittest.main()
