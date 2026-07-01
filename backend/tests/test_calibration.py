import sys, os, tempfile, time, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration import (
    CalibrationCollector, load_profile, load_presets, normalize_presets, save_profile, save_presets,
)
from osc_sender import OSCSender


class TestCalibrationCollector(unittest.TestCase):
    def test_ranges_use_percentiles(self):
        c = CalibrationCollector()
        for v in range(0, 101):  # 0..100 inclusive
            c.add({"energy": float(v)})
        rng = c.ranges()
        self.assertIn("energy", rng)
        lo, hi = rng["energy"]
        self.assertAlmostEqual(lo, 1.0, delta=1.0)   # ~1st percentile, not 0
        self.assertAlmostEqual(hi, 99.0, delta=1.0)  # ~99th percentile, not 100

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

    def test_normalize_presets_accepts_export_wrapper(self):
        data = {
            "version": 1,
            "presets": {
                "stage": {
                    "energy": [1.0, 10.0],
                    "bogus": [0.0, 1.0],
                    "height": [5.0, 5.0],
                }
            },
        }
        presets = normalize_presets(data)
        self.assertEqual(presets["stage"]["energy"], (1.0, 10.0))
        self.assertNotIn("bogus", presets["stage"])
        self.assertNotIn("height", presets["stage"])

    def test_normalize_presets_accepts_single_profile_ranges(self):
        presets = normalize_presets({"ranges": {"sway": [0.2, 0.8]}}, default_name="solo")
        self.assertEqual(presets["solo"]["sway"], (0.2, 0.8))


class TestFixedNormalize(unittest.TestCase):
    def _sender(self):
        return OSCSender(enabled=False, mode="fixed", alpha=1.0)

    def test_fixed_range_maps_to_0_1(self):
        osc = self._sender()
        osc.set_metric_ranges({"expansion": (0.0, 100.0)})
        osc.send_metrics({"expansion": 50.0}, send_keys=set())
        self.assertAlmostEqual(osc.last_prepared_metrics["expansion"], 0.5)

    def test_fixed_energy_lifts_mid_range_response(self):
        osc = self._sender()
        osc.set_metric_ranges({"energy": (0.0, 100.0)})
        osc.send_metrics({"energy": 25.0}, send_keys=set())
        value = osc.last_prepared_metrics["energy"]
        self.assertGreater(value, 0.49)
        self.assertLess(value, 0.51)

    def test_fixed_clamps_out_of_range(self):
        osc = self._sender()
        osc.set_metric_ranges({"energy": (0.0, 10.0)})
        osc.send_metrics({"energy": 999.0}, send_keys=set())
        self.assertAlmostEqual(osc.last_prepared_metrics["energy"], 1.0)

    def test_fixed_jerk_uses_log_range(self):
        osc = self._sender()
        osc.set_metric_ranges({"jerk": (10.0, 100000.0)})
        osc.send_metrics({"jerk": 1000.0}, send_keys=set())
        value = osc.last_prepared_metrics["jerk"]
        self.assertGreater(value, 0.7)
        self.assertLess(value, 0.85)

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


class TestViewerCalibrationFlow(unittest.TestCase):
    def test_calibration_collects_output_ema_values(self):
        import osc_viewer

        original_state = dict(osc_viewer.calibration_state)
        original_mode = osc_viewer.osc_sender.mode
        original_alpha = osc_viewer.osc_sender.alpha
        original_metric_alphas = dict(osc_viewer.osc_sender.metric_alphas)
        try:
            osc_viewer.calibration_collector.reset()
            osc_viewer.osc_sender.configure(mode="raw", alpha=1.0)
            osc_viewer.osc_sender.reset_state()
            for name in osc_viewer.METRIC_NAMES:
                osc_viewer.osc_sender.set_metric_alpha(name, 1.0)
            osc_viewer.osc_sender.set_metric_alpha("energy", 0.5)
            osc_viewer.calibration_state["active"] = True
            osc_viewer.calibration_state["countdown_until"] = None
            osc_viewer.calibration_state["sample_count"] = 0
            osc_viewer.calibration_state["skipped_frames"] = 0

            frame_a = {name: 1.0 for name in osc_viewer.METRIC_NAMES}
            frame_b = {name: 1.0 for name in osc_viewer.METRIC_NAMES}
            frame_a["energy"] = 100.0
            frame_b["energy"] = 0.0

            osc_viewer.set_analysis_result(frame_a, timestamp_ms=1000, pose_valid=True)
            osc_viewer.set_analysis_result(frame_b, timestamp_ms=1100, pose_valid=True)

            energy_samples = osc_viewer.calibration_collector._samples["energy"][-2:]
            self.assertEqual(len(energy_samples), 2)
            self.assertAlmostEqual(energy_samples[0], 100.0)
            self.assertAlmostEqual(energy_samples[1], 50.0)
        finally:
            osc_viewer.calibration_state.update(original_state)
            osc_viewer.calibration_collector.reset()
            osc_viewer.osc_sender.metric_alphas.clear()
            osc_viewer.osc_sender.metric_alphas.update(original_metric_alphas)
            osc_viewer.osc_sender.configure(mode=original_mode, alpha=original_alpha)
            osc_viewer.osc_sender.reset_state()

    def test_calibration_countdown_does_not_collect_samples(self):
        import osc_viewer

        original_state = dict(osc_viewer.calibration_state)
        try:
            osc_viewer.calibration_collector.reset()
            osc_viewer.osc_sender.configure(mode="raw")
            osc_viewer.osc_sender.reset_state()
            osc_viewer.calibration_state["active"] = True
            osc_viewer.calibration_state["countdown_until"] = time.time() + 10.0
            osc_viewer.calibration_state["sample_count"] = 0
            osc_viewer.calibration_state["skipped_frames"] = 0

            frame = {name: 1.0 for name in osc_viewer.METRIC_NAMES}
            frame["energy"] = 100.0
            osc_viewer.set_analysis_result(frame, timestamp_ms=1000, pose_valid=True)

            self.assertEqual(osc_viewer.calibration_collector.count("energy"), 0)
            self.assertEqual(osc_viewer.calibration_state["sample_count"], 0)
        finally:
            osc_viewer.calibration_state.update(original_state)
            osc_viewer.calibration_collector.reset()
            osc_viewer.osc_sender.reset_state()

    def test_calibration_skips_invalid_pose_frames(self):
        import osc_viewer

        original_state = dict(osc_viewer.calibration_state)
        try:
            osc_viewer.calibration_collector.reset()
            osc_viewer.osc_sender.configure(mode="raw")
            osc_viewer.osc_sender.reset_state()
            osc_viewer.calibration_state["active"] = True
            osc_viewer.calibration_state["countdown_until"] = time.time() - 1.0
            osc_viewer.calibration_state["sample_count"] = 0
            osc_viewer.calibration_state["skipped_frames"] = 0

            frame = {name: 1.0 for name in osc_viewer.METRIC_NAMES}
            frame["energy"] = 100.0
            osc_viewer.set_analysis_result(frame, timestamp_ms=1000, pose_valid=False)

            self.assertEqual(osc_viewer.calibration_collector.count("energy"), 0)
            self.assertEqual(osc_viewer.calibration_state["sample_count"], 0)
            self.assertEqual(osc_viewer.calibration_state["skipped_frames"], 1)
        finally:
            osc_viewer.calibration_state.update(original_state)
            osc_viewer.calibration_collector.reset()
            osc_viewer.osc_sender.reset_state()


if __name__ == "__main__":
    unittest.main()
