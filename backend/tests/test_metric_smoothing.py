import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osc_sender import OSCSender


class TestPerMetricSmoothing(unittest.TestCase):
    def _sender(self):
        # enabled=False -> prepare values without actually sending UDP
        return OSCSender(enabled=False, alpha=0.5, mode="raw")

    def test_per_metric_alpha_overrides_global(self):
        osc = self._sender()
        osc.set_metric_alpha("energy", 1.0)  # energy: no smoothing
        osc.send_metrics({"energy": 100.0, "sync_velocity": 100.0}, send_keys=set())
        osc.send_metrics({"energy": 0.0, "sync_velocity": 0.0}, send_keys=set())
        # energy alpha=1.0 -> passes raw; sync_velocity uses global 0.5
        self.assertAlmostEqual(osc.last_prepared_metrics["energy"], 0.0)
        self.assertAlmostEqual(osc.last_prepared_metrics["sync_velocity"], 50.0)

    def test_unknown_metric_ignored(self):
        osc = self._sender()
        osc.set_metric_alpha("not_a_metric", 0.5)  # no raise
        self.assertNotIn("not_a_metric", osc.metric_alphas)

    def test_invalid_alpha_raises(self):
        osc = self._sender()
        with self.assertRaises(ValueError):
            osc.set_metric_alpha("energy", 0.0)
        with self.assertRaises(ValueError):
            osc.set_metric_alpha("energy", 1.5)

    def test_status_exposes_metric_alphas(self):
        osc = self._sender()
        osc.set_metric_alpha("jerk", 0.1)
        status = osc.get_status()
        self.assertIn("metric_alphas", status)
        self.assertAlmostEqual(status["metric_alphas"]["jerk"], 0.1)

    def test_fixed_ranges_apply_after_output_ema(self):
        osc = OSCSender(enabled=False, alpha=1.0, mode="fixed")
        osc.set_metric_ranges({"energy": (0.0, 100.0)})
        osc.set_metric_alpha("energy", 0.5)

        osc.send_metrics({"energy": 100.0}, send_keys=set())
        self.assertAlmostEqual(osc.last_prepared_metrics["energy"], 1.0)

        osc.send_metrics({"energy": 0.0}, send_keys=set())
        self.assertAlmostEqual(osc.last_prepared_metrics["energy"], 0.5)

    def test_viewer_defaults_do_not_smooth_sync_correlation(self):
        import osc_viewer

        alphas = osc_viewer.osc_sender.metric_alphas
        self.assertAlmostEqual(alphas["sync_correlation"], 1.0)
        self.assertAlmostEqual(alphas["energy"], 0.5)


if __name__ == "__main__":
    unittest.main()
