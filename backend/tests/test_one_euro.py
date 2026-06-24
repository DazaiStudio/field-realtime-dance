import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from one_euro import JointSmoother


class TestJointSmoother(unittest.TestCase):
    def test_shape_preserved_and_first_passthrough(self):
        s = JointSmoother()
        x = np.random.default_rng(0).random((17, 3)) * 100
        out = s.filter(x, 0.0)
        self.assertEqual(out.shape, (17, 3))
        np.testing.assert_allclose(out, x)  # first sample passes through

    def test_constant_signal_stays_constant(self):
        s = JointSmoother()
        x = np.full((17, 3), 42.0)
        t = 0.0
        for _ in range(10):
            out = s.filter(x, t)
            t += 1000.0 / 30.0
        np.testing.assert_allclose(out, x, atol=1e-9)

    def test_reduces_high_frequency_noise(self):
        # A still joint with +/- jitter should come out much steadier.
        s = JointSmoother(min_cutoff=1.0, beta=0.0)
        rng = np.random.default_rng(1)
        base = np.full((17, 3), 500.0)
        ins, outs = [], []
        t = 0.0
        for i in range(120):
            x = base + rng.normal(0, 5.0, (17, 3))
            o = s.filter(x, t)
            if i > 10:  # skip warm-up
                ins.append(x.copy())
                outs.append(o.copy())
            t += 1000.0 / 30.0
        in_std = np.std(np.array(ins))
        out_std = np.std(np.array(outs))
        self.assertLess(out_std, in_std * 0.5)  # at least halve the jitter

    def test_tracks_fast_motion_with_low_lag(self):
        # A fast ramp should not lag too far behind (adaptive cutoff opens up).
        s = JointSmoother(min_cutoff=1.0, beta=0.02)
        t = 0.0
        x = np.zeros((17, 3))
        out = None
        for i in range(60):
            x = np.full((17, 3), float(i) * 50.0)  # 50mm/frame ramp
            out = s.filter(x, t)
            t += 1000.0 / 30.0
        # After a sustained fast ramp the output should be close to the input.
        self.assertGreater(float(out.mean()), float(x.mean()) * 0.7)

    def test_reset_clears_state(self):
        s = JointSmoother()
        s.filter(np.ones((17, 3)), 0.0)
        s.reset()
        x = np.full((17, 3), 9.0)
        np.testing.assert_allclose(s.filter(x, 0.0), x)


if __name__ == "__main__":
    unittest.main()
