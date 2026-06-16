import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dance_metrics import DanceMetricsEngine


def _square_positions():
    pos = np.zeros((17, 3))
    pos[:, 0] = np.linspace(0, 100, 17)
    pos[:, 1] = np.linspace(0, 100, 17)
    pos[5, :2] = (0, 100)
    pos[6, :2] = (100, 0)
    return pos


class TestMetrics2D(unittest.TestCase):
    def test_expansion_2d_is_positive(self):
        eng = DanceMetricsEngine(fps=30, is_3d=False)
        area = eng._calculate_expansion(_square_positions())
        self.assertGreater(area, 0.0)

    def test_expansion_defaults_to_3d(self):
        eng = DanceMetricsEngine(fps=30)
        self.assertTrue(eng.is_3d)

    def test_sway_is_horizontal_when_z_zero(self):
        eng = DanceMetricsEngine(fps=30, is_3d=False)
        pos = np.zeros((17, 3))
        # Ankles (BOS) stay at x=0; every other joint moves to x=5, so COM
        # shifts in x while z stays 0. Sway should equal |com_x - bos_x|,
        # i.e. plain horizontal (x) distance, with no z contribution.
        pos[:, 0] = 5.0
        pos[3, 0] = 0.0  # R_ANKLE
        pos[6, 0] = 0.0  # L_ANKLE

        from constants import MASS_WEIGHTS, L_ANKLE, R_ANKLE
        com_x = float(np.sum(pos[:, 0] * MASS_WEIGHTS))
        bos_x = float((pos[L_ANKLE, 0] + pos[R_ANKLE, 0]) / 2.0)
        expected_sway = abs(com_x - bos_x) / 1000.0

        _h, sway = eng._calculate_stability(pos)
        self.assertAlmostEqual(sway, expected_sway, places=6)


if __name__ == "__main__":
    unittest.main()
