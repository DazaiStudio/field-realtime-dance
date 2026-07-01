import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constants import (  # noqa: E402
    HEAD,
    L_ANKLE,
    L_ELBOW,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    NECK,
    PELVIS,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
    SPINE,
    THORAX,
)
from dance_metrics import DanceMetricsEngine  # noqa: E402


def _pose(pelvis_y: float, torso: float = 700.0) -> np.ndarray:
    p = np.zeros((17, 3), dtype=float)
    p[PELVIS] = [0, pelvis_y, 0]
    p[R_HIP] = [-90, pelvis_y, 0]
    p[L_HIP] = [90, pelvis_y, 0]
    p[R_KNEE] = [-110, pelvis_y * 0.55, 20]
    p[L_KNEE] = [110, pelvis_y * 0.55, 20]
    p[R_ANKLE] = [-120, 0, 0]
    p[L_ANKLE] = [120, 0, 0]
    p[SPINE] = [0, pelvis_y + torso * 0.35, 0]
    p[THORAX] = [0, pelvis_y + torso * 0.65, 0]
    p[NECK] = [0, pelvis_y + torso * 0.82, 0]
    p[HEAD] = [0, pelvis_y + torso, 0]
    p[R_SHOULDER] = [-180, pelvis_y + torso * 0.65, 0]
    p[L_SHOULDER] = [180, pelvis_y + torso * 0.65, 0]
    p[R_ELBOW] = [-260, pelvis_y + torso * 0.35, 0]
    p[L_ELBOW] = [260, pelvis_y + torso * 0.35, 0]
    p[R_WRIST] = [-300, pelvis_y + torso * 0.15, 0]
    p[L_WRIST] = [300, pelvis_y + torso * 0.15, 0]
    return p


class TestDanceMetricsStability(unittest.TestCase):
    def test_height_drops_when_body_crouches_toward_feet(self):
        engine = DanceMetricsEngine()
        standing_height, _ = engine._calculate_stability(_pose(pelvis_y=950.0))
        crouched_height, _ = engine._calculate_stability(_pose(pelvis_y=450.0))

        self.assertGreater(standing_height, crouched_height)

    def test_height_is_not_dependent_on_y_axis_sign(self):
        engine = DanceMetricsEngine()
        pose = _pose(pelvis_y=900.0)
        mirrored = pose.copy()
        mirrored[:, 1] *= -1

        height, _ = engine._calculate_stability(pose)
        mirrored_height, _ = engine._calculate_stability(mirrored)

        self.assertAlmostEqual(height, mirrored_height, places=6)

    def test_reset_clears_motion_history(self):
        engine = DanceMetricsEngine()
        engine.update(_pose(pelvis_y=900.0))
        engine.update(_pose(pelvis_y=920.0))
        engine._calculate_transition({
            "Trunk": 1.0,
            "L_Arm": 1.0,
            "R_Arm": 1.0,
            "L_Leg": 1.0,
            "R_Leg": 1.0,
        })

        engine.reset()

        self.assertEqual(engine.positions_history, [])
        self.assertEqual(engine.omega_l_history, [])
        self.assertEqual(getattr(engine, "omega_history", []), [])


if __name__ == "__main__":
    unittest.main()
