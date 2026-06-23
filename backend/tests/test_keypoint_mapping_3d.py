import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from keypoint_mapping import coco17_to_h36m17_3d

NOSE = 0
L_SH, R_SH = 5, 6
L_EL, R_EL = 7, 8
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK = 15, 16


def _kpts3d():
    c = np.zeros((17, 3))
    for i in range(17):
        c[i] = (i * 10, i * 10 + 1, i * 10 + 2)  # distinct z per joint
    return c


class TestCoco3D(unittest.TestCase):
    def test_arms_and_z_preserved(self):
        c = _kpts3d()
        out = coco17_to_h36m17_3d(c)
        self.assertEqual(out.shape, (17, 3))
        np.testing.assert_allclose(out[13], c[L_WR])   # left wrist -> 13, z kept
        np.testing.assert_allclose(out[16], c[R_WR])   # right wrist -> 16
        np.testing.assert_allclose(out[11], c[L_SH])
        np.testing.assert_allclose(out[14], c[R_SH])
        np.testing.assert_allclose(out[10], c[NOSE])   # head
        self.assertNotEqual(out[13][2], 0.0)           # z not zeroed

    def test_spine_chain(self):
        c = _kpts3d()
        out = coco17_to_h36m17_3d(c)
        pelvis = (c[L_HIP] + c[R_HIP]) / 2
        thorax = (c[L_SH] + c[R_SH]) / 2
        np.testing.assert_allclose(out[0], pelvis)
        np.testing.assert_allclose(out[8], thorax)
        np.testing.assert_allclose(out[7], (pelvis + thorax) / 2)
        np.testing.assert_allclose(out[9], (thorax + c[NOSE]) / 2)


if __name__ == "__main__":
    unittest.main()
