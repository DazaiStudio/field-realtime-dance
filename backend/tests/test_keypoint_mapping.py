import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from keypoint_mapping import coco17_to_h36m17

# COCO-17 indices
NOSE = 0
L_SH, R_SH = 5, 6
L_EL, R_EL = 7, 8
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK = 15, 16


def _coco():
    c = np.zeros((17, 2))
    for i in range(17):
        c[i] = (i * 10, i * 10 + 1)
    return c


class TestCocoToH36m(unittest.TestCase):
    def test_shape_and_zero_z(self):
        out = coco17_to_h36m17(_coco())
        self.assertEqual(out.shape, (17, 3))
        self.assertTrue(np.allclose(out[:, 2], 0.0))

    def test_spine_chain_standard_h36m(self):
        # H36M: 0 pelvis, 7 spine, 8 thorax, 9 neck, 10 head
        c = _coco()
        out = coco17_to_h36m17(c)
        pelvis = (c[L_HIP] + c[R_HIP]) / 2
        thorax = (c[L_SH] + c[R_SH]) / 2
        head = c[NOSE]
        np.testing.assert_allclose(out[0][:2], pelvis)
        np.testing.assert_allclose(out[8][:2], thorax)
        np.testing.assert_allclose(out[10][:2], head)
        np.testing.assert_allclose(out[7][:2], (pelvis + thorax) / 2)
        np.testing.assert_allclose(out[9][:2], (thorax + head) / 2)

    def test_legs_direct(self):
        # H36M: 1 RHip 2 RKnee 3 RAnkle, 4 LHip 5 LKnee 6 LAnkle
        c = _coco()
        out = coco17_to_h36m17(c)
        np.testing.assert_allclose(out[1][:2], c[R_HIP])
        np.testing.assert_allclose(out[2][:2], c[R_KNEE])
        np.testing.assert_allclose(out[3][:2], c[R_ANK])
        np.testing.assert_allclose(out[4][:2], c[L_HIP])
        np.testing.assert_allclose(out[5][:2], c[L_KNEE])
        np.testing.assert_allclose(out[6][:2], c[L_ANK])

    def test_arms_standard_h36m_layout(self):
        # Regression guard for the upstream joint-index bug: wrists must land
        # on H36M 13 (left) and 16 (right), shoulders on 11/14 -- NOT shifted.
        c = _coco()
        out = coco17_to_h36m17(c)
        np.testing.assert_allclose(out[11][:2], c[L_SH])
        np.testing.assert_allclose(out[12][:2], c[L_EL])
        np.testing.assert_allclose(out[13][:2], c[L_WR])
        np.testing.assert_allclose(out[14][:2], c[R_SH])
        np.testing.assert_allclose(out[15][:2], c[R_EL])
        np.testing.assert_allclose(out[16][:2], c[R_WR])


if __name__ == "__main__":
    unittest.main()
