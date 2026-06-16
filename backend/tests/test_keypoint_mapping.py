import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from keypoint_mapping import coco17_to_h36m17


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

    def test_derived_joints(self):
        c = _coco()
        out = coco17_to_h36m17(c)
        l_hip, r_hip = c[11], c[12]
        l_sh, r_sh = c[5], c[6]
        np.testing.assert_allclose(out[0][:2], (l_hip + r_hip) / 2)
        np.testing.assert_allclose(out[8][:2], (l_sh + r_sh) / 2)
        np.testing.assert_allclose(out[9][:2], c[0])
        np.testing.assert_allclose(out[7][:2], (out[0][:2] + out[8][:2]) / 2)

    def test_direct_limb_joints(self):
        c = _coco()
        out = coco17_to_h36m17(c)
        np.testing.assert_allclose(out[1][:2], c[12])
        np.testing.assert_allclose(out[3][:2], c[16])
        np.testing.assert_allclose(out[12][:2], c[9])
        np.testing.assert_allclose(out[15][:2], c[10])


if __name__ == "__main__":
    unittest.main()
