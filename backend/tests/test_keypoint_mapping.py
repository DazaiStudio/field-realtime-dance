import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from keypoint_mapping import mp33_to_h36m17, coco17_to_h36m17_3d


class _LM:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def _fake_mp():
    # 33 MediaPipe landmarks; each axis encodes its index so we can assert
    # which source landmark ended up in which H36M slot.
    return [_LM(float(i), float(i) + 0.1, float(i) + 0.2) for i in range(33)]


class TestMp33ToH36m(unittest.TestCase):
    def test_arms_head_legs_standard_layout(self):
        out = mp33_to_h36m17(_fake_mp(), scale=1000.0)
        s = 1000.0
        # Wrists land on 13 (L=MP15) and 16 (R=MP16) -- NOT 12/15.
        np.testing.assert_allclose(out[13], np.array([15, 15.1, 15.2]) * s)
        np.testing.assert_allclose(out[16], np.array([16, 16.1, 16.2]) * s)
        # Shoulders 11/14, elbows 12/15, head 10.
        np.testing.assert_allclose(out[11], np.array([11, 11.1, 11.2]) * s)
        np.testing.assert_allclose(out[12], np.array([13, 13.1, 13.2]) * s)
        np.testing.assert_allclose(out[14], np.array([12, 12.1, 12.2]) * s)
        np.testing.assert_allclose(out[15], np.array([14, 14.1, 14.2]) * s)
        np.testing.assert_allclose(out[10], np.array([0, 0.1, 0.2]) * s)
        # Legs: 3 R-ankle (MP28), 6 L-ankle (MP27).
        np.testing.assert_allclose(out[3], np.array([28, 28.1, 28.2]) * s)
        np.testing.assert_allclose(out[6], np.array([27, 27.1, 27.2]) * s)

    def test_thorax_is_mid_shoulders(self):
        out = mp33_to_h36m17(_fake_mp(), scale=1.0)
        l_sh = np.array([11, 11.1, 11.2]); r_sh = np.array([12, 12.1, 12.2])
        np.testing.assert_allclose(out[8], (l_sh + r_sh) / 2.0)


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
        np.testing.assert_allclose(out[13], c[9])    # left wrist -> 13, z kept
        np.testing.assert_allclose(out[16], c[10])   # right wrist -> 16
        np.testing.assert_allclose(out[11], c[5])    # left shoulder
        np.testing.assert_allclose(out[10], c[0])    # head (nose)
        self.assertNotEqual(out[13][2], 0.0)         # z not zeroed

    def test_spine_chain(self):
        c = _kpts3d()
        out = coco17_to_h36m17_3d(c)
        pelvis = (c[11] + c[12]) / 2
        thorax = (c[5] + c[6]) / 2
        np.testing.assert_allclose(out[0], pelvis)
        np.testing.assert_allclose(out[8], thorax)
        np.testing.assert_allclose(out[7], (pelvis + thorax) / 2)
        np.testing.assert_allclose(out[9], (thorax + c[0]) / 2)


if __name__ == "__main__":
    unittest.main()
