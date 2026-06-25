import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_sources import (
    rtmw3d_primary_h36m, _largest_person, _fit_affine_xy, _apply_affine_xy,
)


def _two_people():
    # person 0 small (coords ~0-10), person 1 large (coords ~0-300)
    kp = np.zeros((2, 133, 3))
    kp2 = np.zeros((2, 133, 2))
    rng = np.random.default_rng(0)
    kp[0, :17, :] = rng.uniform(0, 10, (17, 3))
    kp[1, :17, :] = rng.uniform(0, 300, (17, 3))
    kp2[0, :17, :] = kp[0, :17, :2]
    kp2[1, :17, :] = kp[1, :17, :2]
    return kp, kp2


class TestRtmw3dPrimary(unittest.TestCase):
    def test_picks_largest_person(self):
        kp, kp2 = _two_people()
        self.assertEqual(_largest_person(kp, kp2), 1)
        _h36m, pts2d = rtmw3d_primary_h36m(kp, kp2)
        # overlay points come from the primary (largest) person = index 1
        np.testing.assert_allclose(pts2d, kp2[1, :17, :])

    def test_shape_and_root_centered(self):
        kp, kp2 = _two_people()
        h36m, _ = rtmw3d_primary_h36m(kp, kp2)
        self.assertEqual(h36m.shape, (17, 3))
        # pelvis (index 0) sits at the origin after root-centering
        np.testing.assert_allclose(h36m[0], np.zeros(3), atol=1e-6)
        self.assertTrue(np.all(np.isfinite(h36m)))

    def test_z_is_scaled_not_zero(self):
        kp, kp2 = _two_people()
        h36m, _ = rtmw3d_primary_h36m(kp, kp2)
        # at least one joint has a non-trivial z (z was kept + scaled)
        self.assertGreater(float(np.abs(h36m[:, 2]).max()), 0.0)

    def test_empty_returns_none(self):
        h36m, pts2d = rtmw3d_primary_h36m(np.zeros((0, 133, 3)), None)
        self.assertIsNone(h36m)
        self.assertIsNone(pts2d)

    def test_degenerate_bbox_returns_none(self):
        kp = np.zeros((1, 133, 3))  # all joints collapsed -> bbox height 0
        h36m, pts2d = rtmw3d_primary_h36m(kp, None)
        self.assertIsNone(h36m)

    def test_affine_fits_and_applies(self):
        # crop->full overlay transform: x = crop*10+5, y = crop*11+7
        crop = np.array([[0.0, 0.0], [100.0, 50.0]])
        full = np.array([[5.0, 7.0], [1005.0, 557.0]])
        xf = _fit_affine_xy(crop, full)
        out = _apply_affine_xy(crop, xf)
        np.testing.assert_allclose(out, full, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
