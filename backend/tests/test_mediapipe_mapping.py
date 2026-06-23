import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# mediapipe_backend imports the mediapipe package at module load; skip the
# whole module when it is not installed so the rest of the suite still runs.
try:
    from pose_backends.mediapipe_backend import _mp33_to_h36m17
    HAVE_MP = True
except Exception:
    HAVE_MP = False


class _LM:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def _fake_lms():
    # 33 MediaPipe landmarks; each axis encodes its own index so we can assert
    # which source landmark ended up in which H36M slot.
    return [_LM(float(i), float(i) + 0.1, float(i) + 0.2) for i in range(33)]


@unittest.skipUnless(HAVE_MP, "mediapipe not installed")
class TestMpToH36m(unittest.TestCase):
    def test_arms_and_head_standard_h36m_layout(self):
        out = _mp33_to_h36m17(_fake_lms())
        s = 1000.0  # converter scales metres -> mm
        # Wrists must land on 13 (L=MP15) and 16 (R=MP16), not 12/15.
        np.testing.assert_allclose(out[13], np.array([15, 15.1, 15.2]) * s)
        np.testing.assert_allclose(out[16], np.array([16, 16.1, 16.2]) * s)
        # Shoulders on 11 (MP11) / 14 (MP12); elbows on 12 (MP13) / 15 (MP14).
        np.testing.assert_allclose(out[11], np.array([11, 11.1, 11.2]) * s)
        np.testing.assert_allclose(out[12], np.array([13, 13.1, 13.2]) * s)
        np.testing.assert_allclose(out[14], np.array([12, 12.1, 12.2]) * s)
        np.testing.assert_allclose(out[15], np.array([14, 14.1, 14.2]) * s)
        # Head (nose) on 10.
        np.testing.assert_allclose(out[10], np.array([0, 0.1, 0.2]) * s)

    def test_legs_direct(self):
        out = _mp33_to_h36m17(_fake_lms())
        s = 1000.0
        np.testing.assert_allclose(out[3], np.array([28, 28.1, 28.2]) * s)  # R ankle
        np.testing.assert_allclose(out[6], np.array([27, 27.1, 27.2]) * s)  # L ankle


if __name__ == "__main__":
    unittest.main()
