import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_backend import PersonPose, PoseBackend


class _Dummy:
    def estimate(self, frame, timestamp_ms): return []
    def close(self): pass


class TestPoseBackend(unittest.TestCase):
    def test_personpose_holds_fields(self):
        p = PersonPose(track_id=3, h36m17=np.zeros((17, 3)),
                       bbox=(0, 0, 10, 20), kpts_2d=np.zeros((17, 2)), is_3d=False)
        self.assertEqual(p.track_id, 3)
        self.assertEqual(p.h36m17.shape, (17, 3))
        self.assertFalse(p.is_3d)

    def test_dummy_satisfies_protocol(self):
        self.assertIsInstance(_Dummy(), PoseBackend)


if __name__ == "__main__":
    unittest.main()
