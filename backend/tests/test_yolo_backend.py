import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_backends.yolo_backend import results_to_personposes


class _FakeKpts:
    def __init__(self, xy): self.xy = xy
class _FakeBoxes:
    def __init__(self, xyxy, ids):
        self.xyxy = xyxy
        self.id = ids
class _FakeResult:
    def __init__(self, kpts, boxes):
        self.keypoints = kpts
        self.boxes = boxes


class TestYoloAdaptation(unittest.TestCase):
    def test_builds_personposes_with_ids(self):
        kxy = np.zeros((2, 17, 2)); kxy[0] += 1.0; kxy[1] += 2.0
        boxes = _FakeBoxes(np.array([[0, 0, 10, 20], [5, 5, 30, 40]]),
                           np.array([7, 9]))
        out = results_to_personposes(_FakeResult(_FakeKpts(kxy), boxes))
        self.assertEqual([p.track_id for p in out], [7, 9])
        self.assertTrue(all(p.is_3d is False for p in out))
        self.assertEqual(out[0].h36m17.shape, (17, 3))

    def test_skips_when_no_ids_yet(self):
        kxy = np.zeros((1, 17, 2))
        boxes = _FakeBoxes(np.array([[0, 0, 10, 20]]), None)
        out = results_to_personposes(_FakeResult(_FakeKpts(kxy), boxes))
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
