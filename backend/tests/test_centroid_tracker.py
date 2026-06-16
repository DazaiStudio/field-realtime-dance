import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from centroid_tracker import CentroidTracker


class TestCentroidTracker(unittest.TestCase):
    def test_assigns_ids_and_keeps_them_stable(self):
        t = CentroidTracker(max_distance=50, evict_after=2)
        ids1 = t.update([(10, 10), (200, 200)])
        self.assertEqual(len(set(ids1)), 2)
        ids2 = t.update([(12, 11), (205, 198)])
        self.assertEqual(ids1, ids2)

    def test_new_centroid_gets_new_id(self):
        t = CentroidTracker(max_distance=50, evict_after=2)
        t.update([(10, 10)])
        ids = t.update([(10, 10), (400, 400)])
        self.assertEqual(len(set(ids)), 2)


if __name__ == "__main__":
    unittest.main()
