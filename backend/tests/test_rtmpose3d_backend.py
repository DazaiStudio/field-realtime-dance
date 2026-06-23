"""Tests for the RTMPose3D backend helper (Task 3).

Imports only the pure helper function -- does NOT import rtmlib.
The rtmlib import is lazy (inside RTMPose3DBackend.__init__) so this
test suite runs on any machine without ONNX weights or a GPU.
"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import only the pure helper -- NOT the class (which would trigger a lazy
# rtmlib import on first instantiation, not on module load, so this is safe).
from pose_backends.rtmpose3d_backend import personposes_from_rtmw3d, POSE_SCALE

# COCO-17 indices referenced in the tests
_L_WR = 9   # left wrist in COCO-17 → maps to H36M index 13


def _make_keypoints(n_persons, n_kpts=133, z_val=0.5):
    """Synthetic (n_persons, n_kpts, 3) array with distinct values per person."""
    kpts = np.zeros((n_persons, n_kpts, 3), dtype=np.float32)
    for i in range(n_persons):
        kpts[i, :, 0] = float(i + 1) * 10.0   # x
        kpts[i, :, 1] = float(i + 1) * 20.0   # y
        kpts[i, :, 2] = z_val                  # z (3D depth, all same for simplicity)
    return kpts


def _make_scores(n_persons, n_kpts=133, score=0.9):
    return np.full((n_persons, n_kpts), score, dtype=np.float32)


class TestPersonposesFromRtmw3d(unittest.TestCase):

    def test_returns_correct_count(self):
        kpts = _make_keypoints(3)
        scores = _make_scores(3)
        track_ids = [0, 1, 2]
        out = personposes_from_rtmw3d(kpts, scores, track_ids)
        self.assertEqual(len(out), 3)

    def test_single_person_shape_and_flags(self):
        kpts = _make_keypoints(1)
        scores = _make_scores(1)
        out = personposes_from_rtmw3d(kpts, scores, [5])
        self.assertEqual(len(out), 1)
        p = out[0]
        self.assertEqual(p.h36m17.shape, (17, 3))
        self.assertTrue(p.is_3d)
        self.assertEqual(p.track_id, 5)

    def test_track_ids_pass_through(self):
        kpts = _make_keypoints(3)
        scores = _make_scores(3)
        track_ids = [7, 42, 3]
        out = personposes_from_rtmw3d(kpts, scores, track_ids)
        self.assertEqual([p.track_id for p in out], [7, 42, 3])

    def test_left_wrist_z_scaled_and_mapped_to_h36m13(self):
        """COCO idx 9 (left wrist) must land on H36M idx 13 with z scaled."""
        z_raw = 1.0
        kpts = _make_keypoints(1, z_val=z_raw)
        scores = _make_scores(1)
        out = personposes_from_rtmw3d(kpts, scores, [0])
        p = out[0]
        # H36M index 13 = left wrist (confirmed in keypoint_mapping.py)
        self.assertAlmostEqual(p.h36m17[13, 2], z_raw * POSE_SCALE, places=4)

    def test_kpts_2d_shape(self):
        kpts = _make_keypoints(2)
        scores = _make_scores(2)
        out = personposes_from_rtmw3d(kpts, scores, [0, 1])
        for p in out:
            self.assertEqual(p.kpts_2d.shape, (17, 2))

    def test_bbox_derived_from_2d_keypoints(self):
        """bbox (x1,y1,x2,y2) must cover the person's 2D kpt extent."""
        kpts = _make_keypoints(1)
        scores = _make_scores(1)
        out = personposes_from_rtmw3d(kpts, scores, [0])
        p = out[0]
        self.assertEqual(len(p.bbox), 4)
        x1, y1, x2, y2 = p.bbox
        body_xy = kpts[0, :17, :2]
        self.assertAlmostEqual(x1, float(body_xy[:, 0].min()), places=4)
        self.assertAlmostEqual(y1, float(body_xy[:, 1].min()), places=4)
        self.assertAlmostEqual(x2, float(body_xy[:, 0].max()), places=4)
        self.assertAlmostEqual(y2, float(body_xy[:, 1].max()), places=4)

    def test_skips_when_no_ids_yet(self):
        """Empty track_ids list → return [] (mirrors YOLO backend behaviour)."""
        kpts = _make_keypoints(1)
        scores = _make_scores(1)
        out = personposes_from_rtmw3d(kpts, scores, [])
        self.assertEqual(out, [])

    def test_skips_person_with_negative_id(self):
        """track_id == -1 means the tracker has not assigned a valid id yet."""
        kpts = _make_keypoints(2)
        scores = _make_scores(2)
        # Second person gets -1 (invalid)
        out = personposes_from_rtmw3d(kpts, scores, [3, -1])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].track_id, 3)

    def test_h36m17_z_scaled_by_pose_scale(self):
        """All z values in h36m17 must be multiplied by POSE_SCALE."""
        z_raw = 0.5
        kpts = _make_keypoints(1, z_val=z_raw)
        scores = _make_scores(1)
        out = personposes_from_rtmw3d(kpts, scores, [0])
        p = out[0]
        # The scaled z should be z_raw * POSE_SCALE for non-averaged joints.
        # Left ankle (COCO 15 → H36M 6) is a direct copy, not averaged.
        self.assertAlmostEqual(p.h36m17[6, 2], z_raw * POSE_SCALE, places=4)

    def test_returns_empty_for_empty_input(self):
        kpts = np.zeros((0, 133, 3), dtype=np.float32)
        scores = np.zeros((0, 133), dtype=np.float32)
        out = personposes_from_rtmw3d(kpts, scores, [])
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
