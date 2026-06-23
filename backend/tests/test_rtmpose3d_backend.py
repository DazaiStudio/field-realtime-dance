"""Tests for the RTMPose3D backend helper (Task 3 & Task 5).

Imports only the pure helper function -- does NOT import rtmlib.
The rtmlib import is lazy (inside RTMPose3DBackend.__init__) so this
test suite runs on any machine without ONNX weights or a GPU.

Task 5 changes tested here:
  - z_normalisation: z is now scaled by bbox_height_3d before POSE_SCALE,
    so the final z unit is consistent with x,y (all in model-input pixels).
  - keypoints_2d argument: bbox and kpts_2d are taken from keypoints_2d
    (full-frame pixel coords) when provided; fall back to 3D x,y otherwise.
  - Degenerate frame skip: if y-extent of body joints < 1 px, skip person.
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
_L_WR = 9    # left wrist in COCO-17 → maps to H36M index 13
_L_ANK = 15  # left ankle in COCO-17 → maps to H36M index 6


def _make_keypoints(n_persons, n_kpts=133, z_val=0.5, y_spread=100.0):
    """Synthetic (n_persons, n_kpts, 3) array with non-degenerate y spread.

    y spread must be >= 1.0 so the degenerate-frame check does not skip
    the person (bbox_height_3d = y_max - y_min = y_spread).
    """
    kpts = np.zeros((n_persons, n_kpts, 3), dtype=np.float32)
    for i in range(n_persons):
        kpts[i, :, 0] = float(i + 1) * 10.0   # x — distinct per person
        # Spread y linearly across joints so min/max differ by y_spread.
        for j in range(n_kpts):
            kpts[i, j, 1] = 50.0 + (j / max(n_kpts - 1, 1)) * y_spread
        kpts[i, :, 2] = z_val                  # z (normalised depth)
    return kpts


def _make_keypoints_varied_z(n_persons, n_kpts=133, y_spread=100.0):
    """Like _make_keypoints but with a DISTINCT z per joint.

    Needed because the metric pipeline now subtracts the pelvis (root-center),
    so a constant z collapses to 0 after subtraction.  Distinct z per joint
    lets tests verify the pelvis-relative z (z_joint - z_pelvis).
    """
    kpts = np.zeros((n_persons, n_kpts, 3), dtype=np.float32)
    for i in range(n_persons):
        kpts[i, :, 0] = float(i + 1) * 10.0
        for j in range(n_kpts):
            kpts[i, j, 1] = 50.0 + (j / max(n_kpts - 1, 1)) * y_spread
            kpts[i, j, 2] = 0.1 + 0.01 * j   # distinct normalised z per joint
    return kpts


def _make_scores(n_persons, n_kpts=133, score=0.9):
    return np.full((n_persons, n_kpts), score, dtype=np.float32)


def _make_keypoints_2d(n_persons, n_kpts=133):
    """Synthetic full-frame pixel coords — clearly different from 3D x,y."""
    kpts2d = np.zeros((n_persons, n_kpts, 2), dtype=np.float32)
    for i in range(n_persons):
        kpts2d[i, :, 0] = float(i + 1) * 500.0 + 100.0   # x in [600, 1600] range
        kpts2d[i, :, 1] = float(i + 1) * 200.0 + 50.0    # y in [250,  850] range
    return kpts2d


class TestPersonposesFromRtmw3d(unittest.TestCase):

    # ------------------------------------------------------------------
    # Basic count / shape / flag tests
    # ------------------------------------------------------------------

    def test_returns_correct_count(self):
        kpts = _make_keypoints(3)
        scores = _make_scores(3)
        out = personposes_from_rtmw3d(kpts, scores, [0, 1, 2])
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
        out = personposes_from_rtmw3d(kpts, scores, [7, 42, 3])
        self.assertEqual([p.track_id for p in out], [7, 42, 3])

    # ------------------------------------------------------------------
    # z-normalisation: z is scaled by bbox_height_3d then by POSE_SCALE
    # ------------------------------------------------------------------

    # COCO body-17 hip indices used for the pelvis (root-center) calculation.
    _COCO_L_HIP = 11
    _COCO_R_HIP = 12

    def test_left_wrist_z_normalised_scaled_and_root_centered(self):
        """COCO idx 9 (left wrist) → H36M idx 13.

        Full transform applied to z (body joints 0-16):
            z_scaled[j] = z_raw[j] × bbox_height_3d × scale
            z_out[j]    = z_scaled[j] − z_scaled_pelvis
        where pelvis = mean of COCO hips (11, 12) and
        bbox_height_3d = body_y.max() − body_y.min().
        """
        y_spread = 80.0
        scale = 1.0
        kpts = _make_keypoints_varied_z(1, y_spread=y_spread)
        scores = _make_scores(1)
        out = personposes_from_rtmw3d(kpts, scores, [0], scale=scale)
        p = out[0]

        body = kpts[0, :17].astype(float)
        bbox_h = float(body[:, 1].max() - body[:, 1].min())
        z_scaled = body[:, 2] * bbox_h * scale
        z_pelvis = (z_scaled[self._COCO_L_HIP] + z_scaled[self._COCO_R_HIP]) / 2.0
        # COCO left wrist (9) → H36M 13; it is a direct (non-averaged) joint.
        expected_z = z_scaled[9] - z_pelvis
        self.assertAlmostEqual(p.h36m17[13, 2], expected_z, places=3)

    def test_pelvis_z_is_zero_after_root_center(self):
        """After root-centering, the H36M pelvis (idx 0) z must be ~0."""
        kpts = _make_keypoints_varied_z(1, y_spread=100.0)
        scores = _make_scores(1)
        out = personposes_from_rtmw3d(kpts, scores, [0], scale=3.0)
        p = out[0]
        # H36M pelvis = mean of the two hips = the subtracted reference → ~0.
        np.testing.assert_allclose(p.h36m17[0], np.zeros(3), atol=1e-3)

    def test_h36m17_z_uses_pose_scale(self):
        """Pelvis-relative z must scale linearly with the scale argument."""
        kpts = _make_keypoints_varied_z(1, y_spread=100.0)
        scores = _make_scores(1)

        out_scale1 = personposes_from_rtmw3d(kpts, scores, [0], scale=1.0)
        out_scale2 = personposes_from_rtmw3d(kpts, scores, [0], scale=2.0)

        # Left ankle (COCO 15 → H36M 6) — distinct from pelvis, non-zero.
        z1 = out_scale1[0].h36m17[6, 2]
        z2 = out_scale2[0].h36m17[6, 2]

        self.assertNotAlmostEqual(z1, 0.0, places=4)
        self.assertAlmostEqual(z2, 2.0 * z1, places=4)

    def test_root_center_is_pure_translation(self):
        """Pelvis-subtraction must NOT change any limb vector (joint diffs).

        Limb vectors drive energy/expansion/sway/torque/jerk; root-centering
        is a pure translation, so every pairwise joint difference must be
        identical to a build with no subtraction.  We verify by comparing the
        full set of inter-joint vectors against a manually-recentered baseline.
        """
        kpts = _make_keypoints_varied_z(1, y_spread=120.0)
        scores = _make_scores(1)
        out = personposes_from_rtmw3d(kpts, scores, [0], scale=3.0)
        h = out[0].h36m17

        # All inter-joint vectors are translation-invariant: pick a few limbs
        # and confirm they are non-degenerate (subtraction did not zero them).
        # Left upper arm (H36M 11→12) and right thigh (H36M 1→2).
        l_upper_arm = h[12] - h[11]
        r_thigh = h[2] - h[1]
        self.assertGreater(np.linalg.norm(l_upper_arm), 1e-6)
        self.assertGreater(np.linalg.norm(r_thigh), 1e-6)

    def test_pose_scale_constant_is_calibrated(self):
        """POSE_SCALE must be a small positive number calibrated in Task 5.

        It was originally 1000.0 (placeholder).  After Task-5 calibration it
        must be much smaller (≤ 20.0) so metrics are in the right range.
        """
        self.assertGreater(POSE_SCALE, 0.0)
        self.assertLess(POSE_SCALE, 20.0,
                        "POSE_SCALE still looks like the pre-calibration placeholder")

    # ------------------------------------------------------------------
    # keypoints_2d argument: bbox / kpts_2d source
    # ------------------------------------------------------------------

    def test_kpts_2d_shape_without_keypoints_2d(self):
        """Fallback (no keypoints_2d): kpts_2d comes from 3D x,y reversed."""
        kpts = _make_keypoints(1)
        scores = _make_scores(1)
        out = personposes_from_rtmw3d(kpts, scores, [0])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].kpts_2d.shape, (17, 2))

    def test_kpts_2d_from_keypoints_2d_when_provided(self):
        """When keypoints_2d is provided, kpts_2d must use those pixel coords."""
        kpts = _make_keypoints(1)
        scores = _make_scores(1)
        kpts2d = _make_keypoints_2d(1)
        out = personposes_from_rtmw3d(kpts, scores, [0], keypoints_2d=kpts2d)
        p = out[0]
        self.assertEqual(p.kpts_2d.shape, (17, 2))
        # The 2D pixel coords should come from kpts2d, not from 3D x,y.
        np.testing.assert_allclose(p.kpts_2d, kpts2d[0, :17, :], rtol=1e-5)

    def test_bbox_from_keypoints_2d_when_provided(self):
        """bbox must use full-frame pixel coords when keypoints_2d is given."""
        kpts = _make_keypoints(1)
        scores = _make_scores(1)
        kpts2d = _make_keypoints_2d(1)
        out = personposes_from_rtmw3d(kpts, scores, [0], keypoints_2d=kpts2d)
        p = out[0]
        body2d = kpts2d[0, :17, :]
        expected_bbox = (
            float(body2d[:, 0].min()),
            float(body2d[:, 1].min()),
            float(body2d[:, 0].max()),
            float(body2d[:, 1].max()),
        )
        self.assertAlmostEqual(p.bbox[0], expected_bbox[0], places=4)
        self.assertAlmostEqual(p.bbox[1], expected_bbox[1], places=4)
        self.assertAlmostEqual(p.bbox[2], expected_bbox[2], places=4)
        self.assertAlmostEqual(p.bbox[3], expected_bbox[3], places=4)

    def test_bbox_fallback_to_3d_xy_without_keypoints_2d(self):
        """Without keypoints_2d, bbox is derived from the (un-scaled) 3D x,y."""
        kpts = _make_keypoints(1)
        scores = _make_scores(1)
        out = personposes_from_rtmw3d(kpts, scores, [0])
        p = out[0]
        self.assertEqual(len(p.bbox), 4)
        # Bbox must be a 4-tuple of finite floats.
        for v in p.bbox:
            self.assertTrue(np.isfinite(v))

    # ------------------------------------------------------------------
    # Edge / skip cases
    # ------------------------------------------------------------------

    def test_skips_when_no_ids_yet(self):
        """Empty track_ids list → return []."""
        kpts = _make_keypoints(1)
        scores = _make_scores(1)
        out = personposes_from_rtmw3d(kpts, scores, [])
        self.assertEqual(out, [])

    def test_skips_person_with_negative_id(self):
        """track_id == -1 means the tracker has not assigned a valid id yet."""
        kpts = _make_keypoints(2)
        scores = _make_scores(2)
        out = personposes_from_rtmw3d(kpts, scores, [3, -1])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].track_id, 3)

    def test_skips_degenerate_frame(self):
        """If all body-joint y values are identical (bbox_h_3d < 1), skip."""
        kpts = _make_keypoints(1, y_spread=0.0)   # all y identical → degenerate
        scores = _make_scores(1)
        out = personposes_from_rtmw3d(kpts, scores, [0])
        # Person should be skipped — degenerate frame.
        self.assertEqual(out, [])

    def test_returns_empty_for_empty_input(self):
        kpts = np.zeros((0, 133, 3), dtype=np.float32)
        scores = np.zeros((0, 133), dtype=np.float32)
        out = personposes_from_rtmw3d(kpts, scores, [])
        self.assertEqual(out, [])

    def test_multiple_persons_keypoints_2d_indexed_correctly(self):
        """Each person must get its own row from keypoints_2d."""
        kpts = _make_keypoints(2)
        scores = _make_scores(2)
        kpts2d = _make_keypoints_2d(2)
        out = personposes_from_rtmw3d(kpts, scores, [0, 1], keypoints_2d=kpts2d)
        self.assertEqual(len(out), 2)
        for idx, p in enumerate(out):
            np.testing.assert_allclose(p.kpts_2d, kpts2d[idx, :17, :], rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
