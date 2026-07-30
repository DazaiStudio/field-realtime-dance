import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_backends.azure_kinect import (
    KinectBody,
    body_quality,
    bbox_from_points,
    transform_points_2d,
    pad_to_aspect,
)


def _conf(fill=2):
    return np.full(32, fill, dtype=float)


class BodyQualityTests(unittest.TestCase):
    def test_all_medium_is_valid(self):
        quality, valid = body_quality(_conf(2))
        self.assertAlmostEqual(quality, 0.8)
        self.assertTrue(valid)

    def test_none_core_joint_invalidates(self):
        conf = _conf(3)
        conf[18] = 0  # HIP_LEFT = NONE
        quality, valid = body_quality(conf)
        self.assertFalse(valid)

    def test_none_peripheral_joint_keeps_valid(self):
        conf = _conf(2)
        conf[7] = 0  # WRIST_LEFT
        quality, valid = body_quality(conf)
        self.assertTrue(valid)
        self.assertLess(quality, 0.8)


class Transform2DTests(unittest.TestCase):
    def test_scale_only(self):
        pts = np.array([[640.0, 360.0]])
        out = transform_points_2d(pts, native_size=(1280, 720),
                                  frame_size=(1920, 1080), mirrored=False)
        np.testing.assert_allclose(out, [[960.0, 540.0]])

    def test_mirror_flips_x_in_native_space(self):
        pts = np.array([[100.0, 50.0]])
        out = transform_points_2d(pts, native_size=(1280, 720),
                                  frame_size=(1280, 720), mirrored=True)
        np.testing.assert_allclose(out, [[1180.0, 50.0]])


class BboxTests(unittest.TestCase):
    def test_bbox_padded_and_clamped(self):
        pts = np.array([[10.0, 10.0], [110.0, 210.0]])
        x1, y1, x2, y2 = bbox_from_points(pts, frame_size=(200, 220), pad_frac=0.1)
        self.assertAlmostEqual(x1, 0.0)     # 10 - 10% of 100 = 0
        self.assertAlmostEqual(y1, 0.0)     # 10 - 10% of 200 = -10 -> clamp 0
        self.assertAlmostEqual(x2, 120.0)
        self.assertAlmostEqual(y2, 220.0)   # 230 -> clamp to frame


class PadToAspectTests(unittest.TestCase):
    def test_nfov_depth_to_16_9(self):
        img = np.zeros((576, 640, 3), dtype=np.uint8)
        out, x_off, y_off = pad_to_aspect(img, 16, 9)
        self.assertEqual(out.shape[0], 576)
        self.assertEqual(out.shape[1], 1024)  # 576 * 16/9
        self.assertEqual(x_off, (1024 - 640) // 2)
        self.assertEqual(y_off, 0)

    def test_already_wide_enough(self):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        out, x_off, y_off = pad_to_aspect(img, 16, 9)
        self.assertEqual(out.shape, (720, 1280, 3))
        self.assertEqual((x_off, y_off), (0, 0))


if __name__ == "__main__":
    unittest.main()
