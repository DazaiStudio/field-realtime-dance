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


from pose_backends.azure_kinect import AzureKinectPoseSource  # noqa: E402


class FakeRuntime:
    def __init__(self, bodies=None, view="color", mirrored=False,
                 native_size=(1280, 720)):
        self.last_bodies = bodies or []
        self.view = view
        self.mirrored = mirrored
        self.native_view_size = native_size
        self.last_error = None


def _fake_body(body_id, x_offset=0.0, conf=2):
    joints = np.zeros((32, 4))
    joints[:, 3] = conf
    base = {18: [-100, 0, 1000], 22: [100, 0, 1000],
            19: [-110, 400, 1000], 23: [110, 400, 1000],
            20: [-120, 800, 1000], 24: [120, 800, 1000],
            5: [-180, -500, 1000], 12: [180, -500, 1000],
            6: [-200, -250, 1000], 13: [200, -250, 1000],
            7: [-210, 0, 1000], 14: [210, 0, 1000],
            27: [0, -700, 1000]}
    for i, xyz in base.items():
        joints[i, :3] = xyz
        joints[i, 0] += x_offset
    joints2d = np.zeros((32, 2))
    joints2d[:, 0] = 280.0 + np.arange(32) * 2.0 + x_offset / 5.0  # non-zero width
    joints2d[:, 1] = 300.0
    joints2d[20, 1] = joints2d[24, 1] = 600.0   # ankles lower for a real bbox
    return KinectBody(body_id=body_id, joints=joints, joints2d=joints2d)


def _frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


class PoseSourceSingleTests(unittest.TestCase):
    def test_no_bodies(self):
        source = AzureKinectPoseSource(FakeRuntime())
        frame, h36m = source.estimate(_frame(), 1000.0)
        self.assertIsNone(h36m)
        self.assertFalse(source.last_pose_valid)
        self.assertIsNone(source.last_h36m_by_id)

    def test_single_body_disabled_tracking(self):
        source = AzureKinectPoseSource(FakeRuntime(bodies=[_fake_body(7)]))
        frame, h36m = source.estimate(_frame(), 1000.0)
        self.assertEqual(h36m.shape, (17, 3))
        self.assertTrue(source.last_pose_valid)
        self.assertAlmostEqual(source.last_pose_quality, 0.8)
        self.assertIsNone(source.last_h36m_by_id)      # single-person contract
        # registry still runs: active id present so PoseEngine resets on change
        self.assertEqual(source.last_tracking["active_id"], 1)
        self.assertFalse(source.last_tracking["enabled"])

    def test_invalid_body_returns_none(self):
        body = _fake_body(7)
        body.joints[18, 3] = 0   # HIP_LEFT NONE
        source = AzureKinectPoseSource(FakeRuntime(bodies=[body]))
        frame, h36m = source.estimate(_frame(), 1000.0)
        self.assertIsNone(h36m)
        self.assertFalse(source.last_pose_valid)


class PoseSourceTrackedTests(unittest.TestCase):
    def _source(self, runtime):
        return AzureKinectPoseSource(runtime, tracking_enabled=True)

    def test_two_bodies_get_stable_ids(self):
        runtime = FakeRuntime(bodies=[_fake_body(11), _fake_body(23, x_offset=500)])
        source = self._source(runtime)
        source.estimate(_frame(), 1000.0)
        self.assertEqual(set(source.last_h36m_by_id), {1, 2})
        self.assertEqual(source.last_tracking["count"], 2)
        states = {t["stable_id"]: t["state"] for t in source.last_tracking["tracks"]}
        self.assertEqual(states, {1: "tracking", 2: "tracking"})

    def test_body_id_change_keeps_stable_id(self):
        runtime = FakeRuntime(bodies=[_fake_body(11)])
        source = self._source(runtime)
        source.estimate(_frame(), 1000.0)
        runtime.last_bodies = [_fake_body(99)]   # K4ABT re-assigned the raw id
        source.estimate(_frame(), 1100.0)
        self.assertEqual(list(source.last_h36m_by_id), [1])   # registry re-id

    def test_mirror_flips_skeleton(self):
        runtime = FakeRuntime(bodies=[_fake_body(3)], mirrored=True)
        source = self._source(runtime)
        source.estimate(_frame(), 1000.0)
        h = source.last_h36m_by_id[1]
        # r_ankle should carry the mirrored left ankle x (+120)
        np.testing.assert_allclose(h[3], [120, 800, 0], atol=1e-6)

    def test_configure_tracking_toggles(self):
        runtime = FakeRuntime(bodies=[_fake_body(4)])
        source = self._source(runtime)
        source.configure_tracking(enabled=False)
        source.estimate(_frame(), 1000.0)
        self.assertIsNone(source.last_h36m_by_id)
        source.configure_tracking(enabled=True)
        source.estimate(_frame(), 1100.0)
        self.assertIsInstance(source.last_h36m_by_id, dict)

    def test_runtime_error_reported(self):
        runtime = FakeRuntime()
        runtime.last_error = "device unplugged"
        source = self._source(runtime)
        source.estimate(_frame(), 1000.0)
        self.assertEqual(source.last_tracking["state"], "error")
        self.assertEqual(source.last_tracking["error"], "device unplugged")


if __name__ == "__main__":
    unittest.main()
