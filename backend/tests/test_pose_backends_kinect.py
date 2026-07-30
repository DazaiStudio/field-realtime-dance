import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_backends.azure_kinect import (
    KinectBody,
    body_quality,
    bbox_from_points,
    sanitize_joints2d,
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


class Sanitize2DTests(unittest.TestCase):
    def test_zero_marker_and_conf_gate(self):
        # k4a marks failed 3d->2d projections as (0, 0); NONE-confidence joints
        # are position guesses (occluded behind scenery) — both must not reach
        # the overlay/bbox as if they were real pixels.
        pts = np.array([[0.0, 0.0], [100.0, 50.0], [200.0, 80.0]])
        conf = np.array([2, 0, 2])
        out = sanitize_joints2d(pts, conf)
        self.assertTrue(np.isnan(out[0]).all())   # projection-failed marker
        self.assertTrue(np.isnan(out[1]).all())   # confidence NONE
        np.testing.assert_allclose(out[2], [200.0, 80.0])

    def test_non_finite_input(self):
        pts = np.array([[np.inf, 5.0], [50.0, 60.0]])
        out = sanitize_joints2d(pts, np.array([2, 2]))
        self.assertTrue(np.isnan(out[0]).all())
        np.testing.assert_allclose(out[1], [50.0, 60.0])


class BboxTests(unittest.TestCase):
    def test_bbox_padded_and_clamped(self):
        pts = np.array([[10.0, 10.0], [110.0, 210.0]])
        x1, y1, x2, y2 = bbox_from_points(pts, frame_size=(200, 220), pad_frac=0.1)
        self.assertAlmostEqual(x1, 0.0)     # 10 - 10% of 100 = 0
        self.assertAlmostEqual(y1, 0.0)     # 10 - 10% of 200 = -10 -> clamp 0
        self.assertAlmostEqual(x2, 120.0)
        self.assertAlmostEqual(y2, 220.0)   # 230 -> clamp to frame

    def test_ignores_nan_points(self):
        pts = np.array([[np.nan, np.nan], [10.0, 10.0], [110.0, 210.0]])
        bbox = bbox_from_points(pts, frame_size=(200, 220), pad_frac=0.0)
        self.assertAlmostEqual(bbox[0], 10.0)
        self.assertAlmostEqual(bbox[3], 210.0)

    def test_all_nan_returns_none(self):
        pts = np.full((3, 2), np.nan)
        self.assertIsNone(bbox_from_points(pts, frame_size=(200, 220)))


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

    def test_nan_joints_skipped_in_overlay_and_bbox(self):
        body = _fake_body(6)
        body.joints2d[20] = np.nan          # left ankle unprojectable (occluded)
        runtime = FakeRuntime(bodies=[body])
        source = self._source(runtime)
        source.estimate(_frame(), 1000.0)
        points = source._overlay_points_by_id[1]
        self.assertNotIn(6, points)         # H36M l_ankle not drawn
        self.assertIn(3, points)            # r_ankle still drawn
        bbox = source.last_tracking["tracks"][0]["bbox"]
        self.assertTrue(all(np.isfinite(v) for v in bbox))

    def test_body_with_no_projectable_joints_is_dropped(self):
        body = _fake_body(6)
        body.joints2d[:] = np.nan
        runtime = FakeRuntime(bodies=[body])
        source = self._source(runtime)
        frame, h36m = source.estimate(_frame(), 1000.0)
        self.assertEqual(source.last_h36m_by_id, {})
        self.assertIsNone(h36m)


from pose_backends.azure_kinect import KinectError, KinectRuntime, _guarded  # noqa: E402


class GuardedCallTests(unittest.TestCase):
    def test_converts_system_exit(self):
        def boom():
            raise SystemExit(1)   # pykinect VERIFY() does this
        with self.assertRaises(KinectError):
            _guarded("enqueue", boom)

    def test_converts_exception(self):
        def boom():
            raise RuntimeError("usb reset")
        with self.assertRaises(KinectError):
            _guarded("capture", boom)

    def test_passes_result(self):
        self.assertEqual(_guarded("ok", lambda: 42), 42)


class RuntimeOwnershipTests(unittest.TestCase):
    def test_release_requires_owner(self):
        runtime = KinectRuntime()
        runtime._opened = True
        runtime._owner = 7
        closed = []
        runtime._close_device = lambda: closed.append(True)
        runtime.release(owner=3)      # wrong owner: no-op
        self.assertTrue(runtime._opened)
        runtime.release(owner=7)
        self.assertFalse(runtime._opened)
        self.assertEqual(closed, [True])

    def test_read_when_closed_reports_error(self):
        runtime = KinectRuntime()
        ok, frame = runtime.read()
        self.assertFalse(ok)
        self.assertIsNone(frame)
        self.assertTrue(runtime.last_error)


if __name__ == "__main__":
    unittest.main()
