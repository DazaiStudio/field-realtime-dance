import sys, unittest
from pathlib import Path
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_backends import factory


class TestFactoryRtmpose3d(unittest.TestCase):
    def test_make_backend_rtmpose3d(self):
        # Patch the class where it is defined so make_backend's lazy import
        # resolves to the mock -- no rtmlib / GPU / weights needed.
        with mock.patch(
            "pose_backends.rtmpose3d_backend.RTMPose3DBackend"
        ) as MockBackend:
            out = factory.make_backend("rtmpose3d")
            self.assertIs(out, MockBackend.return_value)
            MockBackend.assert_called_once()

    def test_env_var_selects_rtmpose3d(self):
        with mock.patch(
            "pose_backends.rtmpose3d_backend.RTMPose3DBackend"
        ) as MockBackend, mock.patch.dict(
            "os.environ", {"FIELD_POSE_BACKEND": "rtmpose3d"}
        ):
            out = factory.make_backend()
            self.assertIs(out, MockBackend.return_value)


if __name__ == "__main__":
    unittest.main()
