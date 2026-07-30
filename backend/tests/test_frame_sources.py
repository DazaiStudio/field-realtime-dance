import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frame_sources import KinectFrameSource, OpenCVFrameSource


class FakeCap:
    def __init__(self):
        self.frames = [np.zeros((4, 4, 3), dtype=np.uint8)]

    def read(self):
        return True, self.frames[0]


class OpenCVDelegationTests(unittest.TestCase):
    def _source(self, log):
        cap = FakeCap()
        return OpenCVFrameSource(
            index=2, owner=9,
            open_fn=lambda index, owner: log.append(("open", index, owner)) or cap,
            read_fn=lambda c, owner: log.append(("read", owner)) or c.read(),
            release_fn=lambda owner=None, force=False: log.append(("release", owner)),
        )

    def test_open_read_release(self):
        log = []
        source = self._source(log)
        source.open()
        ok, frame = source.read()
        self.assertTrue(ok)
        source.release()
        self.assertEqual([entry[0] for entry in log], ["open", "read", "release"])
        self.assertEqual(log[0], ("open", 2, 9))
        self.assertEqual(log[1], ("read", 9))
        self.assertEqual(log[2], ("release", 9))

    def test_reopen_releases_then_opens(self):
        log = []
        source = self._source(log)
        source.open()
        source.reopen(sleep_seconds=0.0)
        self.assertEqual([entry[0] for entry in log], ["open", "release", "open"])

    def test_read_before_open_fails(self):
        source = self._source([])
        ok, frame = source.read()
        self.assertFalse(ok)
        self.assertIsNone(frame)

    def test_describe(self):
        self.assertEqual(self._source([]).describe(), "Camera 2")


class KinectFrameSourceTests(unittest.TestCase):
    def test_delegates_to_runtime(self):
        class FakeRuntime:
            def __init__(self):
                self.calls = []

            def read(self):
                self.calls.append("read")
                return True, np.zeros((2, 2, 3), dtype=np.uint8)

            def reopen(self):
                self.calls.append("reopen")

            def release(self, owner=None, force=False):
                self.calls.append(("release", owner))

            def describe(self):
                return "Azure Kinect"

        runtime = FakeRuntime()
        source = KinectFrameSource(runtime, owner=5)
        ok, _ = source.read()
        self.assertTrue(ok)
        source.reopen()
        source.release()
        self.assertEqual(runtime.calls, ["read", "reopen", ("release", 5)])
        self.assertEqual(source.describe(), "Azure Kinect")


if __name__ == "__main__":
    unittest.main()
