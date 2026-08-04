import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import osc_viewer  # noqa: E402


class TestViewerStageTiming(unittest.TestCase):
    """Per-stage timings for the live loop.

    encode_ms and pose_ms alone cannot explain the frame budget: the capture +
    body-tracking call is the one stage nobody was measuring.
    """

    def setUp(self):
        self._saved = dict(osc_viewer.processing_state)

    def tearDown(self):
        osc_viewer.processing_state.clear()
        osc_viewer.processing_state.update(self._saved)

    def test_stream_frame_records_every_stage(self):
        osc_viewer.set_stream_frame(None, encode_ms=12.5, read_ms=31.0, resize_ms=2.25)
        state = osc_viewer.processing_state
        self.assertAlmostEqual(state["encode_ms"], 12.5)
        self.assertAlmostEqual(state["read_ms"], 31.0)
        self.assertAlmostEqual(state["resize_ms"], 2.25)

    def test_stage_timings_reach_the_state_payload(self):
        osc_viewer.set_stream_frame(None, encode_ms=1.0, read_ms=2.0, resize_ms=3.0)
        processing = osc_viewer.state_payload()["processing"]
        self.assertAlmostEqual(processing["read_ms"], 2.0)
        self.assertAlmostEqual(processing["resize_ms"], 3.0)

    def test_unmeasured_paths_report_zero_not_a_stale_reading(self):
        osc_viewer.set_stream_frame(None, encode_ms=9.0, read_ms=40.0, resize_ms=5.0)
        osc_viewer.set_stream_frame(None, encode_ms=9.0)   # e.g. the video path
        state = osc_viewer.processing_state
        self.assertAlmostEqual(state["read_ms"], 0.0)
        self.assertAlmostEqual(state["resize_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
