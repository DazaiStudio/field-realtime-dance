import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import osc_viewer  # noqa: E402


def _frame(w, h):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestResizeNeverUpscales(unittest.TestCase):
    """The performance preset is a cap, not a target.

    resize_frame uses INTER_AREA, which is the shrink-optimised interpolation --
    the presets were written for webcams that overshoot them. The Kinect views
    (1280x720 colour, 1024x576 padded depth) are smaller than the quality
    preset, so the same call was enlarging them: no detail gained (K4ABT has
    already run on native depth by then), but a bigger frame to JPEG-encode and
    for the browser to composite.
    """

    def setUp(self):
        self._saved = (osc_viewer.source_state["width"], osc_viewer.source_state["height"])

    def tearDown(self):
        osc_viewer.source_state["width"], osc_viewer.source_state["height"] = self._saved

    def _target(self, w, h):
        osc_viewer.source_state["width"] = w
        osc_viewer.source_state["height"] = h

    def test_smaller_source_is_left_alone(self):
        self._target(1920, 1080)
        frame = _frame(1280, 720)                      # Kinect colour view
        out = osc_viewer.resize_frame(frame)
        self.assertEqual(out.shape[:2], (720, 1280))

    def test_padded_depth_view_is_left_alone(self):
        self._target(1920, 1080)
        out = osc_viewer.resize_frame(_frame(1024, 576))
        self.assertEqual(out.shape[:2], (576, 1024))

    def test_larger_source_still_downscales(self):
        self._target(1280, 720)
        out = osc_viewer.resize_frame(_frame(1920, 1080))
        self.assertEqual(out.shape[:2], (720, 1280))

    def test_exact_match_is_passed_through_untouched(self):
        self._target(1280, 720)
        frame = _frame(1280, 720)
        self.assertIs(osc_viewer.resize_frame(frame), frame)

    def test_one_oversized_dimension_still_downscales(self):
        # Anything that exceeds the cap on either axis keeps the old behaviour.
        self._target(1280, 720)
        out = osc_viewer.resize_frame(_frame(1920, 480))
        self.assertEqual(out.shape[:2], (720, 1280))


class TestActualFrameSizeIsReported(unittest.TestCase):
    """/api/state must not claim a resolution the stream is not sending."""

    def setUp(self):
        self._saved = dict(osc_viewer.processing_state)

    def tearDown(self):
        osc_viewer.processing_state.clear()
        osc_viewer.processing_state.update(self._saved)

    def test_stream_frame_records_the_size_it_encoded(self):
        osc_viewer.set_stream_frame(_frame(1280, 720), encode_ms=1.0)
        self.assertEqual(osc_viewer.processing_state["frame_size"], [1280, 720])

    def test_reaches_the_state_payload(self):
        osc_viewer.set_stream_frame(_frame(1024, 576), encode_ms=1.0)
        self.assertEqual(osc_viewer.state_payload()["processing"]["frame_size"], [1024, 576])


if __name__ == "__main__":
    unittest.main()
