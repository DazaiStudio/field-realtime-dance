import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from group_extent import GroupExtent
from group_overlay import denormalized_box, normalized_box
from group_smoothing import (
    DEFAULT_FRAMES,
    MAX_FRAMES,
    NOMINAL_ANALYSIS_DT,
    GroupSmoother,
    cutoff_for_frames,
    smoothed_group_outputs,
)

FRAME = (1080, 1920)   # (h, w), the rehearsal capture size


def box_px(x1, y1, x2, y2):
    return (float(x1), float(y1), float(x2), float(y2))


def feed(smoother, boxes, extent=None, start=0.0, step=0.1):
    """Run a sequence of pixel boxes through and collect the normalised output.

    ``step`` defaults to the 10 Hz analysis rate the viewer actually runs at, so
    the filter sees realistic dt values rather than a rate it never meets.
    """
    out = []
    for index, raw in enumerate(boxes):
        _, _, norm = smoothed_group_outputs(
            smoother, extent, raw, FRAME, start + index * step
        )
        out.append(norm)
    return out


class TestOffIsUntouched(unittest.TestCase):
    """cutoff 0 must be exactly the behaviour that existed before this module.

    The group box has not been verified against real dancers yet, so 'off' has
    to be provably the old path, not a very light filter.
    """

    def test_disabled_returns_the_caller_s_own_box(self):
        smoother = GroupSmoother(0.0)
        raw = box_px(100, 200, 400, 800)
        extent = GroupExtent(count=2, width=1.5, depth=2.0, cx=0.1, cz=4.0)
        out_extent, out_px, out_norm = smoothed_group_outputs(
            smoother, extent, raw, FRAME, 1.0
        )
        self.assertIs(out_px, raw)
        self.assertIs(out_extent, extent)
        self.assertEqual(out_norm, normalized_box(raw, FRAME))

    def test_negative_cutoff_is_off_not_inverted(self):
        self.assertFalse(GroupSmoother(-2.0).enabled)


class TestGatesAreNeverSmoothed(unittest.TestCase):
    """count and held are branch conditions for the receiving end (§3).

    A count that eases from 2 to 3 spends several frames at 2.4 dancers, and a
    fractional held stops being a gate at all.
    """

    def test_count_and_held_pass_through_intact(self):
        smoother = GroupSmoother(1.5)
        counts, helds = [], []
        for index in range(30):
            # Count steps 2 -> 4 mid-run; held flips on for the tail.
            count = 2 if index < 10 else 4
            held = index >= 20
            extent = GroupExtent(count=count, width=2.0, depth=1.0,
                                 cx=0.0, cz=4.0, held=held)
            out, _ = smoother.apply(extent, None, index * 0.1)
            counts.append(out.count)
            helds.append(out.held)
        self.assertEqual(set(counts), {2, 4})
        self.assertTrue(all(isinstance(c, int) for c in counts))
        self.assertEqual(set(helds), {False, True})
        self.assertTrue(all(isinstance(h, bool) for h in helds))


class TestJitterIsReduced(unittest.TestCase):
    def test_a_still_box_with_detector_noise_settles(self):
        smoother = GroupSmoother(1.5)
        # A stationary group whose right edge rattles +/- 20 px, which is what a
        # confident-but-noisy YOLO box does frame to frame.
        boxes = [box_px(400, 300, 1200 + (20 if i % 2 else -20), 900)
                 for i in range(40)]
        norms = feed(smoother, boxes)
        tail = [n["x2"] for n in norms[20:]]
        raw_swing = 40.0 / FRAME[1]
        self.assertLess(max(tail) - min(tail), raw_swing * 0.5)

    def test_a_real_move_still_arrives(self):
        """Smoothing must not turn a crossing into a value that never gets there."""
        smoother = GroupSmoother(1.5)
        boxes = [box_px(400, 300, 1200, 900)] * 5
        boxes += [box_px(900, 300, 1700, 900)] * 40
        norms = feed(smoother, boxes)
        self.assertAlmostEqual(norms[-1]["x1"], 900 / FRAME[1], places=2)


class TestDerivedValuesStayConsistent(unittest.TestCase):
    """The derived fields must agree with the smoothed primitives.

    Filtering w/h/cx/cy or area as signals in their own right would let them
    drift away from the corners and widths they are supposed to describe -- and
    a patch reading box_w and box_x1/x2 would see two different rectangles.
    """

    def test_box_w_h_c_match_the_smoothed_corners(self):
        smoother = GroupSmoother(1.5)
        boxes = [box_px(400 + 10 * i, 300, 1200 + 5 * i, 900) for i in range(15)]
        for norm in feed(smoother, boxes):
            self.assertAlmostEqual(norm["w"], norm["x2"] - norm["x1"], places=12)
            self.assertAlmostEqual(norm["h"], norm["y2"] - norm["y1"], places=12)
            self.assertAlmostEqual(norm["cx"], (norm["x1"] + norm["x2"]) / 2, places=12)
            self.assertAlmostEqual(norm["cy"], (norm["y1"] + norm["y2"]) / 2, places=12)

    def test_area_still_equals_width_times_depth(self):
        smoother = GroupSmoother(1.5)
        for index in range(15):
            extent = GroupExtent(count=3, width=2.0 + 0.1 * index,
                                 depth=1.5, cx=0.0, cz=4.0)
            out, _ = smoother.apply(extent, None, index * 0.1)
            self.assertAlmostEqual(out.area, out.width * out.depth, places=12)


class TestCornerOrdering(unittest.TestCase):
    """Each corner gets its own speed-adaptive cutoff, so unlike a shared-alpha
    filter this one can cross x2 under x1 on a fast shrink. A negative box_w on
    the wire looks like a real measurement."""

    def test_width_never_goes_negative_on_a_collapse(self):
        smoother = GroupSmoother(1.5)
        boxes = [box_px(200, 300, 1800, 900)] * 10      # wide
        boxes += [box_px(950, 300, 960, 900)] * 10      # collapses hard
        for norm in feed(smoother, boxes):
            self.assertGreaterEqual(norm["x2"], norm["x1"])
            self.assertGreaterEqual(norm["y2"], norm["y1"])
            self.assertGreaterEqual(norm["w"], 0.0)
            self.assertGreaterEqual(norm["h"], 0.0)


class TestResetBoundaries(unittest.TestCase):
    def test_empty_stage_clears_the_state(self):
        """Otherwise the next entrance slides in from wherever the cast left."""
        smoother = GroupSmoother(1.5)
        feed(smoother, [box_px(100, 300, 400, 900)] * 20)
        smoother.apply(GroupExtent(count=0), None, 2.0)

        entrance = box_px(1500, 300, 1800, 900)
        _, _, norm = smoothed_group_outputs(smoother, GroupExtent(count=1),
                                            entrance, FRAME, 2.1)
        self.assertEqual(norm, normalized_box(entrance, FRAME))

    def test_reconfiguring_the_cutoff_restarts_the_filter(self):
        smoother = GroupSmoother(1.5)
        feed(smoother, [box_px(100, 300, 400, 900)] * 20)
        smoother.configure(0.8)
        moved = box_px(1500, 300, 1800, 900)
        _, _, norm = smoothed_group_outputs(smoother, None, moved, FRAME, 5.0)
        self.assertEqual(norm, normalized_box(moved, FRAME))

    def test_configure_none_is_a_no_op(self):
        smoother = GroupSmoother(1.5)
        smoother.configure(None)
        self.assertEqual(smoother.cutoff, 1.5)


class TestPreviewMatchesTheWire(unittest.TestCase):
    """The drawn rectangle and the OSC values have to be the same rectangle, or
    watching the preview stops being a way to check what is being sent."""

    def test_pixel_box_is_the_normalised_box(self):
        smoother = GroupSmoother(1.5)
        boxes = [box_px(400 + 25 * i, 300, 1200 + 25 * i, 900) for i in range(12)]
        for index, raw in enumerate(boxes):
            _, px, norm = smoothed_group_outputs(smoother, None, raw, FRAME, index * 0.1)
            self.assertEqual(px, denormalized_box(norm, FRAME))

    def test_normalisation_round_trips(self):
        raw = box_px(133, 271, 1444, 907)
        for value, original in zip(denormalized_box(normalized_box(raw, FRAME), FRAME), raw):
            self.assertAlmostEqual(value, original, places=9)

    def test_denormalized_box_handles_missing_input(self):
        self.assertIsNone(denormalized_box(None, FRAME))
        self.assertIsNone(denormalized_box({"x1": 0, "y1": 0, "x2": 1, "y2": 1}, (0, 0)))


class TestResolutionIndependence(unittest.TestCase):
    """Filtering happens in 0-1 units so a performance preset that changes
    capture resolution does not change what the tuning means."""

    def test_same_motion_smooths_the_same_at_two_resolutions(self):
        hd, sd = (1080, 1920), (540, 960)
        hd_smoother, sd_smoother = GroupSmoother(1.5), GroupSmoother(1.5)
        hd_out, sd_out = [], []
        for index in range(20):
            shift = 20 * index
            _, _, hd_norm = smoothed_group_outputs(
                hd_smoother, None, box_px(400 + shift, 300, 1200 + shift, 900),
                hd, index * 0.1)
            _, _, sd_norm = smoothed_group_outputs(
                sd_smoother, None, box_px(200 + shift / 2, 150, 600 + shift / 2, 450),
                sd, index * 0.1)
            hd_out.append(hd_norm["x1"])
            sd_out.append(sd_norm["x1"])
        for hd_value, sd_value in zip(hd_out, sd_out):
            self.assertAlmostEqual(hd_value, sd_value, places=9)


class TestFrameConversion(unittest.TestCase):
    """The UI talks in analysis frames, like the per-metric smoothness sliders."""

    def test_zero_and_one_frames_are_off(self):
        # 1 is off for the same reason it is on the metric sliders: a one-frame
        # average is the identity.
        self.assertEqual(cutoff_for_frames(0), 0.0)
        self.assertEqual(cutoff_for_frames(1), 0.0)
        self.assertFalse(GroupSmoother(cutoff_for_frames(0)).enabled)
        self.assertFalse(GroupSmoother(cutoff_for_frames(1)).enabled)

    def test_more_frames_means_heavier(self):
        cutoffs = [cutoff_for_frames(n) for n in range(2, MAX_FRAMES + 1)]
        self.assertEqual(cutoffs, sorted(cutoffs, reverse=True))
        self.assertTrue(all(c > 0 for c in cutoffs))

    @staticmethod
    def _ema_fraction_covered(frames, steps):
        """Fraction of a step a plain 2/(N+1) EMA has covered after n frames."""
        alpha = 2.0 / (frames + 1.0)
        covered = 0.0
        for _ in range(steps):
            covered = alpha * 1.0 + (1.0 - alpha) * covered
        return covered

    def _one_euro_fraction_covered(self, frames, jump):
        """Same measurement through the real filter, for a step of ``jump``
        (in normalised units) on the box's right edge."""
        smoother = GroupSmoother(cutoff_for_frames(frames))
        width = FRAME[1]
        start, end = 0.5 * width, (0.5 + jump) * width
        boxes = [box_px(0, 0, start, 540)] + [box_px(0, 0, end, 540)] * 6
        norms = feed(smoother, boxes, step=NOMINAL_ANALYSIS_DT)
        return (norms[-1]["x2"] - 0.5) / jump

    def test_matches_the_equivalent_ema_while_the_box_is_near_still(self):
        """N frames here means what N frames means on the metric sliders.

        Only near-still: One-Euro's whole point is that it stops matching a
        fixed alpha once the box moves (see the next test). Checked through the
        real filter rather than the formula, so a change to either side has to
        survive the other.
        """
        frames = 5
        self.assertAlmostEqual(
            self._one_euro_fraction_covered(frames, jump=0.001),
            self._ema_fraction_covered(frames, steps=6),
            places=2,
        )

    def test_a_fast_move_beats_the_equivalent_ema(self):
        """The reason this is One-Euro and not the metrics' fixed-alpha EMA: a
        real crossing must not pay the lag that killing jitter at rest costs."""
        frames = 5
        self.assertGreater(
            self._one_euro_fraction_covered(frames, jump=0.4),
            self._ema_fraction_covered(frames, steps=6) + 0.02,
        )

    def test_out_of_range_input_is_survivable(self):
        self.assertEqual(cutoff_for_frames(None), 0.0)
        self.assertEqual(cutoff_for_frames("nonsense"), 0.0)
        self.assertEqual(cutoff_for_frames(-4), 0.0)
        # Above the slider's top the filter should not keep getting heavier.
        self.assertEqual(cutoff_for_frames(999), cutoff_for_frames(MAX_FRAMES))

    def test_default_is_a_real_setting(self):
        self.assertTrue(GroupSmoother(cutoff_for_frames(DEFAULT_FRAMES)).enabled)


if __name__ == "__main__":
    unittest.main()
