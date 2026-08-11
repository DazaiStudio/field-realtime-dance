import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from group_extent import (
    GroupExtentTracker,
    hip_floor_positions,
    measure_extent,
)
from keypoint_mapping import _K4_L_HIP, _K4_R_HIP, k4abt32_to_h36m17


def make_body(x_mm, z_mm, confidence=2, y_mm=0.0, hip_half_width=150.0):
    """A K4ABT-shaped (32, 4) body standing at (x, z) on the floor."""
    joints = np.zeros((32, 4), dtype=float)
    joints[:, 3] = confidence
    joints[:, 0] = x_mm
    joints[:, 1] = y_mm
    joints[:, 2] = z_mm
    joints[_K4_L_HIP] = (x_mm - hip_half_width, y_mm, z_mm, confidence)
    joints[_K4_R_HIP] = (x_mm + hip_half_width, y_mm, z_mm, confidence)

    class _Body:
        pass

    body = _Body()
    body.joints = joints
    return body


class TestRootCenteringTrap(unittest.TestCase):
    """The reason this module reads raw joints instead of the H36M arrays."""

    def test_h36m_arrays_lose_absolute_position(self):
        far_apart = [make_body(-2000.0, 4000.0), make_body(2000.0, 4000.0)]
        centered = [k4abt32_to_h36m17(b.joints) for b in far_apart]
        # Both dancers' pelvises land on the origin, so any bbox built from
        # H36M would measure zero no matter how far apart they stand.
        for skeleton in centered:
            np.testing.assert_allclose(skeleton[0], np.zeros(3), atol=1e-9)

        # Raw joints keep the 4 m separation the bbox is supposed to report.
        extent = measure_extent(hip_floor_positions(far_apart))
        self.assertAlmostEqual(extent.width, 4.0, places=6)


class TestHipFloorPositions(unittest.TestCase):
    def test_converts_mm_to_metres_on_the_floor_plane(self):
        positions = hip_floor_positions([make_body(1500.0, 3000.0)])
        self.assertEqual(len(positions), 1)
        x, z = positions[0]
        self.assertAlmostEqual(x, 1.5, places=6)
        self.assertAlmostEqual(z, 3.0, places=6)

    def test_mirror_flips_x_only(self):
        body = make_body(1500.0, 3000.0)
        (x, z) = hip_floor_positions([body], mirrored=True)[0]
        self.assertAlmostEqual(x, -1.5, places=6)
        self.assertAlmostEqual(z, 3.0, places=6)

    def test_drops_bodies_with_none_confidence_hips(self):
        good = make_body(0.0, 3000.0, confidence=2)
        guessed = make_body(9000.0, 3000.0, confidence=0)
        positions = hip_floor_positions([good, guessed])
        self.assertEqual(len(positions), 1)
        self.assertAlmostEqual(positions[0][0], 0.0, places=6)

    def test_drops_non_finite_hips(self):
        broken = make_body(0.0, 3000.0)
        broken.joints[_K4_L_HIP, 0] = np.nan
        self.assertEqual(hip_floor_positions([broken]), [])

    def test_empty_input(self):
        self.assertEqual(hip_floor_positions([]), [])
        self.assertEqual(hip_floor_positions(None), [])


class TestMeasureExtent(unittest.TestCase):
    def test_bbox_over_four_dancers(self):
        extent = measure_extent([(-1.0, 3.0), (1.0, 3.0), (0.0, 5.0), (0.5, 4.0)])
        self.assertEqual(extent.count, 4)
        self.assertAlmostEqual(extent.width, 2.0, places=6)
        self.assertAlmostEqual(extent.depth, 2.0, places=6)
        self.assertAlmostEqual(extent.cx, 0.0, places=6)
        self.assertAlmostEqual(extent.cz, 4.0, places=6)

    def test_single_dancer_is_a_point_at_their_position(self):
        extent = measure_extent([(1.2, 4.85)])
        self.assertEqual(extent.count, 1)
        self.assertAlmostEqual(extent.width, 0.0, places=6)
        self.assertAlmostEqual(extent.depth, 0.0, places=6)
        self.assertAlmostEqual(extent.cx, 1.2, places=6)
        self.assertAlmostEqual(extent.cz, 4.85, places=6)

    def test_empty_is_count_zero(self):
        extent = measure_extent([])
        self.assertEqual(extent.count, 0)
        self.assertFalse(extent.held)


class TestGroupExtentTrackerDropoutHold(unittest.TestCase):
    """§4a protection: a lost dancer must not read as the group closing up."""

    def test_holds_last_extent_when_a_body_drops_out(self):
        tracker = GroupExtentTracker(hold_seconds=1.0)
        both = [(-1.0, 4.0), (1.0, 4.0)]
        measured = tracker.update(both, now=0.0)
        self.assertAlmostEqual(measured.width, 2.0, places=6)
        self.assertFalse(measured.held)

        # One dancer drops out. Without the hold this would report width 0 --
        # indistinguishable from the pair standing on the same spot.
        held = tracker.update([(-1.0, 4.0)], now=0.2)
        self.assertAlmostEqual(held.width, 2.0, places=6)
        self.assertTrue(held.held)
        self.assertEqual(held.count, 2)

    def test_accepts_the_lower_count_once_the_hold_expires(self):
        tracker = GroupExtentTracker(hold_seconds=1.0)
        tracker.update([(-1.0, 4.0), (1.0, 4.0)], now=0.0)
        tracker.update([(-1.0, 4.0)], now=0.5)
        real_exit = tracker.update([(-1.0, 4.0)], now=1.6)
        self.assertFalse(real_exit.held)
        self.assertEqual(real_exit.count, 1)
        self.assertAlmostEqual(real_exit.width, 0.0, places=6)

    def test_recovery_resumes_measuring_immediately(self):
        tracker = GroupExtentTracker(hold_seconds=1.0)
        tracker.update([(-1.0, 4.0), (1.0, 4.0)], now=0.0)
        tracker.update([(-1.0, 4.0)], now=0.2)
        back = tracker.update([(-1.5, 4.0), (1.5, 4.0)], now=0.4)
        self.assertFalse(back.held)
        self.assertAlmostEqual(back.width, 3.0, places=6)

    def test_a_genuine_huddle_is_not_masked_by_the_hold(self):
        # Count stays at 2, so closing up is measured, not held.
        tracker = GroupExtentTracker(hold_seconds=1.0)
        tracker.update([(-1.0, 4.0), (1.0, 4.0)], now=0.0)
        close = tracker.update([(-0.1, 4.0), (0.1, 4.0)], now=0.2)
        self.assertFalse(close.held)
        self.assertAlmostEqual(close.width, 0.2, places=6)

    def test_empty_stage_reports_zero_after_the_hold(self):
        tracker = GroupExtentTracker(hold_seconds=1.0)
        tracker.update([(-1.0, 4.0), (1.0, 4.0)], now=0.0)
        self.assertTrue(tracker.update([], now=0.3).held)
        gone = tracker.update([], now=2.0)
        self.assertEqual(gone.count, 0)
        self.assertFalse(gone.held)

    def test_extra_bodies_beyond_max_people_do_not_inflate_the_bbox(self):
        tracker = GroupExtentTracker(hold_seconds=1.0, max_people=4)
        cast = [(-1.0, 4.0), (-0.4, 4.0), (0.4, 4.0), (1.0, 4.0)]
        stray = (12.0, 4.0)  # a reflection or someone in the wings
        extent = tracker.update(cast + [stray], now=0.0)
        self.assertEqual(extent.count, 4)
        self.assertAlmostEqual(extent.width, 2.0, places=6)

    def test_reset_clears_held_state(self):
        tracker = GroupExtentTracker(hold_seconds=1.0)
        tracker.update([(-1.0, 4.0), (1.0, 4.0)], now=0.0)
        tracker.reset()
        fresh = tracker.update([(0.0, 4.0)], now=0.1)
        self.assertFalse(fresh.held)
        self.assertEqual(fresh.count, 1)


class TestSizeParameters(unittest.TestCase):
    def test_area_is_width_times_depth(self):
        extent = measure_extent([(-1.0, 3.0), (1.0, 5.0)])
        self.assertAlmostEqual(extent.width, 2.0, places=6)
        self.assertAlmostEqual(extent.depth, 2.0, places=6)
        self.assertAlmostEqual(extent.area, 4.0, places=6)

    def test_area_collapses_when_two_dancers_line_up(self):
        # The documented trap: 3 m apart but on one Z line -> zero area.
        # diagonal is the honest size signal for a pair.
        extent = measure_extent([(-1.5, 4.0), (1.5, 4.0)])
        self.assertAlmostEqual(extent.area, 0.0, places=6)
        self.assertAlmostEqual(extent.diagonal, 3.0, places=6)

    def test_diagonal_is_the_box_corner_span(self):
        extent = measure_extent([(0.0, 0.0), (3.0, 4.0)])
        self.assertAlmostEqual(extent.diagonal, 5.0, places=6)

    def test_aspect_describes_formation_shape(self):
        wide = measure_extent([(-2.0, 4.0), (2.0, 4.5)])
        deep = measure_extent([(-0.25, 3.0), (0.25, 6.0)])
        self.assertGreater(wide.aspect, 1.0)
        self.assertLess(deep.aspect, 1.0)

    def test_aspect_is_clamped_not_infinite_on_a_flat_box(self):
        flat = measure_extent([(-1.5, 4.0), (1.5, 4.0)])
        self.assertTrue(np.isfinite(flat.aspect))
        self.assertEqual(flat.aspect, 999.0)

    def test_single_dancer_has_zero_size_everywhere(self):
        one = measure_extent([(1.0, 4.0)])
        self.assertEqual(one.area, 0.0)
        self.assertEqual(one.diagonal, 0.0)
        self.assertEqual(one.aspect, 1.0)


class TestUnitScaling(unittest.TestCase):
    def test_metres_is_the_default(self):
        payload = measure_extent([(-1.0, 3.0), (1.0, 5.0)]).as_osc()
        self.assertAlmostEqual(payload["group/width"], 2.0, places=6)
        self.assertEqual(payload["group/units"], 1.0)

    def test_centimetres_scale_lengths_by_100(self):
        payload = measure_extent([(-1.0, 3.0), (1.0, 5.0)]).as_osc(units="cm")
        self.assertAlmostEqual(payload["group/width"], 200.0, places=6)
        self.assertAlmostEqual(payload["group/depth"], 200.0, places=6)
        self.assertAlmostEqual(payload["group/diagonal"], 282.842712, places=4)
        self.assertAlmostEqual(payload["group/cz"], 400.0, places=6)
        self.assertEqual(payload["group/units"], 100.0)

    def test_area_scales_by_the_square(self):
        """m2 -> cm2 is 10000, not 100. Getting this wrong is invisible in the
        UI (the number just looks big) but wrong on the wire."""
        payload = measure_extent([(-1.0, 3.0), (1.0, 5.0)]).as_osc(units="cm")
        self.assertAlmostEqual(payload["group/area"], 40000.0, places=3)

    def test_aspect_is_unitless(self):
        metres = measure_extent([(-2.0, 4.0), (2.0, 4.5)]).as_osc()
        centis = measure_extent([(-2.0, 4.0), (2.0, 4.5)]).as_osc(units="cm")
        self.assertAlmostEqual(metres["group/aspect"], centis["group/aspect"], places=9)

    def test_count_and_held_are_never_scaled(self):
        payload = measure_extent([(-1.0, 3.0), (1.0, 5.0)]).as_osc(units="cm")
        self.assertEqual(payload["group/count"], 2.0)
        self.assertEqual(payload["group/held"], 0.0)

    def test_unknown_unit_falls_back_to_metres(self):
        payload = measure_extent([(-1.0, 3.0), (1.0, 5.0)]).as_osc(units="furlongs")
        self.assertAlmostEqual(payload["group/width"], 2.0, places=6)
        self.assertEqual(payload["group/units"], 1.0)


class TestOscPayload(unittest.TestCase):
    def test_as_osc_names_are_id_free_and_values_float(self):
        extent = measure_extent([(-1.0, 3.0), (1.0, 5.0)])
        payload = extent.as_osc()
        self.assertEqual(
            set(payload),
            {"group/count", "group/width", "group/depth", "group/area",
             "group/diagonal", "group/aspect", "group/cx", "group/cz",
             "group/units", "group/held"},
        )
        for value in payload.values():
            self.assertIsInstance(value, float)
        self.assertEqual(payload["group/count"], 2.0)
        self.assertEqual(payload["group/held"], 0.0)
        self.assertAlmostEqual(payload["group/area"], 4.0, places=6)


if __name__ == "__main__":
    unittest.main()
