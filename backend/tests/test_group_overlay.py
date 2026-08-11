import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from group_extent import GroupExtent, measure_extent, union_bbox
from group_overlay import draw_group_box, normalized_box


def blank(h=720, w=1280):
    return np.zeros((h, w, 3), dtype=np.uint8)


def ink(frame) -> int:
    return int(np.count_nonzero(frame.any(axis=2)))


class TestUnionBbox(unittest.TestCase):
    def test_encloses_every_person_box(self):
        box = union_bbox([(100, 200, 300, 600), (500, 150, 700, 620)])
        self.assertEqual(box, (100.0, 150.0, 700.0, 620.0))

    def test_single_person_is_their_own_box(self):
        self.assertEqual(union_bbox([(10, 20, 30, 40)]), (10.0, 20.0, 30.0, 40.0))

    def test_no_people_is_none(self):
        self.assertIsNone(union_bbox([]))
        self.assertIsNone(union_bbox(None))

    def test_skips_none_and_non_finite_boxes(self):
        box = union_bbox([None, (10, 20, 30, 40), (np.nan, 1, 2, 3)])
        self.assertEqual(box, (10.0, 20.0, 30.0, 40.0))


class TestDrawGroupBox(unittest.TestCase):
    def test_draws_for_a_group(self):
        frame = blank()
        draw_group_box(frame, (200, 150, 900, 640), measure_extent([(-1.0, 3.0), (1.0, 5.0)]))
        self.assertGreater(ink(frame), 0)

    def test_one_person_still_gets_a_box(self):
        """On screen a single dancer has a real, visible rectangle -- unlike
        the floor extent, which would collapse to a point."""
        frame = blank()
        draw_group_box(frame, (400, 200, 700, 640), measure_extent([(0.5, 4.0)]))
        self.assertGreater(ink(frame), 0)

    def test_box_is_drawn_where_the_people_are(self):
        left, right = blank(), blank()
        draw_group_box(left, (100, 200, 400, 600), None)
        draw_group_box(right, (800, 200, 1100, 600), None)
        self.assertGreater(ink(left[:, :500]), 0)
        self.assertEqual(ink(left[:, 600:]), 0)
        self.assertGreater(ink(right[:, 700:]), 0)
        self.assertEqual(ink(right[:, :700]), 0)

    def test_held_box_differs_from_live(self):
        base = measure_extent([(-1.0, 3.0), (1.0, 5.0)])
        held = GroupExtent(base.count, base.width, base.depth, base.cx, base.cz, True)
        a, b = blank(), blank()
        draw_group_box(a, (200, 150, 900, 640), base)
        draw_group_box(b, (200, 150, 900, 640), held)
        self.assertFalse(np.array_equal(a, b))

    def test_none_box_is_a_no_op(self):
        frame = blank()
        draw_group_box(frame, None, measure_extent([(0.0, 4.0)]))
        self.assertEqual(ink(frame), 0)

    def test_degenerate_box_is_a_no_op_not_a_crash(self):
        frame = blank()
        draw_group_box(frame, (300, 300, 300, 300), None)
        self.assertEqual(ink(frame), 0)

    def test_box_beyond_the_frame_is_clamped(self):
        frame = blank()
        draw_group_box(frame, (-500, -400, 5000, 4000), None)
        self.assertGreater(ink(frame), 0)   # clamped and drawn, no exception

    def test_label_survives_a_box_at_the_top_edge(self):
        frame = blank()
        draw_group_box(frame, (100, 0, 500, 300), measure_extent([(0.0, 3.0), (1.0, 4.0)]))
        self.assertGreater(ink(frame), 0)


class TestNormalizedBox(unittest.TestCase):
    def test_fractions_of_the_frame(self):
        norm = normalized_box((320, 180, 960, 540), (720, 1280, 3))
        self.assertAlmostEqual(norm["x1"], 0.25, places=6)
        self.assertAlmostEqual(norm["y1"], 0.25, places=6)
        self.assertAlmostEqual(norm["x2"], 0.75, places=6)
        self.assertAlmostEqual(norm["y2"], 0.75, places=6)
        self.assertAlmostEqual(norm["w"], 0.5, places=6)
        self.assertAlmostEqual(norm["h"], 0.5, places=6)
        self.assertAlmostEqual(norm["cx"], 0.5, places=6)
        self.assertAlmostEqual(norm["cy"], 0.5, places=6)

    def test_resolution_independent(self):
        """The point of normalising: a performance-preset resolution change
        must not move the numbers a patch is mapped to."""
        small = normalized_box((160, 90, 480, 270), (360, 640, 3))
        large = normalized_box((320, 180, 960, 540), (720, 1280, 3))
        for key in small:
            self.assertAlmostEqual(small[key], large[key], places=6)

    def test_none_inputs(self):
        self.assertIsNone(normalized_box(None, (720, 1280, 3)))
        self.assertIsNone(normalized_box((0, 0, 10, 10), None))


if __name__ == "__main__":
    unittest.main()
