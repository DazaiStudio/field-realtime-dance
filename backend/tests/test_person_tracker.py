import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from person_tracker import (
    MultiPersonTrackRegistry,
    PersonTrack,
    StableTrackSelector,
    bbox_area,
    expand_bbox,
    suppress_duplicate_person_tracks,
)


class TestStableTrackSelector(unittest.TestCase):
    def test_keeps_locked_track_even_when_larger_person_appears(self):
        selector = StableTrackSelector(hold_seconds=0.5)
        first = PersonTrack(1, (10, 10, 80, 180), 0.8)
        chosen, state = selector.choose([first], now=1.0)
        self.assertEqual(chosen.track_id, 1)
        self.assertEqual(state, "tracking")

        larger = PersonTrack(2, (0, 0, 300, 300), 0.9)
        chosen, state = selector.choose([larger, first], now=1.1)
        self.assertEqual(chosen.track_id, 1)
        self.assertEqual(state, "tracking")

    def test_holds_recent_bbox_during_short_occlusion(self):
        selector = StableTrackSelector(hold_seconds=0.5)
        first = PersonTrack(7, (10, 10, 80, 180), 0.8)
        selector.choose([first], now=1.0)

        chosen, state = selector.choose([], now=1.3)
        self.assertEqual(chosen.track_id, 7)
        self.assertEqual(state, "holding")

    def test_relocks_after_hold_expires(self):
        selector = StableTrackSelector(hold_seconds=0.5)
        selector.choose([PersonTrack(7, (10, 10, 80, 180), 0.8)], now=1.0)

        chosen, state = selector.choose([PersonTrack(9, (400, 400, 600, 600), 0.8)], now=5.0)
        self.assertEqual(chosen.track_id, 9)
        self.assertEqual(state, "tracking")
        self.assertEqual(selector.stable_id, 2)

    def test_reidentifies_nearby_new_raw_id_as_same_stable_id(self):
        selector = StableTrackSelector(hold_seconds=0.5, reidentify_seconds=3.0)
        selector.choose([PersonTrack(1, (100, 100, 200, 300), 0.8)], now=1.0)

        chosen, state = selector.choose([PersonTrack(3, (115, 165, 225, 330), 0.7)], now=2.2)
        self.assertEqual(chosen.track_id, 3)
        self.assertEqual(state, "reidentified")
        self.assertEqual(selector.stable_id, 1)


class TestTrackGeometry(unittest.TestCase):
    def test_bbox_area_clamps_negative_dimensions(self):
        self.assertEqual(bbox_area((10, 10, 5, 5)), 0.0)

    def test_expand_bbox_clamps_to_frame(self):
        rect = expand_bbox((-10, -5, 30, 50), (100, 120, 3), padding=0.2, min_size=16)
        self.assertIsNotNone(rect)
        x1, y1, x2, y2 = rect
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, 120)
        self.assertLessEqual(y2, 100)
        self.assertGreater(x2 - x1, 0)
        self.assertGreater(y2 - y1, 0)


class TestDuplicateSuppression(unittest.TestCase):
    def test_drops_partial_body_box_inside_full_body_box(self):
        tracks = suppress_duplicate_person_tracks([
            PersonTrack(1, (100, 100, 500, 900), 0.9),
            PersonTrack(2, (260, 240, 450, 700), 0.5),
        ])

        self.assertEqual([track.track_id for track in tracks], [1])

    def test_keeps_two_separate_people(self):
        tracks = suppress_duplicate_person_tracks([
            PersonTrack(1, (100, 100, 500, 900), 0.9),
            PersonTrack(2, (520, 120, 900, 910), 0.8),
        ])

        self.assertEqual([track.track_id for track in tracks], [1, 2])


class TestMultiPersonTrackRegistry(unittest.TestCase):
    def test_assigns_stable_ids_to_multiple_people(self):
        registry = MultiPersonTrackRegistry()
        tracks = registry.update([
            PersonTrack(10, (0, 0, 100, 200), 0.8),
            PersonTrack(20, (300, 0, 420, 220), 0.9),
        ], now=1.0)

        self.assertEqual([track.stable_id for track in tracks], [1, 2])
        self.assertEqual([track.raw_id for track in tracks], [10, 20])

    def test_manual_selection_returns_requested_stable_id(self):
        registry = MultiPersonTrackRegistry()
        registry.update([
            PersonTrack(10, (0, 0, 100, 200), 0.8),
            PersonTrack(20, (300, 0, 420, 220), 0.9),
        ], now=1.0)

        active, state = registry.choose_active("id:2", frame_shape=(720, 1280, 3))
        self.assertEqual(active.stable_id, 2)
        self.assertEqual(state, "tracking")

    def test_reidentifies_changed_raw_id_without_changing_stable_id(self):
        registry = MultiPersonTrackRegistry(hold_seconds=0.5, reidentify_seconds=3.0)
        registry.update([PersonTrack(10, (100, 100, 200, 320), 0.8)], now=1.0)
        registry.update([], now=1.8)
        tracks = registry.update([PersonTrack(99, (110, 145, 220, 335), 0.7)], now=2.1)

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].stable_id, 1)
        self.assertEqual(tracks[0].raw_id, 99)

    def test_reidentifies_after_longer_gap(self):
        registry = MultiPersonTrackRegistry(hold_seconds=0.5, reidentify_seconds=8.0)
        registry.update([
            PersonTrack(10, (100, 100, 300, 800), 0.8),
            PersonTrack(20, (500, 100, 700, 800), 0.8),
        ], now=1.0)
        registry.update([PersonTrack(10, (105, 110, 305, 810), 0.8)], now=2.0)
        tracks = registry.update([
            PersonTrack(10, (110, 110, 310, 810), 0.8),
            PersonTrack(99, (455, 120, 690, 800), 0.7),
        ], now=7.0)

        by_raw = {track.raw_id: track.stable_id for track in tracks if track.raw_id is not None}
        self.assertEqual(by_raw[99], 2)

    def test_registry_filters_partial_duplicate_tracks(self):
        registry = MultiPersonTrackRegistry()
        tracks = registry.update([
            PersonTrack(10, (100, 100, 500, 900), 0.9),
            PersonTrack(20, (260, 240, 450, 700), 0.5),
        ], now=1.0)

        self.assertEqual(len([track for track in tracks if track.state == "tracking"]), 1)

    def test_auto_center_prefers_track_closest_to_frame_center(self):
        registry = MultiPersonTrackRegistry()
        registry.update([
            PersonTrack(10, (0, 0, 100, 200), 0.9),
            PersonTrack(20, (590, 280, 690, 440), 0.6),
        ], now=1.0)

        active, _state = registry.choose_active("auto_center", frame_shape=(720, 1280, 3))
        self.assertEqual(active.stable_id, 2)


class TestAutoLargestStickiness(unittest.TestCase):
    """auto_largest must not flip the active dancer on every frame: two
    similar-sized dancers would otherwise swap the metrics subject
    repeatedly, spiking velocity/jerk over OSC."""

    def test_keeps_incumbent_until_challenger_clearly_larger(self):
        registry = MultiPersonTrackRegistry()
        a = PersonTrack(1, (0, 0, 100, 100), 0.9)  # area 10000
        registry.update([a], now=0.0)
        track, _state = registry.choose_active("auto_largest")
        self.assertEqual(track.stable_id, 1)

        b_slightly_larger = PersonTrack(2, (300, 0, 405, 100), 0.9)  # area 10500
        registry.update([a, b_slightly_larger], now=0.1)
        track, _state = registry.choose_active("auto_largest")
        self.assertEqual(track.stable_id, 1)  # 1.05x must not steal the lock

        b_clearly_larger = PersonTrack(2, (300, 0, 440, 100), 0.9)  # area 14000
        registry.update([a, b_clearly_larger], now=0.2)
        track, _state = registry.choose_active("auto_largest")
        self.assertEqual(track.stable_id, 2)  # 1.4x takes over

    def test_switches_when_incumbent_disappears(self):
        registry = MultiPersonTrackRegistry()
        a = PersonTrack(1, (0, 0, 100, 100), 0.9)
        b = PersonTrack(2, (300, 0, 380, 100), 0.9)  # smaller
        registry.update([a, b], now=0.0)
        registry.choose_active("auto_largest")

        registry.update([b], now=0.5)  # A occluded -> holding
        track, _state = registry.choose_active("auto_largest")
        self.assertEqual(track.stable_id, 2)


if __name__ == "__main__":
    unittest.main()
