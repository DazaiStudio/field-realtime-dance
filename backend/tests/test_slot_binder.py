import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slot_binder import SlotBinder


class TestSlotBinder(unittest.TestCase):
    def test_auto_assigns_lowest_free_slot(self):
        b = SlotBinder(num_slots=4, evict_after=2)
        mapping = b.update([101, 102])
        self.assertEqual(mapping, {101: 1, 102: 2})
        self.assertEqual(b.active_slots(), [1, 2])

    def test_existing_track_keeps_its_slot(self):
        b = SlotBinder(num_slots=4, evict_after=2)
        b.update([101, 102])
        mapping = b.update([102, 101])
        self.assertEqual(mapping, {102: 2, 101: 1})

    def test_slot_evicted_after_threshold(self):
        b = SlotBinder(num_slots=4, evict_after=2)
        b.update([101])
        b.update([])
        self.assertEqual(b.active_slots(), [1])
        b.update([])
        self.assertEqual(b.active_slots(), [])
        self.assertEqual(b.update([200]), {200: 1})

    def test_manual_bind_moves_track(self):
        b = SlotBinder(num_slots=4, evict_after=2)
        b.update([101, 102])
        b.manual_bind(101, 3)
        mapping = b.update([101, 102])
        self.assertEqual(mapping[101], 3)
        self.assertEqual(mapping[102], 2)

    def test_swap_exchanges_two_slots(self):
        b = SlotBinder(num_slots=4, evict_after=2)
        b.update([101, 102])
        b.swap(1, 2)
        mapping = b.update([101, 102])
        self.assertEqual(mapping[101], 2)
        self.assertEqual(mapping[102], 1)


if __name__ == "__main__":
    unittest.main()
