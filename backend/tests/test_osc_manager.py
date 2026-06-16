import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osc_manager import MultiSlotOSC


class TestMultiSlotOSC(unittest.TestCase):
    def test_per_slot_addresses(self):
        m = MultiSlotOSC(num_slots=4, enabled=False, alpha=1.0)
        self.assertEqual(m.sender(1).metric_address("energy"), "/field/1/energy")
        self.assertEqual(m.sender(3).metric_address("sync_velocity"), "/field/3/sync_vel")

    def test_state_isolated_between_slots(self):
        m = MultiSlotOSC(num_slots=4, enabled=False, alpha=1.0, mode="normalize")
        m.sender(1)._prepare_value("energy", 10.0)
        v2 = m.sender(2)._prepare_value("energy", 5.0)
        self.assertAlmostEqual(v2, 1.0)

    def test_configure_broadcasts(self):
        m = MultiSlotOSC(num_slots=4, enabled=False, alpha=1.0)
        m.configure(host="10.0.0.9", port=9001)
        for s in (1, 2, 3, 4):
            self.assertEqual(m.sender(s).host, "10.0.0.9")
            self.assertEqual(m.sender(s).port, 9001)


if __name__ == "__main__":
    unittest.main()
