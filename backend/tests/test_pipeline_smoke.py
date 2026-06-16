import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slot_binder import SlotBinder
from osc_manager import MultiSlotOSC
from dance_metrics import DanceMetricsEngine
from keypoint_mapping import coco17_to_h36m17


class TestPipelineSmoke(unittest.TestCase):
    def test_two_people_get_isolated_slots_and_metrics(self):
        binder = SlotBinder(num_slots=4)
        osc = MultiSlotOSC(num_slots=4, enabled=False, alpha=1.0)
        engines = {}
        rng = np.random.default_rng(0)
        for _frame in range(5):
            people = {7: rng.random((17, 2)) * 100, 12: rng.random((17, 2)) * 100}
            mapping = binder.update(list(people))
            for tid, slot in mapping.items():
                eng = engines.setdefault(slot, DanceMetricsEngine(fps=30, is_3d=False))
                m = eng.update(coco17_to_h36m17(people[tid]))
                osc.send_slot(slot, m)
        self.assertEqual(sorted(binder.active_slots()), [1, 2])
        self.assertIn("energy", osc.prepared_for(1))
        self.assertIn("energy", osc.prepared_for(2))


if __name__ == "__main__":
    unittest.main()
