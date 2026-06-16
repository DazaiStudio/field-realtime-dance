from typing import Dict, List, Optional


class SlotBinder:
    """Maps volatile tracker ids onto a fixed set of slots (1..num_slots).

    - Auto-assigns a new track to the lowest free slot.
    - Frees a slot whose bound track has been absent for > evict_after updates.
    - manual_bind / swap let an operator override the mapping.
    """

    def __init__(self, num_slots: int = 4, evict_after: int = 15):
        self.num_slots = num_slots
        self.evict_after = evict_after
        self.slot_to_track: Dict[int, Optional[int]] = {s: None for s in range(1, num_slots + 1)}
        self.missing: Dict[int, int] = {s: 0 for s in range(1, num_slots + 1)}

    def _track_to_slot(self) -> Dict[int, int]:
        return {t: s for s, t in self.slot_to_track.items() if t is not None}

    def _free_slots(self) -> List[int]:
        return [s for s in range(1, self.num_slots + 1) if self.slot_to_track[s] is None]

    def update(self, present_track_ids: List[int]) -> Dict[int, int]:
        present = list(dict.fromkeys(present_track_ids))  # de-dupe, keep order
        t2s = self._track_to_slot()

        for t in present:
            if t not in t2s:
                free = self._free_slots()
                if not free:
                    continue
                slot = free[0]
                self.slot_to_track[slot] = t
        t2s = self._track_to_slot()

        present_set = set(present)
        for slot, track in list(self.slot_to_track.items()):
            if track is None:
                continue
            if track in present_set:
                self.missing[slot] = 0
            else:
                self.missing[slot] += 1
                if self.missing[slot] >= self.evict_after:
                    self.slot_to_track[slot] = None
                    self.missing[slot] = 0

        final = self._track_to_slot()
        return {t: final[t] for t in present if t in final}

    def manual_bind(self, track_id: int, slot: int) -> None:
        if slot not in self.slot_to_track:
            return
        for s, t in self.slot_to_track.items():
            if t == track_id:
                self.slot_to_track[s] = None
                self.missing[s] = 0
        self.slot_to_track[slot] = track_id
        self.missing[slot] = 0

    def swap(self, slot_a: int, slot_b: int) -> None:
        if slot_a in self.slot_to_track and slot_b in self.slot_to_track:
            self.slot_to_track[slot_a], self.slot_to_track[slot_b] = (
                self.slot_to_track[slot_b], self.slot_to_track[slot_a])
            self.missing[slot_a] = self.missing[slot_b] = 0

    def active_slots(self) -> List[int]:
        return [s for s in range(1, self.num_slots + 1) if self.slot_to_track[s] is not None]
