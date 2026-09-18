from __future__ import annotations

"""Serializable description of one conservative cache invalidation."""

from dataclasses import dataclass

from cubelab.state_slot_propagation import StateSlotKey


@dataclass(frozen=True, slots=True)
class CacheDelta:
    event_type: str
    candidate_id: str | None
    piece_id: str | None
    dirty_slots: tuple[StateSlotKey, ...]
    dirty_move_families: tuple[tuple[str, str], ...]
    previous_version: int
    next_version: int

    def row(self) -> dict[str, object]:
        return {
            "event_type": self.event_type,
            "candidate_id": self.candidate_id,
            "piece_id": self.piece_id,
            "dirty_slots": [
                {
                    "move": key.move,
                    "piece_type": key.piece_type,
                    "state_before_position": key.state_before_position,
                }
                for key in self.dirty_slots
            ],
            "dirty_move_families": [
                list(value) for value in self.dirty_move_families
            ],
            "previous_version": self.previous_version,
            "next_version": self.next_version,
        }


__all__ = ["CacheDelta"]
