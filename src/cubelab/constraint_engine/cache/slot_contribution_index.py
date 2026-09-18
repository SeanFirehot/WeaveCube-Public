from __future__ import annotations

"""Precomputed candidate-to-state-slot incidence index."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

from cubelab.move_demand_balance import MoveDemandVector
from cubelab.state_slot_propagation import StateSlotKey
from cubelab.transformations import PIECES


@dataclass(frozen=True, slots=True)
class CandidateSlotContribution:
    candidate_id: str
    piece_id: str
    piece_type: str
    move: str
    state_before_position: str
    orientation_before: int | None
    active_ordinal: int
    multiplicity: int

    @property
    def slot_key(self) -> StateSlotKey:
        return StateSlotKey(
            self.move,
            self.piece_type,
            self.state_before_position,
        )

    def row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateSlotContributionIndex:
    contributions: tuple[CandidateSlotContribution, ...]
    candidate_counts: Mapping[str, Mapping[StateSlotKey, int]]
    candidate_to_slots: Mapping[str, tuple[StateSlotKey, ...]]
    slot_to_candidates: Mapping[StateSlotKey, tuple[str, ...]]
    piece_to_slots: Mapping[str, tuple[StateSlotKey, ...]]
    candidate_to_piece: Mapping[str, str]

    @classmethod
    def build(
        cls,
        candidates: Iterable[MoveDemandVector],
    ) -> "CandidateSlotContributionIndex":
        unique: dict[str, MoveDemandVector] = {}
        for candidate in candidates:
            previous = unique.get(candidate.candidate_id)
            if previous is not None and previous != candidate:
                raise ValueError(
                    f"candidate id collision: {candidate.candidate_id}"
                )
            unique[candidate.candidate_id] = candidate

        rows: list[CandidateSlotContribution] = []
        candidate_counts: dict[str, dict[StateSlotKey, int]] = {}
        slot_to_candidates: dict[StateSlotKey, set[str]] = defaultdict(set)
        piece_to_slots: dict[str, set[StateSlotKey]] = defaultdict(set)
        candidate_to_piece: dict[str, str] = {}
        for candidate_id, candidate in sorted(unique.items()):
            counter: Counter[StateSlotKey] = Counter()
            candidate_to_piece[candidate_id] = candidate.piece_id
            for ordinal, event in enumerate(candidate.active_events, start=1):
                position = PIECES[event.state_before[0]]
                key = StateSlotKey(
                    event.move,
                    event.piece_type,
                    position,
                )
                counter[key] += 1
                orientation = (
                    int(event.state_before[1])
                    if len(event.state_before) > 1
                    else None
                )
                rows.append(
                    CandidateSlotContribution(
                        candidate_id=candidate_id,
                        piece_id=candidate.piece_id,
                        piece_type=event.piece_type,
                        move=event.move,
                        state_before_position=position,
                        orientation_before=orientation,
                        active_ordinal=ordinal,
                        multiplicity=1,
                    )
                )
            candidate_counts[candidate_id] = dict(counter)
            for key in counter:
                slot_to_candidates[key].add(candidate_id)
                piece_to_slots[candidate.piece_id].add(key)

        return cls(
            contributions=tuple(rows),
            candidate_counts=candidate_counts,
            candidate_to_slots={
                candidate_id: tuple(sorted(counts))
                for candidate_id, counts in sorted(
                    candidate_counts.items()
                )
            },
            slot_to_candidates={
                key: tuple(sorted(candidate_ids))
                for key, candidate_ids in sorted(
                    slot_to_candidates.items()
                )
            },
            piece_to_slots={
                piece: tuple(sorted(keys))
                for piece, keys in sorted(piece_to_slots.items())
            },
            candidate_to_piece=candidate_to_piece,
        )

    def contribution(
        self,
        candidate_id: str,
        key: StateSlotKey,
    ) -> int:
        return int(self.candidate_counts.get(candidate_id, {}).get(key, 0))

    def all_slots(self) -> tuple[StateSlotKey, ...]:
        return tuple(sorted(self.slot_to_candidates))

    def row(self) -> dict[str, object]:
        return {
            "candidate_count": len(self.candidate_counts),
            "slot_count": len(self.slot_to_candidates),
            "contribution_event_count": len(self.contributions),
            "contributions": [value.row() for value in self.contributions],
            "candidate_to_slots": {
                candidate_id: [
                    {
                        "move": key.move,
                        "piece_type": key.piece_type,
                        "state_before_position": key.state_before_position,
                    }
                    for key in keys
                ]
                for candidate_id, keys in sorted(
                    self.candidate_to_slots.items()
                )
            },
        }


__all__ = [
    "CandidateSlotContribution",
    "CandidateSlotContributionIndex",
]
