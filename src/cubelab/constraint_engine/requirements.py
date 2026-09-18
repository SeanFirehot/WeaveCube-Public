from __future__ import annotations

"""Typed, canonical requirements emitted by the v101.5 cascade."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class Requirement:
    source_constraint: str
    sound: bool
    explanation: str

    @property
    def requirement_type(self) -> str:
        return type(self).__name__

    @property
    def requirement_id(self) -> str:
        payload = {
            "type": self.requirement_type,
            "payload": asdict(self),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        return f"REQ-{sha256(encoded).hexdigest()[:20]}"

    def row(self) -> dict[str, object]:
        return {
            "requirement_id": self.requirement_id,
            "requirement_type": self.requirement_type,
            **asdict(self),
        }


@dataclass(frozen=True, slots=True)
class MoveResidueRequirement(Requirement):
    piece_type: str
    move: str
    required_residue_mod4: int
    minimum_additional_count: int
    maximum_additional_count: int | None


@dataclass(frozen=True, slots=True)
class CornerEdgeBalanceRequirement(Requirement):
    move: str
    current_corner_count: int
    current_edge_count: int
    required_delta_direction: str


@dataclass(frozen=True, slots=True)
class StateSlotRequirement(Requirement):
    move: str
    piece_type: str
    required_state_before: tuple[object, ...]
    required_multiplicity: int


@dataclass(frozen=True, slots=True)
class TransitionRequirement(Requirement):
    piece_id: str | None
    move: str
    required_state_before: tuple[int, int]
    required_state_after: tuple[int, int]
    allowed_candidate_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PrecedenceRequirement(Requirement):
    predecessor_event_id: str
    successor_event_id: str
    strict: bool


@dataclass(frozen=True, slots=True)
class ColumnMultiplicityRequirement(Requirement):
    move: str
    minimum_columns: int
    reason_event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ObligationRequirement(Requirement):
    piece_id: str
    obligation_type: str
    required_exit_state: tuple[int, int] | None
    allowed_gap_domain: tuple[object, ...]


AnyRequirement: TypeAlias = (
    MoveResidueRequirement
    | CornerEdgeBalanceRequirement
    | StateSlotRequirement
    | TransitionRequirement
    | PrecedenceRequirement
    | ColumnMultiplicityRequirement
    | ObligationRequirement
)


__all__ = [
    "Requirement",
    "MoveResidueRequirement",
    "CornerEdgeBalanceRequirement",
    "StateSlotRequirement",
    "TransitionRequirement",
    "PrecedenceRequirement",
    "ColumnMultiplicityRequirement",
    "ObligationRequirement",
    "AnyRequirement",
]
