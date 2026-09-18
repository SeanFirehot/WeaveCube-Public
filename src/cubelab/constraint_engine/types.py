from __future__ import annotations

"""Shared status, result, context, and protocol types for v101.5."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Mapping, Protocol, Sequence

from cubelab.move_demand_balance import ColumnInventory, MoveDemandVector

from .requirements import AnyRequirement


class ConstraintStatus(str, Enum):
    SAT = "sat"
    SAT_WITH_REQUIREMENTS = "sat_with_requirements"
    UNSAT_PROVEN = "unsat_proven"
    UNKNOWN_CAP = "unknown_cap"
    HEURISTIC_ONLY = "heuristic_only"


class ConstraintTier(str, Enum):
    HARD = "hard"
    SOUND_RELAXATION = "sound_relaxation"
    HEURISTIC = "heuristic"
    BOUNDED_UNKNOWN = "bounded_unknown"


@dataclass(frozen=True, slots=True)
class CandidateRemoval:
    piece_id: str
    candidate_id: str
    reason: str
    requirement_ids: tuple[str, ...] = ()

    def row(self) -> dict[str, object]:
        payload = asdict(self)
        payload["requirement_ids"] = list(self.requirement_ids)
        return payload


@dataclass(frozen=True, slots=True)
class ForcedAssignment:
    piece_id: str
    candidate_id: str
    reason: str
    requirement_ids: tuple[str, ...] = ()

    def row(self) -> dict[str, object]:
        payload = asdict(self)
        payload["requirement_ids"] = list(self.requirement_ids)
        return payload


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    constraint_name: str
    status: ConstraintStatus
    tier: ConstraintTier
    requirements: tuple[AnyRequirement, ...] = ()
    removals: tuple[CandidateRemoval, ...] = ()
    forced_assignments: tuple[ForcedAssignment, ...] = ()
    runtime_ns: int = 0
    domain_size_before: int = 0
    domain_size_after: int = 0
    rejection_reason: str | None = None
    unknown_reason: str | None = None
    consumed_requirement_ids: tuple[str, ...] = ()
    conflict_candidate_ids: tuple[str, ...] = ()
    metrics: Mapping[str, int | float | str | bool | None] = field(
        default_factory=dict
    )

    @property
    def removed_candidate_ids(self) -> tuple[str, ...]:
        return tuple(value.candidate_id for value in self.removals)

    @property
    def forced_candidate_ids(self) -> tuple[str, ...]:
        return tuple(
            value.candidate_id for value in self.forced_assignments
        )

    def row(self) -> dict[str, object]:
        return {
            "constraint_name": self.constraint_name,
            "status": self.status.value,
            "tier": self.tier.value,
            "requirements": [value.row() for value in self.requirements],
            "removals": [value.row() for value in self.removals],
            "forced_assignments": [
                value.row() for value in self.forced_assignments
            ],
            "runtime_ns": self.runtime_ns,
            "domain_size_before": self.domain_size_before,
            "domain_size_after": self.domain_size_after,
            "rejection_reason": self.rejection_reason,
            "unknown_reason": self.unknown_reason,
            "consumed_requirement_ids": list(
                self.consumed_requirement_ids
            ),
            "conflict_candidate_ids": list(
                self.conflict_candidate_ids
            ),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class ConstraintContext:
    inventory: ColumnInventory | None = None
    contracts: Mapping[str, object] = field(default_factory=dict)
    residue_state_cap: int = 100_000
    state_slot_node_cap: int = 100_000
    coloring_node_cap: int = 50_000
    packing_node_cap: int = 100_000
    linearization_node_cap: int = 100_000
    exact_packing_domain_threshold: int = 20
    solution_limit: int = 64
    max_fixpoint_rounds: int = 100


class IncrementalConstraint(Protocol):
    name: str
    tier: ConstraintTier
    estimated_cost_class: int

    def initialize(self, context: ConstraintContext) -> None:
        ...

    def push_candidate(self, candidate: MoveDemandVector) -> None:
        ...

    def pop_candidate(self, candidate: MoveDemandVector) -> None:
        ...

    def propagate(self, state: object) -> ConstraintResult:
        ...

    def state_hash_payload(self) -> object:
        ...


@dataclass(frozen=True, slots=True)
class CascadeOutcome:
    status: ConstraintStatus
    fixpoint_rounds: int
    results: tuple[ConstraintResult, ...]
    conflict_core_id: str | None
    unknown_count: int
    domain_size_before: int
    domain_size_after: int

    def row(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "fixpoint_rounds": self.fixpoint_rounds,
            "results": [value.row() for value in self.results],
            "conflict_core_id": self.conflict_core_id,
            "unknown_count": self.unknown_count,
            "domain_size_before": self.domain_size_before,
            "domain_size_after": self.domain_size_after,
        }


def selected_vectors(
    values: Mapping[str, MoveDemandVector],
) -> tuple[MoveDemandVector, ...]:
    return tuple(values[piece] for piece in sorted(values))


def flatten_domains(
    domains: Mapping[str, Sequence[MoveDemandVector]],
) -> tuple[MoveDemandVector, ...]:
    return tuple(
        candidate
        for piece in sorted(domains)
        for candidate in domains[piece]
    )


__all__ = [
    "ConstraintStatus",
    "ConstraintTier",
    "CandidateRemoval",
    "ForcedAssignment",
    "ConstraintResult",
    "ConstraintContext",
    "IncrementalConstraint",
    "CascadeOutcome",
    "selected_vectors",
    "flatten_domains",
]
