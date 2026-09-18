from __future__ import annotations

"""Candidate-domain and reversible propagation state."""

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Mapping, Sequence

from cubelab.move_demand_balance import ColumnInventory, MoveDemandVector

from .provenance import ProvenanceGraph
from .requirements import AnyRequirement, ObligationRequirement
from .types import ConstraintStatus


@dataclass(slots=True)
class CandidateDomain:
    piece_id: str
    candidates: dict[str, MoveDemandVector]
    original_size: int
    removal_reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_candidates(
        cls,
        piece_id: str,
        candidates: Sequence[MoveDemandVector],
    ) -> "CandidateDomain":
        mapping: dict[str, MoveDemandVector] = {}
        for candidate in candidates:
            if candidate.piece_id != piece_id:
                raise ValueError(
                    f"candidate {candidate.candidate_id} belongs to "
                    f"{candidate.piece_id}, not {piece_id}"
                )
            if candidate.candidate_id in mapping:
                raise ValueError(
                    f"duplicate candidate id: {candidate.candidate_id}"
                )
            mapping[candidate.candidate_id] = candidate
        return cls(
            piece_id=piece_id,
            candidates=mapping,
            original_size=len(mapping),
        )

    def remove(self, candidate_id: str, reason: str) -> bool:
        if candidate_id not in self.candidates:
            return False
        del self.candidates[candidate_id]
        previous = self.removal_reasons.get(candidate_id, ())
        self.removal_reasons[candidate_id] = previous + (reason,)
        return True

    def row(self) -> dict[str, object]:
        return {
            "piece_id": self.piece_id,
            "original_size": self.original_size,
            "current_size": len(self.candidates),
            "candidate_ids": sorted(self.candidates),
            "removal_reasons": {
                key: list(value)
                for key, value in sorted(self.removal_reasons.items())
            },
        }


@dataclass(frozen=True, slots=True)
class PropagationSnapshot:
    domain_candidates: Mapping[str, Mapping[str, MoveDemandVector]]
    removal_reasons: Mapping[str, Mapping[str, tuple[str, ...]]]
    selected_candidates: Mapping[str, MoveDemandVector]
    generated_requirements: Mapping[str, AnyRequirement]
    obligation_ledger: Mapping[str, ObligationRequirement]
    artifacts: Mapping[str, object]
    unknown_event_count: int
    status: ConstraintStatus
    provenance_snapshot: tuple[int, int]


@dataclass(slots=True)
class PropagationState:
    domains: dict[str, CandidateDomain]
    inventory: ColumnInventory | None = None
    contracts: Mapping[str, object] = field(default_factory=dict)
    selected_candidates: dict[str, MoveDemandVector] = field(
        default_factory=dict
    )
    generated_requirements: dict[str, AnyRequirement] = field(
        default_factory=dict
    )
    obligation_ledger: dict[str, ObligationRequirement] = field(
        default_factory=dict
    )
    artifacts: dict[str, object] = field(default_factory=dict)
    unknown_events: list[dict[str, object]] = field(default_factory=list)
    status: ConstraintStatus = ConstraintStatus.SAT
    provenance: ProvenanceGraph = field(default_factory=ProvenanceGraph)

    @classmethod
    def from_mapping(
        cls,
        domains: Mapping[str, Sequence[MoveDemandVector]],
        *,
        inventory: ColumnInventory | None = None,
        contracts: Mapping[str, object] | None = None,
    ) -> "PropagationState":
        return cls(
            domains={
                piece: CandidateDomain.from_candidates(piece, values)
                for piece, values in domains.items()
            },
            inventory=inventory,
            contracts={} if contracts is None else dict(contracts),
        )

    @property
    def domain_size(self) -> int:
        return sum(len(domain.candidates) for domain in self.domains.values())

    @property
    def complete(self) -> bool:
        return (
            len(self.selected_candidates) == len(self.domains)
            and all(len(domain.candidates) == 1 for domain in self.domains.values())
        )

    def unselected_domains(
        self,
    ) -> dict[str, tuple[MoveDemandVector, ...]]:
        return {
            piece: tuple(domain.candidates.values())
            for piece, domain in self.domains.items()
            if piece not in self.selected_candidates
        }

    def selected(self) -> tuple[MoveDemandVector, ...]:
        return tuple(
            self.selected_candidates[piece]
            for piece in sorted(self.selected_candidates)
        )

    def assign(self, piece_id: str, candidate_id: str) -> bool:
        domain = self.domains[piece_id]
        candidate = domain.candidates.get(candidate_id)
        if candidate is None:
            raise ValueError(
                f"candidate {candidate_id} is not in domain {piece_id}"
            )
        previous = self.selected_candidates.get(piece_id)
        if previous is not None:
            if previous.candidate_id != candidate_id:
                raise ValueError(f"piece {piece_id} already assigned")
            return False
        for other_id in tuple(domain.candidates):
            if other_id != candidate_id:
                domain.remove(other_id, f"ASSIGNED:{candidate_id}")
        self.selected_candidates[piece_id] = candidate
        return True

    def add_requirement(self, requirement: AnyRequirement) -> bool:
        key = requirement.requirement_id
        if key in self.generated_requirements:
            return False
        self.generated_requirements[key] = requirement
        if isinstance(requirement, ObligationRequirement):
            self.obligation_ledger[key] = requirement
        return True

    def snapshot(self) -> PropagationSnapshot:
        return PropagationSnapshot(
            domain_candidates={
                piece: dict(domain.candidates)
                for piece, domain in self.domains.items()
            },
            removal_reasons={
                piece: dict(domain.removal_reasons)
                for piece, domain in self.domains.items()
            },
            selected_candidates=dict(self.selected_candidates),
            generated_requirements=dict(self.generated_requirements),
            obligation_ledger=dict(self.obligation_ledger),
            artifacts=dict(self.artifacts),
            unknown_event_count=len(self.unknown_events),
            status=self.status,
            provenance_snapshot=self.provenance.snapshot(),
        )

    def restore(self, snapshot: PropagationSnapshot) -> None:
        for piece, candidates in snapshot.domain_candidates.items():
            self.domains[piece].candidates = dict(candidates)
            self.domains[piece].removal_reasons = dict(
                snapshot.removal_reasons[piece]
            )
        self.selected_candidates = dict(snapshot.selected_candidates)
        self.generated_requirements = dict(snapshot.generated_requirements)
        self.obligation_ledger = dict(snapshot.obligation_ledger)
        self.artifacts = dict(snapshot.artifacts)
        del self.unknown_events[snapshot.unknown_event_count :]
        self.status = snapshot.status
        self.provenance.restore(snapshot.provenance_snapshot)

    def state_hash(self) -> str:
        payload = {
            "domains": {
                piece: sorted(domain.candidates)
                for piece, domain in sorted(self.domains.items())
            },
            "selected": {
                piece: candidate.candidate_id
                for piece, candidate in sorted(
                    self.selected_candidates.items()
                )
            },
            "requirements": sorted(self.generated_requirements),
            "obligations": sorted(self.obligation_ledger),
            "artifact_keys": sorted(self.artifacts),
            "unknown_count": len(self.unknown_events),
            "status": self.status.value,
            "provenance": self.provenance.state_hash_payload(),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def row(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "inventory": (
                None if self.inventory is None else self.inventory.row()
            ),
            "domain_size": self.domain_size,
            "domains": {
                piece: domain.row()
                for piece, domain in sorted(self.domains.items())
            },
            "selected_candidates": {
                piece: value.candidate_id
                for piece, value in sorted(
                    self.selected_candidates.items()
                )
            },
            "generated_requirements": [
                value.row()
                for _, value in sorted(
                    self.generated_requirements.items()
                )
            ],
            "obligation_ledger": [
                value.row()
                for _, value in sorted(self.obligation_ledger.items())
            ],
            "unknown_events": list(self.unknown_events),
            "artifacts": sorted(self.artifacts),
        }


__all__ = [
    "CandidateDomain",
    "PropagationSnapshot",
    "PropagationState",
]
