from __future__ import annotations

"""Immutable exact-DP input objects for state-slot joint support."""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json

from cubelab.state_slot_propagation import StateSlotKey

from .subproblem_key import JointSupportSubproblemKey


def _digest(payload: object) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class JointSupportCandidate:
    candidate_id: str
    counts: tuple[int, ...]

    def row(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "counts": list(self.counts),
        }


@dataclass(frozen=True, slots=True)
class JointSupportPieceDomain:
    piece_id: str
    candidates: tuple[JointSupportCandidate, ...]
    _domain_signature: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_domain_signature",
            _digest(
                [
                    candidate.row()
                    for candidate in self.candidates
                ]
            ),
        )

    @property
    def domain_signature(self) -> str:
        return self._domain_signature

    def restrict(
        self,
        candidate_id: str,
    ) -> "JointSupportPieceDomain":
        return JointSupportPieceDomain(
            piece_id=self.piece_id,
            candidates=tuple(
                candidate
                for candidate in self.candidates
                if candidate.candidate_id == candidate_id
            ),
        )

    def row(self) -> dict[str, object]:
        return {
            "piece_id": self.piece_id,
            "domain_signature": self.domain_signature,
            "candidates": [
                candidate.row() for candidate in self.candidates
            ],
        }


@dataclass(frozen=True, slots=True)
class JointSupportProblem:
    slot_keys: tuple[StateSlotKey, ...]
    domains: tuple[JointSupportPieceDomain, ...]
    initial_deficits: tuple[int, ...]
    selected_slot_signature: tuple[int, ...]
    contract_signature: str = "EXACT"
    extension_signature: str = "NONE"
    cap_policy_id: str = "NODE_CAP_V1"
    ordering_policy: str = "smallest_domain"

    def key(
        self,
        depth: int,
        deficits: tuple[int, ...],
    ) -> JointSupportSubproblemKey:
        suffix = self.domains[depth:]
        return JointSupportSubproblemKey(
            remaining_piece_domain_ids=tuple(
                domain.piece_id for domain in suffix
            ),
            remaining_domain_signatures=tuple(
                domain.domain_signature for domain in suffix
            ),
            slot_keys_signature=tuple(
                (
                    key.move,
                    key.piece_type,
                    key.state_before_position,
                )
                for key in self.slot_keys
            ),
            slot_deficits=deficits,
            remaining_move_target=sum(deficits),
            selected_slot_signature=self.selected_slot_signature,
            contract_signature=self.contract_signature,
            extension_signature=self.extension_signature,
            cap_policy_id=self.cap_policy_id,
            ordering_policy=self.ordering_policy,
        )

    def with_forced_candidate(
        self,
        piece_id: str,
        candidate_id: str,
    ) -> "JointSupportProblem":
        return JointSupportProblem(
            slot_keys=self.slot_keys,
            domains=tuple(
                (
                    domain.restrict(candidate_id)
                    if domain.piece_id == piece_id
                    else domain
                )
                for domain in self.domains
            ),
            initial_deficits=self.initial_deficits,
            selected_slot_signature=self.selected_slot_signature,
            contract_signature=self.contract_signature,
            extension_signature=self.extension_signature,
            cap_policy_id=self.cap_policy_id,
            ordering_policy=self.ordering_policy,
        )

    def row(self) -> dict[str, object]:
        return {
            "slot_keys": [asdict(key) for key in self.slot_keys],
            "domains": [domain.row() for domain in self.domains],
            "initial_deficits": list(self.initial_deficits),
            "selected_slot_signature": list(
                self.selected_slot_signature
            ),
            "contract_signature": self.contract_signature,
            "extension_signature": self.extension_signature,
            "cap_policy_id": self.cap_policy_id,
            "ordering_policy": self.ordering_policy,
        }


__all__ = [
    "JointSupportCandidate",
    "JointSupportPieceDomain",
    "JointSupportProblem",
]
