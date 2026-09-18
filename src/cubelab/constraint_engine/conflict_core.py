from __future__ import annotations

"""Small, reproducible local conflict cores."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

from .types import ConstraintResult


@dataclass(frozen=True, slots=True)
class ConflictCore:
    core_id: str
    constraint_name: str
    conflict_type: str
    piece_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    requirement_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    minimality_status: str

    def row(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("piece_ids", "candidate_ids", "requirement_ids", "reasons"):
            payload[key] = list(payload[key])
        return payload


def extract_conflict_core(result: ConstraintResult) -> ConflictCore:
    candidate_ids = tuple(dict.fromkeys(result.conflict_candidate_ids))
    piece_ids = tuple(
        dict.fromkeys(
            removal.piece_id
            for removal in result.removals
            if removal.candidate_id in candidate_ids or not candidate_ids
        )
    )
    requirement_ids = tuple(
        dict.fromkeys(
            requirement.requirement_id
            for requirement in result.requirements
        )
    )
    reasons = tuple(
        dict.fromkeys(
            value
            for value in (
                result.rejection_reason,
                *(
                    removal.reason for removal in result.removals
                ),
            )
            if value
        )
    )
    payload = {
        "constraint": result.constraint_name,
        "candidates": candidate_ids,
        "requirements": requirement_ids,
        "reasons": reasons,
    }
    core_id = (
        "CORE-"
        + sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:20]
    )
    return ConflictCore(
        core_id=core_id,
        constraint_name=result.constraint_name,
        conflict_type=(
            result.rejection_reason
            or f"{result.constraint_name.upper()}_CONFLICT"
        ),
        piece_ids=piece_ids,
        candidate_ids=candidate_ids,
        requirement_ids=requirement_ids,
        reasons=reasons,
        minimality_status="LOCAL_DEDUPLICATED_CORE",
    )


__all__ = ["ConflictCore", "extract_conflict_core"]
