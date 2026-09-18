from __future__ import annotations

"""Canonical, conservative key for exact joint-support suffix states."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json


@dataclass(frozen=True, slots=True)
class JointSupportSubproblemKey:
    """Every field can affect the exact suffix result.

    The current solver couples all move/state slots at once.  Consequently
    ``slot_keys_signature`` and ``slot_deficits`` cover the full joint vector,
    rather than unsoundly treating each four-slot move family as independent.
    """

    remaining_piece_domain_ids: tuple[str, ...]
    remaining_domain_signatures: tuple[str, ...]
    slot_keys_signature: tuple[tuple[str, str, str], ...]
    slot_deficits: tuple[int, ...]
    remaining_move_target: int | None
    selected_slot_signature: tuple[int, ...]
    contract_signature: str
    extension_signature: str
    cap_policy_id: str
    ordering_policy: str

    @property
    def key_hash(self) -> str:
        return sha256(
            json.dumps(
                asdict(self),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def row(self) -> dict[str, object]:
        payload = asdict(self)
        payload["key_hash"] = self.key_hash
        payload["remaining_piece_domain_ids"] = list(
            self.remaining_piece_domain_ids
        )
        payload["remaining_domain_signatures"] = list(
            self.remaining_domain_signatures
        )
        payload["slot_keys_signature"] = [
            list(value) for value in self.slot_keys_signature
        ]
        payload["slot_deficits"] = list(self.slot_deficits)
        payload["selected_slot_signature"] = list(
            self.selected_slot_signature
        )
        return payload


__all__ = ["JointSupportSubproblemKey"]
