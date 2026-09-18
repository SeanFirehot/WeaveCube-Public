"""Teacher-free symbolic canonical fixed-bound languages."""

from __future__ import annotations

from hashlib import sha256

from cubelab.ato.regular_support import FACE_ORDER, MOVE_NAMES, canonical_allow

from .symbolic_mdd_v2 import SymbolicMDD


CANONICAL_V356_HASH = sha256(
    b"production-canonical:same-face-forbidden:ordered-opposites:v37.356"
).hexdigest()


def canonical_language(
    horizon: int,
    *,
    previous_face: str | None = None,
    node_cap: int = 5_000_000,
) -> SymbolicMDD:
    sentinel = previous_face

    def transition(state: str | None, move_id: int) -> str:
        return MOVE_NAMES[move_id][0]

    # Canonical legality is already enforced by SymbolicMDD's shared context.
    return SymbolicMDD.from_automaton(
        sentinel,
        int(horizon),
        transition,
        lambda _state: True,
        previous_face=previous_face,
        scope_hash=sha256(
            f"{CANONICAL_V356_HASH}:{horizon}:{previous_face}".encode()
        ).hexdigest(),
        node_cap=node_cap,
        construction="CANONICAL_FIXED_BOUND",
    )


__all__ = ["CANONICAL_V356_HASH", "canonical_language"]
