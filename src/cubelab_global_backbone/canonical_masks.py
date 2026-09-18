"""Declarative last-two-move canonical masks for 18 HTM moves."""

from __future__ import annotations

from cubelab.pdcc import MOVE_ORDER, MOVES


SENTINEL = None
HISTORY = (SENTINEL, *MOVE_ORDER)
MOVE_INDEX = {move: index for index, move in enumerate(MOVE_ORDER)}


def commute(left: str | None, right: str | None) -> bool:
    return left is not None and right is not None and MOVES[left].axis == MOVES[right].axis


def allowed(last2: str | None, last1: str | None, move: str) -> bool:
    if last1 is not None and MOVES[last1].face == MOVES[move].face:
        return False
    if (
        last2 is not None
        and last1 is not None
        and commute(last2, last1)
        and MOVES[last2].face == MOVES[move].face
    ):
        return False
    if commute(last1, move) and MOVE_INDEX[last1] >= MOVE_INDEX[move]:  # type: ignore[index]
        return False
    return True


def build_masks() -> dict[tuple[str | None, str | None], int]:
    return {
        (last2, last1): sum(
            1 << MOVE_INDEX[move]
            for move in MOVE_ORDER
            if allowed(last2, last1, move)
        )
        for last2 in HISTORY
        for last1 in HISTORY
    }


CANONICAL_MASKS = build_masks()


def allowed_moves(last2: str | None, last1: str | None) -> tuple[str, ...]:
    mask = CANONICAL_MASKS[(last2, last1)]
    return tuple(move for move in MOVE_ORDER if mask & (1 << MOVE_INDEX[move]))
