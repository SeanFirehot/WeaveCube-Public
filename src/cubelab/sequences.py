from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from .orientation import OrientationAnalyzer
from .transformations import PIECES, Transformation, from_sequence


def _quarter_turns(token: str) -> tuple[str, int]:
    if not token:
        raise ValueError("Move token cannot be empty")
    face = token[0]
    suffix = token[1:]
    if suffix == "":
        turns = 1
    elif suffix == "2":
        turns = 2
    elif suffix == "'":
        turns = 3
    else:
        raise ValueError(f"Unsupported move token: {token}")
    return face, turns


def _token(face: str, turns: int) -> str | None:
    turns %= 4
    if turns == 0:
        return None
    if turns == 1:
        return face
    if turns == 2:
        return f"{face}2"
    return f"{face}'"


def reduce_sequence(sequence: Iterable[str]) -> Tuple[str, ...]:
    """Reduce adjacent turns of the same face modulo four.

    This is intentionally conservative. It performs only identities that are
    valid without reordering moves, so the full cube effect is guaranteed to be
    preserved when the implementation is correct.
    """
    stack: list[tuple[str, int]] = []
    for raw in sequence:
        face, turns = _quarter_turns(raw)
        if stack and stack[-1][0] == face:
            _, previous = stack.pop()
            combined = (previous + turns) % 4
            if combined:
                stack.append((face, combined))
        else:
            stack.append((face, turns))
    return tuple(token for face, turns in stack if (token := _token(face, turns)) is not None)


def full_effect_signature(transformation: Transformation) -> tuple:
    orientation = OrientationAnalyzer.analyze(transformation)
    return (
        tuple(transformation.mapping[piece] for piece in PIECES),
        tuple((change.piece, change.position, change.twist) for change in orientation.corner_changes),
        tuple((change.piece, change.position, change.orientation) for change in orientation.edge_changes),
    )


def full_effect_equal(left: Transformation, right: Transformation) -> bool:
    return full_effect_signature(left) == full_effect_signature(right)


@dataclass(frozen=True)
class SequenceReductionResult:
    original: Transformation
    reduced: Transformation
    original_sequence: Tuple[str, ...]
    reduced_sequence: Tuple[str, ...]
    removed_move_count: int
    effect_preserved: bool


def reduce_transformation(transformation: Transformation) -> SequenceReductionResult:
    reduced_sequence = reduce_sequence(transformation.sequence)
    reduced = from_sequence(reduced_sequence)
    preserved = full_effect_equal(transformation, reduced)
    return SequenceReductionResult(
        original=transformation,
        reduced=reduced,
        original_sequence=transformation.sequence,
        reduced_sequence=reduced_sequence,
        removed_move_count=len(transformation.sequence) - len(reduced_sequence),
        effect_preserved=preserved,
    )


def inverse_sequence(sequence: Iterable[str]) -> Tuple[str, ...]:
    result = []
    for token in reversed(tuple(sequence)):
        face, turns = _quarter_turns(token)
        inverse = _token(face, -turns)
        if inverse is not None:
            result.append(inverse)
    return tuple(result)


def is_exact_inverse(left: Transformation, right: Transformation) -> bool:
    return full_effect_equal(from_sequence(inverse_sequence(left.sequence)), right)


__all__ = [
    "SequenceReductionResult",
    "reduce_sequence",
    "reduce_transformation",
    "full_effect_signature",
    "full_effect_equal",
    "inverse_sequence",
    "is_exact_inverse",
]
