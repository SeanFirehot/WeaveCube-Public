"""HTM face-turn definitions and parsing for the PDCC engine."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

from .group import (
    CUBE_ROTATIONS,
    Matrix3,
    Vector3,
    matrix_mul,
    rotation_x,
    rotation_y,
    rotation_z,
    transpose,
)


@dataclass(frozen=True, slots=True)
class MoveSpec:
    name: str
    face: str
    rotation_id: int
    axis: int
    layer: int
    amount: int  # +1 quarter, -1 inverse quarter, 2 half turn in notation space

    def is_active_position(self, position: Vector3) -> bool:
        return position[self.axis] == self.layer


# Clockwise when looking directly at the named face from outside the cube.
_BASE_QUARTER: Mapping[str, tuple[Matrix3, int, int]] = MappingProxyType(
    {
        "U": (rotation_y(-1), 1, 1),
        "D": (rotation_y(1), 1, -1),
        "R": (rotation_x(-1), 0, 1),
        "L": (rotation_x(1), 0, -1),
        "F": (rotation_z(-1), 2, 1),
        "B": (rotation_z(1), 2, -1),
    }
)


def _build_moves() -> tuple[tuple[str, ...], Mapping[str, MoveSpec]]:
    order: list[str] = []
    specs: dict[str, MoveSpec] = {}
    for face in "URFDLB":
        base_matrix, axis, layer = _BASE_QUARTER[face]
        for suffix, amount in (("", 1), ("'", -1), ("2", 2)):
            if amount == 1:
                matrix = base_matrix
            elif amount == -1:
                matrix = transpose(base_matrix)
            else:
                matrix = matrix_mul(base_matrix, base_matrix)
            name = f"{face}{suffix}"
            order.append(name)
            specs[name] = MoveSpec(
                name=name,
                face=face,
                rotation_id=CUBE_ROTATIONS.id_of(matrix),
                axis=axis,
                layer=layer,
                amount=amount,
            )
    return tuple(order), MappingProxyType(specs)


MOVE_ORDER, MOVES = _build_moves()


def normalize_move_token(token: str) -> str:
    normalized = token.strip().replace("’", "'").replace("′", "'")
    if normalized not in MOVES:
        raise ValueError(f"Unknown HTM move token: {token!r}")
    return normalized


def parse_word(word: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(word, str):
        if not word.strip():
            return ()
        tokens = word.replace("\n", " ").split()
    else:
        tokens = list(word)
    return tuple(normalize_move_token(token) for token in tokens)


def inverse_move(move: str) -> str:
    move = normalize_move_token(move)
    if move.endswith("2"):
        return move
    if move.endswith("'"):
        return move[0]
    return f"{move}'"


def inverse_word(word: str | Iterable[str]) -> tuple[str, ...]:
    parsed = parse_word(word)
    return tuple(inverse_move(move) for move in reversed(parsed))
