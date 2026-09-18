"""Precomputed exact local transition table: 20 pieces x 24 poses x 18 moves."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .group import CUBE_ROTATIONS
from .model import (
    CORNER_SLOT_BY_POSITION,
    EDGE_SLOT_BY_POSITION,
    PIECE_NAMES,
    PIECE_SPECS,
    PieceKind,
)
from .moves import MOVE_ORDER, MOVES
from .orientation import ORIENTATION_SYSTEM


@dataclass(frozen=True, slots=True)
class LocalTransition:
    piece: str
    pose_before: int
    move: str
    active: bool
    pose_after: int
    slot_before: str
    slot_after: str
    orientation_before: int
    orientation_after: int


def _slot(piece: str, pose_id: int) -> str:
    spec = PIECE_SPECS[piece]
    position = CUBE_ROTATIONS.apply(pose_id, spec.solved_position)
    if spec.kind is PieceKind.CORNER:
        return CORNER_SLOT_BY_POSITION[position]
    return EDGE_SLOT_BY_POSITION[position]


def _build_local_transition_table() -> Mapping[tuple[str, int, str], LocalTransition]:
    table: dict[tuple[str, int, str], LocalTransition] = {}
    for piece in PIECE_NAMES:
        spec = PIECE_SPECS[piece]
        for pose_before in range(24):
            slot_before = _slot(piece, pose_before)
            position = CUBE_ROTATIONS.apply(pose_before, spec.solved_position)
            orientation_before = ORIENTATION_SYSTEM.orientation(piece, pose_before)
            for move in MOVE_ORDER:
                move_spec = MOVES[move]
                active = move_spec.is_active_position(position)
                pose_after = (
                    CUBE_ROTATIONS.compose(move_spec.rotation_id, pose_before)
                    if active
                    else pose_before
                )
                table[(piece, pose_before, move)] = LocalTransition(
                    piece=piece,
                    pose_before=pose_before,
                    move=move,
                    active=active,
                    pose_after=pose_after,
                    slot_before=slot_before,
                    slot_after=_slot(piece, pose_after),
                    orientation_before=orientation_before,
                    orientation_after=ORIENTATION_SYSTEM.orientation(piece, pose_after),
                )
    if len(table) != 20 * 24 * 18:
        raise RuntimeError(f"Expected 8640 local transitions, got {len(table)}")
    return MappingProxyType(table)


LOCAL_TRANSITION_TABLE = _build_local_transition_table()


def local_transition(piece: str, pose_id: int, move: str) -> LocalTransition:
    try:
        return LOCAL_TRANSITION_TABLE[(piece, pose_id, move)]
    except KeyError as exc:
        raise ValueError(f"Unknown local transition key: {piece}/{pose_id}/{move}") from exc
