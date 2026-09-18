"""UD/FB/LR orientation projections derived from full Q poses."""

from __future__ import annotations

from types import MappingProxyType

from cubelab.pdcc.group import CUBE_ROTATIONS
from cubelab.pdcc.model import FACE_NORMALS, PIECE_INDEX, PIECE_NAMES, PIECE_SPECS, SLOT_BY_POSITION
from cubelab.pdcc.orientation import ORIENTATION_SYSTEM

from .matrices import OMEGA_MATRICES
from .pose24 import Q_TO_OMEGA


AXES: tuple[str, ...] = ("UD", "FB", "LR")
AXIS_FACE_MAPS = MappingProxyType(
    {
        "UD": {"U": "U", "D": "D", "R": "R", "L": "L", "F": "F", "B": "B"},
        "FB": {"U": "B", "B": "D", "D": "F", "F": "U", "R": "R", "L": "L"},
        # This is CubeLab domino_reduction's deterministic minimum-key RL->UD
        # normalizer (reported as LR by ATO).  The residual rotation about the
        # destination UD axis is part of the legacy gauge and must be kept.
        "LR": {"R": "U", "L": "D", "U": "B", "D": "F", "F": "L", "B": "R"},
    }
)


def _rotation_for_face_map(face_map: dict[str, str]) -> int:
    for rotation_id in range(24):
        if all(
            CUBE_ROTATIONS.apply(rotation_id, FACE_NORMALS[source])
            == FACE_NORMALS[target]
            for source, target in face_map.items()
        ):
            return rotation_id
    raise RuntimeError(f"Face map is not a proper cube rotation: {face_map}")


AXIS_ROTATION_IDS = MappingProxyType(
    {axis: _rotation_for_face_map(dict(AXIS_FACE_MAPS[axis])) for axis in AXES}
)


def mapped_piece(piece: str, rotation_id: int) -> str:
    position = CUBE_ROTATIONS.apply(rotation_id, PIECE_SPECS[piece].solved_position)
    return SLOT_BY_POSITION[position]


def conjugate_q(rotation_id: int, q: int) -> int:
    group = CUBE_ROTATIONS
    return group.compose(rotation_id, group.compose(q, group.inverse(rotation_id)))


def legacy_orientation(piece: str, q: int, axis: str = "UD") -> int:
    if axis not in AXIS_ROTATION_IDS:
        raise ValueError(f"Unknown orientation axis: {axis}")
    rotation_id = AXIS_ROTATION_IDS[axis]
    rotated_piece = mapped_piece(piece, rotation_id)
    rotated_q = conjugate_q(rotation_id, q)
    return ORIENTATION_SYSTEM.orientation(rotated_piece, rotated_q)


def project_q_vector(q_vector: tuple[int, ...], axis: str = "UD") -> tuple[int, ...]:
    if len(q_vector) != len(PIECE_NAMES):
        raise ValueError(f"Expected {len(PIECE_NAMES)} Q poses")
    return tuple(
        legacy_orientation(piece, q_vector[PIECE_INDEX[piece]], axis)
        for piece in PIECE_NAMES
    )


def omega_conjugacy_class(omega: int) -> str:
    matrix = OMEGA_MATRICES[omega]
    fixed = sum(matrix[index][index] for index in range(3))
    if fixed == 3:
        return "identity"
    if fixed == 1:
        return "transposition"
    if fixed == 0:
        return "3-cycle"
    raise AssertionError(f"Unexpected S3 permutation matrix trace: {fixed}")


def omega_of_q(q: int) -> int:
    return Q_TO_OMEGA[q]


__all__ = [
    "AXES",
    "AXIS_FACE_MAPS",
    "AXIS_ROTATION_IDS",
    "conjugate_q",
    "legacy_orientation",
    "mapped_piece",
    "omega_conjugacy_class",
    "omega_of_q",
    "project_q_vector",
]
