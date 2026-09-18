"""Piece, slot, face-normal, and native-adjacency definitions for PDCC."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from .group import Vector3


class PieceKind(str, Enum):
    CORNER = "corner"
    EDGE = "edge"


U: Vector3 = (0, 1, 0)
D: Vector3 = (0, -1, 0)
R: Vector3 = (1, 0, 0)
L: Vector3 = (-1, 0, 0)
F: Vector3 = (0, 0, 1)
B: Vector3 = (0, 0, -1)

FACE_NORMALS: Mapping[str, Vector3] = MappingProxyType(
    {"U": U, "D": D, "R": R, "L": L, "F": F, "B": B}
)

# The sticker order follows the standard cubie-coordinate ordering.  It is not
# cosmetic: exact ordered-normal matching defines orientation 0 in every slot.
CORNER_FACES: Mapping[str, tuple[Vector3, Vector3, Vector3]] = MappingProxyType(
    {
        "UFR": (U, R, F),
        "UFL": (U, F, L),
        "UBL": (U, L, B),
        "UBR": (U, B, R),
        "DFR": (D, F, R),
        "DFL": (D, L, F),
        "DBL": (D, B, L),
        "DBR": (D, R, B),
    }
)

EDGE_FACES: Mapping[str, tuple[Vector3, Vector3]] = MappingProxyType(
    {
        "UR": (U, R),
        "UF": (U, F),
        "UL": (U, L),
        "UB": (U, B),
        "DR": (D, R),
        "DF": (D, F),
        "DL": (D, L),
        "DB": (D, B),
        "FR": (F, R),
        "FL": (F, L),
        "BL": (B, L),
        "BR": (B, R),
    }
)

CORNER_NAMES: tuple[str, ...] = tuple(CORNER_FACES)
EDGE_NAMES: tuple[str, ...] = tuple(EDGE_FACES)
PIECE_NAMES: tuple[str, ...] = CORNER_NAMES + EDGE_NAMES


def _sum_vectors(vectors: tuple[Vector3, ...]) -> Vector3:
    return tuple(sum(vector[axis] for vector in vectors) for axis in range(3))  # type: ignore[return-value]


CORNER_POSITIONS: Mapping[str, Vector3] = MappingProxyType(
    {name: _sum_vectors(faces) for name, faces in CORNER_FACES.items()}
)
EDGE_POSITIONS: Mapping[str, Vector3] = MappingProxyType(
    {name: _sum_vectors(faces) for name, faces in EDGE_FACES.items()}
)
PIECE_POSITIONS: Mapping[str, Vector3] = MappingProxyType(
    {**CORNER_POSITIONS, **EDGE_POSITIONS}
)

CORNER_SLOT_BY_POSITION: Mapping[Vector3, str] = MappingProxyType(
    {position: name for name, position in CORNER_POSITIONS.items()}
)
EDGE_SLOT_BY_POSITION: Mapping[Vector3, str] = MappingProxyType(
    {position: name for name, position in EDGE_POSITIONS.items()}
)
SLOT_BY_POSITION: Mapping[Vector3, str] = MappingProxyType(
    {**CORNER_SLOT_BY_POSITION, **EDGE_SLOT_BY_POSITION}
)

PIECE_INDEX: Mapping[str, int] = MappingProxyType(
    {name: index for index, name in enumerate(PIECE_NAMES)}
)
CORNER_INDEX: Mapping[str, int] = MappingProxyType(
    {name: index for index, name in enumerate(CORNER_NAMES)}
)
EDGE_INDEX: Mapping[str, int] = MappingProxyType(
    {name: index for index, name in enumerate(EDGE_NAMES)}
)


@dataclass(frozen=True, slots=True)
class PieceSpec:
    name: str
    kind: PieceKind
    solved_position: Vector3
    ordered_stickers: tuple[Vector3, ...]
    orientation_order: int


PIECE_SPECS: Mapping[str, PieceSpec] = MappingProxyType(
    {
        **{
            name: PieceSpec(
                name=name,
                kind=PieceKind.CORNER,
                solved_position=CORNER_POSITIONS[name],
                ordered_stickers=CORNER_FACES[name],
                orientation_order=3,
            )
            for name in CORNER_NAMES
        },
        **{
            name: PieceSpec(
                name=name,
                kind=PieceKind.EDGE,
                solved_position=EDGE_POSITIONS[name],
                ordered_stickers=EDGE_FACES[name],
                orientation_order=2,
            )
            for name in EDGE_NAMES
        },
    }
)

# Native edge-corner incidence graph: 12 edges x 2 endpoint corners = 24 links.
NATIVE_ADJACENCY: tuple[tuple[str, str], ...] = (
    ("UR", "UFR"),
    ("UR", "UBR"),
    ("UF", "UFR"),
    ("UF", "UFL"),
    ("UL", "UFL"),
    ("UL", "UBL"),
    ("UB", "UBL"),
    ("UB", "UBR"),
    ("DR", "DFR"),
    ("DR", "DBR"),
    ("DF", "DFR"),
    ("DF", "DFL"),
    ("DL", "DFL"),
    ("DL", "DBL"),
    ("DB", "DBL"),
    ("DB", "DBR"),
    ("FR", "UFR"),
    ("FR", "DFR"),
    ("FL", "UFL"),
    ("FL", "DFL"),
    ("BL", "UBL"),
    ("BL", "DBL"),
    ("BR", "UBR"),
    ("BR", "DBR"),
)

NATIVE_NEIGHBORS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        name: tuple(
            other
            for left, right in NATIVE_ADJACENCY
            for other in (
                (right,) if left == name else (left,) if right == name else ()
            )
        )
        for name in PIECE_NAMES
    }
)


def piece_kind(name: str) -> PieceKind:
    try:
        return PIECE_SPECS[name].kind
    except KeyError as exc:
        raise ValueError(f"Unknown piece or slot: {name}") from exc


def slots_for_kind(kind: PieceKind) -> tuple[str, ...]:
    return CORNER_NAMES if kind is PieceKind.CORNER else EDGE_NAMES
