from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


Piece = str
Mapping = Dict[Piece, Piece]


PIECES: Tuple[Piece, ...] = (
    "UFR", "UBR", "UBL", "UFL",
    "DFR", "DRB", "DBL", "DLF",
    "UF", "UR", "UB", "UL",
    "FR", "BR", "BL", "FL",
    "DF", "DR", "DB", "DL",
)


def identity_mapping() -> Mapping:
    return {piece: piece for piece in PIECES}


def mapping_from_cycles(cycles: Iterable[Tuple[Piece, ...]]) -> Mapping:
    mapping = identity_mapping()

    for cycle in cycles:
        if len(cycle) < 2:
            continue

        for i, piece in enumerate(cycle):
            mapping[piece] = cycle[(i + 1) % len(cycle)]

    return mapping


@dataclass(frozen=True)
class Transformation:
    canonical_name: str
    sequence: Tuple[str, ...]
    mapping: Mapping
    aliases: Tuple[str, ...] = ()

    def compose_after(self, before: "Transformation") -> "Transformation":
        return compose(self, before)


def compose(after: Transformation, before: Transformation) -> Transformation:
    """
    compose(after, before) means:

        apply `before` first,
        then apply `after`.

    In mathematical notation:

        after ∘ before
    """

    new_mapping = {
        piece: after.mapping[before.mapping[piece]]
        for piece in PIECES
    }

    return Transformation(
        canonical_name=f"{' '.join(before.sequence)} {' '.join(after.sequence)}".strip(),
        sequence=before.sequence + after.sequence,
        mapping=new_mapping,
    )


PRIMITIVES: Dict[str, Transformation] = {
    "R": Transformation(
        canonical_name="R",
        aliases=("Right turn",),
        sequence=("R",),
        mapping=mapping_from_cycles(
            [
                ("UFR", "UBR", "DRB", "DFR"),
                ("UR", "BR", "DR", "FR"),
            ]
        ),
    ),
    "U": Transformation(
        canonical_name="U",
        aliases=("Upper turn",),
        sequence=("U",),
        mapping=mapping_from_cycles(
            [
                ("UFR", "UFL", "UBL", "UBR"),
                ("UF", "UL", "UB", "UR"),
            ]
        ),

    ),
    "D": Transformation(
        canonical_name="D",
        aliases=("Down turn",),
        sequence=("D",),
        mapping=mapping_from_cycles(
            [
                ("DFR", "DRB", "DBL", "DLF"),
                ("DF", "DR", "DB", "DL"),
            ]
        ),
    ),
    "L": Transformation(
        canonical_name="L",
        aliases=("Left turn",),
        sequence=("L",),
        mapping=mapping_from_cycles(
            [
                ("UFL", "DLF", "DBL", "UBL"),
                ("UL", "FL", "DL", "BL"),
            ]
        ),
    ),
    "F": Transformation(
        canonical_name="F",
        aliases=("Front turn",),
        sequence=("F",),
        mapping=mapping_from_cycles(
            [
                ("UFR", "DFR", "DLF", "UFL"),
                ("UF", "FR", "DF", "FL"),
            ]
        ),
    ),
    "B": Transformation(
        canonical_name="B",
        aliases=("Back turn",),
        sequence=("B",),
        mapping=mapping_from_cycles(
            [
                ("UBR", "UBL", "DBL", "DRB"),
                ("UB", "BL", "DB", "BR"),
            ]
        ),
    ),
}


def from_sequence(sequence: Iterable[str]) -> Transformation:
    result = Transformation(
        canonical_name="identity",
        sequence=(),
        mapping=identity_mapping(),
        aliases=("I",),
    )

    for move in sequence:
        base = move[0]
        suffix = move[1:] if len(move) > 1 else ""
        result = compose(make_move(base, suffix), result)

    return result


def moved_pieces(mapping: Mapping) -> Tuple[Piece, ...]:
    return tuple(piece for piece in PIECES if mapping[piece] != piece)


def decompose_cycles(mapping: Mapping) -> Tuple[Tuple[Piece, ...], ...]:
    visited = set()
    cycles = []

    for start in PIECES:
        if start in visited:
            continue

        current = start
        cycle = []

        while current not in visited:
            visited.add(current)
            cycle.append(current)
            current = mapping[current]

        if len(cycle) > 1:
            cycles.append(tuple(cycle))

    return tuple(cycles)


def describe(transformation: Transformation) -> str:
    cycles = decompose_cycles(transformation.mapping)

    lines = [
        f"Transformation: {transformation.canonical_name}",
        f"Sequence: {' '.join(transformation.sequence) or 'identity'}",
        "",
        "Cycles:",
    ]

    if not cycles:
        lines.append("  identity")
    else:
        for cycle in cycles:
            lines.append("  (" + " -> ".join(cycle) + ")")

    return "\n".join(lines)

def inverse_mapping(mapping: Mapping) -> Mapping:
    return {target: source for source, target in mapping.items()}


def power_mapping(mapping: Mapping, power: int) -> Mapping:
    result = identity_mapping()

    for _ in range(power % 4):
        result = {
            piece: mapping[result[piece]]
            for piece in PIECES
        }

    return result


def make_move(base: str, suffix: str = "") -> Transformation:
    primitive = PRIMITIVES[base]

    if suffix == "":
        return primitive

    if suffix == "'":
        return Transformation(
            canonical_name=f"{base}'",
            sequence=(f"{base}'",),
            mapping=inverse_mapping(primitive.mapping),
            aliases=(f"{primitive.canonical_name} inverse",),
        )

    if suffix == "2":
        return Transformation(
            canonical_name=f"{base}2",
            sequence=(f"{base}2",),
            mapping=power_mapping(primitive.mapping, 2),
            aliases=(f"{primitive.canonical_name} double",),
        )

    raise ValueError(f"Unsupported move suffix: {suffix}")