"""Orientation analysis for CubeLab transformations.

The core ``Transformation`` object remains unchanged.  Orientation facts for
primitive face turns live in ``PRIMITIVE_ORIENTATION_RULES`` and are composed
by ``OrientationAnalyzer`` from the transformation's move sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

from .pieces import CORNER_ORDER, EDGE_ORDER, PieceName
from .transformations import PRIMITIVES, Transformation


@dataclass(frozen=True)
class PrimitiveOrientationRule:
    """Orientation deltas caused by one clockwise primitive face turn.

    Deltas are indexed by the piece's position *before* the turn.  Corner
    values are interpreted modulo 3 and edge values modulo 2.
    """

    move: str
    corner_deltas: Mapping[PieceName, int]
    edge_deltas: Mapping[PieceName, int]

    def delta_for(self, position: PieceName) -> int:
        if position in CORNER_ORDER:
            return self.corner_deltas.get(position, 0)
        if position in EDGE_ORDER:
            return self.edge_deltas.get(position, 0)
        raise ValueError(f"Unknown piece position: {position}")


# Source facts, expressed using CubeLab's ordered-frame orientation convention.
PRIMITIVE_ORIENTATION_RULES: Dict[str, PrimitiveOrientationRule] = {
    "U": PrimitiveOrientationRule("U", {}, {}),
    "D": PrimitiveOrientationRule("D", {}, {}),
    "R": PrimitiveOrientationRule(
        "R",
        {"UFR": 1, "UBR": 2, "DRB": 1, "DFR": 2},
        {},
    ),
    "L": PrimitiveOrientationRule(
        "L",
        {"UFL": 2, "DLF": 1, "DBL": 2, "UBL": 1},
        {},
    ),
    "F": PrimitiveOrientationRule(
        "F",
        {"UFR": 2, "DFR": 1, "DLF": 2, "UFL": 1},
        {"UF": 1, "FR": 1, "DF": 1, "FL": 1},
    ),
    "B": PrimitiveOrientationRule(
        "B",
        {"UBR": 1, "UBL": 2, "DBL": 1, "DRB": 2},
        {"UB": 1, "BL": 1, "DB": 1, "BR": 1},
    ),
}


@dataclass(frozen=True)
class CornerOrientationChange:
    piece: PieceName
    position: PieceName
    twist: int

    @property
    def is_oriented(self) -> bool:
        return self.twist == 0


@dataclass(frozen=True)
class EdgeOrientationChange:
    piece: PieceName
    position: PieceName
    flipped: bool

    @property
    def orientation(self) -> int:
        return int(self.flipped)

    @property
    def is_oriented(self) -> bool:
        return not self.flipped


@dataclass(frozen=True)
class OrientationProfile:
    corner_changes: Tuple[CornerOrientationChange, ...]
    edge_changes: Tuple[EdgeOrientationChange, ...]

    @property
    def twisted_corners(self) -> Tuple[CornerOrientationChange, ...]:
        return tuple(change for change in self.corner_changes if change.twist)

    @property
    def flipped_edges(self) -> Tuple[EdgeOrientationChange, ...]:
        return tuple(change for change in self.edge_changes if change.flipped)

    @property
    def orientation_support(self) -> frozenset[PieceName]:
        return frozenset(
            [change.piece for change in self.twisted_corners]
            + [change.piece for change in self.flipped_edges]
        )

    @property
    def is_orientation_preserving(self) -> bool:
        return not self.twisted_corners and not self.flipped_edges

    @property
    def corner_checksum(self) -> int:
        return sum(change.twist for change in self.corner_changes) % 3

    @property
    def edge_checksum(self) -> int:
        return sum(change.orientation for change in self.edge_changes) % 2

    @property
    def invariants_hold(self) -> bool:
        return self.corner_checksum == 0 and self.edge_checksum == 0

    def corner_by_piece(self) -> Dict[PieceName, CornerOrientationChange]:
        return {change.piece: change for change in self.corner_changes}

    def edge_by_piece(self) -> Dict[PieceName, EdgeOrientationChange]:
        return {change.piece: change for change in self.edge_changes}


class OrientationAnalyzer:
    @staticmethod
    def analyze(transformation: Transformation) -> OrientationProfile:
        positions: Dict[PieceName, PieceName] = {
            piece: piece for piece in (*CORNER_ORDER, *EDGE_ORDER)
        }
        corner_orientation = {piece: 0 for piece in CORNER_ORDER}
        edge_orientation = {piece: 0 for piece in EDGE_ORDER}

        for token in transformation.sequence:
            base, turns = _parse_move_token(token)
            rule = PRIMITIVE_ORIENTATION_RULES[base]
            primitive = PRIMITIVES[base]

            for _ in range(turns):
                previous_positions = positions.copy()

                for piece in CORNER_ORDER:
                    source = previous_positions[piece]
                    corner_orientation[piece] = (
                        corner_orientation[piece] + rule.delta_for(source)
                    ) % 3
                    positions[piece] = primitive.mapping[source]

                for piece in EDGE_ORDER:
                    source = previous_positions[piece]
                    edge_orientation[piece] = (
                        edge_orientation[piece] + rule.delta_for(source)
                    ) % 2
                    positions[piece] = primitive.mapping[source]

        # The independently composed orientation path must land at the same
        # positions as the source Transformation mapping.
        for piece, position in positions.items():
            expected = transformation.mapping[piece]
            if position != expected:
                raise ValueError(
                    "Orientation composition disagrees with Transformation "
                    f"mapping for {piece}: {position} != {expected}"
                )

        return OrientationProfile(
            corner_changes=tuple(
                CornerOrientationChange(
                    piece=piece,
                    position=positions[piece],
                    twist=corner_orientation[piece],
                )
                for piece in CORNER_ORDER
            ),
            edge_changes=tuple(
                EdgeOrientationChange(
                    piece=piece,
                    position=positions[piece],
                    flipped=bool(edge_orientation[piece]),
                )
                for piece in EDGE_ORDER
            ),
        )



def _parse_move_token(token: str) -> Tuple[str, int]:
    if not token:
        raise ValueError("Move token cannot be empty")

    base = token[0]
    suffix = token[1:]

    if base not in PRIMITIVE_ORIENTATION_RULES:
        raise ValueError(f"Unsupported orientation move: {token}")

    if suffix == "":
        return base, 1
    if suffix == "2":
        return base, 2
    if suffix == "'":
        return base, 3

    raise ValueError(f"Unsupported move token: {token}")


__all__ = [
    "PrimitiveOrientationRule",
    "PRIMITIVE_ORIENTATION_RULES",
    "CornerOrientationChange",
    "EdgeOrientationChange",
    "OrientationProfile",
    "OrientationAnalyzer",
]

# ---------------------------------------------------------------------------
# Orientation requirements and matching
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CornerTwistRequirement:
    piece: PieceName
    twist: int

    def __post_init__(self) -> None:
        if self.piece not in CORNER_ORDER:
            raise ValueError(f"Corner requirement needs a corner piece: {self.piece}")
        if self.twist not in (0, 1, 2):
            raise ValueError("Corner twist must be 0, 1, or 2")


@dataclass(frozen=True)
class EdgeFlipRequirement:
    piece: PieceName
    flipped: bool = True

    def __post_init__(self) -> None:
        if self.piece not in EDGE_ORDER:
            raise ValueError(f"Edge requirement needs an edge piece: {self.piece}")


@dataclass(frozen=True)
class OrientationRequirement:
    corner_twists: Tuple[CornerTwistRequirement, ...] = ()
    edge_flips: Tuple[EdgeFlipRequirement, ...] = ()

    def __post_init__(self) -> None:
        corner_pieces = [item.piece for item in self.corner_twists]
        edge_pieces = [item.piece for item in self.edge_flips]
        if len(corner_pieces) != len(set(corner_pieces)):
            raise ValueError("Duplicate corner orientation requirement")
        if len(edge_pieces) != len(set(edge_pieces)):
            raise ValueError("Duplicate edge orientation requirement")

    @property
    def is_empty(self) -> bool:
        return not self.corner_twists and not self.edge_flips


@dataclass(frozen=True)
class OrientationMatchResult:
    requirement: OrientationRequirement
    profile: OrientationProfile
    matched_corner_twists: Tuple[CornerTwistRequirement, ...]
    missing_corner_twists: Tuple[CornerTwistRequirement, ...]
    matched_edge_flips: Tuple[EdgeFlipRequirement, ...]
    missing_edge_flips: Tuple[EdgeFlipRequirement, ...]
    incidental_corner_twists: Tuple[CornerOrientationChange, ...]
    incidental_edge_flips: Tuple[EdgeOrientationChange, ...]

    @property
    def satisfied(self) -> bool:
        return not self.missing_corner_twists and not self.missing_edge_flips

    @property
    def incidental_count(self) -> int:
        return len(self.incidental_corner_twists) + len(self.incidental_edge_flips)


class OrientationRequirementMatcher:
    @staticmethod
    def match(
        requirement: OrientationRequirement,
        value: Transformation | OrientationProfile,
    ) -> OrientationMatchResult:
        profile = (
            value if isinstance(value, OrientationProfile)
            else OrientationAnalyzer.analyze(value)
        )
        corner_by_piece = profile.corner_by_piece()
        edge_by_piece = profile.edge_by_piece()

        matched_corners = []
        missing_corners = []
        for item in requirement.corner_twists:
            actual = corner_by_piece[item.piece]
            (matched_corners if actual.twist == item.twist else missing_corners).append(item)

        matched_edges = []
        missing_edges = []
        for item in requirement.edge_flips:
            actual = edge_by_piece[item.piece]
            (matched_edges if actual.flipped == item.flipped else missing_edges).append(item)

        required_corner_pieces = {item.piece for item in requirement.corner_twists}
        required_edge_pieces = {item.piece for item in requirement.edge_flips}

        incidental_corners = tuple(
            change for change in profile.twisted_corners
            if change.piece not in required_corner_pieces
        )
        incidental_edges = tuple(
            change for change in profile.flipped_edges
            if change.piece not in required_edge_pieces
        )

        return OrientationMatchResult(
            requirement=requirement,
            profile=profile,
            matched_corner_twists=tuple(matched_corners),
            missing_corner_twists=tuple(missing_corners),
            matched_edge_flips=tuple(matched_edges),
            missing_edge_flips=tuple(missing_edges),
            incidental_corner_twists=incidental_corners,
            incidental_edge_flips=incidental_edges,
        )


__all__.extend([
    "CornerTwistRequirement",
    "EdgeFlipRequirement",
    "OrientationRequirement",
    "OrientationMatchResult",
    "OrientationRequirementMatcher",
])
