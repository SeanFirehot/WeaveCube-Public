from __future__ import annotations

from dataclasses import dataclass

from .orientation import OrientationAnalyzer
from .pieces import CORNER_ORDER, EDGE_ORDER
from .transformations import PIECES, Transformation

_PIECE_INDEX = {piece: index for index, piece in enumerate(PIECES)}
_CORNER_INDEX = {piece: index for index, piece in enumerate(CORNER_ORDER)}
_EDGE_INDEX = {piece: index for index, piece in enumerate(EDGE_ORDER)}


@dataclass(frozen=True, slots=True)
class CubieEffect:
    """Complete cube effect independent of move-sequence history.

    ``destinations`` and orientation deltas are indexed by source position.
    This is sufficient to compose permutation, corner twist, and edge flip
    without reparsing primitive move sequences.
    """

    destinations: tuple[int, ...]
    corner_twists: tuple[int, ...]
    edge_flips: tuple[int, ...]

    @classmethod
    def identity(cls) -> "CubieEffect":
        return cls(
            destinations=tuple(range(len(PIECES))),
            corner_twists=(0,) * len(CORNER_ORDER),
            edge_flips=(0,) * len(EDGE_ORDER),
        )

    @classmethod
    def from_transformation(cls, transformation: Transformation) -> "CubieEffect":
        profile = OrientationAnalyzer.analyze(transformation)
        corners = profile.corner_by_piece()
        edges = profile.edge_by_piece()
        return cls(
            destinations=tuple(_PIECE_INDEX[transformation.mapping[piece]] for piece in PIECES),
            corner_twists=tuple(corners[piece].twist for piece in CORNER_ORDER),
            edge_flips=tuple(edges[piece].orientation for piece in EDGE_ORDER),
        )

    def compose_after(self, before: "CubieEffect") -> "CubieEffect":
        """Return ``self ∘ before`` (apply ``before`` first)."""
        destinations = tuple(self.destinations[mid] for mid in before.destinations)

        corner_twists = []
        for source_index, source in enumerate(CORNER_ORDER):
            mid_piece_index = before.destinations[_PIECE_INDEX[source]]
            mid_position = PIECES[mid_piece_index]
            corner_twists.append(
                (before.corner_twists[source_index] + self.corner_twists[_CORNER_INDEX[mid_position]]) % 3
            )

        edge_flips = []
        for source_index, source in enumerate(EDGE_ORDER):
            mid_piece_index = before.destinations[_PIECE_INDEX[source]]
            mid_position = PIECES[mid_piece_index]
            edge_flips.append(
                (before.edge_flips[source_index] + self.edge_flips[_EDGE_INDEX[mid_position]]) % 2
            )

        return CubieEffect(destinations, tuple(corner_twists), tuple(edge_flips))

    def inverse(self) -> "CubieEffect":
        inverse_destinations = [0] * len(PIECES)
        for source, destination in enumerate(self.destinations):
            inverse_destinations[destination] = source

        corner_twists = [0] * len(CORNER_ORDER)
        for source_index, source in enumerate(CORNER_ORDER):
            destination = PIECES[self.destinations[_PIECE_INDEX[source]]]
            corner_twists[_CORNER_INDEX[destination]] = (-self.corner_twists[source_index]) % 3

        edge_flips = [0] * len(EDGE_ORDER)
        for source_index, source in enumerate(EDGE_ORDER):
            destination = PIECES[self.destinations[_PIECE_INDEX[source]]]
            edge_flips[_EDGE_INDEX[destination]] = (-self.edge_flips[source_index]) % 2

        return CubieEffect(tuple(inverse_destinations), tuple(corner_twists), tuple(edge_flips))

    @property
    def signature(self) -> tuple:
        return self.destinations, self.corner_twists, self.edge_flips


__all__ = ["CubieEffect"]
