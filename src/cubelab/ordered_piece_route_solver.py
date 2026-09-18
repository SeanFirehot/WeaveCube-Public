from __future__ import annotations

from functools import lru_cache

from .cubie_effect import CubieEffect
from .piece_route_graph import PieceState, RouteWitness, SinglePieceRouteGraph
from .pieces import CORNER_ORDER, EDGE_ORDER
from .transformations import PIECES

_PIECE_INDEX = {piece: index for index, piece in enumerate(PIECES)}
_CORNER_INDEX = {piece: index for index, piece in enumerate(CORNER_ORDER)}
_EDGE_INDEX = {piece: index for index, piece in enumerate(EDGE_ORDER)}


class OrderedPieceRouteSolver:
    """Exact ordered route portfolios in single-cubie projection graphs.

    This class is the shared low-level route provider used by the route,
    joint-pair, component, hypergraph, and deferred-amount solvers.  A cubie's
    projected state is read directly from :class:`CubieEffect`; route witnesses
    are then enumerated in deterministic ``(length, sequence)`` order.
    """

    def __init__(self) -> None:
        self.corner_graph = SinglePieceRouteGraph("corner")
        self.edge_graph = SinglePieceRouteGraph("edge")

    def piece_state(self, effect: CubieEffect, piece: str) -> PieceState:
        try:
            source_index = _PIECE_INDEX[piece]
        except KeyError as exc:
            raise ValueError(f"Unknown cubie: {piece}") from exc

        position = PIECES[effect.destinations[source_index]]
        if piece in _CORNER_INDEX:
            orientation = effect.corner_twists[_CORNER_INDEX[piece]]
        elif piece in _EDGE_INDEX:
            orientation = effect.edge_flips[_EDGE_INDEX[piece]]
        else:  # Defensive guard if PIECES and piece-order constants diverge.
            raise ValueError(f"Cubie is neither corner nor edge: {piece}")
        return PieceState(position, orientation)

    @lru_cache(maxsize=None)
    def _routes_cached(
        self,
        piece_type: str,
        source: PieceState,
        target: PieceState,
        slack: int,
        max_witnesses: int | None,
    ) -> tuple[RouteWitness, ...]:
        graph = self.corner_graph if piece_type == "corner" else self.edge_graph if piece_type == "edge" else None
        if graph is None:
            raise ValueError("piece_type must be 'corner' or 'edge'")
        return graph.routes_with_slack(
            source,
            target,
            slack=slack,
            max_witnesses=max_witnesses,
        )

    def _routes(
        self,
        piece_type: str,
        source: PieceState,
        target: PieceState,
        *,
        slack: int = 0,
        max_witnesses: int | None = None,
    ) -> tuple[RouteWitness, ...]:
        if slack < 0:
            raise ValueError("slack must be non-negative")
        if max_witnesses is not None and max_witnesses < 1:
            return ()
        return self._routes_cached(piece_type, source, target, slack, max_witnesses)


__all__ = ["OrderedPieceRouteSolver"]
