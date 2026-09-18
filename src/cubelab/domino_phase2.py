from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from .cubie_effect import CubieEffect
from .domino_reduction import (
    DRAxis,
    DR_SUBGROUP_MOVES_UD,
    _axis_to_ud_rotation,
    _move_is_canonical,
    diagnose_transformation,
    find_dr_portfolio_ida,
    normalize_transformation_to_ud,
)
from .pieces import CORNER_ORDER, EDGE_ORDER
from .transformations import PIECES, Transformation, compose, from_sequence

_PIECE_INDEX = {piece: i for i, piece in enumerate(PIECES)}
_CORNER_INDEX = {piece: i for i, piece in enumerate(CORNER_ORDER)}
_EDGE_INDEX = {piece: i for i, piece in enumerate(EDGE_ORDER)}

UD_EDGE_POSITIONS = tuple(edge for edge in EDGE_ORDER if edge not in {"FR", "BR", "BL", "FL"})
SLICE_EDGE_POSITIONS = ("FR", "BR", "BL", "FL")
PHASE2_MOVES_UD: tuple[str, ...] = DR_SUBGROUP_MOVES_UD


def _rank_permutation(values: tuple[int, ...]) -> int:
    rank = 0
    n = len(values)
    for i in range(n):
        smaller = sum(values[j] < values[i] for j in range(i + 1, n))
        rank = rank * (n - i) + smaller
    return rank


def _unrank_permutation(rank: int, n: int) -> tuple[int, ...]:
    digits = [0] * n
    for i in range(n - 1, -1, -1):
        base = n - i
        digits[i] = rank % base
        rank //= base
    available = list(range(n))
    result = []
    for digit in digits:
        result.append(available.pop(digit))
    return tuple(result)


def _occupants_by_position(transformation: Transformation, order: tuple[str, ...]) -> tuple[str, ...]:
    effect = CubieEffect.from_transformation(transformation)
    occupants: list[str | None] = [None] * len(order)
    position_index = {piece: i for i, piece in enumerate(order)}
    for source in order:
        destination = PIECES[effect.destinations[_PIECE_INDEX[source]]]
        occupants[position_index[destination]] = source
    if any(item is None for item in occupants):
        raise AssertionError("incomplete cubie permutation")
    return tuple(item for item in occupants if item is not None)


@dataclass(frozen=True, slots=True)
class Phase2Coordinates:
    corner_permutation_index: int
    ud_edge_permutation_index: int
    slice_edge_permutation_index: int

    @property
    def is_solved(self) -> bool:
        return (
            self.corner_permutation_index == 0
            and self.ud_edge_permutation_index == 0
            and self.slice_edge_permutation_index == 0
        )


def phase2_coordinates_from_transformation(
    transformation: Transformation,
    *,
    axis: DRAxis | str = DRAxis.UD,
) -> Phase2Coordinates:
    axis = DRAxis(axis)
    diagnostic = diagnose_transformation(transformation, axis=axis)
    if not diagnostic.is_reduced:
        raise ValueError(f"state is not in {axis.value}-axis DR")
    normalized = normalize_transformation_to_ud(transformation, axis)

    corner_occupants = _occupants_by_position(normalized, CORNER_ORDER)
    corner_perm = tuple(_CORNER_INDEX[piece] for piece in corner_occupants)

    edge_occupants = _occupants_by_position(normalized, EDGE_ORDER)
    edge_at = dict(zip(EDGE_ORDER, edge_occupants))
    ud_index = {piece: i for i, piece in enumerate(UD_EDGE_POSITIONS)}
    slice_index = {piece: i for i, piece in enumerate(SLICE_EDGE_POSITIONS)}
    ud_perm = tuple(ud_index[edge_at[position]] for position in UD_EDGE_POSITIONS)
    slice_perm = tuple(slice_index[edge_at[position]] for position in SLICE_EDGE_POSITIONS)

    return Phase2Coordinates(
        _rank_permutation(corner_perm),
        _rank_permutation(ud_perm),
        _rank_permutation(slice_perm),
    )


def _position_destinations(move: str, order: tuple[str, ...]) -> tuple[int, ...]:
    effect = CubieEffect.from_transformation(from_sequence((move,)))
    index = {piece: i for i, piece in enumerate(order)}
    result = []
    for position in order:
        destination = PIECES[effect.destinations[_PIECE_INDEX[position]]]
        result.append(index[destination])
    return tuple(result)


def _transition_permutation(values: tuple[int, ...], destinations: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * len(values)
    for old_position, new_position in enumerate(destinations):
        result[new_position] = values[old_position]
    return tuple(result)


def _build_permutation_move_table(n: int, destinations_by_move: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    from math import factorial
    rows = []
    for index in range(factorial(n)):
        values = _unrank_permutation(index, n)
        rows.append(tuple(
            _rank_permutation(_transition_permutation(values, destinations))
            for destinations in destinations_by_move
        ))
    return tuple(rows)


def _build_pair_distance(
    first: tuple[tuple[int, ...], ...],
    second: tuple[tuple[int, ...], ...],
) -> bytes:
    width = len(second)
    unseen = 255
    distance = bytearray([unseen]) * (len(first) * width)
    distance[0] = 0
    queue: deque[int] = deque([0])
    while queue:
        state = queue.popleft()
        a, b = divmod(state, width)
        next_distance = distance[state] + 1
        row_a = first[a]
        row_b = second[b]
        for move_index in range(len(row_a)):
            neighbour = row_a[move_index] * width + row_b[move_index]
            if distance[neighbour] == unseen:
                distance[neighbour] = next_distance
                queue.append(neighbour)
    if unseen in distance:
        raise AssertionError("phase-2 pair graph was not fully reached")
    return bytes(distance)


@dataclass(frozen=True, slots=True)
class Phase2PruningTables:
    moves: tuple[str, ...]
    corner_move: tuple[tuple[int, ...], ...]
    ud_edge_move: tuple[tuple[int, ...], ...]
    slice_edge_move: tuple[tuple[int, ...], ...]
    corner_slice_distance: bytes
    ud_edge_slice_distance: bytes

    def heuristic(self, coordinates: Phase2Coordinates) -> int:
        sl = coordinates.slice_edge_permutation_index
        return max(
            self.corner_slice_distance[coordinates.corner_permutation_index * 24 + sl],
            self.ud_edge_slice_distance[coordinates.ud_edge_permutation_index * 24 + sl],
        )


@lru_cache(maxsize=1)
def build_phase2_pruning_tables() -> Phase2PruningTables:
    moves = PHASE2_MOVES_UD
    corner_destinations = tuple(_position_destinations(move, CORNER_ORDER) for move in moves)
    full_edge_destinations = tuple(_position_destinations(move, EDGE_ORDER) for move in moves)
    edge_order_index = {piece: i for i, piece in enumerate(EDGE_ORDER)}

    def restricted_destinations(group: tuple[str, ...], full: tuple[int, ...]) -> tuple[int, ...]:
        group_index = {piece: i for i, piece in enumerate(group)}
        result = []
        for position in group:
            destination_piece = EDGE_ORDER[full[edge_order_index[position]]]
            result.append(group_index[destination_piece])
        return tuple(result)

    ud_destinations = tuple(restricted_destinations(UD_EDGE_POSITIONS, full) for full in full_edge_destinations)
    slice_destinations = tuple(restricted_destinations(SLICE_EDGE_POSITIONS, full) for full in full_edge_destinations)

    corner_move = _build_permutation_move_table(8, corner_destinations)
    ud_edge_move = _build_permutation_move_table(8, ud_destinations)
    slice_edge_move = _build_permutation_move_table(4, slice_destinations)
    return Phase2PruningTables(
        moves=moves,
        corner_move=corner_move,
        ud_edge_move=ud_edge_move,
        slice_edge_move=slice_edge_move,
        corner_slice_distance=_build_pair_distance(corner_move, slice_edge_move),
        ud_edge_slice_distance=_build_pair_distance(ud_edge_move, slice_edge_move),
    )


def phase2_lower_bound(
    transformation: Transformation,
    *,
    axis: DRAxis | str = DRAxis.UD,
    tables: Phase2PruningTables | None = None,
) -> int:
    """Admissible lower bound for solving a DR state inside Phase 2."""
    tables = tables or build_phase2_pruning_tables()
    return tables.heuristic(phase2_coordinates_from_transformation(transformation, axis=axis))



def rank_dr_terminals_by_phase2_lower_bound(
    terminals: Iterable[object],
    *,
    tables: Phase2PruningTables | None = None,
) -> tuple[object, ...]:
    """Rank DR terminals by DR depth plus the admissible Phase-2 lower bound."""
    tables = tables or build_phase2_pruning_tables()
    return tuple(sorted(
        terminals,
        key=lambda terminal: (
            len(terminal.sequence)
            + phase2_lower_bound(terminal.transformation, axis=terminal.axis, tables=tables),
            phase2_lower_bound(terminal.transformation, axis=terminal.axis, tables=tables),
            len(terminal.sequence),
            terminal.axis.value,
            terminal.sequence,
        ),
    ))


def select_dr_terminals_for_exact_evaluation(
    terminals: Iterable[object],
    *,
    lower_bound_top_k: int | None,
    include_shortest_per_axis: bool = False,
    tables: Phase2PruningTables | None = None,
) -> tuple[object, ...]:
    """Select a deterministic exact-evaluation subset from one fixed portfolio."""
    ranked = rank_dr_terminals_by_phase2_lower_bound(terminals, tables=tables)
    if lower_bound_top_k is None:
        return ranked
    if lower_bound_top_k <= 0:
        return ()
    selected = list(ranked[:lower_bound_top_k])
    if include_shortest_per_axis:
        shortest_by_axis: dict[DRAxis, object] = {}
        for terminal in ranked:
            current = shortest_by_axis.get(terminal.axis)
            key = (len(terminal.sequence), terminal.sequence)
            if current is None or key < (len(current.sequence), current.sequence):
                shortest_by_axis[terminal.axis] = terminal
        seen = {(terminal.axis, terminal.sequence) for terminal in selected}
        for axis in (DRAxis.UD, DRAxis.FB, DRAxis.RL):
            terminal = shortest_by_axis.get(axis)
            if terminal is not None and (terminal.axis, terminal.sequence) not in seen:
                selected.append(terminal)
                seen.add((terminal.axis, terminal.sequence))
    return tuple(selected)

@dataclass(frozen=True, slots=True)
class Phase2Solution:
    axis: DRAxis
    sequence: tuple[str, ...]
    depth: int
    nodes: int


def _axis_phase2_moves(axis: DRAxis, tables: Phase2PruningTables) -> tuple[tuple[str, int], ...]:
    if axis is DRAxis.UD:
        return tuple((move, i) for i, move in enumerate(tables.moves))
    rotation = _axis_to_ud_rotation(axis)
    all_moves = tuple(
        f"{face}{suffix}" for face in "UDRLFB" for suffix in ("", "'", "2")
    )
    physical = {move: from_sequence((move,)) for move in all_moves}
    normalized = {move: from_sequence((move,)) for move in tables.moves}
    result = []
    for normalized_name, normalized_transformation in normalized.items():
        matches = [
            move for move, transformation in physical.items()
            if rotation.apply(transformation) == normalized_transformation
        ]
        if len(matches) != 1:
            raise AssertionError(f"could not map phase-2 move {normalized_name} to {axis.value}")
        result.append((matches[0], tables.moves.index(normalized_name)))
    return tuple(result)


def solve_phase2(
    transformation: Transformation,
    *,
    axis: DRAxis | str = DRAxis.UD,
    max_depth: int = 18,
    tables: Phase2PruningTables | None = None,
) -> Phase2Solution | None:
    axis = DRAxis(axis)
    tables = tables or build_phase2_pruning_tables()
    start = phase2_coordinates_from_transformation(transformation, axis=axis)
    moves = _axis_phase2_moves(axis, tables)
    start_bound = tables.heuristic(start)
    total_nodes = 0

    for bound in range(start_bound, max_depth + 1):
        answer: tuple[str, ...] | None = None

        def visit(cp: int, ep8: int, ep4: int, path: tuple[str, ...]) -> bool:
            nonlocal total_nodes, answer
            total_nodes += 1
            coordinates = Phase2Coordinates(cp, ep8, ep4)
            if len(path) + tables.heuristic(coordinates) > bound:
                return False
            if coordinates.is_solved:
                answer = path
                return True
            if len(path) == bound:
                return False
            for move, move_index in moves:
                if not _move_is_canonical(path, move):
                    continue
                if visit(
                    tables.corner_move[cp][move_index],
                    tables.ud_edge_move[ep8][move_index],
                    tables.slice_edge_move[ep4][move_index],
                    path + (move,),
                ):
                    return True
            return False

        if visit(
            start.corner_permutation_index,
            start.ud_edge_permutation_index,
            start.slice_edge_permutation_index,
            (),
        ):
            assert answer is not None
            current = transformation
            move_transformations = {move: from_sequence((move,)) for move, _ in moves}
            for move in answer:
                current = compose(move_transformations[move], current)
            if current.mapping != from_sequence(()).mapping:
                raise AssertionError("phase-2 coordinate solution failed physical replay")
            return Phase2Solution(axis, answer, len(answer), total_nodes)
    return None


@dataclass(frozen=True, slots=True)
class DRFullSolution:
    scramble: tuple[str, ...]
    axis: DRAxis
    dr_sequence: tuple[str, ...]
    phase2_sequence: tuple[str, ...]
    full_sequence: tuple[str, ...]
    dr_depth: int
    phase2_depth: int
    total_depth: int


def solve_with_dr(
    scramble: Iterable[str],
    *,
    max_dr_depth: int = 12,
    dr_extra_depth: int = 1,
    terminals_per_axis: int = 16,
    max_phase2_depth: int = 18,
    phase2_exact_candidates: int | None = None,
    include_shortest_per_axis: bool = False,
) -> DRFullSolution | None:
    scramble_tuple = tuple(scramble)
    start = from_sequence(scramble_tuple)
    portfolio = find_dr_portfolio_ida(
        start,
        max_depth=max_dr_depth,
        extra_depth=dr_extra_depth,
        max_terminals_per_axis=terminals_per_axis,
    )
    phase2_tables = build_phase2_pruning_tables()
    ranked_terminals = list(select_dr_terminals_for_exact_evaluation(
        portfolio.terminals,
        lower_bound_top_k=phase2_exact_candidates,
        include_shortest_per_axis=include_shortest_per_axis,
        tables=phase2_tables,
    ))
    if phase2_exact_candidates is not None and not ranked_terminals:
        return None

    best: DRFullSolution | None = None
    for terminal in ranked_terminals:
        phase2 = solve_phase2(
            terminal.transformation,
            axis=terminal.axis,
            max_depth=max_phase2_depth,
            tables=phase2_tables,
        )
        if phase2 is None:
            continue
        full = terminal.sequence + phase2.sequence
        candidate = DRFullSolution(
            scramble=scramble_tuple,
            axis=terminal.axis,
            dr_sequence=terminal.sequence,
            phase2_sequence=phase2.sequence,
            full_sequence=full,
            dr_depth=len(terminal.sequence),
            phase2_depth=phase2.depth,
            total_depth=len(full),
        )
        if best is None or (candidate.total_depth, candidate.full_sequence) < (best.total_depth, best.full_sequence):
            best = candidate
    if best is not None and from_sequence(best.scramble + best.full_sequence).mapping != from_sequence(()).mapping:
        raise AssertionError("full DR solution failed physical replay")
    return best


__all__ = [
    "UD_EDGE_POSITIONS",
    "SLICE_EDGE_POSITIONS",
    "PHASE2_MOVES_UD",
    "Phase2Coordinates",
    "Phase2PruningTables",
    "Phase2Solution",
    "DRFullSolution",
    "phase2_coordinates_from_transformation",
    "build_phase2_pruning_tables",
    "phase2_lower_bound",
    "rank_dr_terminals_by_phase2_lower_bound",
    "select_dr_terminals_for_exact_evaluation",
    "solve_phase2",
    "solve_with_dr",
]
