from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import comb
from functools import lru_cache
from typing import Iterable, Literal

from .cubie_effect import CubieEffect
from .pieces import CORNER_ORDER, EDGE_ORDER
from .rotations import CUBE_ROTATIONS, CubeRotation
from .transformations import PIECES, Transformation, compose, from_sequence

_PIECE_INDEX = {piece: index for index, piece in enumerate(PIECES)}
_EDGE_INDEX = {piece: index for index, piece in enumerate(EDGE_ORDER)}


class DRAxis(str, Enum):
    UD = "UD"
    FB = "FB"
    RL = "RL"


_AXIS_FACES: dict[DRAxis, frozenset[str]] = {
    DRAxis.UD: frozenset({"U", "D"}),
    DRAxis.FB: frozenset({"F", "B"}),
    DRAxis.RL: frozenset({"R", "L"}),
}

# In the UD-normalized frame these four edge pieces must occupy the four
# middle-slice positions.  CubieEffect.destinations is indexed by source piece.
_UD_SLICE_EDGES = frozenset({"FR", "BR", "BL", "FL"})


@dataclass(frozen=True, slots=True)
class DRCoordinates:
    axis: DRAxis
    corner_orientation: tuple[int, ...]
    edge_orientation: tuple[int, ...]
    slice_occupancy: tuple[int, ...]

    @property
    def corner_orientation_index(self) -> int:
        value = 0
        for twist in self.corner_orientation[:-1]:
            value = value * 3 + twist
        return value

    @property
    def edge_orientation_index(self) -> int:
        value = 0
        for flip in self.edge_orientation[:-1]:
            value = value * 2 + flip
        return value

    @property
    def slice_combination_index(self) -> int:
        """Rank the occupied four-position subset among C(12, 4)=495 subsets."""
        selected = [index for index, bit in enumerate(self.slice_occupancy) if bit]
        rank = 0
        previous = -1
        remaining = 4
        for chosen in selected:
            for candidate in range(previous + 1, chosen):
                rank += comb(12 - candidate - 1, remaining - 1)
            previous = chosen
            remaining -= 1
        solved_rank = 425
        return (rank - solved_rank) % 495

    @property
    def is_reduced(self) -> bool:
        return (
            all(value == 0 for value in self.corner_orientation)
            and all(value == 0 for value in self.edge_orientation)
            and self.slice_occupancy == (0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0)
        )


@dataclass(frozen=True, slots=True)
class DRDiagnostic:
    coordinates: DRCoordinates
    corner_orientation_solved: bool
    edge_orientation_solved: bool
    slice_edges_in_slice: bool
    misplaced_slice_edges: tuple[str, ...]
    non_slice_edges_in_slice: tuple[str, ...]

    @property
    def is_reduced(self) -> bool:
        return (
            self.corner_orientation_solved
            and self.edge_orientation_solved
            and self.slice_edges_in_slice
        )


def _axis_to_ud_rotation(axis: DRAxis) -> CubeRotation:
    source = _AXIS_FACES[axis]
    matches = [
        rotation
        for rotation in CUBE_ROTATIONS
        if rotation.map_face_set(source) == _AXIS_FACES[DRAxis.UD]
    ]
    if not matches:
        raise AssertionError(f"No cube rotation maps {axis.value} to UD")
    return min(matches, key=lambda item: item.key)


def normalize_transformation_to_ud(
    transformation: Transformation,
    axis: DRAxis | str,
) -> Transformation:
    axis = DRAxis(axis)
    if axis is DRAxis.UD:
        return transformation
    return _axis_to_ud_rotation(axis).apply(transformation)


def coordinates_from_effect(
    effect: CubieEffect,
    *,
    axis: DRAxis | str = DRAxis.UD,
) -> DRCoordinates:
    """Extract DR coordinates from a UD-normalized CubieEffect.

    Non-UD axes require relabelling the complete physical transformation, not
    merely reinterpreting the stored UD orientation digits.  Therefore this
    low-level function deliberately accepts UD only.  Use
    ``coordinates_from_transformation`` or ``coordinates_from_sequence`` for
    FB/RL axes.
    """
    axis = DRAxis(axis)
    if axis is not DRAxis.UD:
        raise ValueError(
            "CubieEffect orientation digits are UD-normalized; use "
            "coordinates_from_transformation/sequence for FB or RL axes"
        )

    slice_occupancy = []
    for position in EDGE_ORDER:
        occupant_is_slice = False
        for source in _UD_SLICE_EDGES:
            destination = PIECES[effect.destinations[_PIECE_INDEX[source]]]
            if destination == position:
                occupant_is_slice = True
                break
        slice_occupancy.append(1 if occupant_is_slice else 0)

    # Search coordinates must be indexed by *position*, not by source piece.
    # CubieEffect stores orientation deltas by source piece, so invert the
    # permutation here.  Position-indexed orientation is a true quotient
    # coordinate: its transition under a face turn does not depend on the
    # hidden permutation.
    corner_by_position = [0] * len(CORNER_ORDER)
    for source_index, source in enumerate(CORNER_ORDER):
        destination = PIECES[effect.destinations[_PIECE_INDEX[source]]]
        corner_by_position[CORNER_ORDER.index(destination)] = effect.corner_twists[source_index]

    edge_by_position = [0] * len(EDGE_ORDER)
    for source_index, source in enumerate(EDGE_ORDER):
        destination = PIECES[effect.destinations[_PIECE_INDEX[source]]]
        edge_by_position[EDGE_ORDER.index(destination)] = effect.edge_flips[source_index]

    return DRCoordinates(
        axis=DRAxis.UD,
        corner_orientation=tuple(corner_by_position),
        edge_orientation=tuple(edge_by_position),
        slice_occupancy=tuple(slice_occupancy),
    )


def coordinates_from_transformation(
    transformation: Transformation,
    *,
    axis: DRAxis | str = DRAxis.UD,
) -> DRCoordinates:
    axis = DRAxis(axis)
    normalized = normalize_transformation_to_ud(transformation, axis)
    base = coordinates_from_effect(CubieEffect.from_transformation(normalized))
    return DRCoordinates(
        axis=axis,
        corner_orientation=base.corner_orientation,
        edge_orientation=base.edge_orientation,
        slice_occupancy=base.slice_occupancy,
    )


def coordinates_from_sequence(
    sequence: Iterable[str],
    *,
    axis: DRAxis | str = DRAxis.UD,
) -> DRCoordinates:
    return coordinates_from_transformation(from_sequence(tuple(sequence)), axis=axis)


def diagnose_transformation(
    transformation: Transformation,
    *,
    axis: DRAxis | str = DRAxis.UD,
) -> DRDiagnostic:
    axis = DRAxis(axis)
    normalized = normalize_transformation_to_ud(transformation, axis)
    effect = CubieEffect.from_transformation(normalized)
    coordinates = coordinates_from_effect(effect)
    coordinates = DRCoordinates(
        axis=axis,
        corner_orientation=coordinates.corner_orientation,
        edge_orientation=coordinates.edge_orientation,
        slice_occupancy=coordinates.slice_occupancy,
    )

    misplaced_slice: list[str] = []
    non_slice_in_slice: list[str] = []
    for source in EDGE_ORDER:
        destination = PIECES[effect.destinations[_PIECE_INDEX[source]]]
        if source in _UD_SLICE_EDGES and destination not in _UD_SLICE_EDGES:
            misplaced_slice.append(source)
        if source not in _UD_SLICE_EDGES and destination in _UD_SLICE_EDGES:
            non_slice_in_slice.append(source)

    return DRDiagnostic(
        coordinates=coordinates,
        corner_orientation_solved=all(v == 0 for v in coordinates.corner_orientation),
        edge_orientation_solved=all(v == 0 for v in coordinates.edge_orientation),
        slice_edges_in_slice=not misplaced_slice,
        misplaced_slice_edges=tuple(misplaced_slice),
        non_slice_edges_in_slice=tuple(non_slice_in_slice),
    )


def diagnose_sequence(
    sequence: Iterable[str],
    *,
    axis: DRAxis | str = DRAxis.UD,
) -> DRDiagnostic:
    return diagnose_transformation(from_sequence(tuple(sequence)), axis=axis)


def is_dr_sequence(sequence: Iterable[str], *, axis: DRAxis | str = DRAxis.UD) -> bool:
    return diagnose_sequence(sequence, axis=axis).is_reduced


DR_SUBGROUP_MOVES_UD: tuple[str, ...] = (
    "U", "U'", "U2", "D", "D'", "D2", "R2", "L2", "F2", "B2"
)


__all__ = [
    "DRAxis",
    "DRCoordinates",
    "DRDiagnostic",
    "DR_SUBGROUP_MOVES_UD",
    "normalize_transformation_to_ud",
    "coordinates_from_effect",
    "coordinates_from_transformation",
    "coordinates_from_sequence",
    "diagnose_transformation",
    "diagnose_sequence",
    "is_dr_sequence",
    "ALL_FACE_TURNS",
    "DRTerminal",
    "find_dr_terminals",
    "find_dr_terminals_from_sequence",
    "DRPruningTables",
    "build_dr_pruning_tables",
    "dr_heuristic",
    "find_dr_terminals_ida",
    "find_dr_terminals_ida_from_sequence",
]

ALL_FACE_TURNS: tuple[str, ...] = tuple(
    f"{face}{suffix}"
    for face in "UDRLFB"
    for suffix in ("", "'", "2")
)

_OPPOSITE_FACE = {"U": "D", "D": "U", "R": "L", "L": "R", "F": "B", "B": "F"}
_FACE_ORDER = {face: index for index, face in enumerate("UDRLFB")}


@dataclass(frozen=True, slots=True)
class DRTerminal:
    axis: DRAxis
    sequence: tuple[str, ...]
    transformation: Transformation
    diagnostic: DRDiagnostic

    @property
    def depth(self) -> int:
        return len(self.sequence)


def _move_is_canonical(path: tuple[str, ...], move: str) -> bool:
    if not path:
        return True
    previous_face = path[-1][0]
    face = move[0]
    if face == previous_face:
        return False
    # Opposite faces commute. Keep only one ordering to avoid duplicate branches.
    if _OPPOSITE_FACE[previous_face] == face and _FACE_ORDER[face] < _FACE_ORDER[previous_face]:
        return False
    return True


def find_dr_terminals(
    transformation: Transformation,
    *,
    axes: Iterable[DRAxis | str] = tuple(DRAxis),
    max_depth: int = 6,
    max_terminals: int = 128,
    moves: Iterable[str] = ALL_FACE_TURNS,
) -> tuple[DRTerminal, ...]:
    """Find unique shortest DR terminals for every requested axis.

    This DR-1 search is deliberately table-free. It is intended for semantic
    verification and shallow research probes before coordinate pruning tables
    are introduced in DR-2. Each axis is searched independently, so a shallow
    terminal on one axis does not hide a deeper shortest terminal on another.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if max_terminals <= 0:
        return ()

    requested_axes = tuple(dict.fromkeys(DRAxis(axis) for axis in axes))
    move_names = tuple(moves)
    move_transformations = {name: from_sequence((name,)) for name in move_names}
    all_found: list[DRTerminal] = []

    for axis in requested_axes:
        axis_found: dict[tuple[str, ...], DRTerminal] = {}
        remaining_capacity = max_terminals - len(all_found)
        if remaining_capacity <= 0:
            break

        for depth_limit in range(max_depth + 1):
            axis_found.clear()

            def visit(current: Transformation, path: tuple[str, ...], remaining: int) -> None:
                if len(axis_found) >= remaining_capacity:
                    return
                diagnostic = diagnose_transformation(current, axis=axis)
                if diagnostic.is_reduced:
                    mapping_key = tuple(current.mapping[piece] for piece in PIECES)
                    axis_found.setdefault(
                        mapping_key,
                        DRTerminal(axis, path, current, diagnostic),
                    )
                    return
                if remaining == 0:
                    return
                for move in move_names:
                    if not _move_is_canonical(path, move):
                        continue
                    next_transformation = compose(move_transformations[move], current)
                    visit(next_transformation, path + (move,), remaining - 1)
                    if len(axis_found) >= remaining_capacity:
                        return

            visit(transformation, (), depth_limit)
            if axis_found:
                all_found.extend(axis_found.values())
                break

    return tuple(
        sorted(all_found, key=lambda item: (item.depth, item.axis.value, item.sequence))
    )


def find_dr_terminals_from_sequence(
    sequence: Iterable[str],
    **kwargs: object,
) -> tuple[DRTerminal, ...]:
    return find_dr_terminals(from_sequence(tuple(sequence)), **kwargs)


# ---------------------------------------------------------------------------
# DR-2: coordinate move tables, pruning tables, and IDA* search
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DRPruningTables:
    moves: tuple[str, ...]
    corner_orientation_move: tuple[tuple[int, ...], ...]
    edge_orientation_move: tuple[tuple[int, ...], ...]
    slice_move: tuple[tuple[int, ...], ...]
    corner_orientation_distance: bytes
    edge_orientation_distance: bytes
    slice_distance: bytes
    corner_slice_distance: bytes
    edge_slice_distance: bytes

    def heuristic(self, coordinates: DRCoordinates) -> int:
        co = coordinates.corner_orientation_index
        eo = coordinates.edge_orientation_index
        sl = coordinates.slice_combination_index
        return max(
            self.corner_orientation_distance[co],
            self.edge_orientation_distance[eo],
            self.slice_distance[sl],
            self.corner_slice_distance[co * 495 + sl],
            self.edge_slice_distance[eo * 495 + sl],
        )


def _decode_orientation(index: int, base: int, length: int) -> tuple[int, ...]:
    values = [0] * length
    total = 0
    for position in range(length - 2, -1, -1):
        values[position] = index % base
        total += values[position]
        index //= base
    values[-1] = (-total) % base
    return tuple(values)


def _encode_orientation(values: tuple[int, ...], base: int) -> int:
    value = 0
    for digit in values[:-1]:
        value = value * base + digit
    return value


def _raw_slice_rank(occupancy: tuple[int, ...]) -> int:
    selected = [index for index, bit in enumerate(occupancy) if bit]
    rank = 0
    previous = -1
    remaining = 4
    for chosen in selected:
        for candidate in range(previous + 1, chosen):
            rank += comb(12 - candidate - 1, remaining - 1)
        previous = chosen
        remaining -= 1
    return rank


def _decode_raw_slice_rank(rank: int) -> tuple[int, ...]:
    selected: list[int] = []
    previous = -1
    remaining = 4
    for _ in range(4):
        for candidate in range(previous + 1, 12):
            count = comb(12 - candidate - 1, remaining - 1) if remaining > 1 else 1
            if rank < count:
                selected.append(candidate)
                previous = candidate
                remaining -= 1
                break
            rank -= count
    occupancy = [0] * 12
    for index in selected:
        occupancy[index] = 1
    return tuple(occupancy)


def _decode_slice_index(index: int) -> tuple[int, ...]:
    # Public index is shifted so the solved slice subset has index zero.
    return _decode_raw_slice_rank((index + 425) % 495)


def _encode_slice_index(occupancy: tuple[int, ...]) -> int:
    return (_raw_slice_rank(occupancy) - 425) % 495


def _move_position_data(move: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    effect = CubieEffect.from_transformation(from_sequence((move,)))
    corner_destination = []
    for position in CORNER_ORDER:
        destination = PIECES[effect.destinations[_PIECE_INDEX[position]]]
        corner_destination.append(CORNER_ORDER.index(destination))
    edge_destination = []
    for position in EDGE_ORDER:
        destination = PIECES[effect.destinations[_PIECE_INDEX[position]]]
        edge_destination.append(EDGE_ORDER.index(destination))
    return (
        tuple(corner_destination),
        effect.corner_twists,
        tuple(edge_destination),
        effect.edge_flips,
    )


def _transition_orientation(
    values: tuple[int, ...],
    destination: tuple[int, ...],
    delta: tuple[int, ...],
    base: int,
) -> tuple[int, ...]:
    result = [0] * len(values)
    for old_position, new_position in enumerate(destination):
        result[new_position] = (values[old_position] + delta[old_position]) % base
    return tuple(result)


def _transition_occupancy(
    occupancy: tuple[int, ...], destination: tuple[int, ...]
) -> tuple[int, ...]:
    result = [0] * len(occupancy)
    for old_position, new_position in enumerate(destination):
        result[new_position] = occupancy[old_position]
    return tuple(result)


def _build_distance_table(move_table: tuple[tuple[int, ...], ...]) -> bytes:
    unseen = 255
    distance = bytearray([unseen]) * len(move_table)
    distance[0] = 0
    queue: deque[int] = deque([0])
    while queue:
        state = queue.popleft()
        next_distance = distance[state] + 1
        for neighbour in move_table[state]:
            if distance[neighbour] == unseen:
                distance[neighbour] = next_distance
                queue.append(neighbour)
    if unseen in distance:
        raise AssertionError("coordinate graph was not fully reached")
    return bytes(distance)


def _build_pair_distance_table(
    first_move_table: tuple[tuple[int, ...], ...],
    slice_move_table: tuple[tuple[int, ...], ...],
) -> bytes:
    unseen = 255
    width = 495
    distance = bytearray([unseen]) * (len(first_move_table) * width)
    distance[0] = 0
    queue: deque[int] = deque([0])
    while queue:
        state = queue.popleft()
        first, slice_index = divmod(state, width)
        next_distance = distance[state] + 1
        first_row = first_move_table[first]
        slice_row = slice_move_table[slice_index]
        for move_index in range(len(first_row)):
            neighbour = first_row[move_index] * width + slice_row[move_index]
            if distance[neighbour] == unseen:
                distance[neighbour] = next_distance
                queue.append(neighbour)
    if unseen in distance:
        raise AssertionError("pair coordinate graph was not fully reached")
    return bytes(distance)


@lru_cache(maxsize=1)
def build_dr_pruning_tables() -> DRPruningTables:
    """Build reusable DR-2 coordinate transition and distance tables.

    The compact single-coordinate tables are supplemented by the standard
    CO+Slice and EO+Slice pair pattern databases.  Their maximum is an
    admissible Phase-1/DR heuristic and is strong enough for practical IDA*.
    """
    moves = ALL_FACE_TURNS
    move_data = tuple(_move_position_data(move) for move in moves)

    co_rows: list[tuple[int, ...]] = []
    for index in range(2187):
        values = _decode_orientation(index, 3, 8)
        co_rows.append(tuple(
            _encode_orientation(
                _transition_orientation(values, corner_dest, corner_delta, 3), 3
            )
            for corner_dest, corner_delta, _, _ in move_data
        ))

    eo_rows: list[tuple[int, ...]] = []
    for index in range(2048):
        values = _decode_orientation(index, 2, 12)
        eo_rows.append(tuple(
            _encode_orientation(
                _transition_orientation(values, edge_dest, edge_delta, 2), 2
            )
            for _, _, edge_dest, edge_delta in move_data
        ))

    slice_rows: list[tuple[int, ...]] = []
    for index in range(495):
        occupancy = _decode_slice_index(index)
        slice_rows.append(tuple(
            _encode_slice_index(_transition_occupancy(occupancy, edge_dest))
            for _, _, edge_dest, _ in move_data
        ))

    co_table = tuple(co_rows)
    eo_table = tuple(eo_rows)
    slice_table = tuple(slice_rows)
    return DRPruningTables(
        moves=moves,
        corner_orientation_move=co_table,
        edge_orientation_move=eo_table,
        slice_move=slice_table,
        corner_orientation_distance=_build_distance_table(co_table),
        edge_orientation_distance=_build_distance_table(eo_table),
        slice_distance=_build_distance_table(slice_table),
        corner_slice_distance=_build_pair_distance_table(co_table, slice_table),
        edge_slice_distance=_build_pair_distance_table(eo_table, slice_table),
    )


def dr_heuristic(
    transformation: Transformation,
    *,
    axis: DRAxis | str = DRAxis.UD,
    tables: DRPruningTables | None = None,
) -> int:
    tables = tables or build_dr_pruning_tables()
    return tables.heuristic(coordinates_from_transformation(transformation, axis=axis))


def find_dr_terminals_ida(
    transformation: Transformation,
    *,
    axes: Iterable[DRAxis | str] = tuple(DRAxis),
    max_depth: int = 12,
    max_terminals: int = 128,
    moves: Iterable[str] = ALL_FACE_TURNS,
    tables: DRPruningTables | None = None,
) -> tuple[DRTerminal, ...]:
    """Find shortest DR terminals using the DR-2 admissible heuristic."""
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if max_terminals <= 0:
        return ()
    tables = tables or build_dr_pruning_tables()
    requested_axes = tuple(dict.fromkeys(DRAxis(axis) for axis in axes))
    move_names = tuple(moves)
    move_transformations = {name: from_sequence((name,)) for name in move_names}
    all_found: list[DRTerminal] = []

    for axis in requested_axes:
        remaining_capacity = max_terminals - len(all_found)
        if remaining_capacity <= 0:
            break
        start_bound = dr_heuristic(transformation, axis=axis, tables=tables)
        for bound in range(start_bound, max_depth + 1):
            found: dict[tuple[str, ...], DRTerminal] = {}

            def visit(current: Transformation, path: tuple[str, ...]) -> None:
                if len(found) >= remaining_capacity:
                    return
                diagnostic = diagnose_transformation(current, axis=axis)
                estimate = tables.heuristic(diagnostic.coordinates)
                if len(path) + estimate > bound:
                    return
                if diagnostic.is_reduced:
                    mapping_key = tuple(current.mapping[piece] for piece in PIECES)
                    found.setdefault(mapping_key, DRTerminal(axis, path, current, diagnostic))
                    return
                if len(path) == bound:
                    return
                for move in move_names:
                    if not _move_is_canonical(path, move):
                        continue
                    visit(compose(move_transformations[move], current), path + (move,))
                    if len(found) >= remaining_capacity:
                        return

            visit(transformation, ())
            if found:
                all_found.extend(found.values())
                break

    return tuple(sorted(all_found, key=lambda item: (item.depth, item.axis.value, item.sequence)))


def find_dr_terminals_ida_from_sequence(
    sequence: Iterable[str], **kwargs: object
) -> tuple[DRTerminal, ...]:
    return find_dr_terminals_ida(from_sequence(tuple(sequence)), **kwargs)


# ---------------------------------------------------------------------------
# DR-3: coordinate-only IDA* portfolio search and benchmark statistics
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DRBoundStat:
    axis: DRAxis
    bound: int
    nodes: int
    terminals: int


@dataclass(frozen=True, slots=True)
class DRSearchResult:
    terminals: tuple[DRTerminal, ...]
    shortest_depth_by_axis: tuple[tuple[DRAxis, int | None], ...]
    bound_stats: tuple[DRBoundStat, ...]

    def shortest_depth(self, axis: DRAxis | str) -> int | None:
        target = DRAxis(axis)
        return dict(self.shortest_depth_by_axis)[target]


def _coordinate_indices(
    transformation: Transformation,
    axis: DRAxis,
) -> tuple[int, int, int]:
    coordinates = coordinates_from_transformation(transformation, axis=axis)
    return (
        coordinates.corner_orientation_index,
        coordinates.edge_orientation_index,
        coordinates.slice_combination_index,
    )


def find_dr_portfolio_ida(
    transformation: Transformation,
    *,
    axes: Iterable[DRAxis | str] = tuple(DRAxis),
    max_depth: int = 12,
    extra_depth: int = 0,
    max_terminals_per_axis: int = 32,
    moves: Iterable[str] = ALL_FACE_TURNS,
    tables: DRPruningTables | None = None,
) -> DRSearchResult:
    """Coordinate-only IDA* returning shortest and extra-depth DR terminals.

    The search state is only ``(CO, EO, Slice)``. Full transformations are
    replayed only when a DR leaf is reached, which makes general-scramble DR
    searches substantially faster than the DR-2 reference implementation.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if extra_depth < 0:
        raise ValueError("extra_depth must be non-negative")
    if max_terminals_per_axis <= 0:
        return DRSearchResult((), tuple((DRAxis(a), None) for a in axes), ())

    tables = tables or build_dr_pruning_tables()
    requested_axes = tuple(dict.fromkeys(DRAxis(axis) for axis in axes))
    move_names = tuple(moves)
    try:
        move_indices = tuple(tables.moves.index(move) for move in move_names)
    except ValueError as exc:
        raise ValueError("all search moves must exist in the pruning tables") from exc
    move_transformations = {name: from_sequence((name,)) for name in move_names}
    table_move_transformations = {name: from_sequence((name,)) for name in tables.moves}

    all_terminals: list[DRTerminal] = []
    shortest_rows: list[tuple[DRAxis, int | None]] = []
    stats: list[DRBoundStat] = []

    for axis in requested_axes:
        start = _coordinate_indices(transformation, axis)
        if axis is DRAxis.UD:
            axis_move_indices = move_indices
        else:
            rotation = _axis_to_ud_rotation(axis)
            mapped_indices: list[int] = []
            for move in move_names:
                normalized_move = rotation.apply(move_transformations[move])
                matches = [
                    index
                    for index, table_move in enumerate(tables.moves)
                    if table_move_transformations[table_move] == normalized_move
                ]
                if len(matches) != 1:
                    raise AssertionError(f"could not normalize move {move} for axis {axis.value}")
                mapped_indices.append(matches[0])
            axis_move_indices = tuple(mapped_indices)
        start_h = max(
            tables.corner_slice_distance[start[0] * 495 + start[2]],
            tables.edge_slice_distance[start[1] * 495 + start[2]],
        )
        shortest: int | None = None
        axis_found: dict[tuple[str, ...], DRTerminal] = {}

        for bound in range(start_h, max_depth + 1):
            if shortest is not None and bound > shortest + extra_depth:
                break
            nodes = 0
            found_this_bound = 0

            def visit(
                co: int,
                eo: int,
                sl: int,
                path: tuple[str, ...],
            ) -> None:
                nonlocal nodes, found_this_bound, shortest
                if len(axis_found) >= max_terminals_per_axis:
                    return
                nodes += 1
                estimate = max(
                    tables.corner_slice_distance[co * 495 + sl],
                    tables.edge_slice_distance[eo * 495 + sl],
                )
                if len(path) + estimate > bound:
                    return
                if co == 0 and eo == 0 and sl == 0:
                    if shortest is None:
                        shortest = len(path)
                    current = transformation
                    for move in path:
                        current = compose(move_transformations[move], current)
                    diagnostic = diagnose_transformation(current, axis=axis)
                    if not diagnostic.is_reduced:
                        raise AssertionError("coordinate DR leaf failed physical replay")
                    mapping_key = tuple(current.mapping[piece] for piece in PIECES)
                    if mapping_key not in axis_found:
                        axis_found[mapping_key] = DRTerminal(axis, path, current, diagnostic)
                        found_this_bound += 1
                    return
                if len(path) == bound:
                    return
                for move, move_index in zip(move_names, axis_move_indices):
                    if not _move_is_canonical(path, move):
                        continue
                    visit(
                        tables.corner_orientation_move[co][move_index],
                        tables.edge_orientation_move[eo][move_index],
                        tables.slice_move[sl][move_index],
                        path + (move,),
                    )
                    if len(axis_found) >= max_terminals_per_axis:
                        return

            visit(*start, ())
            stats.append(DRBoundStat(axis, bound, nodes, found_this_bound))
            if shortest is not None and bound >= shortest + extra_depth:
                break

        shortest_rows.append((axis, shortest))
        all_terminals.extend(axis_found.values())

    return DRSearchResult(
        terminals=tuple(sorted(
            all_terminals,
            key=lambda item: (item.depth, item.axis.value, item.sequence),
        )),
        shortest_depth_by_axis=tuple(shortest_rows),
        bound_stats=tuple(stats),
    )


def find_dr_portfolio_ida_from_sequence(
    sequence: Iterable[str], **kwargs: object
) -> DRSearchResult:
    return find_dr_portfolio_ida(from_sequence(tuple(sequence)), **kwargs)


__all__ += [
    "DRBoundStat",
    "DRSearchResult",
    "find_dr_portfolio_ida",
    "find_dr_portfolio_ida_from_sequence",
]
