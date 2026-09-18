from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations
import time
from typing import Iterable, Literal, Sequence

from .corner_optimal_ftm import (
    CornerProjectedState,
    apply_move as apply_corner_move,
    heuristic as corner_heuristic,
    solve_corner_optimal_ftm,
)
from .cubie_effect import CubieEffect
from .dense_projection_search_v64 import DensePairPolicy, DenseProjectionRepository
from .dense_selected_quad_projection_v70 import (
    DenseQuadPolicyBankV70,
    DenseQuadPolicyV70,
)
from .domino_reduction import (
    DRAxis,
    DRPruningTables,
    _axis_to_ud_rotation,
    build_dr_pruning_tables,
    coordinates_from_transformation,
    diagnose_transformation,
)
from .effect_library_v2 import MOVES
from .grammar_guided_gateway_v72 import (
    EXACT_TWELVE_SCRAMBLE,
    EXACT_TWELVE_WITNESS,
)
from .piece_route_graph import FACE_MOVES, PieceState
from .pieces import CORNER_ORDER, EDGE_ORDER
from .rotations import RotationCanonicalizer
from .sequences import reduce_sequence
from .transformations import PIECES, from_sequence


_PIECE_INDEX = {piece: index for index, piece in enumerate(PIECES)}
_CORNER_INDEX = {piece: index for index, piece in enumerate(CORNER_ORDER)}
_EDGE_INDEX = {piece: index for index, piece in enumerate(EDGE_ORDER)}
_OPPOSITE = {"U": "D", "D": "U", "F": "B", "B": "F", "L": "R", "R": "L"}
_AXIS = {"U": 0, "D": 0, "R": 1, "L": 1, "F": 2, "B": 2}
_FACE_ORDER = {"U": 0, "D": 1, "R": 0, "L": 1, "F": 0, "B": 1}
_CORNER_MOVES = tuple(MOVES)


def _effect(sequence: Sequence[str]) -> CubieEffect:
    return CubieEffect.from_transformation(from_sequence(tuple(sequence)))


def _piece_solved(effect: CubieEffect, piece: str) -> bool:
    piece_index = _PIECE_INDEX[piece]
    if effect.destinations[piece_index] != piece_index:
        return False
    if piece in _CORNER_INDEX:
        return effect.corner_twists[_CORNER_INDEX[piece]] == 0
    return effect.edge_flips[_EDGE_INDEX[piece]] == 0


def f2l_target_pieces(frame: str) -> tuple[str, ...]:
    """The 12 solved cubies when ``frame`` is the completed base face.

    The opposite face is the last layer.  Consequently every cubie not
    containing the opposite face belongs to the solved two-layer target.
    """
    if frame not in _OPPOSITE:
        raise ValueError(f"unknown F2L frame: {frame}")
    last_layer_face = _OPPOSITE[frame]
    return tuple(piece for piece in PIECES if last_layer_face not in piece)


def f2l_complete(effect: CubieEffect, frame: str) -> bool:
    return all(_piece_solved(effect, piece) for piece in f2l_target_pieces(frame))


def corners_solved(effect: CubieEffect) -> bool:
    return all(_piece_solved(effect, piece) for piece in CORNER_ORDER)


def full_edges_solved(effect: CubieEffect) -> bool:
    return all(_piece_solved(effect, piece) for piece in EDGE_ORDER)


def edge_permutation_solved(effect: CubieEffect) -> bool:
    return all(
        effect.destinations[_PIECE_INDEX[piece]] == _PIECE_INDEX[piece]
        for piece in EDGE_ORDER
    )


def edge_orientation_solved(effect: CubieEffect) -> bool:
    return all(value == 0 for value in effect.edge_flips)


def f2l_last_layer_structure(effect: CubieEffect, frame: str) -> dict:
    last_layer_face = _OPPOSITE[frame]
    corners = tuple(piece for piece in CORNER_ORDER if last_layer_face in piece)
    edges = tuple(piece for piece in EDGE_ORDER if last_layer_face in piece)
    return {
        "last_layer_face": last_layer_face,
        "corner_sources": list(corners),
        "corner_destinations": [
            PIECES[effect.destinations[_PIECE_INDEX[piece]]] for piece in corners
        ],
        "corner_orientations": [
            effect.corner_twists[_CORNER_INDEX[piece]] for piece in corners
        ],
        "edge_sources": list(edges),
        "edge_destinations": [
            PIECES[effect.destinations[_PIECE_INDEX[piece]]] for piece in edges
        ],
        "edge_orientations": [
            effect.edge_flips[_EDGE_INDEX[piece]] for piece in edges
        ],
    }


def _cycle_type(effect: CubieEffect, pieces: Sequence[str]) -> tuple[int, ...]:
    local_index = {piece: index for index, piece in enumerate(pieces)}
    permutation = tuple(
        local_index[PIECES[effect.destinations[_PIECE_INDEX[piece]]]]
        for piece in pieces
    )
    seen: set[int] = set()
    cycles: list[int] = []
    for start in range(len(pieces)):
        if start in seen:
            continue
        cursor = start
        length = 0
        while cursor not in seen:
            seen.add(cursor)
            length += 1
            cursor = permutation[cursor]
        cycles.append(length)
    return tuple(sorted(cycles, reverse=True))


def structural_features(
    effect: CubieEffect,
    repository: DenseProjectionRepository,
    pair_policies: Sequence[DensePairPolicy] | None = None,
) -> dict:
    codes = repository.encode_effect(effect)
    policies = tuple(pair_policies) if pair_policies is not None else repository.all_policies()
    pair_lower_bound = max((policy.distance(codes) for policy in policies), default=0)
    return {
        "pair_pdb_lower_bound": pair_lower_bound,
        "unresolved_corner_count": sum(
            not _piece_solved(effect, piece) for piece in CORNER_ORDER
        ),
        "unresolved_edge_count": sum(
            not _piece_solved(effect, piece) for piece in EDGE_ORDER
        ),
        "unresolved_edge_orientation_count": sum(effect.edge_flips),
        "corner_cycle_type": list(_cycle_type(effect, CORNER_ORDER)),
        "edge_cycle_type": list(_cycle_type(effect, EDGE_ORDER)),
        "misplaced_piece_support": sum(
            effect.destinations[index] != index for index in range(len(PIECES))
        ),
        "orientation_defect_support": (
            sum(value != 0 for value in effect.corner_twists)
            + sum(effect.edge_flips)
        ),
    }


def rotation_canonical_effect_signature(sequence: Sequence[str]) -> tuple:
    return RotationCanonicalizer.analyze(
        from_sequence(tuple(sequence))
    ).canonical.signature


@dataclass(frozen=True, slots=True)
class IndependentTarget:
    name: str
    description: str
    allowed_codes: tuple[tuple[int, tuple[int, ...]], ...]
    frame: str | None = None

    @property
    def constrained_indices(self) -> tuple[int, ...]:
        return tuple(index for index, _ in self.allowed_codes)

    def satisfied(self, codes: Sequence[int]) -> bool:
        return all(codes[index] in allowed for index, allowed in self.allowed_codes)

    def unsatisfied_count(self, codes: Sequence[int]) -> int:
        return sum(codes[index] not in allowed for index, allowed in self.allowed_codes)


def exact_piece_target(
    name: str,
    description: str,
    pieces: Sequence[str],
    repository: DenseProjectionRepository,
    *,
    frame: str | None = None,
) -> IndependentTarget:
    identity = repository.encode_effect(CubieEffect.identity())
    return IndependentTarget(
        name=name,
        description=description,
        allowed_codes=tuple(
            (_PIECE_INDEX[piece], (identity[_PIECE_INDEX[piece]],))
            for piece in pieces
        ),
        frame=frame,
    )


def f2l_target(frame: str, repository: DenseProjectionRepository) -> IndependentTarget:
    return exact_piece_target(
        f"F2L-{frame}",
        f"all cubies outside the {_OPPOSITE[frame]} last layer are solved",
        f2l_target_pieces(frame),
        repository,
        frame=frame,
    )


def full_edge_target(repository: DenseProjectionRepository) -> IndependentTarget:
    return exact_piece_target(
        "E1-full-edge-solved",
        "all 12 edge positions and orientations are solved",
        EDGE_ORDER,
        repository,
    )


def edge_permutation_target(repository: DenseProjectionRepository) -> IndependentTarget:
    allowed = []
    for piece in EDGE_ORDER:
        codes = tuple(
            repository._edge_index[PieceState(piece, orientation)]
            for orientation in (0, 1)
        )
        allowed.append((_PIECE_INDEX[piece], codes))
    return IndependentTarget(
        name="E2-edge-permutation-solved",
        description="all 12 edges are at home; edge flips remain free",
        allowed_codes=tuple(allowed),
    )


@dataclass(frozen=True, slots=True)
class MultiGoalPairPolicy:
    pieces: tuple[str, str]
    piece_indices: tuple[int, int]
    distances: tuple[int, ...]

    def distance(self, piece_codes: Sequence[int]) -> int:
        first, second = self.piece_indices
        value = self.distances[piece_codes[first] * 24 + piece_codes[second]]
        if value < 0:
            raise ValueError(f"unreachable target projection for {self.pieces}")
        return value


PairPolicy = DensePairPolicy | MultiGoalPairPolicy


def build_target_pair_policies(
    target: IndependentTarget,
    repository: DenseProjectionRepository,
) -> tuple[PairPolicy, ...]:
    allowed_by_index = dict(target.allowed_codes)
    identity = repository.encode_effect(CubieEffect.identity())
    policies: list[PairPolicy] = []
    for first_index, second_index in combinations(target.constrained_indices, 2):
        first = PIECES[first_index]
        second = PIECES[second_index]
        first_goals = allowed_by_index[first_index]
        second_goals = allowed_by_index[second_index]
        if (
            first_goals == (identity[first_index],)
            and second_goals == (identity[second_index],)
        ):
            policies.append(repository.policy(first, second))
            continue

        transitions_a = repository._transitions_for_piece(first_index)
        transitions_b = repository._transitions_for_piece(second_index)
        distances = [-1] * (24 * 24)
        queue: deque[int] = deque()
        for first_goal in first_goals:
            for second_goal in second_goals:
                state = first_goal * 24 + second_goal
                if distances[state] < 0:
                    distances[state] = 0
                    queue.append(state)
        while queue:
            state = queue.popleft()
            code_a, code_b = divmod(state, 24)
            next_depth = distances[state] + 1
            for move_index in range(len(FACE_MOVES)):
                nxt = (
                    transitions_a[code_a][move_index] * 24
                    + transitions_b[code_b][move_index]
                )
                if distances[nxt] < 0:
                    distances[nxt] = next_depth
                    queue.append(nxt)
        policies.append(MultiGoalPairPolicy(
            pieces=(first, second),
            piece_indices=(first_index, second_index),
            distances=tuple(distances),
        ))
    return tuple(policies)


def select_exact_target_quads(
    target: IndependentTarget,
    root_codes: Sequence[int],
    repository: DenseProjectionRepository,
    bank: DenseQuadPolicyBankV70,
    *,
    candidate_limit: int = 4,
    selected_limit: int = 2,
) -> tuple[tuple[DenseQuadPolicyV70, ...], dict]:
    """Select capped exact quads only when every selected goal is identity."""
    identity = repository.encode_effect(CubieEffect.identity())
    exact_pieces = tuple(
        PIECES[index]
        for index, allowed in target.allowed_codes
        if allowed == (identity[index],)
    )
    unsolved = tuple(
        piece for piece in exact_pieces
        if root_codes[_PIECE_INDEX[piece]] != identity[_PIECE_INDEX[piece]]
    )
    if len(unsolved) < 4 or candidate_limit < 1 or selected_limit < 1:
        return (), {"candidates": [], "selected": [], "build_seconds": 0.0}

    features = [
        bank.candidate_features(root_codes, quad)
        for quad in combinations(unsolved, 4)
    ]
    features.sort(key=bank._candidate_key, reverse=True)
    candidates = features[:candidate_limit]
    evaluated = []
    for row in candidates:
        policy, cache_hit = bank.policy(row.pieces)
        distance = policy.distance(root_codes)
        evaluated.append((policy, row, distance, distance - row.max_pair_distance, cache_hit))
    evaluated.sort(
        key=lambda row: (row[2], row[3], bank._candidate_key(row[1])),
        reverse=True,
    )
    selected = tuple(row[0] for row in evaluated[:selected_limit])
    return selected, {
        "candidates": [
            {
                "pieces": list(row[0].pieces),
                "distance": row[2],
                "pair_gain": row[3],
                "cache_hit": row[4],
                "build_seconds": row[0].build_seconds,
            }
            for row in evaluated
        ],
        "selected": [list(policy.pieces) for policy in selected],
        "root_distances": [policy.distance(root_codes) for policy in selected],
        "build_seconds": sum(row[0].build_seconds for row in evaluated if not row[4]),
    }


@dataclass(frozen=True, slots=True)
class ManifoldTerminal:
    target: str
    sequence: tuple[str, ...]
    effect: CubieEffect
    frame: str | None = None
    source: str = "search"

    @property
    def phase1_cost(self) -> int:
        return len(self.sequence)


@dataclass(frozen=True, slots=True)
class TargetBoundStat:
    bound: int
    expanded: int
    generated: int
    pruned_lower_bound: int
    pruned_duplicate: int
    terminals_added: int
    truncated: bool
    seconds: float


@dataclass(frozen=True, slots=True)
class TargetPortfolioResult:
    target: IndependentTarget
    terminals: tuple[ManifoldTerminal, ...]
    root_lower_bound: int
    shortest_depth: int | None
    complete_through: int
    bound_stats: tuple[TargetBoundStat, ...]
    quad_selection: dict

    @property
    def truncated(self) -> bool:
        return any(row.truncated for row in self.bound_stats)


def _canonical_successor(previous: int | None, move_index: int) -> bool:
    if previous is None:
        return True
    previous_face = FACE_MOVES[previous][0]
    face = FACE_MOVES[move_index][0]
    if previous_face == face:
        return False
    return not (
        _AXIS[previous_face] == _AXIS[face]
        and _FACE_ORDER[face] < _FACE_ORDER[previous_face]
    )


def _canonical_named_successor(
    moves: Sequence[str], previous: int | None, move_index: int
) -> bool:
    if previous is None:
        return True
    previous_face = moves[previous][0]
    face = moves[move_index][0]
    if previous_face == face:
        return False
    return not (
        _AXIS[previous_face] == _AXIS[face]
        and _FACE_ORDER[face] < _FACE_ORDER[previous_face]
    )


def find_independent_target_portfolio(
    scramble: Sequence[str],
    target: IndependentTarget,
    repository: DenseProjectionRepository,
    *,
    max_depth: int,
    extra_depth: int = 0,
    max_terminals: int = 16,
    max_expanded_per_bound: int = 1_500_000,
    quad_bank: DenseQuadPolicyBankV70 | None = None,
    quad_candidate_limit: int = 4,
    quad_selected_limit: int = 2,
) -> TargetPortfolioResult:
    """Cost-bounded IDA portfolio for an independent cubie-state target.

    Full 20-cubie codes are the transposition key, so distinct residual
    effects are not collapsed merely because they agree on the target subset.
    """
    if max_depth < 0 or extra_depth < 0:
        raise ValueError("depth limits must be non-negative")
    if max_terminals < 1 or max_expanded_per_bound < 1:
        raise ValueError("portfolio and node limits must be positive")
    start_effect = _effect(scramble)
    start_codes = repository.encode_effect(start_effect)
    pair_policies = build_target_pair_policies(target, repository)
    quads: tuple[DenseQuadPolicyV70, ...] = ()
    quad_metrics: dict = {"candidates": [], "selected": [], "build_seconds": 0.0}
    if quad_bank is not None:
        quads, quad_metrics = select_exact_target_quads(
            target,
            start_codes,
            repository,
            quad_bank,
            candidate_limit=quad_candidate_limit,
            selected_limit=quad_selected_limit,
        )

    def lower_bound(codes: Sequence[int]) -> int:
        value = 0
        for policy in pair_policies:
            value = max(value, policy.distance(codes))
        for policy in quads:
            value = max(value, policy.distance(codes))
        return value

    root_lower_bound = lower_bound(start_codes)
    move_effects = tuple(_effect((move,)) for move in FACE_MOVES)
    retained: dict[tuple, ManifoldTerminal] = {}
    shortest: int | None = None
    complete_through = root_lower_bound - 1
    stats: list[TargetBoundStat] = []

    for bound in range(root_lower_bound, max_depth + 1):
        if shortest is not None and bound > shortest + extra_depth:
            break
        started = time.perf_counter()
        expanded = generated = pruned_lb = pruned_duplicate = 0
        before_count = len(retained)
        truncated = False
        path: list[int] = []
        seen: dict[tuple[bytes, int | None], int] = {}

        def replay() -> CubieEffect:
            current = start_effect
            for index in path:
                current = move_effects[index].compose_after(current)
            return current

        def visit(
            codes: tuple[int, ...],
            depth: int,
            previous: int | None,
            known_h: int | None = None,
        ) -> None:
            nonlocal expanded, generated, pruned_lb, pruned_duplicate
            nonlocal truncated, shortest
            if truncated:
                return
            estimate = lower_bound(codes) if known_h is None else known_h
            if depth + estimate > bound:
                pruned_lb += 1
                return
            if target.satisfied(codes):
                terminal_effect = replay()
                sequence = tuple(FACE_MOVES[index] for index in path)
                old = retained.get(terminal_effect.signature)
                if old is None or (len(sequence), sequence) < (
                    old.phase1_cost,
                    old.sequence,
                ):
                    retained[terminal_effect.signature] = ManifoldTerminal(
                        target=target.name,
                        sequence=sequence,
                        effect=terminal_effect,
                        frame=target.frame,
                    )
                if shortest is None:
                    shortest = depth
                if len(retained) >= max_terminals:
                    truncated = True
                return
            if depth == bound:
                return
            if expanded >= max_expanded_per_bound:
                truncated = True
                return
            remaining = bound - depth
            key = (bytes(codes), previous)
            old_remaining = seen.get(key)
            if old_remaining is not None and old_remaining >= remaining:
                pruned_duplicate += 1
                return
            seen[key] = remaining
            expanded += 1

            children = []
            for move_index in range(len(FACE_MOVES)):
                if not _canonical_successor(previous, move_index):
                    continue
                generated += 1
                nxt = repository.advance_codes(codes, move_index)
                child_h = lower_bound(nxt)
                children.append((
                    child_h,
                    target.unsatisfied_count(nxt),
                    move_index,
                    nxt,
                ))
            children.sort(key=lambda row: (row[0], row[1], row[2]))
            for child_h, _, move_index, nxt in children:
                path.append(move_index)
                visit(nxt, depth + 1, move_index, child_h)
                path.pop()
                if truncated:
                    return

        visit(tuple(start_codes), 0, None)
        if not truncated:
            complete_through = bound
        stats.append(TargetBoundStat(
            bound=bound,
            expanded=expanded,
            generated=generated,
            pruned_lower_bound=pruned_lb,
            pruned_duplicate=pruned_duplicate,
            terminals_added=len(retained) - before_count,
            truncated=truncated,
            seconds=time.perf_counter() - started,
        ))
        if truncated and len(retained) >= max_terminals:
            break

    terminals = tuple(sorted(
        retained.values(),
        key=lambda item: (item.phase1_cost, item.sequence),
    ))
    return TargetPortfolioResult(
        target=target,
        terminals=terminals,
        root_lower_bound=root_lower_bound,
        shortest_depth=shortest,
        complete_through=complete_through,
        bound_stats=tuple(stats),
        quad_selection=quad_metrics,
    )


def expand_f2l_last_layer_orbit(
    scramble: Sequence[str],
    terminals: Sequence[ManifoldTerminal],
) -> tuple[ManifoldTerminal, ...]:
    """Add the three one-turn LL neighbours of every F2L terminal."""
    retained: dict[tuple, ManifoldTerminal] = {
        terminal.effect.signature: terminal for terminal in terminals
    }
    for terminal in tuple(terminals):
        if terminal.frame is None:
            continue
        face = _OPPOSITE[terminal.frame]
        for suffix in ("", "'", "2"):
            sequence = reduce_sequence(terminal.sequence + (f"{face}{suffix}",))
            effect = _effect(tuple(scramble) + sequence)
            if not f2l_complete(effect, terminal.frame):
                raise AssertionError("last-layer orbit move broke F2L")
            candidate = ManifoldTerminal(
                target=terminal.target,
                sequence=sequence,
                effect=effect,
                frame=terminal.frame,
                source="last_layer_orbit",
            )
            old = retained.get(effect.signature)
            if old is None or (candidate.phase1_cost, candidate.sequence) < (
                old.phase1_cost,
                old.sequence,
            ):
                retained[effect.signature] = candidate
    return tuple(sorted(
        retained.values(),
        key=lambda item: (item.phase1_cost, item.sequence),
    ))


def canonicalize_terminals(
    scramble: Sequence[str],
    terminals: Sequence[ManifoldTerminal],
) -> tuple[ManifoldTerminal, ...]:
    retained: dict[tuple, ManifoldTerminal] = {}
    for terminal in terminals:
        key = rotation_canonical_effect_signature(tuple(scramble) + terminal.sequence)
        old = retained.get(key)
        if old is None or (terminal.phase1_cost, terminal.sequence, terminal.frame or "") < (
            old.phase1_cost,
            old.sequence,
            old.frame or "",
        ):
            retained[key] = terminal
    return tuple(sorted(
        retained.values(),
        key=lambda item: (item.phase1_cost, item.sequence, item.frame or ""),
    ))


CoordinateMode = Literal["eo", "eo_slice"]


@dataclass(frozen=True, slots=True)
class CoordinatePortfolioResult:
    mode: CoordinateMode
    axes: tuple[DRAxis, ...]
    terminals: tuple[ManifoldTerminal, ...]
    shortest_depth_by_axis: tuple[tuple[DRAxis, int | None], ...]
    bound_stats: tuple[tuple[DRAxis, TargetBoundStat], ...]


def _axis_move_indices(
    axis: DRAxis,
    moves: Sequence[str],
    tables: DRPruningTables,
) -> tuple[int, ...]:
    if axis is DRAxis.UD:
        return tuple(tables.moves.index(move) for move in moves)
    rotation = _axis_to_ud_rotation(axis)
    table_effects = tuple(from_sequence((move,)) for move in tables.moves)
    mapped: list[int] = []
    for move in moves:
        normalized = rotation.apply(from_sequence((move,)))
        matches = [
            index for index, table_effect in enumerate(table_effects)
            if table_effect == normalized
        ]
        if len(matches) != 1:
            raise AssertionError(f"could not map {move} on {axis.value}")
        mapped.append(matches[0])
    return tuple(mapped)


def find_coordinate_target_portfolio(
    scramble: Sequence[str],
    *,
    mode: CoordinateMode,
    axes: Iterable[DRAxis | str] = (DRAxis.UD,),
    max_depth: int = 12,
    extra_depth: int = 1,
    max_terminals_per_axis: int = 12,
    max_expanded_per_bound: int = 500_000,
    tables: DRPruningTables | None = None,
) -> CoordinatePortfolioResult:
    if mode not in ("eo", "eo_slice"):
        raise ValueError(f"unknown coordinate target mode: {mode}")
    tables = tables or build_dr_pruning_tables()
    requested_axes = tuple(dict.fromkeys(DRAxis(axis) for axis in axes))
    start_transformation = from_sequence(tuple(scramble))
    start_effect = _effect(scramble)
    move_effects = tuple(_effect((move,)) for move in FACE_MOVES)
    all_terminals: dict[tuple, ManifoldTerminal] = {}
    shortest_rows: list[tuple[DRAxis, int | None]] = []
    stats: list[tuple[DRAxis, TargetBoundStat]] = []

    for axis in requested_axes:
        coordinates = coordinates_from_transformation(start_transformation, axis=axis)
        start_eo = coordinates.edge_orientation_index
        start_slice = coordinates.slice_combination_index
        move_indices = _axis_move_indices(axis, FACE_MOVES, tables)

        def heuristic(eo: int, sl: int) -> int:
            if mode == "eo":
                return tables.edge_orientation_distance[eo]
            return max(
                tables.edge_orientation_distance[eo],
                tables.slice_distance[sl],
                tables.edge_slice_distance[eo * 495 + sl],
            )

        def goal(eo: int, sl: int) -> bool:
            return eo == 0 and (mode == "eo" or sl == 0)

        start_h = heuristic(start_eo, start_slice)
        shortest: int | None = None
        axis_signatures: set[tuple] = set()
        for bound in range(start_h, max_depth + 1):
            if shortest is not None and bound > shortest + extra_depth:
                break
            started = time.perf_counter()
            expanded = generated = pruned = 0
            before_count = len(axis_signatures)
            truncated = False
            path: list[int] = []

            def replay() -> CubieEffect:
                current = start_effect
                for index in path:
                    current = move_effects[index].compose_after(current)
                return current

            def visit(eo: int, sl: int, depth: int, previous: int | None) -> None:
                nonlocal expanded, generated, pruned, truncated, shortest
                if truncated:
                    return
                estimate = heuristic(eo, sl)
                if depth + estimate > bound:
                    pruned += 1
                    return
                if goal(eo, sl):
                    terminal_effect = replay()
                    sequence = tuple(FACE_MOVES[index] for index in path)
                    signature = terminal_effect.signature
                    axis_signatures.add(signature)
                    old = all_terminals.get(signature)
                    candidate = ManifoldTerminal(
                        target=("E3-edge-orientation-solved" if mode == "eo" else "E4-EO-slice"),
                        sequence=sequence,
                        effect=terminal_effect,
                        frame=axis.value,
                    )
                    if old is None or (candidate.phase1_cost, candidate.sequence) < (
                        old.phase1_cost,
                        old.sequence,
                    ):
                        all_terminals[signature] = candidate
                    if shortest is None:
                        shortest = depth
                    if len(axis_signatures) >= max_terminals_per_axis:
                        truncated = True
                    return
                if depth == bound:
                    return
                if expanded >= max_expanded_per_bound:
                    truncated = True
                    return
                expanded += 1
                children = []
                for move_index, table_move_index in enumerate(move_indices):
                    if not _canonical_successor(previous, move_index):
                        continue
                    generated += 1
                    next_eo = tables.edge_orientation_move[eo][table_move_index]
                    next_slice = tables.slice_move[sl][table_move_index]
                    children.append((
                        heuristic(next_eo, next_slice),
                        move_index,
                        next_eo,
                        next_slice,
                    ))
                children.sort(key=lambda row: (row[0], row[1]))
                for _, move_index, next_eo, next_slice in children:
                    path.append(move_index)
                    visit(next_eo, next_slice, depth + 1, move_index)
                    path.pop()
                    if truncated:
                        return

            visit(start_eo, start_slice, 0, None)
            stats.append((axis, TargetBoundStat(
                bound=bound,
                expanded=expanded,
                generated=generated,
                pruned_lower_bound=pruned,
                pruned_duplicate=0,
                terminals_added=len(axis_signatures) - before_count,
                truncated=truncated,
                seconds=time.perf_counter() - started,
            )))
            if truncated and len(axis_signatures) >= max_terminals_per_axis:
                break
        shortest_rows.append((axis, shortest))

    return CoordinatePortfolioResult(
        mode=mode,
        axes=requested_axes,
        terminals=tuple(sorted(
            all_terminals.values(),
            key=lambda item: (item.phase1_cost, item.frame or "", item.sequence),
        )),
        shortest_depth_by_axis=tuple(shortest_rows),
        bound_stats=tuple(stats),
    )


def find_corner_portfolio(
    scramble: Sequence[str],
    *,
    extra_depth: int = 2,
    max_terminals: int = 24,
) -> tuple[ManifoldTerminal, ...]:
    """Reuse the corner PDB without importing the optional SymPy edge stack."""
    start_effect = _effect(scramble)
    start_corner = CornerProjectedState.from_effect(start_effect)
    optimum = solve_corner_optimal_ftm(start_corner)
    if optimum is None:
        return ()
    max_depth = optimum.distance + extra_depth
    retained: dict[tuple, ManifoldTerminal] = {}
    move_effects = tuple(_effect((move,)) for move in _CORNER_MOVES)
    layer_count = extra_depth + 1
    per_layer_limit = max(1, max_terminals // layer_count)

    # Enumerate the exact cost layers separately.  A single DFS at the largest
    # bound can otherwise fill the portfolio with long terminals before it
    # ever visits the known optimal layer.
    for bound in range(optimum.distance, max_depth + 1):
        layer: dict[tuple, ManifoldTerminal] = {}
        path: list[int] = []

        def replay() -> CubieEffect:
            current = start_effect
            for index in path:
                current = move_effects[index].compose_after(current)
            return current

        def visit(
            state: CornerProjectedState,
            depth: int,
            previous: int | None,
        ) -> None:
            if len(layer) >= per_layer_limit:
                return
            if state.solved:
                if depth != bound:
                    return
                terminal_effect = replay()
                if not corners_solved(terminal_effect):
                    raise AssertionError("corner projection terminal failed replay")
                sequence = tuple(_CORNER_MOVES[index] for index in path)
                layer.setdefault(
                    terminal_effect.signature,
                    ManifoldTerminal(
                        target="corner-solved",
                        sequence=sequence,
                        effect=terminal_effect,
                        source=f"corner_exact_depth_{bound}",
                    ),
                )
                return
            if depth == bound or depth + corner_heuristic(state) > bound:
                return
            children = []
            for move_index in range(len(_CORNER_MOVES)):
                if not _canonical_named_successor(_CORNER_MOVES, previous, move_index):
                    continue
                nxt = apply_corner_move(state, move_index)
                children.append((corner_heuristic(nxt), move_index, nxt))
            children.sort(key=lambda row: (row[0], row[1]))
            for _, move_index, nxt in children:
                path.append(move_index)
                visit(nxt, depth + 1, move_index)
                path.pop()
                if len(layer) >= per_layer_limit:
                    return

        visit(start_corner, 0, None)
        for signature, candidate in layer.items():
            old = retained.get(signature)
            if old is None or (candidate.phase1_cost, candidate.sequence) < (
                old.phase1_cost,
                old.sequence,
            ):
                retained[signature] = candidate
        if len(retained) >= max_terminals:
            break
    return tuple(sorted(
        retained.values(),
        key=lambda item: (item.phase1_cost, item.sequence),
    ))


def terminal_matches_definition(terminal: ManifoldTerminal) -> bool:
    if terminal.target.startswith("F2L-"):
        return terminal.frame is not None and f2l_complete(terminal.effect, terminal.frame)
    if terminal.target == "E1-full-edge-solved":
        return full_edges_solved(terminal.effect)
    if terminal.target == "E2-edge-permutation-solved":
        return edge_permutation_solved(terminal.effect)
    if terminal.target == "E3-edge-orientation-solved":
        return edge_orientation_solved(terminal.effect)
    if terminal.target == "E4-EO-slice":
        if terminal.frame is None:
            return False
        diagnostic = diagnose_transformation(
            from_sequence(EXACT_TWELVE_SCRAMBLE + terminal.sequence),
            axis=DRAxis(terminal.frame),
        )
        return diagnostic.edge_orientation_solved and diagnostic.slice_edges_in_slice
    if terminal.target == "corner-solved":
        return corners_solved(terminal.effect)
    return True


__all__ = [
    "EXACT_TWELVE_SCRAMBLE",
    "EXACT_TWELVE_WITNESS",
    "CoordinatePortfolioResult",
    "IndependentTarget",
    "ManifoldTerminal",
    "MultiGoalPairPolicy",
    "TargetBoundStat",
    "TargetPortfolioResult",
    "build_target_pair_policies",
    "canonicalize_terminals",
    "corners_solved",
    "edge_orientation_solved",
    "edge_permutation_solved",
    "edge_permutation_target",
    "exact_piece_target",
    "expand_f2l_last_layer_orbit",
    "f2l_complete",
    "f2l_last_layer_structure",
    "f2l_target",
    "f2l_target_pieces",
    "find_coordinate_target_portfolio",
    "find_corner_portfolio",
    "find_independent_target_portfolio",
    "full_edge_target",
    "full_edges_solved",
    "rotation_canonical_effect_signature",
    "select_exact_target_quads",
    "structural_features",
    "terminal_matches_definition",
]
