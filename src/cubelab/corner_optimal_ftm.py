from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from .cubie_effect import CubieEffect
from .effect_library_v2 import MOVES
from .pieces import CORNER_ORDER
from .transformations import PIECES, from_sequence

_CIDX = {piece: i for i, piece in enumerate(CORNER_ORDER)}
_PIDX = {piece: i for i, piece in enumerate(PIECES)}
_MOVE_NAMES = tuple(MOVES)
_FACES = tuple(m[0] for m in _MOVE_NAMES)
_AXIS = {'U': 0, 'D': 0, 'R': 1, 'L': 1, 'F': 2, 'B': 2}
_FACE_ORDER = {'U': 0, 'D': 1, 'R': 0, 'L': 1, 'F': 0, 'B': 1}


@dataclass(frozen=True, slots=True)
class CornerProjectedState:
    destinations: tuple[int, ...]
    twists: tuple[int, ...]

    @classmethod
    def identity(cls) -> 'CornerProjectedState':
        return cls(tuple(range(8)), (0,) * 8)

    @classmethod
    def from_effect(cls, effect: CubieEffect) -> 'CornerProjectedState':
        destinations = tuple(_CIDX[PIECES[effect.destinations[_PIDX[p]]]] for p in CORNER_ORDER)
        return cls(destinations, tuple(t % 3 for t in effect.corner_twists))

    @property
    def solved(self) -> bool:
        return self.destinations == tuple(range(8)) and self.twists == (0,) * 8


@dataclass(frozen=True, slots=True)
class CornerOptimalResult:
    distance: int
    sequence: tuple[str, ...]
    expanded: int
    thresholds: tuple[int, ...]


@lru_cache(maxsize=1)
def move_tables() -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    rows = []
    for name in _MOVE_NAMES:
        effect = CubieEffect.from_transformation(from_sequence((name,)))
        state = CornerProjectedState.from_effect(effect)
        rows.append((state.destinations, state.twists))
    return tuple(rows)


@lru_cache(maxsize=500000)
def apply_move(state: CornerProjectedState, move_index: int) -> CornerProjectedState:
    move_dest, move_twist = move_tables()[move_index]
    destinations = tuple(move_dest[mid] for mid in state.destinations)
    twists = tuple((state.twists[src] + move_twist[state.destinations[src]]) % 3 for src in range(8))
    return CornerProjectedState(destinations, twists)


def _permutation_projection(state: CornerProjectedState) -> tuple[int, ...]:
    return state.destinations


def _orientation_projection(state: CornerProjectedState) -> tuple[int, ...]:
    return state.twists


@lru_cache(maxsize=1)
def corner_permutation_pdb() -> dict[tuple[int, ...], int]:
    start = tuple(range(8))
    distances = {start: 0}
    q = deque([start])
    tables = move_tables()
    while q:
        perm = q.popleft()
        depth = distances[perm] + 1
        for move_dest, _ in tables:
            nxt = tuple(move_dest[mid] for mid in perm)
            if nxt not in distances:
                distances[nxt] = depth
                q.append(nxt)
    return distances


@lru_cache(maxsize=1)
def corner_orientation_pdb() -> dict[tuple[int, ...], int]:
    start = (0,) * 8
    distances = {start: 0}
    q = deque([start])
    tables = move_tables()
    while q:
        twists = q.popleft()
        depth = distances[twists] + 1
        for move_dest, move_twist in tables:
            # For an orientation-only projection, source cubies are currently at
            # positions encoded by an implicit identity permutation only at the
            # root. Orientation transport depends on permutation, so a twist tuple
            # indexed by source is not Markov by itself. Use position-indexed
            # orientation instead: nxt at destination = old at source + delta.
            pos = [0] * 8
            for src, dest in enumerate(move_dest):
                pos[dest] = (twists[src] + move_twist[src]) % 3
            nxt = tuple(pos)
            if nxt not in distances:
                distances[nxt] = depth
                q.append(nxt)
    return distances


def _position_orientations(state: CornerProjectedState) -> tuple[int, ...]:
    pos = [0] * 8
    for src, dest in enumerate(state.destinations):
        pos[dest] = state.twists[src]
    return tuple(pos)


@lru_cache(maxsize=500000)
def heuristic(state: CornerProjectedState) -> int:
    return max(
        corner_permutation_pdb()[_permutation_projection(state)],
        corner_orientation_pdb()[_position_orientations(state)],
    )


def _canonical_successor(prev_move: int | None, move_index: int) -> bool:
    if prev_move is None:
        return True
    prev_face = _FACES[prev_move]
    face = _FACES[move_index]
    if face == prev_face:
        return False
    # Opposite faces commute. Keep only one canonical order within an axis.
    if _AXIS[face] == _AXIS[prev_face] and _FACE_ORDER[face] < _FACE_ORDER[prev_face]:
        return False
    return True


def solve_corner_optimal_ftm(
    effect_or_state: CubieEffect | CornerProjectedState,
    *,
    max_depth: int = 20,
) -> CornerOptimalResult | None:
    state = effect_or_state if isinstance(effect_or_state, CornerProjectedState) else CornerProjectedState.from_effect(effect_or_state)
    if state.solved:
        return CornerOptimalResult(0, (), 0, (0,))

    threshold = heuristic(state)
    thresholds = []
    expanded = 0
    path: list[int] = []

    def dfs(cur: CornerProjectedState, g: int, bound: int, prev_move: int | None):
        nonlocal expanded
        h = heuristic(cur)
        f = g + h
        if f > bound:
            return f
        if cur.solved:
            return True
        expanded += 1
        minimum = 10**9
        ordered = []
        for idx in range(len(_MOVE_NAMES)):
            if not _canonical_successor(prev_move, idx):
                continue
            nxt = apply_move(cur, idx)
            ordered.append((heuristic(nxt), idx, nxt))
        ordered.sort(key=lambda row: (row[0], _MOVE_NAMES[row[1]]))
        for _, idx, nxt in ordered:
            path.append(idx)
            result = dfs(nxt, g + 1, bound, idx)
            if result is True:
                return True
            path.pop()
            if result < minimum:
                minimum = result
        return minimum

    while threshold <= max_depth:
        thresholds.append(threshold)
        seen: dict[tuple[CornerProjectedState, int | None], int] = {}

        def dfs_with_tt(cur: CornerProjectedState, g: int, bound: int, prev_move: int | None):
            nonlocal expanded
            h = heuristic(cur)
            f = g + h
            if f > bound:
                return f
            if cur.solved:
                return True
            key = (cur, prev_move)
            old_g = seen.get(key)
            if old_g is not None and old_g <= g:
                return 10**9
            seen[key] = g
            expanded += 1
            minimum = 10**9
            ordered = []
            for idx in range(len(_MOVE_NAMES)):
                if not _canonical_successor(prev_move, idx):
                    continue
                nxt = apply_move(cur, idx)
                ordered.append((heuristic(nxt), idx, nxt))
            ordered.sort(key=lambda row: (row[0], _MOVE_NAMES[row[1]]))
            for _, idx, nxt in ordered:
                path.append(idx)
                result = dfs_with_tt(nxt, g + 1, bound, idx)
                if result is True:
                    return True
                path.pop()
                if result < minimum:
                    minimum = result
            return minimum

        result = dfs_with_tt(state, 0, threshold, None)
        if result is True:
            return CornerOptimalResult(len(path), tuple(_MOVE_NAMES[i] for i in path), expanded, tuple(thresholds))
        if result == 10**9:
            return None
        threshold = int(result)
    return None


__all__ = [
    'CornerProjectedState', 'CornerOptimalResult', 'move_tables', 'apply_move',
    'corner_permutation_pdb', 'corner_orientation_pdb', 'heuristic',
    'solve_corner_optimal_ftm',
]
