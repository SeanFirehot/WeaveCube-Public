"""Exact deterministic CubeLab IDA/IDDFS reference kernel."""

from __future__ import annotations

from dataclasses import dataclass, field

from .canonical_masks import allowed_moves
from .coord_state import CoordinateState
from .heuristic_provider import HeuristicProvider


@dataclass(slots=True)
class SearchStats:
    nodes_by_target: dict[int, int] = field(default_factory=dict)
    prunes_by_target: dict[int, int] = field(default_factory=dict)
    generated_by_target: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResult:
    solved: bool
    distance: int | None
    word: tuple[str, ...] | None
    stats: SearchStats


def solve_optimal(
    root: CoordinateState,
    heuristic: HeuristicProvider,
    *,
    maximum_depth: int,
) -> SearchResult:
    stats = SearchStats()

    def visit(
        state: CoordinateState,
        target: int,
        path: list[str],
        last2: str | None,
        last1: str | None,
    ) -> tuple[str, ...] | None:
        stats.nodes_by_target[target] = stats.nodes_by_target.get(target, 0) + 1
        g = len(path)
        lower = heuristic.estimate(state)
        if g + lower > target:
            stats.prunes_by_target[target] = stats.prunes_by_target.get(target, 0) + 1
            return None
        if state.is_solved():
            return tuple(path)
        if g == target:
            return None
        remaining = target - g
        forbidden = heuristic.forbidden_moves(state, remaining)
        for move in allowed_moves(last2, last1):
            if move in forbidden:
                continue
            stats.generated_by_target[target] = stats.generated_by_target.get(target, 0) + 1
            path.append(move)
            result = visit(state.move(move), target, path, last1, move)
            path.pop()
            if result is not None:
                return result
        return None

    first = heuristic.estimate(root)
    for target in range(first, maximum_depth + 1):
        result = visit(root, target, [], None, None)
        if result is not None:
            return SearchResult(True, len(result), result, stats)
    return SearchResult(False, None, None, stats)
