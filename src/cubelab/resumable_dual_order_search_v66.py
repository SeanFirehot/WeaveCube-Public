from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, Sequence

from .adaptive_projection_search_v65 import (
    AdaptiveProjectionCompleteSolverV65,
    VoteOrderDecision,
    shared_adaptive_solver_v65,
)
from .cubie_effect import CubieEffect
from .dense_projection_search_v64 import DensePairPolicy, DenseProjectionRepository, Pair
from .piece_route_graph import FACE_MOVES, inverse_sequence
from .transformations import from_sequence

LaneName = Literal["natural", "adaptive"]
WinnerName = Literal["root", "natural", "adaptive"]


@dataclass(frozen=True, slots=True)
class PortfolioLaneMetrics:
    name: LaneName
    processed: int
    expanded: int
    generated: int
    pruned_pair: int
    claimed_root_branches: int
    ordering_applied: int
    ordering_bypassed_depth: int
    ordering_bypassed_weak: int
    vote_cache_hits: int
    vote_cache_misses: int


@dataclass(frozen=True, slots=True)
class ResumableDualOrderSearchResult:
    solved: bool
    sequence: tuple[str, ...]
    shortest_depth: int | None
    mode: Literal["resumable_dual_order"]
    pair_count: int
    portfolio: tuple[Pair, ...]
    expanded: int
    generated: int
    pruned_pair: int
    root_expanded: int
    thresholds: tuple[int, ...]
    suffix_states: int
    lane_quantum: int
    dual_lane_min_threshold: int
    winning_lane: WinnerName | None
    natural: PortfolioLaneMetrics
    adaptive: PortfolioLaneMetrics
    claimed_root_branches: tuple[tuple[int, LaneName, str], ...]
    truncated: bool


@dataclass(slots=True)
class _SearchNode:
    effect: CubieEffect
    piece_codes: tuple[int, ...]
    last_face: str
    path: tuple[str, ...]


@dataclass(slots=True)
class _LaneCounters:
    name: LaneName
    processed: int = 0
    expanded: int = 0
    generated: int = 0
    pruned_pair: int = 0
    claimed_root_branches: int = 0
    ordering_applied: int = 0
    ordering_bypassed_depth: int = 0
    ordering_bypassed_weak: int = 0
    vote_cache_hits: int = 0
    vote_cache_misses: int = 0

    def freeze(self) -> PortfolioLaneMetrics:
        return PortfolioLaneMetrics(
            name=self.name,
            processed=self.processed,
            expanded=self.expanded,
            generated=self.generated,
            pruned_pair=self.pruned_pair,
            claimed_root_branches=self.claimed_root_branches,
            ordering_applied=self.ordering_applied,
            ordering_bypassed_depth=self.ordering_bypassed_depth,
            ordering_bypassed_weak=self.ordering_bypassed_weak,
            vote_cache_hits=self.vote_cache_hits,
            vote_cache_misses=self.vote_cache_misses,
        )


@dataclass(slots=True)
class _LaneState:
    counters: _LaneCounters
    root_order: tuple[int, ...]
    root_cursor: int = 0
    stack: list[_SearchNode] = field(default_factory=list)
    seen: dict[tuple[tuple, str], int] = field(default_factory=dict)
    vote_cache: dict[tuple, VoteOrderDecision] = field(default_factory=dict)


class ResumableDualOrderCompleteSolverV66(AdaptiveProjectionCompleteSolverV65):
    """Complete exact search with two resumable, non-duplicating root lanes.

    The natural and adaptive orders claim each root move at most once per
    threshold.  Each lane owns an explicit DFS stack and runs for a bounded
    expansion quantum before yielding.  Exact full-cube agreement is never a
    partial-state score.
    """

    def __init__(
        self,
        repository: DenseProjectionRepository | None = None,
        *,
        suffix_depth: int = 4,
        runtime: AdaptiveProjectionCompleteSolverV65 | None = None,
    ) -> None:
        if runtime is None:
            super().__init__(repository, suffix_depth=suffix_depth)
            return
        if repository is not None and repository is not runtime.repository:
            raise ValueError("repository and runtime.repository must be identical")
        if suffix_depth != runtime.suffix_depth:
            raise ValueError("suffix_depth must match the supplied runtime")
        self.repository = runtime.repository
        self.suffix_depth = runtime.suffix_depth
        self.identity = runtime.identity
        self.move_effects = runtime.move_effects
        self.suffix_table = runtime.suffix_table

    def _adaptive_order(
        self,
        piece_codes: tuple[int, ...],
        portfolio: Sequence[DensePairPolicy],
        *,
        last_face: str,
        path_depth: int,
        threshold: int,
        ordering_prefix_depth: int,
        child_order_min_threshold: int,
        min_shortest_votes: int,
        min_vote_spread: int,
        min_child_lb_spread: int,
        max_best_child_fraction: float,
        counters: _LaneCounters,
        vote_cache: dict[tuple, VoteOrderDecision],
    ) -> tuple[tuple[int, ...], dict[int, tuple[int, ...]]]:
        legal = self._legal_move_indices(last_face)
        if path_depth >= ordering_prefix_depth:
            counters.ordering_bypassed_depth += 1
            return legal, {}

        if threshold >= child_order_min_threshold:
            child_rows = self.repository.rank_moves(
                piece_codes,
                portfolio,
                last_face=last_face,
            )
            child_lb_spread = (
                max(row.lower_bound_after for row in child_rows)
                - min(row.lower_bound_after for row in child_rows)
            )
            best_lower_bound = min(row.lower_bound_after for row in child_rows)
            best_child_count = sum(
                row.lower_bound_after == best_lower_bound for row in child_rows
            )
            concentrated = (
                best_child_count <= len(child_rows) * max_best_child_fraction
            )
            if child_lb_spread >= min_child_lb_spread and concentrated:
                counters.ordering_applied += 1
                return (
                    tuple(row.move_index for row in child_rows),
                    {
                        row.move_index: row.next_piece_codes
                        for row in child_rows
                    },
                )
            counters.ordering_bypassed_weak += 1
            return legal, {}

        cache_key = self._vote_cache_key(piece_codes, portfolio, last_face)
        decision = vote_cache.get(cache_key)
        if decision is None:
            counters.vote_cache_misses += 1
            decision = self.vote_order(
                piece_codes,
                portfolio,
                last_face=last_face,
            )
            vote_cache[cache_key] = decision
        else:
            counters.vote_cache_hits += 1
        strong_enough = (
            decision.discriminative
            and decision.top_shortest_votes >= min_shortest_votes
            and max(decision.shortest_vote_spread, decision.slack_vote_spread)
            >= min_vote_spread
        )
        if strong_enough:
            counters.ordering_applied += 1
            return decision.move_indices, {}
        counters.ordering_bypassed_weak += 1
        return legal, {}

    def solve(
        self,
        scramble: Sequence[str],
        *,
        bound: int,
        pair_limit: int = 8,
        lane_quantum: int = 32,
        dual_lane_min_threshold: int = 8,
        ordering_prefix_depth: int = 1,
        child_order_min_threshold: int = 9,
        min_shortest_votes: int = 1,
        min_vote_spread: int = 1,
        min_child_lb_spread: int = 1,
        max_best_child_fraction: float = 0.5,
        max_nodes: int = 2_000_000,
    ) -> ResumableDualOrderSearchResult:
        if bound < 0:
            raise ValueError("bound must be non-negative")
        if lane_quantum < 1:
            raise ValueError("lane_quantum must be positive")
        if dual_lane_min_threshold < 0:
            raise ValueError("dual_lane_min_threshold must be non-negative")
        if ordering_prefix_depth < 0:
            raise ValueError("ordering_prefix_depth must be non-negative")
        if child_order_min_threshold < 0:
            raise ValueError("child_order_min_threshold must be non-negative")
        if min_shortest_votes < 0 or min_vote_spread < 0 or min_child_lb_spread < 0:
            raise ValueError("ordering thresholds must be non-negative")
        if not 0 < max_best_child_fraction <= 1:
            raise ValueError("max_best_child_fraction must be in (0, 1]")
        if max_nodes < 1:
            raise ValueError("max_nodes must be positive")

        start = CubieEffect.from_transformation(from_sequence(tuple(scramble)))
        start_codes = self.repository.encode_effect(start)
        portfolio = self.repository.select_portfolio(start_codes, limit=pair_limit)
        first_threshold = self.repository.lower_bound(start_codes, portfolio)
        natural_counters = _LaneCounters("natural")
        adaptive_counters = _LaneCounters("adaptive")
        root_expanded = 0
        total_expanded = 0
        attempted: list[int] = []
        claim_trace: list[tuple[int, LaneName, str]] = []
        solved = False
        answer: tuple[str, ...] = ()
        winning_lane: WinnerName | None = None
        truncated = False

        for threshold in range(first_threshold, bound + 1):
            attempted.append(threshold)
            if total_expanded >= max_nodes:
                truncated = True
                break

            suffix = self.suffix_table.get(start.signature)
            if suffix is not None and len(suffix) <= threshold:
                candidate = inverse_sequence(suffix)
                if self._verify(start, candidate):
                    solved = True
                    answer = candidate
                    winning_lane = "root"
                    break

            forward_limit = max(0, threshold - self.suffix_depth)
            if forward_limit == 0:
                continue
            if self.repository.lower_bound(start_codes, portfolio) > threshold:
                continue

            root_expanded += 1
            total_expanded += 1
            if total_expanded > max_nodes:
                truncated = True
                break

            natural_order = self._legal_move_indices("")
            adaptive_order, root_precomputed_codes = self._adaptive_order(
                start_codes,
                portfolio,
                last_face="",
                path_depth=0,
                threshold=threshold,
                ordering_prefix_depth=ordering_prefix_depth,
                child_order_min_threshold=child_order_min_threshold,
                min_shortest_votes=min_shortest_votes,
                min_vote_spread=min_vote_spread,
                min_child_lb_spread=min_child_lb_spread,
                max_best_child_fraction=max_best_child_fraction,
                counters=adaptive_counters,
                vote_cache={},
            )
            natural_lane = _LaneState(natural_counters, natural_order)
            adaptive_lane = _LaneState(adaptive_counters, adaptive_order)
            lanes = (
                (adaptive_lane, natural_lane)
                if threshold >= dual_lane_min_threshold
                else (adaptive_lane,)
            )
            claimed: set[int] = set()

            def claim_root(lane: _LaneState) -> bool:
                while lane.root_cursor < len(lane.root_order):
                    move_index = lane.root_order[lane.root_cursor]
                    lane.root_cursor += 1
                    if move_index in claimed:
                        continue
                    claimed.add(move_index)
                    move = FACE_MOVES[move_index]
                    next_codes = root_precomputed_codes.get(move_index)
                    if next_codes is None:
                        next_codes = self.repository.advance_codes(
                            start_codes,
                            move_index,
                        )
                    nxt = self.move_effects[move].compose_after(start)
                    lane.stack.append(
                        _SearchNode(nxt, next_codes, move[0], (move,))
                    )
                    lane.counters.generated += 1
                    lane.counters.claimed_root_branches += 1
                    claim_trace.append((threshold, lane.counters.name, move))
                    return True
                return False

            def run_quantum(lane: _LaneState) -> Literal[
                "yielded", "idle", "solved", "truncated"
            ]:
                nonlocal total_expanded, solved, answer, winning_lane, truncated
                expanded_at_start = lane.counters.expanded
                while lane.counters.expanded - expanded_at_start < lane_quantum:
                    if total_expanded >= max_nodes:
                        truncated = True
                        return "truncated"
                    if not lane.stack and not claim_root(lane):
                        return "idle"

                    node = lane.stack.pop()
                    lane.counters.processed += 1
                    suffix_at_node = self.suffix_table.get(node.effect.signature)
                    if (
                        suffix_at_node is not None
                        and len(node.path) + len(suffix_at_node) <= threshold
                    ):
                        candidate = node.path + inverse_sequence(suffix_at_node)
                        if self._verify(start, candidate):
                            solved = True
                            answer = candidate
                            winning_lane = lane.counters.name
                            return "solved"

                    if len(node.path) == forward_limit:
                        continue

                    remaining = threshold - len(node.path)
                    lower_bound = self.repository.lower_bound(
                        node.piece_codes,
                        portfolio,
                    )
                    if lower_bound > remaining:
                        lane.counters.pruned_pair += 1
                        continue

                    transposition_key = (node.effect.signature, node.last_face)
                    forward_remaining = forward_limit - len(node.path)
                    previous = lane.seen.get(transposition_key)
                    if previous is not None and previous >= forward_remaining:
                        continue
                    lane.seen[transposition_key] = forward_remaining
                    lane.counters.expanded += 1
                    total_expanded += 1

                    if lane.counters.name == "adaptive":
                        move_indices, precomputed_codes = self._adaptive_order(
                            node.piece_codes,
                            portfolio,
                            last_face=node.last_face,
                            path_depth=len(node.path),
                            threshold=threshold,
                            ordering_prefix_depth=ordering_prefix_depth,
                            child_order_min_threshold=child_order_min_threshold,
                            min_shortest_votes=min_shortest_votes,
                            min_vote_spread=min_vote_spread,
                            min_child_lb_spread=min_child_lb_spread,
                            max_best_child_fraction=max_best_child_fraction,
                            counters=lane.counters,
                            vote_cache=lane.vote_cache,
                        )
                    else:
                        move_indices = self._legal_move_indices(node.last_face)
                        precomputed_codes = {}

                    for move_index in reversed(move_indices):
                        move = FACE_MOVES[move_index]
                        next_codes = precomputed_codes.get(move_index)
                        if next_codes is None:
                            next_codes = self.repository.advance_codes(
                                node.piece_codes,
                                move_index,
                            )
                        nxt = self.move_effects[move].compose_after(node.effect)
                        lane.stack.append(
                            _SearchNode(
                                nxt,
                                next_codes,
                                move[0],
                                node.path + (move,),
                            )
                        )
                        lane.counters.generated += 1
                return "yielded"

            lane_index = 0
            while True:
                if total_expanded >= max_nodes:
                    truncated = True
                    break
                outcome = run_quantum(lanes[lane_index])
                if outcome in ("solved", "truncated"):
                    break
                if len(claimed) == len(natural_order) and all(
                    not lane.stack for lane in lanes
                ):
                    break
                lane_index = (lane_index + 1) % len(lanes)

            if solved or truncated:
                break

        natural = natural_counters.freeze()
        adaptive = adaptive_counters.freeze()
        return ResumableDualOrderSearchResult(
            solved=solved,
            sequence=answer,
            shortest_depth=len(answer) if solved else None,
            mode="resumable_dual_order",
            pair_count=len(portfolio),
            portfolio=tuple(policy.pieces for policy in portfolio),
            expanded=total_expanded,
            generated=natural.generated + adaptive.generated,
            pruned_pair=natural.pruned_pair + adaptive.pruned_pair,
            root_expanded=root_expanded,
            thresholds=tuple(attempted),
            suffix_states=len(self.suffix_table),
            lane_quantum=lane_quantum,
            dual_lane_min_threshold=dual_lane_min_threshold,
            winning_lane=winning_lane,
            natural=natural,
            adaptive=adaptive,
            claimed_root_branches=tuple(claim_trace),
            truncated=truncated,
        )


@lru_cache(maxsize=4)
def shared_resumable_solver_v66(
    suffix_depth: int = 4,
) -> ResumableDualOrderCompleteSolverV66:
    runtime = shared_adaptive_solver_v65(suffix_depth)
    return ResumableDualOrderCompleteSolverV66(
        suffix_depth=suffix_depth,
        runtime=runtime,
    )


__all__ = [
    "LaneName",
    "PortfolioLaneMetrics",
    "ResumableDualOrderCompleteSolverV66",
    "ResumableDualOrderSearchResult",
    "WinnerName",
    "shared_resumable_solver_v66",
]
