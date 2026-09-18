from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Sequence

from .cubie_effect import CubieEffect
from .dense_projection_search_v64 import (
    DenseMoveEvaluation,
    DensePairPolicy,
    DenseProjectionRepository,
    Pair,
)
from .piece_route_graph import FACE_MOVES, inverse_sequence
from .transformations import from_sequence

AdaptiveSearchMode = Literal[
    "plain",
    "pair_prune",
    "vote_order",
    "adaptive_vote",
    "adaptive_child",
    "adaptive_promote",
    "threshold_adaptive",
]


@dataclass(frozen=True, slots=True)
class VoteOrderDecision:
    move_indices: tuple[int, ...]
    discriminative: bool
    top_shortest_votes: int
    shortest_vote_spread: int
    slack_vote_spread: int


@dataclass(frozen=True, slots=True)
class AdaptiveProjectionSearchResult:
    solved: bool
    sequence: tuple[str, ...]
    shortest_depth: int | None
    mode: AdaptiveSearchMode
    pair_count: int
    portfolio: tuple[Pair, ...]
    expanded: int
    generated: int
    pruned_pair: int
    thresholds: tuple[int, ...]
    suffix_states: int
    ordering_applied: int
    ordering_bypassed_depth: int
    ordering_bypassed_weak: int
    vote_cache_hits: int
    vote_cache_misses: int
    truncated: bool


class AdaptiveProjectionCompleteSolverV65:
    """Complete exact prefix search with threshold-adaptive projected guidance.

    Shallow thresholds use cheap precomputed route masks.  Deeper thresholds
    may afford a full projected child-LB comparison at only the top prefix.
    Neither policy drops a legal move or scores partial full-cube exact
    agreement.
    """

    def __init__(
        self,
        repository: DenseProjectionRepository | None = None,
        *,
        suffix_depth: int = 4,
    ) -> None:
        if suffix_depth < 0:
            raise ValueError("suffix_depth must be non-negative")
        self.repository = repository or DenseProjectionRepository()
        self.suffix_depth = suffix_depth
        self.identity = CubieEffect.identity()
        self.move_effects = {
            move: CubieEffect.from_transformation(from_sequence((move,)))
            for move in FACE_MOVES
        }
        self.suffix_table = self._build_suffix_table()

    def _build_suffix_table(self) -> dict[tuple, tuple[str, ...]]:
        table = {self.identity.signature: ()}
        queue = deque([(self.identity, (), "")])
        while queue:
            effect, path, last_face = queue.popleft()
            if len(path) == self.suffix_depth:
                continue
            for move in FACE_MOVES:
                if last_face and move[0] == last_face:
                    continue
                nxt = self.move_effects[move].compose_after(effect)
                if nxt.signature in table:
                    continue
                next_path = path + (move,)
                table[nxt.signature] = next_path
                queue.append((nxt, next_path, move[0]))
        return table

    @staticmethod
    def _legal_move_indices(last_face: str) -> tuple[int, ...]:
        return tuple(
            index
            for index, move in enumerate(FACE_MOVES)
            if not last_face or move[0] != last_face
        )

    @staticmethod
    def _promote_best_children(
        legal: Sequence[int],
        child_rows: Sequence[DenseMoveEvaluation],
        *,
        best_lower_bound: int,
        promotion_limit: int,
    ) -> tuple[int, ...]:
        """Promote only a bounded best-LB prefix and preserve the base order."""
        promoted = tuple(
            row.move_index
            for row in child_rows
            if row.lower_bound_after == best_lower_bound
        )[:promotion_limit]
        promoted_set = set(promoted)
        return promoted + tuple(
            move_index for move_index in legal if move_index not in promoted_set
        )

    @staticmethod
    def _vote_cache_key(
        piece_codes: Sequence[int],
        portfolio: Sequence[DensePairPolicy],
        last_face: str,
    ) -> tuple:
        return (
            tuple(policy.state_index(piece_codes) for policy in portfolio),
            last_face,
        )

    def vote_order(
        self,
        piece_codes: Sequence[int],
        portfolio: Sequence[DensePairPolicy],
        *,
        last_face: str,
    ) -> VoteOrderDecision:
        legal = self._legal_move_indices(last_face)
        shortest_votes = [0] * len(FACE_MOVES)
        slack_votes = [0] * len(FACE_MOVES)
        for policy in portfolio:
            state_index = policy.state_index(piece_codes)
            shortest_mask = policy.shortest_masks[state_index]
            slack_mask = policy.slack_one_masks[state_index]
            for move_index in legal:
                bit = 1 << move_index
                shortest_votes[move_index] += int(bool(shortest_mask & bit))
                slack_votes[move_index] += int(bool(slack_mask & bit))

        ranked = tuple(
            sorted(
                legal,
                key=lambda move_index: (
                    -shortest_votes[move_index],
                    -slack_votes[move_index],
                    move_index,
                ),
            )
        )
        shortest_values = [shortest_votes[index] for index in legal]
        slack_values = [slack_votes[index] for index in legal]
        shortest_spread = max(shortest_values) - min(shortest_values)
        slack_spread = max(slack_values) - min(slack_values)
        return VoteOrderDecision(
            move_indices=ranked,
            discriminative=(shortest_spread > 0 or slack_spread > 0),
            top_shortest_votes=max(shortest_values),
            shortest_vote_spread=shortest_spread,
            slack_vote_spread=slack_spread,
        )

    def _verify(self, start: CubieEffect, sequence: Sequence[str]) -> bool:
        action = CubieEffect.from_transformation(from_sequence(tuple(sequence)))
        return action.compose_after(start) == self.identity

    def solve(
        self,
        scramble: Sequence[str],
        *,
        bound: int,
        mode: AdaptiveSearchMode = "threshold_adaptive",
        pair_limit: int = 8,
        ordering_prefix_depth: int = 2,
        min_shortest_votes: int = 1,
        min_vote_spread: int = 1,
        min_child_lb_spread: int = 1,
        max_best_child_fraction: float = 0.5,
        promotion_limit: int = 3,
        child_order_min_threshold: int = 9,
        max_nodes: int = 2_000_000,
    ) -> AdaptiveProjectionSearchResult:
        if bound < 0:
            raise ValueError("bound must be non-negative")
        if ordering_prefix_depth < 0:
            raise ValueError("ordering_prefix_depth must be non-negative")
        if min_shortest_votes < 0 or min_vote_spread < 0 or min_child_lb_spread < 0:
            raise ValueError("ordering thresholds must be non-negative")
        if not 0 < max_best_child_fraction <= 1:
            raise ValueError("max_best_child_fraction must be in (0, 1]")
        if promotion_limit < 1:
            raise ValueError("promotion_limit must be positive")
        if child_order_min_threshold < 0:
            raise ValueError("child_order_min_threshold must be non-negative")
        if max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        if mode not in (
            "plain",
            "pair_prune",
            "vote_order",
            "adaptive_vote",
            "adaptive_child",
            "adaptive_promote",
            "threshold_adaptive",
        ):
            raise ValueError(f"Unsupported search mode: {mode}")

        start = CubieEffect.from_transformation(from_sequence(tuple(scramble)))
        start_codes = self.repository.encode_effect(start)
        portfolio = (
            ()
            if mode == "plain"
            else self.repository.select_portfolio(start_codes, limit=pair_limit)
        )
        root_lower_bound = self.repository.lower_bound(start_codes, portfolio)
        first_threshold = root_lower_bound if portfolio else 0
        expanded = 0
        generated = 0
        pruned_pair = 0
        ordering_applied = 0
        ordering_bypassed_depth = 0
        ordering_bypassed_weak = 0
        vote_cache_hits = 0
        vote_cache_misses = 0
        truncated = False
        solved = False
        answer: tuple[str, ...] = ()
        attempted: list[int] = []

        for threshold in range(first_threshold, bound + 1):
            attempted.append(threshold)
            forward_limit = max(0, threshold - self.suffix_depth)
            seen: dict[tuple[tuple, str], int] = {}
            vote_cache: dict[tuple, VoteOrderDecision] = {}
            path: list[str] = []

            def dfs(
                effect: CubieEffect,
                piece_codes: tuple[int, ...],
                last_face: str,
            ) -> bool:
                nonlocal expanded, generated, pruned_pair
                nonlocal ordering_applied, ordering_bypassed_depth
                nonlocal ordering_bypassed_weak, vote_cache_hits, vote_cache_misses
                nonlocal truncated, solved, answer
                if expanded >= max_nodes:
                    truncated = True
                    return False

                suffix = self.suffix_table.get(effect.signature)
                if suffix is not None and len(path) + len(suffix) <= threshold:
                    candidate = tuple(path) + inverse_sequence(suffix)
                    if self._verify(start, candidate):
                        solved = True
                        answer = candidate
                        return True

                if len(path) == forward_limit:
                    return False

                remaining = threshold - len(path)
                if portfolio:
                    lower_bound = self.repository.lower_bound(piece_codes, portfolio)
                    if lower_bound > remaining:
                        pruned_pair += 1
                        return False

                transposition_key = (effect.signature, last_face)
                previous = seen.get(transposition_key)
                forward_remaining = forward_limit - len(path)
                if previous is not None and previous >= forward_remaining:
                    return False
                seen[transposition_key] = forward_remaining
                expanded += 1

                legal = self._legal_move_indices(last_face)
                move_indices = legal
                precomputed_codes: dict[int, tuple[int, ...]] = {}
                use_child_ordering = mode in (
                    "adaptive_child",
                    "adaptive_promote",
                ) or (
                    mode == "threshold_adaptive"
                    and threshold >= child_order_min_threshold
                )
                if use_child_ordering:
                    if len(path) >= ordering_prefix_depth:
                        ordering_bypassed_depth += 1
                    else:
                        child_rows = self.repository.rank_moves(
                            piece_codes,
                            portfolio,
                            last_face=last_face,
                        )
                        child_lb_spread = (
                            max(row.lower_bound_after for row in child_rows)
                            - min(row.lower_bound_after for row in child_rows)
                        )
                        best_lower_bound = min(
                            row.lower_bound_after for row in child_rows
                        )
                        best_child_count = sum(
                            row.lower_bound_after == best_lower_bound
                            for row in child_rows
                        )
                        concentrated = (
                            best_child_count
                            <= len(child_rows) * max_best_child_fraction
                        )
                        if child_lb_spread >= min_child_lb_spread and concentrated:
                            ordering_applied += 1
                            if mode in ("adaptive_child", "threshold_adaptive"):
                                move_indices = tuple(
                                    row.move_index for row in child_rows
                                )
                            else:
                                move_indices = self._promote_best_children(
                                    legal,
                                    child_rows,
                                    best_lower_bound=best_lower_bound,
                                    promotion_limit=promotion_limit,
                                )
                            precomputed_codes = {
                                row.move_index: row.next_piece_codes for row in child_rows
                            }
                        else:
                            ordering_bypassed_weak += 1
                use_adaptive_vote_ordering = mode == "adaptive_vote" or (
                    mode == "threshold_adaptive" and not use_child_ordering
                )
                should_evaluate_votes = mode == "vote_order" or (
                    use_adaptive_vote_ordering
                    and len(path) < ordering_prefix_depth
                )
                if use_adaptive_vote_ordering and not should_evaluate_votes:
                    ordering_bypassed_depth += 1
                if should_evaluate_votes:
                    cache_key = self._vote_cache_key(piece_codes, portfolio, last_face)
                    decision = vote_cache.get(cache_key)
                    if decision is None:
                        vote_cache_misses += 1
                        decision = self.vote_order(
                            piece_codes,
                            portfolio,
                            last_face=last_face,
                        )
                        vote_cache[cache_key] = decision
                    else:
                        vote_cache_hits += 1
                    strong_enough = (
                        decision.discriminative
                        and decision.top_shortest_votes >= min_shortest_votes
                        and max(
                            decision.shortest_vote_spread,
                            decision.slack_vote_spread,
                        )
                        >= min_vote_spread
                    )
                    if mode == "vote_order" or strong_enough:
                        ordering_applied += 1
                        move_indices = decision.move_indices
                    else:
                        ordering_bypassed_weak += 1

                for move_index in move_indices:
                    move = FACE_MOVES[move_index]
                    generated += 1
                    next_codes = precomputed_codes.get(move_index)
                    if next_codes is None:
                        next_codes = self.repository.advance_codes(piece_codes, move_index)
                    nxt = self.move_effects[move].compose_after(effect)
                    path.append(move)
                    if dfs(nxt, next_codes, move[0]):
                        return True
                    path.pop()
                    if truncated:
                        return False
                return False

            if dfs(start, start_codes, ""):
                break
            if truncated:
                break

        return AdaptiveProjectionSearchResult(
            solved=solved,
            sequence=answer,
            shortest_depth=len(answer) if solved else None,
            mode=mode,
            pair_count=len(portfolio),
            portfolio=tuple(policy.pieces for policy in portfolio),
            expanded=expanded,
            generated=generated,
            pruned_pair=pruned_pair,
            thresholds=tuple(attempted),
            suffix_states=len(self.suffix_table),
            ordering_applied=ordering_applied,
            ordering_bypassed_depth=ordering_bypassed_depth,
            ordering_bypassed_weak=ordering_bypassed_weak,
            vote_cache_hits=vote_cache_hits,
            vote_cache_misses=vote_cache_misses,
            truncated=truncated,
        )


@lru_cache(maxsize=1)
def shared_dense_repository_v65() -> DenseProjectionRepository:
    repository = DenseProjectionRepository()
    repository.all_policies()
    return repository


@lru_cache(maxsize=4)
def shared_adaptive_solver_v65(
    suffix_depth: int = 4,
) -> AdaptiveProjectionCompleteSolverV65:
    return AdaptiveProjectionCompleteSolverV65(
        shared_dense_repository_v65(),
        suffix_depth=suffix_depth,
    )


__all__ = [
    "AdaptiveProjectionCompleteSolverV65",
    "AdaptiveProjectionSearchResult",
    "AdaptiveSearchMode",
    "VoteOrderDecision",
    "shared_adaptive_solver_v65",
    "shared_dense_repository_v65",
]
