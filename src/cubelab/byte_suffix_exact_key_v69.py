from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import time
from typing import Literal, Sequence

from .completed_proof_lookup_gate_v68 import (
    CompletedProofLookupGateSolverV68,
    CompletedProofLookupLaneMetrics,
    ProofLookupRemainingMetrics,
    _BRANCH_BITS,
    _BRANCH_MASK,
    _FACE_SLOT,
    _LookupCounters,
    _LookupLane,
    _freeze_lookup_buckets,
    _new_lookup_buckets,
    shared_completed_proof_lookup_solver_v68,
)
from .cubie_effect import CubieEffect
from .dense_projection_search_v64 import DenseProjectionRepository, Pair
from .piece_route_graph import FACE_MOVES, inverse_sequence
from .resumable_dual_order_search_v66 import LaneName, WinnerName
from .transformations import from_sequence


@dataclass(frozen=True, slots=True)
class ByteSuffixIndexV69:
    suffix_depth: int
    table: dict[bytes, tuple[str, ...]]
    build_seconds: float


@dataclass(slots=True)
class _ByteSearchNode:
    piece_codes: tuple[int, ...]
    last_face: str
    path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ByteSuffixExactKeyResult:
    solved: bool
    sequence: tuple[str, ...]
    shortest_depth: int | None
    mode: Literal["byte_suffix_exact_key"]
    pair_count: int
    portfolio: tuple[Pair, ...]
    expanded: int
    generated: int
    pruned_pair: int
    pruned_completed_cross_lane: int
    completed_proofs_available: int
    root_expanded: int
    thresholds: tuple[int, ...]
    completed_table_sizes: tuple[tuple[int, int], ...]
    suffix_states: int
    signature_suffix_states: int
    suffix_key_mode: Literal["piece-code-bytes"]
    suffix_index_build_seconds: float
    lane_quantum: int
    dual_lane_min_threshold: int
    min_shared_forward_remaining: int
    min_lookup_forward_remaining: int
    max_lookup_forward_remaining: int | None
    sharing_enabled: bool
    lookup_instrumented: bool
    lookup_by_remaining: tuple[ProofLookupRemainingMetrics, ...]
    winning_lane: WinnerName | None
    natural: CompletedProofLookupLaneMetrics
    adaptive: CompletedProofLookupLaneMetrics
    claimed_root_branches: tuple[tuple[int, LaneName, str], ...]
    truncated: bool


def _build_byte_suffix_index(
    repository: DenseProjectionRepository,
    *,
    suffix_depth: int,
) -> ByteSuffixIndexV69:
    if suffix_depth < 0:
        raise ValueError("suffix_depth must be non-negative")
    started = time.perf_counter()
    identity_codes = repository.encode_effect(CubieEffect.identity())
    identity_key = bytes(identity_codes)
    table: dict[bytes, tuple[str, ...]] = {identity_key: ()}
    queue = deque([(identity_codes, (), "")])
    while queue:
        piece_codes, path, last_face = queue.popleft()
        if len(path) == suffix_depth:
            continue
        for move_index, move in enumerate(FACE_MOVES):
            if last_face and move[0] == last_face:
                continue
            next_codes = repository.advance_codes(piece_codes, move_index)
            state_key = bytes(next_codes)
            if state_key in table:
                continue
            next_path = path + (move,)
            table[state_key] = next_path
            queue.append((next_codes, next_path, move[0]))
    return ByteSuffixIndexV69(
        suffix_depth=suffix_depth,
        table=table,
        build_seconds=time.perf_counter() - started,
    )


@lru_cache(maxsize=4)
def shared_byte_suffix_index_v69(suffix_depth: int = 4) -> ByteSuffixIndexV69:
    runtime = shared_completed_proof_lookup_solver_v68(suffix_depth)
    return _build_byte_suffix_index(
        runtime.repository,
        suffix_depth=suffix_depth,
    )


class ByteSuffixExactKeySolverV69(CompletedProofLookupGateSolverV68):
    """v68 exact search driven only by incremental exact piece codes.

    The byte key is created once when a node is processed and is reused for the
    exact suffix join, lane-local transposition identity, and completed proof
    lookup.  Because suffix lookup no longer needs ``CubieEffect.signature``,
    search nodes also stop composing and retaining an otherwise-unused full
    ``CubieEffect``.  No partial-state similarity score or move filter is
    introduced.
    """

    def __init__(
        self,
        *,
        runtime: CompletedProofLookupGateSolverV68,
        byte_suffix_index: ByteSuffixIndexV69,
    ) -> None:
        if runtime.suffix_depth != byte_suffix_index.suffix_depth:
            raise ValueError("runtime and byte suffix depths must match")
        super().__init__(
            suffix_depth=runtime.suffix_depth,
            runtime=runtime,
        )
        self.byte_suffix_table = byte_suffix_index.table
        self.byte_suffix_index_build_seconds = byte_suffix_index.build_seconds

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
        share_completed: bool = True,
        min_shared_forward_remaining: int = 1,
        min_lookup_forward_remaining: int = 1,
        max_lookup_forward_remaining: int | None = 2,
        instrument_lookups: bool = False,
        initial_last_face: str = "",
        max_nodes: int = 2_000_000,
    ) -> ByteSuffixExactKeyResult:
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
        if min_shared_forward_remaining < 1:
            raise ValueError("min_shared_forward_remaining must be positive")
        if min_lookup_forward_remaining < 1:
            raise ValueError("min_lookup_forward_remaining must be positive")
        if (
            max_lookup_forward_remaining is not None
            and max_lookup_forward_remaining < min_lookup_forward_remaining
        ):
            raise ValueError(
                "max_lookup_forward_remaining must be at least the minimum"
            )
        if max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        if initial_last_face not in ("", "U", "D", "F", "B", "L", "R"):
            raise ValueError("initial_last_face must be empty or one cube face")

        start = CubieEffect.from_transformation(from_sequence(tuple(scramble)))
        start_codes = self.repository.encode_effect(start)
        start_key = bytes(start_codes)
        portfolio = self.repository.select_portfolio(start_codes, limit=pair_limit)
        first_threshold = self.repository.lower_bound(start_codes, portfolio)
        natural_counters = _LookupCounters("natural")
        adaptive_counters = _LookupCounters("adaptive")
        root_expanded = 0
        total_expanded = 0
        attempted: list[int] = []
        completed_table_sizes: list[tuple[int, int]] = []
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

            suffix = self.byte_suffix_table.get(start_key)
            if suffix is not None and len(suffix) <= threshold:
                candidate = inverse_sequence(suffix)
                if (
                    (not candidate or candidate[0][0] != initial_last_face)
                    and self._verify(start, candidate)
                ):
                    solved = True
                    answer = candidate
                    winning_lane = "root"
                    break

            # Counterfactual first-column audits can arrive here after a fixed
            # move.  If the sole cached suffix starts on that same face, keep
            # one forward ply available so an alternative reduced witness can
            # still be found instead of treating the cache witness as unique.
            forward_limit = max(
                1 if initial_last_face and threshold > 0 else 0,
                threshold - self.suffix_depth,
            )
            if forward_limit == 0:
                completed_table_sizes.append((threshold, 0))
                continue
            if self.repository.lower_bound(start_codes, portfolio) > threshold:
                completed_table_sizes.append((threshold, 0))
                continue

            root_expanded += 1
            total_expanded += 1
            if total_expanded > max_nodes:
                truncated = True
                break

            natural_order = self._legal_move_indices(initial_last_face)
            adaptive_order, root_precomputed_codes = self._adaptive_order(
                start_codes,
                portfolio,
                last_face=initial_last_face,
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
            natural_lane = _LookupLane(natural_counters, natural_order)
            adaptive_lane = _LookupLane(adaptive_counters, adaptive_order)
            lanes = (
                (adaptive_lane, natural_lane)
                if threshold >= dual_lane_min_threshold
                else (adaptive_lane,)
            )
            sharing_active = share_completed and len(lanes) > 1
            claimed: set[int] = set()

            def publish_branch(lane: _LookupLane) -> None:
                if not lane.branch_active:
                    return
                lane.counters.completed_root_branches += 1
                lane.completed_branch_serial = lane.active_branch_serial
                lane.branch_active = False

            def claim_root(lane: _LookupLane) -> bool:
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
                    lane.stack.append(
                        _ByteSearchNode(next_codes, move[0], (move,))
                    )
                    lane.next_branch_serial += 1
                    lane.active_branch_serial = lane.next_branch_serial
                    lane.branch_active = True
                    lane.counters.generated += 1
                    lane.counters.claimed_root_branches += 1
                    claim_trace.append((threshold, lane.counters.name, move))
                    return True
                return False

            def run_quantum(lane: _LookupLane) -> Literal[
                "yielded", "idle", "solved", "truncated"
            ]:
                nonlocal total_expanded, solved, answer, winning_lane, truncated
                expanded_at_start = lane.counters.expanded
                while lane.counters.expanded - expanded_at_start < lane_quantum:
                    if total_expanded >= max_nodes:
                        truncated = True
                        return "truncated"
                    if not lane.stack:
                        publish_branch(lane)
                        if not claim_root(lane):
                            return "idle"

                    node = lane.stack.pop()
                    lane.counters.processed += 1
                    state_key = bytes(node.piece_codes)
                    suffix_at_node = self.byte_suffix_table.get(state_key)
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

                    forward_remaining = forward_limit - len(node.path)
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

                    face_slot = _FACE_SLOT[node.last_face]
                    if sharing_active:
                        other = (
                            natural_lane
                            if lane.counters.name == "adaptive"
                            else adaptive_lane
                        )
                        if other.completed_branch_serial:
                            bucket = lane.counters.lookup_buckets[forward_remaining]
                            if instrument_lookups:
                                bucket[0] += 1
                            if (
                                forward_remaining < min_lookup_forward_remaining
                                or (
                                    max_lookup_forward_remaining is not None
                                    and forward_remaining
                                    > max_lookup_forward_remaining
                                )
                            ):
                                if instrument_lookups:
                                    bucket[1] += 1
                            else:
                                if instrument_lookups:
                                    bucket[2] += 1
                                proof = other.seen_by_face[face_slot].get(state_key)
                                if proof is not None and proof:
                                    if instrument_lookups:
                                        bucket[3] += 1
                                    if (
                                        (proof & _BRANCH_MASK)
                                        <= other.completed_branch_serial
                                    ):
                                        if instrument_lookups:
                                            bucket[4] += 1
                                        proof_remaining = proof >> _BRANCH_BITS
                                        if (
                                            proof_remaining
                                            >= min_shared_forward_remaining
                                            and proof_remaining >= forward_remaining
                                        ):
                                            if instrument_lookups:
                                                bucket[5] += 1
                                            lane.counters.pruned_completed_cross_lane += 1
                                            continue

                    previous = lane.seen_by_face[face_slot].get(state_key)
                    if (
                        previous is not None
                        and (
                            (previous >> _BRANCH_BITS)
                            if sharing_active
                            else previous
                        )
                        >= forward_remaining
                    ):
                        continue
                    lane.seen_by_face[face_slot][state_key] = (
                        (
                            (forward_remaining << _BRANCH_BITS)
                            | lane.active_branch_serial
                        )
                        if sharing_active
                        else forward_remaining
                    )
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
                        lane.stack.append(
                            _ByteSearchNode(
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
                    not lane.stack and not lane.branch_active for lane in lanes
                ):
                    break
                lane_index = (lane_index + 1) % len(lanes)

            available_by_lane = (
                tuple(
                    sum(
                        (record & _BRANCH_MASK)
                        <= lane.completed_branch_serial
                        and (record >> _BRANCH_BITS)
                        >= min_shared_forward_remaining
                        for seen in lane.seen_by_face
                        for record in seen.values()
                    )
                    for lane in (natural_lane, adaptive_lane)
                )
                if sharing_active
                else (0, 0)
            )
            natural_counters.completed_proofs_available += available_by_lane[0]
            adaptive_counters.completed_proofs_available += available_by_lane[1]
            completed_table_sizes.append((threshold, sum(available_by_lane)))
            if solved or truncated:
                break

        natural = natural_counters.freeze()
        adaptive = adaptive_counters.freeze()
        aggregate_buckets = _new_lookup_buckets()
        if instrument_lookups:
            for lane_buckets in (
                natural_counters.lookup_buckets,
                adaptive_counters.lookup_buckets,
            ):
                for remaining, counts in enumerate(lane_buckets):
                    for index, count in enumerate(counts):
                        aggregate_buckets[remaining][index] += count
        return ByteSuffixExactKeyResult(
            solved=solved,
            sequence=answer,
            shortest_depth=len(answer) if solved else None,
            mode="byte_suffix_exact_key",
            pair_count=len(portfolio),
            portfolio=tuple(policy.pieces for policy in portfolio),
            expanded=total_expanded,
            generated=natural.generated + adaptive.generated,
            pruned_pair=natural.pruned_pair + adaptive.pruned_pair,
            pruned_completed_cross_lane=(
                natural.pruned_completed_cross_lane
                + adaptive.pruned_completed_cross_lane
            ),
            completed_proofs_available=(
                natural.completed_proofs_available
                + adaptive.completed_proofs_available
            ),
            root_expanded=root_expanded,
            thresholds=tuple(attempted),
            completed_table_sizes=tuple(completed_table_sizes),
            suffix_states=len(self.byte_suffix_table),
            signature_suffix_states=len(self.suffix_table),
            suffix_key_mode="piece-code-bytes",
            suffix_index_build_seconds=self.byte_suffix_index_build_seconds,
            lane_quantum=lane_quantum,
            dual_lane_min_threshold=dual_lane_min_threshold,
            min_shared_forward_remaining=min_shared_forward_remaining,
            min_lookup_forward_remaining=min_lookup_forward_remaining,
            max_lookup_forward_remaining=max_lookup_forward_remaining,
            sharing_enabled=share_completed,
            lookup_instrumented=instrument_lookups,
            lookup_by_remaining=(
                _freeze_lookup_buckets(aggregate_buckets)
                if instrument_lookups
                else ()
            ),
            winning_lane=winning_lane,
            natural=natural,
            adaptive=adaptive,
            claimed_root_branches=tuple(claim_trace),
            truncated=truncated,
        )


@lru_cache(maxsize=4)
def shared_byte_suffix_solver_v69(
    suffix_depth: int = 4,
) -> ByteSuffixExactKeySolverV69:
    runtime = shared_completed_proof_lookup_solver_v68(suffix_depth)
    byte_suffix_index = shared_byte_suffix_index_v69(suffix_depth)
    return ByteSuffixExactKeySolverV69(
        runtime=runtime,
        byte_suffix_index=byte_suffix_index,
    )


__all__ = [
    "ByteSuffixExactKeyResult",
    "ByteSuffixExactKeySolverV69",
    "ByteSuffixIndexV69",
    "shared_byte_suffix_index_v69",
    "shared_byte_suffix_solver_v69",
]
