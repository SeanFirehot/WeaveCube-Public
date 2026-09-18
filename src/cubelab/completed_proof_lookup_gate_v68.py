from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, Sequence

from .adaptive_projection_search_v65 import VoteOrderDecision
from .cubie_effect import CubieEffect
from .dense_projection_search_v64 import Pair
from .piece_route_graph import FACE_MOVES, inverse_sequence
from .resumable_dual_order_search_v66 import (
    LaneName,
    ResumableDualOrderCompleteSolverV66,
    WinnerName,
    shared_resumable_solver_v66,
)
from .transformations import from_sequence

_TranspositionKey = tuple[tuple, str]
_SeenRecord = int
_BRANCH_BITS = 5
_BRANCH_MASK = (1 << _BRANCH_BITS) - 1
_LOOKUP_BUCKETS = 32
_FACES = ("R", "U", "F", "L", "D", "B")
_FACE_SLOT = {face: index for index, face in enumerate(_FACES)}


@dataclass(frozen=True, slots=True)
class ProofLookupRemainingMetrics:
    forward_remaining: int
    eligible: int
    gated: int
    attempted: int
    key_present: int
    completed_record: int
    pruned: int


@dataclass(frozen=True, slots=True)
class CompletedProofLookupLaneMetrics:
    name: LaneName
    processed: int
    expanded: int
    generated: int
    pruned_pair: int
    claimed_root_branches: int
    completed_root_branches: int
    ordering_applied: int
    ordering_bypassed_depth: int
    ordering_bypassed_weak: int
    vote_cache_hits: int
    vote_cache_misses: int
    completed_proofs_available: int
    pruned_completed_cross_lane: int
    lookup_by_remaining: tuple[ProofLookupRemainingMetrics, ...]


@dataclass(frozen=True, slots=True)
class CompletedProofLookupGateResult:
    solved: bool
    sequence: tuple[str, ...]
    shortest_depth: int | None
    mode: Literal["completed_proof_lookup_gate"]
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
    lane_quantum: int
    dual_lane_min_threshold: int
    min_shared_forward_remaining: int
    min_lookup_forward_remaining: int
    max_lookup_forward_remaining: int | None
    unified_exact_index: bool
    sharing_enabled: bool
    lookup_instrumented: bool
    lookup_by_remaining: tuple[ProofLookupRemainingMetrics, ...]
    winning_lane: WinnerName | None
    natural: CompletedProofLookupLaneMetrics
    adaptive: CompletedProofLookupLaneMetrics
    claimed_root_branches: tuple[tuple[int, LaneName, str], ...]
    truncated: bool


@dataclass(slots=True)
class _SearchNode:
    effect: CubieEffect
    piece_codes: tuple[int, ...]
    last_face: str
    path: tuple[str, ...]


def _new_lookup_buckets() -> list[list[int]]:
    # eligible, gated, attempted, key present, completed record, pruned
    return [[0, 0, 0, 0, 0, 0] for _ in range(_LOOKUP_BUCKETS)]


def _new_seen_by_face() -> tuple[dict[bytes, _SeenRecord], ...]:
    return tuple({} for _face in _FACES)


def _freeze_lookup_buckets(
    buckets: list[list[int]],
) -> tuple[ProofLookupRemainingMetrics, ...]:
    return tuple(
        ProofLookupRemainingMetrics(remaining, *counts)
        for remaining, counts in enumerate(buckets)
        if any(counts)
    )


@dataclass(slots=True)
class _LookupCounters:
    name: LaneName
    processed: int = 0
    expanded: int = 0
    generated: int = 0
    pruned_pair: int = 0
    claimed_root_branches: int = 0
    completed_root_branches: int = 0
    ordering_applied: int = 0
    ordering_bypassed_depth: int = 0
    ordering_bypassed_weak: int = 0
    vote_cache_hits: int = 0
    vote_cache_misses: int = 0
    completed_proofs_available: int = 0
    pruned_completed_cross_lane: int = 0
    lookup_buckets: list[list[int]] = field(default_factory=_new_lookup_buckets)

    def freeze(self) -> CompletedProofLookupLaneMetrics:
        return CompletedProofLookupLaneMetrics(
            name=self.name,
            processed=self.processed,
            expanded=self.expanded,
            generated=self.generated,
            pruned_pair=self.pruned_pair,
            claimed_root_branches=self.claimed_root_branches,
            completed_root_branches=self.completed_root_branches,
            ordering_applied=self.ordering_applied,
            ordering_bypassed_depth=self.ordering_bypassed_depth,
            ordering_bypassed_weak=self.ordering_bypassed_weak,
            vote_cache_hits=self.vote_cache_hits,
            vote_cache_misses=self.vote_cache_misses,
            completed_proofs_available=self.completed_proofs_available,
            pruned_completed_cross_lane=self.pruned_completed_cross_lane,
            lookup_by_remaining=_freeze_lookup_buckets(self.lookup_buckets),
        )


@dataclass(slots=True)
class _LookupLane:
    counters: _LookupCounters
    root_order: tuple[int, ...]
    root_cursor: int = 0
    stack: list[_SearchNode] = field(default_factory=list)
    seen_by_face: tuple[dict[bytes, _SeenRecord], ...] = field(
        default_factory=_new_seen_by_face
    )
    vote_cache: dict[tuple, VoteOrderDecision] = field(default_factory=dict)
    branch_active: bool = False
    next_branch_serial: int = 0
    active_branch_serial: int = 0
    completed_branch_serial: int = 0


class CompletedProofLookupGateSolverV68(ResumableDualOrderCompleteSolverV66):
    """v67 proof sharing with an incremental exact key and safe lookup gate.

    The exact key reuses the already-incremental 20-piece code tuple instead of
    hashing the wider CubieEffect signature.  The default retains v67's simple
    lane-local dictionaries, partitioned into six last-face buckets so no
    wrapper key tuple is allocated.  It consults the other lane only for the
    measured productive remaining budgets 1 and 2.  A gated proof check falls
    back to complete local search and never removes a legal move.  A unified
    packed index remains available for ablation, but is not the production
    default.  Instrumentation is disabled for timing.
    """

    @staticmethod
    def _proof_key(node: _SearchNode) -> _TranspositionKey:
        return node.piece_codes, node.last_face

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
        unified_exact_index: bool = False,
        instrument_lookups: bool = False,
        max_nodes: int = 2_000_000,
    ) -> CompletedProofLookupGateResult:
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

        start = CubieEffect.from_transformation(from_sequence(tuple(scramble)))
        start_codes = self.repository.encode_effect(start)
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
            natural_lane = _LookupLane(natural_counters, natural_order)
            adaptive_lane = _LookupLane(adaptive_counters, adaptive_order)
            lanes = (
                (adaptive_lane, natural_lane)
                if threshold >= dual_lane_min_threshold
                else (adaptive_lane,)
            )
            sharing_active = share_completed and len(lanes) > 1
            unified_active = sharing_active and unified_exact_index
            shared_slot_bits = max(
                _BRANCH_BITS + 1,
                forward_limit.bit_length() + _BRANCH_BITS,
            )
            shared_slot_mask = (1 << shared_slot_bits) - 1
            shared_seen: dict[_TranspositionKey, int] = {}
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
                    nxt = self.move_effects[move].compose_after(start)
                    lane.stack.append(
                        _SearchNode(nxt, next_codes, move[0], (move,))
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

                    state_key = bytes(node.piece_codes)
                    face_slot = _FACE_SLOT[node.last_face]
                    key = self._proof_key(node) if unified_active else None
                    packed_seen = 0
                    local_shift = 0
                    if sharing_active:
                        other = (
                            natural_lane
                            if lane.counters.name == "adaptive"
                            else adaptive_lane
                        )
                        if unified_active:
                            local_shift = (
                                0
                                if lane.counters.name == "natural"
                                else shared_slot_bits
                            )
                            other_shift = (
                                shared_slot_bits
                                if lane.counters.name == "natural"
                                else 0
                            )
                            packed_seen = shared_seen.get(key, 0)
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
                                proof = (
                                    (
                                        packed_seen >> other_shift
                                    ) & shared_slot_mask
                                    if unified_active
                                    else other.seen_by_face[face_slot].get(
                                        state_key
                                    )
                                )
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

                    previous = (
                        (packed_seen >> local_shift) & shared_slot_mask
                        if unified_active
                        else lane.seen_by_face[face_slot].get(state_key)
                    )
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
                    next_seen_record = (
                        (
                            (forward_remaining << _BRANCH_BITS)
                            | lane.active_branch_serial
                        )
                        if sharing_active
                        else forward_remaining
                    )
                    if unified_active:
                        shared_seen[key] = (
                            packed_seen
                            & ~(shared_slot_mask << local_shift)
                        ) | (next_seen_record << local_shift)
                    else:
                        lane.seen_by_face[face_slot][state_key] = next_seen_record
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
                    not lane.stack and not lane.branch_active for lane in lanes
                ):
                    break
                lane_index = (lane_index + 1) % len(lanes)

            if unified_active:
                available_counts: list[int] = []
                for lane, shift in (
                    (natural_lane, 0),
                    (adaptive_lane, shared_slot_bits),
                ):
                    available_counts.append(
                        sum(
                            bool(record)
                            and (record & _BRANCH_MASK)
                            <= lane.completed_branch_serial
                            and (record >> _BRANCH_BITS)
                            >= min_shared_forward_remaining
                            for packed in shared_seen.values()
                            if (
                                record := (
                                    packed >> shift
                                ) & shared_slot_mask
                            )
                        )
                    )
                available_by_lane = tuple(available_counts)
            elif sharing_active:
                available_by_lane = tuple(
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
            else:
                available_by_lane = (0, 0)
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
        return CompletedProofLookupGateResult(
            solved=solved,
            sequence=answer,
            shortest_depth=len(answer) if solved else None,
            mode="completed_proof_lookup_gate",
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
            suffix_states=len(self.suffix_table),
            lane_quantum=lane_quantum,
            dual_lane_min_threshold=dual_lane_min_threshold,
            min_shared_forward_remaining=min_shared_forward_remaining,
            min_lookup_forward_remaining=min_lookup_forward_remaining,
            max_lookup_forward_remaining=max_lookup_forward_remaining,
            unified_exact_index=unified_exact_index,
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
def shared_completed_proof_lookup_solver_v68(
    suffix_depth: int = 4,
) -> CompletedProofLookupGateSolverV68:
    runtime = shared_resumable_solver_v66(suffix_depth)
    return CompletedProofLookupGateSolverV68(
        suffix_depth=suffix_depth,
        runtime=runtime,
    )


__all__ = [
    "CompletedProofLookupGateResult",
    "CompletedProofLookupGateSolverV68",
    "CompletedProofLookupLaneMetrics",
    "ProofLookupRemainingMetrics",
    "shared_completed_proof_lookup_solver_v68",
]
