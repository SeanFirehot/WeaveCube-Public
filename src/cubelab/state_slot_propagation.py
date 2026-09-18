from __future__ import annotations

"""State-indexed physical slot propagation for CubeLab v101.4.2.

Every outer-face column owns one cell for each of its four corner positions
and four edge positions.  This module keeps that position-indexed condition
separate from move-count balance and from local transition validity.

The joint propagator is exact within its finite domains.  If its node cap is
reached it retains every unresolved candidate and returns ``UNKNOWN_*``.
"""

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .move_demand_balance import (
    MOVE_ORDER,
    ColumnInventory,
    MoveDemandVector,
)
from .pieces import CORNER_ORDER, EDGE_ORDER
from .transformations import PIECES


@dataclass(frozen=True, slots=True, order=True)
class StateSlotKey:
    move: str
    piece_type: str
    state_before_position: str


@dataclass(frozen=True, slots=True)
class StateSlotDemandVector:
    counts: Mapping[StateSlotKey, int]

    def row(self) -> dict[str, object]:
        return {
            "counts": [
                {
                    **asdict(key),
                    "count": int(value),
                }
                for key, value in sorted(self.counts.items())
                if value
            ]
        }


@dataclass(frozen=True, slots=True)
class SlotBoundTrace:
    key: StateSlotKey
    current_demand: int
    minimum_future_demand: int
    maximum_future_demand: int
    reachable_future_counts: tuple[int, ...]
    exact_required_target: int

    def row(self) -> dict[str, object]:
        payload = asdict(self)
        payload["key"] = asdict(self.key)
        payload["reachable_future_counts"] = list(
            self.reachable_future_counts
        )
        return payload


@dataclass(frozen=True, slots=True)
class StateSlotCoverResult:
    status: str
    exact_cover: bool
    demand: StateSlotDemandVector
    target: StateSlotDemandVector
    failure_reasons: tuple[str, ...]

    def row(self) -> dict[str, object]:
        return {
            "status": self.status,
            "exact_cover": self.exact_cover,
            "demand": self.demand.row(),
            "target": self.target.row(),
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True, slots=True)
class StateSlotPropagationResult:
    status: str
    bound_traces: tuple[SlotBoundTrace, ...]
    surviving_candidate_ids: Mapping[str, tuple[str, ...]]
    pruned_candidate_ids: Mapping[str, tuple[str, ...]]
    explored_nodes: int
    cap_hit: bool
    joint_completion_exists: bool | None
    failure_reasons: tuple[str, ...]

    def row(self) -> dict[str, object]:
        return {
            "status": self.status,
            "bound_traces": [trace.row() for trace in self.bound_traces],
            "surviving_candidate_ids": {
                piece: list(values)
                for piece, values in self.surviving_candidate_ids.items()
            },
            "pruned_candidate_ids": {
                piece: list(values)
                for piece, values in self.pruned_candidate_ids.items()
            },
            "explored_nodes": self.explored_nodes,
            "cap_hit": self.cap_hit,
            "joint_completion_exists": self.joint_completion_exists,
            "failure_reasons": list(self.failure_reasons),
        }


def candidate_state_slot_vector(
    candidate: MoveDemandVector,
) -> StateSlotDemandVector:
    counts: Counter[StateSlotKey] = Counter()
    for event in candidate.active_events:
        counts[
            StateSlotKey(
                move=event.move,
                piece_type=event.piece_type,
                state_before_position=PIECES[event.state_before[0]],
            )
        ] += 1
    return StateSlotDemandVector(dict(counts))


def aggregate_state_slot_demand(
    candidates: Sequence[MoveDemandVector],
) -> StateSlotDemandVector:
    counts: Counter[StateSlotKey] = Counter()
    for candidate in candidates:
        counts.update(candidate_state_slot_vector(candidate).counts)
    return StateSlotDemandVector(dict(counts))


def inventory_state_slot_target(
    inventory: ColumnInventory,
) -> StateSlotDemandVector:
    counts: dict[StateSlotKey, int] = {}
    for move, multiplicity in inventory.move_counts.items():
        face = move[0]
        for piece_type, positions in (
            ("CORNER", CORNER_ORDER),
            ("EDGE", EDGE_ORDER),
        ):
            for position in positions:
                if face in position:
                    counts[
                        StateSlotKey(move, piece_type, position)
                    ] = int(multiplicity)
    return StateSlotDemandVector(counts)


def evaluate_state_slot_cover(
    candidates: Sequence[MoveDemandVector],
    inventory: ColumnInventory,
) -> StateSlotCoverResult:
    demand = aggregate_state_slot_demand(candidates)
    target = inventory_state_slot_target(inventory)
    failures = []
    for key in sorted(set(demand.counts) | set(target.counts)):
        actual = int(demand.counts.get(key, 0))
        required = int(target.counts.get(key, 0))
        if actual != required:
            failures.append(
                "STATE_SLOT:"
                f"{key.move}:{key.piece_type}:{key.state_before_position}:"
                f"{actual}!={required}"
            )
    return StateSlotCoverResult(
        status=(
            "SAT_STATE_SLOT_COVER"
            if not failures
            else "UNSAT_STATE_SLOT_COVER"
        ),
        exact_cover=not failures,
        demand=demand,
        target=target,
        failure_reasons=tuple(failures),
    )


def _slot_counter(candidate: MoveDemandVector) -> Counter[StateSlotKey]:
    return Counter(candidate_state_slot_vector(candidate).counts)


def propagate_state_slot_requirements(
    *,
    selected: Sequence[MoveDemandVector],
    remaining_domains: Mapping[str, Sequence[MoveDemandVector]],
    inventory: ColumnInventory,
    node_cap: int = 200_000,
    compute_candidate_support: bool = True,
) -> StateSlotPropagationResult:
    """Propagate exact physical slot demand through finite piece domains."""

    if node_cap < 1:
        raise ValueError("node_cap must be positive")
    target = Counter(inventory_state_slot_target(inventory).counts)
    current = Counter(aggregate_state_slot_demand(selected).counts)
    ordered = tuple(
        (piece, tuple(domain))
        for piece, domain in sorted(
            remaining_domains.items(),
            key=lambda item: (len(item[1]), PIECES.index(item[0])),
        )
    )
    all_keys = tuple(sorted(set(target) | set(current)))
    for _, domain in ordered:
        for candidate in domain:
            all_keys = tuple(
                sorted(set(all_keys) | set(_slot_counter(candidate)))
            )

    failures: list[str] = []
    for key in all_keys:
        if current[key] > target[key]:
            failures.append(
                "CURRENT_EXCEEDS_TARGET:"
                f"{key.move}:{key.piece_type}:{key.state_before_position}"
            )

    traces = []
    for key in all_keys:
        per_piece_values = []
        for _, domain in ordered:
            values = {
                _slot_counter(candidate)[key]
                for candidate in domain
            }
            if not values:
                values = {0}
            per_piece_values.append(values)
        reachable = {0}
        for values in per_piece_values:
            reachable = {
                left + right
                for left in reachable
                for right in values
                if left + right <= target[key]
            }
        minimum = sum(min(values) for values in per_piece_values)
        maximum = sum(max(values) for values in per_piece_values)
        needed = target[key] - current[key]
        traces.append(
            SlotBoundTrace(
                key=key,
                current_demand=current[key],
                minimum_future_demand=minimum,
                maximum_future_demand=maximum,
                reachable_future_counts=tuple(sorted(reachable)),
                exact_required_target=target[key],
            )
        )
        if needed < minimum or needed > maximum:
            failures.append(
                "SCALAR_BOUND_UNREACHABLE:"
                f"{key.move}:{key.piece_type}:{key.state_before_position}"
            )
        elif needed not in reachable:
            failures.append(
                "SCALAR_RESIDUE_UNREACHABLE:"
                f"{key.move}:{key.piece_type}:{key.state_before_position}"
            )

    empty = [piece for piece, domain in ordered if not domain]
    if empty:
        failures.extend(f"EMPTY_DOMAIN:{piece}" for piece in empty)

    domain_ids = {
        piece: tuple(candidate.candidate_id for candidate in domain)
        for piece, domain in ordered
    }
    if failures:
        return StateSlotPropagationResult(
            status="UNSAT_STATE_SLOT_BOUNDS",
            bound_traces=tuple(traces),
            surviving_candidate_ids={piece: () for piece, _ in ordered},
            pruned_candidate_ids=domain_ids,
            explored_nodes=0,
            cap_hit=False,
            joint_completion_exists=False,
            failure_reasons=tuple(failures),
        )

    keys = tuple(sorted(target))
    start_remaining = tuple(target[key] - current[key] for key in keys)
    candidate_counts = {
        candidate.candidate_id: tuple(
            _slot_counter(candidate)[key] for key in keys
        )
        for _, domain in ordered
        for candidate in domain
    }
    explored = 0
    cap_hit = False

    def exists(
        forced_piece: str | None = None,
        forced_candidate_id: str | None = None,
    ) -> bool | None:
        nonlocal explored, cap_hit
        memo: dict[tuple[int, tuple[int, ...]], bool] = {}

        def visit(depth: int, remaining: tuple[int, ...]) -> bool | None:
            nonlocal explored, cap_hit
            explored += 1
            if explored > node_cap:
                cap_hit = True
                return None
            if depth == len(ordered):
                return not any(remaining)
            memo_key = (depth, remaining)
            if memo_key in memo:
                return memo[memo_key]
            piece, domain = ordered[depth]
            choices = (
                tuple(
                    candidate
                    for candidate in domain
                    if candidate.candidate_id == forced_candidate_id
                )
                if piece == forced_piece
                else domain
            )
            for candidate in choices:
                counts = candidate_counts[candidate.candidate_id]
                if any(
                    counts[index] > remaining[index]
                    for index in range(len(keys))
                ):
                    continue
                next_remaining = tuple(
                    remaining[index] - counts[index]
                    for index in range(len(keys))
                )
                result = visit(depth + 1, next_remaining)
                if result is None:
                    return None
                if result:
                    memo[memo_key] = True
                    return True
            memo[memo_key] = False
            return False

        return visit(0, start_remaining)

    overall = exists()
    if overall is None:
        return StateSlotPropagationResult(
            status="UNKNOWN_STATE_SLOT_CAP",
            bound_traces=tuple(traces),
            surviving_candidate_ids=domain_ids,
            pruned_candidate_ids={piece: () for piece, _ in ordered},
            explored_nodes=explored,
            cap_hit=True,
            joint_completion_exists=None,
            failure_reasons=("JOINT_SUPPORT_NODE_CAP",),
        )
    if not overall:
        return StateSlotPropagationResult(
            status="UNSAT_STATE_SLOT_JOINT_SUPPORT",
            bound_traces=tuple(traces),
            surviving_candidate_ids={piece: () for piece, _ in ordered},
            pruned_candidate_ids=domain_ids,
            explored_nodes=explored,
            cap_hit=False,
            joint_completion_exists=False,
            failure_reasons=("NO_JOINT_SLOT_COMPLETION",),
        )

    surviving: dict[str, tuple[str, ...]] = {}
    pruned: dict[str, tuple[str, ...]] = {}
    if compute_candidate_support:
        for piece, domain in ordered:
            keep = []
            drop = []
            for candidate in domain:
                result = exists(piece, candidate.candidate_id)
                if result is None:
                    keep.append(candidate.candidate_id)
                elif result:
                    keep.append(candidate.candidate_id)
                else:
                    drop.append(candidate.candidate_id)
            surviving[piece] = tuple(keep)
            pruned[piece] = tuple(drop)
    else:
        surviving = domain_ids
        pruned = {piece: () for piece, _ in ordered}

    return StateSlotPropagationResult(
        status=(
            "UNKNOWN_STATE_SLOT_CAP"
            if cap_hit
            else "SAT_STATE_SLOT_SUPPORT"
        ),
        bound_traces=tuple(traces),
        surviving_candidate_ids=surviving,
        pruned_candidate_ids=pruned,
        explored_nodes=explored,
        cap_hit=cap_hit,
        joint_completion_exists=True,
        failure_reasons=(
            ("CANDIDATE_SUPPORT_NODE_CAP",) if cap_hit else ()
        ),
    )


__all__ = [
    "StateSlotKey",
    "StateSlotDemandVector",
    "SlotBoundTrace",
    "StateSlotCoverResult",
    "StateSlotPropagationResult",
    "candidate_state_slot_vector",
    "aggregate_state_slot_demand",
    "inventory_state_slot_target",
    "evaluate_state_slot_cover",
    "propagate_state_slot_requirements",
]
