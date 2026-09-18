from __future__ import annotations

"""Exact joint state-slot DP with branch-shared suffix memoization."""

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from cubelab.move_demand_balance import ColumnInventory, MoveDemandVector
from cubelab.state_slot_propagation import (
    SlotBoundTrace,
    StateSlotKey,
    StateSlotPropagationResult,
    aggregate_state_slot_demand,
    candidate_state_slot_vector,
    inventory_state_slot_target,
)
from cubelab.transformations import PIECES

from .joint_support_state import (
    JointSupportCandidate,
    JointSupportPieceDomain,
    JointSupportProblem,
)
from .suffix_memo import ExactSuffixMemoStore


SUPPORTED = "SUPPORTED"
UNSUPPORTED_PROVEN = "UNSUPPORTED_PROVEN"
UNKNOWN_CAP = "UNKNOWN_CAP"


@dataclass(frozen=True, slots=True)
class JointSupportMemoResult:
    status: str
    witness_candidate_ids: tuple[str, ...] | None
    forced_candidate_ids: tuple[str, ...] = ()
    impossible_candidate_ids: tuple[str, ...] = ()
    explored_nodes: int = 0
    cap_hit: bool = False
    logical_nodes: int = 0
    unknown_budget: int = 0

    def row(self) -> dict[str, object]:
        payload = asdict(self)
        payload["witness_candidate_ids"] = (
            None
            if self.witness_candidate_ids is None
            else list(self.witness_candidate_ids)
        )
        payload["forced_candidate_ids"] = list(
            self.forced_candidate_ids
        )
        payload["impossible_candidate_ids"] = list(
            self.impossible_candidate_ids
        )
        return payload


@dataclass(frozen=True, slots=True)
class JointSupportSolveMetrics:
    logical_nodes: int
    expanded_nodes: int
    memo_hits: int
    memo_misses: int
    reused_logical_nodes: int
    solve_calls: int
    key_rows: tuple[dict[str, object], ...]

    def row(self) -> dict[str, object]:
        payload = asdict(self)
        payload["key_rows"] = list(self.key_rows)
        return payload


class _Budget:
    def __init__(self, node_cap: int) -> None:
        self.node_cap = node_cap
        self.logical_nodes = 0
        self.expanded_nodes = 0
        self.memo_hits = 0
        self.memo_misses = 0
        self.reused_logical_nodes = 0
        self.solve_calls = 0
        self.key_rows: list[dict[str, object]] = []

    @property
    def remaining(self) -> int:
        return max(0, self.node_cap - self.logical_nodes)

    def consume(self, amount: int = 1) -> bool:
        self.logical_nodes += amount
        return self.logical_nodes <= self.node_cap


def _ordered_domains(
    domains: Sequence[JointSupportPieceDomain],
    policy: str,
) -> tuple[JointSupportPieceDomain, ...]:
    if policy in {"fixed", "reference"}:
        return tuple(domains)
    if policy == "smallest_domain":
        return tuple(
            sorted(
                domains,
                key=lambda domain: (
                    len(domain.candidates),
                    PIECES.index(domain.piece_id),
                ),
            )
        )

    def scarcity(domain: JointSupportPieceDomain) -> tuple[object, ...]:
        supplied = Counter(
            index
            for candidate in domain.candidates
            for index, count in enumerate(candidate.counts)
            if count
        )
        rare = min(supplied.values(), default=10**9)
        nonzero = len(supplied)
        coupling = max(
            (
                sum(bool(count) for count in candidate.counts)
                for candidate in domain.candidates
            ),
            default=0,
        )
        if policy == "highest_slot_scarcity":
            return (rare, -nonzero, len(domain.candidates), PIECES.index(domain.piece_id))
        if policy == "highest_coupling":
            return (-coupling, len(domain.candidates), rare, PIECES.index(domain.piece_id))
        raise ValueError(f"unsupported joint-support ordering: {policy}")

    return tuple(sorted(domains, key=scarcity))


def _solve_problem(
    problem: JointSupportProblem,
    *,
    node_cap: int,
    memo_store: ExactSuffixMemoStore[JointSupportMemoResult] | None,
    budget: _Budget,
) -> JointSupportMemoResult:
    budget.solve_calls += 1
    local: dict[
        tuple[int, tuple[int, ...]], JointSupportMemoResult
    ] = {}

    def visit(
        depth: int,
        deficits: tuple[int, ...],
    ) -> JointSupportMemoResult:
        start = budget.logical_nodes
        if not budget.consume():
            return JointSupportMemoResult(
                status=UNKNOWN_CAP,
                witness_candidate_ids=None,
                explored_nodes=0,
                cap_hit=True,
                logical_nodes=1,
                unknown_budget=0,
            )
        if depth == len(problem.domains):
            status = SUPPORTED if not any(deficits) else UNSUPPORTED_PROVEN
            return JointSupportMemoResult(
                status=status,
                witness_candidate_ids=() if status == SUPPORTED else None,
                explored_nodes=1,
                logical_nodes=1,
            )

        local_key = (depth, deficits)
        cached_local = local.get(local_key)
        if cached_local is not None:
            return JointSupportMemoResult(
                status=cached_local.status,
                witness_candidate_ids=cached_local.witness_candidate_ids,
                explored_nodes=1,
                cap_hit=cached_local.cap_hit,
                logical_nodes=1,
                unknown_budget=cached_local.unknown_budget,
            )

        key = problem.key(depth, deficits)
        if len(budget.key_rows) < 64:
            budget.key_rows.append(key.row())
        if memo_store is not None:
            lookup = memo_store.get(
                key,
                remaining_node_budget=budget.remaining + 1,
            )
            if lookup.hit:
                budget.memo_hits += 1
                value = lookup.value
                assert value is not None
                if value.status == UNKNOWN_CAP:
                    local[local_key] = value
                    return JointSupportMemoResult(
                        status=UNKNOWN_CAP,
                        witness_candidate_ids=None,
                        explored_nodes=1,
                        cap_hit=True,
                        logical_nodes=1,
                        unknown_budget=budget.remaining + 1,
                    )
                additional = max(0, value.logical_nodes - 1)
                if additional and not budget.consume(additional):
                    return JointSupportMemoResult(
                        status=UNKNOWN_CAP,
                        witness_candidate_ids=None,
                        explored_nodes=1,
                        cap_hit=True,
                        logical_nodes=budget.logical_nodes - start,
                        unknown_budget=max(0, budget.remaining),
                    )
                budget.reused_logical_nodes += additional
                local[local_key] = value
                return JointSupportMemoResult(
                    status=value.status,
                    witness_candidate_ids=value.witness_candidate_ids,
                    explored_nodes=1,
                    logical_nodes=budget.logical_nodes - start,
                )
            budget.memo_misses += 1

        budget.expanded_nodes += 1
        domain = problem.domains[depth]
        for candidate in domain.candidates:
            if any(
                count > deficits[index]
                for index, count in enumerate(candidate.counts)
            ):
                continue
            next_deficits = tuple(
                deficits[index] - count
                for index, count in enumerate(candidate.counts)
            )
            child = visit(depth + 1, next_deficits)
            if child.status == UNKNOWN_CAP:
                result = JointSupportMemoResult(
                    status=UNKNOWN_CAP,
                    witness_candidate_ids=None,
                    explored_nodes=budget.logical_nodes - start,
                    cap_hit=True,
                    logical_nodes=budget.logical_nodes - start,
                    unknown_budget=max(0, budget.remaining),
                )
                if memo_store is not None:
                    memo_store.put(key, result)
                return result
            if child.status == SUPPORTED:
                result = JointSupportMemoResult(
                    status=SUPPORTED,
                    witness_candidate_ids=(
                        candidate.candidate_id,
                    )
                    + tuple(child.witness_candidate_ids or ()),
                    explored_nodes=budget.logical_nodes - start,
                    logical_nodes=budget.logical_nodes - start,
                )
                local[local_key] = result
                if memo_store is not None:
                    memo_store.put(key, result)
                return result

        result = JointSupportMemoResult(
            status=UNSUPPORTED_PROVEN,
            witness_candidate_ids=None,
            explored_nodes=budget.logical_nodes - start,
            logical_nodes=budget.logical_nodes - start,
        )
        local[local_key] = result
        if memo_store is not None:
            memo_store.put(key, result)
        return result

    return visit(0, problem.initial_deficits)


def _candidate_counts(
    candidate: MoveDemandVector,
    keys: tuple[StateSlotKey, ...],
) -> tuple[int, ...]:
    counts = candidate_state_slot_vector(candidate).counts
    return tuple(int(counts.get(key, 0)) for key in keys)


def _problem(
    *,
    ordered: Sequence[tuple[str, Sequence[MoveDemandVector]]],
    keys: tuple[StateSlotKey, ...],
    deficits: tuple[int, ...],
    selected_signature: tuple[int, ...],
    ordering_policy: str,
    contract_signature: str,
    extension_signature: str,
) -> JointSupportProblem:
    raw = tuple(
        JointSupportPieceDomain(
            piece_id=piece,
            candidates=tuple(
                JointSupportCandidate(
                    candidate_id=candidate.candidate_id,
                    counts=_candidate_counts(candidate, keys),
                )
                for candidate in domain
            ),
        )
        for piece, domain in ordered
    )
    return JointSupportProblem(
        slot_keys=keys,
        domains=_ordered_domains(raw, ordering_policy),
        initial_deficits=deficits,
        selected_slot_signature=selected_signature,
        contract_signature=contract_signature,
        extension_signature=extension_signature,
        ordering_policy=ordering_policy,
    )


def propagate_state_slot_requirements_memoized(
    *,
    selected: Sequence[MoveDemandVector],
    remaining_domains: Mapping[str, Sequence[MoveDemandVector]],
    inventory: ColumnInventory,
    node_cap: int = 200_000,
    compute_candidate_support: bool = True,
    memo_store: ExactSuffixMemoStore[JointSupportMemoResult] | None = None,
    ordering_policy: str = "smallest_domain",
    contract_signature: str = "EXACT",
    extension_signature: str = "NONE",
) -> tuple[StateSlotPropagationResult, JointSupportSolveMetrics]:
    """Return the v101.4.2 logical result plus exact-DP metrics."""

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
    all_keys = set(target) | set(current)
    vectors = {
        candidate.candidate_id: candidate_state_slot_vector(candidate)
        for _, domain in ordered
        for candidate in domain
    }
    for vector in vectors.values():
        all_keys.update(vector.counts)
    sorted_all_keys = tuple(sorted(all_keys))

    failures: list[str] = []
    for key in sorted_all_keys:
        if current[key] > target[key]:
            failures.append(
                "CURRENT_EXCEEDS_TARGET:"
                f"{key.move}:{key.piece_type}:{key.state_before_position}"
            )

    traces = []
    for key in sorted_all_keys:
        per_piece_values = []
        for _, domain in ordered:
            values = {
                int(vectors[candidate.candidate_id].counts.get(key, 0))
                for candidate in domain
            }
            per_piece_values.append(values or {0})
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
    failures.extend(f"EMPTY_DOMAIN:{piece}" for piece in empty)
    domain_ids = {
        piece: tuple(candidate.candidate_id for candidate in domain)
        for piece, domain in ordered
    }
    zero_metrics = JointSupportSolveMetrics(0, 0, 0, 0, 0, 0, ())
    if failures:
        return (
            StateSlotPropagationResult(
                status="UNSAT_STATE_SLOT_BOUNDS",
                bound_traces=tuple(traces),
                surviving_candidate_ids={piece: () for piece, _ in ordered},
                pruned_candidate_ids=domain_ids,
                explored_nodes=0,
                cap_hit=False,
                joint_completion_exists=False,
                failure_reasons=tuple(failures),
            ),
            zero_metrics,
        )

    keys = tuple(sorted(target))
    start_remaining = tuple(target[key] - current[key] for key in keys)
    selected_signature = tuple(current[key] for key in keys)
    problem = _problem(
        ordered=ordered,
        keys=keys,
        deficits=start_remaining,
        selected_signature=selected_signature,
        ordering_policy=ordering_policy,
        contract_signature=contract_signature,
        extension_signature=extension_signature,
    )
    budget = _Budget(node_cap)
    overall = _solve_problem(
        problem,
        node_cap=node_cap,
        memo_store=memo_store,
        budget=budget,
    )
    if overall.status == UNKNOWN_CAP:
        result = StateSlotPropagationResult(
            status="UNKNOWN_STATE_SLOT_CAP",
            bound_traces=tuple(traces),
            surviving_candidate_ids=domain_ids,
            pruned_candidate_ids={piece: () for piece, _ in ordered},
            explored_nodes=budget.logical_nodes,
            cap_hit=True,
            joint_completion_exists=None,
            failure_reasons=("JOINT_SUPPORT_NODE_CAP",),
        )
    elif overall.status == UNSUPPORTED_PROVEN:
        result = StateSlotPropagationResult(
            status="UNSAT_STATE_SLOT_JOINT_SUPPORT",
            bound_traces=tuple(traces),
            surviving_candidate_ids={piece: () for piece, _ in ordered},
            pruned_candidate_ids=domain_ids,
            explored_nodes=budget.logical_nodes,
            cap_hit=False,
            joint_completion_exists=False,
            failure_reasons=("NO_JOINT_SLOT_COMPLETION",),
        )
    else:
        surviving: dict[str, tuple[str, ...]] = {}
        pruned: dict[str, tuple[str, ...]] = {}
        cap_hit = False
        if compute_candidate_support:
            for piece, domain in ordered:
                keep = []
                drop = []
                for candidate in domain:
                    forced = _solve_problem(
                        problem.with_forced_candidate(
                            piece,
                            candidate.candidate_id,
                        ),
                        node_cap=node_cap,
                        memo_store=memo_store,
                        budget=budget,
                    )
                    if forced.status == UNKNOWN_CAP:
                        cap_hit = True
                        keep.append(candidate.candidate_id)
                    elif forced.status == SUPPORTED:
                        keep.append(candidate.candidate_id)
                    else:
                        drop.append(candidate.candidate_id)
                surviving[piece] = tuple(keep)
                pruned[piece] = tuple(drop)
        else:
            surviving = domain_ids
            pruned = {piece: () for piece, _ in ordered}
        result = StateSlotPropagationResult(
            status=(
                "UNKNOWN_STATE_SLOT_CAP"
                if cap_hit
                else "SAT_STATE_SLOT_SUPPORT"
            ),
            bound_traces=tuple(traces),
            surviving_candidate_ids=surviving,
            pruned_candidate_ids=pruned,
            explored_nodes=budget.logical_nodes,
            cap_hit=cap_hit,
            joint_completion_exists=True,
            failure_reasons=(
                ("CANDIDATE_SUPPORT_NODE_CAP",) if cap_hit else ()
            ),
        )

    metrics = JointSupportSolveMetrics(
        logical_nodes=budget.logical_nodes,
        expanded_nodes=budget.expanded_nodes,
        memo_hits=budget.memo_hits,
        memo_misses=budget.memo_misses,
        reused_logical_nodes=budget.reused_logical_nodes,
        solve_calls=budget.solve_calls,
        key_rows=tuple(budget.key_rows),
    )
    return result, metrics


__all__ = [
    "SUPPORTED",
    "UNSUPPORTED_PROVEN",
    "UNKNOWN_CAP",
    "JointSupportMemoResult",
    "JointSupportSolveMetrics",
    "propagate_state_slot_requirements_memoized",
]
