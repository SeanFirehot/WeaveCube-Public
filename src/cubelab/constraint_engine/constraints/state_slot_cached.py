from __future__ import annotations

"""v101.5.1 cached adapter for the sound state-slot propagator."""

from time import perf_counter_ns

from cubelab.move_demand_balance import MoveDemandVector

from ..cache import StateSlotSupportCache
from ..domains import PropagationState
from ..requirements import (
    CornerEdgeBalanceRequirement,
    MoveResidueRequirement,
    StateSlotRequirement,
)
from ..types import CandidateRemoval, ConstraintResult, ConstraintStatus
from .core import BaseConstraint, _result


class CachedStateSlotConstraint(BaseConstraint):
    name = "state_slot_capacity"
    tier = BaseConstraint.tier
    estimated_cost_class = 2

    def __init__(
        self,
        *,
        maximum_joint_entries: int = 256,
        maximum_subproblem_entries: int = 100_000,
        dirty_updates_enabled: bool = True,
        joint_cache_enabled: bool = True,
        subproblem_memo_enabled: bool = False,
        subproblem_ordering_policy: str = "smallest_domain",
        fine_grained_dirty_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.maximum_joint_entries = maximum_joint_entries
        self.maximum_subproblem_entries = maximum_subproblem_entries
        self.dirty_updates_enabled = dirty_updates_enabled
        self.joint_cache_enabled = joint_cache_enabled
        self.subproblem_memo_enabled = subproblem_memo_enabled
        self.subproblem_ordering_policy = subproblem_ordering_policy
        self.fine_grained_dirty_enabled = fine_grained_dirty_enabled
        self.cache: StateSlotSupportCache | None = None
        self._cache_snapshot_stack: list[dict[str, object] | None] = []
        self._propagation_calls = 0

    def initialize(self, context) -> None:
        super().initialize(context)
        self.cache = None
        self._cache_snapshot_stack.clear()
        self._propagation_calls = 0

    def push_candidate(self, candidate: MoveDemandVector) -> None:
        self._cache_snapshot_stack.append(
            None
            if self.cache is None
            else self.cache.logical_snapshot()
        )
        super().push_candidate(candidate)

    def pop_candidate(self, candidate: MoveDemandVector) -> None:
        super().pop_candidate(candidate)
        if not self._cache_snapshot_stack:
            raise AssertionError("state-slot cache snapshot underflow")
        snapshot = self._cache_snapshot_stack.pop()
        if snapshot is not None:
            if self.cache is None:
                raise AssertionError("state-slot cache disappeared")
            self.cache.restore_logical_snapshot(snapshot)

    def candidate_affects_slots(self, candidate_id: str) -> bool:
        return (
            self.cache is None
            or self.cache.candidate_affects_slots(candidate_id)
        )

    def state_hash_payload(self) -> object:
        return {
            "selection_stack": tuple(self._selection_stack),
            "cache_logical_hash": (
                None if self.cache is None else self.cache.logical_hash()
            ),
            "cache_snapshot_depth": len(self._cache_snapshot_stack),
        }

    def propagate(self, state: PropagationState) -> ConstraintResult:
        inventory = state.inventory or self.context.inventory
        if inventory is None:
            return _result(
                self,
                metrics={
                    "deferred": "NO_FIXED_INVENTORY",
                    "execution_path": "INVENTORY_UNKNOWN_PATH",
                },
            )
        if self.cache is None:
            self.cache = StateSlotSupportCache.from_state(
                state,
                maximum_joint_entries=self.maximum_joint_entries,
                maximum_subproblem_entries=(
                    self.maximum_subproblem_entries
                ),
                dirty_updates_enabled=self.dirty_updates_enabled,
                joint_cache_enabled=self.joint_cache_enabled,
                subproblem_memo_enabled=self.subproblem_memo_enabled,
                subproblem_ordering_policy=(
                    self.subproblem_ordering_policy
                ),
                fine_grained_dirty_enabled=(
                    self.fine_grained_dirty_enabled
                ),
            )
        self._propagation_calls += 1
        evaluation = self.cache.evaluate(
            state,
            inventory=inventory,
            node_cap=self.context.state_slot_node_cap,
            compute_candidate_support=True,
            propagation_round=self._propagation_calls,
        )
        result = evaluation.result

        requirement_started = perf_counter_ns()
        requirements = []
        for trace in result.bound_traces:
            needed = (
                trace.exact_required_target - trace.current_demand
            )
            if needed <= 0:
                continue
            requirements.append(
                StateSlotRequirement(
                    source_constraint=self.name,
                    sound=True,
                    explanation=(
                        f"remaining domain must supply {needed} "
                        f"{trace.key.move}@"
                        f"{trace.key.state_before_position} cells"
                    ),
                    move=trace.key.move,
                    piece_type=trace.key.piece_type,
                    required_state_before=(
                        trace.key.state_before_position,
                    ),
                    required_multiplicity=needed,
                )
            )
        requirement_ns = perf_counter_ns() - requirement_started

        filtering_started = perf_counter_ns()
        requirement_by_key = {
            (
                requirement.move,
                requirement.piece_type,
                requirement.required_state_before[0],
            ): requirement.requirement_id
            for requirement in requirements
        }
        requirement_ids = tuple(
            sorted(set(requirement_by_key.values()))
        )
        removals = tuple(
            CandidateRemoval(
                piece_id=piece,
                candidate_id=candidate_id,
                reason="NO_JOINT_STATE_SLOT_SUPPORT",
                requirement_ids=requirement_ids,
            )
            for piece, candidate_ids in sorted(
                result.pruned_candidate_ids.items()
            )
            for candidate_id in candidate_ids
        )
        filtering_ns = perf_counter_ns() - filtering_started
        consumed = tuple(
            requirement_id
            for requirement_id, requirement in (
                state.generated_requirements.items()
            )
            if isinstance(
                requirement,
                (
                    MoveResidueRequirement,
                    CornerEdgeBalanceRequirement,
                ),
            )
        )
        profile = self.cache.replace_last_profile(
            requirement_generation_ns=requirement_ns,
            candidate_filtering_ns=filtering_ns,
            total_ns=(
                evaluation.profile.total_ns
                + requirement_ns
                + filtering_ns
            ),
        )
        metrics = {
            "explored_nodes": result.explored_nodes,
            "joint_completion_exists": result.joint_completion_exists,
            "full_recompute": profile.full_recompute,
            "joint_cache_hit": profile.joint_cache_hit,
            "affected_slots": profile.affected_slots,
            "affected_moves": len(profile.affected_moves),
            "domain_changes_since_last_call": (
                profile.domain_changes_since_last_call
            ),
            "execution_path": profile.execution_path,
            "subproblem_memo_hits": profile.subproblem_memo_hits,
            "subproblem_memo_misses": profile.subproblem_memo_misses,
            "subproblem_expanded_nodes": (
                profile.subproblem_expanded_nodes
            ),
            "subproblem_reused_nodes": (
                profile.subproblem_reused_nodes
            ),
        }
        if result.status.startswith("UNSAT"):
            return _result(
                self,
                ConstraintStatus.UNSAT_PROVEN,
                requirements=tuple(requirements),
                removals=removals,
                consumed_requirement_ids=consumed,
                rejection_reason=result.status,
                metrics={
                    **metrics,
                    "failure_count": len(result.failure_reasons),
                },
            )
        if result.cap_hit:
            return _result(
                self,
                ConstraintStatus.UNKNOWN_CAP,
                requirements=tuple(requirements),
                removals=removals,
                consumed_requirement_ids=consumed,
                unknown_reason="STATE_SLOT_NODE_CAP",
                metrics=metrics,
            )
        return _result(
            self,
            ConstraintStatus.SAT_WITH_REQUIREMENTS,
            requirements=tuple(requirements),
            removals=removals,
            consumed_requirement_ids=consumed,
            metrics=metrics,
        )


__all__ = ["CachedStateSlotConstraint"]
