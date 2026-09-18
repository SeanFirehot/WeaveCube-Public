from __future__ import annotations

"""Incremental, fixed-order, fixpoint cascade for CubeLab v101.5."""

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from time import perf_counter_ns
from typing import Iterable

from cubelab.move_demand_balance import MoveDemandVector

from .conflict_core import ConflictCore, extract_conflict_core
from .constraints import (
    ContractValidationConstraint,
    CornerEdgeEqualityConstraint,
    DemandBoundsConstraint,
    ExactSlotPackingConstraint,
    IncompatibilityGraphConstraint,
    LinearizationReplayConstraint,
    MoveMod4Constraint,
    MoveMultiplicityConstraint,
    OrientationParityConstraint,
    PrecedenceConstraint,
    StateFlowConstraint,
    StateIndexedResidueConstraint,
    StateSlotConstraint,
)
from .domains import PropagationSnapshot, PropagationState
from .metrics import ConstraintMetrics
from .registry import ConstraintRegistry
from .scheduler import FixedConstraintScheduler
from .types import (
    CandidateRemoval,
    CascadeOutcome,
    ConstraintContext,
    ConstraintResult,
    ConstraintStatus,
    ForcedAssignment,
    IncrementalConstraint,
)


@dataclass(slots=True)
class _BranchFrame:
    snapshot: PropagationSnapshot
    expected_state_hash: str
    assigned_candidates: list[MoveDemandVector]


def default_constraints() -> tuple[IncrementalConstraint, ...]:
    """The declared v101.5 cheap-to-expensive fixed order."""

    return (
        ContractValidationConstraint(),
        DemandBoundsConstraint(),
        MoveMod4Constraint(),
        CornerEdgeEqualityConstraint(),
        MoveMultiplicityConstraint(),
        StateSlotConstraint(),
        StateIndexedResidueConstraint(),
        StateFlowConstraint(),
        IncompatibilityGraphConstraint(),
        OrientationParityConstraint(),
        ExactSlotPackingConstraint(),
        PrecedenceConstraint(),
        LinearizationReplayConstraint(),
    )


class CascadedConstraintEngine:
    def __init__(
        self,
        *,
        context: ConstraintContext | None = None,
        constraints: Iterable[IncrementalConstraint] | None = None,
    ) -> None:
        self.context = context or ConstraintContext()
        self.registry = ConstraintRegistry(
            default_constraints() if constraints is None else constraints
        )
        self.scheduler = FixedConstraintScheduler.from_sequence(
            self.registry.constraints
        )
        self.metrics = ConstraintMetrics()
        self.conflict_cores: list[ConflictCore] = []
        self._frames: list[_BranchFrame] = []
        self._initialized = False

    @property
    def constraints(self) -> tuple[IncrementalConstraint, ...]:
        return self.scheduler.ordered_constraints()

    def initialize(self, state: PropagationState) -> None:
        if (
            self.context.inventory is not None
            and state.inventory is not None
            and self.context.inventory.row() != state.inventory.row()
        ):
            raise ValueError("context and state inventory differ")
        for constraint in self.constraints:
            constraint.initialize(self.context)
        self.metrics = ConstraintMetrics()
        self.conflict_cores.clear()
        self._frames.clear()
        self._initialized = True
        for piece, domain in sorted(state.domains.items()):
            if len(domain.candidates) == 1:
                candidate_id = next(iter(domain.candidates))
                self._assign(
                    state,
                    piece,
                    candidate_id,
                    reason="INITIAL_SINGLETON_DOMAIN",
                )

    def _assign(
        self,
        state: PropagationState,
        piece_id: str,
        candidate_id: str,
        *,
        reason: str,
    ) -> bool:
        candidate = state.domains[piece_id].candidates[candidate_id]
        if not state.assign(piece_id, candidate_id):
            return False
        for constraint in self.constraints:
            constraint.push_candidate(candidate)
        if self._frames:
            self._frames[-1].assigned_candidates.append(candidate)
        state.provenance.record_assignment(piece_id, candidate_id, reason)
        return True

    def push_candidate(
        self,
        state: PropagationState,
        candidate: MoveDemandVector,
    ) -> str:
        if not self._initialized:
            raise RuntimeError("engine must be initialized before push")
        expected = self.combined_state_hash(state)
        frame = _BranchFrame(
            snapshot=state.snapshot(),
            expected_state_hash=expected,
            assigned_candidates=[],
        )
        self._frames.append(frame)
        self._assign(
            state,
            candidate.piece_id,
            candidate.candidate_id,
            reason="BRANCH_SELECTION",
        )
        return expected

    def pop_candidate(self, state: PropagationState) -> str:
        if not self._frames:
            raise RuntimeError("no branch frame to pop")
        frame = self._frames.pop()
        for candidate in reversed(frame.assigned_candidates):
            for constraint in reversed(self.constraints):
                constraint.pop_candidate(candidate)
        state.restore(frame.snapshot)
        actual = self.combined_state_hash(state)
        if actual != frame.expected_state_hash:
            raise AssertionError(
                "incremental push/pop failed to restore the exact state hash"
            )
        return actual

    def combined_state_hash(self, state: PropagationState) -> str:
        payload = {
            "propagation_state": state.state_hash(),
            "constraints": {
                constraint.name: constraint.state_hash_payload()
                for constraint in self.constraints
            },
        }
        return sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
        ).hexdigest()

    def _record_new_requirements(
        self,
        state: PropagationState,
        result: ConstraintResult,
    ) -> tuple[object, ...]:
        output = []
        for requirement in result.requirements:
            if not state.add_requirement(requirement):
                continue
            output.append(requirement)
            state.provenance.record_requirement(
                requirement.requirement_id,
                result.constraint_name,
                requirement.requirement_type,
            )
        return tuple(output)

    def _apply_removals(
        self,
        state: PropagationState,
        result: ConstraintResult,
    ) -> tuple[CandidateRemoval, ...]:
        applied = []
        for removal in result.removals:
            selected = state.selected_candidates.get(removal.piece_id)
            if (
                selected is not None
                and selected.candidate_id == removal.candidate_id
            ):
                raise AssertionError(
                    "a propagator attempted to remove a selected candidate"
                )
            domain = state.domains[removal.piece_id]
            if not domain.remove(
                removal.candidate_id,
                f"{result.constraint_name}:{removal.reason}",
            ):
                continue
            applied.append(removal)
            state.provenance.record_removal(
                removal.piece_id,
                removal.candidate_id,
                result.constraint_name,
                removal.reason,
                removal.requirement_ids,
            )
        return tuple(applied)

    def _apply_forced_assignments(
        self,
        state: PropagationState,
        result: ConstraintResult,
    ) -> tuple[ForcedAssignment, ...]:
        applied = []
        for forced in result.forced_assignments:
            if self._assign(
                state,
                forced.piece_id,
                forced.candidate_id,
                reason=f"{result.constraint_name}:{forced.reason}",
            ):
                applied.append(forced)
        return tuple(applied)

    def _force_singletons(
        self,
        state: PropagationState,
        *,
        source_constraint: str,
    ) -> tuple[ForcedAssignment, ...]:
        output = []
        for piece, domain in sorted(state.domains.items()):
            if piece in state.selected_candidates:
                continue
            if len(domain.candidates) != 1:
                continue
            candidate_id = next(iter(domain.candidates))
            forced = ForcedAssignment(
                piece_id=piece,
                candidate_id=candidate_id,
                reason=f"SINGLETON_AFTER:{source_constraint}",
            )
            if self._assign(
                state,
                piece,
                candidate_id,
                reason=forced.reason,
            ):
                output.append(forced)
        return tuple(output)

    def _empty_domain_result(
        self,
        state: PropagationState,
        source: ConstraintResult,
    ) -> ConstraintResult | None:
        empty = tuple(
            piece
            for piece, domain in sorted(state.domains.items())
            if not domain.candidates
        )
        if not empty:
            return None
        candidate_ids = tuple(
            removal.candidate_id
            for removal in source.removals
            if removal.piece_id in empty
        )
        return ConstraintResult(
            constraint_name=source.constraint_name,
            status=ConstraintStatus.UNSAT_PROVEN,
            tier=source.tier,
            rejection_reason=(
                "EMPTY_DOMAIN_AFTER_PROPAGATION:" + ",".join(empty)
            ),
            conflict_candidate_ids=candidate_ids,
            metrics={"empty_piece_count": len(empty)},
        )

    def _finish_unsat(
        self,
        state: PropagationState,
        result: ConstraintResult,
        *,
        round_index: int,
        trace: list[ConstraintResult],
        domain_before: int,
    ) -> CascadeOutcome:
        state.status = ConstraintStatus.UNSAT_PROVEN
        core = extract_conflict_core(result)
        self.conflict_cores.append(core)
        state.artifacts["last_conflict_core"] = core
        trace.append(result)
        self.metrics.record(result, round_index=round_index)
        return CascadeOutcome(
            status=ConstraintStatus.UNSAT_PROVEN,
            fixpoint_rounds=round_index,
            results=tuple(trace),
            conflict_core_id=core.core_id,
            unknown_count=sum(
                item.status is ConstraintStatus.UNKNOWN_CAP
                for item in trace
            ),
            domain_size_before=domain_before,
            domain_size_after=state.domain_size,
        )

    def propagate_to_fixpoint(
        self,
        state: PropagationState,
    ) -> CascadeOutcome:
        if not self._initialized:
            self.initialize(state)
        domain_before = state.domain_size
        trace: list[ConstraintResult] = []
        for round_index in range(1, self.context.max_fixpoint_rounds + 1):
            changed = False
            for constraint in self.constraints:
                before = state.domain_size
                started = perf_counter_ns()
                raw = constraint.propagate(state)
                runtime = perf_counter_ns() - started
                dedup_started = perf_counter_ns()
                new_requirements = self._record_new_requirements(state, raw)
                dedup_runtime = perf_counter_ns() - dedup_started
                provenance_started = perf_counter_ns()
                removals = self._apply_removals(state, raw)
                provenance_runtime = (
                    perf_counter_ns() - provenance_started
                )
                cache = getattr(constraint, "cache", None)
                if (
                    raw.constraint_name == "state_slot_capacity"
                    and cache is not None
                    and cache.profiles
                ):
                    previous_profile = cache.profiles[-1]
                    cache.replace_last_profile(
                        deduplication_ns=dedup_runtime,
                        provenance_ns=provenance_runtime,
                        total_ns=(
                            previous_profile.total_ns
                            + dedup_runtime
                            + provenance_runtime
                        ),
                    )
                forced = self._apply_forced_assignments(state, raw)
                forced += self._force_singletons(
                    state,
                    source_constraint=raw.constraint_name,
                )
                effective = replace(
                    raw,
                    requirements=new_requirements,
                    removals=removals,
                    forced_assignments=forced,
                    runtime_ns=runtime,
                    domain_size_before=before,
                    domain_size_after=state.domain_size,
                )
                if raw.status is ConstraintStatus.UNKNOWN_CAP:
                    state.unknown_events.append(
                        {
                            "constraint_name": raw.constraint_name,
                            "round_index": round_index,
                            "unknown_reason": raw.unknown_reason,
                        }
                    )
                if raw.status is ConstraintStatus.UNSAT_PROVEN:
                    return self._finish_unsat(
                        state,
                        effective,
                        round_index=round_index,
                        trace=trace,
                        domain_before=domain_before,
                    )
                empty = self._empty_domain_result(state, effective)
                if empty is not None:
                    empty = replace(
                        empty,
                        runtime_ns=runtime,
                        domain_size_before=before,
                        domain_size_after=state.domain_size,
                    )
                    return self._finish_unsat(
                        state,
                        empty,
                        round_index=round_index,
                        trace=trace,
                        domain_before=domain_before,
                    )
                trace.append(effective)
                self.metrics.record(
                    effective,
                    round_index=round_index,
                )
                changed |= bool(new_requirements or removals or forced)
            if not changed:
                status = (
                    ConstraintStatus.SAT_WITH_REQUIREMENTS
                    if state.generated_requirements
                    else ConstraintStatus.SAT
                )
                state.status = status
                return CascadeOutcome(
                    status=status,
                    fixpoint_rounds=round_index,
                    results=tuple(trace),
                    conflict_core_id=None,
                    unknown_count=sum(
                        item.status is ConstraintStatus.UNKNOWN_CAP
                        for item in trace
                    ),
                    domain_size_before=domain_before,
                    domain_size_after=state.domain_size,
                )
        result = ConstraintResult(
            constraint_name="cascade_fixpoint",
            status=ConstraintStatus.UNKNOWN_CAP,
            tier=self.constraints[-1].tier,
            unknown_reason="FIXPOINT_ROUND_CAP",
            domain_size_before=domain_before,
            domain_size_after=state.domain_size,
        )
        state.unknown_events.append(
            {
                "constraint_name": result.constraint_name,
                "round_index": self.context.max_fixpoint_rounds,
                "unknown_reason": result.unknown_reason,
            }
        )
        trace.append(result)
        self.metrics.record(
            result,
            round_index=self.context.max_fixpoint_rounds,
        )
        state.status = ConstraintStatus.UNKNOWN_CAP
        return CascadeOutcome(
            status=ConstraintStatus.UNKNOWN_CAP,
            fixpoint_rounds=self.context.max_fixpoint_rounds,
            results=tuple(trace),
            conflict_core_id=None,
            unknown_count=sum(
                item.status is ConstraintStatus.UNKNOWN_CAP
                for item in trace
            ),
            domain_size_before=domain_before,
            domain_size_after=state.domain_size,
        )

    def row(self, state: PropagationState) -> dict[str, object]:
        return {
            "scheduler": self.scheduler.row(),
            "constraint_registry": list(self.registry.rows()),
            "state": state.row(),
            "constraint_metrics": list(self.metrics.rows()),
            "requirement_reuse": self.metrics.requirement_reuse_row(
                state.generated_requirements
            ),
            "provenance": state.provenance.row(),
            "conflict_cores": [
                core.row() for core in self.conflict_cores
            ],
        }


__all__ = ["default_constraints", "CascadedConstraintEngine"]
