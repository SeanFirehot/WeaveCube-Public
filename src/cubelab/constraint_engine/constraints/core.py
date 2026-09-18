from __future__ import annotations

"""Sound fixed-order propagators for the v101.5 cascade."""

from collections import Counter, defaultdict
from dataclasses import replace
from typing import Iterable, Mapping, Sequence

from cubelab.column_adjacency_graph import (
    linearize_columns_with_state_flow,
)
from cubelab.column_transition_templates import pack_column_templates
from cubelab.fresh_candidate_balance_v101_4_1 import (
    CandidateContractType,
    build_incompatibility_graphs,
    evaluate_state_indexed_residue,
)
from cubelab.move_demand_balance import (
    MOVE_INDEX,
    MOVE_ORDER,
    MoveDemandVector,
    build_precedence_graph,
    concrete_replay,
    evaluate_state_conservation,
)
from cubelab.state_flow_conservation import evaluate_piece_state_flow
from cubelab.state_slot_propagation import (
    propagate_state_slot_requirements,
)
from cubelab.transformations import PIECES

from ..domains import PropagationState
from ..requirements import (
    ColumnMultiplicityRequirement,
    CornerEdgeBalanceRequirement,
    MoveResidueRequirement,
    ObligationRequirement,
    PrecedenceRequirement,
    StateSlotRequirement,
    TransitionRequirement,
)
from ..types import (
    CandidateRemoval,
    ConstraintContext,
    ConstraintResult,
    ConstraintStatus,
    ConstraintTier,
)


def _result(
    constraint: "BaseConstraint",
    status: ConstraintStatus = ConstraintStatus.SAT,
    **kwargs: object,
) -> ConstraintResult:
    return ConstraintResult(
        constraint_name=constraint.name,
        status=status,
        tier=constraint.tier,
        **kwargs,
    )


def _selected_count(
    state: PropagationState,
    piece_type: str,
    move_index: int,
) -> int:
    return sum(
        candidate.counts[move_index]
        for candidate in state.selected_candidates.values()
        if candidate.piece_type == piece_type
    )


def _unselected_by_type(
    state: PropagationState,
    piece_type: str,
) -> tuple[tuple[str, tuple[MoveDemandVector, ...]], ...]:
    return tuple(
        (piece, tuple(domain.candidates.values()))
        for piece, domain in sorted(state.domains.items())
        if piece not in state.selected_candidates
        and any(
            candidate.piece_type == piece_type
            for candidate in domain.candidates.values()
        )
    )


def _merge_removal(
    removals: dict[tuple[str, str], tuple[list[str], set[str]]],
    piece_id: str,
    candidate_id: str,
    reason: str,
    requirement_ids: Iterable[str] = (),
) -> None:
    reasons, identifiers = removals.setdefault(
        (piece_id, candidate_id), ([], set())
    )
    if reason not in reasons:
        reasons.append(reason)
    identifiers.update(requirement_ids)


def _removal_rows(
    removals: Mapping[tuple[str, str], tuple[list[str], set[str]]],
) -> tuple[CandidateRemoval, ...]:
    return tuple(
        CandidateRemoval(
            piece_id=piece,
            candidate_id=candidate_id,
            reason=";".join(reasons),
            requirement_ids=tuple(sorted(requirement_ids)),
        )
        for (piece, candidate_id), (reasons, requirement_ids) in sorted(
            removals.items()
        )
    )


class BaseConstraint:
    name = "base"
    tier = ConstraintTier.SOUND_RELAXATION
    estimated_cost_class = 1

    def __init__(self) -> None:
        self.context = ConstraintContext()
        self._selection_stack: list[str] = []

    def initialize(self, context: ConstraintContext) -> None:
        self.context = context
        self._selection_stack.clear()

    def push_candidate(self, candidate: MoveDemandVector) -> None:
        self._selection_stack.append(candidate.candidate_id)

    def pop_candidate(self, candidate: MoveDemandVector) -> None:
        if (
            not self._selection_stack
            or self._selection_stack[-1] != candidate.candidate_id
        ):
            raise AssertionError(
                f"{self.name} incremental pop order mismatch"
            )
        self._selection_stack.pop()

    def state_hash_payload(self) -> object:
        return tuple(self._selection_stack)


class ContractValidationConstraint(BaseConstraint):
    name = "contract_validation"
    tier = ConstraintTier.HARD
    estimated_cost_class = 1

    def propagate(self, state: PropagationState) -> ConstraintResult:
        removals: dict[tuple[str, str], tuple[list[str], set[str]]] = {}
        requirements = []
        for piece, domain in state.domains.items():
            for candidate in tuple(domain.candidates.values()):
                reasons = []
                if candidate.piece_id != piece:
                    reasons.append("PIECE_ID_MISMATCH")
                if len(candidate.counts) != len(MOVE_ORDER):
                    reasons.append("MOVE_ORDER_WIDTH")
                if sum(candidate.counts) != candidate.active_length:
                    reasons.append("ACTIVE_LENGTH_COUNT_MISMATCH")
                contract = state.contracts.get(candidate.candidate_id)
                if contract is not None:
                    if contract.piece_id != piece:
                        reasons.append("CONTRACT_PIECE_MISMATCH")
                    if (
                        contract.contract_type
                        is CandidateContractType.EXACT_PROJECTION
                        and contract.exact_active_skeleton
                        != candidate.active_skeleton
                    ):
                        reasons.append("EXACT_CONTRACT_SKELETON_MISMATCH")
                    if (
                        contract.contract_type
                        is not CandidateContractType.EXACT_PROJECTION
                        and not contract.allowed_extra_event_domains
                    ):
                        # Extensible candidates remain possible, but the exact
                        # engine cannot turn missing finite support into UNSAT.
                        continue
                    for domain in contract.allowed_extra_event_domains:
                        exit_state = (
                            domain.allowed_state_after[0]
                            if len(domain.allowed_state_after) == 1
                            else None
                        )
                        requirements.append(
                            ObligationRequirement(
                                source_constraint=self.name,
                                sound=True,
                                explanation=(
                                    "extensible candidate preserves a finite "
                                    "active-gap obligation domain"
                                ),
                                piece_id=piece,
                                obligation_type=domain.obligation_effect,
                                required_exit_state=exit_state,
                                allowed_gap_domain=(
                                    domain.placement_interval,
                                    domain.allowed_moves,
                                ),
                            )
                        )
                if not reasons:
                    continue
                if piece in state.selected_candidates:
                    return _result(
                        self,
                        ConstraintStatus.UNSAT_PROVEN,
                        requirements=tuple(requirements),
                        rejection_reason="SELECTED_CONTRACT_INVALID",
                        conflict_candidate_ids=(candidate.candidate_id,),
                        metrics={"reasons": "|".join(reasons)},
                    )
                _merge_removal(
                    removals,
                    piece,
                    candidate.candidate_id,
                    "|".join(reasons),
                )
        return _result(
            self,
            (
                ConstraintStatus.SAT_WITH_REQUIREMENTS
                if removals or requirements
                else ConstraintStatus.SAT
            ),
            requirements=tuple(requirements),
            removals=_removal_rows(removals),
            metrics={"validated_candidates": state.domain_size},
        )


class DemandBoundsConstraint(BaseConstraint):
    name = "demand_min_max_bounds"
    tier = ConstraintTier.SOUND_RELAXATION
    estimated_cost_class = 1

    def propagate(self, state: PropagationState) -> ConstraintResult:
        inventory = state.inventory or self.context.inventory
        if inventory is None:
            return _result(
                self,
                metrics={"deferred": "NO_FIXED_INVENTORY"},
            )
        requirements = []
        removals: dict[tuple[str, str], tuple[list[str], set[str]]] = {}
        for piece_type in ("CORNER", "EDGE"):
            domains = _unselected_by_type(state, piece_type)
            if any(not domain for _, domain in domains):
                return _result(
                    self,
                    ConstraintStatus.UNSAT_PROVEN,
                    rejection_reason="EMPTY_DOMAIN",
                )
            for move_index, move in enumerate(MOVE_ORDER):
                target = 4 * int(inventory.move_counts.get(move, 0))
                current = _selected_count(
                    state, piece_type, move_index
                )
                if current > target:
                    return _result(
                        self,
                        ConstraintStatus.UNSAT_PROVEN,
                        rejection_reason=(
                            f"DEMAND_EXCEEDS_TARGET:{piece_type}:{move}"
                        ),
                        conflict_candidate_ids=tuple(
                            candidate.candidate_id
                            for candidate in state.selected_candidates.values()
                            if candidate.piece_type == piece_type
                            and candidate.counts[move_index]
                        ),
                    )
                mins = {
                    piece: min(c.counts[move_index] for c in domain)
                    for piece, domain in domains
                }
                maxs = {
                    piece: max(c.counts[move_index] for c in domain)
                    for piece, domain in domains
                }
                minimum = sum(mins.values())
                maximum = sum(maxs.values())
                needed = target - current
                requirement = MoveResidueRequirement(
                    source_constraint=self.name,
                    sound=True,
                    explanation=(
                        f"{piece_type} {move} must add exactly {needed} "
                        "events within the remaining domains"
                    ),
                    piece_type=piece_type,
                    move=move,
                    required_residue_mod4=needed % 4,
                    minimum_additional_count=needed,
                    maximum_additional_count=needed,
                )
                requirements.append(requirement)
                if needed < minimum or needed > maximum:
                    return _result(
                        self,
                        ConstraintStatus.UNSAT_PROVEN,
                        requirements=tuple(requirements),
                        rejection_reason=(
                            f"DEMAND_BOUND_UNREACHABLE:{piece_type}:{move}"
                        ),
                    )
                for piece, domain in domains:
                    other_min = minimum - mins[piece]
                    other_max = maximum - maxs[piece]
                    for candidate in domain:
                        final_low = (
                            current
                            + candidate.counts[move_index]
                            + other_min
                        )
                        final_high = (
                            current
                            + candidate.counts[move_index]
                            + other_max
                        )
                        if final_low <= target <= final_high:
                            continue
                        _merge_removal(
                            removals,
                            piece,
                            candidate.candidate_id,
                            f"DEMAND_BOUND:{piece_type}:{move}",
                            (requirement.requirement_id,),
                        )
        return _result(
            self,
            ConstraintStatus.SAT_WITH_REQUIREMENTS,
            requirements=tuple(requirements),
            removals=_removal_rows(removals),
            metrics={
                "fixed_inventory": True,
                "dimension_count": 2 * len(MOVE_ORDER),
            },
        )


class MoveMod4Constraint(BaseConstraint):
    name = "move_mod4"
    tier = ConstraintTier.HARD
    estimated_cost_class = 1

    def propagate(self, state: PropagationState) -> ConstraintResult:
        requirements = []
        removals: dict[tuple[str, str], tuple[list[str], set[str]]] = {}
        support_checks = 0
        for piece_type in ("CORNER", "EDGE"):
            domains = _unselected_by_type(state, piece_type)
            for move_index, move in enumerate(MOVE_ORDER):
                current = _selected_count(
                    state, piece_type, move_index
                )
                needed = (-current) % 4
                requirement = MoveResidueRequirement(
                    source_constraint=self.name,
                    sound=True,
                    explanation=(
                        f"remaining {piece_type} candidates must supply "
                        f"{move} residue {needed} mod 4"
                    ),
                    piece_type=piece_type,
                    move=move,
                    required_residue_mod4=needed,
                    minimum_additional_count=needed,
                    maximum_additional_count=None,
                )
                requirements.append(requirement)
                prefix = [{0}]
                for _, domain in domains:
                    prefix.append(
                        {
                            (left + candidate.counts[move_index]) % 4
                            for left in prefix[-1]
                            for candidate in domain
                        }
                    )
                suffix: list[set[int]] = [set() for _ in range(len(domains) + 1)]
                suffix[-1] = {0}
                for index in range(len(domains) - 1, -1, -1):
                    suffix[index] = {
                        (candidate.counts[move_index] + right) % 4
                        for candidate in domains[index][1]
                        for right in suffix[index + 1]
                    }
                if needed not in prefix[-1]:
                    return _result(
                        self,
                        ConstraintStatus.UNSAT_PROVEN,
                        requirements=tuple(requirements),
                        rejection_reason=(
                            f"NO_MOD4_SUPPORT:{piece_type}:{move}"
                        ),
                    )
                for index, (piece, domain) in enumerate(domains):
                    for candidate in domain:
                        support_checks += 1
                        supported = any(
                            (
                                current
                                + left
                                + candidate.counts[move_index]
                                + right
                            )
                            % 4
                            == 0
                            for left in prefix[index]
                            for right in suffix[index + 1]
                        )
                        if not supported:
                            _merge_removal(
                                removals,
                                piece,
                                candidate.candidate_id,
                                f"MOD4_SUPPORT:{piece_type}:{move}",
                                (requirement.requirement_id,),
                            )
        return _result(
            self,
            ConstraintStatus.SAT_WITH_REQUIREMENTS,
            requirements=tuple(requirements),
            removals=_removal_rows(removals),
            metrics={"candidate_dimension_support_checks": support_checks},
        )


def _sum_prefix_suffix(
    domains: Sequence[tuple[str, tuple[MoveDemandVector, ...]]],
    move_index: int,
) -> tuple[list[set[int]], list[set[int]]]:
    prefix = [{0}]
    for _, domain in domains:
        prefix.append(
            {
                left + candidate.counts[move_index]
                for left in prefix[-1]
                for candidate in domain
            }
        )
    suffix: list[set[int]] = [set() for _ in range(len(domains) + 1)]
    suffix[-1] = {0}
    for index in range(len(domains) - 1, -1, -1):
        suffix[index] = {
            candidate.counts[move_index] + right
            for candidate in domains[index][1]
            for right in suffix[index + 1]
        }
    return prefix, suffix


class CornerEdgeEqualityConstraint(BaseConstraint):
    name = "corner_edge_equality"
    tier = ConstraintTier.HARD
    estimated_cost_class = 1

    def propagate(self, state: PropagationState) -> ConstraintResult:
        removals: dict[tuple[str, str], tuple[list[str], set[str]]] = {}
        requirements = []
        support_checks = 0
        corner_domains = _unselected_by_type(state, "CORNER")
        edge_domains = _unselected_by_type(state, "EDGE")
        for move_index, move in enumerate(MOVE_ORDER):
            corner_current = _selected_count(state, "CORNER", move_index)
            edge_current = _selected_count(state, "EDGE", move_index)
            corner_prefix, corner_suffix = _sum_prefix_suffix(
                corner_domains, move_index
            )
            edge_prefix, edge_suffix = _sum_prefix_suffix(
                edge_domains, move_index
            )
            corner_totals = {
                corner_current + value for value in corner_prefix[-1]
            }
            edge_totals = {
                edge_current + value for value in edge_prefix[-1]
            }
            intersection = corner_totals & edge_totals
            direction = (
                "balanced"
                if corner_current == edge_current
                else "edge_must_catch_up"
                if corner_current > edge_current
                else "corner_must_catch_up"
            )
            requirement = CornerEdgeBalanceRequirement(
                source_constraint=self.name,
                sound=True,
                explanation=(
                    f"corner and edge {move} totals must share a final value"
                ),
                move=move,
                current_corner_count=corner_current,
                current_edge_count=edge_current,
                required_delta_direction=direction,
            )
            requirements.append(requirement)
            if not intersection:
                return _result(
                    self,
                    ConstraintStatus.UNSAT_PROVEN,
                    requirements=tuple(requirements),
                    rejection_reason=f"NO_EQUAL_TOTAL_SUPPORT:{move}",
                )
            for piece_type, domains, prefix, suffix, current, other_totals in (
                (
                    "CORNER",
                    corner_domains,
                    corner_prefix,
                    corner_suffix,
                    corner_current,
                    edge_totals,
                ),
                (
                    "EDGE",
                    edge_domains,
                    edge_prefix,
                    edge_suffix,
                    edge_current,
                    corner_totals,
                ),
            ):
                for index, (piece, domain) in enumerate(domains):
                    for candidate in domain:
                        support_checks += 1
                        candidate_totals = {
                            current
                            + left
                            + candidate.counts[move_index]
                            + right
                            for left in prefix[index]
                            for right in suffix[index + 1]
                        }
                        if candidate_totals & other_totals:
                            continue
                        _merge_removal(
                            removals,
                            piece,
                            candidate.candidate_id,
                            f"EQUALITY_SUPPORT:{piece_type}:{move}",
                            (requirement.requirement_id,),
                        )
        consumed = tuple(
            requirement_id
            for requirement_id, requirement in state.generated_requirements.items()
            if isinstance(requirement, MoveResidueRequirement)
        )
        return _result(
            self,
            ConstraintStatus.SAT_WITH_REQUIREMENTS,
            requirements=tuple(requirements),
            removals=_removal_rows(removals),
            consumed_requirement_ids=consumed,
            metrics={"candidate_dimension_support_checks": support_checks},
        )


class MoveMultiplicityConstraint(BaseConstraint):
    name = "move_multiplicity"
    tier = ConstraintTier.SOUND_RELAXATION
    estimated_cost_class = 1

    def propagate(self, state: PropagationState) -> ConstraintResult:
        inventory = state.inventory or self.context.inventory
        if inventory is None:
            return _result(self, metrics={"deferred": "NO_FIXED_INVENTORY"})
        removals: dict[tuple[str, str], tuple[list[str], set[str]]] = {}
        requirements = []
        for move_index, move in enumerate(MOVE_ORDER):
            columns = int(inventory.move_counts.get(move, 0))
            requirements.append(
                ColumnMultiplicityRequirement(
                    source_constraint=self.name,
                    sound=True,
                    explanation=(
                        f"same-piece {move} events require distinct columns"
                    ),
                    move=move,
                    minimum_columns=max(
                        (
                            candidate.counts[move_index]
                            for candidate in state.selected_candidates.values()
                        ),
                        default=0,
                    ),
                    reason_event_ids=(),
                )
            )
            for piece, domain in state.domains.items():
                if piece in state.selected_candidates:
                    candidate = state.selected_candidates[piece]
                    if candidate.counts[move_index] > columns:
                        return _result(
                            self,
                            ConstraintStatus.UNSAT_PROVEN,
                            requirements=tuple(requirements),
                            rejection_reason=(
                                f"SELECTED_MULTIPLICITY:{piece}:{move}"
                            ),
                            conflict_candidate_ids=(
                                candidate.candidate_id,
                            ),
                        )
                    continue
                for candidate in domain.candidates.values():
                    if candidate.counts[move_index] <= columns:
                        continue
                    _merge_removal(
                        removals,
                        piece,
                        candidate.candidate_id,
                        f"MULTIPLICITY:{move}",
                        (requirements[-1].requirement_id,),
                    )
        return _result(
            self,
            ConstraintStatus.SAT_WITH_REQUIREMENTS,
            requirements=tuple(requirements),
            removals=_removal_rows(removals),
        )


class StateSlotConstraint(BaseConstraint):
    name = "state_slot_capacity"
    tier = ConstraintTier.SOUND_RELAXATION
    estimated_cost_class = 2

    def propagate(self, state: PropagationState) -> ConstraintResult:
        inventory = state.inventory or self.context.inventory
        if inventory is None:
            return _result(self, metrics={"deferred": "NO_FIXED_INVENTORY"})
        result = propagate_state_slot_requirements(
            selected=state.selected(),
            remaining_domains=state.unselected_domains(),
            inventory=inventory,
            node_cap=self.context.state_slot_node_cap,
            compute_candidate_support=True,
        )
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
        requirement_by_key = {
            (
                requirement.move,
                requirement.piece_type,
                requirement.required_state_before[0],
            ): requirement.requirement_id
            for requirement in requirements
        }
        removals = []
        for piece, candidate_ids in result.pruned_candidate_ids.items():
            for candidate_id in candidate_ids:
                removals.append(
                    CandidateRemoval(
                        piece_id=piece,
                        candidate_id=candidate_id,
                        reason="NO_JOINT_STATE_SLOT_SUPPORT",
                        requirement_ids=tuple(
                            sorted(
                                set(requirement_by_key.values())
                            )
                        ),
                    )
                )
        consumed = tuple(
            requirement_id
            for requirement_id, requirement in state.generated_requirements.items()
            if isinstance(
                requirement,
                (MoveResidueRequirement, CornerEdgeBalanceRequirement),
            )
        )
        if result.status.startswith("UNSAT"):
            return _result(
                self,
                ConstraintStatus.UNSAT_PROVEN,
                requirements=tuple(requirements),
                removals=tuple(removals),
                consumed_requirement_ids=consumed,
                rejection_reason=result.status,
                metrics={
                    "explored_nodes": result.explored_nodes,
                    "failure_count": len(result.failure_reasons),
                },
            )
        if result.cap_hit:
            return _result(
                self,
                ConstraintStatus.UNKNOWN_CAP,
                requirements=tuple(requirements),
                removals=tuple(removals),
                consumed_requirement_ids=consumed,
                unknown_reason="STATE_SLOT_NODE_CAP",
                metrics={
                    "explored_nodes": result.explored_nodes,
                    "joint_completion_exists": result.joint_completion_exists,
                },
            )
        return _result(
            self,
            ConstraintStatus.SAT_WITH_REQUIREMENTS,
            requirements=tuple(requirements),
            removals=tuple(removals),
            consumed_requirement_ids=consumed,
            metrics={
                "explored_nodes": result.explored_nodes,
                "joint_completion_exists": result.joint_completion_exists,
            },
        )


class StateIndexedResidueConstraint(BaseConstraint):
    name = "state_indexed_residue"
    tier = ConstraintTier.HARD
    estimated_cost_class = 2

    def propagate(self, state: PropagationState) -> ConstraintResult:
        inventory = state.inventory or self.context.inventory
        if inventory is None:
            return _result(self, metrics={"deferred": "NO_FIXED_INVENTORY"})
        if not state.complete:
            return _result(
                self,
                metrics={"deferred": "INCOMPLETE_SELECTION"},
            )
        result = evaluate_state_indexed_residue(
            state.selected(),
            inventory,
        )
        state.artifacts["state_indexed_residue"] = result
        if result.status.startswith("UNSAT"):
            return _result(
                self,
                ConstraintStatus.UNSAT_PROVEN,
                rejection_reason=result.status,
                conflict_candidate_ids=tuple(
                    candidate.candidate_id for candidate in state.selected()
                ),
                metrics={
                    "failure_count": len(result.failure_reasons),
                },
            )
        consumed = tuple(
            requirement_id
            for requirement_id, requirement in state.generated_requirements.items()
            if isinstance(requirement, StateSlotRequirement)
        )
        return _result(
            self,
            consumed_requirement_ids=consumed,
            metrics={"state_indexed_rows": len(result.rows)},
        )


class StateFlowConstraint(BaseConstraint):
    name = "state_flow_transition"
    tier = ConstraintTier.HARD
    estimated_cost_class = 3

    def propagate(self, state: PropagationState) -> ConstraintResult:
        removals: dict[tuple[str, str], tuple[list[str], set[str]]] = {}
        requirements = []
        checked = 0
        for piece, domain in state.domains.items():
            for candidate in tuple(domain.candidates.values()):
                checked += 1
                result = evaluate_piece_state_flow(
                    (candidate,),
                    require_local_transitions=True,
                )
                if result.valid:
                    continue
                if piece in state.selected_candidates:
                    return _result(
                        self,
                        ConstraintStatus.UNSAT_PROVEN,
                        rejection_reason="SELECTED_STATE_FLOW_INVALID",
                        conflict_candidate_ids=(candidate.candidate_id,),
                        metrics={
                            "failure_reasons": "|".join(
                                result.failure_reasons
                            )
                        },
                    )
                _merge_removal(
                    removals,
                    piece,
                    candidate.candidate_id,
                    "STATE_FLOW_PATH_INVALID",
                )
        for candidate in state.selected():
            for event in candidate.active_events:
                requirements.append(
                    TransitionRequirement(
                        source_constraint=self.name,
                        sound=True,
                        explanation=(
                            f"{candidate.piece_id} event "
                            f"{event.active_ordinal} carries exact state flow"
                        ),
                        piece_id=candidate.piece_id,
                        move=event.move,
                        required_state_before=event.state_before,
                        required_state_after=event.state_after,
                        allowed_candidate_ids=(
                            candidate.candidate_id,
                        ),
                    )
                )
        consumed = tuple(
            requirement_id
            for requirement_id, requirement in state.generated_requirements.items()
            if isinstance(requirement, StateSlotRequirement)
        )
        return _result(
            self,
            (
                ConstraintStatus.SAT_WITH_REQUIREMENTS
                if requirements or removals
                else ConstraintStatus.SAT
            ),
            requirements=tuple(requirements),
            removals=_removal_rows(removals),
            consumed_requirement_ids=consumed,
            metrics={"candidate_paths_checked": checked},
        )


class IncompatibilityGraphConstraint(BaseConstraint):
    name = "incompatibility_graph"
    tier = ConstraintTier.SOUND_RELAXATION
    estimated_cost_class = 3

    def propagate(self, state: PropagationState) -> ConstraintResult:
        inventory = state.inventory or self.context.inventory
        if inventory is None or not state.selected_candidates:
            return _result(
                self,
                metrics={"deferred": "NO_INVENTORY_OR_SELECTION"},
            )
        graphs = build_incompatibility_graphs(
            state.selected(),
            exact_coloring_node_cap=self.context.coloring_node_cap,
        )
        requirements = []
        cap_moves = []
        for graph in graphs:
            requirements.append(
                ColumnMultiplicityRequirement(
                    source_constraint=self.name,
                    sound=True,
                    explanation=(
                        f"{graph.move} incompatibility clique needs at "
                        f"least {graph.clique_lower_bound} columns"
                    ),
                    move=graph.move,
                    minimum_columns=graph.clique_lower_bound,
                    reason_event_ids=(),
                )
            )
            if graph.clique_lower_bound > int(
                inventory.move_counts.get(graph.move, 0)
            ):
                return _result(
                    self,
                    ConstraintStatus.UNSAT_PROVEN,
                    requirements=tuple(requirements),
                    rejection_reason=(
                        f"CLIQUE_LOWER_BOUND:{graph.move}:"
                        f"{graph.clique_lower_bound}"
                    ),
                    conflict_candidate_ids=tuple(
                        candidate.candidate_id
                        for candidate in state.selected()
                        if candidate.counts[MOVE_INDEX[graph.move]]
                    ),
                )
            if graph.exact_coloring_cap_hit:
                cap_moves.append(graph.move)
        if cap_moves:
            return _result(
                self,
                ConstraintStatus.UNKNOWN_CAP,
                requirements=tuple(requirements),
                unknown_reason="EXACT_COLORING_CAP",
                metrics={"cap_moves": ",".join(cap_moves)},
            )
        return _result(
            self,
            (
                ConstraintStatus.SAT_WITH_REQUIREMENTS
                if requirements
                else ConstraintStatus.SAT
            ),
            requirements=tuple(requirements),
            metrics={"graph_count": len(graphs)},
        )


class OrientationParityConstraint(BaseConstraint):
    name = "orientation_parity"
    tier = ConstraintTier.HARD
    estimated_cost_class = 3

    def propagate(self, state: PropagationState) -> ConstraintResult:
        if not state.complete:
            return _result(self, metrics={"deferred": "INCOMPLETE_SELECTION"})
        conservation = evaluate_state_conservation(state.selected())
        state.artifacts["state_conservation"] = conservation
        if not conservation.valid:
            return _result(
                self,
                ConstraintStatus.UNSAT_PROVEN,
                rejection_reason="ORIENTATION_PARITY_INVALID",
                conflict_candidate_ids=tuple(
                    candidate.candidate_id for candidate in state.selected()
                ),
                metrics={
                    "failure_reasons": "|".join(
                        conservation.failure_reasons
                    )
                },
            )
        return _result(self, metrics={"complete_state_valid": True})


class ExactSlotPackingConstraint(BaseConstraint):
    name = "exact_slot_packing"
    tier = ConstraintTier.HARD
    estimated_cost_class = 4

    def propagate(self, state: PropagationState) -> ConstraintResult:
        inventory = state.inventory or self.context.inventory
        if inventory is None:
            return _result(self, metrics={"deferred": "NO_FIXED_INVENTORY"})
        if not state.complete:
            return _result(self, metrics={"deferred": "INCOMPLETE_SELECTION"})
        if state.domain_size > self.context.exact_packing_domain_threshold:
            return _result(
                self,
                ConstraintStatus.UNKNOWN_CAP,
                unknown_reason="PACKING_DOMAIN_THRESHOLD",
            )
        packing, validations = pack_column_templates(
            state.selected(),
            inventory,
            node_cap=self.context.packing_node_cap,
            prefer_source_columns=False,
        )
        if packing.cap_hit:
            return _result(
                self,
                ConstraintStatus.UNKNOWN_CAP,
                unknown_reason="EXACT_PACKING_NODE_CAP",
                metrics={"explored_nodes": packing.explored_nodes},
            )
        if packing.status != "SAT_PACKED" or not all(
            validation.valid for validation in validations
        ):
            return _result(
                self,
                ConstraintStatus.UNSAT_PROVEN,
                rejection_reason=packing.status,
                conflict_candidate_ids=tuple(
                    candidate.candidate_id for candidate in state.selected()
                ),
                metrics={
                    "explored_nodes": packing.explored_nodes,
                    "template_validation_failures": sum(
                        not validation.valid for validation in validations
                    ),
                },
            )
        state.artifacts["packing"] = packing
        state.artifacts["column_template_validations"] = validations
        return _result(
            self,
            metrics={
                "explored_nodes": packing.explored_nodes,
                "column_count": len(packing.columns),
            },
        )


class PrecedenceConstraint(BaseConstraint):
    name = "precedence"
    tier = ConstraintTier.HARD
    estimated_cost_class = 3

    def propagate(self, state: PropagationState) -> ConstraintResult:
        packing = state.artifacts.get("packing")
        requirements = []
        if packing is None:
            for candidate in state.selected():
                for left, right in zip(
                    candidate.active_events,
                    candidate.active_events[1:],
                ):
                    requirements.append(
                        PrecedenceRequirement(
                            source_constraint=self.name,
                            sound=True,
                            explanation=(
                                f"{candidate.piece_id} active ordinals "
                                f"{left.active_ordinal} < "
                                f"{right.active_ordinal}"
                            ),
                            predecessor_event_id=(
                                f"{candidate.candidate_id}:"
                                f"{left.active_ordinal}"
                            ),
                            successor_event_id=(
                                f"{candidate.candidate_id}:"
                                f"{right.active_ordinal}"
                            ),
                            strict=True,
                        )
                    )
            return _result(
                self,
                (
                    ConstraintStatus.SAT_WITH_REQUIREMENTS
                    if requirements
                    else ConstraintStatus.SAT
                ),
                requirements=tuple(requirements),
                metrics={"deferred": "NO_PACKING_WITNESS"},
            )
        precedence = build_precedence_graph(
            state.selected(),
            packing.columns,
        )
        state.artifacts["precedence"] = precedence
        for left, right in precedence.edges:
            requirements.append(
                PrecedenceRequirement(
                    source_constraint=self.name,
                    sound=True,
                    explanation=f"packed column {left} precedes {right}",
                    predecessor_event_id=left,
                    successor_event_id=right,
                    strict=True,
                )
            )
        consumed = tuple(
            requirement_id
            for requirement_id, requirement in state.generated_requirements.items()
            if isinstance(requirement, TransitionRequirement)
        )
        if precedence.status != "SAT_ACYCLIC":
            return _result(
                self,
                ConstraintStatus.UNSAT_PROVEN,
                requirements=tuple(requirements),
                consumed_requirement_ids=consumed,
                rejection_reason=precedence.status,
                conflict_candidate_ids=tuple(
                    candidate.candidate_id for candidate in state.selected()
                ),
                metrics={
                    "cycle_nodes": ",".join(precedence.cycle_nodes),
                    "ordinal_collapses": len(
                        precedence.same_column_ordinal_collapses
                    ),
                },
            )
        return _result(
            self,
            (
                ConstraintStatus.SAT_WITH_REQUIREMENTS
                if requirements
                else ConstraintStatus.SAT
            ),
            requirements=tuple(requirements),
            consumed_requirement_ids=consumed,
            metrics={"edge_count": len(precedence.edges)},
        )


class LinearizationReplayConstraint(BaseConstraint):
    name = "linearization_and_replay"
    tier = ConstraintTier.HARD
    estimated_cost_class = 6

    def propagate(self, state: PropagationState) -> ConstraintResult:
        packing = state.artifacts.get("packing")
        precedence = state.artifacts.get("precedence")
        if packing is None or precedence is None or not state.complete:
            return _result(
                self,
                metrics={"deferred": "PACKING_OR_PRECEDENCE_MISSING"},
            )
        linearization = linearize_columns_with_state_flow(
            state.selected(),
            packing.columns,
            precedence,
            node_cap=self.context.linearization_node_cap,
            solution_limit=self.context.solution_limit,
        )
        state.artifacts["linearization"] = linearization
        if linearization.cap_hit and not linearization.valid_orders:
            return _result(
                self,
                ConstraintStatus.UNKNOWN_CAP,
                unknown_reason="LINEARIZATION_NODE_CAP",
                metrics={"explored_nodes": linearization.explored_nodes},
            )
        if not linearization.valid_orders:
            return _result(
                self,
                ConstraintStatus.UNSAT_PROVEN,
                rejection_reason=linearization.status,
                conflict_candidate_ids=tuple(
                    candidate.candidate_id for candidate in state.selected()
                ),
                metrics={"explored_nodes": linearization.explored_nodes},
            )
        replay_precedence = replace(
            precedence,
            topological_order=linearization.valid_orders[0],
        )
        replay = concrete_replay(
            state.selected(),
            packing.columns,
            replay_precedence,
        )
        state.artifacts["replay"] = replay
        if not replay.valid:
            return _result(
                self,
                ConstraintStatus.UNSAT_PROVEN,
                rejection_reason="CONCRETE_REPLAY_INVALID",
                conflict_candidate_ids=tuple(
                    candidate.candidate_id for candidate in state.selected()
                ),
                metrics={
                    "replay_mismatches": "|".join(
                        replay.mismatch_reasons
                    )
                },
            )
        consumed = tuple(
            requirement_id
            for requirement_id, requirement in state.generated_requirements.items()
            if isinstance(requirement, PrecedenceRequirement)
        )
        return _result(
            self,
            consumed_requirement_ids=consumed,
            metrics={
                "explored_nodes": linearization.explored_nodes,
                "valid_order_count": len(linearization.valid_orders),
                "replay_attempts": 1,
            },
        )


__all__ = [
    "BaseConstraint",
    "ContractValidationConstraint",
    "DemandBoundsConstraint",
    "MoveMod4Constraint",
    "CornerEdgeEqualityConstraint",
    "MoveMultiplicityConstraint",
    "StateSlotConstraint",
    "StateIndexedResidueConstraint",
    "StateFlowConstraint",
    "IncompatibilityGraphConstraint",
    "OrientationParityConstraint",
    "ExactSlotPackingConstraint",
    "PrecedenceConstraint",
    "LinearizationReplayConstraint",
]
