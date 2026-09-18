from __future__ import annotations

"""Piece-path and inter-column state-flow conservation for v101.4.2."""

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .column_transition_templates import event_matches_template
from .domino_reduction_short_census import (
    compose_packed,
    effect_from_word,
    state_codes,
)
from .move_demand_balance import (
    MOVE_INDEX,
    MOVE_ORDER,
    ActiveEventDemand,
    MoveDemandVector,
    PackedColumn,
)
from .transformations import PIECES


MOVE_EFFECTS = tuple(effect_from_word((move,)) for move in MOVE_ORDER)


@dataclass(frozen=True, slots=True)
class StateFlowEquation:
    piece_id: str
    state: tuple[int, int]
    inflow: int
    outflow: int
    expected_out_minus_in: int
    actual_out_minus_in: int

    def row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PieceStateFlowResult:
    status: str
    valid: bool
    local_transitions_valid: bool
    ordered_path_valid: bool
    equation_balance_valid: bool
    boundary_valid: bool
    equations: tuple[StateFlowEquation, ...]
    failure_reasons: tuple[str, ...]

    def row(self) -> dict[str, object]:
        return {
            "status": self.status,
            "valid": self.valid,
            "local_transitions_valid": self.local_transitions_valid,
            "ordered_path_valid": self.ordered_path_valid,
            "equation_balance_valid": self.equation_balance_valid,
            "boundary_valid": self.boundary_valid,
            "equations": [equation.row() for equation in self.equations],
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True, slots=True)
class InterColumnFlowStep:
    column_id: str
    move: str
    before_states: tuple[tuple[int, int], ...]
    after_states: tuple[tuple[int, int], ...]

    def row(self) -> dict[str, object]:
        return {
            "column_id": self.column_id,
            "move": self.move,
            "before_states": [list(value) for value in self.before_states],
            "after_states": [list(value) for value in self.after_states],
        }


@dataclass(frozen=True, slots=True)
class InterColumnStateFlowResult:
    status: str
    valid: bool
    order: tuple[str, ...]
    steps: tuple[InterColumnFlowStep, ...]
    final_state_matches: bool
    active_event_accounting_complete: bool
    failure_reasons: tuple[str, ...]

    def row(self) -> dict[str, object]:
        return {
            "status": self.status,
            "valid": self.valid,
            "order": list(self.order),
            "steps": [step.row() for step in self.steps],
            "final_state_matches": self.final_state_matches,
            "active_event_accounting_complete": (
                self.active_event_accounting_complete
            ),
            "failure_reasons": list(self.failure_reasons),
        }


def _local_transition_valid(event: ActiveEventDemand) -> bool:
    return event_matches_template(event)


def evaluate_piece_state_flow(
    candidates: Sequence[MoveDemandVector],
    *,
    require_local_transitions: bool = False,
) -> PieceStateFlowResult:
    """Validate each selected candidate as one conserved ordered state path."""

    failures = []
    equations = []
    local_valid = True
    ordered_valid = True
    boundary_valid = True
    equation_valid = True
    seen_pieces = set()

    for candidate in candidates:
        if candidate.piece_id in seen_pieces:
            failures.append(f"DUPLICATE_PIECE:{candidate.piece_id}")
        seen_pieces.add(candidate.piece_id)
        events = tuple(
            sorted(
                candidate.active_events,
                key=lambda event: event.active_ordinal,
            )
        )
        if tuple(event.active_ordinal for event in events) != tuple(
            range(len(events))
        ):
            ordered_valid = False
            failures.append(
                f"ORDINAL_DOMAIN:{candidate.candidate_id}"
            )
        for event in events:
            if not _local_transition_valid(event):
                local_valid = False
                if require_local_transitions:
                    failures.append(f"LOCAL_TRANSITION:{event.key}")
        if events:
            if candidate.initial_state != events[0].state_before:
                boundary_valid = False
                failures.append(
                    f"INITIAL_SUPPLY:{candidate.candidate_id}"
                )
            for left, right in zip(events, events[1:]):
                if left.state_after != right.state_before:
                    ordered_valid = False
                    failures.append(
                        "CHAIN_DISCONNECT:"
                        f"{candidate.candidate_id}:"
                        f"{left.active_ordinal}->{right.active_ordinal}"
                    )
            if events[-1].state_after != candidate.final_state:
                boundary_valid = False
                failures.append(
                    f"FINAL_DEMAND:{candidate.candidate_id}"
                )
        elif candidate.initial_state != candidate.final_state:
            boundary_valid = False
            failures.append(f"EMPTY_PATH_BOUNDARY:{candidate.candidate_id}")

        inflow: Counter[tuple[int, int]] = Counter()
        outflow: Counter[tuple[int, int]] = Counter()
        for event in events:
            outflow[event.state_before] += 1
            inflow[event.state_after] += 1
        states = (
            set(inflow)
            | set(outflow)
            | {candidate.initial_state, candidate.final_state}
        )
        for state in sorted(states):
            expected = int(state == candidate.initial_state) - int(
                state == candidate.final_state
            )
            actual = outflow[state] - inflow[state]
            equations.append(
                StateFlowEquation(
                    piece_id=candidate.piece_id,
                    state=state,
                    inflow=inflow[state],
                    outflow=outflow[state],
                    expected_out_minus_in=expected,
                    actual_out_minus_in=actual,
                )
            )
            if actual != expected:
                equation_valid = False
                failures.append(
                    "FLOW_EQUATION:"
                    f"{candidate.candidate_id}:{state}:{actual}!={expected}"
                )

    valid = (
        ordered_valid
        and equation_valid
        and boundary_valid
        and (local_valid or not require_local_transitions)
    )
    return PieceStateFlowResult(
        status=(
            "SAT_STATE_FLOW" if valid else "UNSAT_STATE_FLOW"
        ),
        valid=valid,
        local_transitions_valid=local_valid,
        ordered_path_valid=ordered_valid,
        equation_balance_valid=equation_valid,
        boundary_valid=boundary_valid,
        equations=tuple(equations),
        failure_reasons=tuple(dict.fromkeys(failures)),
    )


def _packed_state(
    candidates: Sequence[MoveDemandVector],
    *,
    final: bool,
) -> bytes:
    by_piece = {candidate.piece_id: candidate for candidate in candidates}
    if set(by_piece) != set(PIECES):
        raise ValueError("state flow requires exactly twenty piece candidates")
    payload = bytearray(40)
    for index, piece in enumerate(PIECES):
        state = (
            by_piece[piece].final_state
            if final
            else by_piece[piece].initial_state
        )
        payload[index] = state[0]
        payload[20 + index] = state[1]
    return bytes(payload)


def evaluate_intercolumn_state_flow(
    candidates: Sequence[MoveDemandVector],
    columns: Sequence[PackedColumn],
    order: Sequence[str],
) -> InterColumnStateFlowResult:
    """Replay a proposed column order while matching every event cell."""

    by_id = {column.column_id: column for column in columns}
    order = tuple(order)
    failures = []
    if len(order) != len(columns) or set(order) != set(by_id):
        return InterColumnStateFlowResult(
            status="UNSAT_COLUMN_ORDER_DOMAIN",
            valid=False,
            order=order,
            steps=(),
            final_state_matches=False,
            active_event_accounting_complete=False,
            failure_reasons=("ORDER_IS_NOT_A_COLUMN_PERMUTATION",),
        )
    current = _packed_state(candidates, final=False)
    seen: set[tuple[str, int]] = set()
    steps = []
    for ordinal, column_id in enumerate(order):
        column = by_id[column_id]
        before_codes = state_codes(current)
        after = compose_packed(
            MOVE_EFFECTS[MOVE_INDEX[column.move]],
            current,
        )
        after_codes = state_codes(after)
        assigned = {event.piece_id: event for event in column.all_events}
        if len(assigned) != len(column.all_events):
            failures.append(f"C{ordinal}:DUPLICATE_PIECE_ASSIGNMENT")
        before_states = []
        after_states = []
        for piece_index, piece in enumerate(PIECES):
            before_state = (
                int(before_codes[piece_index] % 20),
                int(before_codes[piece_index] // 20),
            )
            after_state = (
                int(after_codes[piece_index] % 20),
                int(after_codes[piece_index] // 20),
            )
            before_states.append(before_state)
            after_states.append(after_state)
            event = assigned.get(piece)
            changed = before_state != after_state
            if changed != (event is not None):
                failures.append(
                    f"C{ordinal}:{piece}:ACTIVE_IDENTITY_MISMATCH"
                )
                continue
            if event is not None:
                seen.add(event.key)
                if event.move != column.move:
                    failures.append(
                        f"C{ordinal}:{piece}:MOVE_MISMATCH"
                    )
                if event.state_before != before_state:
                    failures.append(
                        f"C{ordinal}:{piece}:STATE_SUPPLY_MISMATCH"
                    )
                if event.state_after != after_state:
                    failures.append(
                        f"C{ordinal}:{piece}:STATE_DEMAND_MISMATCH"
                    )
        steps.append(
            InterColumnFlowStep(
                column_id=column_id,
                move=column.move,
                before_states=tuple(before_states),
                after_states=tuple(after_states),
            )
        )
        current = after

    expected_events = {
        event.key
        for candidate in candidates
        for event in candidate.active_events
    }
    accounting = seen == expected_events
    if not accounting:
        failures.append("ACTIVE_EVENT_ACCOUNTING_INCOMPLETE")
    final_match = current == _packed_state(candidates, final=True)
    if not final_match:
        failures.append("FINAL_STATE_DEMAND_MISMATCH")
    return InterColumnStateFlowResult(
        status=(
            "SAT_INTERCOLUMN_STATE_FLOW"
            if not failures
            else "UNSAT_INTERCOLUMN_STATE_FLOW"
        ),
        valid=not failures,
        order=order,
        steps=tuple(steps),
        final_state_matches=final_match,
        active_event_accounting_complete=accounting,
        failure_reasons=tuple(failures[:300]),
    )


__all__ = [
    "StateFlowEquation",
    "PieceStateFlowResult",
    "InterColumnFlowStep",
    "InterColumnStateFlowResult",
    "evaluate_piece_state_flow",
    "evaluate_intercolumn_state_flow",
]
