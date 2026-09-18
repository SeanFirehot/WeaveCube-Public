from __future__ import annotations

"""Deterministic outer-turn column templates for CubeLab v101.4.2."""

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Sequence

from .domino_reduction_short_census import effect_from_word
from .move_demand_balance import (
    MOVE_ORDER,
    ActiveEventDemand,
    ColumnInventory,
    MoveDemandVector,
    PackedColumn,
    PackingResult,
    pack_full_columns,
)
from .pieces import CORNER_ORDER, EDGE_ORDER
from .transformations import PIECES


PIECE_INDEX = {piece: index for index, piece in enumerate(PIECES)}


@dataclass(frozen=True, slots=True)
class TemplateTransition:
    piece_type: str
    before_position: str
    before_orientation: int
    after_position: str
    after_orientation: int


@dataclass(frozen=True, slots=True)
class ColumnTemplate:
    move: str
    corner_before_slots: tuple[str, ...]
    edge_before_slots: tuple[str, ...]
    corner_transitions: tuple[TemplateTransition, ...]
    edge_transitions: tuple[TemplateTransition, ...]

    def row(self) -> dict[str, object]:
        return {
            "move": self.move,
            "corner_before_slots": list(self.corner_before_slots),
            "edge_before_slots": list(self.edge_before_slots),
            "corner_transitions": [
                asdict(value) for value in self.corner_transitions
            ],
            "edge_transitions": [
                asdict(value) for value in self.edge_transitions
            ],
        }


@dataclass(frozen=True, slots=True)
class ColumnTemplateValidation:
    valid: bool
    column_id: str
    move: str
    matched_event_keys: tuple[tuple[str, int], ...]
    failure_reasons: tuple[str, ...]

    def row(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "column_id": self.column_id,
            "move": self.move,
            "matched_event_keys": [
                list(value) for value in self.matched_event_keys
            ],
            "failure_reasons": list(self.failure_reasons),
        }


def _transition(
    move: str,
    piece_type: str,
    position: str,
    orientation: int,
) -> TemplateTransition:
    effect = effect_from_word((move,))
    position_index = PIECE_INDEX[position]
    modulus = 3 if piece_type == "CORNER" else 2
    return TemplateTransition(
        piece_type=piece_type,
        before_position=position,
        before_orientation=orientation,
        after_position=PIECES[effect[position_index]],
        after_orientation=(
            orientation + effect[20 + position_index]
        ) % modulus,
    )


@lru_cache(maxsize=18)
def column_template(move: str) -> ColumnTemplate:
    if move not in MOVE_ORDER:
        raise ValueError(f"unsupported outer move: {move}")
    face = move[0]
    corners = tuple(position for position in CORNER_ORDER if face in position)
    edges = tuple(position for position in EDGE_ORDER if face in position)
    return ColumnTemplate(
        move=move,
        corner_before_slots=corners,
        edge_before_slots=edges,
        corner_transitions=tuple(
            _transition(move, "CORNER", position, orientation)
            for position in corners
            for orientation in range(3)
        ),
        edge_transitions=tuple(
            _transition(move, "EDGE", position, orientation)
            for position in edges
            for orientation in range(2)
        ),
    )


def all_column_templates() -> tuple[ColumnTemplate, ...]:
    return tuple(column_template(move) for move in MOVE_ORDER)


def event_matches_template(event: ActiveEventDemand) -> bool:
    template = column_template(event.move)
    transitions = (
        template.corner_transitions
        if event.piece_type == "CORNER"
        else template.edge_transitions
    )
    before_position = PIECES[event.state_before[0]]
    after_position = PIECES[event.state_after[0]]
    return any(
        value.before_position == before_position
        and value.before_orientation == event.state_before[1]
        and value.after_position == after_position
        and value.after_orientation == event.state_after[1]
        for value in transitions
    )


def validate_packed_column_template(
    column: PackedColumn,
) -> ColumnTemplateValidation:
    template = column_template(column.move)
    failures = []
    for piece_type, events, expected_slots in (
        ("CORNER", column.corner_events, template.corner_before_slots),
        ("EDGE", column.edge_events, template.edge_before_slots),
    ):
        actual_slots = tuple(
            PIECES[event.state_before[0]] for event in events
        )
        if len(events) != 4 or set(actual_slots) != set(expected_slots):
            failures.append(f"{piece_type}:PHYSICAL_SLOT_COVER")
        if len({event.piece_id for event in events}) != len(events):
            failures.append(f"{piece_type}:DUPLICATE_PIECE")
        for event in events:
            if event.move != column.move:
                failures.append(f"{piece_type}:MOVE_MISMATCH:{event.key}")
            if not event_matches_template(event):
                failures.append(
                    f"{piece_type}:TRANSITION_MISMATCH:{event.key}"
                )
    return ColumnTemplateValidation(
        valid=not failures,
        column_id=column.column_id,
        move=column.move,
        matched_event_keys=tuple(
            event.key for event in column.all_events
        ),
        failure_reasons=tuple(failures),
    )


def pack_column_templates(
    candidates: Sequence[MoveDemandVector],
    inventory: ColumnInventory,
    *,
    node_cap: int = 100_000,
    prefer_source_columns: bool = False,
) -> tuple[PackingResult, tuple[ColumnTemplateValidation, ...]]:
    packing = pack_full_columns(
        candidates,
        inventory,
        node_cap=node_cap,
        prefer_source_columns=prefer_source_columns,
    )
    validations = tuple(
        validate_packed_column_template(column)
        for column in packing.columns
    )
    return packing, validations


__all__ = [
    "TemplateTransition",
    "ColumnTemplate",
    "ColumnTemplateValidation",
    "column_template",
    "all_column_templates",
    "event_matches_template",
    "validate_packed_column_template",
    "pack_column_templates",
]
