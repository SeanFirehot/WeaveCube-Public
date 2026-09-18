from __future__ import annotations

"""Move-demand balance and full-column packing for CubeLab.

The module treats every outer face turn as one physical column with exactly
four active corner cells and four active edge cells.  Per-piece candidates are
runtime-state trajectories: demand is counted from their active events, never
from the full source word.

All pruning predicates exposed here are necessary conditions.  Bounded search
exhaustion is reported as ``UNKNOWN_*`` and is never converted to ``UNSAT``.
"""

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence

from .domino_reduction_short_census import (
    IDENTITY_KEY,
    compose_packed,
    effect_from_word,
    state_codes,
)
from .pieces import CORNER_ORDER, EDGE_ORDER
from .transformations import PIECES


MOVE_ORDER = (
    "U", "U'", "U2",
    "D", "D'", "D2",
    "L", "L'", "L2",
    "R", "R'", "R2",
    "F", "F'", "F2",
    "B", "B'", "B2",
)
MOVE_INDEX = {move: index for index, move in enumerate(MOVE_ORDER)}
PIECE_INDEX = {piece: index for index, piece in enumerate(PIECES)}
CORNER_SET = frozenset(CORNER_ORDER)
EDGE_SET = frozenset(EDGE_ORDER)
MOVE_EFFECTS = tuple(effect_from_word((move,)) for move in MOVE_ORDER)


def _state_tuple(code: int) -> tuple[int, int]:
    return int(code % 20), int(code // 20)


def _state_row(state: tuple[int, int]) -> dict[str, object]:
    position, orientation = state
    return {
        "position_index": position,
        "position": PIECES[position],
        "orientation": orientation,
    }


def _candidate_key(candidate_id: str, ordinal: int) -> tuple[str, int]:
    return candidate_id, int(ordinal)


@dataclass(frozen=True, slots=True)
class ActiveEventDemand:
    piece_id: str
    piece_type: str
    candidate_id: str
    active_ordinal: int
    move: str
    state_before: tuple[int, int]
    state_after: tuple[int, int]
    predecessor_event: tuple[str, int] | None
    successor_event: tuple[str, int] | None
    source_column: int | None = None

    @property
    def key(self) -> tuple[str, int]:
        return _candidate_key(self.candidate_id, self.active_ordinal)

    def row(self) -> dict[str, object]:
        payload = asdict(self)
        payload["state_before"] = _state_row(self.state_before)
        payload["state_after"] = _state_row(self.state_after)
        if self.predecessor_event is not None:
            payload["predecessor_event"] = list(self.predecessor_event)
        if self.successor_event is not None:
            payload["successor_event"] = list(self.successor_event)
        return payload


@dataclass(frozen=True, slots=True)
class MoveDemandVector:
    piece_id: str
    piece_type: str
    candidate_id: str
    counts: tuple[int, ...]
    active_length: int
    active_skeleton: tuple[str, ...]
    source: str
    source_word: tuple[str, ...]
    initial_state: tuple[int, int]
    final_state: tuple[int, int]
    active_events: tuple[ActiveEventDemand, ...]

    def __post_init__(self) -> None:
        if len(self.counts) != len(MOVE_ORDER):
            raise ValueError("move-demand counts must use the fixed 18-move order")
        if sum(self.counts) != self.active_length:
            raise ValueError("sum(counts) must equal active skeleton length")
        if len(self.active_skeleton) != self.active_length:
            raise ValueError("active skeleton length mismatch")
        if len(self.active_events) != self.active_length:
            raise ValueError("active event length mismatch")
        if self.piece_type not in {"CORNER", "EDGE"}:
            raise ValueError("piece_type must be CORNER or EDGE")

    def row(self) -> dict[str, object]:
        return {
            "piece_id": self.piece_id,
            "piece_type": self.piece_type,
            "candidate_id": self.candidate_id,
            "counts": {
                move: self.counts[index]
                for index, move in enumerate(MOVE_ORDER)
                if self.counts[index]
            },
            "counts_fixed_order": list(self.counts),
            "move_order": list(MOVE_ORDER),
            "active_length": self.active_length,
            "active_skeleton": list(self.active_skeleton),
            "source": self.source,
            "source_word": list(self.source_word),
            "initial_state": _state_row(self.initial_state),
            "final_state": _state_row(self.final_state),
            "active_events": [event.row() for event in self.active_events],
        }


@dataclass(frozen=True, slots=True)
class GlobalMoveDemand:
    corner_counts: Mapping[str, int]
    edge_counts: Mapping[str, int]
    exact_balanced: bool
    residue_signature: tuple[int, ...]
    inferred_column_inventory: Mapping[str, int] | None
    failure_reasons: tuple[str, ...]

    @property
    def total_active_demand(self) -> int:
        return sum(self.corner_counts.values()) + sum(self.edge_counts.values())

    def row(self) -> dict[str, object]:
        return {
            "corner_counts": dict(self.corner_counts),
            "edge_counts": dict(self.edge_counts),
            "exact_balanced": self.exact_balanced,
            "residue_signature": list(self.residue_signature),
            "inferred_column_inventory": (
                None
                if self.inferred_column_inventory is None
                else dict(self.inferred_column_inventory)
            ),
            "total_active_demand": self.total_active_demand,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True, slots=True)
class ResiduePropagationResult:
    status: str
    current_residue_signature: tuple[int, ...]
    corner_needed_mod4: Mapping[str, int]
    edge_needed_mod4: Mapping[str, int]
    balance_delta: Mapping[str, int]
    surviving_candidate_ids: Mapping[str, tuple[str, ...]]
    pruned_candidate_ids: Mapping[str, tuple[str, ...]]
    explored_states: int
    cap_hit: bool
    exact_completion_count_lower_bound: int

    def row(self) -> dict[str, object]:
        payload = asdict(self)
        payload["current_residue_signature"] = list(
            self.current_residue_signature
        )
        payload["surviving_candidate_ids"] = {
            piece: list(values)
            for piece, values in self.surviving_candidate_ids.items()
        }
        payload["pruned_candidate_ids"] = {
            piece: list(values)
            for piece, values in self.pruned_candidate_ids.items()
        }
        return payload


@dataclass(frozen=True, slots=True)
class ColumnInventory:
    move_counts: Mapping[str, int]
    total_columns: int

    def row(self) -> dict[str, object]:
        return {
            "move_counts": dict(self.move_counts),
            "total_columns": self.total_columns,
        }


@dataclass(frozen=True, slots=True)
class PackedColumn:
    column_id: str
    move: str
    corner_events: tuple[ActiveEventDemand, ...]
    edge_events: tuple[ActiveEventDemand, ...]
    source_column: int | None = None

    @property
    def all_events(self) -> tuple[ActiveEventDemand, ...]:
        return self.corner_events + self.edge_events

    def row(self) -> dict[str, object]:
        return {
            "column_id": self.column_id,
            "move": self.move,
            "source_column": self.source_column,
            "corner_events": [event.row() for event in self.corner_events],
            "edge_events": [event.row() for event in self.edge_events],
        }


@dataclass(frozen=True, slots=True)
class PackingResult:
    status: str
    columns: tuple[PackedColumn, ...]
    failure_reasons: tuple[str, ...]
    explored_nodes: int
    cap_hit: bool
    used_source_column_witness: bool
    face_support_exact_cover: bool
    column_transition_closure: bool

    def row(self) -> dict[str, object]:
        return {
            "status": self.status,
            "columns": [column.row() for column in self.columns],
            "failure_reasons": list(self.failure_reasons),
            "explored_nodes": self.explored_nodes,
            "cap_hit": self.cap_hit,
            "used_source_column_witness": self.used_source_column_witness,
            "face_support_exact_cover": self.face_support_exact_cover,
            "column_transition_closure": self.column_transition_closure,
        }


@dataclass(frozen=True, slots=True)
class PrecedenceResult:
    status: str
    edges: tuple[tuple[str, str], ...]
    topological_order: tuple[str, ...]
    cycle_nodes: tuple[str, ...]
    same_column_ordinal_collapses: tuple[tuple[str, int, int], ...]

    def row(self) -> dict[str, object]:
        return {
            "status": self.status,
            "edges": [list(edge) for edge in self.edges],
            "topological_order": list(self.topological_order),
            "cycle_nodes": list(self.cycle_nodes),
            "same_column_ordinal_collapses": [
                list(value) for value in self.same_column_ordinal_collapses
            ],
        }


@dataclass(frozen=True, slots=True)
class StateConservationResult:
    valid: bool
    initial_valid: bool
    final_valid: bool
    corner_orientation_residue: int | None
    edge_orientation_residue: int | None
    corner_permutation_parity: int | None
    edge_permutation_parity: int | None
    failure_reasons: tuple[str, ...]

    def row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ConcreteReplayResult:
    valid: bool
    global_word: tuple[str, ...]
    final_state_matches: bool
    active_event_accounting_complete: bool
    mismatch_reasons: tuple[str, ...]

    def row(self) -> dict[str, object]:
        payload = asdict(self)
        payload["global_word"] = list(self.global_word)
        payload["mismatch_reasons"] = list(self.mismatch_reasons)
        return payload


@dataclass(frozen=True, slots=True)
class SelectionAnalysis:
    status: str
    demand: GlobalMoveDemand
    inventory: ColumnInventory | None
    packing: PackingResult | None
    precedence: PrecedenceResult | None
    state_conservation: StateConservationResult
    move_multiplicity_lower_bound_satisfied: bool
    replay: ConcreteReplayResult | None
    failure_stage: str | None

    @property
    def replay_valid(self) -> bool:
        return bool(self.replay is not None and self.replay.valid)

    def row(self) -> dict[str, object]:
        return {
            "status": self.status,
            "demand": self.demand.row(),
            "inventory": None if self.inventory is None else self.inventory.row(),
            "packing": None if self.packing is None else self.packing.row(),
            "precedence": (
                None if self.precedence is None else self.precedence.row()
            ),
            "state_conservation": self.state_conservation.row(),
            "move_multiplicity_lower_bound_satisfied": (
                self.move_multiplicity_lower_bound_satisfied
            ),
            "replay": None if self.replay is None else self.replay.row(),
            "failure_stage": self.failure_stage,
        }


def build_candidate_vectors(
    word: Sequence[str],
    *,
    start_key: bytes = IDENTITY_KEY,
    source: str = "CONCRETE_REPLAY",
    selection_id: str = "schedule",
) -> tuple[MoveDemandVector, ...]:
    """Project one concrete global word into twenty runtime-state candidates."""

    tokens = tuple(str(move) for move in word)
    if any(move not in MOVE_INDEX for move in tokens):
        raise ValueError("word contains a move outside the 18 outer turns")
    if len(start_key) != 40:
        raise ValueError("packed start state must contain 40 bytes")

    current = start_key
    rows = [state_codes(current)]
    for move in tokens:
        current = compose_packed(MOVE_EFFECTS[MOVE_INDEX[move]], current)
        rows.append(state_codes(current))

    vectors: list[MoveDemandVector] = []
    for piece_index, piece_id in enumerate(PIECES):
        raw_events: list[tuple[int, str, tuple[int, int], tuple[int, int]]] = []
        for column, move in enumerate(tokens):
            before = _state_tuple(rows[column][piece_index])
            after = _state_tuple(rows[column + 1][piece_index])
            if before != after:
                raw_events.append((column, move, before, after))
        candidate_id = f"{selection_id}:{piece_id}"
        events = []
        for ordinal, (column, move, before, after) in enumerate(raw_events):
            events.append(
                ActiveEventDemand(
                    piece_id=piece_id,
                    piece_type=(
                        "CORNER" if piece_id in CORNER_SET else "EDGE"
                    ),
                    candidate_id=candidate_id,
                    active_ordinal=ordinal,
                    move=move,
                    state_before=before,
                    state_after=after,
                    predecessor_event=(
                        None
                        if ordinal == 0
                        else _candidate_key(candidate_id, ordinal - 1)
                    ),
                    successor_event=(
                        None
                        if ordinal + 1 == len(raw_events)
                        else _candidate_key(candidate_id, ordinal + 1)
                    ),
                    source_column=column,
                )
            )
        counter = Counter(event.move for event in events)
        vectors.append(
            MoveDemandVector(
                piece_id=piece_id,
                piece_type=("CORNER" if piece_id in CORNER_SET else "EDGE"),
                candidate_id=candidate_id,
                counts=tuple(counter[move] for move in MOVE_ORDER),
                active_length=len(events),
                active_skeleton=tuple(event.move for event in events),
                source=source,
                source_word=tokens,
                initial_state=_state_tuple(rows[0][piece_index]),
                final_state=_state_tuple(rows[-1][piece_index]),
                active_events=tuple(events),
            )
        )
    return tuple(vectors)


def aggregate_move_demand(
    candidates: Sequence[MoveDemandVector],
) -> GlobalMoveDemand:
    corner = [0] * len(MOVE_ORDER)
    edge = [0] * len(MOVE_ORDER)
    failures: list[str] = []
    seen_pieces: set[str] = set()
    for candidate in candidates:
        if candidate.piece_id in seen_pieces:
            failures.append(f"DUPLICATE_PIECE:{candidate.piece_id}")
        seen_pieces.add(candidate.piece_id)
        target = corner if candidate.piece_type == "CORNER" else edge
        for index, value in enumerate(candidate.counts):
            target[index] += int(value)

    for index, move in enumerate(MOVE_ORDER):
        if corner[index] % 4:
            failures.append(f"CORNER_MOD4:{move}:{corner[index] % 4}")
        if edge[index] % 4:
            failures.append(f"EDGE_MOD4:{move}:{edge[index] % 4}")
        if corner[index] != edge[index]:
            failures.append(
                f"CORNER_EDGE_DELTA:{move}:{corner[index] - edge[index]}"
            )
    exact = not failures
    inventory = (
        {
            move: corner[index] // 4
            for index, move in enumerate(MOVE_ORDER)
            if corner[index]
        }
        if exact
        else None
    )
    residue = tuple(value % 4 for value in corner)
    residue += tuple(value % 4 for value in edge)
    residue += tuple(
        corner[index] - edge[index] for index in range(len(MOVE_ORDER))
    )
    return GlobalMoveDemand(
        corner_counts={
            move: corner[index] for index, move in enumerate(MOVE_ORDER)
        },
        edge_counts={
            move: edge[index] for index, move in enumerate(MOVE_ORDER)
        },
        exact_balanced=exact,
        residue_signature=residue,
        inferred_column_inventory=inventory,
        failure_reasons=tuple(failures),
    )


def synthesize_column_inventory(
    demand: GlobalMoveDemand,
) -> ColumnInventory:
    if not demand.exact_balanced or demand.inferred_column_inventory is None:
        raise ValueError("column inventory requires exact move-demand balance")
    move_counts = dict(demand.inferred_column_inventory)
    return ColumnInventory(
        move_counts=move_counts,
        total_columns=sum(move_counts.values()),
    )


def _add_counts(
    counts: tuple[int, ...], candidate: MoveDemandVector
) -> tuple[int, ...]:
    split = len(MOVE_ORDER)
    offset = 0 if candidate.piece_type == "CORNER" else split
    values = list(counts)
    for index, value in enumerate(candidate.counts):
        values[offset + index] += int(value)
    return tuple(values)


def _counts_balanced(counts: tuple[int, ...]) -> bool:
    split = len(MOVE_ORDER)
    return all(
        counts[index] == counts[split + index]
        and counts[index] % 4 == 0
        for index in range(split)
    )


def propagate_residue_requirements(
    *,
    selected: Sequence[MoveDemandVector],
    remaining_domains: Mapping[str, Sequence[MoveDemandVector]],
    state_cap: int = 100_000,
) -> ResiduePropagationResult:
    """Exactly filter finite remaining domains, with cap -> UNKNOWN."""

    if state_cap < 1:
        raise ValueError("state_cap must be positive")
    selected_demand = aggregate_move_demand(selected)
    split = len(MOVE_ORDER)
    start_counts = tuple(
        selected_demand.corner_counts[move] for move in MOVE_ORDER
    ) + tuple(selected_demand.edge_counts[move] for move in MOVE_ORDER)
    ordered = tuple(
        (piece, tuple(domain))
        for piece, domain in sorted(remaining_domains.items())
    )
    if any(not domain for _, domain in ordered):
        return ResiduePropagationResult(
            status="UNSAT_EMPTY_DOMAIN",
            current_residue_signature=selected_demand.residue_signature,
            corner_needed_mod4={
                move: (-selected_demand.corner_counts[move]) % 4
                for move in MOVE_ORDER
            },
            edge_needed_mod4={
                move: (-selected_demand.edge_counts[move]) % 4
                for move in MOVE_ORDER
            },
            balance_delta={
                move: (
                    selected_demand.corner_counts[move]
                    - selected_demand.edge_counts[move]
                )
                for move in MOVE_ORDER
            },
            surviving_candidate_ids={piece: () for piece, _ in ordered},
            pruned_candidate_ids={
                piece: tuple(candidate.candidate_id for candidate in domain)
                for piece, domain in ordered
            },
            explored_states=0,
            cap_hit=False,
            exact_completion_count_lower_bound=0,
        )

    explored = 0
    cap_hit = False

    def exists(
        forced_piece: str | None = None,
        forced_id: str | None = None,
    ) -> tuple[bool, bool]:
        nonlocal explored, cap_hit
        frontier = {start_counts}
        for piece, domain in ordered:
            choices = (
                tuple(c for c in domain if c.candidate_id == forced_id)
                if piece == forced_piece
                else domain
            )
            if not choices:
                return False, False
            next_frontier: set[tuple[int, ...]] = set()
            for counts in frontier:
                for candidate in choices:
                    explored += 1
                    if explored > state_cap:
                        cap_hit = True
                        return False, True
                    next_frontier.add(_add_counts(counts, candidate))
            frontier = next_frontier
        return any(_counts_balanced(counts) for counts in frontier), False

    any_completion, unknown = exists()
    surviving: dict[str, tuple[str, ...]] = {}
    pruned: dict[str, tuple[str, ...]] = {}
    if unknown:
        for piece, domain in ordered:
            surviving[piece] = tuple(candidate.candidate_id for candidate in domain)
            pruned[piece] = ()
        status = "UNKNOWN_RESIDUE_CAP"
    elif not any_completion:
        for piece, domain in ordered:
            surviving[piece] = ()
            pruned[piece] = tuple(candidate.candidate_id for candidate in domain)
        status = "UNSAT_RESIDUE_SUPPORT"
    else:
        for piece, domain in ordered:
            keep: list[str] = []
            drop: list[str] = []
            for candidate in domain:
                sat, item_unknown = exists(piece, candidate.candidate_id)
                if item_unknown:
                    keep.append(candidate.candidate_id)
                elif sat:
                    keep.append(candidate.candidate_id)
                else:
                    drop.append(candidate.candidate_id)
            surviving[piece] = tuple(keep)
            pruned[piece] = tuple(drop)
        status = "UNKNOWN_RESIDUE_CAP" if cap_hit else "SAT_RESIDUE_SUPPORT"

    return ResiduePropagationResult(
        status=status,
        current_residue_signature=selected_demand.residue_signature,
        corner_needed_mod4={
            move: (-selected_demand.corner_counts[move]) % 4
            for move in MOVE_ORDER
        },
        edge_needed_mod4={
            move: (-selected_demand.edge_counts[move]) % 4
            for move in MOVE_ORDER
        },
        balance_delta={
            move: (
                selected_demand.corner_counts[move]
                - selected_demand.edge_counts[move]
            )
            for move in MOVE_ORDER
        },
        surviving_candidate_ids=surviving,
        pruned_candidate_ids=pruned,
        explored_states=explored,
        cap_hit=cap_hit,
        exact_completion_count_lower_bound=int(any_completion),
    )


def _face_support(move: str, piece_type: str) -> tuple[int, ...]:
    face = move[0]
    population = CORNER_ORDER if piece_type == "CORNER" else EDGE_ORDER
    return tuple(PIECE_INDEX[position] for position in population if face in position)


def _event_transition_valid(event: ActiveEventDemand) -> bool:
    position, orientation = event.state_before
    effect = MOVE_EFFECTS[MOVE_INDEX[event.move]]
    expected_position = effect[position]
    base = 3 if event.piece_type == "CORNER" else 2
    expected_orientation = (orientation + effect[20 + position]) % base
    return event.state_after == (expected_position, expected_orientation)


def _column_valid(column: PackedColumn) -> tuple[bool, bool, tuple[str, ...]]:
    reasons: list[str] = []
    exact_cover = True
    transition = True
    for piece_type, events in (
        ("CORNER", column.corner_events),
        ("EDGE", column.edge_events),
    ):
        expected = set(_face_support(column.move, piece_type))
        positions = [event.state_before[0] for event in events]
        after_positions = [event.state_after[0] for event in events]
        if len(events) != 4 or set(positions) != expected or len(set(positions)) != 4:
            exact_cover = False
            reasons.append(f"FACE_SUPPORT_EXACT_COVER:{piece_type}")
        if len({event.piece_id for event in events}) != len(events):
            exact_cover = False
            reasons.append(f"DUPLICATE_PHYSICAL_PIECE:{piece_type}")
        if (
            any(not _event_transition_valid(event) for event in events)
            or set(after_positions) != expected
            or len(set(after_positions)) != 4
        ):
            transition = False
            reasons.append(f"COLUMN_TRANSITION_CLOSURE:{piece_type}")
    return exact_cover, transition, tuple(reasons)


def _source_column_packing(
    candidates: Sequence[MoveDemandVector],
    inventory: ColumnInventory,
) -> PackingResult | None:
    words = {candidate.source_word for candidate in candidates}
    if len(words) != 1:
        return None
    source_word = next(iter(words))
    if len(source_word) != inventory.total_columns:
        return None
    if any(event.source_column is None for c in candidates for event in c.active_events):
        return None
    by_column: dict[int, list[ActiveEventDemand]] = defaultdict(list)
    for candidate in candidates:
        for event in candidate.active_events:
            by_column[int(event.source_column)].append(event)
    if set(by_column) != set(range(len(source_word))):
        return None
    columns: list[PackedColumn] = []
    failures: list[str] = []
    exact = True
    transition = True
    for source_column, move in enumerate(source_word):
        events = by_column[source_column]
        corners = tuple(
            sorted(
                (event for event in events if event.piece_type == "CORNER"),
                key=lambda event: PIECE_INDEX[event.piece_id],
            )
        )
        edges = tuple(
            sorted(
                (event for event in events if event.piece_type == "EDGE"),
                key=lambda event: PIECE_INDEX[event.piece_id],
            )
        )
        if any(event.move != move for event in events):
            failures.append(f"SOURCE_MOVE_MISMATCH:{source_column}")
        column = PackedColumn(
            column_id=f"S{source_column:03d}",
            move=move,
            corner_events=corners,
            edge_events=edges,
            source_column=source_column,
        )
        cover_ok, transition_ok, reasons = _column_valid(column)
        exact &= cover_ok
        transition &= transition_ok
        failures.extend(
            f"{column.column_id}:{reason}" for reason in reasons
        )
        columns.append(column)
    status = "SAT_PACKED" if not failures else "UNSAT_SOURCE_PACKING"
    return PackingResult(
        status=status,
        columns=tuple(columns),
        failure_reasons=tuple(failures),
        explored_nodes=len(columns),
        cap_hit=False,
        used_source_column_witness=True,
        face_support_exact_cover=exact,
        column_transition_closure=transition,
    )


def _partition_face_events(
    *,
    events: Sequence[ActiveEventDemand],
    move: str,
    piece_type: str,
    column_count: int,
    node_cap: int,
    explored: list[int],
) -> tuple[tuple[tuple[ActiveEventDemand, ...], ...] | None, bool]:
    support = _face_support(move, piece_type)
    buckets: dict[int, list[ActiveEventDemand]] = {
        position: [] for position in support
    }
    for event in events:
        if event.state_before[0] not in buckets:
            return None, False
        buckets[event.state_before[0]].append(event)
    if any(len(values) != column_count for values in buckets.values()):
        return None, False
    for values in buckets.values():
        values.sort(
            key=lambda event: (
                event.source_column if event.source_column is not None else 9999,
                event.piece_id,
                event.active_ordinal,
                event.candidate_id,
            )
        )

    used: set[tuple[str, int]] = set()
    groups: list[tuple[ActiveEventDemand, ...]] = []

    def visit() -> tuple[tuple[tuple[ActiveEventDemand, ...], ...] | None, bool]:
        if len(groups) == column_count:
            return tuple(groups), False
        anchor = next(
            (event for event in buckets[support[0]] if event.key not in used),
            None,
        )
        if anchor is None:
            return None, False
        choices = []
        for position in support[1:]:
            available = tuple(
                event for event in buckets[position] if event.key not in used
            )
            if not available:
                return None, False
            choices.append(available)
        for tail in product(*choices):
            explored[0] += 1
            if explored[0] > node_cap:
                return None, True
            group = (anchor,) + tuple(tail)
            if len({event.key for event in group}) != 4:
                continue
            if len({event.piece_id for event in group}) != 4:
                continue
            if any(not _event_transition_valid(event) for event in group):
                continue
            if set(event.state_after[0] for event in group) != set(support):
                continue
            for event in group:
                used.add(event.key)
            groups.append(group)
            result, capped = visit()
            if result is not None or capped:
                return result, capped
            groups.pop()
            for event in group:
                used.remove(event.key)
        return None, False

    return visit()


def pack_full_columns(
    candidates: Sequence[MoveDemandVector],
    inventory: ColumnInventory,
    *,
    node_cap: int = 100_000,
    prefer_source_columns: bool = True,
) -> PackingResult:
    """Pack exact move demands into physical 4C+4E columns."""

    if node_cap < 1:
        raise ValueError("node_cap must be positive")
    if prefer_source_columns:
        source = _source_column_packing(candidates, inventory)
        if source is not None:
            return source

    events_by_move_type: dict[tuple[str, str], list[ActiveEventDemand]] = (
        defaultdict(list)
    )
    for candidate in candidates:
        for event in candidate.active_events:
            events_by_move_type[(event.move, event.piece_type)].append(event)

    explored = [0]
    columns: list[PackedColumn] = []
    failures: list[str] = []
    cap_hit = False
    for move in MOVE_ORDER:
        count = int(inventory.move_counts.get(move, 0))
        if count == 0:
            continue
        corner_groups, corner_cap = _partition_face_events(
            events=events_by_move_type[(move, "CORNER")],
            move=move,
            piece_type="CORNER",
            column_count=count,
            node_cap=node_cap,
            explored=explored,
        )
        edge_groups, edge_cap = _partition_face_events(
            events=events_by_move_type[(move, "EDGE")],
            move=move,
            piece_type="EDGE",
            column_count=count,
            node_cap=node_cap,
            explored=explored,
        )
        cap_hit |= corner_cap or edge_cap
        if corner_groups is None or edge_groups is None:
            failures.append(f"NO_4C4E_EXACT_COVER:{move}")
            continue
        for index, (corner_group, edge_group) in enumerate(
            zip(corner_groups, edge_groups)
        ):
            columns.append(
                PackedColumn(
                    column_id=f"M{MOVE_INDEX[move]:02d}-{index:03d}",
                    move=move,
                    corner_events=corner_group,
                    edge_events=edge_group,
                )
            )
    if cap_hit:
        status = "UNKNOWN_PACKING_CAP"
    elif failures:
        status = "UNSAT_SLOT_PACKING"
    else:
        status = "SAT_PACKED"
    exact = False
    transition = False
    if status == "SAT_PACKED":
        checks = [_column_valid(column) for column in columns]
        exact = all(item[0] for item in checks)
        transition = all(item[1] for item in checks)
        for column, item in zip(columns, checks):
            failures.extend(
                f"{column.column_id}:{reason}" for reason in item[2]
            )
        if failures:
            status = "UNSAT_COLUMN_SEMANTICS"
    return PackingResult(
        status=status,
        columns=tuple(columns),
        failure_reasons=tuple(failures),
        explored_nodes=explored[0],
        cap_hit=cap_hit,
        used_source_column_witness=False,
        face_support_exact_cover=exact,
        column_transition_closure=transition,
    )


def build_precedence_graph(
    candidates: Sequence[MoveDemandVector],
    columns: Sequence[PackedColumn],
) -> PrecedenceResult:
    event_to_column: dict[tuple[str, int], str] = {}
    for column in columns:
        for event in column.all_events:
            if event.key in event_to_column:
                raise ValueError(f"event packed twice: {event.key}")
            event_to_column[event.key] = column.column_id
    edges: set[tuple[str, str]] = set()
    collapses: list[tuple[str, int, int]] = []
    for candidate in candidates:
        for left, right in zip(candidate.active_events, candidate.active_events[1:]):
            left_column = event_to_column.get(left.key)
            right_column = event_to_column.get(right.key)
            if left_column is None or right_column is None:
                continue
            if left_column == right_column:
                collapses.append(
                    (
                        candidate.candidate_id,
                        left.active_ordinal,
                        right.active_ordinal,
                    )
                )
            else:
                edges.add((left_column, right_column))

    nodes = tuple(column.column_id for column in columns)
    outgoing: dict[str, set[str]] = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for left, right in edges:
        if right not in outgoing[left]:
            outgoing[left].add(right)
            indegree[right] += 1
    source_order = {
        column.column_id: (
            column.source_column
            if column.source_column is not None
            else len(columns) + index
        )
        for index, column in enumerate(columns)
    }
    ready = sorted(
        (node for node in nodes if indegree[node] == 0),
        key=lambda node: (source_order[node], node),
    )
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for target in sorted(outgoing[node]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort(key=lambda item: (source_order[item], item))
    cycle_nodes = tuple(sorted(node for node in nodes if indegree[node] > 0))
    if collapses:
        status = "UNSAT_ORDINAL_COLLAPSE"
    elif cycle_nodes:
        status = "UNSAT_PRECEDENCE_CYCLE"
    else:
        status = "SAT_ACYCLIC"
    return PrecedenceResult(
        status=status,
        edges=tuple(sorted(edges)),
        topological_order=tuple(order),
        cycle_nodes=cycle_nodes,
        same_column_ordinal_collapses=tuple(collapses),
    )


def _permutation_parity(values: Sequence[int]) -> int:
    inversions = sum(
        values[left] > values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    return inversions & 1


def _state_legality(
    states: Mapping[str, tuple[int, int]],
    label: str,
) -> tuple[bool, int | None, int | None, int | None, int | None, list[str]]:
    reasons: list[str] = []
    corner_positions = [states[piece][0] for piece in CORNER_ORDER]
    edge_positions = [states[piece][0] for piece in EDGE_ORDER]
    corner_bijection = sorted(corner_positions) == list(range(8))
    edge_bijection = sorted(edge_positions) == list(range(8, 20))
    if not corner_bijection:
        reasons.append(f"{label}:CORNER_POSITION_NOT_BIJECTIVE")
    if not edge_bijection:
        reasons.append(f"{label}:EDGE_POSITION_NOT_BIJECTIVE")
    corner_orientation = sum(states[piece][1] for piece in CORNER_ORDER) % 3
    edge_orientation = sum(states[piece][1] for piece in EDGE_ORDER) % 2
    if corner_orientation:
        reasons.append(f"{label}:CORNER_ORIENTATION_MOD3")
    if edge_orientation:
        reasons.append(f"{label}:EDGE_ORIENTATION_MOD2")
    corner_parity = (
        _permutation_parity(corner_positions) if corner_bijection else None
    )
    edge_parity = (
        _permutation_parity([value - 8 for value in edge_positions])
        if edge_bijection
        else None
    )
    if (
        corner_parity is not None
        and edge_parity is not None
        and corner_parity != edge_parity
    ):
        reasons.append(f"{label}:CORNER_EDGE_PARITY_MISMATCH")
    return (
        not reasons,
        corner_orientation,
        edge_orientation,
        corner_parity,
        edge_parity,
        reasons,
    )


def evaluate_state_conservation(
    candidates: Sequence[MoveDemandVector],
) -> StateConservationResult:
    by_piece = {candidate.piece_id: candidate for candidate in candidates}
    missing = [piece for piece in PIECES if piece not in by_piece]
    if missing or len(by_piece) != 20:
        return StateConservationResult(
            valid=False,
            initial_valid=False,
            final_valid=False,
            corner_orientation_residue=None,
            edge_orientation_residue=None,
            corner_permutation_parity=None,
            edge_permutation_parity=None,
            failure_reasons=(
                "SELECTION_DOES_NOT_CONTAIN_EXACTLY_20_UNIQUE_PIECES",
            ),
        )
    initial = {
        piece: by_piece[piece].initial_state for piece in PIECES
    }
    final = {piece: by_piece[piece].final_state for piece in PIECES}
    initial_row = _state_legality(initial, "INITIAL")
    final_row = _state_legality(final, "FINAL")
    return StateConservationResult(
        valid=initial_row[0] and final_row[0],
        initial_valid=initial_row[0],
        final_valid=final_row[0],
        corner_orientation_residue=final_row[1],
        edge_orientation_residue=final_row[2],
        corner_permutation_parity=final_row[3],
        edge_permutation_parity=final_row[4],
        failure_reasons=tuple(initial_row[5] + final_row[5]),
    )


def move_multiplicity_lower_bound_satisfied(
    candidates: Sequence[MoveDemandVector],
    inventory: ColumnInventory,
) -> bool:
    for index, move in enumerate(MOVE_ORDER):
        maximum = max(
            (candidate.counts[index] for candidate in candidates),
            default=0,
        )
        if int(inventory.move_counts.get(move, 0)) < maximum:
            return False
    return True


def _packed_state_from_candidates(
    candidates: Sequence[MoveDemandVector],
    *,
    final: bool,
) -> bytes:
    by_piece = {candidate.piece_id: candidate for candidate in candidates}
    payload = bytearray(40)
    for piece_index, piece in enumerate(PIECES):
        state = (
            by_piece[piece].final_state
            if final
            else by_piece[piece].initial_state
        )
        payload[piece_index] = state[0]
        payload[20 + piece_index] = state[1]
    return bytes(payload)


def concrete_replay(
    candidates: Sequence[MoveDemandVector],
    columns: Sequence[PackedColumn],
    precedence: PrecedenceResult,
) -> ConcreteReplayResult:
    if precedence.status != "SAT_ACYCLIC":
        return ConcreteReplayResult(
            valid=False,
            global_word=(),
            final_state_matches=False,
            active_event_accounting_complete=False,
            mismatch_reasons=("PRECEDENCE_NOT_SAT",),
        )
    by_id = {column.column_id: column for column in columns}
    ordered = tuple(by_id[column_id] for column_id in precedence.topological_order)
    word = tuple(column.move for column in ordered)
    by_piece = {candidate.piece_id: candidate for candidate in candidates}
    if set(by_piece) != set(PIECES):
        return ConcreteReplayResult(
            valid=False,
            global_word=word,
            final_state_matches=False,
            active_event_accounting_complete=False,
            mismatch_reasons=("SELECTION_NOT_20_UNIQUE_PIECES",),
        )
    current = _packed_state_from_candidates(candidates, final=False)
    reasons: list[str] = []
    seen_events: set[tuple[str, int]] = set()
    for column_index, column in enumerate(ordered):
        after = compose_packed(MOVE_EFFECTS[MOVE_INDEX[column.move]], current)
        before_codes = state_codes(current)
        after_codes = state_codes(after)
        assigned = {event.piece_id: event for event in column.all_events}
        for piece_index, piece in enumerate(PIECES):
            before_state = _state_tuple(before_codes[piece_index])
            after_state = _state_tuple(after_codes[piece_index])
            changed = before_state != after_state
            event = assigned.get(piece)
            if changed != (event is not None):
                reasons.append(
                    f"C{column_index}:{piece}:ACTIVE_ASSIGNMENT_MISMATCH"
                )
                continue
            if event is not None:
                seen_events.add(event.key)
                if event.move != column.move:
                    reasons.append(f"C{column_index}:{piece}:MOVE_MISMATCH")
                if event.state_before != before_state:
                    reasons.append(
                        f"C{column_index}:{piece}:STATE_BEFORE_MISMATCH"
                    )
                if event.state_after != after_state:
                    reasons.append(
                        f"C{column_index}:{piece}:STATE_AFTER_MISMATCH"
                    )
        current = after
    expected_events = {
        event.key for candidate in candidates for event in candidate.active_events
    }
    accounting = seen_events == expected_events
    if not accounting:
        reasons.append("ACTIVE_EVENT_ACCOUNTING_INCOMPLETE")
    target = _packed_state_from_candidates(candidates, final=True)
    final_match = current == target
    if not final_match:
        reasons.append("FINAL_STATE_MISMATCH")
    return ConcreteReplayResult(
        valid=not reasons,
        global_word=word,
        final_state_matches=final_match,
        active_event_accounting_complete=accounting,
        mismatch_reasons=tuple(reasons[:200]),
    )


def analyze_candidate_selection(
    candidates: Sequence[MoveDemandVector],
    *,
    packing_node_cap: int = 100_000,
    prefer_source_columns: bool = True,
) -> SelectionAnalysis:
    """Run the necessary-condition cascade and concrete replay."""

    candidates = tuple(candidates)
    demand = aggregate_move_demand(candidates)
    conservation = evaluate_state_conservation(candidates)
    if not demand.exact_balanced:
        return SelectionAnalysis(
            status="UNSAT",
            demand=demand,
            inventory=None,
            packing=None,
            precedence=None,
            state_conservation=conservation,
            move_multiplicity_lower_bound_satisfied=False,
            replay=None,
            failure_stage="MOVE_DEMAND_BALANCE",
        )
    inventory = synthesize_column_inventory(demand)
    multiplicity = move_multiplicity_lower_bound_satisfied(
        candidates, inventory
    )
    if not multiplicity:
        return SelectionAnalysis(
            status="UNSAT",
            demand=demand,
            inventory=inventory,
            packing=None,
            precedence=None,
            state_conservation=conservation,
            move_multiplicity_lower_bound_satisfied=False,
            replay=None,
            failure_stage="MOVE_MULTIPLICITY_LOWER_BOUND",
        )
    if not conservation.valid:
        return SelectionAnalysis(
            status="UNSAT",
            demand=demand,
            inventory=inventory,
            packing=None,
            precedence=None,
            state_conservation=conservation,
            move_multiplicity_lower_bound_satisfied=True,
            replay=None,
            failure_stage="ORIENTATION_PERMUTATION_CONSERVATION",
        )
    packing = pack_full_columns(
        candidates,
        inventory,
        node_cap=packing_node_cap,
        prefer_source_columns=prefer_source_columns,
    )
    if packing.status.startswith("UNKNOWN"):
        return SelectionAnalysis(
            status="UNKNOWN",
            demand=demand,
            inventory=inventory,
            packing=packing,
            precedence=None,
            state_conservation=conservation,
            move_multiplicity_lower_bound_satisfied=True,
            replay=None,
            failure_stage="PACKING_CAP",
        )
    if packing.status != "SAT_PACKED":
        return SelectionAnalysis(
            status="UNSAT",
            demand=demand,
            inventory=inventory,
            packing=packing,
            precedence=None,
            state_conservation=conservation,
            move_multiplicity_lower_bound_satisfied=True,
            replay=None,
            failure_stage="SLOT_PACKING",
        )
    precedence = build_precedence_graph(candidates, packing.columns)
    if precedence.status != "SAT_ACYCLIC":
        return SelectionAnalysis(
            status="UNSAT",
            demand=demand,
            inventory=inventory,
            packing=packing,
            precedence=precedence,
            state_conservation=conservation,
            move_multiplicity_lower_bound_satisfied=True,
            replay=None,
            failure_stage="PRECEDENCE",
        )
    replay = concrete_replay(candidates, packing.columns, precedence)
    return SelectionAnalysis(
        status="SAT" if replay.valid else "UNSAT",
        demand=demand,
        inventory=inventory,
        packing=packing,
        precedence=precedence,
        state_conservation=conservation,
        move_multiplicity_lower_bound_satisfied=True,
        replay=replay,
        failure_stage=None if replay.valid else "CONCRETE_REPLAY",
    )


def exact_word_inventory(word: Sequence[str]) -> ColumnInventory:
    counts = Counter(str(move) for move in word)
    return ColumnInventory(
        move_counts={
            move: counts[move] for move in MOVE_ORDER if counts[move]
        },
        total_columns=len(tuple(word)),
    )


__all__ = [
    "MOVE_ORDER",
    "MOVE_INDEX",
    "ActiveEventDemand",
    "MoveDemandVector",
    "GlobalMoveDemand",
    "ResiduePropagationResult",
    "ColumnInventory",
    "PackedColumn",
    "PackingResult",
    "PrecedenceResult",
    "StateConservationResult",
    "ConcreteReplayResult",
    "SelectionAnalysis",
    "build_candidate_vectors",
    "aggregate_move_demand",
    "propagate_residue_requirements",
    "synthesize_column_inventory",
    "pack_full_columns",
    "build_precedence_graph",
    "evaluate_state_conservation",
    "move_multiplicity_lower_bound_satisfied",
    "concrete_replay",
    "analyze_candidate_selection",
    "exact_word_inventory",
]
