from __future__ import annotations

"""Fresh candidate portfolio balance grounding for CubeLab v101.4.1.

The independent search in this module consumes only:

* a disclosed scramble,
* a DR axis,
* the one-piece runtime transition graphs, and
* the public move-demand/full-column constraints.

It deliberately does not accept a reference solution.  Reference comparison
is implemented by the grounding/report layer after the portfolio and the
independent result have been sealed.
"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from enum import Enum
from hashlib import sha256
from itertools import combinations_with_replacement
import json
from time import perf_counter
from typing import Iterable, Mapping, Sequence

from .cubie_effect import CubieEffect
from .domino_reduction import DRAxis, diagnose_sequence
from .domino_reduction_short_census import effect_from_word
from .move_demand_balance import (
    MOVE_INDEX,
    MOVE_ORDER,
    ActiveEventDemand,
    ColumnInventory,
    MoveDemandVector,
    SelectionAnalysis,
    analyze_candidate_selection,
)
from .piece_route_graph import FACE_MOVES, PieceState, SinglePieceRouteGraph
from .pieces import CORNER_ORDER, EDGE_ORDER
from .rotations import CUBE_ROTATIONS, CubeRotation
from .solution_atlas_composition_v80 import piece_state
from .transformations import PIECES, from_sequence


SCHEMA = "cubelab.fresh-candidate-balance.v101.4.1"
FIXED_SEED = 20260725
FIXED_SCRAMBLE = ("B'", "F", "U2", "R'", "L", "D", "R", "B")
FIXED_DR_AXIS = DRAxis.FB
CORNER_SET = frozenset(CORNER_ORDER)
EDGE_SET = frozenset(EDGE_ORDER)
UD_SLICE_EDGES = frozenset({"FR", "BR", "BL", "FL"})
PIECE_INDEX = {piece: index for index, piece in enumerate(PIECES)}
INVERSE_MOVE = {
    move: (
        move
        if move.endswith("2")
        else move[0]
        if move.endswith("'")
        else move + "'"
    )
    for move in MOVE_ORDER
}


def _stable_digest(value: object, length: int = 20) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:length]


def _piece_type(piece: str) -> str:
    return "CORNER" if piece in CORNER_SET else "EDGE"


def _state_tuple(state: PieceState) -> tuple[int, int]:
    return PIECE_INDEX[state.position], int(state.orientation)


def _piece_state(value: tuple[int, int]) -> PieceState:
    return PieceState(PIECES[value[0]], int(value[1]))


def normalizing_rotation(axis: DRAxis | str) -> CubeRotation:
    """Return the deterministic rotation that maps ``axis`` onto UD."""

    axis = DRAxis(axis)
    source = {
        DRAxis.UD: frozenset({"U", "D"}),
        DRAxis.FB: frozenset({"F", "B"}),
        DRAxis.RL: frozenset({"R", "L"}),
    }[axis]
    matches = [
        rotation
        for rotation in CUBE_ROTATIONS
        if rotation.map_face_set(source) == frozenset({"U", "D"})
    ]
    if not matches:
        raise AssertionError(f"no rotation normalizes {axis.value}")
    return min(matches, key=lambda item: item.key)


def denormalize_word(
    word: Sequence[str],
    rotation: CubeRotation,
) -> tuple[str, ...]:
    inverse_faces = {
        normalized: original
        for original, normalized in rotation.face_map.items()
    }
    return tuple(inverse_faces[move[0]] + move[1:] for move in word)


def _target_state_allowed(piece: str, state: PieceState) -> bool:
    """Per-piece UD-DR target domain after axis normalization."""

    if piece in CORNER_SET:
        return state.orientation == 0 and state.position in CORNER_SET
    same_slice_class = (
        (piece in UD_SLICE_EDGES)
        == (state.position in UD_SLICE_EDGES)
    )
    return (
        state.orientation == 0
        and state.position in EDGE_SET
        and same_slice_class
    )


class CandidateContractType(str, Enum):
    EXACT_PROJECTION = "exact_projection"
    REQUIRED_SUBSEQUENCE = "required_subsequence"
    ACTIVE_GAP_EXTENSIBLE = "active_gap_extensible"


class ExtensionStatus(str, Enum):
    EXACTLY_SUPPORTED = "EXACTLY_SUPPORTED"
    SUPPORTED_BY_FINITE_EXTENSION = "SUPPORTED_BY_FINITE_EXTENSION"
    UNKNOWN_EXTENSION_CAP = "UNKNOWN_EXTENSION_CAP"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True, slots=True)
class OptionalActiveEventDomain:
    piece_id: str
    candidate_id: str
    placement_interval: tuple[int, int] | None
    allowed_moves: tuple[str, ...]
    allowed_state_before: tuple[tuple[int, int], ...]
    allowed_state_after: tuple[tuple[int, int], ...]
    obligation_effect: str
    maximum_multiplicity: int

    def row(self) -> dict[str, object]:
        return {
            "piece_id": self.piece_id,
            "candidate_id": self.candidate_id,
            "placement_interval": (
                None
                if self.placement_interval is None
                else list(self.placement_interval)
            ),
            "allowed_moves": list(self.allowed_moves),
            "allowed_state_before": [
                list(value) for value in self.allowed_state_before
            ],
            "allowed_state_after": [
                list(value) for value in self.allowed_state_after
            ],
            "obligation_effect": self.obligation_effect,
            "maximum_multiplicity": self.maximum_multiplicity,
        }


@dataclass(frozen=True, slots=True)
class CandidateContractMetadata:
    candidate_id: str
    piece_id: str
    piece_type: str
    contract_type: CandidateContractType
    required_active_skeleton: tuple[str, ...]
    exact_active_skeleton: tuple[str, ...] | None
    allowed_extra_event_domains: tuple[OptionalActiveEventDomain, ...]
    gap_mapping_requirements: tuple[object, ...]
    source: str
    provenance: tuple[str, ...]
    search_completeness_status: str

    def row(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "piece_id": self.piece_id,
            "piece_type": self.piece_type,
            "contract_type": self.contract_type.value,
            "required_active_skeleton": list(
                self.required_active_skeleton
            ),
            "exact_active_skeleton": (
                None
                if self.exact_active_skeleton is None
                else list(self.exact_active_skeleton)
            ),
            "allowed_extra_event_domains": [
                value.row() for value in self.allowed_extra_event_domains
            ],
            "gap_mapping_requirements": list(
                self.gap_mapping_requirements
            ),
            "source": self.source,
            "provenance": list(self.provenance),
            "search_completeness_status": self.search_completeness_status,
        }


@dataclass(frozen=True, slots=True)
class FreshCandidate:
    vector: MoveDemandVector
    contract: CandidateContractMetadata
    generation_rank: int

    def row(self) -> dict[str, object]:
        return {
            "generation_rank": self.generation_rank,
            "contract": self.contract.row(),
            "move_demand_vector": self.vector.row(),
        }


@dataclass(frozen=True, slots=True)
class FreshPortfolio:
    scramble: tuple[str, ...]
    normalized_scramble: tuple[str, ...]
    axis: DRAxis
    rotation_key: str
    active_depth_limit: int
    candidates_by_piece: Mapping[str, tuple[FreshCandidate, ...]]
    generation_elapsed_seconds: float
    seal_sha256: str

    @property
    def candidate_count(self) -> int:
        return sum(len(values) for values in self.candidates_by_piece.values())

    def exact_vectors(
        self,
        *,
        active_depth_limit: int | None = None,
    ) -> dict[str, tuple[MoveDemandVector, ...]]:
        limit = (
            self.active_depth_limit
            if active_depth_limit is None
            else int(active_depth_limit)
        )
        return {
            piece: tuple(
                candidate.vector
                for candidate in self.candidates_by_piece[piece]
                if (
                    candidate.contract.contract_type
                    is CandidateContractType.EXACT_PROJECTION
                    and candidate.vector.active_length <= limit
                )
            )
            for piece in PIECES
        }

    def summary(self) -> dict[str, object]:
        rows: dict[str, object] = {}
        for piece in PIECES:
            candidates = self.candidates_by_piece[piece]
            contract_counts = Counter(
                candidate.contract.contract_type.value
                for candidate in candidates
            )
            lengths = Counter(
                candidate.vector.active_length for candidate in candidates
            )
            exact = [
                candidate
                for candidate in candidates
                if candidate.contract.contract_type
                is CandidateContractType.EXACT_PROJECTION
            ]
            final_states = {
                candidate.vector.final_state for candidate in exact
            }
            exact_skeletons = {
                (
                    candidate.vector.final_state,
                    candidate.vector.active_skeleton,
                    tuple(
                        (
                            event.state_before,
                            event.state_after,
                        )
                        for event in candidate.vector.active_events
                    ),
                )
                for candidate in exact
            }
            rows[piece] = {
                "piece_type": _piece_type(piece),
                "raw_candidate_count": len(candidates),
                "exact_projection_count": len(exact),
                "effect_deduplicated_final_state_count": len(final_states),
                "trajectory_deduplicated_count": len(exact_skeletons),
                "contract_type_counts": dict(sorted(contract_counts.items())),
                "active_length_distribution": {
                    str(key): value for key, value in sorted(lengths.items())
                },
                "equivalent_witness_excess": max(
                    0, len(exact) - len(final_states)
                ),
                "cap_hit": False,
                "timeout_hit": False,
                "search_completeness_status": (
                    f"COMPLETE_ACTIVE_DEPTH_0_TO_{self.active_depth_limit}"
                ),
            }
        return {
            "schema": SCHEMA + ".portfolio-summary",
            "seed": FIXED_SEED,
            "scramble": list(self.scramble),
            "normalized_scramble": list(self.normalized_scramble),
            "axis": self.axis.value,
            "rotation_key": self.rotation_key,
            "active_depth_limit": self.active_depth_limit,
            "candidate_count": self.candidate_count,
            "generation_elapsed_seconds": self.generation_elapsed_seconds,
            "blind_reference_consumed": False,
            "seal_sha256": self.seal_sha256,
            "pieces": rows,
        }


def _build_vector(
    *,
    piece: str,
    initial_state: PieceState,
    path: Sequence[tuple[str, PieceState, PieceState]],
    candidate_id: str,
    source: str,
) -> MoveDemandVector:
    events: list[ActiveEventDemand] = []
    path = tuple(path)
    for ordinal, (move, before, after) in enumerate(path):
        events.append(
            ActiveEventDemand(
                piece_id=piece,
                piece_type=_piece_type(piece),
                candidate_id=candidate_id,
                active_ordinal=ordinal,
                move=move,
                state_before=_state_tuple(before),
                state_after=_state_tuple(after),
                predecessor_event=(
                    None
                    if ordinal == 0
                    else (candidate_id, ordinal - 1)
                ),
                successor_event=(
                    None
                    if ordinal + 1 == len(path)
                    else (candidate_id, ordinal + 1)
                ),
                source_column=None,
            )
        )
    skeleton = tuple(event.move for event in events)
    counts = Counter(skeleton)
    final_state = (
        initial_state if not path else path[-1][2]
    )
    return MoveDemandVector(
        piece_id=piece,
        piece_type=_piece_type(piece),
        candidate_id=candidate_id,
        counts=tuple(counts[move] for move in MOVE_ORDER),
        active_length=len(events),
        active_skeleton=skeleton,
        source=source,
        source_word=(),
        initial_state=_state_tuple(initial_state),
        final_state=_state_tuple(final_state),
        active_events=tuple(events),
    )


def _path_states(vector: MoveDemandVector) -> tuple[PieceState, ...]:
    output = [_piece_state(vector.initial_state)]
    output.extend(_piece_state(event.state_after) for event in vector.active_events)
    return tuple(output)


def _finite_detour_domains(
    vector: MoveDemandVector,
    *,
    limit: int = 4,
) -> tuple[OptionalActiveEventDomain, ...]:
    graph = SinglePieceRouteGraph(
        "corner" if vector.piece_type == "CORNER" else "edge"
    )
    rows: list[OptionalActiveEventDomain] = []
    for gap, state in enumerate(_path_states(vector)):
        for move in FACE_MOVES:
            after = graph.transitions[state][move]
            if after == state:
                continue
            inverse = INVERSE_MOVE[move]
            restored = graph.transitions[after][inverse]
            if restored != state:
                raise AssertionError("inverse detour did not restore piece state")
            rows.append(
                OptionalActiveEventDomain(
                    piece_id=vector.piece_id,
                    candidate_id=vector.candidate_id,
                    placement_interval=(gap, gap),
                    allowed_moves=(move, inverse),
                    allowed_state_before=(
                        _state_tuple(state),
                        _state_tuple(after),
                    ),
                    allowed_state_after=(
                        _state_tuple(after),
                        _state_tuple(restored),
                    ),
                    obligation_effect=(
                        "FINITE_ACTIVE_DETOUR_PRESERVES_GAP_ENTRY_EXIT_STATE"
                    ),
                    maximum_multiplicity=1,
                )
            )
            if len(rows) >= limit:
                return tuple(rows)
    return tuple(rows)


def _contract_copy(
    candidate: FreshCandidate,
    contract_type: CandidateContractType,
    *,
    generation_rank: int,
) -> FreshCandidate:
    suffix = {
        CandidateContractType.REQUIRED_SUBSEQUENCE: "REQ",
        CandidateContractType.ACTIVE_GAP_EXTENSIBLE: "EXT",
    }[contract_type]
    candidate_id = candidate.vector.candidate_id + f":{suffix}"
    vector = _reidentify_vector(
        candidate.vector,
        candidate_id=candidate_id,
        source=(
            "FRESH_REQUIRED_SUBSEQUENCE"
            if contract_type is CandidateContractType.REQUIRED_SUBSEQUENCE
            else "FRESH_ACTIVE_GAP_EXTENSIBLE"
        ),
    )
    domains = (
        ()
        if contract_type is CandidateContractType.REQUIRED_SUBSEQUENCE
        else _finite_detour_domains(vector)
    )
    return FreshCandidate(
        vector=vector,
        contract=CandidateContractMetadata(
            candidate_id=candidate_id,
            piece_id=vector.piece_id,
            piece_type=vector.piece_type,
            contract_type=contract_type,
            required_active_skeleton=vector.active_skeleton,
            exact_active_skeleton=None,
            allowed_extra_event_domains=domains,
            gap_mapping_requirements=tuple(
                {
                    "gap_index": gap,
                    "entry_state": list(state),
                    "exit_state": list(state),
                    "require_same_state": True,
                }
                for gap, state in enumerate(
                    [
                        vector.initial_state,
                        *(
                            event.state_after
                            for event in vector.active_events
                        ),
                    ]
                )
            ),
            source=vector.source,
            provenance=(
                "ONE_PIECE_RUNTIME_TRANSITION_GRAPH",
                "UD_NORMALIZED_DR_TARGET_DOMAIN",
                "NO_REFERENCE_WORD",
            ),
            search_completeness_status=(
                "FINITE_EXTENSION_DOMAIN"
                if domains
                else "REQUIRED_SUBSEQUENCE_ONLY"
            ),
        ),
        generation_rank=generation_rank,
    )


def _reidentify_vector(
    vector: MoveDemandVector,
    *,
    candidate_id: str,
    source: str,
    events: Sequence[ActiveEventDemand] | None = None,
) -> MoveDemandVector:
    raw = vector.active_events if events is None else tuple(events)
    rebuilt = tuple(
        replace(
            event,
            candidate_id=candidate_id,
            active_ordinal=ordinal,
            predecessor_event=(
                None if ordinal == 0 else (candidate_id, ordinal - 1)
            ),
            successor_event=(
                None
                if ordinal + 1 == len(raw)
                else (candidate_id, ordinal + 1)
            ),
            source_column=None,
        )
        for ordinal, event in enumerate(raw)
    )
    counts = Counter(event.move for event in rebuilt)
    return replace(
        vector,
        candidate_id=candidate_id,
        counts=tuple(counts[move] for move in MOVE_ORDER),
        active_length=len(rebuilt),
        active_skeleton=tuple(event.move for event in rebuilt),
        active_events=rebuilt,
        source=source,
        source_word=(),
    )


def materialize_finite_extension(
    candidate: FreshCandidate,
    domain: OptionalActiveEventDomain,
) -> MoveDemandVector:
    if (
        candidate.contract.contract_type
        is not CandidateContractType.ACTIVE_GAP_EXTENSIBLE
    ):
        raise ValueError("finite extensions require an extensible contract")
    if domain.candidate_id != candidate.vector.candidate_id:
        raise ValueError("extension domain belongs to another candidate")
    if domain.maximum_multiplicity < 1:
        raise ValueError("extension domain has zero multiplicity")
    if domain.placement_interval is None:
        raise ValueError("finite detour requires a concrete insertion gap")
    gap = domain.placement_interval[0]
    if domain.placement_interval != (gap, gap):
        raise ValueError("this materializer accepts a fixed gap only")
    insertion = tuple(
        ActiveEventDemand(
            piece_id=candidate.vector.piece_id,
            piece_type=candidate.vector.piece_type,
            candidate_id=candidate.vector.candidate_id,
            active_ordinal=0,
            move=move,
            state_before=before,
            state_after=after,
            predecessor_event=None,
            successor_event=None,
            source_column=None,
        )
        for move, before, after in zip(
            domain.allowed_moves,
            domain.allowed_state_before,
            domain.allowed_state_after,
        )
    )
    raw = (
        candidate.vector.active_events[:gap]
        + insertion
        + candidate.vector.active_events[gap:]
    )
    candidate_id = (
        candidate.vector.candidate_id
        + ":FINITE:"
        + _stable_digest(domain.row(), 10)
    )
    return _reidentify_vector(
        candidate.vector,
        candidate_id=candidate_id,
        source="SUPPORTED_BY_FINITE_EXTENSION",
        events=raw,
    )


def generate_fresh_candidate_portfolio(
    *,
    scramble: Sequence[str] = FIXED_SCRAMBLE,
    axis: DRAxis | str = FIXED_DR_AXIS,
    active_depth_limit: int = 3,
) -> FreshPortfolio:
    """Completely enumerate one-piece active paths within the fixed depth.

    No reference word or reference inventory is accepted by this API.
    """

    if active_depth_limit < 0:
        raise ValueError("active_depth_limit must be nonnegative")
    started = perf_counter()
    scramble = tuple(str(move) for move in scramble)
    axis = DRAxis(axis)
    rotation = normalizing_rotation(axis)
    normalized_scramble = rotation.map_sequence(scramble)
    effect = CubieEffect.from_transformation(
        from_sequence(normalized_scramble)
    )
    output: dict[str, tuple[FreshCandidate, ...]] = {}
    seal_rows = []

    for piece in PIECES:
        graph = SinglePieceRouteGraph(
            "corner" if piece in CORNER_SET else "edge"
        )
        initial = piece_state(effect, piece)
        exact: list[FreshCandidate] = []
        frontier: list[
            tuple[PieceState, tuple[tuple[str, PieceState, PieceState], ...]]
        ] = [(initial, ())]
        rank = 0
        for depth in range(active_depth_limit + 1):
            next_frontier = []
            for state, path in frontier:
                if _target_state_allowed(piece, state):
                    candidate_id = (
                        f"FRESH-{piece}-"
                        + _stable_digest(
                            (
                                normalized_scramble,
                                piece,
                                tuple(move for move, _, _ in path),
                                tuple(
                                    (
                                        before.position,
                                        before.orientation,
                                        after.position,
                                        after.orientation,
                                    )
                                    for _, before, after in path
                                ),
                            ),
                            16,
                        )
                    )
                    vector = _build_vector(
                        piece=piece,
                        initial_state=initial,
                        path=path,
                        candidate_id=candidate_id,
                        source="FRESH_SINGLE_PIECE_ACTIVE_PATH",
                    )
                    rank += 1
                    exact.append(
                        FreshCandidate(
                            vector=vector,
                            contract=CandidateContractMetadata(
                                candidate_id=candidate_id,
                                piece_id=piece,
                                piece_type=_piece_type(piece),
                                contract_type=(
                                    CandidateContractType.EXACT_PROJECTION
                                ),
                                required_active_skeleton=(
                                    vector.active_skeleton
                                ),
                                exact_active_skeleton=(
                                    vector.active_skeleton
                                ),
                                allowed_extra_event_domains=(),
                                gap_mapping_requirements=(),
                                source=vector.source,
                                provenance=(
                                    "ONE_PIECE_RUNTIME_TRANSITION_GRAPH",
                                    "UD_NORMALIZED_DR_TARGET_DOMAIN",
                                    "NO_REFERENCE_WORD",
                                ),
                                search_completeness_status=(
                                    "COMPLETE_WITHIN_ACTIVE_DEPTH_BOUND"
                                ),
                            ),
                            generation_rank=rank,
                        )
                    )
                if depth == active_depth_limit:
                    continue
                for move in FACE_MOVES:
                    after = graph.transitions[state][move]
                    if after == state:
                        continue
                    next_frontier.append(
                        (after, path + ((move, state, after),))
                    )
            frontier = next_frontier

        exact.sort(
            key=lambda candidate: (
                candidate.vector.active_length,
                tuple(
                    MOVE_INDEX[move]
                    for move in candidate.vector.active_skeleton
                ),
                candidate.vector.final_state,
                candidate.vector.candidate_id,
            )
        )
        reranked = [
            replace(candidate, generation_rank=index + 1)
            for index, candidate in enumerate(exact)
        ]
        contract_variants: list[FreshCandidate] = []
        if reranked:
            shortest = reranked[0]
            contract_variants.append(
                _contract_copy(
                    shortest,
                    CandidateContractType.REQUIRED_SUBSEQUENCE,
                    generation_rank=len(reranked) + 1,
                )
            )
            contract_variants.append(
                _contract_copy(
                    shortest,
                    CandidateContractType.ACTIVE_GAP_EXTENSIBLE,
                    generation_rank=len(reranked) + 2,
                )
            )
        values = tuple(reranked + contract_variants)
        output[piece] = values
        seal_rows.extend(
            (
                candidate.vector.candidate_id,
                candidate.contract.contract_type.value,
                candidate.vector.active_skeleton,
                candidate.vector.initial_state,
                candidate.vector.final_state,
            )
            for candidate in values
        )

    seal = sha256(
        json.dumps(
            {
                "schema": SCHEMA,
                "scramble": scramble,
                "normalized_scramble": normalized_scramble,
                "axis": axis.value,
                "active_depth_limit": active_depth_limit,
                "candidates": seal_rows,
                "reference": "NOT_CONSUMED",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return FreshPortfolio(
        scramble=scramble,
        normalized_scramble=normalized_scramble,
        axis=axis,
        rotation_key=rotation.key,
        active_depth_limit=active_depth_limit,
        candidates_by_piece=output,
        generation_elapsed_seconds=perf_counter() - started,
        seal_sha256=seal,
    )


@dataclass(frozen=True, slots=True)
class StateIndexedMoveDemand:
    move: str
    piece_type: str
    state_before_position: str
    orientation_before: int | None
    count: int

    def row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StateIndexedResidueResult:
    status: str
    rows: tuple[StateIndexedMoveDemand, ...]
    expected_slot_count: Mapping[str, int]
    failure_reasons: tuple[str, ...]

    def row(self) -> dict[str, object]:
        return {
            "status": self.status,
            "rows": [value.row() for value in self.rows],
            "expected_slot_count": dict(self.expected_slot_count),
            "failure_reasons": list(self.failure_reasons),
        }


def evaluate_state_indexed_residue(
    candidates: Sequence[MoveDemandVector],
    inventory: ColumnInventory,
) -> StateIndexedResidueResult:
    counts = Counter(
        (
            event.move,
            event.piece_type,
            PIECES[event.state_before[0]],
            event.state_before[1],
        )
        for candidate in candidates
        for event in candidate.active_events
    )
    position_counts = Counter(
        (
            event.move,
            event.piece_type,
            PIECES[event.state_before[0]],
        )
        for candidate in candidates
        for event in candidate.active_events
    )
    failures = []
    expected = {}
    for move, column_count in inventory.move_counts.items():
        for piece_type, positions in (
            ("CORNER", CORNER_ORDER),
            ("EDGE", EDGE_ORDER),
        ):
            for position in positions:
                if move[0] not in position:
                    continue
                key = f"{move}:{piece_type}:{position}"
                expected[key] = int(column_count)
                actual = position_counts[(move, piece_type, position)]
                if actual != int(column_count):
                    failures.append(
                        f"STATE_SLOT:{key}:{actual}!={column_count}"
                    )
    return StateIndexedResidueResult(
        status=(
            "SAT_STATE_INDEXED_RESIDUE"
            if not failures
            else "UNSAT_STATE_INDEXED_RESIDUE"
        ),
        rows=tuple(
            StateIndexedMoveDemand(
                move=move,
                piece_type=piece_type,
                state_before_position=position,
                orientation_before=orientation,
                count=count,
            )
            for (
                move,
                piece_type,
                position,
                orientation,
            ), count in sorted(counts.items())
        ),
        expected_slot_count=expected,
        failure_reasons=tuple(failures),
    )


@dataclass(frozen=True, slots=True)
class TransitionSignatureResult:
    status: str
    signature_counts: Mapping[str, int]
    invalid_event_keys: tuple[tuple[str, int], ...]

    def row(self) -> dict[str, object]:
        return {
            "status": self.status,
            "signature_counts": dict(self.signature_counts),
            "invalid_event_keys": [
                list(value) for value in self.invalid_event_keys
            ],
        }


def evaluate_transition_signature_balance(
    candidates: Sequence[MoveDemandVector],
) -> TransitionSignatureResult:
    graphs = {
        "CORNER": SinglePieceRouteGraph("corner"),
        "EDGE": SinglePieceRouteGraph("edge"),
    }
    counts: Counter[str] = Counter()
    invalid = []
    for candidate in candidates:
        graph = graphs[candidate.piece_type]
        for event in candidate.active_events:
            before = _piece_state(event.state_before)
            expected = graph.transitions[before][event.move]
            after = _piece_state(event.state_after)
            signature = (
                f"{event.move}|{before.position}:{before.orientation}"
                f"->{after.position}:{after.orientation}"
            )
            counts[signature] += 1
            if expected != after or expected == before:
                invalid.append(event.key)
    return TransitionSignatureResult(
        status=(
            "SAT_TRANSITION_SIGNATURE"
            if not invalid
            else "UNSAT_TRANSITION_SIGNATURE"
        ),
        signature_counts=dict(sorted(counts.items())),
        invalid_event_keys=tuple(invalid),
    )


@dataclass(frozen=True, slots=True)
class IncompatibilityGraphResult:
    move: str
    node_count: int
    edge_count: int
    clique_lower_bound: int
    greedy_coloring_upper_bound: int
    exact_chromatic_number: int | None
    exact_coloring_cap_hit: bool
    edge_reason_counts: Mapping[str, int]

    def row(self) -> dict[str, object]:
        return asdict(self)


def _maximum_clique(adjacency: Sequence[set[int]]) -> int:
    best = 0

    def visit(chosen: set[int], possible: set[int]) -> None:
        nonlocal best
        if len(chosen) + len(possible) <= best:
            return
        if not possible:
            best = max(best, len(chosen))
            return
        while possible:
            if len(chosen) + len(possible) <= best:
                return
            node = max(possible, key=lambda value: len(adjacency[value]))
            possible.remove(node)
            visit(chosen | {node}, possible & adjacency[node])
        best = max(best, len(chosen))

    visit(set(), set(range(len(adjacency))))
    return best


def _greedy_dsatur(adjacency: Sequence[set[int]]) -> int:
    if not adjacency:
        return 0
    colors: dict[int, int] = {}
    while len(colors) < len(adjacency):
        node = max(
            (index for index in range(len(adjacency)) if index not in colors),
            key=lambda index: (
                len({colors[n] for n in adjacency[index] if n in colors}),
                len(adjacency[index]),
                -index,
            ),
        )
        forbidden = {colors[n] for n in adjacency[node] if n in colors}
        color = 0
        while color in forbidden:
            color += 1
        colors[node] = color
    return max(colors.values(), default=-1) + 1


def _bounded_exact_coloring(
    adjacency: Sequence[set[int]],
    lower: int,
    upper: int,
    *,
    node_cap: int,
) -> tuple[int | None, bool]:
    if lower == upper:
        return lower, False
    nodes = [0]
    order = sorted(
        range(len(adjacency)),
        key=lambda node: (-len(adjacency[node]), node),
    )

    def feasible(color_count: int) -> bool | None:
        colors = [-1] * len(adjacency)

        def visit(depth: int) -> bool | None:
            nodes[0] += 1
            if nodes[0] > node_cap:
                return None
            if depth == len(order):
                return True
            node = order[depth]
            forbidden = {
                colors[neighbour]
                for neighbour in adjacency[node]
                if colors[neighbour] >= 0
            }
            for color in range(color_count):
                if color in forbidden:
                    continue
                colors[node] = color
                result = visit(depth + 1)
                if result is not False:
                    return result
                colors[node] = -1
            return False

        return visit(0)

    for color_count in range(lower, upper + 1):
        result = feasible(color_count)
        if result is None:
            return None, True
        if result:
            return color_count, False
    raise AssertionError("greedy upper bound must be colorable")


def build_incompatibility_graphs(
    candidates: Sequence[MoveDemandVector],
    *,
    exact_coloring_node_cap: int = 50_000,
) -> tuple[IncompatibilityGraphResult, ...]:
    by_move: dict[str, list[ActiveEventDemand]] = defaultdict(list)
    for candidate in candidates:
        for event in candidate.active_events:
            by_move[event.move].append(event)
    output = []
    for move in MOVE_ORDER:
        events = by_move.get(move, [])
        if not events:
            continue
        adjacency = [set() for _ in events]
        reasons: Counter[str] = Counter()
        for left in range(len(events)):
            for right in range(left + 1, len(events)):
                labels = []
                if events[left].piece_id == events[right].piece_id:
                    labels.append("SAME_PIECE_DIFFERENT_ORDINAL")
                if (
                    events[left].state_before[0]
                    == events[right].state_before[0]
                ):
                    labels.append("DUPLICATE_STATE_BEFORE_SLOT")
                if labels:
                    adjacency[left].add(right)
                    adjacency[right].add(left)
                    reasons.update(labels)
        clique = _maximum_clique(adjacency)
        greedy = _greedy_dsatur(adjacency)
        exact, capped = _bounded_exact_coloring(
            adjacency,
            clique,
            greedy,
            node_cap=exact_coloring_node_cap,
        )
        output.append(
            IncompatibilityGraphResult(
                move=move,
                node_count=len(events),
                edge_count=sum(map(len, adjacency)) // 2,
                clique_lower_bound=clique,
                greedy_coloring_upper_bound=greedy,
                exact_chromatic_number=exact,
                exact_coloring_cap_hit=capped,
                edge_reason_counts=dict(sorted(reasons.items())),
            )
        )
    return tuple(output)


def _inventory_superkeys(
    counts: tuple[int, ...],
    total_columns: int,
) -> set[tuple[int, ...]]:
    used = sum(counts)
    if used > total_columns:
        return set()
    extra = total_columns - used
    output = set()
    for additions in combinations_with_replacement(
        range(len(MOVE_ORDER)), extra
    ):
        values = list(counts)
        for index in additions:
            values[index] += 1
        output.add(tuple(values))
    return output


def _candidate_fits_inventory(
    candidate: MoveDemandVector,
    inventory: tuple[int, ...],
) -> bool:
    if any(
        candidate.counts[index] > inventory[index]
        for index in range(len(MOVE_ORDER))
    ):
        return False
    slot_counts = Counter(
        (MOVE_INDEX[event.move], event.state_before[0])
        for event in candidate.active_events
    )
    return all(
        count <= inventory[move_index]
        for (move_index, _), count in slot_counts.items()
    )


def _slot_capacity(
    inventory: tuple[int, ...],
    piece_type: str,
) -> Counter[tuple[int, int]]:
    positions = CORNER_ORDER if piece_type == "CORNER" else EDGE_ORDER
    return Counter(
        {
            (move_index, PIECE_INDEX[position]): count
            for move_index, count in enumerate(inventory)
            if count
            for position in positions
            if MOVE_ORDER[move_index][0] in position
        }
    )


def _candidate_slot_counts(
    candidate: MoveDemandVector,
) -> Counter[tuple[int, int]]:
    return Counter(
        (MOVE_INDEX[event.move], event.state_before[0])
        for event in candidate.active_events
    )


@dataclass(frozen=True, slots=True)
class TypeExactCoverResult:
    status: str
    piece_type: str
    solutions: tuple[tuple[MoveDemandVector, ...], ...]
    nodes: int
    cap_hit: bool
    eligible_domain_sizes: Mapping[str, int]

    def row(self) -> dict[str, object]:
        return {
            "status": self.status,
            "piece_type": self.piece_type,
            "solution_candidate_ids": [
                [candidate.candidate_id for candidate in solution]
                for solution in self.solutions
            ],
            "nodes": self.nodes,
            "cap_hit": self.cap_hit,
            "eligible_domain_sizes": dict(self.eligible_domain_sizes),
        }


def _exact_cover_piece_type(
    domains: Mapping[str, Sequence[MoveDemandVector]],
    inventory: tuple[int, ...],
    piece_type: str,
    *,
    node_cap: int,
    solution_limit: int,
) -> TypeExactCoverResult:
    pieces = CORNER_ORDER if piece_type == "CORNER" else EDGE_ORDER
    capacity = _slot_capacity(inventory, piece_type)
    candidate_slots = {
        candidate.candidate_id: _candidate_slot_counts(candidate)
        for piece in pieces
        for candidate in domains[piece]
    }
    eligible = {
        piece: tuple(
            candidate
            for candidate in domains[piece]
            if (
                _candidate_fits_inventory(candidate, inventory)
                and all(
                    count <= capacity[key]
                    for key, count in candidate_slots[
                        candidate.candidate_id
                    ].items()
                )
            )
        )
        for piece in pieces
    }
    if any(not eligible[piece] for piece in pieces):
        return TypeExactCoverResult(
            status="UNSAT_EMPTY_DOMAIN",
            piece_type=piece_type,
            solutions=(),
            nodes=0,
            cap_hit=False,
            eligible_domain_sizes={
                piece: len(eligible[piece]) for piece in pieces
            },
        )
    initial_order = tuple(
        sorted(
            pieces,
            key=lambda piece: (
                len(eligible[piece]),
                pieces.index(piece),
            ),
        )
    )
    remaining = Counter(capacity)
    selected: list[MoveDemandVector] = []
    solutions: list[tuple[MoveDemandVector, ...]] = []
    nodes = 0
    cap_hit = False

    def visit(unassigned: tuple[str, ...]) -> None:
        nonlocal nodes, cap_hit
        if cap_hit or len(solutions) >= solution_limit:
            return
        nodes += 1
        if nodes > node_cap:
            cap_hit = True
            return
        if not unassigned:
            if all(value == 0 for value in remaining.values()):
                by_piece = {
                    candidate.piece_id: candidate
                    for candidate in selected
                }
                solutions.append(
                    tuple(by_piece[piece] for piece in pieces)
                )
            return
        required_total = sum(remaining.values())
        fitting = {
            piece: tuple(
                candidate
                for candidate in eligible[piece]
                if all(
                    count <= remaining[key]
                    for key, count in candidate_slots[
                        candidate.candidate_id
                    ].items()
                )
            )
            for piece in unassigned
        }
        if any(not fitting[piece] for piece in unassigned):
            return
        minimum = sum(
            min(candidate.active_length for candidate in fitting[piece])
            for piece in unassigned
        )
        maximum = sum(
            max(candidate.active_length for candidate in fitting[piece])
            for piece in unassigned
        )
        if not (minimum <= required_total <= maximum):
            return
        piece = min(
            unassigned,
            key=lambda value: (
                len(fitting[value]),
                initial_order.index(value),
            ),
        )
        next_unassigned = tuple(
            value for value in unassigned if value != piece
        )
        for candidate in fitting[piece]:
            slots = candidate_slots[candidate.candidate_id]
            for key, count in slots.items():
                remaining[key] -= count
            selected.append(candidate)
            visit(next_unassigned)
            selected.pop()
            for key, count in slots.items():
                remaining[key] += count
            if cap_hit or len(solutions) >= solution_limit:
                break

    visit(initial_order)
    return TypeExactCoverResult(
        status=(
            "SAT"
            if solutions
            else "UNKNOWN_NODE_CAP"
            if cap_hit
            else "UNSAT_EXACT_COVER"
        ),
        piece_type=piece_type,
        solutions=tuple(solutions),
        nodes=nodes,
        cap_hit=cap_hit,
        eligible_domain_sizes={
            piece: len(eligible[piece]) for piece in pieces
        },
    )


@dataclass(frozen=True, slots=True)
class FreshInventorySolution:
    normalized_word: tuple[str, ...]
    original_word: tuple[str, ...]
    inventory: ColumnInventory
    selected_candidates: tuple[MoveDemandVector, ...]
    analysis: SelectionAnalysis
    state_indexed: StateIndexedResidueResult
    transition_signatures: TransitionSignatureResult
    incompatibility_graphs: tuple[IncompatibilityGraphResult, ...]
    dr_reduced_normalized: bool
    dr_reduced_original: bool

    def row(self) -> dict[str, object]:
        return {
            "normalized_word": list(self.normalized_word),
            "original_word": list(self.original_word),
            "inventory": self.inventory.row(),
            "selected_candidate_ids": [
                candidate.candidate_id
                for candidate in self.selected_candidates
            ],
            "selected_candidate_sources": [
                candidate.source for candidate in self.selected_candidates
            ],
            "analysis": self.analysis.row(),
            "state_indexed_residue": self.state_indexed.row(),
            "transition_signature_balance": (
                self.transition_signatures.row()
            ),
            "incompatibility_graphs": [
                graph.row() for graph in self.incompatibility_graphs
            ],
            "dr_reduced_normalized": self.dr_reduced_normalized,
            "dr_reduced_original": self.dr_reduced_original,
        }


@dataclass(frozen=True, slots=True)
class FreshInventorySearchResult:
    status: str
    mode: str
    active_candidate_depth: int
    global_length_limit: int
    inventory_hypotheses_generated: int
    inventory_hypotheses_with_all_piece_domains: int
    inventory_hypotheses_examined: int
    state_indexed_rejections: int
    corner_cover_rejections: int
    edge_cover_rejections: int
    graph_lower_bound_rejections: int
    slot_packing_rejections: int
    precedence_rejections: int
    replay_rejections: int
    unknown_count: int
    exact_cover_nodes: int
    candidate_combinations_visited: int
    runtime_seconds: float
    solutions: tuple[FreshInventorySolution, ...]
    cap_hit: bool

    def row(self) -> dict[str, object]:
        payload = asdict(self)
        payload["solutions"] = [value.row() for value in self.solutions]
        return payload


def search_fresh_inventory(
    portfolio: FreshPortfolio,
    *,
    mode: str = "EXACT_ONLY",
    active_candidate_depth: int = 3,
    global_length_limit: int = 5,
    exact_cover_node_cap: int = 200_000,
    per_type_solution_limit: int = 12,
    solution_limit: int = 4,
    extra_vectors: Mapping[str, Sequence[MoveDemandVector]] | None = None,
) -> FreshInventorySearchResult:
    """Search candidate combinations and inventories before word ordering."""

    if mode not in {"EXACT_ONLY", "EXTENSIBLE", "DOMINO_REDUCTION_SUPPLIED"}:
        raise ValueError(f"unsupported search mode: {mode}")
    started = perf_counter()
    domains = portfolio.exact_vectors(
        active_depth_limit=active_candidate_depth
    )
    if extra_vectors:
        domains = {
            piece: tuple(domains[piece]) + tuple(extra_vectors.get(piece, ()))
            for piece in PIECES
        }
    if any(not domains[piece] for piece in PIECES):
        return FreshInventorySearchResult(
            status="UNKNOWN_INCOMPLETE_CANDIDATE_DOMAIN",
            mode=mode,
            active_candidate_depth=active_candidate_depth,
            global_length_limit=global_length_limit,
            inventory_hypotheses_generated=0,
            inventory_hypotheses_with_all_piece_domains=0,
            inventory_hypotheses_examined=0,
            state_indexed_rejections=0,
            corner_cover_rejections=0,
            edge_cover_rejections=0,
            graph_lower_bound_rejections=0,
            slot_packing_rejections=0,
            precedence_rejections=0,
            replay_rejections=0,
            unknown_count=1,
            exact_cover_nodes=0,
            candidate_combinations_visited=0,
            runtime_seconds=perf_counter() - started,
            solutions=(),
            cap_hit=False,
        )

    generated = 0
    with_domains = 0
    examined = 0
    state_rejections = 0
    corner_rejections = 0
    edge_rejections = 0
    graph_rejections = 0
    packing_rejections = 0
    precedence_rejections = 0
    replay_rejections = 0
    unknown = 0
    nodes = 0
    combinations = 0
    cap_hit = False
    solutions: list[FreshInventorySolution] = []

    for global_length in range(global_length_limit + 1):
        inventory_sets: list[set[tuple[int, ...]]] = []
        for piece in PIECES:
            possible: set[tuple[int, ...]] = set()
            signatures = {
                candidate.counts
                for candidate in domains[piece]
                if candidate.active_length <= active_candidate_depth
            }
            for signature in signatures:
                possible.update(
                    _inventory_superkeys(signature, global_length)
                )
            inventory_sets.append(possible)
        generated += len(
            set(combinations_with_replacement(
                range(len(MOVE_ORDER)), global_length
            ))
        )
        if any(not values for values in inventory_sets):
            continue
        common = set.intersection(*inventory_sets)
        with_domains += len(common)
        for inventory_tuple in sorted(common):
            examined += 1
            corner = _exact_cover_piece_type(
                domains,
                inventory_tuple,
                "CORNER",
                node_cap=exact_cover_node_cap,
                solution_limit=per_type_solution_limit,
            )
            nodes += corner.nodes
            cap_hit |= corner.cap_hit
            if corner.status.startswith("UNKNOWN"):
                unknown += 1
            if not corner.solutions:
                corner_rejections += 1
                continue
            edge = _exact_cover_piece_type(
                domains,
                inventory_tuple,
                "EDGE",
                node_cap=exact_cover_node_cap,
                solution_limit=per_type_solution_limit,
            )
            nodes += edge.nodes
            cap_hit |= edge.cap_hit
            if edge.status.startswith("UNKNOWN"):
                unknown += 1
            if not edge.solutions:
                edge_rejections += 1
                continue
            inventory = ColumnInventory(
                move_counts={
                    MOVE_ORDER[index]: value
                    for index, value in enumerate(inventory_tuple)
                    if value
                },
                total_columns=sum(inventory_tuple),
            )
            for corner_solution in corner.solutions:
                for edge_solution in edge.solutions:
                    combinations += 1
                    selection = corner_solution + edge_solution
                    state_indexed = evaluate_state_indexed_residue(
                        selection, inventory
                    )
                    if not state_indexed.status.startswith("SAT"):
                        state_rejections += 1
                        continue
                    transition = evaluate_transition_signature_balance(
                        selection
                    )
                    if not transition.status.startswith("SAT"):
                        state_rejections += 1
                        continue
                    graphs = build_incompatibility_graphs(selection)
                    lower_invalid = any(
                        graph.clique_lower_bound
                        > int(inventory.move_counts.get(graph.move, 0))
                        for graph in graphs
                    )
                    if lower_invalid:
                        graph_rejections += 1
                        continue
                    analysis = analyze_candidate_selection(
                        selection,
                        packing_node_cap=exact_cover_node_cap,
                        prefer_source_columns=False,
                    )
                    if analysis.status == "UNKNOWN":
                        unknown += 1
                        continue
                    if analysis.status != "SAT" or analysis.replay is None:
                        if analysis.failure_stage == "SLOT_PACKING":
                            packing_rejections += 1
                        elif analysis.failure_stage == "PRECEDENCE":
                            precedence_rejections += 1
                        else:
                            replay_rejections += 1
                        continue
                    normalized_word = analysis.replay.global_word
                    rotation = normalizing_rotation(portfolio.axis)
                    original_word = denormalize_word(
                        normalized_word, rotation
                    )
                    normalized_dr = diagnose_sequence(
                        portfolio.normalized_scramble + normalized_word,
                        axis=DRAxis.UD,
                    ).is_reduced
                    original_dr = diagnose_sequence(
                        portfolio.scramble + original_word,
                        axis=portfolio.axis,
                    ).is_reduced
                    if not (normalized_dr and original_dr):
                        replay_rejections += 1
                        continue
                    solutions.append(
                        FreshInventorySolution(
                            normalized_word=normalized_word,
                            original_word=original_word,
                            inventory=inventory,
                            selected_candidates=selection,
                            analysis=analysis,
                            state_indexed=state_indexed,
                            transition_signatures=transition,
                            incompatibility_graphs=graphs,
                            dr_reduced_normalized=normalized_dr,
                            dr_reduced_original=original_dr,
                        )
                    )
                    if len(solutions) >= solution_limit:
                        break
                if len(solutions) >= solution_limit:
                    break
            if len(solutions) >= solution_limit:
                break
        if solutions:
            break

    status = (
        "SAT"
        if solutions
        else "UNKNOWN_CAP"
        if cap_hit or unknown
        else "UNSAT_WITHIN_COMPLETE_BOUNDS"
    )
    return FreshInventorySearchResult(
        status=status,
        mode=mode,
        active_candidate_depth=active_candidate_depth,
        global_length_limit=global_length_limit,
        inventory_hypotheses_generated=generated,
        inventory_hypotheses_with_all_piece_domains=with_domains,
        inventory_hypotheses_examined=examined,
        state_indexed_rejections=state_rejections,
        corner_cover_rejections=corner_rejections,
        edge_cover_rejections=edge_rejections,
        graph_lower_bound_rejections=graph_rejections,
        slot_packing_rejections=packing_rejections,
        precedence_rejections=precedence_rejections,
        replay_rejections=replay_rejections,
        unknown_count=unknown,
        exact_cover_nodes=nodes,
        candidate_combinations_visited=combinations,
        runtime_seconds=perf_counter() - started,
        solutions=tuple(solutions),
        cap_hit=cap_hit,
    )


def finite_extension_vectors(
    portfolio: FreshPortfolio,
    *,
    per_piece_limit: int = 4,
) -> dict[str, tuple[MoveDemandVector, ...]]:
    output = {}
    for piece in PIECES:
        rows = []
        for candidate in portfolio.candidates_by_piece[piece]:
            if (
                candidate.contract.contract_type
                is not CandidateContractType.ACTIVE_GAP_EXTENSIBLE
            ):
                continue
            for domain in candidate.contract.allowed_extra_event_domains:
                rows.append(materialize_finite_extension(candidate, domain))
                if len(rows) >= per_piece_limit:
                    break
            if len(rows) >= per_piece_limit:
                break
        output[piece] = tuple(rows)
    return output


def portfolio_contains_projection(
    portfolio: FreshPortfolio,
    vectors: Sequence[MoveDemandVector],
) -> dict[str, object]:
    """Post-seal coverage check for an externally supplied projection."""

    by_piece = {vector.piece_id: vector for vector in vectors}
    rows = {}
    for piece in PIECES:
        target = by_piece[piece]
        matches = [
            candidate.vector.candidate_id
            for candidate in portfolio.candidates_by_piece[piece]
            if (
                candidate.contract.contract_type
                is CandidateContractType.EXACT_PROJECTION
                and candidate.vector.active_skeleton
                == target.active_skeleton
                and tuple(
                    (
                        event.state_before,
                        event.state_after,
                    )
                    for event in candidate.vector.active_events
                )
                == tuple(
                    (
                        event.state_before,
                        event.state_after,
                    )
                    for event in target.active_events
                )
                and candidate.vector.final_state == target.final_state
            )
        ]
        rows[piece] = {
            "covered": bool(matches),
            "matching_candidate_ids": matches,
        }
    return {
        "all_twenty_piece_projections_covered": all(
            row["covered"] for row in rows.values()
        ),
        "piece_rows": rows,
        "portfolio_seal_sha256": portfolio.seal_sha256,
        "comparison_executed_after_seal": True,
    }


__all__ = [
    "SCHEMA",
    "FIXED_SEED",
    "FIXED_SCRAMBLE",
    "FIXED_DR_AXIS",
    "CandidateContractType",
    "ExtensionStatus",
    "OptionalActiveEventDomain",
    "CandidateContractMetadata",
    "FreshCandidate",
    "FreshPortfolio",
    "StateIndexedMoveDemand",
    "StateIndexedResidueResult",
    "TransitionSignatureResult",
    "IncompatibilityGraphResult",
    "TypeExactCoverResult",
    "FreshInventorySolution",
    "FreshInventorySearchResult",
    "normalizing_rotation",
    "denormalize_word",
    "generate_fresh_candidate_portfolio",
    "materialize_finite_extension",
    "evaluate_state_indexed_residue",
    "evaluate_transition_signature_balance",
    "build_incompatibility_graphs",
    "search_fresh_inventory",
    "finite_extension_vectors",
    "portfolio_contains_projection",
]
