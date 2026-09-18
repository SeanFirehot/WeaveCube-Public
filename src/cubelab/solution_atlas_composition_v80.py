from __future__ import annotations

"""Atlas-guided, grammar-agnostic piece-candidate composition POC.

The module deliberately separates three concerns:

* an atlas is evidence about complete, verified solution words;
* a :class:`Candidate` is a partial trajectory constraint, not a macro block;
* the composer searches only moves requested by candidate frontiers.

The atlas extractor is an oracle diagnostic.  ``generate_blind_candidates``
does not receive solution words and is the corresponding blind baseline.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .cubie_effect import CubieEffect
from .multi_core_exact7_screening_v75 import q2_commute_reduce_normal_form
from .piece_route_graph import FACE_MOVES, PieceState, SinglePieceRouteGraph
from .pieces import CORNER_ORDER, EDGE_ORDER
from .sequences import reduce_sequence
from .transformations import PIECES, from_sequence


_PIECE_INDEX = {piece: index for index, piece in enumerate(PIECES)}
_CORNER_INDEX = {piece: index for index, piece in enumerate(CORNER_ORDER)}
_EDGE_INDEX = {piece: index for index, piece in enumerate(EDGE_ORDER)}
_MOVE_INDEX = {move: index for index, move in enumerate(FACE_MOVES)}
_MOVE_EFFECTS = {
    move: CubieEffect.from_transformation(from_sequence((move,)))
    for move in FACE_MOVES
}
_IDENTITY = CubieEffect.identity()


def effect_after(effect: CubieEffect, move: str) -> CubieEffect:
    return _MOVE_EFFECTS[move].compose_after(effect)


def piece_state(effect: CubieEffect, piece: str) -> PieceState:
    destination = PIECES[effect.destinations[_PIECE_INDEX[piece]]]
    if piece in _CORNER_INDEX:
        orientation = effect.corner_twists[_CORNER_INDEX[piece]]
    else:
        orientation = effect.edge_flips[_EDGE_INDEX[piece]]
    return PieceState(destination, orientation)


def piece_is_solved(effect: CubieEffect, piece: str) -> bool:
    return piece_state(effect, piece) == PieceState(piece, 0)


def sequence_solves(scramble: Sequence[str], solution: Sequence[str]) -> bool:
    effect = CubieEffect.from_transformation(from_sequence(tuple(scramble)))
    for move in solution:
        effect = effect_after(effect, move)
    return effect == _IDENTITY


@dataclass(frozen=True, slots=True)
class AtlasSolution:
    case_id: str
    scramble: tuple[str, ...]
    exact_distance: int
    solution: tuple[str, ...]
    family_ids: tuple[str, ...] = ()
    provenance: str = ""

    def verify(self) -> None:
        if not sequence_solves(self.scramble, self.solution):
            raise AssertionError(f"atlas witness does not solve {self.case_id}: {self.solution}")


@dataclass(slots=True)
class SolutionAtlas:
    rows: list[AtlasSolution] = field(default_factory=list)

    def add(self, row: AtlasSolution) -> None:
        row.verify()
        if row not in self.rows:
            self.rows.append(row)

    def cases(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(row.case_id for row in self.rows))

    def for_case(self, case_id: str) -> tuple[AtlasSolution, ...]:
        return tuple(row for row in self.rows if row.case_id == case_id)

    @classmethod
    def load_v76_cores(
        cls,
        path: str | Path,
        *,
        case_id: str,
        scramble: Sequence[str],
        exact_distance: int,
        max_length: int = 10,
        limit: int | None = None,
    ) -> "SolutionAtlas":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        selected = [row for row in payload if row["length"] <= max_length]
        selected.sort(key=lambda row: (row["length"], row["core_id"]))
        if limit is not None:
            selected = selected[:limit]
        atlas = cls()
        for row in selected:
            atlas.add(AtlasSolution(
                case_id=case_id,
                scramble=tuple(scramble),
                exact_distance=exact_distance,
                solution=tuple(row["solution"]),
                family_ids=tuple(row.get("route_labels", ())) + (row["core_id"],),
                provenance=f"{path}:{row['core_id']}",
            ))
        return atlas

    @classmethod
    def load_v78_benchmarks(
        cls,
        path: str | Path,
        *,
        selected_labels: Iterable[str] | None = None,
    ) -> "SolutionAtlas":
        wanted = None if selected_labels is None else set(selected_labels)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        atlas = cls()
        for row in payload:
            if wanted is not None and row["label"] not in wanted:
                continue
            witnesses = {tuple(row["certified_witness"])}
            for result in row.get("strategies", {}).values():
                if result.get("solved") and result.get("solution"):
                    witnesses.add(tuple(result["solution"]))
            for index, solution in enumerate(sorted(witnesses, key=lambda word: (len(word), word))):
                atlas.add(AtlasSolution(
                    case_id=row["label"],
                    scramble=tuple(row["scramble"]),
                    exact_distance=row["exact_distance"],
                    solution=solution,
                    family_ids=(f"v78-witness-{index}",),
                    provenance=f"{path}:{row['label']}",
                ))
        return atlas


@dataclass(frozen=True, slots=True)
class PieceTrajectory:
    piece: str
    states: tuple[PieceState, ...]
    active_move_indices: tuple[int, ...]
    active_subsequence: tuple[str, ...]
    active_states: tuple[PieceState, ...]
    first_solved_time: int | None
    broken_again_times: tuple[int, ...]
    final_lock_time: int | None


@dataclass(frozen=True, slots=True)
class SolutionTrace:
    atlas_row: AtlasSolution
    piece_trajectories: tuple[PieceTrajectory, ...]
    active_pieces_by_move: tuple[tuple[str, ...], ...]

    def trajectory(self, piece: str) -> PieceTrajectory:
        return next(row for row in self.piece_trajectories if row.piece == piece)


def _first_and_final_lock(states: Sequence[PieceState], piece: str) -> tuple[int | None, tuple[int, ...], int | None]:
    solved = PieceState(piece, 0)
    solved_times = [index for index, state in enumerate(states) if state == solved]
    if not solved_times:
        return None, (), None
    first = solved_times[0]
    broken = tuple(
        index
        for index in range(first + 1, len(states))
        if states[index - 1] == solved and states[index] != solved
    )
    final_lock = next(
        index
        for index in range(len(states))
        if states[index] == solved and all(state == solved for state in states[index:])
    )
    return first, broken, final_lock


def trace_solution(row: AtlasSolution) -> SolutionTrace:
    row.verify()
    effect = CubieEffect.from_transformation(from_sequence(row.scramble))
    all_states: dict[str, list[PieceState]] = {
        piece: [piece_state(effect, piece)] for piece in PIECES
    }
    active_indices: dict[str, list[int]] = {piece: [] for piece in PIECES}
    active_words: dict[str, list[str]] = {piece: [] for piece in PIECES}
    active_pieces_by_move: list[tuple[str, ...]] = []

    for move_index, move in enumerate(row.solution):
        after = effect_after(effect, move)
        active: list[str] = []
        for piece in PIECES:
            before_state = all_states[piece][-1]
            after_state = piece_state(after, piece)
            all_states[piece].append(after_state)
            if after_state != before_state:
                active.append(piece)
                active_indices[piece].append(move_index)
                active_words[piece].append(move)
        active_pieces_by_move.append(tuple(active))
        effect = after

    trajectories: list[PieceTrajectory] = []
    for piece in PIECES:
        states = tuple(all_states[piece])
        indices = tuple(active_indices[piece])
        active_states = (states[0],) + tuple(states[index + 1] for index in indices)
        first, broken, final_lock = _first_and_final_lock(states, piece)
        trajectories.append(PieceTrajectory(
            piece=piece,
            states=states,
            active_move_indices=indices,
            active_subsequence=tuple(active_words[piece]),
            active_states=active_states,
            first_solved_time=first,
            broken_again_times=broken,
            final_lock_time=final_lock,
        ))
    return SolutionTrace(row, tuple(trajectories), tuple(active_pieces_by_move))


@dataclass(frozen=True, slots=True)
class PieceRequirement:
    piece: str
    state: PieceState


@dataclass(frozen=True, slots=True)
class GapConstraint:
    interval: int
    state: PieceState
    allowed_faces: tuple[str, ...]
    rule: str = "move must leave every target cubie's local state unchanged"


@dataclass(frozen=True, slots=True)
class ShareableMoveAnnotation:
    active_index: int
    move: str
    before: PieceState
    after: PieceState


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    source_grammar: str
    target_piece_set: tuple[str, ...]
    optional_support_piece_set: tuple[str, ...]
    start_preconditions: tuple[PieceRequirement, ...]
    target_requirements: tuple[PieceRequirement, ...]
    guaranteed_terminal_piece_state: tuple[PieceRequirement, ...]
    witness_sequence: tuple[str, ...]
    required_active_subsequence: tuple[str, ...]
    active_move_indices: tuple[int, ...]
    gap_constraints: tuple[GapConstraint, ...]
    allowed_gap_faces_by_interval: tuple[tuple[str, ...], ...]
    state_dependent_gap_rules: tuple[str, ...]
    affected_piece_set: tuple[str, ...]
    preserved_piece_set: tuple[str, ...]
    interference_footprint: tuple[str, ...]
    equivalent_witness_ids: tuple[str, ...]
    shareable_move_annotations: tuple[ShareableMoveAnnotation, ...]
    boundary_signature: tuple[str | None, str | None]
    execution_metadata: Mapping[str, object]
    provenance: Mapping[str, object]
    target_state_trajectories: tuple[tuple[str, tuple[PieceState, ...]], ...]

    def trajectory(self, piece: str) -> tuple[PieceState, ...]:
        return next(states for name, states in self.target_state_trajectories if name == piece)


def _affected_by_word(sequence: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    effect = CubieEffect.from_transformation(from_sequence(tuple(sequence)))
    affected = tuple(piece for piece in PIECES if not piece_is_solved(effect, piece))
    return affected, tuple(piece for piece in PIECES if piece not in affected)


def _gap_constraints_for_trajectory(states: Sequence[PieceState]) -> tuple[GapConstraint, ...]:
    rows = []
    for index, state in enumerate(states):
        allowed = tuple(face for face in "RUFLDB" if face not in state.position)
        rows.append(GapConstraint(index, state, allowed))
    return tuple(rows)


def candidate_from_piece_trajectory(
    trajectory: PieceTrajectory,
    *,
    candidate_id: str,
    source_grammar: str,
    provenance: Mapping[str, object],
) -> Candidate:
    piece = trajectory.piece
    required = trajectory.active_subsequence
    states = trajectory.active_states
    affected, preserved = _affected_by_word(required)
    gaps = _gap_constraints_for_trajectory(states)
    return Candidate(
        candidate_id=candidate_id,
        source_grammar=source_grammar,
        target_piece_set=(piece,),
        optional_support_piece_set=tuple(p for p in affected if p != piece),
        start_preconditions=(PieceRequirement(piece, states[0]),),
        target_requirements=(PieceRequirement(piece, PieceState(piece, 0)),),
        guaranteed_terminal_piece_state=(PieceRequirement(piece, states[-1]),),
        witness_sequence=required,
        required_active_subsequence=required,
        active_move_indices=trajectory.active_move_indices,
        gap_constraints=gaps,
        allowed_gap_faces_by_interval=tuple(gap.allowed_faces for gap in gaps),
        state_dependent_gap_rules=("inactive iff full replay leaves target local state unchanged",),
        affected_piece_set=affected,
        preserved_piece_set=preserved,
        interference_footprint=tuple(p for p in affected if p != piece),
        equivalent_witness_ids=(),
        shareable_move_annotations=tuple(
            ShareableMoveAnnotation(index, move, states[index], states[index + 1])
            for index, move in enumerate(required)
        ),
        boundary_signature=(required[0] if required else None, required[-1] if required else None),
        execution_metadata={
            "active_length": len(required),
            "original_active_indices": list(trajectory.active_move_indices),
        },
        provenance=dict(provenance),
        target_state_trajectories=((piece, states),),
    )


def oracle_candidates(traces: Sequence[SolutionTrace]) -> tuple[Candidate, ...]:
    """Extract and deduplicate per-piece constraints from atlas solutions."""
    grouped: dict[tuple, list[tuple[SolutionTrace, PieceTrajectory]]] = defaultdict(list)
    for trace in traces:
        for trajectory in trace.piece_trajectories:
            key = (
                trajectory.piece,
                trajectory.active_subsequence,
                trajectory.active_states,
            )
            grouped[key].append((trace, trajectory))

    candidates: list[Candidate] = []
    for index, key in enumerate(sorted(grouped, key=repr)):
        occurrences = grouped[key]
        trace, trajectory = occurrences[0]
        candidate = candidate_from_piece_trajectory(
            trajectory,
            candidate_id=f"oracle-{index:04d}",
            source_grammar="solution-atlas-active-trajectory",
            provenance={
                "mode": "oracle-assisted-only",
                "case_ids": sorted({row.atlas_row.case_id for row, _ in occurrences}),
                "solutions": [list(row.atlas_row.solution) for row, _ in occurrences],
            },
        )
        candidates.append(candidate)
    return link_equivalent_witnesses(tuple(candidates))


def link_equivalent_witnesses(candidates: Sequence[Candidate]) -> tuple[Candidate, ...]:
    """Annotate target-effect alternatives without changing candidate identity.

    This is the C5 supply layer.  A local target-effect class is deliberately
    weaker than an exact-trajectory class; the full composer still verifies
    every substituted witness against its own state trajectory and finally
    replays the complete cube.
    """
    groups: dict[tuple, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        key = (
            candidate.target_piece_set,
            candidate.start_preconditions,
            candidate.target_requirements,
            candidate.guaranteed_terminal_piece_state,
        )
        groups[key].append(candidate)
    equivalents = {
        candidate.candidate_id: tuple(
            other.candidate_id
            for other in group
            if other.candidate_id != candidate.candidate_id
        )
        for group in groups.values()
        for candidate in group
    }
    return tuple(
        replace(candidate, equivalent_witness_ids=equivalents[candidate.candidate_id])
        for candidate in candidates
    )


def _local_active_routes(
    graph: SinglePieceRouteGraph,
    source: PieceState,
    target: PieceState,
    *,
    slack: int,
    max_witnesses: int,
    expansion_cap: int = 100_000,
) -> tuple[tuple[str, ...], ...]:
    """Enumerate short local trajectories, including non-empty solved loops.

    Only moves active on the tracked cubie are emitted.  Consecutive inverse
    requirements are retained because another candidate may occupy their gap.
    """
    minimum = graph.distance(source, target)
    maximum = minimum + slack
    queue = deque([(source, ())])
    results: set[tuple[str, ...]] = set()
    expanded = 0
    while queue and expanded < expansion_cap:
        state, path = queue.popleft()
        expanded += 1
        if state == target:
            results.add(path)
        if len(path) == maximum:
            continue
        for move in graph.moves:
            next_state = graph.transitions[state][move]
            if next_state == state:
                continue
            next_path = path + (move,)
            if len(next_path) + graph.distance(next_state, target) > maximum:
                continue
            queue.append((next_state, next_path))

    ordered = sorted(results, key=lambda word: (len(word), _word_key(word)))
    if len(ordered) <= max_witnesses:
        return tuple(ordered)
    # Preserve first-move diversity before filling from the deterministic rank.
    selected: list[tuple[str, ...]] = []
    seen_first: set[str | None] = set()
    for word in ordered:
        first = word[0] if word else None
        if first not in seen_first:
            selected.append(word)
            seen_first.add(first)
            if len(selected) == max_witnesses:
                return tuple(selected)
    for word in ordered:
        if word not in selected:
            selected.append(word)
            if len(selected) == max_witnesses:
                break
    return tuple(selected)


def _word_key(word: Sequence[str]) -> tuple[int, ...]:
    return tuple(_MOVE_INDEX[move] for move in word)


def generate_blind_candidates(
    scramble: Sequence[str],
    *,
    slack: int = 2,
    max_witnesses_per_piece: int = 12,
) -> tuple[Candidate, ...]:
    """Generate local candidates without consulting an atlas solution."""
    start = CubieEffect.from_transformation(from_sequence(tuple(scramble)))
    graphs = {
        "corner": SinglePieceRouteGraph("corner"),
        "edge": SinglePieceRouteGraph("edge"),
    }
    result: list[Candidate] = []
    serial = 0
    for piece in PIECES:
        graph = graphs["corner" if piece in _CORNER_INDEX else "edge"]
        source = piece_state(start, piece)
        target = PieceState(piece, 0)
        for word in _local_active_routes(
            graph,
            source,
            target,
            slack=slack,
            max_witnesses=max_witnesses_per_piece,
        ):
            states = [source]
            state = source
            for move in word:
                state = graph.transitions[state][move]
                states.append(state)
            trajectory = PieceTrajectory(
                piece=piece,
                states=tuple(states),
                active_move_indices=tuple(range(len(word))),
                active_subsequence=word,
                active_states=tuple(states),
                first_solved_time=None,
                broken_again_times=(),
                final_lock_time=None,
            )
            result.append(candidate_from_piece_trajectory(
                trajectory,
                candidate_id=f"blind-{serial:04d}",
                source_grammar="single-piece-active-route-bfs",
                provenance={
                    "mode": "blind",
                    "slack": slack,
                    "minimum_local_distance": graph.distance(source, target),
                },
            ))
            serial += 1
    return link_equivalent_witnesses(tuple(result))


def candidate_match_level(candidate: Candidate, trace: SolutionTrace) -> int | None:
    if len(candidate.target_piece_set) != 1:
        return None
    piece = candidate.target_piece_set[0]
    observed = trace.trajectory(piece)
    witness = candidate.witness_sequence
    full = trace.atlas_row.solution
    if witness and any(tuple(full[index:index + len(witness)]) == witness for index in range(len(full) - len(witness) + 1)):
        return 0
    if (
        candidate.required_active_subsequence == observed.active_subsequence
        and candidate.trajectory(piece) == observed.active_states
    ):
        return 1
    if candidate.trajectory(piece) == observed.active_states:
        return 2
    if (
        candidate.start_preconditions[0].state == observed.states[0]
        and candidate.guaranteed_terminal_piece_state[0].state == observed.states[-1]
    ):
        return 3
    return None


@dataclass(frozen=True, slots=True)
class CoverageReport:
    solutions: int
    piece_requirements: int
    matched_by_level: tuple[int, int, int, int]
    covered_level1: int
    covered_level3: int
    candidate_coverage: float
    terminal_effect_coverage: float
    family_count: int
    families_covered_level1: int


def analyze_candidate_coverage(
    candidates: Sequence[Candidate],
    traces: Sequence[SolutionTrace],
) -> CoverageReport:
    levels = [0, 0, 0, 0]
    level1 = 0
    level3 = 0
    covered_families: set[str] = set()
    all_families = {family for trace in traces for family in trace.atlas_row.family_ids}
    by_piece: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        for piece in candidate.target_piece_set:
            by_piece[piece].append(candidate)
    for trace in traces:
        solution_level1 = True
        for piece in PIECES:
            matches = [
                level
                for candidate in by_piece[piece]
                if (level := candidate_match_level(candidate, trace)) is not None
            ]
            if matches:
                levels[min(matches)] += 1
                level3 += 1
            if not matches or min(matches) > 1:
                solution_level1 = False
            else:
                level1 += 1
        if solution_level1:
            covered_families.update(trace.atlas_row.family_ids)
    total = len(traces) * len(PIECES)
    return CoverageReport(
        solutions=len(traces),
        piece_requirements=total,
        matched_by_level=tuple(levels),  # type: ignore[arg-type]
        covered_level1=level1,
        covered_level3=level3,
        candidate_coverage=level1 / total if total else 0.0,
        terminal_effect_coverage=level3 / total if total else 0.0,
        family_count=len(all_families),
        families_covered_level1=len(covered_families),
    )


@dataclass(frozen=True, slots=True)
class CandidateProgress:
    candidate_index: int
    progress: int


DomainState = tuple[tuple[CandidateProgress, ...], ...]


@dataclass(frozen=True, slots=True)
class ComposedStep:
    move: str
    active_pieces: tuple[str, ...]
    advanced_candidates: tuple[str, ...]
    shared_move: bool
    inactive_gap_pieces: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompositionResult:
    stage: str
    solved: bool
    raw_sequence: tuple[str, ...]
    normalized_sequence: tuple[str, ...]
    nodes: int
    generated: int
    shared_move_count: int
    shared_move_ratio: float
    boundary_reduction: int
    steps: tuple[ComposedStep, ...]
    completed_piece_count: int
    residual_effect: CubieEffect
    budget_exhausted: bool
    failure_type: str | None


def _candidate_groups(
    candidates: Sequence[Candidate],
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    by_piece: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        if len(candidate.target_piece_set) != 1:
            continue
        by_piece[candidate.target_piece_set[0]].append(index)
    constrained = tuple(piece for piece in PIECES if by_piece[piece])
    return constrained, tuple(tuple(by_piece[piece]) for piece in constrained)


def _initial_domains(
    effect: CubieEffect,
    candidates: Sequence[Candidate],
    constrained_pieces: Sequence[str],
    groups: Sequence[Sequence[int]],
) -> DomainState | None:
    domains = []
    for piece, group in zip(constrained_pieces, groups):
        actual = piece_state(effect, piece)
        domain = tuple(
            CandidateProgress(index, 0)
            for index in group
            if candidates[index].trajectory(piece)[0] == actual
        )
        if not domain:
            return None
        domains.append(domain)
    return tuple(domains)


def _next_moves(domains: DomainState, candidates: Sequence[Candidate]) -> tuple[str, ...]:
    moves = {
        candidate.required_active_subsequence[item.progress]
        for domain in domains
        for item in domain
        for candidate in (candidates[item.candidate_index],)
        if item.progress < len(candidate.required_active_subsequence)
    }
    return tuple(sorted(moves, key=_MOVE_INDEX.__getitem__))


def _advance_domains(
    effect: CubieEffect,
    move: str,
    domains: DomainState,
    candidates: Sequence[Candidate],
    constrained_pieces: Sequence[str],
    *,
    allow_shared: bool,
    fixed_owner_piece: int | None,
) -> tuple[CubieEffect, DomainState, ComposedStep] | None:
    after = effect_after(effect, move)
    next_domains: list[tuple[CandidateProgress, ...]] = []
    active_pieces: list[str] = []
    advanced_ids: set[str] = set()
    inactive_gap_pieces: list[str] = []
    for piece_index, (piece, domain) in enumerate(zip(constrained_pieces, domains)):
        before_state = piece_state(effect, piece)
        after_state = piece_state(after, piece)
        if before_state == after_state:
            next_domains.append(domain)
            if any(
                item.progress < len(candidates[item.candidate_index].required_active_subsequence)
                for item in domain
            ):
                inactive_gap_pieces.append(piece)
            continue
        active_pieces.append(piece)
        advanced: list[CandidateProgress] = []
        for item in domain:
            candidate = candidates[item.candidate_index]
            required = candidate.required_active_subsequence
            states = candidate.trajectory(piece)
            if item.progress >= len(required) or required[item.progress] != move:
                continue
            if states[item.progress] != before_state or states[item.progress + 1] != after_state:
                continue
            advanced.append(CandidateProgress(item.candidate_index, item.progress + 1))
            advanced_ids.add(candidate.candidate_id)
        if not advanced:
            return None
        next_domains.append(tuple(advanced))

    if not active_pieces:
        return None
    if not allow_shared and len(active_pieces) != 1:
        return None
    if fixed_owner_piece is not None:
        if len(active_pieces) != 1 or constrained_pieces[fixed_owner_piece] != active_pieces[0]:
            return None
    step = ComposedStep(
        move=move,
        active_pieces=tuple(active_pieces),
        advanced_candidates=tuple(sorted(advanced_ids)),
        shared_move=len(active_pieces) > 1,
        inactive_gap_pieces=tuple(inactive_gap_pieces),
    )
    return after, tuple(next_domains), step


def _completed_pieces(domains: DomainState, candidates: Sequence[Candidate]) -> int:
    return sum(
        any(
            item.progress == len(candidates[item.candidate_index].required_active_subsequence)
            for item in domain
        )
        for domain in domains
    )


def _remaining_lower_bound(domains: DomainState, candidates: Sequence[Candidate]) -> int:
    return max(
        (
            min(
                len(candidates[item.candidate_index].required_active_subsequence) - item.progress
                for item in domain
            )
            for domain in domains
        ),
        default=0,
    )


def compose_candidates(
    scramble: Sequence[str],
    candidates: Sequence[Candidate],
    *,
    stage: str,
    max_depth: int,
    node_limit: int = 500_000,
) -> CompositionResult:
    """Search a common global word for a candidate portfolio.

    ``C2`` uses a fixed first-unfinished-piece owner, ``C3`` permits any
    partial-order frontier but no shared active transition, and ``C4`` permits
    one physical turn to satisfy multiple candidate requirements.
    """
    if stage not in {"C2", "C3", "C4", "C5"}:
        raise ValueError("constraint composer stage must be C2, C3, C4, or C5")
    start = CubieEffect.from_transformation(from_sequence(tuple(scramble)))
    constrained_pieces, groups = _candidate_groups(candidates)
    domains = _initial_domains(start, candidates, constrained_pieces, groups)
    if domains is None:
        return CompositionResult(
            stage, False, (), (), 0, 0, 0, 0.0, 0, (), 0, start, False, "F2",
        )

    root_key = (start.signature, domains, "")
    queue = deque([(start, domains, (), (), "")])
    seen = {root_key}
    nodes = 0
    generated = 0
    best = (start, domains, (), ())
    exhausted = False
    while queue:
        effect, current_domains, path, steps, last_face = queue.popleft()
        nodes += 1
        if nodes > node_limit:
            exhausted = True
            break
        completed = _completed_pieces(current_domains, candidates)
        if completed > _completed_pieces(best[1], candidates):
            best = (effect, current_domains, path, steps)
        if effect == _IDENTITY and completed == len(constrained_pieces):
            normalized = q2_commute_reduce_normal_form(reduce_sequence(path))
            if not sequence_solves(scramble, normalized):
                normalized = reduce_sequence(path)
            shared = sum(step.shared_move for step in steps)
            return CompositionResult(
                stage=stage,
                solved=True,
                raw_sequence=path,
                normalized_sequence=normalized,
                nodes=nodes,
                generated=generated,
                shared_move_count=shared,
                shared_move_ratio=shared / len(path) if path else 0.0,
                boundary_reduction=len(path) - len(normalized),
                steps=steps,
                completed_piece_count=completed,
                residual_effect=effect,
                budget_exhausted=False,
                failure_type=None,
            )
        if len(path) >= max_depth or len(path) + _remaining_lower_bound(current_domains, candidates) > max_depth:
            continue

        fixed_owner = None
        if stage == "C2":
            fixed_owner = next(
                (
                    index
                    for index, domain in enumerate(current_domains)
                    if not any(
                        item.progress == len(candidates[item.candidate_index].required_active_subsequence)
                        for item in domain
                    )
                ),
                None,
            )
        for move in _next_moves(current_domains, candidates):
            if last_face and move[0] == last_face:
                continue
            advanced = _advance_domains(
                effect,
                move,
                current_domains,
                candidates,
                constrained_pieces,
                allow_shared=stage in {"C4", "C5"},
                fixed_owner_piece=fixed_owner,
            )
            if advanced is None:
                continue
            after, next_domain, step = advanced
            generated += 1
            key = (after.signature, next_domain, move[0])
            if key in seen:
                continue
            seen.add(key)
            queue.append((after, next_domain, path + (move,), steps + (step,), move[0]))

    effect, best_domains, path, steps = best
    completed = _completed_pieces(best_domains, candidates)
    failure = "F8" if exhausted else ("F4" if stage == "C2" else "F5" if stage == "C3" else "F9")
    normalized = reduce_sequence(path)
    return CompositionResult(
        stage=stage,
        solved=False,
        raw_sequence=path,
        normalized_sequence=normalized,
        nodes=nodes,
        generated=generated,
        shared_move_count=sum(step.shared_move for step in steps),
        shared_move_ratio=(sum(step.shared_move for step in steps) / len(path) if path else 0.0),
        boundary_reduction=len(path) - len(normalized),
        steps=steps,
        completed_piece_count=completed,
        residual_effect=effect,
        budget_exhausted=exhausted,
        failure_type=failure,
    )


def block_baseline(
    scramble: Sequence[str],
    candidates: Sequence[Candidate],
    *,
    stage: str = "C0",
    ordering_limit: int = 20_000,
) -> CompositionResult:
    """C0/C1 baseline over one shortest candidate per piece.

    C1 uses a bounded best-first block ordering; neither mode interleaves a
    block.  It exists to reproduce the old composition limitation, not as the
    new engine's backbone.
    """
    if stage not in {"C0", "C1"}:
        raise ValueError("block baseline stage must be C0 or C1")
    constrained_pieces, groups = _candidate_groups(candidates)
    if not groups or any(not group for group in groups):
        start = CubieEffect.from_transformation(from_sequence(tuple(scramble)))
        return CompositionResult(stage, False, (), (), 0, 0, 0, 0.0, 0, (), 0, start, False, "F1")
    chosen = [
        min(
            (candidates[index] for index in group),
            key=lambda candidate: (len(candidate.witness_sequence), candidate.candidate_id),
        )
        for group in groups
    ]

    ordering_expanded = 0
    if stage == "C0":
        orders = [tuple(range(len(chosen)))]
    else:
        # Beam over block orders, preferring words that solve more pieces.
        start_effect = CubieEffect.from_transformation(from_sequence(tuple(scramble)))
        frontier = [(start_effect, (), tuple(range(len(chosen))))]
        completed_orders: list[tuple[int, ...]] = []
        expanded = 0
        while frontier and expanded < ordering_limit:
            effect, order, remaining = frontier.pop(0)
            expanded += 1
            if not remaining:
                completed_orders.append(order)
                if effect == _IDENTITY:
                    break
                continue
            children = []
            for index in remaining:
                after = effect
                for move in chosen[index].witness_sequence:
                    after = effect_after(after, move)
                rest = tuple(item for item in remaining if item != index)
                score = -sum(piece_is_solved(after, piece) for piece in PIECES)
                children.append((score, after, order + (index,), rest))
            children.sort(key=lambda row: (row[0], row[2]))
            frontier.extend((after, order, rest) for _, after, order, rest in children[:3])
            frontier = frontier[:128]
        ordering_expanded = expanded
        orders = completed_orders or [tuple(range(len(chosen)))]

    nodes = 0
    best_word: tuple[str, ...] = ()
    best_effect = CubieEffect.from_transformation(from_sequence(tuple(scramble)))
    best_solved = sum(piece_is_solved(best_effect, piece) for piece in PIECES)
    for order in orders:
        nodes += 1
        word = tuple(move for index in order for move in chosen[index].witness_sequence)
        effect = CubieEffect.from_transformation(from_sequence(tuple(scramble) + word))
        solved_count = sum(piece_is_solved(effect, piece) for piece in PIECES)
        if solved_count > best_solved:
            best_word, best_effect, best_solved = word, effect, solved_count
        if effect == _IDENTITY:
            best_word, best_effect, best_solved = word, effect, len(PIECES)
            break
    normalized = q2_commute_reduce_normal_form(reduce_sequence(best_word))
    solved = sequence_solves(scramble, normalized)
    return CompositionResult(
        stage=stage,
        solved=solved,
        raw_sequence=best_word,
        normalized_sequence=normalized,
        nodes=nodes + ordering_expanded,
        generated=nodes + ordering_expanded,
        shared_move_count=0,
        shared_move_ratio=0.0,
        boundary_reduction=len(best_word) - len(normalized),
        steps=(),
        completed_piece_count=best_solved,
        residual_effect=best_effect,
        budget_exhausted=stage == "C1" and ordering_expanded >= ordering_limit,
        failure_type=None if solved else "F3",
    )


def classify_failure(
    *,
    candidate_coverage: float,
    result: CompositionResult,
    oracle_representable: bool,
) -> tuple[str, tuple[str, ...]]:
    if result.solved:
        return "F10" if not oracle_representable else "SUCCESS", ()
    secondary: list[str] = []
    if candidate_coverage < 1.0:
        primary = "F1"
    elif result.failure_type == "F2":
        primary = "F2"
    elif result.budget_exhausted:
        primary = "F8"
    else:
        primary = result.failure_type or "F9"
    if result.completed_piece_count >= len(PIECES) - 4:
        secondary.append("F11")
    if oracle_representable and primary in {"F4", "F5", "F9"}:
        secondary.append("search_or_stage_deficiency")
    return primary, tuple(secondary)


def normalize_composition(
    scramble: Sequence[str],
    result: CompositionResult,
) -> CompositionResult:
    """C6 normalization with effect-preserving fallback and attribution kept."""
    normalized = q2_commute_reduce_normal_form(reduce_sequence(result.raw_sequence))
    raw_effect = CubieEffect.from_transformation(
        from_sequence(tuple(scramble) + result.raw_sequence)
    )
    normalized_effect = CubieEffect.from_transformation(
        from_sequence(tuple(scramble) + normalized)
    )
    if normalized_effect != raw_effect:
        normalized = reduce_sequence(result.raw_sequence)
    return replace(
        result,
        stage="C6",
        normalized_sequence=normalized,
        boundary_reduction=len(result.raw_sequence) - len(normalized),
    )


def apply_small_residual_fallback(
    scramble: Sequence[str],
    result: CompositionResult,
    *,
    max_distance: int = 4,
) -> CompositionResult:
    """C7 exact residual closure, reported separately from grammar-only solve."""
    if max_distance < 0 or max_distance > 4:
        raise ValueError("the POC fallback is intentionally capped at distance 4")
    if result.solved:
        return replace(result, stage="C7")
    from .exact_mitm_oracle_v63 import ExactMitmOracleV63

    residual_scramble = tuple(scramble) + result.raw_sequence
    probe = ExactMitmOracleV63(backward_depth=min(4, max_distance)).solve(
        residual_scramble,
        bound=max_distance,
        max_solutions=8,
    )
    if not probe.solved:
        return replace(result, stage="C7", failure_type="F11")
    raw = result.raw_sequence + probe.sequence
    normalized = q2_commute_reduce_normal_form(reduce_sequence(raw))
    if not sequence_solves(scramble, normalized):
        normalized = reduce_sequence(raw)
    if not sequence_solves(scramble, normalized):
        raise AssertionError("exact residual witness failed full-state replay")
    return CompositionResult(
        stage="C7",
        solved=True,
        raw_sequence=raw,
        normalized_sequence=normalized,
        nodes=result.nodes + probe.forward_expanded,
        generated=result.generated,
        shared_move_count=result.shared_move_count,
        shared_move_ratio=result.shared_move_count / len(raw) if raw else 0.0,
        boundary_reduction=len(raw) - len(normalized),
        steps=result.steps,
        completed_piece_count=len(PIECES),
        residual_effect=_IDENTITY,
        budget_exhausted=False,
        failure_type=None,
    )


__all__ = [
    "AtlasSolution",
    "SolutionAtlas",
    "PieceTrajectory",
    "SolutionTrace",
    "PieceRequirement",
    "GapConstraint",
    "ShareableMoveAnnotation",
    "Candidate",
    "CoverageReport",
    "ComposedStep",
    "CompositionResult",
    "trace_solution",
    "oracle_candidates",
    "link_equivalent_witnesses",
    "generate_blind_candidates",
    "candidate_match_level",
    "analyze_candidate_coverage",
    "compose_candidates",
    "block_baseline",
    "classify_failure",
    "normalize_composition",
    "apply_small_residual_fallback",
    "piece_state",
    "sequence_solves",
]
