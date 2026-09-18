from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import math
import random
import time
from typing import Iterable, Sequence

from .cubie_effect import CubieEffect
from .dense_projection_search_v64 import DenseProjectionRepository
from .dense_selected_quad_projection_v70 import DenseQuadPolicyBankV70
from .exact_mitm_oracle_v63 import ExactMitmOracleV63
from .intermediate_manifold_benchmark_v73 import structural_features
from .piece_route_graph import FACE_MOVES, inverse_sequence
from .pieces import CORNER_ORDER, EDGE_ORDER
from .rotations import RotationCanonicalizer
from .solution_space_atlas_v74 import AllWitnessSolutionEnumeratorV74
from .transformations import PIECES, from_sequence


_AXIS = {"R": 0, "L": 0, "U": 1, "D": 1, "F": 2, "B": 2}
_AXIS_FACE_ORDER = {0: ("L", "R"), 1: ("D", "U"), 2: ("B", "F")}
_OPPOSITE = {"R": "L", "L": "R", "U": "D", "D": "U", "F": "B", "B": "F"}
_MOVE_INDEX = {move: index for index, move in enumerate(FACE_MOVES)}
_FIXED_QUADS = (
    ("UBR", "UF", "BL", "DL"),
    ("DBL", "UF", "BL", "DL"),
    ("UF", "UB", "BL", "DL"),
)


def _move_amount(move: str) -> int:
    return 2 if move.endswith("2") else 3 if move.endswith("'") else 1


def _move_token(face: str, amount: int) -> str:
    amount %= 4
    if amount == 1:
        return face
    if amount == 2:
        return face + "2"
    if amount == 3:
        return face + "'"
    raise ValueError("amount zero is the identity, not a move token")


def q2_commute_reduce_normal_form(sequence: Sequence[str]) -> tuple[str, ...]:
    """Canonical form for same-face reduction plus opposite-face commuting.

    The rewrite group is the free product of the three axis groups
    ``<R,L>``, ``<U,D>``, and ``<F,B>``.  Each axis group is ``C4 x C4``:
    opposite faces commute and each face amount adds modulo four.  Reducing a
    block to identity exposes neighboring blocks, so a stack gives the unique
    normal form without enumerating a commutation closure.
    """

    stack: list[tuple[int, dict[str, int]]] = []
    for move in sequence:
        face = move[0]
        axis = _AXIS[face]
        amount = _move_amount(move)
        if stack and stack[-1][0] == axis:
            values = stack[-1][1]
            values[face] = (values.get(face, 0) + amount) % 4
            if all(value % 4 == 0 for value in values.values()):
                stack.pop()
        else:
            stack.append((axis, {face: amount}))

    result: list[str] = []
    for axis, values in stack:
        for face in _AXIS_FACE_ORDER[axis]:
            amount = values.get(face, 0) % 4
            if amount:
                result.append(_move_token(face, amount))
    return tuple(result)


def _word_key(word: Sequence[str]) -> tuple[int, ...]:
    return tuple(_MOVE_INDEX[move] for move in word)


def _effect_id(effect: CubieEffect) -> str:
    return sha256(repr(effect.signature).encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True, slots=True)
class ExactSevenCandidateV75:
    candidate_id: str
    scramble: tuple[str, ...]
    exact_distance: int
    shortest_witness: tuple[str, ...]
    state_effect_id: str
    profile: dict


@dataclass(frozen=True, slots=True)
class CandidateGenerationStatsV75:
    requested: int
    generated: int
    attempts: int
    rejected_face_usage: int
    rejected_full_state_duplicate: int
    rejected_rotation_duplicate: int
    rejected_distance_below_seven: int
    exact_six_oracle_seconds: float
    total_seconds: float
    fixed_quads: tuple[tuple[str, ...], ...]
    fixed_quad_build_seconds: float


@dataclass(frozen=True, slots=True)
class CoreSummaryV75:
    core_id: str
    length: int
    solution: tuple[str, ...]
    first_move: str
    exact_word_count: int
    observed_lengths: tuple[tuple[int, int], ...]
    representative_witness: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateScreenResultV75:
    candidate_id: str
    scramble: tuple[str, ...]
    solution_counts: tuple[tuple[int, int], ...]
    raw_mitm_joins: tuple[tuple[int, int], ...]
    q2_classes_by_length: tuple[tuple[int, int], ...]
    new_q2_classes_by_length: tuple[tuple[int, int], ...]
    total_solution_count: int
    q2_class_count: int
    shortest_solution_count: int
    shortest_core_count: int
    productive_core_count: int
    distinct_core_first_moves: int
    trivial_inflation_only: bool
    passed: bool
    pass_reasons: tuple[str, ...]
    cores: tuple[CoreSummaryV75, ...]
    screening_seconds: float


def _face_usage_profile(scramble: Sequence[str]) -> dict:
    faces = [move[0] for move in scramble]
    axes = [_AXIS[face] for face in faces]
    return {
        "distinct_faces": sorted(set(faces)),
        "distinct_face_count": len(set(faces)),
        "distinct_axis_count": len(set(axes)),
        "half_turn_count": sum(move.endswith("2") for move in scramble),
        "quarter_turn_count": sum(not move.endswith("2") for move in scramble),
        "opposite_face_adjacencies": sum(
            _OPPOSITE[left] == right for left, right in zip(faces, faces[1:])
        ),
        "axis_changes": sum(left != right for left, right in zip(axes, axes[1:])),
        "face_histogram": dict(sorted(Counter(faces).items())),
    }


def _candidate_profile(
    scramble: Sequence[str],
    effect: CubieEffect,
    repository: DenseProjectionRepository,
    pair_policies,
    fixed_quad_policies,
) -> dict:
    codes = repository.encode_effect(effect)
    features = structural_features(effect, repository, pair_policies)
    rotation = RotationCanonicalizer.analyze(from_sequence(tuple(scramble)))
    solved_piece_count = (
        len(PIECES)
        - features["unresolved_corner_count"]
        - features["unresolved_edge_count"]
    )
    return {
        **_face_usage_profile(scramble),
        **features,
        "solved_piece_count": solved_piece_count,
        "corner_orientation_defect_support": sum(
            value != 0 for value in effect.corner_twists
        ),
        "edge_orientation_defect_support": sum(effect.edge_flips),
        "rotation_class_size": rotation.class_size,
        "rotation_stabilizer_size": 24 // rotation.class_size,
        "fixed_quad_lower_bound": max(
            (policy.distance(codes) for policy in fixed_quad_policies),
            default=0,
        ),
        "fixed_quad_pieces": [list(policy.pieces) for policy in fixed_quad_policies],
    }


def generate_exact_seven_candidates(
    *,
    count: int,
    seed: int,
    repository: DenseProjectionRepository,
    min_distinct_faces: int = 5,
) -> tuple[tuple[ExactSevenCandidateV75, ...], CandidateGenerationStatsV75]:
    if count < 1:
        raise ValueError("count must be positive")
    if not 1 <= min_distinct_faces <= 6:
        raise ValueError("min_distinct_faces must be in 1..6")
    started = time.perf_counter()
    rng = random.Random(seed)
    oracle = ExactMitmOracleV63(backward_depth=4)
    pair_policies = repository.all_policies()
    quad_bank = DenseQuadPolicyBankV70(
        repository,
        cutoff_depth=6,
        max_cached_tables=len(_FIXED_QUADS),
    )
    quad_started = time.perf_counter()
    fixed_quad_policies = tuple(quad_bank.policy(quad)[0] for quad in _FIXED_QUADS)
    fixed_quad_build_seconds = time.perf_counter() - quad_started

    attempts = 0
    rejected_face = rejected_full = rejected_rotation = rejected_short = 0
    oracle_seconds = 0.0
    seen_full: set[tuple] = set()
    seen_rotation: set[tuple] = set()
    candidates: list[ExactSevenCandidateV75] = []
    faces = tuple("RUFLDB")
    suffixes = ("", "'", "2")
    while len(candidates) < count:
        attempts += 1
        scramble: list[str] = []
        last_face = ""
        while len(scramble) < 7:
            face = rng.choice(faces)
            if face == last_face:
                continue
            scramble.append(face + rng.choice(suffixes))
            last_face = face
        sequence = tuple(scramble)
        if len({move[0] for move in sequence}) < min_distinct_faces:
            rejected_face += 1
            continue
        transformation = from_sequence(sequence)
        effect = CubieEffect.from_transformation(transformation)
        if effect.signature in seen_full:
            rejected_full += 1
            continue
        rotation_key = RotationCanonicalizer.analyze(transformation).canonical.signature
        if rotation_key in seen_rotation:
            rejected_rotation += 1
            continue
        proof_started = time.perf_counter()
        below = oracle.solve(sequence, bound=6)
        oracle_seconds += time.perf_counter() - proof_started
        if below.solved:
            rejected_short += 1
            continue
        witness = inverse_sequence(sequence)
        if CubieEffect.from_transformation(from_sequence(sequence + witness)) != CubieEffect.identity():
            raise AssertionError("inverse witness failed full-state replay")
        seen_full.add(effect.signature)
        seen_rotation.add(rotation_key)
        candidate_id = f"C{len(candidates):04d}"
        candidates.append(
            ExactSevenCandidateV75(
                candidate_id=candidate_id,
                scramble=sequence,
                exact_distance=7,
                shortest_witness=witness,
                state_effect_id=_effect_id(effect),
                profile=_candidate_profile(
                    sequence,
                    effect,
                    repository,
                    pair_policies,
                    fixed_quad_policies,
                ),
            )
        )
    return (
        tuple(candidates),
        CandidateGenerationStatsV75(
            requested=count,
            generated=len(candidates),
            attempts=attempts,
            rejected_face_usage=rejected_face,
            rejected_full_state_duplicate=rejected_full,
            rejected_rotation_duplicate=rejected_rotation,
            rejected_distance_below_seven=rejected_short,
            exact_six_oracle_seconds=oracle_seconds,
            total_seconds=time.perf_counter() - started,
            fixed_quads=tuple(tuple(policy.pieces) for policy in fixed_quad_policies),
            fixed_quad_build_seconds=fixed_quad_build_seconds,
        ),
    )


def screen_candidate(
    candidate: ExactSevenCandidateV75,
    *,
    enumerator: AllWitnessSolutionEnumeratorV74,
    lengths: Sequence[int] = (7, 8, 9, 10),
) -> CandidateScreenResultV75:
    started = time.perf_counter()
    requested = tuple(sorted(set(lengths)))
    if not requested or requested[0] < 7 or requested[-1] > 10:
        raise ValueError("screening lengths must be a non-empty subset of 7..10")

    by_core: dict[tuple[str, ...], list[tuple[int, tuple[str, ...]]]] = defaultdict(list)
    solution_counts: list[tuple[int, int]] = []
    raw_joins: list[tuple[int, int]] = []
    q2_by_length: list[tuple[int, int]] = []
    new_q2_by_length: list[tuple[int, int]] = []
    seen_cores: set[tuple[str, ...]] = set()
    shortest_solution_count = 0
    shortest_cores: set[tuple[str, ...]] = set()
    for length in requested:
        result = enumerator.enumerate_length(candidate.scramble, length)
        solution_counts.append((length, result.reduced_solution_count))
        raw_joins.append((length, result.raw_mitm_joins))
        current_cores: set[tuple[str, ...]] = set()
        for word in result.reduced_solutions:
            core = q2_commute_reduce_normal_form(word)
            current_cores.add(core)
            by_core[core].append((length, word))
        q2_by_length.append((length, len(current_cores)))
        new_q2_by_length.append((length, len(current_cores - seen_cores)))
        seen_cores.update(current_cores)
        if length == 7:
            shortest_solution_count = result.reduced_solution_count
            shortest_cores.update(current_cores)

    ordered_cores = sorted(by_core, key=lambda word: (len(word), _word_key(word)))
    core_rows: list[CoreSummaryV75] = []
    for index, core in enumerate(ordered_cores):
        witnesses = by_core[core]
        representative = min(
            (word for _, word in witnesses),
            key=lambda word: (len(word), _word_key(word)),
        )
        core_rows.append(
            CoreSummaryV75(
                core_id=f"Q2-{index:03d}",
                length=len(core),
                solution=core,
                first_move=core[0] if core else "",
                exact_word_count=len(witnesses),
                observed_lengths=tuple(sorted(Counter(length for length, _ in witnesses).items())),
                representative_witness=representative,
            )
        )

    productive = sum(len(core) > 7 for core in ordered_cores)
    total_q2 = len(ordered_cores)
    reasons: list[str] = []
    if len(shortest_cores) >= 2:
        reasons.append("multiple_shortest_q2_cores")
    if productive:
        reasons.append("productive_nonshortest_q2_core")
    if total_q2 >= 2:
        reasons.append("multiple_q2_cores")
    trivial_only = (
        total_q2 == 1
        and bool(ordered_cores)
        and len(ordered_cores[0]) == 7
        and productive == 0
    )
    return CandidateScreenResultV75(
        candidate_id=candidate.candidate_id,
        scramble=candidate.scramble,
        solution_counts=tuple(solution_counts),
        raw_mitm_joins=tuple(raw_joins),
        q2_classes_by_length=tuple(q2_by_length),
        new_q2_classes_by_length=tuple(new_q2_by_length),
        total_solution_count=sum(count for _, count in solution_counts),
        q2_class_count=total_q2,
        shortest_solution_count=shortest_solution_count,
        shortest_core_count=len(shortest_cores),
        productive_core_count=productive,
        distinct_core_first_moves=len({core[0] for core in ordered_cores if core}),
        trivial_inflation_only=trivial_only,
        passed=total_q2 >= 2,
        pass_reasons=tuple(reasons),
        cores=tuple(core_rows),
        screening_seconds=time.perf_counter() - started,
    )


_FORK_ENUMERATOR: AllWitnessSolutionEnumeratorV74 | None = None


def install_fork_enumerator(enumerator: AllWitnessSolutionEnumeratorV74) -> None:
    global _FORK_ENUMERATOR
    _FORK_ENUMERATOR = enumerator


def screen_candidate_in_fork(
    candidate: ExactSevenCandidateV75,
) -> CandidateScreenResultV75:
    if _FORK_ENUMERATOR is None:
        raise RuntimeError("fork enumerator was not installed before worker start")
    return screen_candidate(candidate, enumerator=_FORK_ENUMERATOR)


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    delta_left = [value - mean_left for value in left]
    delta_right = [value - mean_right for value in right]
    denominator = math.sqrt(
        sum(value * value for value in delta_left)
        * sum(value * value for value in delta_right)
    )
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(delta_left, delta_right)) / denominator


def _rate_table(
    candidates: Sequence[ExactSevenCandidateV75],
    results_by_id: dict[str, CandidateScreenResultV75],
    feature: str,
) -> dict[str, dict]:
    groups: dict[str, list[CandidateScreenResultV75]] = defaultdict(list)
    for candidate in candidates:
        value = candidate.profile[feature]
        if isinstance(value, list):
            value = "|".join(map(str, value))
        groups[str(value)].append(results_by_id[candidate.candidate_id])
    return {
        key: {
            "states": len(rows),
            "passed": sum(row.passed for row in rows),
            "pass_rate": sum(row.passed for row in rows) / len(rows),
            "mean_q2_classes": sum(row.q2_class_count for row in rows) / len(rows),
            "max_q2_classes": max(row.q2_class_count for row in rows),
        }
        for key, rows in sorted(groups.items())
    }


def screening_summary(
    candidates: Sequence[ExactSevenCandidateV75],
    results: Sequence[CandidateScreenResultV75],
    generation: CandidateGenerationStatsV75,
) -> dict:
    by_id = {row.candidate_id: row for row in results}
    if set(by_id) != {candidate.candidate_id for candidate in candidates}:
        raise ValueError("candidate/result id sets differ")
    passed = [row for row in results if row.passed]
    numeric_features = (
        "distinct_face_count",
        "half_turn_count",
        "opposite_face_adjacencies",
        "axis_changes",
        "pair_pdb_lower_bound",
        "fixed_quad_lower_bound",
        "orientation_defect_support",
        "corner_orientation_defect_support",
        "edge_orientation_defect_support",
        "misplaced_piece_support",
        "solved_piece_count",
        "rotation_class_size",
    )
    correlations = {}
    for feature in numeric_features:
        values = [float(candidate.profile[feature]) for candidate in candidates]
        correlations[feature] = {
            "with_pass": _pearson(values, [float(by_id[c.candidate_id].passed) for c in candidates]),
            "with_q2_class_count": _pearson(
                values,
                [float(by_id[c.candidate_id].q2_class_count) for c in candidates],
            ),
        }

    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}

    def rank_key(row: CandidateScreenResultV75) -> tuple:
        profile = candidate_by_id[row.candidate_id].profile
        return (
            row.q2_class_count,
            row.shortest_core_count,
            row.distinct_core_first_moves,
            row.productive_core_count,
            profile["distinct_face_count"],
            -profile["rotation_stabilizer_size"],
            profile["misplaced_piece_support"],
            profile["orientation_defect_support"],
            -profile["half_turn_count"],
            row.candidate_id,
        )

    ranked = sorted(passed, key=rank_key, reverse=True)
    return {
        "generation": asdict(generation),
        "screened_states": len(results),
        "passed_states": len(passed),
        "pass_rate": len(passed) / len(results) if results else 0.0,
        "maximum_q2_class_count": max((row.q2_class_count for row in results), default=0),
        "maximum_shortest_core_count": max((row.shortest_core_count for row in results), default=0),
        "q2_class_count_distribution": dict(
            sorted(Counter(row.q2_class_count for row in results).items())
        ),
        "shortest_core_count_distribution": dict(
            sorted(Counter(row.shortest_core_count for row in results).items())
        ),
        "productive_core_count_distribution": dict(
            sorted(Counter(row.productive_core_count for row in results).items())
        ),
        "screening_seconds_sum": sum(row.screening_seconds for row in results),
        "top_candidate_ids": [row.candidate_id for row in ranked[:20]],
        "passing_candidate_ids": [row.candidate_id for row in ranked],
        "rejected_candidate_ids": [
            row.candidate_id for row in results if not row.passed
        ],
        "feature_correlations": correlations,
        "pass_rates_by_feature": {
            feature: _rate_table(candidates, by_id, feature)
            for feature in (
                "half_turn_count",
                "opposite_face_adjacencies",
                "distinct_face_count",
                "pair_pdb_lower_bound",
                "fixed_quad_lower_bound",
                "orientation_defect_support",
                "corner_cycle_type",
                "edge_cycle_type",
            )
        },
    }


__all__ = [
    "CandidateGenerationStatsV75",
    "CandidateScreenResultV75",
    "CoreSummaryV75",
    "ExactSevenCandidateV75",
    "generate_exact_seven_candidates",
    "install_fork_enumerator",
    "q2_commute_reduce_normal_form",
    "screen_candidate",
    "screen_candidate_in_fork",
    "screening_summary",
]
