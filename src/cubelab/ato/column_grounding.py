"""Exact short-horizon COLUMN grounding for CubeLab v37.352 M5B/M5C."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json
import random
import resource

import numpy as np

from cubelab.pdcc.model import PIECE_INDEX, PIECE_NAMES, PIECE_SPECS, PieceKind
from cubelab.pdcc.moves import inverse_move

from .pose24 import MOVE_INDEX as ATO_MOVE_INDEX, Q_TO_OMEGA
from .projections import AXES, legacy_orientation
from .regular_support import (
    MOVE_INDEX,
    MOVE_NAMES,
    RegularFactor,
    canonical_allow,
    canonical_counts,
    canonical_word_arrays,
    exact_supported_domains,
    is_canonical_word,
    propagate_regular_support,
    relation_from_deterministic,
)
from .transitions import NEXT_Q


SCHEMA_VERSION = "cubelab.ato-column-grounding.v37.352"
DEFAULT_SEED = 20260822
SUBSETS: dict[str, tuple[int, ...]] = {
    "CORNERS8": tuple(range(8)),
    "EDGES12": tuple(range(8, 20)),
    "ALL20": tuple(range(20)),
}
AXIS_ARMS = {"A-UD": "UD", "A-FB": "FB", "A-LR": "LR"}
BASE_ARMS = (
    "A-UD",
    "A-FB",
    "A-LR",
    "B-TRI",
    "C-OMEGA-L",
    "D-Q-L",
    "C-OMEGA-A",
    "D-Q-A",
)
PLANE_R_ARMS = ("A-UD", "A-FB", "A-LR", "B-TRI", "C-OMEGA-L", "D-Q-L")


_PROD_TO_ATO = np.asarray([ATO_MOVE_INDEX[move] for move in MOVE_NAMES], dtype=np.uint8)
Q_NEXT = NEXT_Q[:, :, _PROD_TO_ATO]
Q_TO_OMEGA_ARRAY = np.asarray(Q_TO_OMEGA, dtype=np.uint8)
PIECE_ROWS = np.arange(len(PIECE_NAMES), dtype=np.int64)


def _rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return float(value / 1024.0 if value < 10**9 else value / (1024.0 * 1024.0))


def apply_words(
    start_q: np.ndarray | tuple[int, ...],
    words: np.ndarray,
) -> np.ndarray:
    rows = np.asarray(words, dtype=np.uint8)
    state = np.broadcast_to(
        np.asarray(start_q, dtype=np.uint8), (len(rows), len(PIECE_NAMES))
    ).copy()
    for column in range(rows.shape[1]):
        state = Q_NEXT[
            PIECE_ROWS[None, :],
            state,
            rows[:, column, None],
        ]
    return state


def apply_word(start_q: tuple[int, ...], word: tuple[str, ...]) -> tuple[int, ...]:
    state = np.asarray(start_q, dtype=np.uint8)
    for move in word:
        state = Q_NEXT[PIECE_ROWS, state, MOVE_INDEX[move]]
    return tuple(int(value) for value in state)


def inverse_word(word: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(inverse_move(move) for move in reversed(word))


def canonicalize_opposite_pairs(word: tuple[str, ...]) -> tuple[str, ...]:
    """Canonicalize a reduced word using only adjacent opposite commutation."""

    out = list(word)
    changed = True
    while changed:
        changed = False
        for index in range(len(out) - 1):
            left, right = out[index], out[index + 1]
            if not canonical_allow(left[0], right):
                # For an already face-reduced word, this is exactly the
                # descending adjacent-opposite case.
                out[index], out[index + 1] = right, left
                changed = True
    result = tuple(out)
    if not is_canonical_word(result):
        raise AssertionError(f"Failed to canonicalize inverse word: {word} -> {result}")
    return result


@dataclass(frozen=True, slots=True)
class RelationTables:
    axis_states: dict[str, np.ndarray]
    axis_relations: dict[str, tuple[np.ndarray, ...]]
    omega_relation: tuple[np.ndarray, ...]
    q_relation: tuple[np.ndarray, ...]
    omega_legacy_target: tuple[int, ...]
    axis_lifts: dict[str, np.ndarray]
    omega_lift: np.ndarray


def _existential_relation(states: np.ndarray, piece_index: int, state_count: int) -> np.ndarray:
    relation = np.zeros((state_count, len(MOVE_NAMES)), dtype=np.uint64)
    for q in range(24):
        source = int(states[piece_index, q])
        for move_id in range(len(MOVE_NAMES)):
            target_q = int(Q_NEXT[piece_index, q, move_id])
            target = int(states[piece_index, target_q])
            relation[source, move_id] |= np.uint64(1 << target)
    return relation


def _lift_relation(relation: np.ndarray, maximum_mask: int) -> np.ndarray:
    lift = np.zeros((maximum_mask, len(MOVE_NAMES)), dtype=np.uint64)
    for mask in range(maximum_mask):
        for move_id in range(len(MOVE_NAMES)):
            remaining = mask
            target = 0
            while remaining:
                low = remaining & -remaining
                state = low.bit_length() - 1
                target |= int(relation[state, move_id])
                remaining ^= low
            lift[mask, move_id] = np.uint64(target)
    return lift


def build_relation_tables() -> RelationTables:
    axis_states: dict[str, np.ndarray] = {}
    axis_relations: dict[str, tuple[np.ndarray, ...]] = {}
    axis_lifts: dict[str, np.ndarray] = {}

    for axis in AXES:
        states = np.asarray(
            [
                [legacy_orientation(piece, q, axis) for q in range(24)]
                for piece in PIECE_NAMES
            ],
            dtype=np.uint8,
        )
        axis_states[axis] = states
        relations: list[np.ndarray] = []
        lifts = np.zeros((20, 8, len(MOVE_NAMES)), dtype=np.uint64)
        for piece_index, piece in enumerate(PIECE_NAMES):
            state_count = 3 if PIECE_SPECS[piece].kind is PieceKind.CORNER else 2
            relation = _existential_relation(states, piece_index, state_count)
            relations.append(relation)
            lifted = _lift_relation(relation, 1 << state_count)
            lifts[piece_index, : len(lifted)] = lifted
        axis_relations[axis] = tuple(relations)
        axis_lifts[axis] = lifts

    omega_states = np.broadcast_to(Q_TO_OMEGA_ARRAY, (20, 24))
    omega_relations: list[np.ndarray] = []
    omega_lift = np.zeros((20, 64, len(MOVE_NAMES)), dtype=np.uint64)
    omega_legacy_target: list[int] = []
    for piece_index in range(20):
        relation = _existential_relation(omega_states, piece_index, 6)
        omega_relations.append(relation)
        omega_lift[piece_index] = _lift_relation(relation, 64)
        target = 0
        for omega in range(6):
            values = {
                tuple(int(axis_states[axis][piece_index, q]) for axis in AXES)
                for q in range(24)
                if int(Q_TO_OMEGA_ARRAY[q]) == omega
            }
            if len(values) != 1:
                raise AssertionError(
                    f"Omega does not determine the legacy tuple for {PIECE_NAMES[piece_index]}/{omega}: {values}"
                )
            if next(iter(values)) == (0, 0, 0):
                target |= 1 << omega
        omega_legacy_target.append(target)

    q_relations = tuple(
        relation_from_deterministic(Q_NEXT[piece_index])
        for piece_index in range(20)
    )
    return RelationTables(
        axis_states=axis_states,
        axis_relations=axis_relations,
        omega_relation=tuple(omega_relations),
        q_relation=q_relations,
        omega_legacy_target=tuple(omega_legacy_target),
        axis_lifts=axis_lifts,
        omega_lift=omega_lift,
    )


RELATIONS = build_relation_tables()


def _factor(
    name: str,
    relation: np.ndarray,
    initial_state: int,
    target_mask: int,
) -> RegularFactor:
    return RegularFactor(
        name=name,
        state_count=relation.shape[0],
        initial_mask=1 << int(initial_state),
        target_mask=int(target_mask),
        relation=relation,
    )


def factors_for_arm(
    arm: str,
    subset: tuple[int, ...],
    start_q: tuple[int, ...],
) -> tuple[RegularFactor, ...]:
    factors: list[RegularFactor] = []
    if arm in AXIS_ARMS:
        axis = AXIS_ARMS[arm]
        for piece_index in subset:
            relation = RELATIONS.axis_relations[axis][piece_index]
            factors.append(
                _factor(
                    f"{arm}:{PIECE_NAMES[piece_index]}",
                    relation,
                    int(RELATIONS.axis_states[axis][piece_index, start_q[piece_index]]),
                    1,
                )
            )
    elif arm == "B-TRI":
        for axis in AXES:
            for piece_index in subset:
                relation = RELATIONS.axis_relations[axis][piece_index]
                factors.append(
                    _factor(
                        f"B-{axis}:{PIECE_NAMES[piece_index]}",
                        relation,
                        int(RELATIONS.axis_states[axis][piece_index, start_q[piece_index]]),
                        1,
                    )
                )
    elif arm in ("C-OMEGA-L", "C-OMEGA-A"):
        for piece_index in subset:
            initial = int(Q_TO_OMEGA_ARRAY[start_q[piece_index]])
            target = (
                RELATIONS.omega_legacy_target[piece_index]
                if arm.endswith("-L")
                else 1
            )
            factors.append(
                _factor(
                    f"{arm}:{PIECE_NAMES[piece_index]}",
                    RELATIONS.omega_relation[piece_index],
                    initial,
                    target,
                )
            )
    elif arm in ("D-Q-L", "D-Q-A", "E-FULL-Q"):
        for piece_index in subset:
            if arm == "D-Q-L":
                target = sum(
                    1 << q
                    for q in range(24)
                    if all(
                        int(RELATIONS.axis_states[axis][piece_index, q]) == 0
                        for axis in AXES
                    )
                )
            elif arm == "D-Q-A":
                target = sum(1 << q for q in range(24) if int(Q_TO_OMEGA_ARRAY[q]) == 0)
            else:
                target = 1
            factors.append(
                _factor(
                    f"{arm}:{PIECE_NAMES[piece_index]}",
                    RELATIONS.q_relation[piece_index],
                    start_q[piece_index],
                    target,
                )
            )
    else:
        raise ValueError(f"Unknown arm: {arm}")
    return tuple(factors)


def _step_masks(
    masks: np.ndarray,
    moves: np.ndarray,
    lift: np.ndarray,
) -> np.ndarray:
    return lift[
        PIECE_ROWS[None, :],
        masks,
        moves[:, None],
    ]


def exact_acceptance(
    start_q: tuple[int, ...],
    words: np.ndarray,
    *,
    batch_size: int = 8192,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int]]:
    """Ground every arm in the same explicit word-ID array."""

    rows = np.asarray(words, dtype=np.uint8)
    accepted = {
        arm: {subset: np.zeros(len(rows), dtype=np.bool_) for subset in SUBSETS}
        for arm in (*BASE_ARMS, "E-FULL-Q")
    }
    piece_legacy_only = {"corners": 0, "edges": 0, "all": 0}
    q_start = np.asarray(start_q, dtype=np.uint8)

    for begin in range(0, len(rows), int(batch_size)):
        end = min(len(rows), begin + int(batch_size))
        block = rows[begin:end]
        q = np.broadcast_to(q_start, (len(block), 20)).copy()
        axis_masks = {
            axis: np.asarray(
                1 << RELATIONS.axis_states[axis][PIECE_ROWS, q_start],
                dtype=np.uint64,
            )[None, :].repeat(len(block), axis=0)
            for axis in AXES
        }
        omega_masks = np.asarray(
            1 << Q_TO_OMEGA_ARRAY[q_start], dtype=np.uint64
        )[None, :].repeat(len(block), axis=0)

        for column in range(block.shape[1]):
            move_ids = block[:, column]
            q = Q_NEXT[PIECE_ROWS[None, :], q, move_ids[:, None]]
            for axis in AXES:
                axis_masks[axis] = _step_masks(
                    axis_masks[axis], move_ids, RELATIONS.axis_lifts[axis]
                )
            omega_masks = _step_masks(
                omega_masks, move_ids, RELATIONS.omega_lift
            )

        per_piece: dict[str, np.ndarray] = {}
        for arm, axis in AXIS_ARMS.items():
            per_piece[arm] = (axis_masks[axis] & np.uint64(1)) != 0
        per_piece["B-TRI"] = np.logical_and.reduce(
            [(axis_masks[axis] & np.uint64(1)) != 0 for axis in AXES]
        )
        omega_legacy_targets = np.asarray(
            RELATIONS.omega_legacy_target, dtype=np.uint64
        )
        per_piece["C-OMEGA-L"] = (
            omega_masks & omega_legacy_targets[None, :]
        ) != 0

        legacy_exact = np.logical_and.reduce(
            [
                RELATIONS.axis_states[axis][PIECE_ROWS[None, :], q] == 0
                for axis in AXES
            ]
        )
        ato_exact = Q_TO_OMEGA_ARRAY[q] == 0
        per_piece["D-Q-L"] = legacy_exact
        per_piece["C-OMEGA-A"] = (omega_masks & np.uint64(1)) != 0
        per_piece["D-Q-A"] = ato_exact
        per_piece["E-FULL-Q"] = q == 0

        legacy_only = legacy_exact & ~ato_exact
        piece_legacy_only["corners"] += int(legacy_only[:, :8].sum())
        piece_legacy_only["edges"] += int(legacy_only[:, 8:].sum())
        piece_legacy_only["all"] += int(legacy_only.sum())

        for arm, values in per_piece.items():
            for subset_name, subset in SUBSETS.items():
                accepted[arm][subset_name][begin:end] = values[:, subset].all(axis=1)
    return accepted, piece_legacy_only


def _word_tuple(words: np.ndarray, index: int) -> tuple[str, ...]:
    return tuple(MOVE_NAMES[int(value)] for value in words[int(index)])


def _first_set(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return None if not len(indices) else int(indices[0])


def _trajectory(start_q: tuple[int, ...], word: tuple[str, ...]) -> list[dict[str, object]]:
    q = tuple(start_q)
    trace = [
        {
            "column": 0,
            "move": None,
            "q": list(q),
            "omega": [int(Q_TO_OMEGA[value]) for value in q],
        }
    ]
    for column, move in enumerate(word, start=1):
        q = apply_word(q, (move,))
        trace.append(
            {
                "column": column,
                "move": move,
                "q": list(q),
                "omega": [int(Q_TO_OMEGA[value]) for value in q],
            }
        )
    return trace


def _unique_prefix_count(words: np.ndarray, accepted: np.ndarray, width: int) -> int | None:
    if words.shape[1] < width:
        return None
    selected = words[accepted, :width]
    if not len(selected):
        return 0
    code = np.zeros(len(selected), dtype=np.uint32)
    for column in range(width):
        code = code * np.uint32(18) + selected[:, column]
    return int(np.unique(code).size)


def _relation_cache_arrays() -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "production_to_ato_move": _PROD_TO_ATO,
        "q_next": Q_NEXT,
        "q_to_omega": Q_TO_OMEGA_ARRAY,
        "omega_legacy_target": np.asarray(RELATIONS.omega_legacy_target, dtype=np.uint8),
    }
    for axis in AXES:
        arrays[f"{axis.lower()}_state"] = RELATIONS.axis_states[axis]
        padded = np.zeros((20, 3, len(MOVE_NAMES)), dtype=np.uint64)
        for piece_index, relation in enumerate(RELATIONS.axis_relations[axis]):
            padded[piece_index, : relation.shape[0]] = relation
        arrays[f"{axis.lower()}_relation"] = padded
    arrays["omega_relation"] = np.stack(RELATIONS.omega_relation)
    return arrays


def write_relation_cache(path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **_relation_cache_arrays())
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def run_m5b_audit() -> dict[str, object]:
    started = perf_counter()
    production = canonical_counts(3)
    foundation = (1, 18, 270, 4050)
    first_mismatch = None
    for word in (("D", "U"), ("L", "R"), ("B", "F")):
        if not is_canonical_word(word):
            first_mismatch = {
                "word": list(word),
                "foundation_no_same_face": True,
                "production_canonical": False,
                "reason": "descending adjacent opposite-face pair",
            }
            break

    false_deletions = []
    relation_edges = defaultdict(int)
    for piece_index, piece in enumerate(PIECE_NAMES):
        for q in range(24):
            for move_id, move in enumerate(MOVE_NAMES):
                target_q = int(Q_NEXT[piece_index, q, move_id])
                for axis in AXES:
                    source = int(RELATIONS.axis_states[axis][piece_index, q])
                    target = int(RELATIONS.axis_states[axis][piece_index, target_q])
                    relation_edges[f"axis_{axis}"] += 1
                    if not int(RELATIONS.axis_relations[axis][piece_index][source, move_id]) & (1 << target):
                        false_deletions.append({"piece": piece, "q": q, "move": move, "arm": axis})
                source_omega = int(Q_TO_OMEGA_ARRAY[q])
                target_omega = int(Q_TO_OMEGA_ARRAY[target_q])
                relation_edges["omega"] += 1
                if not int(RELATIONS.omega_relation[piece_index][source_omega, move_id]) & (1 << target_omega):
                    false_deletions.append({"piece": piece, "q": q, "move": move, "arm": "OMEGA"})

    # Direct same-word kernel check on a deterministic nontrivial state.
    start = apply_word((0,) * 20, ("R", "U2", "F'"))
    factors = factors_for_arm("D-Q-A", SUBSETS["ALL20"], start)
    kernel = propagate_regular_support(factors, 3)
    passed = not false_deletions and production == (1, 18, 243, 3240)
    return {
        "schema": "cubelab.ato-m5b-regular-support.v37.352",
        "canonical_parity": {
            "production_semantics": "same face forbidden; adjacent opposite faces retain URFDLB ascending order",
            "production_counts_length_0_3": list(production),
            "v37_351_no_same_face_counts_length_0_3": list(foundation),
            "semantics_identical": production == foundation,
            "first_mismatch_word": first_mismatch,
            "v37_352_cache_regenerated": production != foundation,
        },
        "local_relation_recall": {
            "exact_transitions_checked": 8640,
            "axis_checks": 8640 * 3,
            "omega_checks": 8640,
            "false_deletions": len(false_deletions),
            "first_false_deletion": false_deletions[0] if false_deletions else None,
            "recall": 1.0 if not false_deletions else 0.0,
        },
        "same_word_kernel_smoke": {
            "length": 3,
            "factor_count": len(factors),
            "gac_nonempty": kernel["gac_nonempty"],
            "iterations": kernel["iterations"],
            "domain_width_after": list(kernel["domain_width_after"]),
        },
        "column_identity_contract": "All factors consume one shared explicit production-canonical move tuple; GAC and exact-word evidence are reported separately.",
        "branch_search": "NONE",
        "elapsed_wall_seconds": perf_counter() - started,
        "peak_rss_mib": _rss_mib(),
        "result": "PASS" if passed else "FAIL",
        "decision": (
            "V37_352_M5B_REGULAR_SUPPORT_PASS"
            if passed
            else "V37_352_M5B_REGULAR_SUPPORT_FAIL"
        ),
    }


def _random_nonopposite_word(rng: random.Random, length: int, *, stress: bool) -> tuple[str, ...]:
    word: list[str] = []
    while len(word) < length:
        choices = [
            move
            for move in MOVE_NAMES
            if (
                not word
                or (
                    move[0] != word[-1][0]
                    and {move[0], word[-1][0]} not in ({"U", "D"}, {"R", "L"}, {"F", "B"})
                )
            )
        ]
        if stress:
            half_count = sum(move.endswith("2") for move in word)
            remaining = length - len(word)
            if half_count < 2 and remaining <= 2:
                half = [move for move in choices if move.endswith("2")]
                if half:
                    choices = half
        word.append(rng.choice(choices))
    return tuple(word)


def _depth4_state_map() -> dict[bytes, int]:
    solved = (0,) * 20
    depths: dict[bytes, int] = {bytes(solved): 0}
    for depth in range(1, 5):
        words = canonical_word_arrays(depth)
        endpoints = apply_words(solved, words)
        for row in endpoints:
            depths.setdefault(bytes(row), depth)
    return depths


def build_exact5_corpus(
    *,
    seed: int = DEFAULT_SEED,
    unbiased_count: int = 12,
    transport_count: int = 6,
) -> dict[str, object]:
    started = perf_counter()
    rng = random.Random(int(seed))
    shallow = _depth4_state_map()
    seen_orbits: set[bytes] = set()
    rows: list[dict[str, object]] = []

    def collect(kind: str, count: int, stress: bool) -> None:
        attempts = 0
        while sum(row["corpus"] == kind for row in rows) < count:
            attempts += 1
            if attempts > 200000:
                raise RuntimeError(f"Unable to construct {count} unique {kind} exact-5 controls")
            scramble = _random_nonopposite_word(rng, 5, stress=stress)
            q = apply_word((0,) * 20, scramble)
            key = bytes(q)
            if key in shallow:
                continue
            inverse = canonicalize_opposite_pairs(inverse_word(scramble))
            inverse_q = apply_word((0,) * 20, inverse_word(scramble))
            orbit_key = min(key, bytes(inverse_q))
            if orbit_key in seen_orbits:
                continue
            if stress:
                half_turns = sum(move.endswith("2") for move in scramble)
                quarter_turns = len(scramble) - half_turns
                if half_turns < 2 or quarter_turns < 2:
                    continue
            solved = apply_word(q, inverse)
            if solved != (0,) * 20:
                raise AssertionError(f"Exact-5 witness replay failed: {scramble}/{inverse}")
            seen_orbits.add(orbit_key)
            rows.append(
                {
                    "id": f"E5-{'STRESS' if stress else 'UNBIASED'}-{sum(row['corpus'] == kind for row in rows)+1:02d}",
                    "corpus": kind,
                    "scramble": list(scramble),
                    "scramble_string": " ".join(scramble),
                    "solution": list(inverse),
                    "solution_string": " ".join(inverse),
                    "exact_distance": 5,
                    "shorter_depth_0_4_unsat": True,
                    "witness_replay": True,
                    "state_sha256": sha256(key).hexdigest(),
                    "inverse_equivalent_deduplicated": True,
                    "half_turn_count": sum(move.endswith("2") for move in scramble),
                }
            )

    collect("unbiased", int(unbiased_count), False)
    collect("transport_stress", int(transport_count), True)
    return {
        "seed": int(seed),
        "cases": rows,
        "counts": {
            "unbiased": sum(row["corpus"] == "unbiased" for row in rows),
            "transport_stress": sum(row["corpus"] == "transport_stress" for row in rows),
            "total": len(rows),
        },
        "certifier": {
            "method": "complete production-canonical layers 0..4 plus replay-valid length-5 inverse",
            "states_depth_0_4": len(shallow),
        },
        "elapsed_wall_seconds": perf_counter() - started,
    }


def _arm_target(arm: str) -> str:
    if arm.endswith("-L") or arm.startswith("A-") or arm == "B-TRI":
        return "T_LEGACY"
    if arm.endswith("-A"):
        return "T_ATO"
    return "T_FULL"


def run_exact5_census(
    *,
    seed: int = DEFAULT_SEED,
    exact5_count: int = 12,
    exact5_transport_count: int = 6,
    batch_size: int = 8192,
) -> dict[str, object]:
    started = perf_counter()
    rss_start = _rss_mib()
    corpus = build_exact5_corpus(
        seed=seed,
        unbiased_count=exact5_count,
        transport_count=exact5_transport_count,
    )
    words_by_length = {depth: canonical_word_arrays(depth) for depth in range(6)}
    rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    case_costs: list[dict[str, object]] = []
    violations: list[dict[str, object]] = []
    comparability_counterexamples: list[dict[str, object]] = []
    witnesses: dict[str, object | None] = {
        "c_strict_vs_b": None,
        "c_strict_vs_b_edge_or_all": None,
        "c_false_positive_vs_d": None,
        "q_strict_vs_b_edge_or_all": None,
        "legacy_target_not_ato": None,
    }
    strictness = {
        subset: {
            "omega_strict_vs_tri_words": 0,
            "q_strict_vs_tri_words": 0,
            "q_strict_vs_omega_words": 0,
        }
        for subset in SUBSETS
    }
    total_piece_legacy_only = {"corners": 0, "edges": 0, "all": 0}
    full_solution_checks = 0

    for case in corpus["cases"]:
        case_started = perf_counter()
        start_q = apply_word((0,) * 20, tuple(case["scramble"]))
        for depth, words in words_by_length.items():
            accepted, piece_legacy_only = exact_acceptance(
                start_q, words, batch_size=batch_size
            )
            for key in total_piece_legacy_only:
                total_piece_legacy_only[key] += piece_legacy_only[key]

            # Mandatory set inclusions and first explicit counterexamples.
            for subset_name in SUBSETS:
                d_l = accepted["D-Q-L"][subset_name]
                d_a = accepted["D-Q-A"][subset_name]
                for abstract in ("A-UD", "A-FB", "A-LR", "B-TRI", "C-OMEGA-L"):
                    bad = d_l & ~accepted[abstract][subset_name]
                    index = _first_set(bad)
                    if index is not None:
                        violations.append(
                            {
                                "type": "FALSE_DELETION",
                                "case": case["id"],
                                "depth": depth,
                                "subset": subset_name,
                                "exact_arm": "D-Q-L",
                                "abstract_arm": abstract,
                                "word": list(_word_tuple(words, index)),
                            }
                        )
                bad = d_a & ~accepted["C-OMEGA-A"][subset_name]
                index = _first_set(bad)
                if index is not None:
                    violations.append(
                        {
                            "type": "FALSE_DELETION",
                            "case": case["id"],
                            "depth": depth,
                            "subset": subset_name,
                            "exact_arm": "D-Q-A",
                            "abstract_arm": "C-OMEGA-A",
                            "word": list(_word_tuple(words, index)),
                        }
                    )

                c_not_b = accepted["C-OMEGA-L"][subset_name] & ~accepted["B-TRI"][subset_name]
                index = _first_set(c_not_b)
                if index is not None:
                    comparability_counterexamples.append(
                        {
                            "type": "EXPECTED_RELATION_BREAK_C_NOT_SUBSET_B",
                            "case": case["id"],
                            "depth": depth,
                            "subset": subset_name,
                            "word": list(_word_tuple(words, index)),
                        }
                    )

                if witnesses["c_strict_vs_b"] is None:
                    strict = accepted["B-TRI"][subset_name] & ~accepted["C-OMEGA-L"][subset_name]
                    index = _first_set(strict)
                    if index is not None:
                        word = _word_tuple(words, index)
                        witnesses["c_strict_vs_b"] = {
                            "case": case["id"],
                            "corpus": case["corpus"],
                            "depth": depth,
                            "subset": subset_name,
                            "word": list(word),
                            "word_id": index,
                            "B-TRI": True,
                            "C-OMEGA-L": False,
                            "exact_q_trajectory": _trajectory(start_q, word),
                        }
                strict = accepted["B-TRI"][subset_name] & ~accepted["C-OMEGA-L"][subset_name]
                q_strict_tri = accepted["B-TRI"][subset_name] & ~d_l
                q_strict_omega = accepted["C-OMEGA-L"][subset_name] & ~d_l
                strictness[subset_name]["omega_strict_vs_tri_words"] += int(strict.sum())
                strictness[subset_name]["q_strict_vs_tri_words"] += int(q_strict_tri.sum())
                strictness[subset_name]["q_strict_vs_omega_words"] += int(q_strict_omega.sum())
                if (
                    subset_name in ("EDGES12", "ALL20")
                    and witnesses["c_strict_vs_b_edge_or_all"] is None
                ):
                    index = _first_set(strict)
                    if index is not None:
                        word = _word_tuple(words, index)
                        witnesses["c_strict_vs_b_edge_or_all"] = {
                            "case": case["id"],
                            "corpus": case["corpus"],
                            "depth": depth,
                            "subset": subset_name,
                            "word": list(word),
                            "word_id": index,
                            "B-TRI": True,
                            "C-OMEGA-L": False,
                            "exact_q_trajectory": _trajectory(start_q, word),
                        }
                if (
                    subset_name in ("EDGES12", "ALL20")
                    and witnesses["q_strict_vs_b_edge_or_all"] is None
                ):
                    index = _first_set(q_strict_tri)
                    if index is not None:
                        word = _word_tuple(words, index)
                        witnesses["q_strict_vs_b_edge_or_all"] = {
                            "case": case["id"],
                            "corpus": case["corpus"],
                            "depth": depth,
                            "subset": subset_name,
                            "word": list(word),
                            "word_id": index,
                            "B-TRI": True,
                            "D-Q-L": False,
                            "exact_q_trajectory": _trajectory(start_q, word),
                        }
                if witnesses["c_false_positive_vs_d"] is None:
                    false_positive = accepted["C-OMEGA-L"][subset_name] & ~d_l
                    index = _first_set(false_positive)
                    if index is not None:
                        word = _word_tuple(words, index)
                        witnesses["c_false_positive_vs_d"] = {
                            "case": case["id"],
                            "corpus": case["corpus"],
                            "depth": depth,
                            "subset": subset_name,
                            "word": list(word),
                            "word_id": index,
                            "C-OMEGA-L": True,
                            "D-Q-L": False,
                            "exact_q_trajectory": _trajectory(start_q, word),
                        }
                if witnesses["legacy_target_not_ato"] is None:
                    legacy_only = d_l & ~d_a
                    index = _first_set(legacy_only)
                    if index is not None:
                        word = _word_tuple(words, index)
                        witnesses["legacy_target_not_ato"] = {
                            "case": case["id"],
                            "corpus": case["corpus"],
                            "depth": depth,
                            "subset": subset_name,
                            "word": list(word),
                            "word_id": index,
                            "D-Q-L": True,
                            "D-Q-A": False,
                            "exact_q_trajectory": _trajectory(start_q, word),
                        }

                target_rows.append(
                    {
                        "case": case["id"],
                        "corpus": case["corpus"],
                        "length": depth,
                        "subset": subset_name,
                        "legacy_zero_word_endpoints": int(d_l.sum()),
                        "ato_identity_word_endpoints": int(d_a.sum()),
                        "legacy_only_word_endpoints": int((d_l & ~d_a).sum()),
                        "ato_only_word_endpoints": int((d_a & ~d_l).sum()),
                        "ato_to_legacy_ratio": (
                            None if not int(d_l.sum()) else float(d_a.sum() / d_l.sum())
                        ),
                    }
                )

            full = accepted["E-FULL-Q"]["ALL20"]
            if full.any():
                for arm in BASE_ARMS:
                    missing = full & ~accepted[arm]["ALL20"]
                    full_solution_checks += int(full.sum())
                    index = _first_set(missing)
                    if index is not None:
                        violations.append(
                            {
                                "type": "FULL_SOLUTION_INCLUSION_FAIL",
                                "case": case["id"],
                                "depth": depth,
                                "arm": arm,
                                "word": list(_word_tuple(words, index)),
                            }
                        )

            for subset_name, subset in SUBSETS.items():
                arm_metrics: dict[str, dict[str, object]] = {}
                for arm in BASE_ARMS:
                    exact = accepted[arm][subset_name]
                    factors = factors_for_arm(arm, subset, start_q)
                    propagated = propagate_regular_support(factors, depth)
                    exact_domains = exact_supported_domains(words, exact)
                    gac_domains = tuple(int(value) for value in propagated["domains_after"])
                    false_masks = tuple(
                        gac & ~supported for gac, supported in zip(gac_domains, exact_domains)
                    )
                    reference = (
                        accepted["D-Q-A"][subset_name]
                        if arm == "C-OMEGA-A"
                        else accepted["D-Q-L"][subset_name]
                        if arm not in ("D-Q-A",)
                        else accepted["D-Q-A"][subset_name]
                    )
                    metric = {
                        "case": case["id"],
                        "corpus": case["corpus"],
                        "length": depth,
                        "arm": arm,
                        "target": _arm_target(arm),
                        "subset": subset_name,
                        "factor_count": len(factors),
                        "factor_state_count_sum": sum(factor.state_count for factor in factors),
                        "gac_iterations": int(propagated["iterations"]),
                        "domain_width_before": list(propagated["domain_width_before"]),
                        "domain_width_after": list(propagated["domain_width_after"]),
                        "domain_sum_before": int(propagated["domain_sum_before"]),
                        "domain_sum_after": int(propagated["domain_sum_after"]),
                        "exact_supported_domain_width": [mask.bit_count() for mask in exact_domains],
                        "exact_supported_domain_sum": sum(mask.bit_count() for mask in exact_domains),
                        "gac_false_domain_count": sum(mask.bit_count() for mask in false_masks),
                        "accepted_exact_common_word_count": int(exact.sum()),
                        "common_word_empty": not bool(exact.any()),
                        "all_factors_nonempty": bool(propagated["all_factors_nonempty"]),
                        "factor_nonempty_but_no_common_word": bool(
                            propagated["all_factors_nonempty"] and not exact.any()
                        ),
                        "supported_word2_prefix_count": _unique_prefix_count(words, exact, 2),
                        "supported_word3_prefix_count": _unique_prefix_count(words, exact, 3),
                        "false_positive_words_vs_exact_q": int((exact & ~reference).sum()),
                        "false_positive_ratio_vs_exact_q": (
                            0.0 if not int(exact.sum()) else float((exact & ~reference).sum() / exact.sum())
                        ),
                        "factor_state_visits": int(propagated["factor_state_visits"]),
                        "gac_wall_seconds": float(propagated["wall_seconds"]),
                    }
                    rows.append(metric)
                    arm_metrics[arm] = metric

                # Optimistic case-local legacy baseline.
                best_axis = min(
                    AXIS_ARMS,
                    key=lambda name: (
                        arm_metrics[name]["accepted_exact_common_word_count"],
                        arm_metrics[name]["exact_supported_domain_sum"],
                        tuple(AXIS_ARMS).index(name),
                    ),
                )
                best = dict(arm_metrics[best_axis])
                best["arm"] = "A-BEST1"
                best["selected_axis_arm"] = best_axis
                rows.append(best)

        case_costs.append(
            {
                "case": case["id"],
                "corpus": case["corpus"],
                "wall_seconds": perf_counter() - case_started,
                "peak_rss_mib": _rss_mib(),
            }
        )

    passed = not violations
    total_words_per_case = sum(len(words) for words in words_by_length.values())
    return {
        "schema": "cubelab.ato-m5c-exact5-column-census.v37.352",
        "parameters": {
            "seed": seed,
            "batch_size": batch_size,
            "exact5_count": exact5_count,
            "exact5_transport_count": exact5_transport_count,
        },
        "canonical_language": {
            "semantics": "production same-face reduction plus URFDLB adjacent-opposite quotient",
            "exact_length_counts_0_5": list(canonical_counts(5)),
            "words_per_case_total_0_5": total_words_per_case,
        },
        "corpus": corpus,
        "representation_rows": rows,
        "target_rows": target_rows,
        "piece_endpoint_legacy_zero_but_omega_nonidentity": total_piece_legacy_only,
        "representation_strictness": strictness,
        "soundness": {
            "violations": violations,
            "false_deletions": sum(row["type"] == "FALSE_DELETION" for row in violations),
            "full_solution_inclusion_checks": full_solution_checks,
            "same_word_identity": True,
            "exact_word_ids_shared_across_arms": True,
        },
        "non_assumed_comparability_counterexamples": comparability_counterexamples,
        "counterexamples": witnesses,
        "case_costs": case_costs,
        "cost": {
            "elapsed_wall_seconds": perf_counter() - started,
            "peak_rss_mib": _rss_mib(),
            "incremental_peak_rss_mib": max(0.0, _rss_mib() - rss_start),
            "word_case_evaluations": total_words_per_case * len(corpus["cases"]),
        },
        "production_changes": "NONE",
        "search_pruning_changes": "NONE",
        "result": "PASS" if passed else "FAIL",
        "decision": (
            "V37_352_M5C_EXACT5_GROUNDING_PASS"
            if passed
            else "V37_352_M5C_EXACT5_GROUNDING_FAIL"
        ),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "BASE_ARMS",
    "DEFAULT_SEED",
    "Q_NEXT",
    "RELATIONS",
    "SCHEMA_VERSION",
    "SUBSETS",
    "apply_word",
    "apply_words",
    "build_exact5_corpus",
    "build_relation_tables",
    "canonicalize_opposite_pairs",
    "exact_acceptance",
    "factors_for_arm",
    "inverse_word",
    "run_exact5_census",
    "run_m5b_audit",
    "write_json",
    "write_relation_cache",
]
