#!/usr/bin/env python3
"""
CubeLab v37.83 — COMPILED COLUMN-WORD SUPERPOSITION BITSET

Background
----------
v37.82 full-tree result:

    HIST M11 L7 : 34 -> 22 nodes  (-35.29%)
    FRESH L7    : 32 -> 27 nodes  (-15.62%)
    FRESH L8    : 60 -> 52 nodes  (-13.33%)

Exact solution parity was preserved on every control.

But the Python regular-language fixpoint cost ~20x baseline wall time.

Do NOT change the model.
Compile it.

Core representation
-------------------
For each remaining horizon R <= 4, enumerate every reduced HTM global word:

    R=1 : 18
    R=2 : 270
    R=3 : 4,050
    R=4 : 60,750

Each word is one complete future COLUMN schedule.

For each:
    piece p
    current pose s
    horizon R

precompute a Python-int bitset:

    W[p,s,R]

whose set bits are exactly the reduced global words of length R that take
that piece from s into its local-DR target set.

Thus:

    intersection over p=1..20 of W[p,s_p,R]

is the EXACT shared-column superposition of all 20 piece languages for that
horizon.

No piece candidate is chosen.
No GAC iteration is needed.
All cross-column correlations are preserved because a bit represents one
complete GLOBAL COLUMN WORD.

Previous-face reduction is another precomputed word mask.

Q synergy
---------
At a Q-tight node (h_Q == R), compute the ordinary Q-admissible FIRST moves:

    phase1_lb(q after m) <= R-1

Intersect the shared word bitset with the union of words beginning with those
Q-admissible moves.

This is the compiled form of:

    Q first-column filtering
        -> 20-piece superposition
        -> future-column correlation
        -> back to supported first columns

The final shared bitset gives:
    * UNSAT if empty;
    * exact supported C0 move values otherwise.

Modes
-----
Q_ONLY

WORD_PRUNE_T4
    exact 20-piece shared-word intersection at Q-tight R<=4;
    use EMPTY only.

WORD_FILTER_T4
    same, also restrict ordinary Q next branches to supported C0 values.

QWORD_FILTER_T4
    add Q first-column mask BEFORE testing shared-word emptiness/support.

QWORD_FILTER_T3
    same but R<=3.

Controls
--------
Same as v37.82:
    M12 exact-3
    historical M11 exact-7
    deterministic fresh exact-7
    deterministic fresh exact-8

Metrics
-------
    precompute wall
    precomputed bitset memory estimate
    exact solution parity
    nodes
    oracle calls
    empty-wordset prunes
    C0 values removed
    surviving word counts
    hot-search wall ratio

This is the compiled/cache form of the combined
COLUMN + SUPERPOSITION model.

No production changes.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"

if SRC.is_dir():
    s = str(SRC)
    if s not in sys.path:
        sys.path.insert(0, s)
if SCRIPTS.is_dir():
    s = str(SCRIPTS)
    if s not in sys.path:
        sys.path.insert(0, s)

import search_twist_skeleton as ts
import search_pdcc_guided as pg

import audit_fast_local_dr_superposition_v37_57 as v57
import audit_fast_requirement_nonuniform_v37_70 as v70
import audit_fast_column_superposition_v37_80 as v80


NMOVES = len(ts.MOVE_ORDER)
ALL_MOVE_MASK = (1 << NMOVES) - 1
MOVE_RANK = {m: i for i, m in enumerate(ts.MOVE_ORDER)}

FACE_ORDER = ("U", "D", "R", "L", "F", "B")
FACE_INDEX = {f: i for i, f in enumerate(FACE_ORDER)}
MOVE_FACE = tuple(FACE_INDEX[m[0]] for m in ts.MOVE_ORDER)


def q_lb(q, qtables):
    return int(
        ts.phase1_lb(
            int(q),
            qtables[3],
            qtables[4],
        )
    )


def q_step(q, move, qtables):
    return int(
        ts.q_move(
            int(q),
            ts.MI[move],
            qtables[0],
            qtables[1],
            qtables[2],
        )
    )


def step_state(poses, q, move, qtables):
    return v80.step_state(
        poses,
        q,
        move,
        qtables,
    )


def internalize(word, axis):
    return tuple(
        v70.internalize(
            tuple(word),
            axis,
        )
    )


def externalize(word, axis):
    return tuple(
        v70.externalize(
            tuple(word),
            axis,
        )
    )


def pack_bool_to_int(flags):
    arr = np.asarray(
        flags,
        dtype=np.uint8,
    )
    packed = np.packbits(
        arr,
        bitorder="little",
    )
    return int.from_bytes(
        packed.tobytes(),
        "little",
        signed=False,
    )


def generate_reduced_words(max_horizon):
    """
    Return:
        words[R] : np.ndarray shape (N_R, R), move indices
        first_move_masks[R][mi]
        first_face_masks[R][face]
        universe_masks[R]
    """
    words = {
        0: np.zeros(
            (1, 0),
            dtype=np.uint8,
        )
    }

    first_move_masks = {}
    first_face_masks = {}
    universe_masks = {}

    for R in range(
        1,
        int(max_horizon) + 1,
    ):
        rows = []

        def rec(prefix, last_face):
            if len(prefix) == R:
                rows.append(
                    tuple(prefix)
                )
                return

            for mi, move in enumerate(
                ts.MOVE_ORDER
            ):
                face = MOVE_FACE[mi]

                if (
                    last_face is not None
                    and face == last_face
                ):
                    continue

                prefix.append(mi)
                rec(
                    prefix,
                    face,
                )
                prefix.pop()

        rec([], None)

        arr = np.asarray(
            rows,
            dtype=np.uint8,
        )

        words[R] = arr

        N = int(
            arr.shape[0]
        )

        universe_masks[R] = (
            (1 << N) - 1
        )

        fm = []
        for mi in range(NMOVES):
            fm.append(
                pack_bool_to_int(
                    arr[:, 0] == mi
                )
            )
        first_move_masks[R] = tuple(
            fm
        )

        ff = []
        for face in range(6):
            ff.append(
                pack_bool_to_int(
                    np.asarray(
                        [
                            MOVE_FACE[
                                int(mi)
                            ]
                            == face
                            for mi in arr[:, 0]
                        ],
                        dtype=np.uint8,
                    )
                )
            )
        first_face_masks[R] = tuple(
            ff
        )

    return (
        words,
        first_move_masks,
        first_face_masks,
        universe_masks,
    )


class CompiledWordSuperposition:
    def __init__(
        self,
        dr_targets,
        qtables,
        max_horizon=4,
    ):
        self.max_horizon = int(
            max_horizon
        )
        self.qtables = qtables

        (
            self.words,
            self.first_move_masks,
            self.first_face_masks,
            self.universe_masks,
        ) = generate_reduced_words(
            self.max_horizon
        )

        self.pose_next = np.asarray(
            pg.POSE_NEXT,
            dtype=np.uint8,
        )

        self.target = np.zeros(
            (20, 24),
            dtype=bool,
        )

        for p in range(20):
            for pose in dr_targets[p]:
                self.target[
                    p,
                    int(pose),
                ] = True

        self.piece_masks = {}
        self.build_wall = 0.0
        self.bitset_bytes_estimate = 0

        self._build_all_piece_masks()

    def _build_all_piece_masks(
        self,
    ):
        t0 = time.perf_counter()

        total_bytes = 0

        for R in range(
            1,
            self.max_horizon + 1,
        ):
            moves = self.words[R]
            N = int(
                moves.shape[0]
            )

            # 24 possible start poses for every word, vectorized.
            start_template = np.broadcast_to(
                np.arange(
                    24,
                    dtype=np.uint8,
                ),
                (
                    N,
                    24,
                ),
            ).copy()

            for p in range(20):
                states = (
                    start_template.copy()
                )

                for k in range(R):
                    mis = moves[
                        :,
                        k,
                    ].astype(
                        np.int64,
                        copy=False,
                    )

                    states = (
                        self.pose_next[
                            mis[:, None],
                            p,
                            states,
                        ]
                    )

                target_flags = (
                    self.target[
                        p,
                        states,
                    ]
                )

                for pose in range(24):
                    mask = (
                        pack_bool_to_int(
                            target_flags[
                                :,
                                pose,
                            ]
                        )
                    )

                    self.piece_masks[
                        (
                            R,
                            p,
                            pose,
                        )
                    ] = mask

                    total_bytes += (
                        mask.bit_length()
                        + 7
                    ) // 8

        self.build_wall = (
            time.perf_counter()
            - t0
        )
        self.bitset_bytes_estimate = (
            total_bytes
        )

    def previous_face_mask(
        self,
        R,
        last_face,
    ):
        if last_face is None:
            return self.universe_masks[
                R
            ]

        face = FACE_INDEX[
            last_face
        ]

        return (
            self.universe_masks[R]
            & ~self.first_face_masks[
                R
            ][face]
        )

    def q_first_allowed_moves(
        self,
        q,
        R,
        last_face,
    ):
        allowed = []

        child_rem = int(R) - 1

        for mi, move in enumerate(
            ts.MOVE_ORDER
        ):
            if (
                last_face is not None
                and move[0]
                == last_face
            ):
                continue

            nq = q_step(
                q,
                move,
                self.qtables,
            )

            if (
                q_lb(
                    nq,
                    self.qtables,
                )
                <= child_rem
            ):
                allowed.append(
                    mi
                )

        return tuple(
            allowed
        )

    def query(
        self,
        *,
        poses,
        q,
        R,
        last_face,
        add_q_first,
    ):
        R = int(R)

        if (
            R <= 0
            or R
            > self.max_horizon
        ):
            raise ValueError(
                f"unsupported horizon {R}"
            )

        mask = self.previous_face_mask(
            R,
            last_face,
        )

        # Exact shared-word intersection across all 20 piece future languages.
        for p in range(20):
            mask &= self.piece_masks[
                (
                    R,
                    p,
                    int(
                        poses[p]
                    ),
                )
            ]

            if mask == 0:
                return {
                    "feasible": False,
                    "word_mask": 0,
                    "word_count": 0,
                    "first_move_mask": 0,
                    "q_allowed_move_count": (
                        None
                    ),
                }

        q_allowed = None

        if add_q_first:
            q_allowed = (
                self.q_first_allowed_moves(
                    q,
                    R,
                    last_face,
                )
            )

            qword = 0

            for mi in q_allowed:
                qword |= (
                    self.first_move_masks[
                        R
                    ][mi]
                )

            mask &= qword

            if mask == 0:
                return {
                    "feasible": False,
                    "word_mask": 0,
                    "word_count": 0,
                    "first_move_mask": 0,
                    "q_allowed_move_count": len(
                        q_allowed
                    ),
                }

        first_mask = 0

        for mi in range(
            NMOVES
        ):
            if (
                mask
                & self.first_move_masks[
                    R
                ][mi]
            ):
                first_mask |= (
                    1 << mi
                )

        return {
            "feasible": True,
            "word_mask": mask,
            "word_count": int(
                mask.bit_count()
            ),
            "first_move_mask": (
                first_mask
            ),
            "q_allowed_move_count": (
                None
                if q_allowed is None
                else len(
                    q_allowed
                )
            ),
        }


@dataclass
class SearchResult:
    mode: str
    nodes: int
    q_prunes: int
    oracle_calls: int
    oracle_empty_prunes: int
    first_column_values_removed: int
    surviving_word_total: int
    trigger_histogram: dict
    solutions: tuple[tuple[str, ...], ...]
    wall: float
    cut_reason: str | None


def exact_search(
    *,
    mode,
    scramble_internal,
    depth_limit,
    qtables,
    oracle,
    node_cap,
    time_cap,
):
    start = ts.engine.from_word(
        " ".join(
            scramble_internal
        )
    )
    poses0 = tuple(
        int(x)
        for x in start.poses
    )
    q0 = int(
        ts.q_of(start)
    )

    deadline = (
        time.perf_counter()
        + float(time_cap)
    )
    t0 = time.perf_counter()

    nodes = 0
    q_prunes = 0
    oracle_calls = 0
    oracle_empty_prunes = 0
    first_column_values_removed = 0
    surviving_word_total = 0
    trigger_hist = Counter()

    solutions = []
    path = []

    class Cut(Exception):
        pass

    def check_cut():
        if nodes >= int(node_cap):
            raise Cut(
                "node_cap"
            )

        if (
            time.perf_counter()
            >= deadline
        ):
            raise Cut(
                "time_cap"
            )

    def mode_config(
        mode_name,
    ):
        if mode_name == "Q_ONLY":
            return None

        if mode_name == "WORD_PRUNE_T4":
            return {
                "max_r": 4,
                "add_q_first": False,
                "filter_c0": False,
            }

        if mode_name == "WORD_FILTER_T4":
            return {
                "max_r": 4,
                "add_q_first": False,
                "filter_c0": True,
            }

        if mode_name == "QWORD_FILTER_T4":
            return {
                "max_r": 4,
                "add_q_first": True,
                "filter_c0": True,
            }

        if mode_name == "QWORD_FILTER_T3":
            return {
                "max_r": 3,
                "add_q_first": True,
                "filter_c0": True,
            }

        raise ValueError(
            mode_name
        )

    config = mode_config(
        mode
    )

    def dfs(
        poses,
        q,
        depth,
        last_face,
    ):
        nonlocal nodes
        nonlocal q_prunes
        nonlocal oracle_calls
        nonlocal oracle_empty_prunes
        nonlocal first_column_values_removed
        nonlocal surviving_word_total

        check_cut()

        rem = int(
            depth_limit
        ) - depth

        hq = q_lb(
            q,
            qtables,
        )

        if hq > rem:
            q_prunes += 1
            return

        first_mask = (
            ALL_MOVE_MASK
        )

        if (
            config is not None
            and rem > 0
            and rem
            <= int(
                config[
                    "max_r"
                ]
            )
            and hq == rem
        ):
            trigger_hist[
                int(rem)
            ] += 1

            oracle_calls += 1

            qr = oracle.query(
                poses=poses,
                q=q,
                R=rem,
                last_face=last_face,
                add_q_first=bool(
                    config[
                        "add_q_first"
                    ]
                ),
            )

            if not qr[
                "feasible"
            ]:
                oracle_empty_prunes += 1
                return

            surviving_word_total += int(
                qr[
                    "word_count"
                ]
            )

            if config[
                "filter_c0"
            ]:
                first_mask = int(
                    qr[
                        "first_move_mask"
                    ]
                )

                legal = sum(
                    1
                    for move in ts.MOVE_ORDER
                    if (
                        last_face is None
                        or move[0]
                        != last_face
                    )
                )

                first_column_values_removed += max(
                    0,
                    legal
                    - first_mask.bit_count(),
                )

        nodes += 1

        if rem == 0:
            if int(q) == int(
                ts.GOAL_Q
            ):
                word = tuple(
                    path
                )

                final = (
                    start.apply_word(
                        word
                    )
                )

                if not ts.is_dr(
                    final
                ):
                    raise RuntimeError(
                        "q-goal / explicit DR mismatch"
                    )

                solutions.append(
                    word
                )

            return

        for mi, move in enumerate(
            ts.MOVE_ORDER
        ):
            if not (
                first_mask
                & (1 << mi)
            ):
                continue

            if (
                last_face
                is not None
                and move[0]
                == last_face
            ):
                continue

            nposes, nq = (
                step_state(
                    poses,
                    q,
                    move,
                    qtables,
                )
            )

            h2 = q_lb(
                nq,
                qtables,
            )

            if h2 > rem - 1:
                q_prunes += 1
                continue

            path.append(
                move
            )

            dfs(
                nposes,
                nq,
                depth + 1,
                move[0],
            )

            path.pop()

    try:
        dfs(
            poses0,
            q0,
            0,
            None,
        )
        cut = None
    except Cut as exc:
        cut = str(
            exc
        )

    return SearchResult(
        mode=mode,
        nodes=nodes,
        q_prunes=q_prunes,
        oracle_calls=oracle_calls,
        oracle_empty_prunes=(
            oracle_empty_prunes
        ),
        first_column_values_removed=(
            first_column_values_removed
        ),
        surviving_word_total=(
            surviving_word_total
        ),
        trigger_histogram=dict(
            sorted(
                trigger_hist.items()
            )
        ),
        solutions=tuple(
            sorted(
                set(
                    solutions
                )
            )
        ),
        wall=(
            time.perf_counter()
            - t0
        ),
        cut_reason=cut,
    )


def evaluate_control(
    *,
    name,
    scramble_original,
    axis,
    depth,
    qtables,
    oracle,
    node_cap,
    time_cap,
):
    sw = internalize(
        scramble_original,
        axis,
    )

    print(
        f"## {name} axis={axis} "
        f"exactDepth={depth}"
    )
    print(
        "scramble             :",
        " ".join(
            scramble_original
        ),
    )

    modes = (
        "Q_ONLY",
        "WORD_PRUNE_T4",
        "WORD_FILTER_T4",
        "QWORD_FILTER_T4",
        "QWORD_FILTER_T3",
    )

    rows = []

    baseline_set = None
    baseline_nodes = None
    baseline_wall = None

    for mode in modes:
        res = exact_search(
            mode=mode,
            scramble_internal=sw,
            depth_limit=depth,
            qtables=qtables,
            oracle=oracle,
            node_cap=node_cap,
            time_cap=time_cap,
        )

        external = {
            externalize(
                word,
                axis,
            )
            for word in res.solutions
        }

        if mode == "Q_ONLY":
            baseline_set = set(
                external
            )
            baseline_nodes = int(
                res.nodes
            )
            baseline_wall = float(
                res.wall
            )

        parity = (
            res.cut_reason is None
            and set(
                external
            )
            == baseline_set
        )

        node_gain = (
            0.0
            if not baseline_nodes
            else 1.0
            - res.nodes
            / baseline_nodes
        )

        wall_ratio = (
            1.0
            if not baseline_wall
            else res.wall
            / baseline_wall
        )

        print(
            f"{mode:<17}: "
            f"nodes={res.nodes:4d} "
            f"Qpr={res.q_prunes:5d} "
            f"Wcalls={res.oracle_calls:3d} "
            f"Wempty={res.oracle_empty_prunes:3d} "
            f"C0rm={res.first_column_values_removed:4d} "
            f"sol={len(external):3d} "
            f"gain={node_gain:6.2%} "
            f"wall={res.wall:.4f}s "
            f"xQ={wall_ratio:.2f} "
            f"trigger={res.trigger_histogram} "
            f"parity={parity}"
        )

        rows.append({
            "mode": mode,
            "nodes": res.nodes,
            "q_prunes": (
                res.q_prunes
            ),
            "oracle_calls": (
                res.oracle_calls
            ),
            "oracle_empty_prunes": (
                res.oracle_empty_prunes
            ),
            "first_column_values_removed": (
                res.first_column_values_removed
            ),
            "surviving_word_total": (
                res.surviving_word_total
            ),
            "trigger_histogram": (
                res.trigger_histogram
            ),
            "solution_parity": (
                parity
            ),
            "solution_count": len(
                external
            ),
            "node_gain": (
                node_gain
            ),
            "wall": res.wall,
            "wall_ratio_vs_q": (
                wall_ratio
            ),
            "cut_reason": (
                res.cut_reason
            ),
        })

    print()

    return {
        "name": name,
        "axis": axis,
        "depth": int(
            depth
        ),
        "scramble_original": list(
            scramble_original
        ),
        "baseline_nodes": (
            baseline_nodes
        ),
        "baseline_wall": (
            baseline_wall
        ),
        "baseline_solution_count": len(
            baseline_set
        ),
        "modes": (
            rows
        ),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--seed",
        type=int,
        default=20260925,
    )
    ap.add_argument(
        "--fresh-depths",
        nargs="+",
        type=int,
        default=[7, 8],
    )
    ap.add_argument(
        "--min-dead-per-fresh",
        type=int,
        default=3,
    )
    ap.add_argument(
        "--max-control-tries",
        type=int,
        default=30,
    )
    ap.add_argument(
        "--max-generation-tries",
        type=int,
        default=30000,
    )
    ap.add_argument(
        "--cache",
        default=(
            "reports/pdcc_cache/"
            "twist_skeleton_tables_v1.pkl"
        ),
    )
    ap.add_argument(
        "--node-cap",
        type=int,
        default=100000,
    )
    ap.add_argument(
        "--time-cap",
        type=float,
        default=3.0,
    )
    ap.add_argument(
        "--output",
        default=(
            "reports/v37/"
            "compiled_column_wordset_v37_83.json"
        ),
    )

    args = ap.parse_args()

    print(
        "# CubeLab v37.83 - "
        "COMPILED COLUMN-WORD SUPERPOSITION BITSET"
    )
    print(
        "combined model       :",
        "global columns + 20-piece superposition",
    )
    print(
        "compiled horizon     :",
        "R<=4 reduced HTM words",
    )
    print(
        "representation       :",
        "Python-int word bitsets",
    )
    print(
        "GAC loops            :",
        "NONE",
    )
    print(
        "Q synergy            :",
        "optional first-column Q mask before wordset test",
    )
    print(
        "production changes   :",
        "NONE",
    )
    print()

    t0 = (
        time.perf_counter()
    )

    qtables, loaded = (
        ts.cache_load_or_build(
            Path(
                args.cache
            )
        )
    )

    print(
        "phase tables         :",
        "cache"
        if loaded
        else "built",
    )
    print(
        "table wall           :",
        f"{time.perf_counter()-t0:.3f}s",
    )

    dr_targets = (
        v57.local_dr_pose_sets()
    )

    oracle_t0 = (
        time.perf_counter()
    )

    oracle = (
        CompiledWordSuperposition(
            dr_targets,
            qtables,
            max_horizon=4,
        )
    )

    oracle_total_wall = (
        time.perf_counter()
        - oracle_t0
    )

    print(
        "word universe counts :",
        {
            R: int(
                oracle.words[
                    R
                ].shape[0]
            )
            for R in range(
                1,
                5,
            )
        },
    )
    print(
        "piece-bitset build   :",
        f"{oracle.build_wall:.3f}s",
    )
    print(
        "oracle total build   :",
        f"{oracle_total_wall:.3f}s",
    )
    print(
        "bitset bytes estimate:",
        f"{oracle.bitset_bytes_estimate / (1024*1024):.2f} MiB",
    )
    print()

    rows = []

    m12 = next(
        c
        for c in v57.CONTROLS
        if c["name"]
        == "M12-02"
    )

    rows.append(
        evaluate_control(
            name="M12-02",
            scramble_original=tuple(
                m12[
                    "scramble"
                ]
            ),
            axis=m12[
                "axis"
            ],
            depth=3,
            qtables=qtables,
            oracle=oracle,
            node_cap=int(
                args.node_cap
            ),
            time_cap=float(
                args.time_cap
            ),
        )
    )

    m11 = next(
        c
        for c in v57.CONTROLS
        if c["name"]
        == "M11-01"
    )

    rows.append(
        evaluate_control(
            name="HIST-M11-L7",
            scramble_original=tuple(
                m11[
                    "scramble"
                ]
            ),
            axis=m11[
                "axis"
            ],
            depth=7,
            qtables=qtables,
            oracle=oracle,
            node_cap=int(
                args.node_cap
            ),
            time_cap=float(
                args.time_cap
            ),
        )
    )

    rng = random.Random(
        int(
            args.seed
        )
    )

    fresh_controls = []

    for depth in (
        int(x)
        for x in (
            args.fresh_depths
        )
    ):
        (
            control,
            _truth,
            control_try,
        ) = (
            v80.fresh_control_with_dead(
                depth=depth,
                rng=rng,
                qtables=qtables,
                min_dead=int(
                    args.min_dead_per_fresh
                ),
                max_control_tries=int(
                    args.max_control_tries
                ),
                max_generation_tries=int(
                    args.max_generation_tries
                ),
            )
        )

        control = {
            **control,
            "selection_control_try": (
                control_try
            ),
        }

        fresh_controls.append(
            control
        )

        rows.append(
            evaluate_control(
                name=f"FRESH-L{depth}",
                scramble_original=tuple(
                    control[
                        "scramble"
                    ]
                ),
                axis="UD",
                depth=depth,
                qtables=qtables,
                oracle=oracle,
                node_cap=int(
                    args.node_cap
                ),
                time_cap=float(
                    args.time_cap
                ),
            )
        )

    print("# DECISION")

    nonbase = [
        mode
        for row in rows
        for mode in row[
            "modes"
        ]
        if mode[
            "mode"
        ] != "Q_ONLY"
    ]

    sound = all(
        mode[
            "solution_parity"
        ]
        and mode[
            "cut_reason"
        ] is None
        for mode in nonbase
    )

    qword_rows = [
        next(
            mode
            for mode in row[
                "modes"
            ]
            if mode[
                "mode"
            ]
            == "QWORD_FILTER_T4"
        )
        for row in rows
    ]

    fresh_qword = [
        mode
        for row, mode in zip(
            rows,
            qword_rows,
        )
        if row[
            "name"
        ].startswith(
            "FRESH-"
        )
    ]

    any_fresh_gain = any(
        mode[
            "node_gain"
        ] > 0
        for mode in fresh_qword
    )

    mean_fresh_gain = (
        sum(
            mode[
                "node_gain"
            ]
            for mode in fresh_qword
        )
        / len(
            fresh_qword
        )
        if fresh_qword
        else 0.0
    )

    mean_fresh_wall = (
        sum(
            mode[
                "wall_ratio_vs_q"
            ]
            for mode in fresh_qword
        )
        / len(
            fresh_qword
        )
        if fresh_qword
        else float(
            "inf"
        )
    )

    hist_qword = next(
        mode
        for row, mode in zip(
            rows,
            qword_rows,
        )
        if row[
            "name"
        ] == "HIST-M11-L7"
    )

    if not sound:
        decision = (
            "COMPILED_WORDSET_CONTRACT_FAIL"
        )
        note = (
            "The compiled shared-word bitsets changed an exact solution set or "
            "cut. Fix the compilation before interpretation."
        )

    elif (
        any_fresh_gain
        and mean_fresh_wall <= 1.25
    ):
        decision = (
            "COMPILED_COLUMN_SUPERPOSITION_RUNTIME_PASS"
        )
        note = (
            "Exact shared-column word bitsets preserve the combined model's "
            "fresh-control node pruning while reducing hot-search runtime close "
            "to Q baseline. This is the first practical compiled form of column "
            "+ superposition."
        )

    elif any_fresh_gain:
        decision = (
            "COMPILED_COLUMN_SUPERPOSITION_NODE_PASS_RUNTIME_NEGATIVE"
        )
        note = (
            "The compiled wordset preserves real fresh-control pruning, but hot "
            "runtime is still too high. Keep the exact wordset formulation and "
            "optimize/cache the remaining query path."
        )

    elif (
        hist_qword[
            "node_gain"
        ] > 0
    ):
        decision = (
            "COMPILED_WORDSET_HISTORICAL_ONLY"
        )
        note = (
            "The compiled wordset reproduces the historical signal but loses the "
            "fresh full-tree gains seen with the iterative fixpoint. The Q/wordset "
            "interaction needs richer prefix-Q propagation."
        )

    else:
        decision = (
            "COMPILED_WORDSET_NO_SIGNAL"
        )
        note = (
            "The simple shared-word compilation does not reproduce the useful "
            "combined-model full-tree pruning."
        )

    print(
        decision
    )
    print(
        note
    )
    print(
        "mean fresh QWORD node gain:",
        f"{mean_fresh_gain:.2%}",
    )
    print(
        "mean fresh QWORD wall ratio:",
        f"{mean_fresh_wall:.2f}x",
    )
    print(
        "oracle reusable build:",
        f"{oracle_total_wall:.3f}s / "
        f"{oracle.bitset_bytes_estimate/(1024*1024):.2f} MiB",
    )

    payload = {
        "version": "v37.83",
        "mode": (
            "COMPILED_COLUMN_WORD_SUPERPOSITION_BITSET"
        ),
        "seed": int(
            args.seed
        ),
        "oracle": {
            "max_horizon": 4,
            "word_counts": {
                str(R): int(
                    oracle.words[
                        R
                    ].shape[0]
                )
                for R in range(
                    1,
                    5,
                )
            },
            "piece_bitset_build_wall": (
                oracle.build_wall
            ),
            "total_build_wall": (
                oracle_total_wall
            ),
            "bitset_bytes_estimate": (
                oracle.bitset_bytes_estimate
            ),
        },
        "controls": (
            rows
        ),
        "fresh_controls": [
            {
                **control,
                "scramble": list(
                    control[
                        "scramble"
                    ]
                ),
                "planted_inverse": list(
                    control[
                        "planted_inverse"
                    ]
                ),
            }
            for control in (
                fresh_controls
            )
        ],
        "decision": (
            decision
        ),
        "note": (
            note
        ),
        "mean_fresh_qword_node_gain": (
            mean_fresh_gain
        ),
        "mean_fresh_qword_wall_ratio": (
            mean_fresh_wall
        ),
    }

    out = Path(
        args.output
    )
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=repr,
        ),
        encoding="utf-8",
    )

    print(
        "JSON                 :",
        out,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
