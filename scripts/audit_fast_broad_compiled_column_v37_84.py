#!/usr/bin/env python3
"""
CubeLab v37.84 — BROAD COMPILED COLUMN-SUPERPOSITION A/B

Why
---
v37.83 compiled the validated:

    GLOBAL COLUMNS + 20-PIECE SUPERPOSITION

model into exact reduced-word bitsets for remaining horizon R<=4.

Promising hot-path result on selected controls:

    HIST M11 : 34 -> 22 nodes, WORD_FILTER wall ~0.80x
    FRESH L7 : 32 -> 27 nodes, WORD_FILTER wall ~0.70x
    FRESH L8 : 60 -> 52 nodes, WORD_FILTER wall ~0.74x

But the fresh L7/L8 controls were deliberately selected to contain several
Q-surviving dead prefixes.  That is an OPPORTUNITY-RICH sample.

Before integration into the stable search baseline, test an UNBIASED fresh
exact-depth corpus.

Fresh corpus
------------
Default:
    depths = 5 6 7 8
    5 independent exact controls per depth
    = 20 controls

Controls are generated only by:
    random reduced planted word length L
    + root phase1_lb == L

No filtering for dead-prefix count or oracle opportunity.

Modes
-----
Q_ONLY
WORD_PRUNE_T4
WORD_FILTER_T4

WORD_FILTER_T4 is the practical candidate:
    exact shared 20-piece column-word intersection
    -> supported first-column moves
    -> ordinary Q only on surviving branches

No extra Q computation is performed inside the oracle.

Timing
------
Searches are tiny, so one wall sample is noisy.

For each control/mode:
    run --repeats times (default 7)
    require deterministic node/solution parity
    report median wall

Oracle cache
------------
The R<=4 compiled bitsets are scramble-independent.

This script serializes only the reusable compiled data (NOT qtables) to:

    reports/pdcc_cache/compiled_column_wordset_r4_v37_84.pkl

On later runs it loads the cache directly.

Decision
--------
BROAD_COMPILED_COLUMN_RUNTIME_PASS
    exact parity on every case
    fresh median node gain >0
    median WORD_FILTER wall ratio <1
    majority of fresh cases are runtime wins/ties

BROAD_COMPILED_COLUMN_NODE_PASS_RUNTIME_MIXED
    fresh node signal generalizes but wall result is mixed

BROAD_COMPILED_COLUMN_OPPORTUNITY_SPECIFIC
    little/no node gain on unbiased corpus

This remains an exact-depth Q microbenchmark.
Even a pass must next be A/B integrated into the stable v37.24.1 search;
it is not itself a production change.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

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

import audit_fast_local_dr_superposition_v37_57 as v57
import audit_fast_fresh_exact5_exact8_v37_73 as v73
import audit_fast_compiled_column_wordset_v37_83 as v83


CACHE_VERSION = "compiled-column-wordset-r4-v37.84"


def save_oracle_cache(path, oracle):
    payload = {
        "version": CACHE_VERSION,
        "max_horizon": int(
            oracle.max_horizon
        ),
        "words": oracle.words,
        "first_move_masks": (
            oracle.first_move_masks
        ),
        "first_face_masks": (
            oracle.first_face_masks
        ),
        "universe_masks": (
            oracle.universe_masks
        ),
        "piece_masks": (
            oracle.piece_masks
        ),
        "bitset_bytes_estimate": int(
            oracle.bitset_bytes_estimate
        ),
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "wb"
    ) as f:
        pickle.dump(
            payload,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_or_build_oracle(
    *,
    cache_path,
    dr_targets,
    qtables,
):
    if cache_path.is_file():
        t0 = time.perf_counter()

        try:
            with cache_path.open(
                "rb"
            ) as f:
                payload = pickle.load(
                    f
                )

            if (
                payload.get(
                    "version"
                )
                != CACHE_VERSION
            ):
                raise ValueError(
                    "cache version mismatch"
                )

            oracle = object.__new__(
                v83.CompiledWordSuperposition
            )

            oracle.max_horizon = int(
                payload[
                    "max_horizon"
                ]
            )
            oracle.qtables = qtables
            oracle.words = payload[
                "words"
            ]
            oracle.first_move_masks = (
                payload[
                    "first_move_masks"
                ]
            )
            oracle.first_face_masks = (
                payload[
                    "first_face_masks"
                ]
            )
            oracle.universe_masks = (
                payload[
                    "universe_masks"
                ]
            )
            oracle.piece_masks = (
                payload[
                    "piece_masks"
                ]
            )
            oracle.bitset_bytes_estimate = int(
                payload[
                    "bitset_bytes_estimate"
                ]
            )
            oracle.build_wall = 0.0

            return (
                oracle,
                "cache",
                time.perf_counter()
                - t0,
            )

        except Exception as exc:
            print(
                "oracle cache reload failed:",
                repr(exc),
            )
            print(
                "rebuilding oracle..."
            )

    t0 = time.perf_counter()

    oracle = (
        v83.CompiledWordSuperposition(
            dr_targets,
            qtables,
            max_horizon=4,
        )
    )

    elapsed = (
        time.perf_counter()
        - t0
    )

    save_oracle_cache(
        cache_path,
        oracle,
    )

    return (
        oracle,
        "built",
        elapsed,
    )


def run_mode_repeated(
    *,
    mode,
    scramble,
    depth,
    qtables,
    oracle,
    node_cap,
    time_cap,
    repeats,
):
    walls = []
    reference = None

    for rep in range(
        int(repeats)
    ):
        res = v83.exact_search(
            mode=mode,
            scramble_internal=tuple(
                scramble
            ),
            depth_limit=int(
                depth
            ),
            qtables=qtables,
            oracle=oracle,
            node_cap=int(
                node_cap
            ),
            time_cap=float(
                time_cap
            ),
        )

        if res.cut_reason is not None:
            return {
                "cut_reason": (
                    res.cut_reason
                ),
                "repeat": rep,
            }

        signature = (
            int(res.nodes),
            int(res.q_prunes),
            int(res.oracle_calls),
            int(
                res.oracle_empty_prunes
            ),
            int(
                res.first_column_values_removed
            ),
            tuple(
                res.solutions
            ),
            tuple(
                sorted(
                    res.trigger_histogram.items()
                )
            ),
        )

        if reference is None:
            reference = (
                res,
                signature,
            )
        elif signature != reference[1]:
            raise RuntimeError(
                f"{mode}: nondeterministic search signature"
            )

        walls.append(
            float(
                res.wall
            )
        )

    res = reference[0]

    return {
        "cut_reason": None,
        "nodes": int(
            res.nodes
        ),
        "q_prunes": int(
            res.q_prunes
        ),
        "oracle_calls": int(
            res.oracle_calls
        ),
        "oracle_empty_prunes": int(
            res.oracle_empty_prunes
        ),
        "first_column_values_removed": int(
            res.first_column_values_removed
        ),
        "surviving_word_total": int(
            res.surviving_word_total
        ),
        "trigger_histogram": dict(
            res.trigger_histogram
        ),
        "solutions": tuple(
            res.solutions
        ),
        "wall_samples": walls,
        "median_wall": float(
            statistics.median(
                walls
            )
        ),
        "min_wall": float(
            min(
                walls
            )
        ),
        "max_wall": float(
            max(
                walls
            )
        ),
    }


def evaluate_control(
    *,
    case_id,
    control,
    qtables,
    oracle,
    node_cap,
    time_cap,
    repeats,
):
    depth = int(
        control[
            "depth"
        ]
    )
    scramble = tuple(
        control[
            "scramble"
        ]
    )

    mode_rows = {}

    for mode in (
        "Q_ONLY",
        "WORD_PRUNE_T4",
        "WORD_FILTER_T4",
    ):
        mode_rows[
            mode
        ] = run_mode_repeated(
            mode=mode,
            scramble=scramble,
            depth=depth,
            qtables=qtables,
            oracle=oracle,
            node_cap=node_cap,
            time_cap=time_cap,
            repeats=repeats,
        )

    if any(
        row[
            "cut_reason"
        ] is not None
        for row in mode_rows.values()
    ):
        return {
            "case_id": case_id,
            "depth": depth,
            "scramble": list(
                scramble
            ),
            "cut": True,
            "modes": mode_rows,
        }

    q = mode_rows[
        "Q_ONLY"
    ]
    p = mode_rows[
        "WORD_PRUNE_T4"
    ]
    f = mode_rows[
        "WORD_FILTER_T4"
    ]

    qset = set(
        q[
            "solutions"
        ]
    )

    parity_prune = (
        set(
            p[
                "solutions"
            ]
        )
        == qset
    )
    parity_filter = (
        set(
            f[
                "solutions"
            ]
        )
        == qset
    )

    node_gain_prune = (
        1.0
        - p[
            "nodes"
        ]
        / q[
            "nodes"
        ]
        if q[
            "nodes"
        ]
        else 0.0
    )

    node_gain_filter = (
        1.0
        - f[
            "nodes"
        ]
        / q[
            "nodes"
        ]
        if q[
            "nodes"
        ]
        else 0.0
    )

    wall_ratio_prune = (
        p[
            "median_wall"
        ]
        / q[
            "median_wall"
        ]
        if q[
            "median_wall"
        ] > 0
        else float(
            "inf"
        )
    )

    wall_ratio_filter = (
        f[
            "median_wall"
        ]
        / q[
            "median_wall"
        ]
        if q[
            "median_wall"
        ] > 0
        else float(
            "inf"
        )
    )

    print(
        f"{case_id:<12} "
        f"L={depth} "
        f"Qnodes={q['nodes']:4d} "
        f"P={p['nodes']:4d} "
        f"F={f['nodes']:4d} "
        f"gainF={node_gain_filter:6.2%} "
        f"wall P/F="
        f"{wall_ratio_prune:.2f}x/"
        f"{wall_ratio_filter:.2f}x "
        f"Wempty={f['oracle_empty_prunes']:3d} "
        f"C0rm={f['first_column_values_removed']:4d} "
        f"sol={len(qset):3d} "
        f"parity={parity_prune}/{parity_filter}"
    )

    return {
        "case_id": case_id,
        "depth": depth,
        "scramble": list(
            scramble
        ),
        "generation_attempt": int(
            control[
                "generation_attempt"
            ]
        ),
        "cut": False,
        "solution_count": len(
            qset
        ),
        "parity_prune": (
            parity_prune
        ),
        "parity_filter": (
            parity_filter
        ),
        "node_gain_prune": (
            node_gain_prune
        ),
        "node_gain_filter": (
            node_gain_filter
        ),
        "wall_ratio_prune": (
            wall_ratio_prune
        ),
        "wall_ratio_filter": (
            wall_ratio_filter
        ),
        "modes": {
            mode: {
                k: (
                    [
                        list(x)
                        for x in v[
                            "solutions"
                        ]
                    ]
                    if k
                    == "solutions"
                    else v[k]
                )
                for k in v
                if k
                != "solutions"
            }
            | {
                "solutions": [
                    list(x)
                    for x in v[
                        "solutions"
                    ]
                ]
            }
            for mode, v in (
                mode_rows.items()
            )
        },
    }


def aggregate_rows(
    rows,
):
    valid = [
        row
        for row in rows
        if not row[
            "cut"
        ]
    ]

    if not valid:
        return {}

    def mean(name):
        return sum(
            float(
                row[
                    name
                ]
            )
            for row in valid
        ) / len(
            valid
        )

    filter_ratios = [
        float(
            row[
                "wall_ratio_filter"
            ]
        )
        for row in valid
    ]

    prune_ratios = [
        float(
            row[
                "wall_ratio_prune"
            ]
        )
        for row in valid
    ]

    filter_gains = [
        float(
            row[
                "node_gain_filter"
            ]
        )
        for row in valid
    ]

    runtime_wins = sum(
        ratio < 0.98
        for ratio in filter_ratios
    )
    runtime_ties = sum(
        0.98 <= ratio <= 1.02
        for ratio in filter_ratios
    )
    runtime_losses = sum(
        ratio > 1.02
        for ratio in filter_ratios
    )

    node_wins = sum(
        gain > 0
        for gain in filter_gains
    )

    return {
        "case_count": len(
            valid
        ),
        "mean_filter_node_gain": mean(
            "node_gain_filter"
        ),
        "median_filter_node_gain": float(
            statistics.median(
                filter_gains
            )
        ),
        "mean_filter_wall_ratio": mean(
            "wall_ratio_filter"
        ),
        "median_filter_wall_ratio": float(
            statistics.median(
                filter_ratios
            )
        ),
        "mean_prune_wall_ratio": mean(
            "wall_ratio_prune"
        ),
        "median_prune_wall_ratio": float(
            statistics.median(
                prune_ratios
            )
        ),
        "filter_runtime_wins": int(
            runtime_wins
        ),
        "filter_runtime_ties": int(
            runtime_ties
        ),
        "filter_runtime_losses": int(
            runtime_losses
        ),
        "filter_node_gain_cases": int(
            node_wins
        ),
        "all_solution_parity": all(
            row[
                "parity_prune"
            ]
            and row[
                "parity_filter"
            ]
            for row in valid
        ),
        "total_word_empty_prunes_filter": sum(
            int(
                row[
                    "modes"
                ][
                    "WORD_FILTER_T4"
                ][
                    "oracle_empty_prunes"
                ]
            )
            for row in valid
        ),
        "total_first_column_values_removed": sum(
            int(
                row[
                    "modes"
                ][
                    "WORD_FILTER_T4"
                ][
                    "first_column_values_removed"
                ]
            )
            for row in valid
        ),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=[
            5,
            6,
            7,
            8,
        ],
    )
    ap.add_argument(
        "--count-per-depth",
        type=int,
        default=5,
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=7,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=20260926,
    )
    ap.add_argument(
        "--max-generation-tries",
        type=int,
        default=30000,
    )
    ap.add_argument(
        "--phase-cache",
        default=(
            "reports/pdcc_cache/"
            "twist_skeleton_tables_v1.pkl"
        ),
    )
    ap.add_argument(
        "--oracle-cache",
        default=(
            "reports/pdcc_cache/"
            "compiled_column_wordset_r4_v37_84.pkl"
        ),
    )
    ap.add_argument(
        "--node-cap",
        type=int,
        default=1_000_000,
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
            "broad_compiled_column_ab_v37_84.json"
        ),
    )

    args = ap.parse_args()

    depths = tuple(
        int(x)
        for x in args.depths
    )

    print(
        "# CubeLab v37.84 - "
        "BROAD COMPILED COLUMN-SUPERPOSITION A/B"
    )
    print(
        "fresh corpus         :",
        f"depths={list(depths)} "
        f"count/depth={args.count_per_depth}",
    )
    print(
        "selection            :",
        "UNBIASED exact controls; no dead-prefix filter",
    )
    print(
        "modes                :",
        "Q_ONLY / WORD_PRUNE_T4 / WORD_FILTER_T4",
    )
    print(
        "timing               :",
        f"median of {args.repeats} repeats",
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
                args.phase_cache
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
        "phase table wall     :",
        f"{time.perf_counter()-t0:.3f}s",
    )

    dr_targets = (
        v57.local_dr_pose_sets()
    )

    (
        oracle,
        oracle_source,
        oracle_wall,
    ) = load_or_build_oracle(
        cache_path=Path(
            args.oracle_cache
        ),
        dr_targets=dr_targets,
        qtables=qtables,
    )

    print(
        "compiled oracle      :",
        oracle_source,
    )
    print(
        "oracle load/build wall:",
        f"{oracle_wall:.3f}s",
    )
    print(
        "oracle footprint     :",
        f"{oracle.bitset_bytes_estimate/(1024*1024):.2f} MiB",
    )
    print()

    rng = random.Random(
        int(
            args.seed
        )
    )

    controls = []

    for depth in depths:
        for idx in range(
            int(
                args.count_per_depth
            )
        ):
            control = (
                v73.make_fresh_exact_control(
                    target_depth=depth,
                    rng=rng,
                    qtables=qtables,
                    max_tries=int(
                        args.max_generation_tries
                    ),
                )
            )

            controls.append(
                (
                    f"L{depth}-{idx+1}",
                    control,
                )
            )

    rows = []

    total_t0 = (
        time.perf_counter()
    )

    for case_id, control in controls:
        rows.append(
            evaluate_control(
                case_id=case_id,
                control=control,
                qtables=qtables,
                oracle=oracle,
                node_cap=int(
                    args.node_cap
                ),
                time_cap=float(
                    args.time_cap
                ),
                repeats=int(
                    args.repeats
                ),
            )
        )

    total_wall = (
        time.perf_counter()
        - total_t0
    )

    aggregate = (
        aggregate_rows(
            rows
        )
    )

    by_depth = {}

    for depth in depths:
        block = [
            row
            for row in rows
            if (
                not row[
                    "cut"
                ]
                and int(
                    row[
                        "depth"
                    ]
                )
                == depth
            )
        ]

        by_depth[
            str(
                depth
            )
        ] = aggregate_rows(
            block
        )

    print()
    print("# AGGREGATE FRESH")
    print(
        "solution parity      :",
        aggregate.get(
            "all_solution_parity"
        ),
    )
    print(
        "FILTER node gain mean/median:",
        f"{aggregate.get('mean_filter_node_gain', 0):.2%} / "
        f"{aggregate.get('median_filter_node_gain', 0):.2%}",
    )
    print(
        "FILTER wall ratio mean/median:",
        f"{aggregate.get('mean_filter_wall_ratio', float('nan')):.3f}x / "
        f"{aggregate.get('median_filter_wall_ratio', float('nan')):.3f}x",
    )
    print(
        "FILTER runtime W/T/L :",
        f"{aggregate.get('filter_runtime_wins', 0)}/"
        f"{aggregate.get('filter_runtime_ties', 0)}/"
        f"{aggregate.get('filter_runtime_losses', 0)}",
    )
    print(
        "FILTER node-gain cases:",
        f"{aggregate.get('filter_node_gain_cases', 0)}/"
        f"{aggregate.get('case_count', 0)}",
    )
    print(
        "word-empty prunes    :",
        aggregate.get(
            "total_word_empty_prunes_filter"
        ),
    )
    print(
        "C0 values removed    :",
        aggregate.get(
            "total_first_column_values_removed"
        ),
    )

    for depth in depths:
        a = by_depth[
            str(
                depth
            )
        ]

        print(
            f"L{depth}: "
            f"nodeGain={a.get('mean_filter_node_gain', 0):.2%} "
            f"wallMed={a.get('median_filter_wall_ratio', float('nan')):.3f}x "
            f"nodeCases={a.get('filter_node_gain_cases', 0)}/"
            f"{a.get('case_count', 0)}"
        )

    print()
    print("# DECISION")

    sound = bool(
        aggregate.get(
            "all_solution_parity",
            False,
        )
    )

    median_gain = float(
        aggregate.get(
            "median_filter_node_gain",
            0.0,
        )
    )
    mean_gain = float(
        aggregate.get(
            "mean_filter_node_gain",
            0.0,
        )
    )
    median_wall = float(
        aggregate.get(
            "median_filter_wall_ratio",
            float(
                "inf"
            ),
        )
    )

    wins = int(
        aggregate.get(
            "filter_runtime_wins",
            0,
        )
    )
    ties = int(
        aggregate.get(
            "filter_runtime_ties",
            0,
        )
    )
    losses = int(
        aggregate.get(
            "filter_runtime_losses",
            0,
        )
    )

    case_count = int(
        aggregate.get(
            "case_count",
            0,
        )
    )

    node_gain_cases = int(
        aggregate.get(
            "filter_node_gain_cases",
            0,
        )
    )

    if not sound:
        decision = (
            "BROAD_COMPILED_COLUMN_CONTRACT_FAIL"
        )
        note = (
            "At least one unbiased fresh control changed the exact solution set. "
            "Do not integrate the compiled wordset."
        )

    elif (
        mean_gain > 0
        and median_wall < 1.0
        and wins + ties
        >= math.ceil(
            0.6
            * case_count
        )
    ):
        decision = (
            "BROAD_COMPILED_COLUMN_RUNTIME_PASS"
        )
        note = (
            "The compiled column-superposition filter preserves every exact "
            "solution set on the unbiased fresh corpus, gives positive average "
            "node reduction, and improves median hot-search runtime. Proceed to "
            "an A/B integration against the stable v37.24.1 search baseline."
        )

    elif (
        mean_gain > 0
        and node_gain_cases > 0
    ):
        decision = (
            "BROAD_COMPILED_COLUMN_NODE_PASS_RUNTIME_MIXED"
        )
        note = (
            "The compiled column-superposition signal generalizes to the "
            "unbiased corpus at the node level, but timing is mixed. Optimize "
            "query placement/cache before stable-baseline integration."
        )

    else:
        decision = (
            "BROAD_COMPILED_COLUMN_OPPORTUNITY_SPECIFIC"
        )
        note = (
            "The large v37.83 gains were concentrated in opportunity-rich "
            "controls; the unbiased corpus shows little reusable node signal. "
            "Keep the combined model for composer/diagnostic use rather than "
            "hot-path integration."
        )

    print(
        decision
    )
    print(
        note
    )
    print(
        "total benchmark wall:",
        f"{total_wall:.3f}s",
    )

    payload = {
        "version": "v37.84",
        "mode": (
            "BROAD_UNBIASED_COMPILED_COLUMN_SUPERPOSITION_AB"
        ),
        "seed": int(
            args.seed
        ),
        "depths": list(
            depths
        ),
        "count_per_depth": int(
            args.count_per_depth
        ),
        "repeats": int(
            args.repeats
        ),
        "oracle": {
            "source": (
                oracle_source
            ),
            "load_or_build_wall": (
                oracle_wall
            ),
            "bitset_bytes_estimate": int(
                oracle.bitset_bytes_estimate
            ),
            "cache_path": str(
                args.oracle_cache
            ),
        },
        "controls": [
            {
                "case_id": case_id,
                **{
                    k: (
                        list(v)
                        if isinstance(
                            v,
                            tuple,
                        )
                        else v
                    )
                    for k, v in control.items()
                },
            }
            for case_id, control
            in controls
        ],
        "results": rows,
        "aggregate": (
            aggregate
        ),
        "by_depth": (
            by_depth
        ),
        "decision": (
            decision
        ),
        "note": (
            note
        ),
        "total_benchmark_wall": (
            total_wall
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
