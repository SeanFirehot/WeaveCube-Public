#!/usr/bin/env python3
"""
CubeLab v37.79 — BROAD OPPORTUNITY-NORMALIZED GAC SURVEY

Why
---
v37.65/v37.77 historical M11:
    one Q-surviving dead prefix
    full regular-language GAC detects it
    exact minimum core = UR/BL/BR

v37.78:
    one fresh exact-5 + one fresh exact-8
    zero full-GAC conflicts

That is far too small a sample to conclude non-generality.

Also:
    "no GAC conflict"
does NOT imply:
    "no joint incompatibility"

because arc consistency is only a relaxation.

This survey measures OPPORTUNITY-NORMALIZED detection.

For each fresh exact-depth control:
    1. Generate a random reduced scramble with root phase1_lb == target depth.
       The planted inverse proves an upper bound of the same depth, so the
       DR distance is exact.
    2. Traverse the entire exact-depth Q frontier.
    3. For EVERY Q-surviving prefix, compute ground truth:
           LIVE = has at least one exact DR completion in remaining depth
           DEAD = has no exact DR completion
    4. Run full 20-piece regular-language GAC on that same prefix.
    5. Classify:
           true_dead
           GAC_conflict
           GAC_detected_dead
           false_positive (must be zero)
    6. For up to a small global number of GAC conflicts, extract minimum
       low-order cores (exact through configurable size, default 4).

Fresh corpus
------------
Default:
    depths = 5 6 7 8
    3 independent exact controls per depth
    = 12 fresh controls

Stop core extraction after a few hits; continue cheap conflict counting.

Historical M11 exact-7 is included as a positive calibration and must
recover the known UR/BL/BR conflict/core.

Key metrics
-----------
Per control:
    Q frontier nodes
    Q-dead prefix count
    GAC conflict count
    GAC detection recall among dead prefixes
    whether there was any opportunity at all

Aggregate fresh:
    controls with dead opportunities
    controls with GAC-detected conflicts
    total dead opportunities
    total GAC-detected dead prefixes
    detection recall
    unique low-order core families

Interpretation
--------------
A. Conflicts replicate across independent scrambles:
       regular-superposition signal is general enough to continue.

B. Many dead Q-prefix opportunities exist, but GAC detects essentially none:
       current GAC relaxation is too weak/general signal not operational.

C. Fresh controls contain almost no Q-dead opportunities:
       survey is inconclusive; Q is already too tight on this corpus.

D. Fresh conflicts occur with different <=4-piece cores:
       reusable object is a dynamic low-order core family, not UR/BL/BR.

Diagnostic only.
No production changes.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
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
import search_pdcc_guided as pg

import audit_fast_local_dr_superposition_v37_57 as v57
import audit_fast_regular_superposition_v37_64 as v64
import audit_fast_regular_conflict_core_v37_65 as v65
import audit_fast_requirement_nonuniform_v37_70 as v70
import audit_fast_fresh_exact5_exact8_v37_73 as v73


PIECE_NAMES = tuple(v64.PIECE_NAMES)
HIST_CORE = frozenset(("UR", "BL", "BR"))


def q_lb(q, qtables):
    return int(
        ts.phase1_lb(
            int(q),
            qtables[3],
            qtables[4],
        )
    )


def step_state(poses, q, move, qtables):
    mi = ts.MI[move]

    nposes = tuple(
        int(x)
        for x in pg.move_poses(
            poses,
            mi,
        )
    )

    nq = int(
        ts.q_move(
            int(q),
            mi,
            qtables[0],
            qtables[1],
            qtables[2],
        )
    )

    return nposes, nq


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


def q_frontier_with_truth(
    *,
    scramble_internal,
    depth_limit,
    qtables,
):
    """
    Build the exact Q-surviving tree and annotate each node with whether it has
    at least one exact DR descendant.

    This deliberately uses the tree (not aggressive transposition merging),
    because our object of study is prefix-level conflict opportunity.
    """
    start = ts.engine.from_word(
        " ".join(scramble_internal)
    )
    poses0 = tuple(
        int(x)
        for x in start.poses
    )
    q0 = int(ts.q_of(start))

    nodes = []
    path = []

    def rec(poses, q, depth, last_face):
        rem = int(depth_limit) - depth
        h = q_lb(q, qtables)

        if h > rem:
            return False

        node_index = len(nodes)

        row = {
            "prefix_internal": tuple(path),
            "poses": tuple(poses),
            "q": int(q),
            "depth": int(depth),
            "remaining": int(rem),
            "last_face": last_face,
            "q_lb": int(h),
            "live": None,
        }
        nodes.append(row)

        if rem == 0:
            live = int(q) == int(ts.GOAL_Q)
            row["live"] = bool(live)

            if live:
                final = start.apply_word(
                    tuple(path)
                )
                if not ts.is_dr(final):
                    raise RuntimeError(
                        "q-goal / explicit DR mismatch"
                    )

            return bool(live)

        live = False

        for move in ts.MOVE_ORDER:
            if (
                last_face is not None
                and move[0] == last_face
            ):
                continue

            nposes, nq = step_state(
                poses,
                q,
                move,
                qtables,
            )

            h2 = q_lb(
                nq,
                qtables,
            )

            if h2 > rem - 1:
                continue

            path.append(move)
            child_live = rec(
                nposes,
                nq,
                depth + 1,
                move[0],
            )
            path.pop()

            live = live or child_live

        row["live"] = bool(live)
        return bool(live)

    root_live = rec(
        poses0,
        q0,
        0,
        None,
    )

    if not root_live:
        raise RuntimeError(
            "exact control root unexpectedly has no exact-depth solution"
        )

    return {
        "nodes": nodes,
        "root_live": root_live,
        "solution_count": sum(
            1
            for row in nodes
            if (
                row["remaining"] == 0
                and row["live"]
            )
        ),
    }


def gac_result_for_node(
    row,
    full_gac,
):
    gres = full_gac.propagate(
        row["poses"],
        row["remaining"],
        row["last_face"],
    )

    return {
        "gac_feasible": bool(
            gres.feasible
        ),
        "gac_conflict": not bool(
            gres.feasible
        ),
        "gac_iterations": int(
            gres.iterations
        ),
        "gac_removed_values": int(
            gres.removed_values
        ),
        "gac_domain_sizes": [
            x.bit_count()
            for x in gres.domains
        ],
    }


def make_conflict_adapter(row, gres_info):
    """
    Adapt our truth-tree row to the dictionary shape expected by v65 core tools.
    """
    return {
        "prefix_internal": tuple(
            row[
                "prefix_internal"
            ]
        ),
        "poses": tuple(
            row["poses"]
        ),
        "q": int(row["q"]),
        "depth": int(
            row["depth"]
        ),
        "remaining": int(
            row["remaining"]
        ),
        "last_face": row[
            "last_face"
        ],
        "q_lb": int(
            row["q_lb"]
        ),
        "gac_iterations": int(
            gres_info[
                "gac_iterations"
            ]
        ),
        "gac_removed_values": int(
            gres_info[
                "gac_removed_values"
            ]
        ),
        "gac_domains": tuple(
            (1 << int(size)) - 1
            for size in []
        ),
    }


def extract_core(
    *,
    row,
    subset_gac,
    max_exact_core_size,
):
    """
    v65 core functions only need poses/remaining/last_face for propagation.
    Construct a minimal conflict dict.
    """
    conflict = {
        "poses": tuple(
            row["poses"]
        ),
        "remaining": int(
            row["remaining"]
        ),
        "last_face": row[
            "last_face"
        ],
    }

    greedy_t0 = time.perf_counter()
    greedy = v65.greedy_core(
        subset_gac,
        conflict,
        tuple(range(20)),
    )
    greedy_wall = (
        time.perf_counter()
        - greedy_t0
    )

    exact_t0 = time.perf_counter()

    exact_core, checked, exact_res = (
        v65.smallest_core_up_to(
            subset_gac,
            conflict,
            max_size=int(
                max_exact_core_size
            ),
            upper_core=greedy,
        )
    )

    exact_wall = (
        time.perf_counter()
        - exact_t0
    )

    if exact_core is not None:
        selected = tuple(
            exact_core
        )
        core_type = (
            "EXACT_MIN_WITHIN_SEARCH"
        )
        core_res = exact_res
    else:
        selected = tuple(
            greedy
        )
        core_type = (
            "GREEDY_IRREDUCIBLE"
        )
        core_res = subset_gac.propagate(
            conflict["poses"],
            conflict[
                "remaining"
            ],
            conflict[
                "last_face"
            ],
            selected,
        )

    return {
        "greedy_core": [
            PIECE_NAMES[i]
            for i in greedy
        ],
        "greedy_core_size": len(
            greedy
        ),
        "greedy_wall": (
            greedy_wall
        ),
        "selected_core": [
            PIECE_NAMES[i]
            for i in selected
        ],
        "selected_core_indices": list(
            selected
        ),
        "selected_core_size": len(
            selected
        ),
        "selected_core_type": (
            core_type
        ),
        "subset_checks": int(
            checked
        ),
        "core_search_wall": (
            exact_wall
        ),
        "failure_stage": (
            None
            if core_res is None
            else core_res.get(
                "failure_stage"
            )
        ),
    }


def analyze_control(
    *,
    name,
    scramble_original,
    axis,
    depth,
    qtables,
    full_gac,
    subset_gac,
    max_exact_core_size,
    core_budget_state,
):
    sw = internalize(
        scramble_original,
        axis,
    )

    t0 = time.perf_counter()

    truth = q_frontier_with_truth(
        scramble_internal=sw,
        depth_limit=depth,
        qtables=qtables,
    )

    rows = truth["nodes"]

    dead_rows = [
        row
        for row in rows
        if not row["live"]
    ]

    # Leaves that are non-goal are Q-surviving dead nodes too, but GAC at R=0
    # trivially sees them. For structural pruning opportunity, report both all
    # dead and nonterminal dead.
    nonterminal_dead = [
        row
        for row in dead_rows
        if row["remaining"] > 0
    ]

    gac_conflicts = []
    false_positives = []
    detected_dead = []

    for row in rows:
        gi = gac_result_for_node(
            row,
            full_gac,
        )

        row["gac_conflict"] = bool(
            gi[
                "gac_conflict"
            ]
        )
        row["gac_iterations"] = (
            gi[
                "gac_iterations"
            ]
        )
        row[
            "gac_removed_values"
        ] = gi[
            "gac_removed_values"
        ]
        row[
            "gac_domain_sizes"
        ] = gi[
            "gac_domain_sizes"
        ]

        if row["gac_conflict"]:
            gac_conflicts.append(
                row
            )

            if row["live"]:
                false_positives.append(
                    row
                )
            else:
                detected_dead.append(
                    row
                )

    if false_positives:
        raise RuntimeError(
            f"{name}: GAC false positive on a live exact prefix"
        )

    conflict_details = []

    for row in gac_conflicts:
        prefix_original = (
            externalize(
                row[
                    "prefix_internal"
                ],
                axis,
            )
        )

        detail = {
            "prefix_internal": list(
                row[
                    "prefix_internal"
                ]
            ),
            "prefix_original": list(
                prefix_original
            ),
            "depth": int(
                row["depth"]
            ),
            "remaining": int(
                row["remaining"]
            ),
            "q_lb": int(
                row["q_lb"]
            ),
            "live": bool(
                row["live"]
            ),
            "gac_removed_values": int(
                row[
                    "gac_removed_values"
                ]
            ),
            "gac_domain_sizes": list(
                row[
                    "gac_domain_sizes"
                ]
            ),
        }

        if (
            core_budget_state[
                "remaining"
            ] > 0
        ):
            core = extract_core(
                row=row,
                subset_gac=subset_gac,
                max_exact_core_size=(
                    max_exact_core_size
                ),
            )

            detail.update(
                core
            )

            core_budget_state[
                "remaining"
            ] -= 1
        else:
            detail[
                "core_skipped"
            ] = True

        conflict_details.append(
            detail
        )

    nonterminal_detected = [
        row
        for row in detected_dead
        if row["remaining"] > 0
    ]

    elapsed = (
        time.perf_counter()
        - t0
    )

    detection_recall = (
        len(
            nonterminal_detected
        )
        / len(
            nonterminal_dead
        )
        if nonterminal_dead
        else None
    )

    print(
        f"{name:<18}: "
        f"Qnodes={len(rows):3d} "
        f"solutions={truth['solution_count']:2d} "
        f"dead={len(nonterminal_dead):3d} "
        f"GAC={len(gac_conflicts):2d} "
        f"detDead={len(nonterminal_detected):2d} "
        f"recall="
        + (
            f"{detection_recall:.1%}"
            if detection_recall
            is not None
            else "N/A"
        )
        + f" wall={elapsed:.3f}s"
    )

    for detail in (
        conflict_details
    ):
        core_text = (
            "/".join(
                detail.get(
                    "selected_core",
                    [],
                )
            )
            or "SKIP"
        )

        print(
            "    conflict:",
            " ".join(
                detail[
                    "prefix_original"
                ]
            )
            or "(root)",
            f"rem={detail['remaining']} "
            f"Q={detail['q_lb']} "
            f"core={core_text}",
        )

    return {
        "name": name,
        "axis": axis,
        "depth": int(depth),
        "scramble_original": list(
            scramble_original
        ),
        "q_frontier_nodes": len(
            rows
        ),
        "q_solution_count": int(
            truth[
                "solution_count"
            ]
        ),
        "dead_all_count": len(
            dead_rows
        ),
        "dead_nonterminal_count": len(
            nonterminal_dead
        ),
        "gac_conflict_count": len(
            gac_conflicts
        ),
        "gac_detected_dead_count": len(
            detected_dead
        ),
        "gac_detected_nonterminal_dead_count": len(
            nonterminal_detected
        ),
        "gac_false_positive_count": len(
            false_positives
        ),
        "nonterminal_dead_detection_recall": (
            detection_recall
        ),
        "conflicts": (
            conflict_details
        ),
        "wall": elapsed,
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
        default=3,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=20260924,
    )
    ap.add_argument(
        "--max-generation-tries",
        type=int,
        default=30000,
    )
    ap.add_argument(
        "--max-exact-core-size",
        type=int,
        default=4,
    )
    ap.add_argument(
        "--max-core-extractions",
        type=int,
        default=5,
    )
    ap.add_argument(
        "--cache",
        default=(
            "reports/pdcc_cache/"
            "twist_skeleton_tables_v1.pkl"
        ),
    )
    ap.add_argument(
        "--output",
        default=(
            "reports/v37/"
            "broad_gac_opportunity_survey_v37_79.json"
        ),
    )

    args = ap.parse_args()

    depths = tuple(
        int(x)
        for x in args.depths
    )

    print(
        "# CubeLab v37.79 - "
        "BROAD OPPORTUNITY-NORMALIZED GAC SURVEY"
    )
    print(
        "fresh depths         :",
        list(
            depths
        ),
    )
    print(
        "fresh count/depth    :",
        int(
            args.count_per_depth
        ),
    )
    print(
        "ground truth         :",
        "exact Q-prefix LIVE/DEAD descendant classification",
    )
    print(
        "GAC metric           :",
        "detected dead / all nonterminal dead opportunities",
    )
    print(
        "core extraction cap  :",
        int(
            args.max_core_extractions
        ),
    )
    print(
        "production changes   :",
        "NONE",
    )
    print()

    t0 = time.perf_counter()

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

    full_gac = (
        v64.RegularSuperposition(
            dr_targets
        )
    )

    subset_gac = (
        v65.SubsetGAC(
            dr_targets
        )
    )

    core_budget = {
        "remaining": int(
            args.max_core_extractions
        )
    }

    m11 = next(
        c
        for c in v57.CONTROLS
        if c[
            "name"
        ] == "M11-01"
    )

    print()

    total_t0 = (
        time.perf_counter()
    )

    # Historical calibration.
    historical = analyze_control(
        name="HIST-M11-L7",
        scramble_original=tuple(
            m11[
                "scramble"
            ]
        ),
        axis=m11["axis"],
        depth=7,
        qtables=qtables,
        full_gac=full_gac,
        subset_gac=subset_gac,
        max_exact_core_size=int(
            args.max_exact_core_size
        ),
        core_budget_state=(
            core_budget
        ),
    )

    rng = random.Random(
        int(
            args.seed
        )
    )

    fresh_controls = []
    fresh_rows = []

    print()

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

            fresh_controls.append(
                control
            )

            row = analyze_control(
                name=(
                    f"FRESH-L{depth}-"
                    f"{idx+1}"
                ),
                scramble_original=tuple(
                    control[
                        "scramble"
                    ]
                ),
                axis="UD",
                depth=depth,
                qtables=qtables,
                full_gac=full_gac,
                subset_gac=subset_gac,
                max_exact_core_size=int(
                    args.max_exact_core_size
                ),
                core_budget_state=(
                    core_budget
                ),
            )

            fresh_rows.append(
                row
            )

    total_wall = (
        time.perf_counter()
        - total_t0
    )

    # Historical positive-control validation.
    hist_cores = {
        frozenset(
            c.get(
                "selected_core",
                [],
            )
        )
        for c in historical[
            "conflicts"
        ]
        if c.get(
            "selected_core"
        )
    }

    hist_ok = (
        HIST_CORE
        in hist_cores
    )

    fresh_dead = sum(
        row[
            "dead_nonterminal_count"
        ]
        for row in (
            fresh_rows
        )
    )

    fresh_detected = sum(
        row[
            "gac_detected_nonterminal_dead_count"
        ]
        for row in (
            fresh_rows
        )
    )

    fresh_conflicts = sum(
        row[
            "gac_conflict_count"
        ]
        for row in (
            fresh_rows
        )
    )

    controls_with_dead = sum(
        row[
            "dead_nonterminal_count"
        ] > 0
        for row in (
            fresh_rows
        )
    )

    controls_with_gac = sum(
        row[
            "gac_conflict_count"
        ] > 0
        for row in (
            fresh_rows
        )
    )

    aggregate_recall = (
        fresh_detected
        / fresh_dead
        if fresh_dead
        else None
    )

    core_counter = Counter()

    for row in fresh_rows:
        for conflict in row[
            "conflicts"
        ]:
            core = conflict.get(
                "selected_core"
            )
            if core:
                core_counter[
                    tuple(
                        sorted(
                            core
                        )
                    )
                ] += 1

    fresh_low_order = sum(
        count
        for core, count in (
            core_counter.items()
        )
        if len(core) <= 4
    )

    print()
    print(
        "# AGGREGATE FRESH"
    )
    print(
        "fresh controls       :",
        len(
            fresh_rows
        ),
    )
    print(
        "controls with dead   :",
        f"{controls_with_dead}/"
        f"{len(fresh_rows)}",
    )
    print(
        "controls with GAC    :",
        f"{controls_with_gac}/"
        f"{len(fresh_rows)}",
    )
    print(
        "nonterminal dead opp :",
        fresh_dead,
    )
    print(
        "GAC detected dead    :",
        fresh_detected,
    )
    print(
        "GAC detection recall :",
        (
            f"{aggregate_recall:.1%}"
            if aggregate_recall
            is not None
            else "N/A"
        ),
    )
    print(
        "fresh GAC conflicts  :",
        fresh_conflicts,
    )

    if core_counter:
        print(
            "fresh core families :"
        )
        for core, count in (
            core_counter.most_common()
        ):
            print(
                "    ",
                "/".join(
                    core
                ),
                "x",
                count,
            )

    print()
    print(
        "# DECISION"
    )

    if not hist_ok:
        decision = (
            "BROAD_GAC_SURVEY_HISTORICAL_CONTROL_FAIL"
        )
        note = (
            "The broad survey failed to reproduce the known historical "
            "UR/BL/BR core. Fix the survey before interpreting fresh statistics."
        )

    elif fresh_dead == 0:
        decision = (
            "BROAD_GAC_SURVEY_INCONCLUSIVE_NO_DEAD_OPPORTUNITY"
        )
        note = (
            "The fresh exact corpus contains no nonterminal Q-surviving dead "
            "prefixes, so GAC had no independent pruning opportunity. The corpus "
            "is too Q-tight to judge superposition generality."
        )

    elif controls_with_gac >= 2:
        decision = (
            "BROAD_GAC_SIGNAL_REPLICATES_ACROSS_SCRAMBLES"
        )
        note = (
            "Q-surviving full-GAC conflicts appear in multiple independent fresh "
            "scrambles. The regular-superposition signal is not specific to the "
            "historical M11 scramble. Inspect the recovered core families to "
            "decide whether low-order dynamic factor compilation is viable."
        )

    elif fresh_detected > 0:
        decision = (
            "BROAD_GAC_SIGNAL_RARE_BUT_REPLICATED_ONCE"
        )
        note = (
            "A fresh Q-surviving dead prefix is detected by full GAC, so the "
            "historical phenomenon is not unique, but incidence is low in this "
            "pilot. Expand the corpus before engineering reusable factors."
        )

    else:
        decision = (
            "BROAD_GAC_NO_DETECTION_DESPITE_DEAD_OPPORTUNITIES"
        )
        note = (
            "The fresh corpus contains genuine Q-surviving dead prefixes, but "
            "full 20-piece GAC detects none of them. The current arc-consistency "
            "relaxation is not a general operational pruning signal; keep the "
            "historical result as structural evidence only."
        )

    print(
        decision
    )
    print(
        note
    )
    print(
        "historical core OK   :",
        hist_ok,
    )
    print(
        "full GAC cache       :",
        len(
            full_gac.cache
        ),
    )
    print(
        "subset GAC cache     :",
        len(
            subset_gac.cache
        ),
    )
    print(
        "total wall           :",
        f"{total_wall:.3f}s",
    )

    payload = {
        "version": "v37.79",
        "mode": (
            "BROAD_OPPORTUNITY_NORMALIZED_GAC_SURVEY"
        ),
        "depths": list(
            depths
        ),
        "count_per_depth": int(
            args.count_per_depth
        ),
        "seed": int(
            args.seed
        ),
        "historical": historical,
        "historical_core_recovered": (
            hist_ok
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
        "fresh_results": (
            fresh_rows
        ),
        "aggregate": {
            "fresh_control_count": len(
                fresh_rows
            ),
            "controls_with_dead_opportunity": (
                controls_with_dead
            ),
            "controls_with_gac_conflict": (
                controls_with_gac
            ),
            "nonterminal_dead_opportunities": (
                fresh_dead
            ),
            "gac_detected_nonterminal_dead": (
                fresh_detected
            ),
            "gac_detection_recall": (
                aggregate_recall
            ),
            "fresh_gac_conflicts": (
                fresh_conflicts
            ),
            "core_families": [
                {
                    "core": list(
                        core
                    ),
                    "count": int(
                        count
                    ),
                }
                for core, count in (
                    core_counter.most_common()
                )
            ],
            "fresh_low_order_core_count_le4": (
                fresh_low_order
            ),
        },
        "decision": (
            decision
        ),
        "note": (
            note
        ),
        "full_gac_cache_entries": (
            len(
                full_gac.cache
            )
        ),
        "subset_gac_cache_entries": (
            len(
                subset_gac.cache
            )
        ),
        "total_wall": (
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
