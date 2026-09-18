#!/usr/bin/env python3
"""
CubeLab v37.70 — REQUIREMENT-DIRECTED NONUNIFORM LENGTH WIDENING

Motivation
----------
v37.69 measured very large heterogeneity:

M11 exact-7:
    witness-minimal log10 portfolio = 41.18
    uniform slack-5 log10 portfolio = 84.64
    saving = 43.46 orders

All 20 pieces need some excursion, but by very different amounts.

We now try to infer those amounts WITHOUT using the known solution words.

Sound prefix requirement
------------------------
At a global prefix, for piece p:

    a_p = active global events already consumed
    d_p = exact active-only distance from current pose to local DR

Then any completion must have final active length at least:

    req_p = a_p + d_p

Because req_p is monotone non-decreasing along a path, a cap vector C can
soundly reject a prefix whenever:

    exists p: req_p > C_p

Widening policy
---------------
Start:
    C_p = d_p(start)   (local shortest)

Run exact-depth Q search with the cap-requirement gate.

When no complete solution is admitted:
    collect Q-surviving prefixes blocked ONLY by req_p > C_p.

Choose the blocker using:
    1. deepest prefix first
    2. tightest Q slack (remaining - phase1_lb)
    3. smallest increase in candidate-portfolio log volume
    4. smallest total cap increment

Then jump directly:

    C_p <- max(C_p, req_p(blocker))

for all pieces that need it.

This is requirement propagation, not +1 fallback.

The algorithm never sees the historical exact solution words.
Known words are used only after the run for diagnostic validation and for
measuring distance from the witness-minimal cap vector.

Controls
--------
M12 exact depth 3
M11 exact depth 7
M11 exact depth 4 UNSAT

FAST target:
    converge in a small number of widening rounds,
    recover known exact solution sets,
    produce a cap portfolio much smaller than uniform v37.67.

No GAC.
No production changes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
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
import search_pdcc_practical_axes as axes

import audit_fast_local_dr_superposition_v37_57 as v57
import audit_fast_depth_conditioned_piece_bound_v37_60 as v60
import audit_fast_layered_active_length_v37_67 as v67


MOVE_RANK = {m: i for i, m in enumerate(ts.MOVE_ORDER)}

M12_EXPECTED = {
    ("B", "L", "B"),
    ("B", "L", "B'"),
}
M11_EXPECTED = {
    tuple(x)
    for x in v60.M11_EXACT_ORIGINAL
}


def piece_names():
    names = getattr(pg, "PIECE_NAMES", None)
    if names is not None and len(names) == 20:
        return tuple(str(x) for x in names)
    pieces = getattr(ts, "PIECES", None)
    if pieces is not None and len(pieces) == 20:
        return tuple(str(x) for x in pieces)
    return tuple(f"P{i}" for i in range(20))


PIECE_NAMES = piece_names()


def inverse_face_map(fmap):
    return {dst: src for src, dst in fmap.items()}


def internalize(word, axis):
    return tuple(
        axes.map_word(
            tuple(word),
            axes.AXES[axis],
        )
    )


def externalize(word, axis):
    inv = inverse_face_map(axes.AXES[axis])
    return tuple(
        axes.map_word(
            tuple(word),
            inv,
        )
    )


def q_lb(q, qtables):
    return int(
        ts.phase1_lb(
            int(q),
            qtables[3],
            qtables[4],
        )
    )


def step_state(poses, active_counts, q, move, qtables):
    mi = ts.MI[move]

    nposes = tuple(
        int(x)
        for x in pg.move_poses(
            poses,
            mi,
        )
    )

    counts = list(active_counts)
    for i, (a, b) in enumerate(zip(poses, nposes)):
        if int(a) != int(b):
            counts[i] += 1

    nq = int(
        ts.q_move(
            int(q),
            mi,
            qtables[0],
            qtables[1],
            qtables[2],
        )
    )

    return nposes, tuple(counts), nq


def requirement_vector(
    poses,
    active_counts,
    dist_tables,
):
    return tuple(
        int(active_counts[i])
        + int(dist_tables[i][int(poses[i])])
        for i in range(20)
    )


def caps_dominate(caps, req):
    return all(
        int(caps[i]) >= int(req[i])
        for i in range(20)
    )


def raise_caps(caps, req, L):
    return tuple(
        min(
            int(L),
            max(int(caps[i]), int(req[i])),
        )
        for i in range(20)
    )


def candidate_log_table(
    start_poses,
    dr_targets,
    L,
):
    table = []

    for i in range(20):
        row = []

        for cap in range(int(L) + 1):
            c = v67.candidate_count_proxy(
                i,
                int(start_poses[i]),
                dr_targets[i],
                cap,
            )
            if c <= 0:
                row.append(float("-inf"))
            else:
                row.append(math.log10(c))

        table.append(tuple(row))

    return tuple(table)


def cap_log_volume(caps, log_table):
    total = 0.0

    for i, cap in enumerate(caps):
        v = log_table[i][int(cap)]
        if not math.isfinite(v):
            return float("-inf")
        total += v

    return total


def delta_log_volume(caps, newcaps, log_table):
    before = cap_log_volume(
        caps,
        log_table,
    )
    after = cap_log_volume(
        newcaps,
        log_table,
    )
    return after - before


@dataclass
class BlockedPrefix:
    prefix: tuple[str, ...]
    poses: tuple[int, ...]
    active_counts: tuple[int, ...]
    q: int
    depth: int
    remaining: int
    qlb: int
    requirement: tuple[int, ...]
    overflow_pieces: tuple[int, ...]


@dataclass
class CapSearchResult:
    nodes: int
    q_prunes: int
    cap_prunes: int
    solutions: tuple[tuple[str, ...], ...]
    solution_counts: dict
    blocked: tuple[BlockedPrefix, ...]
    wall: float
    cut_reason: str | None


def search_under_caps(
    *,
    scramble_internal,
    depth_limit,
    caps,
    qtables,
    dist_tables,
    node_cap,
    time_cap,
):
    start = ts.engine.from_word(
        " ".join(scramble_internal)
    )
    poses0 = tuple(int(x) for x in start.poses)
    q0 = int(ts.q_of(start))

    deadline = time.perf_counter() + float(time_cap)
    t0 = time.perf_counter()

    nodes = 0
    q_prunes = 0
    cap_prunes = 0
    solutions = {}
    blocked = []
    path = []

    class Cut(Exception):
        pass

    def check_cut():
        if nodes >= int(node_cap):
            raise Cut("node_cap")
        if time.perf_counter() >= deadline:
            raise Cut("time_cap")

    def dfs(poses, counts, q, depth, last_face):
        nonlocal nodes, q_prunes, cap_prunes

        check_cut()
        rem = int(depth_limit) - depth
        h = q_lb(q, qtables)

        if h > rem:
            q_prunes += 1
            return

        req = requirement_vector(
            poses,
            counts,
            dist_tables,
        )

        if not caps_dominate(caps, req):
            cap_prunes += 1
            overflow = tuple(
                i
                for i in range(20)
                if int(req[i]) > int(caps[i])
            )
            blocked.append(
                BlockedPrefix(
                    prefix=tuple(path),
                    poses=tuple(poses),
                    active_counts=tuple(counts),
                    q=int(q),
                    depth=int(depth),
                    remaining=int(rem),
                    qlb=int(h),
                    requirement=tuple(req),
                    overflow_pieces=overflow,
                )
            )
            return

        nodes += 1

        if rem == 0:
            if int(q) == int(ts.GOAL_Q):
                word = tuple(path)
                state = start.apply_word(word)

                if not ts.is_dr(state):
                    raise RuntimeError(
                        "q-goal / explicit DR mismatch"
                    )

                solutions[word] = tuple(counts)
            return

        children = []

        for move in ts.MOVE_ORDER:
            if last_face is not None and move[0] == last_face:
                continue

            nposes, ncounts, nq = step_state(
                poses,
                counts,
                q,
                move,
                qtables,
            )

            h2 = q_lb(nq, qtables)
            if h2 > rem - 1:
                q_prunes += 1
                continue

            nreq = requirement_vector(
                nposes,
                ncounts,
                dist_tables,
            )

            overflow_count = sum(
                int(nreq[i]) > int(caps[i])
                for i in range(20)
            )
            overflow_sum = sum(
                max(
                    0,
                    int(nreq[i]) - int(caps[i]),
                )
                for i in range(20)
            )

            children.append((
                h2,
                overflow_count,
                overflow_sum,
                MOVE_RANK[move],
                move,
                nposes,
                ncounts,
                nq,
            ))

        children.sort()

        for (
            _h,
            _oc,
            _os,
            _rank,
            move,
            nposes,
            ncounts,
            nq,
        ) in children:
            path.append(move)
            dfs(
                nposes,
                ncounts,
                nq,
                depth + 1,
                move[0],
            )
            path.pop()

    try:
        dfs(
            poses0,
            (0,) * 20,
            q0,
            0,
            None,
        )
        cut = None
    except Cut as exc:
        cut = str(exc)

    return CapSearchResult(
        nodes=nodes,
        q_prunes=q_prunes,
        cap_prunes=cap_prunes,
        solutions=tuple(sorted(solutions)),
        solution_counts={
            word: solutions[word]
            for word in sorted(solutions)
        },
        blocked=tuple(blocked),
        wall=time.perf_counter() - t0,
        cut_reason=cut,
    )


def choose_blocker(
    blocked,
    caps,
    L,
    log_table,
):
    rows = []

    for b in blocked:
        newcaps = raise_caps(
            caps,
            b.requirement,
            L,
        )

        if newcaps == caps:
            continue

        dlog = delta_log_volume(
            caps,
            newcaps,
            log_table,
        )
        dsum = sum(newcaps) - sum(caps)
        qslack = int(b.remaining) - int(b.qlb)

        rows.append((
            -int(b.depth),      # deepest first
            int(qslack),        # tighter Q first
            float(dlog),        # cheapest portfolio increase
            int(dsum),
            len(b.overflow_pieces),
            tuple(MOVE_RANK[m] for m in b.prefix),
            b,
            newcaps,
        ))

    if not rows:
        return None

    rows.sort(
        key=lambda x: x[:6]
    )

    *_, blocker, newcaps = rows[0]

    return blocker, newcaps


def replay_witness_caps(
    scramble_internal,
    words_original,
    axis,
    qtables,
):
    """
    Diagnostic oracle ONLY: coordinate-wise max active counts of known words.
    Never used by the widening policy.
    """
    start = ts.engine.from_word(
        " ".join(scramble_internal)
    )
    out = []

    for word_original in words_original:
        word = internalize(
            word_original,
            axis,
        )
        poses = tuple(
            int(x)
            for x in start.poses
        )
        counts = [0] * 20

        for move in word:
            nposes = tuple(
                int(x)
                for x in pg.move_poses(
                    poses,
                    ts.MI[move],
                )
            )
            for i, (a, b) in enumerate(
                zip(poses, nposes)
            ):
                if int(a) != int(b):
                    counts[i] += 1
            poses = nposes

        out.append(tuple(counts))

    return tuple(
        max(row[i] for row in out)
        for i in range(20)
    )


def run_widening(
    *,
    name,
    control,
    L,
    expected_original,
    qtables,
    dr_targets,
    dist_tables,
    node_cap,
    time_cap,
    max_rounds,
):
    axis = control["axis"]
    scramble_original = tuple(
        control["scramble"]
    )
    scramble_internal = internalize(
        scramble_original,
        axis,
    )

    start = ts.engine.from_word(
        " ".join(scramble_internal)
    )
    start_poses = tuple(
        int(x)
        for x in start.poses
    )

    dmin = tuple(
        int(dist_tables[i][start_poses[i]])
        for i in range(20)
    )
    caps = dmin

    log_table = candidate_log_table(
        start_poses,
        dr_targets,
        L,
    )

    expected = set(expected_original)

    witness_caps = replay_witness_caps(
        scramble_internal,
        expected_original,
        axis,
        qtables,
    )

    witness_log = cap_log_volume(
        witness_caps,
        log_table,
    )

    uniform_slack = (
        3 if int(L) == 3
        else 5 if int(L) == 7
        else int(L)
    )
    uniform_caps = tuple(
        min(
            int(L),
            int(dmin[i]) + int(uniform_slack),
        )
        for i in range(20)
    )
    uniform_log = cap_log_volume(
        uniform_caps,
        log_table,
    )

    print(
        f"## {name} axis={axis} exactDepth={L}"
    )
    print(
        "initial caps         :",
        " ".join(
            f"{PIECE_NAMES[i]}:{caps[i]}"
            for i in range(20)
        ),
    )
    print(
        "initial sum / 8L     :",
        f"{sum(caps)} / {8*L}",
    )
    print(
        "witness/uniform log10:",
        f"{witness_log:.2f} / {uniform_log:.2f}",
    )
    print()

    rounds = []
    first_sat_round = None
    full_recall_round = None
    final_search = None

    for rnd in range(int(max_rounds) + 1):
        res = search_under_caps(
            scramble_internal=scramble_internal,
            depth_limit=L,
            caps=caps,
            qtables=qtables,
            dist_tables=dist_tables,
            node_cap=node_cap,
            time_cap=time_cap,
        )
        final_search = res

        external = {
            externalize(
                word,
                axis,
            )
            for word in res.solutions
        }

        recall = (
            1.0
            if not expected
            else len(external & expected) / len(expected)
        )

        clog = cap_log_volume(
            caps,
            log_table,
        )

        print(
            f"round {rnd:2d}: "
            f"nodes={res.nodes:3d} "
            f"blocked={len(res.blocked):3d} "
            f"capSum={sum(caps):3d}/{8*L} "
            f"log10D={clog:.2f} "
            f"solutions={len(external)} "
            f"recall={recall:.1%} "
            f"wall={res.wall:.4f}s"
        )

        if external and first_sat_round is None:
            first_sat_round = rnd

        if recall == 1.0 and res.cut_reason is None:
            full_recall_round = rnd
            rounds.append({
                "round": rnd,
                "caps": list(caps),
                "cap_sum": sum(caps),
                "log10_candidate_volume": clog,
                "nodes": res.nodes,
                "blocked_count": len(res.blocked),
                "solutions_original": [
                    list(x)
                    for x in sorted(external)
                ],
                "recall": recall,
                "selected_blocker": None,
            })
            break

        choice = choose_blocker(
            res.blocked,
            caps,
            L,
            log_table,
        )

        if choice is None:
            rounds.append({
                "round": rnd,
                "caps": list(caps),
                "cap_sum": sum(caps),
                "log10_candidate_volume": clog,
                "nodes": res.nodes,
                "blocked_count": len(res.blocked),
                "solutions_original": [
                    list(x)
                    for x in sorted(external)
                ],
                "recall": recall,
                "selected_blocker": None,
            })
            print(
                "    no expandable Q-blocker remains"
            )
            break

        blocker, newcaps = choice

        prefix_original = externalize(
            blocker.prefix,
            axis,
        )

        changed = [
            i
            for i in range(20)
            if int(newcaps[i]) > int(caps[i])
        ]

        dlog = delta_log_volume(
            caps,
            newcaps,
            log_table,
        )

        print(
            "    choose prefix     :",
            " ".join(prefix_original)
            if prefix_original
            else "(root)",
            f"depth={blocker.depth} "
            f"rem={blocker.remaining} "
            f"Q={blocker.qlb}"
        )
        print(
            "    requirement jump :",
            " ".join(
                f"{PIECE_NAMES[i]} "
                f"{caps[i]}->{newcaps[i]}"
                for i in changed
            ),
        )
        print(
            "    delta cap/log10D  :",
            f"+{sum(newcaps)-sum(caps)} / +{dlog:.2f}",
        )

        rounds.append({
            "round": rnd,
            "caps": list(caps),
            "cap_sum": sum(caps),
            "log10_candidate_volume": clog,
            "nodes": res.nodes,
            "blocked_count": len(res.blocked),
            "solutions_original": [
                list(x)
                for x in sorted(external)
            ],
            "recall": recall,
            "selected_blocker": {
                "prefix_internal": list(
                    blocker.prefix
                ),
                "prefix_original": list(
                    prefix_original
                ),
                "depth": blocker.depth,
                "remaining": blocker.remaining,
                "q_lb": blocker.qlb,
                "q_slack": (
                    blocker.remaining
                    - blocker.qlb
                ),
                "requirement": list(
                    blocker.requirement
                ),
                "overflow_pieces": [
                    PIECE_NAMES[i]
                    for i in blocker.overflow_pieces
                ],
                "changed_pieces": [
                    PIECE_NAMES[i]
                    for i in changed
                ],
                "new_caps": list(
                    newcaps
                ),
                "delta_cap_sum": (
                    sum(newcaps) - sum(caps)
                ),
                "delta_log10_volume": dlog,
            },
        })

        caps = newcaps

    final_external = {
        externalize(
            word,
            axis,
        )
        for word in (
            final_search.solutions
            if final_search is not None
            else ()
        )
    }

    final_log = cap_log_volume(
        caps,
        log_table,
    )

    excess_vs_witness = tuple(
        int(caps[i]) - int(witness_caps[i])
        for i in range(20)
    )

    exact_match_caps = (
        tuple(caps)
        == tuple(witness_caps)
    )

    print()
    print(
        "final caps           :",
        " ".join(
            f"{PIECE_NAMES[i]}:{caps[i]}"
            for i in range(20)
        ),
    )
    print(
        "witness caps         :",
        " ".join(
            f"{PIECE_NAMES[i]}:{witness_caps[i]}"
            for i in range(20)
        ),
    )
    print(
        "cap sum final/witness:",
        f"{sum(caps)} / {sum(witness_caps)}",
    )
    print(
        "log10D final/witness/uniform:",
        f"{final_log:.2f} / {witness_log:.2f} / {uniform_log:.2f}",
    )
    print(
        "exact witness cap match:",
        exact_match_caps,
    )
    print(
        "excess vs witness    :",
        " ".join(
            f"{PIECE_NAMES[i]}:{x:+d}"
            for i, x in enumerate(excess_vs_witness)
            if x != 0
        ) or "-",
    )
    print(
        "first SAT/full recall:",
        f"{first_sat_round} / {full_recall_round}",
    )
    print()

    return {
        "name": name,
        "axis": axis,
        "depth": L,
        "initial_caps": list(dmin),
        "final_caps": list(caps),
        "witness_caps_diagnostic": list(
            witness_caps
        ),
        "uniform_caps_reference": list(
            uniform_caps
        ),
        "final_cap_sum": sum(caps),
        "witness_cap_sum": sum(
            witness_caps
        ),
        "final_log10_candidate_volume": (
            final_log
        ),
        "witness_log10_candidate_volume": (
            witness_log
        ),
        "uniform_log10_candidate_volume": (
            uniform_log
        ),
        "orders_saved_vs_uniform": (
            uniform_log - final_log
        ),
        "orders_above_witness_minimum": (
            final_log - witness_log
        ),
        "exact_witness_cap_match": (
            exact_match_caps
        ),
        "excess_vs_witness": list(
            excess_vs_witness
        ),
        "first_sat_round": (
            first_sat_round
        ),
        "full_recall_round": (
            full_recall_round
        ),
        "solution_parity_expected": (
            final_external == expected
        ),
        "rounds": rounds,
    }


def main():
    ap = argparse.ArgumentParser()
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
        default=1_000_000,
    )
    ap.add_argument(
        "--time-cap",
        type=float,
        default=1.0,
    )
    ap.add_argument(
        "--max-rounds",
        type=int,
        default=20,
    )
    ap.add_argument(
        "--output",
        default=(
            "reports/v37/"
            "requirement_directed_nonuniform_v37_70.json"
        ),
    )
    args = ap.parse_args()

    print(
        "# CubeLab v37.70 - "
        "REQUIREMENT-DIRECTED NONUNIFORM LENGTH WIDENING"
    )
    print(
        "initial caps         :",
        "local shortest active lengths",
    )
    print(
        "prefix requirement   :",
        "active_used + local_DR_distance",
    )
    print(
        "widening jump        :",
        "component-wise to chosen blocked prefix requirement",
    )
    print(
        "selection            :",
        "deepest -> Q-tight -> min portfolio growth",
    )
    print(
        "known words in policy:",
        "NO",
    )
    print(
        "GAC                  :",
        "NONE",
    )
    print(
        "production changes   :",
        "NONE",
    )
    print()

    t0 = time.perf_counter()
    qtables, loaded = ts.cache_load_or_build(
        Path(args.cache)
    )
    print(
        "phase tables         :",
        "cache" if loaded else "built",
    )
    print(
        "table wall           :",
        f"{time.perf_counter()-t0:.3f}s",
    )

    dr_targets = v57.local_dr_pose_sets()
    dist_tables = (
        v60.build_piece_distance_tables(
            dr_targets
        )
    )
    print()

    m12 = next(
        c
        for c in v57.CONTROLS
        if c["name"] == "M12-02"
    )
    m11 = next(
        c
        for c in v57.CONTROLS
        if c["name"] == "M11-01"
    )

    total_t0 = time.perf_counter()

    m12_row = run_widening(
        name="M12-02",
        control=m12,
        L=3,
        expected_original=M12_EXPECTED,
        qtables=qtables,
        dr_targets=dr_targets,
        dist_tables=dist_tables,
        node_cap=int(args.node_cap),
        time_cap=float(args.time_cap),
        max_rounds=int(args.max_rounds),
    )

    m11_row = run_widening(
        name="M11-01-EXACT7",
        control=m11,
        L=7,
        expected_original=M11_EXPECTED,
        qtables=qtables,
        dr_targets=dr_targets,
        dist_tables=dist_tables,
        node_cap=int(args.node_cap),
        time_cap=float(args.time_cap),
        max_rounds=int(args.max_rounds),
    )

    # M11 exact depth 4 UNSAT:
    # use shortest caps; Q should reject root before widening is needed.
    sw4 = internalize(
        tuple(m11["scramble"]),
        m11["axis"],
    )
    start4 = ts.engine.from_word(
        " ".join(sw4)
    )
    poses4 = tuple(
        int(x)
        for x in start4.poses
    )
    caps4 = tuple(
        int(
            dist_tables[i][poses4[i]]
        )
        for i in range(20)
    )
    unsat4 = search_under_caps(
        scramble_internal=sw4,
        depth_limit=4,
        caps=caps4,
        qtables=qtables,
        dist_tables=dist_tables,
        node_cap=int(args.node_cap),
        time_cap=float(args.time_cap),
    )

    total_wall = (
        time.perf_counter()
        - total_t0
    )

    print("## M11-01 EXACT4 UNSAT")
    print(
        "nodes / solutions    :",
        unsat4.nodes,
        "/",
        len(unsat4.solutions),
    )
    print()

    print("# DECISION")

    controls_ok = (
        m12_row["solution_parity_expected"]
        and m11_row[
            "solution_parity_expected"
        ]
        and len(unsat4.solutions) == 0
    )

    substantial_saving = (
        m11_row[
            "orders_saved_vs_uniform"
        ] >= 6.0
    )

    near_witness = (
        m11_row[
            "orders_above_witness_minimum"
        ] <= 6.0
    )

    if (
        controls_ok
        and substantial_saving
        and near_witness
    ):
        decision = (
            "REQUIREMENT_DIRECTED_NONUNIFORM_STRONG_PASS"
        )
        note = (
            "Prefix requirement propagation recovers the exact controls "
            "without witness guidance and reaches a candidate portfolio close "
            "to the measured witness-minimal vector while saving many orders "
            "versus uniform widening."
        )
    elif (
        controls_ok
        and substantial_saving
    ):
        decision = (
            "REQUIREMENT_DIRECTED_NONUNIFORM_PASS"
        )
        note = (
            "Requirement-directed widening recovers the exact controls and "
            "substantially reduces candidate volume versus the uniform ladder, "
            "though the inferred cap vector remains noticeably above the "
            "diagnostic witness minimum."
        )
    elif controls_ok:
        decision = (
            "NONUNIFORM_CORRECT_LOW_SAVING"
        )
        note = (
            "The requirement-directed policy is correct on the controls but "
            "does not materially improve candidate volume over uniform widening."
        )
    else:
        decision = (
            "REQUIREMENT_DIRECTED_NONUNIFORM_FAIL"
        )
        note = (
            "The online requirement-directed cap policy did not recover the "
            "known exact controls within the FAST round budget."
        )

    print(decision)
    print(note)
    print(
        "M11 saving vs uniform:",
        f"{m11_row['orders_saved_vs_uniform']:.2f} log10 orders",
    )
    print(
        "M11 above witness min:",
        f"{m11_row['orders_above_witness_minimum']:.2f} log10 orders",
    )
    print(
        "total wall           :",
        f"{total_wall:.3f}s",
    )

    payload = {
        "version": "v37.70",
        "mode": (
            "REQUIREMENT_DIRECTED_NONUNIFORM_LENGTH_WIDENING"
        ),
        "controls": [
            m12_row,
            m11_row,
        ],
        "m11_exact4_unsat": {
            "nodes": unsat4.nodes,
            "solution_count": len(
                unsat4.solutions
            ),
            "wall": unsat4.wall,
        },
        "decision": decision,
        "note": note,
        "total_wall": total_wall,
    }

    out = Path(args.output)
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
