#!/usr/bin/env python3
"""
CubeLab v37.67 — LAYERED ACTIVE-LENGTH BUDGET SEARCH

Motivation
----------
v37.66 widened one piece (+1) at a time and repeatedly ran expensive GAC.

But every HTM face turn moves exactly 8 movable cubies.

Therefore every exact global word of length L has:

    sum_p active_count_p == 8*L

If per-piece candidate active-length caps are cap_p, then a NECESSARY
condition for ANY length-L global word to fit is:

    sum_p cap_p >= 8*L

This explains v37.66:
    M12 L=3 requires cap sum >=24.
    M11 L=7 requires cap sum >=56.
    v37.66 stopped M11 at cap sum 51, so success was impossible.

Experiment
----------
Use the user's simple staged idea directly.

For piece p:
    d_p = exact shortest active-only distance to local DR

At uniform excursion layer s:

    cap_p(s) = min(L, d_p + s)

This means:
    allow ALL piece active skeleton lengths <= d_p+s

No concrete candidate words are materialized.

Before search:
    if sum cap_p < 8L:
        skip the layer completely (resource impossible)

Search:
    exact-depth Q search
    + maintain actual active-event count for each of 20 pieces
    + prune branch immediately if active_count_p > cap_p

No GAC.
No full-horizon tree parity.
No known solution injection into search.

We increase s until:
    first SAT
and continue only until:
    the historical exact solution set is fully recovered
for diagnostic validation.

Controls:
    M12-02 exact depth 3
    M11-01 exact depth 7
    M11-01 exact depth 4 UNSAT

Expected diagnostic:
    M12 first/full recall layer likely s=3
    M11 first/full recall layer likely s=5

FAST: Q frontier is tiny on these controls.
No production changes.
"""

from __future__ import annotations

import argparse
import json
import math
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
import search_pdcc_practical_axes as axes

import audit_fast_local_dr_superposition_v37_57 as v57
import audit_fast_depth_conditioned_piece_bound_v37_60 as v60


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


def q_lb(q, qtables):
    return int(
        ts.phase1_lb(
            int(q),
            qtables[3],
            qtables[4],
        )
    )


def exact_search_with_caps(
    *,
    scramble_internal,
    depth_limit,
    caps,
    qtables,
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
    solutions = []
    solution_counts = []
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

        if q_lb(q, qtables) > rem:
            q_prunes += 1
            return

        # Candidate-length budget.
        for i in range(20):
            if int(counts[i]) > int(caps[i]):
                cap_prunes += 1
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

                solutions.append(word)
                solution_counts.append(tuple(counts))
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

            # Immediate cap check before child recursion.
            if any(
                int(ncounts[i]) > int(caps[i])
                for i in range(20)
            ):
                cap_prunes += 1
                continue

            h2 = q_lb(nq, qtables)
            if h2 > rem - 1:
                q_prunes += 1
                continue

            # Prefer stronger Q then lower budget utilization.
            max_util = max(
                ncounts[i] / caps[i]
                if caps[i] > 0
                else (0.0 if ncounts[i] == 0 else 999.0)
                for i in range(20)
            )

            children.append((
                h2,
                max_util,
                sum(ncounts),
                MOVE_RANK[move],
                move,
                nposes,
                ncounts,
                nq,
            ))

        children.sort()

        for (
            _h,
            _util,
            _sum,
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

    unique = {}
    for word, counts in zip(solutions, solution_counts):
        unique[word] = counts

    return {
        "nodes": nodes,
        "q_prunes": q_prunes,
        "cap_prunes": cap_prunes,
        "solutions_internal": tuple(sorted(unique)),
        "solution_active_counts": {
            word: unique[word]
            for word in sorted(unique)
        },
        "wall": time.perf_counter() - t0,
        "cut_reason": cut,
    }


def candidate_count_proxy(
    piece_i,
    start_pose,
    targets,
    cap,
):
    """
    Number of active-only local candidate sequences of length <=cap.
    Diagnostic only. Exact move words are never stored.
    """
    targets = set(int(x) for x in targets)

    dp = [0] * 24
    dp[int(start_pose)] = 1

    total = (
        1 if int(start_pose) in targets else 0
    )

    for _depth in range(1, int(cap) + 1):
        ndp = [0] * 24

        for pose, count in enumerate(dp):
            if count == 0:
                continue

            for move in ts.MOVE_ORDER:
                mi = ts.MI[move]
                nxt = int(
                    pg.POSE_NEXT[mi][piece_i][pose]
                )

                if nxt == pose:
                    continue

                ndp[nxt] += count

        dp = ndp
        total += sum(
            dp[p]
            for p in targets
        )

    return int(total)


def volume_log10(start_poses, dr_targets, caps):
    counts = []
    logv = 0.0

    for i in range(20):
        c = candidate_count_proxy(
            i,
            int(start_poses[i]),
            dr_targets[i],
            int(caps[i]),
        )
        counts.append(c)

        if c <= 0:
            return float("-inf"), tuple(counts)

        logv += math.log10(c)

    return logv, tuple(counts)


def run_ladder(
    *,
    name,
    scramble,
    axis,
    L,
    expected_original,
    qtables,
    dr_targets,
    dist_tables,
    node_cap,
    time_cap,
):
    sw = internalize(
        scramble,
        axis,
    )
    start = ts.engine.from_word(
        " ".join(sw)
    )
    start_poses = tuple(
        int(x)
        for x in start.poses
    )

    dmin = tuple(
        int(dist_tables[i][start_poses[i]])
        for i in range(20)
    )

    active_total = 8 * int(L)

    print(
        f"## {name} axis={axis} exactDepth={L}"
    )
    print(
        "shortest lengths     :",
        " ".join(
            f"{PIECE_NAMES[i]}:{dmin[i]}"
            for i in range(20)
        ),
    )
    print(
        "shortest sum / exact active total:",
        f"{sum(dmin)} / {active_total}",
    )

    expected = set(expected_original)
    rows = []
    first_sat = None
    first_full = None

    full_caps = (int(L),) * 20
    full_log, _ = volume_log10(
        start_poses,
        dr_targets,
        full_caps,
    )

    # The first potentially meaningful uniform layer can be found without search.
    first_capacity_layer = None

    for slack in range(int(L) + 1):
        caps = tuple(
            min(int(L), int(d) + slack)
            for d in dmin
        )
        capsum = sum(caps)

        if capsum >= active_total:
            first_capacity_layer = slack
            break

    print(
        "first capacity layer :",
        first_capacity_layer,
    )
    print()

    for slack in range(int(L) + 1):
        caps = tuple(
            min(int(L), int(d) + slack)
            for d in dmin
        )
        capsum = sum(caps)

        logv, _counts = volume_log10(
            start_poses,
            dr_targets,
            caps,
        )

        if capsum < active_total:
            print(
                f"slack={slack}: "
                f"capSum={capsum}/{active_total} "
                f"log10D={logv:.2f} "
                f"SKIP(capacity)"
            )
            rows.append({
                "slack": slack,
                "caps": list(caps),
                "cap_sum": capsum,
                "active_total": active_total,
                "capacity_possible": False,
                "candidate_log10_volume": logv,
                "nodes": 0,
                "solutions_original": [],
                "recall": 0.0 if expected else 1.0,
                "cut_reason": None,
            })
            continue

        res = exact_search_with_caps(
            scramble_internal=sw,
            depth_limit=L,
            caps=caps,
            qtables=qtables,
            node_cap=node_cap,
            time_cap=time_cap,
        )

        external = {
            externalize(
                word,
                axis,
            )
            for word in res["solutions_internal"]
        }

        recall = (
            1.0
            if not expected
            else len(external & expected) / len(expected)
        )

        print(
            f"slack={slack}: "
            f"capSum={capsum}/{active_total} "
            f"log10D={logv:.2f}/{full_log:.2f} "
            f"nodes={res['nodes']:,} "
            f"qPrune={res['q_prunes']:,} "
            f"capPrune={res['cap_prunes']:,} "
            f"solutions={len(external)} "
            f"recall={recall:.1%} "
            f"wall={res['wall']:.4f}s "
            f"cut={res['cut_reason']}"
        )

        for word in sorted(external):
            counts = res["solution_active_counts"][
                internalize(
                    word,
                    axis,
                )
            ]
            excursion = tuple(
                int(counts[i]) - int(dmin[i])
                for i in range(20)
            )
            print(
                "    ",
                " ".join(word),
                f"activeSum={sum(counts)} "
                f"maxExcursion={max(excursion)}",
            )

        if external and first_sat is None:
            first_sat = slack

        if recall == 1.0 and first_full is None:
            first_full = slack

        rows.append({
            "slack": slack,
            "caps": list(caps),
            "cap_sum": capsum,
            "active_total": active_total,
            "capacity_possible": True,
            "candidate_log10_volume": logv,
            "candidate_log10_volume_full": full_log,
            "nodes": res["nodes"],
            "q_prunes": res["q_prunes"],
            "cap_prunes": res["cap_prunes"],
            "solutions_original": [
                list(x)
                for x in sorted(external)
            ],
            "recall": recall,
            "wall": res["wall"],
            "cut_reason": res["cut_reason"],
        })

        # FAST diagnostic: once the complete known exact set is recovered,
        # higher slack layers add no information for this question.
        if recall == 1.0 and res["cut_reason"] is None:
            break

    print(
        "first SAT / full recall:",
        f"{first_sat} / {first_full}",
    )
    print()

    return {
        "name": name,
        "axis": axis,
        "depth": L,
        "shortest_lengths": list(dmin),
        "shortest_sum": sum(dmin),
        "exact_active_total": active_total,
        "first_capacity_layer": first_capacity_layer,
        "first_sat_slack": first_sat,
        "first_full_recall_slack": first_full,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache",
        default="reports/pdcc_cache/twist_skeleton_tables_v1.pkl",
    )
    ap.add_argument(
        "--node-cap",
        type=int,
        default=1_000_000,
    )
    ap.add_argument(
        "--time-cap",
        type=float,
        default=2.0,
    )
    ap.add_argument(
        "--output",
        default="reports/v37/layered_active_length_budget_v37_67.json",
    )
    args = ap.parse_args()

    print("# CubeLab v37.67 - LAYERED ACTIVE-LENGTH BUDGET SEARCH")
    print("candidate policy     : cap_p=min(L,dmin_p+slack)")
    print("slack schedule       : 0 -> 1 -> ... -> L")
    print("capacity gate        : sum caps >= 8L")
    print("search               : Q + per-piece active-count budget")
    print("GAC                  : NONE")
    print("full-tree parity     : NOT REQUIRED")
    print("production changes   : NONE")
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
    dist_tables = v60.build_piece_distance_tables(
        dr_targets
    )
    print()

    m12 = next(
        c for c in v57.CONTROLS
        if c["name"] == "M12-02"
    )
    m11 = next(
        c for c in v57.CONTROLS
        if c["name"] == "M11-01"
    )

    total_t0 = time.perf_counter()

    m12_row = run_ladder(
        name="M12-02",
        scramble=tuple(m12["scramble"]),
        axis=m12["axis"],
        L=3,
        expected_original=M12_EXPECTED,
        qtables=qtables,
        dr_targets=dr_targets,
        dist_tables=dist_tables,
        node_cap=int(args.node_cap),
        time_cap=float(args.time_cap),
    )

    m11_row = run_ladder(
        name="M11-01-EXACT7",
        scramble=tuple(m11["scramble"]),
        axis=m11["axis"],
        L=7,
        expected_original=M11_EXPECTED,
        qtables=qtables,
        dr_targets=dr_targets,
        dist_tables=dist_tables,
        node_cap=int(args.node_cap),
        time_cap=float(args.time_cap),
    )

    # Exact depth 4 UNSAT check using full cap; Q should kill root.
    sw4 = internalize(
        tuple(m11["scramble"]),
        m11["axis"],
    )
    unsat4 = exact_search_with_caps(
        scramble_internal=sw4,
        depth_limit=4,
        caps=(4,) * 20,
        qtables=qtables,
        node_cap=int(args.node_cap),
        time_cap=float(args.time_cap),
    )

    print("## M11-01 EXACT4 UNSAT")
    print(
        "nodes / solutions    :",
        unsat4["nodes"],
        "/",
        len(unsat4["solutions_internal"]),
    )
    print()

    total_wall = time.perf_counter() - total_t0

    print("# DECISION")

    controls_ok = (
        m12_row["first_full_recall_slack"] is not None
        and m11_row["first_full_recall_slack"] is not None
        and len(unsat4["solutions_internal"]) == 0
    )

    m11_saving = (
        m11_row["first_full_recall_slack"] is not None
        and m11_row["first_full_recall_slack"] < 7
    )

    if controls_ok and m11_saving:
        decision = "LAYERED_ACTIVE_LENGTH_PASS"
        note = (
            "Uniform staged active-length widening recovers the complete exact "
            "controls without opening M11 all the way to the full global horizon. "
            "The 8L capacity gate skips provably impossible early layers."
        )
    elif controls_ok:
        decision = "LAYERED_ACTIVE_LENGTH_CORRECT_NO_HORIZON_SAVING"
        note = (
            "The staged length ladder is correct, but these controls require the "
            "full horizon before complete recall."
        )
    else:
        decision = "LAYERED_ACTIVE_LENGTH_FAIL"
        note = (
            "The staged cap ladder failed to recover a known exact control. "
            "Inspect active-count bookkeeping before further widening work."
        )

    print(decision)
    print(note)
    print(
        "M12 full-recall slack:",
        m12_row["first_full_recall_slack"],
    )
    print(
        "M11 full-recall slack:",
        m11_row["first_full_recall_slack"],
    )
    print(
        "total wall           :",
        f"{total_wall:.3f}s",
    )

    payload = {
        "version": "v37.67",
        "mode": "LAYERED_ACTIVE_LENGTH_BUDGET_SEARCH",
        "controls": [
            m12_row,
            m11_row,
        ],
        "m11_exact4_unsat": {
            "nodes": unsat4["nodes"],
            "solution_count": len(
                unsat4["solutions_internal"]
            ),
            "wall": unsat4["wall"],
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

    print("JSON                 :", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
