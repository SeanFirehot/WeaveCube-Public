#!/usr/bin/env python3
"""
CubeLab v37.72 — PREFIX REQUIREMENT / CAPACITY-COMPLETION BEST-FIRST

Why
---
v37.71.1 best-first cap-lattice search spent 5s expanding 1,175 different
20-dimensional cap vectors but had not reached the necessary M11 active
capacity sum 56.

Invert the search.

A concrete Q-search PREFIX already determines a sound per-piece lower bound:

    r_p = active_used_p + local_DR_distance_p

Any exact length-L DR completion has final active-count vector c satisfying:

    c_p >= r_p
    0 <= c_p <= L
    sum_p c_p = 8L

For each piece p and cap c, let:

    cost_p(c) = log10(number of active-only local-DR candidates of length <= c
                      from the ORIGINAL scrambled pose)

Then compute at every prefix the exact optimistic capacity-completion bound:

    Hcap(r) =
        min sum_p cost_p(c_p)
        s.t.
            c_p >= r_p
            c_p <= L
            sum c_p = 8L

This is a tiny 20 x (8L) dynamic program.

Properties
----------
* Every exact descendant solution's final active-count vector is a feasible
  completion in this DP.
* Therefore Hcap is an admissible lower bound on its final candidate portfolio.
* r_p is monotone non-decreasing along a path.
* The feasible completion set only shrinks as r grows, so Hcap is monotone.
* At a DR goal, local distances are zero and sum active counts = 8L, so the
  DP completion collapses to the exact solution active-count vector.

Therefore best-first search over Q-surviving prefixes ordered by Hcap returns
a minimum-candidate-portfolio exact solution WITHOUT enumerating the cap
lattice and WITHOUT using known exact words.

Known words are used only after search for validation.

Controls:
    M12 exact depth 3
    M11 exact depth 7
    M11 exact depth 4 UNSAT

No GAC.
No production changes.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
import time
from functools import lru_cache
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
import audit_fast_depth_conditioned_piece_bound_v37_60 as v60
import audit_fast_layered_active_length_v37_67 as v67
import audit_fast_requirement_nonuniform_v37_70 as v70


PIECE_NAMES = v70.PIECE_NAMES
MOVE_RANK = v70.MOVE_RANK

M12_EXPECTED = {
    ("B", "L", "B"),
    ("B", "L", "B'"),
}
M11_EXPECTED = {
    tuple(x)
    for x in v60.M11_EXACT_ORIGINAL
}


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
    for i, (a, b) in enumerate(
        zip(poses, nposes)
    ):
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


class CapacityCompletion:
    def __init__(
        self,
        start_poses,
        dr_targets,
        L,
    ):
        self.L = int(L)
        self.target_sum = 8 * int(L)

        raw = v70.candidate_log_table(
            start_poses,
            dr_targets,
            L,
        )
        self.cost_table = tuple(
            tuple(float(x) for x in row)
            for row in raw
        )

        self.cache = {}

    def solve(self, requirement):
        req = tuple(
            int(x)
            for x in requirement
        )

        cached = self.cache.get(req)
        if cached is not None:
            return cached

        L = self.L
        target = self.target_sum

        if any(x < 0 or x > L for x in req):
            result = None
            self.cache[req] = result
            return result

        if sum(req) > target:
            result = None
            self.cache[req] = result
            return result

        # DP: sum -> (cost, cap_prefix)
        dp = {
            0: (
                0.0,
                (),
            )
        }

        for i in range(20):
            ndp = {}

            min_c = int(req[i])

            for used_sum, (
                base_cost,
                prefix,
            ) in dp.items():
                remaining_pieces = (
                    20 - i - 1
                )

                for c in range(
                    min_c,
                    L + 1,
                ):
                    c_cost = (
                        self.cost_table[i][c]
                    )

                    if not math.isfinite(
                        c_cost
                    ):
                        continue

                    ns = int(used_sum) + c

                    if ns > target:
                        break

                    # Capacity feasibility for remaining pieces.
                    min_rest = sum(
                        req[j]
                        for j in range(
                            i + 1,
                            20,
                        )
                    )
                    max_rest = (
                        remaining_pieces * L
                    )

                    if (
                        ns + min_rest
                        > target
                    ):
                        continue

                    if (
                        ns + max_rest
                        < target
                    ):
                        continue

                    ncost = (
                        float(base_cost)
                        + float(c_cost)
                    )
                    ncaps = (
                        prefix + (c,)
                    )

                    prev = ndp.get(ns)

                    if (
                        prev is None
                        or ncost
                        < prev[0] - 1e-12
                        or (
                            abs(
                                ncost
                                - prev[0]
                            )
                            <= 1e-12
                            and ncaps
                            < prev[1]
                        )
                    ):
                        ndp[ns] = (
                            ncost,
                            ncaps,
                        )

            dp = ndp

            if not dp:
                result = None
                self.cache[req] = result
                return result

        result = dp.get(
            target
        )
        self.cache[req] = result
        return result


def search_best_prefix(
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
):
    axis = control["axis"]

    scramble_original = tuple(
        control["scramble"]
    )
    scramble_internal = (
        v70.internalize(
            scramble_original,
            axis,
        )
    )

    start = ts.engine.from_word(
        " ".join(scramble_internal)
    )

    start_poses = tuple(
        int(x)
        for x in start.poses
    )
    q0 = int(
        ts.q_of(start)
    )

    capacity = CapacityCompletion(
        start_poses,
        dr_targets,
        L,
    )

    start_counts = (0,) * 20
    start_req = (
        v70.requirement_vector(
            start_poses,
            start_counts,
            dist_tables,
        )
    )
    root_completion = (
        capacity.solve(
            start_req
        )
    )

    if root_completion is None:
        raise RuntimeError(
            f"{name}: root has no 8L "
            "capacity completion"
        )

    root_lb_cost, root_opt_caps = (
        root_completion
    )

    witness_caps = (
        v70.replay_witness_caps(
            scramble_internal,
            expected_original,
            axis,
            qtables,
        )
    )
    witness_cost = (
        v70.cap_log_volume(
            witness_caps,
            capacity.cost_table,
        )
    )

    uniform_slack = (
        3 if int(L) == 3
        else 5 if int(L) == 7
        else int(L)
    )
    uniform_caps = tuple(
        min(
            int(L),
            int(start_req[i])
            + int(uniform_slack),
        )
        for i in range(20)
    )
    uniform_cost = (
        v70.cap_log_volume(
            uniform_caps,
            capacity.cost_table,
        )
    )

    print(
        f"## {name} axis={axis} "
        f"exactDepth={L}"
    )
    print(
        "root requirement sum :",
        f"{sum(start_req)} / {8*L}",
    )
    print(
        "root Hcap            :",
        f"{root_lb_cost:.2f}",
    )
    print(
        "root optimistic caps :",
        " ".join(
            f"{PIECE_NAMES[i]}:"
            f"{root_opt_caps[i]}"
            for i in range(20)
        ),
    )
    print(
        "witness/uniform log10:",
        f"{witness_cost:.2f} / "
        f"{uniform_cost:.2f}",
    )
    print()

    # Heap record:
    # (Hcap, -depth, qslack, seq,
    #  poses, counts, q, last_face, path,
    #  requirement, optimistic_caps)
    heap = []
    seq = 0

    root_q = q_lb(
        q0,
        qtables,
    )

    if root_q > int(L):
        return {
            "name": name,
            "axis": axis,
            "depth": L,
            "root_q_lb": root_q,
            "q_root_unsat": True,
            "found": False,
            "solutions_original": [],
            "popped_nodes": 0,
            "generated_nodes": 0,
            "wall": 0.0,
        }

    heapq.heappush(
        heap,
        (
            float(root_lb_cost),
            0,
            int(L) - root_q,
            seq,
            start_poses,
            start_counts,
            q0,
            None,
            (),
            start_req,
            root_opt_caps,
        ),
    )

    deadline = (
        time.perf_counter()
        + float(time_cap)
    )
    t0 = time.perf_counter()

    popped = 0
    generated = 1
    q_prunes = 0
    capacity_prunes = 0

    first_goal_cost = None
    goal_rows = []

    min_open_history = []

    while heap:
        if popped >= int(node_cap):
            cut_reason = "node_cap"
            break

        if time.perf_counter() >= deadline:
            cut_reason = "time_cap"
            break

        (
            hcap,
            neg_depth,
            _qslack,
            _seq,
            poses,
            counts,
            q,
            last_face,
            path,
            req,
            optimistic_caps,
        ) = heapq.heappop(heap)

        if (
            first_goal_cost is not None
            and hcap
            > first_goal_cost + 1e-12
        ):
            cut_reason = None
            break

        depth = len(path)
        rem = int(L) - depth

        popped += 1

        if (
            popped <= 20
            or popped % 25 == 0
        ):
            print(
                f"pop {popped:3d}: "
                f"depth={depth} "
                f"rem={rem} "
                f"Hcap={hcap:.2f} "
                f"reqSum={sum(req):2d}/{8*L} "
                f"Q={q_lb(q, qtables)} "
                f"open={len(heap)} "
                f"path="
                f"{' '.join(v70.externalize(path, axis)) or '(root)'}"
            )

        if rem == 0:
            if int(q) == int(ts.GOAL_Q):
                state = start.apply_word(
                    tuple(path)
                )

                if not ts.is_dr(state):
                    raise RuntimeError(
                        "q-goal / explicit DR mismatch"
                    )

                final_req = tuple(
                    int(x)
                    for x in req
                )
                final_cost = (
                    v70.cap_log_volume(
                        final_req,
                        capacity.cost_table,
                    )
                )

                if (
                    first_goal_cost
                    is None
                ):
                    first_goal_cost = (
                        final_cost
                    )
                    print()
                    print(
                        "FIRST GOAL:"
                    )

                if (
                    abs(
                        final_cost
                        - first_goal_cost
                    )
                    <= 1e-12
                ):
                    goal_rows.append({
                        "word_internal": list(
                            path
                        ),
                        "word_original": list(
                            v70.externalize(
                                path,
                                axis,
                            )
                        ),
                        "caps": list(
                            final_req
                        ),
                        "log10_candidate_volume": (
                            final_cost
                        ),
                    })

                    print(
                        "    ",
                        " ".join(
                            v70.externalize(
                                path,
                                axis,
                            )
                        ),
                        f"log10D={final_cost:.2f}"
                    )

            continue

        # Expand only Q-surviving reduced-word children.
        children = []

        for move in ts.MOVE_ORDER:
            if (
                last_face is not None
                and move[0] == last_face
            ):
                continue

            (
                nposes,
                ncounts,
                nq,
            ) = step_state(
                poses,
                counts,
                q,
                move,
                qtables,
            )

            hq = q_lb(
                nq,
                qtables,
            )

            if hq > rem - 1:
                q_prunes += 1
                continue

            nreq = (
                v70.requirement_vector(
                    nposes,
                    ncounts,
                    dist_tables,
                )
            )

            completion = (
                capacity.solve(
                    nreq
                )
            )

            if completion is None:
                capacity_prunes += 1
                continue

            (
                nhcap,
                nopt_caps,
            ) = completion

            # Once an incumbent minimum goal cost exists, this child cannot
            # improve/equal it if its admissible lower bound is larger.
            if (
                first_goal_cost
                is not None
                and nhcap
                > first_goal_cost
                + 1e-12
            ):
                capacity_prunes += 1
                continue

            children.append(
                (
                    float(nhcap),
                    -(depth + 1),
                    (rem - 1) - hq,
                    MOVE_RANK[move],
                    move,
                    nposes,
                    ncounts,
                    nq,
                    nreq,
                    nopt_caps,
                )
            )

        children.sort()

        for (
            nhcap,
            nneg_depth,
            nqslack,
            _rank,
            move,
            nposes,
            ncounts,
            nq,
            nreq,
            nopt_caps,
        ) in children:
            seq += 1
            npath = (
                tuple(path)
                + (move,)
            )

            heapq.heappush(
                heap,
                (
                    nhcap,
                    nneg_depth,
                    nqslack,
                    seq,
                    nposes,
                    ncounts,
                    nq,
                    move[0],
                    npath,
                    nreq,
                    nopt_caps,
                ),
            )
            generated += 1

    else:
        cut_reason = None

    wall = (
        time.perf_counter()
        - t0
    )

    expected = set(
        expected_original
    )
    found_words = {
        tuple(row["word_original"])
        for row in goal_rows
    }

    if goal_rows:
        goal_caps_set = {
            tuple(row["caps"])
            for row in goal_rows
        }

        # Multiple minimum-volume goal words may share the same cap vector.
        best_caps = min(
            goal_caps_set
        )
        best_cost = float(
            first_goal_cost
        )
        exact_witness_cap_match = (
            len(goal_caps_set) == 1
            and next(
                iter(goal_caps_set)
            ) == tuple(witness_caps)
        )
    else:
        goal_caps_set = set()
        best_caps = None
        best_cost = None
        exact_witness_cap_match = False

    qref = (
        v67.exact_search_with_caps(
            scramble_internal=scramble_internal,
            depth_limit=L,
            caps=(int(L),) * 20,
            qtables=qtables,
            node_cap=int(node_cap),
            time_cap=float(time_cap),
        )
    )

    print()
    print(
        "BEST-FIRST PREFIX RESULT"
    )
    print(
        "minimum goal cost    :",
        (
            f"{best_cost:.2f}"
            if best_cost is not None
            else "NONE"
        ),
    )
    print(
        "witness minimum cost :",
        f"{witness_cost:.2f}",
    )
    print(
        "uniform reference    :",
        f"{uniform_cost:.2f}",
    )
    print(
        "goal caps            :",
        (
            " / ".join(
                " ".join(
                    f"{PIECE_NAMES[i]}:{c[i]}"
                    for i in range(20)
                )
                for c in sorted(
                    goal_caps_set
                )
            )
            if goal_caps_set
            else "-"
        ),
    )
    print(
        "witness caps         :",
        " ".join(
            f"{PIECE_NAMES[i]}:"
            f"{witness_caps[i]}"
            for i in range(20)
        ),
    )
    print(
        "exact witness cap match:",
        exact_witness_cap_match,
    )
    print(
        "min-cost solution parity:",
        found_words == expected,
    )
    print(
        "popped/generated     :",
        f"{popped} / {generated}",
    )
    print(
        "Q reference nodes    :",
        qref["nodes"],
    )
    print(
        "Q/capacity prunes    :",
        f"{q_prunes} / {capacity_prunes}",
    )
    print(
        "completion cache     :",
        len(capacity.cache),
    )
    print(
        "wall                 :",
        f"{wall:.4f}s",
    )
    print()

    return {
        "name": name,
        "axis": axis,
        "depth": L,
        "root_q_lb": root_q,
        "root_requirement": list(
            start_req
        ),
        "root_hcap": root_lb_cost,
        "root_optimistic_caps": list(
            root_opt_caps
        ),
        "found": bool(
            goal_rows
        ),
        "cut_reason": cut_reason,
        "minimum_goal_cost": (
            best_cost
        ),
        "witness_cost_diagnostic": (
            witness_cost
        ),
        "uniform_cost_reference": (
            uniform_cost
        ),
        "witness_caps_diagnostic": list(
            witness_caps
        ),
        "goal_cap_vectors": [
            list(x)
            for x in sorted(
                goal_caps_set
            )
        ],
        "exact_witness_cap_match": (
            exact_witness_cap_match
        ),
        "minimum_cost_solutions": (
            goal_rows
        ),
        "solution_parity_expected": (
            found_words == expected
        ),
        "popped_nodes": popped,
        "generated_nodes": generated,
        "q_reference_nodes": int(
            qref["nodes"]
        ),
        "q_prunes": q_prunes,
        "capacity_prunes": (
            capacity_prunes
        ),
        "completion_cache_entries": (
            len(capacity.cache)
        ),
        "wall": wall,
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
        default=100_000,
    )
    ap.add_argument(
        "--time-cap",
        type=float,
        default=2.0,
    )
    ap.add_argument(
        "--output",
        default=(
            "reports/v37/"
            "prefix_capacity_best_first_v37_72.json"
        ),
    )
    args = ap.parse_args()

    print(
        "# CubeLab v37.72 - "
        "PREFIX REQUIREMENT / CAPACITY-COMPLETION BEST-FIRST"
    )
    print(
        "outer cap lattice    :",
        "NONE",
    )
    print(
        "prefix requirement   :",
        "active_used + local_DR_distance",
    )
    print(
        "exact resource       :",
        "final active-count sum = 8L",
    )
    print(
        "priority             :",
        "minimum possible final candidate log10 volume",
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
    qtables, loaded = (
        ts.cache_load_or_build(
            Path(args.cache)
        )
    )
    print(
        "phase tables         :",
        "cache" if loaded else "built",
    )
    print(
        "table wall           :",
        f"{time.perf_counter()-t0:.3f}s",
    )

    dr_targets = (
        v57.local_dr_pose_sets()
    )
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

    total_t0 = (
        time.perf_counter()
    )

    m12_row = search_best_prefix(
        name="M12-02",
        control=m12,
        L=3,
        expected_original=M12_EXPECTED,
        qtables=qtables,
        dr_targets=dr_targets,
        dist_tables=dist_tables,
        node_cap=int(args.node_cap),
        time_cap=float(args.time_cap),
    )

    m11_row = search_best_prefix(
        name="M11-01-EXACT7",
        control=m11,
        L=7,
        expected_original=M11_EXPECTED,
        qtables=qtables,
        dr_targets=dr_targets,
        dist_tables=dist_tables,
        node_cap=int(args.node_cap),
        time_cap=float(args.time_cap),
    )

    # Exact-4 UNSAT Q reference.
    sw4 = v70.internalize(
        tuple(m11["scramble"]),
        m11["axis"],
    )
    q4 = (
        v67.exact_search_with_caps(
            scramble_internal=sw4,
            depth_limit=4,
            caps=(4,) * 20,
            qtables=qtables,
            node_cap=int(args.node_cap),
            time_cap=float(args.time_cap),
        )
    )

    total_wall = (
        time.perf_counter()
        - total_t0
    )

    print(
        "## M11-01 EXACT4 UNSAT"
    )
    print(
        "nodes / solutions    :",
        q4["nodes"],
        "/",
        len(
            q4["solutions_internal"]
        ),
    )
    print()

    print("# DECISION")

    controls_ok = (
        m12_row.get(
            "solution_parity_expected",
            False,
        )
        and m11_row.get(
            "solution_parity_expected",
            False,
        )
        and len(
            q4["solutions_internal"]
        ) == 0
    )

    exact_match = (
        m11_row.get(
            "exact_witness_cap_match",
            False,
        )
    )

    cost_gap = (
        (
            m11_row[
                "minimum_goal_cost"
            ]
            - m11_row[
                "witness_cost_diagnostic"
            ]
        )
        if (
            m11_row.get(
                "minimum_goal_cost"
            )
            is not None
        )
        else float("inf")
    )

    if (
        controls_ok
        and exact_match
    ):
        decision = (
            "PREFIX_CAPACITY_BEST_FIRST_EXACT_PASS"
        )
        note = (
            "Prefix-carried requirement lower bounds plus exact 8L capacity "
            "completion recover the exact controls without cap-lattice "
            "enumeration or witness guidance, and identify the same minimum "
            "per-piece candidate cap vector measured from the exact M11 "
            "witnesses."
        )
    elif (
        controls_ok
        and cost_gap <= 1e-9
    ):
        decision = (
            "PREFIX_CAPACITY_BEST_FIRST_COST_PASS"
        )
        note = (
            "The prefix best-first search reaches the same minimum candidate "
            "portfolio cost as the diagnostic witnesses, though the minimum "
            "cap vector is not unique."
        )
    elif controls_ok:
        decision = (
            "PREFIX_CAPACITY_BEST_FIRST_PASS"
        )
        note = (
            "The prefix/capacity ordering recovers the exact controls, but the "
            "first minimum-cost goal found differs from the diagnostic witness "
            "portfolio."
        )
    else:
        decision = (
            "PREFIX_CAPACITY_BEST_FIRST_FAIL"
        )
        note = (
            "The prefix/capacity FAST search did not recover the known exact "
            "controls within its budget."
        )

    print(decision)
    print(note)
    print(
        "M11 goal/witness gap :",
        (
            f"{cost_gap:.6f} log10"
            if math.isfinite(
                cost_gap
            )
            else "N/A"
        ),
    )
    print(
        "M11 popped vs Q nodes:",
        f"{m11_row.get('popped_nodes')} / "
        f"{m11_row.get('q_reference_nodes')}",
    )
    print(
        "total wall           :",
        f"{total_wall:.3f}s",
    )

    payload = {
        "version": "v37.72",
        "mode": (
            "PREFIX_REQUIREMENT_CAPACITY_COMPLETION_BEST_FIRST"
        ),
        "controls": [
            m12_row,
            m11_row,
        ],
        "m11_exact4_unsat": {
            "nodes": q4["nodes"],
            "solution_count": len(
                q4["solutions_internal"]
            ),
            "wall": q4["wall"],
        },
        "decision": decision,
        "note": note,
        "m11_goal_witness_cost_gap": (
            cost_gap
        ),
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
