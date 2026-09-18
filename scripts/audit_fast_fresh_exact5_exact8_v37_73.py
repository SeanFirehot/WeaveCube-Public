#!/usr/bin/env python3
"""
CubeLab v37.73 — FRESH EXACT-5 / EXACT-8 PREFIX-CAPACITY HOLDOUT

Purpose
-------
v37.72 on historical controls:
    M12 exact-3: exact witness-minimal cap recovered
    M11 exact-7: exact witness-minimal cap recovered
    but no extra feasibility pruning beyond phase1_lb.

Now satisfy the short-control generalization requirement with FRESH
exact-depth controls at 5 and 8 moves.

Fresh control construction
--------------------------
Start from solved cube, which is DR.

Generate a deterministic random reduced word W of target length L.
Let S = solved.apply(W).

Because W^{-1} returns S to solved/DR:
    exact_DR_distance(S) <= L.

Accept the generated control only when:
    phase1_lb(S) == L.

Since phase1_lb is admissible:
    exact_DR_distance(S) >= L.

Therefore:
    exact_DR_distance(S) == L

without using any solution search.

We generate:
    one exact-5 control
    one exact-8 control

using a fixed RNG seed.

Validation
----------
For each fresh control:

1. Enumerate ALL exact-depth-L DR words under Q pruning.
2. Replay each and compute its exact 20-piece active-count cap vector.
3. Compute exact candidate-portfolio log10 cost for every exact solution.
4. Determine the true minimum cost and all minimum-cost exact solutions/caps.
5. Run v37.72 prefix-capacity best-first WITHOUT using these words in its
   ordering.
6. Compare:
       minimum cost
       minimum-cost solution set
       minimum cap-vector set

Ordering diagnostic
-------------------
Also compare time-to-FIRST solution under:
    Q default move-order DFS
    Hcap best-first

This distinguishes:
    * correctness/optimization signal
    * actual first-hit ordering benefit

No production changes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
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
import audit_fast_requirement_nonuniform_v37_70 as v70
import audit_fast_prefix_capacity_best_first_v37_72 as v72


MOVE_RANK = v70.MOVE_RANK
PIECE_NAMES = v70.PIECE_NAMES


def inverse_move(move):
    if move.endswith("2"):
        return move
    if move.endswith("'"):
        return move[:-1]
    return move + "'"


def inverse_word(word):
    return tuple(
        inverse_move(m)
        for m in reversed(tuple(word))
    )


def random_reduced_word(rng, length):
    out = []
    last_face = None

    for _ in range(int(length)):
        choices = [
            m
            for m in ts.MOVE_ORDER
            if (
                last_face is None
                or m[0] != last_face
            )
        ]
        move = rng.choice(choices)
        out.append(move)
        last_face = move[0]

    return tuple(out)


def q_lb_of_state(state, qtables):
    q = int(ts.q_of(state))
    return int(
        ts.phase1_lb(
            q,
            qtables[3],
            qtables[4],
        )
    )


def make_fresh_exact_control(
    *,
    target_depth,
    rng,
    qtables,
    max_tries,
):
    for attempt in range(
        1,
        int(max_tries) + 1,
    ):
        word = random_reduced_word(
            rng,
            target_depth,
        )

        state = ts.engine.from_word(
            " ".join(word)
        )

        h = q_lb_of_state(
            state,
            qtables,
        )

        if h != int(target_depth):
            continue

        if ts.is_dr(state):
            continue

        return {
            "scramble": word,
            "axis": "UD",
            "depth": int(target_depth),
            "root_q_lb": int(h),
            "planted_inverse": inverse_word(
                word
            ),
            "generation_attempt": attempt,
        }

    raise RuntimeError(
        f"could not generate exact-{target_depth} control "
        f"with root Q-LB={target_depth} in {max_tries} tries"
    )


def step_q(poses, q, move, qtables):
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


def enumerate_all_exact(
    *,
    scramble,
    depth,
    qtables,
    node_cap,
    time_cap,
):
    start = ts.engine.from_word(
        " ".join(scramble)
    )
    poses0 = tuple(
        int(x)
        for x in start.poses
    )
    q0 = int(ts.q_of(start))

    deadline = (
        time.perf_counter()
        + float(time_cap)
    )
    t0 = time.perf_counter()

    nodes = 0
    q_prunes = 0
    solutions = []
    path = []
    first_goal_node = None
    first_goal_word = None

    class Cut(Exception):
        pass

    def check_cut():
        if nodes >= int(node_cap):
            raise Cut("node_cap")
        if time.perf_counter() >= deadline:
            raise Cut("time_cap")

    def dfs(poses, q, d, last_face):
        nonlocal nodes, q_prunes
        nonlocal first_goal_node, first_goal_word

        check_cut()

        rem = int(depth) - d
        h = int(
            ts.phase1_lb(
                int(q),
                qtables[3],
                qtables[4],
            )
        )

        if h > rem:
            q_prunes += 1
            return

        nodes += 1

        if rem == 0:
            if int(q) == int(ts.GOAL_Q):
                word = tuple(path)
                final = start.apply_word(
                    word
                )

                if not ts.is_dr(final):
                    raise RuntimeError(
                        "q-goal / explicit DR mismatch"
                    )

                solutions.append(word)

                if first_goal_node is None:
                    first_goal_node = nodes
                    first_goal_word = word

            return

        for move in ts.MOVE_ORDER:
            if (
                last_face is not None
                and move[0] == last_face
            ):
                continue

            nposes, nq = step_q(
                poses,
                q,
                move,
                qtables,
            )

            h2 = int(
                ts.phase1_lb(
                    nq,
                    qtables[3],
                    qtables[4],
                )
            )

            if h2 > rem - 1:
                q_prunes += 1
                continue

            path.append(move)
            dfs(
                nposes,
                nq,
                d + 1,
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
        cut = str(exc)

    return {
        "nodes": nodes,
        "q_prunes": q_prunes,
        "solutions": tuple(
            sorted(
                set(solutions)
            )
        ),
        "first_goal_node": (
            first_goal_node
        ),
        "first_goal_word": (
            first_goal_word
        ),
        "wall": (
            time.perf_counter()
            - t0
        ),
        "cut_reason": cut,
    }


def active_counts_for_word(
    scramble,
    word,
):
    start = ts.engine.from_word(
        " ".join(scramble)
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

    return tuple(counts)


def hcap_first_goal(
    *,
    scramble,
    depth,
    qtables,
    dr_targets,
    dist_tables,
    node_cap,
    time_cap,
):
    start = ts.engine.from_word(
        " ".join(scramble)
    )
    start_poses = tuple(
        int(x)
        for x in start.poses
    )
    q0 = int(ts.q_of(start))

    capacity = v72.CapacityCompletion(
        start_poses,
        dr_targets,
        depth,
    )

    start_counts = (0,) * 20
    start_req = v70.requirement_vector(
        start_poses,
        start_counts,
        dist_tables,
    )
    root = capacity.solve(
        start_req
    )

    if root is None:
        raise RuntimeError(
            "fresh control root has no capacity completion"
        )

    h0, caps0 = root

    heap = []
    seq = 0

    heap.append(
        (
            float(h0),
            0,
            int(depth)
            - int(
                ts.phase1_lb(
                    q0,
                    qtables[3],
                    qtables[4],
                )
            ),
            seq,
            start_poses,
            start_counts,
            q0,
            None,
            (),
            start_req,
            caps0,
        )
    )

    import heapq
    heapq.heapify(heap)

    deadline = (
        time.perf_counter()
        + float(time_cap)
    )
    t0 = time.perf_counter()

    popped = 0
    generated = 1
    q_prunes = 0
    capacity_prunes = 0

    while heap:
        if popped >= int(node_cap):
            return {
                "found": False,
                "cut_reason": "node_cap",
                "popped": popped,
                "generated": generated,
                "wall": time.perf_counter() - t0,
            }

        if time.perf_counter() >= deadline:
            return {
                "found": False,
                "cut_reason": "time_cap",
                "popped": popped,
                "generated": generated,
                "wall": time.perf_counter() - t0,
            }

        (
            hcap,
            _negdepth,
            _qslack,
            _seq,
            poses,
            counts,
            q,
            last_face,
            path,
            req,
            optcaps,
        ) = heapq.heappop(heap)

        popped += 1
        rem = int(depth) - len(path)

        if rem == 0:
            if int(q) == int(ts.GOAL_Q):
                final = start.apply_word(
                    tuple(path)
                )
                if not ts.is_dr(final):
                    raise RuntimeError(
                        "q-goal / DR mismatch in first-goal search"
                    )

                return {
                    "found": True,
                    "word": tuple(path),
                    "popped": popped,
                    "generated": generated,
                    "goal_cost": float(hcap),
                    "goal_caps": tuple(req),
                    "q_prunes": q_prunes,
                    "capacity_prunes": capacity_prunes,
                    "completion_cache": len(
                        capacity.cache
                    ),
                    "wall": (
                        time.perf_counter()
                        - t0
                    ),
                    "cut_reason": None,
                }

            continue

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
            ) = v72.step_state(
                poses,
                counts,
                q,
                move,
                qtables,
            )

            hq = int(
                ts.phase1_lb(
                    nq,
                    qtables[3],
                    qtables[4],
                )
            )

            if hq > rem - 1:
                q_prunes += 1
                continue

            nreq = v70.requirement_vector(
                nposes,
                ncounts,
                dist_tables,
            )

            completion = capacity.solve(
                nreq
            )

            if completion is None:
                capacity_prunes += 1
                continue

            nhcap, nopt = completion

            children.append(
                (
                    float(nhcap),
                    -(len(path) + 1),
                    (rem - 1) - hq,
                    MOVE_RANK[move],
                    move,
                    nposes,
                    ncounts,
                    nq,
                    nreq,
                    nopt,
                )
            )

        children.sort()

        for (
            nhcap,
            negd,
            qslack,
            _rank,
            move,
            nposes,
            ncounts,
            nq,
            nreq,
            nopt,
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
                    negd,
                    qslack,
                    seq,
                    nposes,
                    ncounts,
                    nq,
                    move[0],
                    npath,
                    nreq,
                    nopt,
                )
            )
            generated += 1

    return {
        "found": False,
        "cut_reason": "open_exhausted",
        "popped": popped,
        "generated": generated,
        "wall": time.perf_counter() - t0,
    }


def evaluate_control(
    *,
    control,
    qtables,
    dr_targets,
    dist_tables,
    node_cap,
    enum_time_cap,
    hcap_time_cap,
):
    scramble = tuple(
        control["scramble"]
    )
    L = int(
        control["depth"]
    )

    print(
        f"## FRESH EXACT-{L}"
    )
    print(
        "generated scramble   :",
        " ".join(scramble),
    )
    print(
        "generation attempt   :",
        control[
            "generation_attempt"
        ],
    )
    print(
        "root Q-LB / depth    :",
        f"{control['root_q_lb']} / {L}",
    )
    print(
        "planted inverse      :",
        " ".join(
            control[
                "planted_inverse"
            ]
        ),
    )

    enum = enumerate_all_exact(
        scramble=scramble,
        depth=L,
        qtables=qtables,
        node_cap=node_cap,
        time_cap=enum_time_cap,
    )

    if enum["cut_reason"] is not None:
        print(
            "ENUM CUT:",
            enum["cut_reason"],
        )
        return {
            "control": control,
            "enum_cut": enum[
                "cut_reason"
            ],
            "passed": False,
        }

    if not enum["solutions"]:
        raise RuntimeError(
            f"exact-{L} control has no exact solution despite planted inverse"
        )

    start = ts.engine.from_word(
        " ".join(scramble)
    )
    start_poses = tuple(
        int(x)
        for x in start.poses
    )

    capacity = v72.CapacityCompletion(
        start_poses,
        dr_targets,
        L,
    )

    solution_rows = []

    for word in enum["solutions"]:
        caps = active_counts_for_word(
            scramble,
            word,
        )
        cost = v70.cap_log_volume(
            caps,
            capacity.cost_table,
        )
        solution_rows.append({
            "word": word,
            "caps": caps,
            "cost": float(cost),
        })

    min_cost = min(
        row["cost"]
        for row in solution_rows
    )

    min_rows = [
        row
        for row in solution_rows
        if abs(
            row["cost"] - min_cost
        ) <= 1e-12
    ]

    min_solution_set = {
        row["word"]
        for row in min_rows
    }
    min_cap_set = {
        row["caps"]
        for row in min_rows
    }

    print(
        "exact Q nodes        :",
        enum["nodes"],
    )
    print(
        "exact solution count :",
        len(
            enum["solutions"]
        ),
    )
    print(
        "Q first-goal node    :",
        enum[
            "first_goal_node"
        ],
    )
    print(
        "true min log10D      :",
        f"{min_cost:.2f}",
    )
    print(
        "min-cost sol count   :",
        len(
            min_solution_set
        ),
    )
    print(
        "min cap-vector count :",
        len(
            min_cap_set
        ),
    )

    # Full v37.72 minimum-cost collector.
    pseudo_control = {
        "scramble": scramble,
        "axis": "UD",
    }

    v72row = v72.search_best_prefix(
        name=f"FRESH-EXACT{L}",
        control=pseudo_control,
        L=L,
        expected_original=min_solution_set,
        qtables=qtables,
        dr_targets=dr_targets,
        dist_tables=dist_tables,
        node_cap=node_cap,
        time_cap=hcap_time_cap,
    )

    returned_cost = (
        v72row[
            "minimum_goal_cost"
        ]
    )
    returned_solutions = {
        tuple(
            row[
                "word_original"
            ]
        )
        for row in v72row[
            "minimum_cost_solutions"
        ]
    }
    returned_caps = {
        tuple(x)
        for x in v72row[
            "goal_cap_vectors"
        ]
    }

    cost_match = (
        returned_cost is not None
        and abs(
            float(returned_cost)
            - float(min_cost)
        ) <= 1e-12
    )
    sol_match = (
        returned_solutions
        == min_solution_set
    )
    cap_match = (
        returned_caps
        == min_cap_set
    )

    # First-hit ordering A/B.
    hfirst = hcap_first_goal(
        scramble=scramble,
        depth=L,
        qtables=qtables,
        dr_targets=dr_targets,
        dist_tables=dist_tables,
        node_cap=node_cap,
        time_cap=hcap_time_cap,
    )

    hfirst_valid = False
    if hfirst.get(
        "found"
    ):
        hword = tuple(
            hfirst["word"]
        )
        hfirst_valid = (
            hword in set(
                enum["solutions"]
            )
        )

    print(
        "v37.72 cost match    :",
        cost_match,
    )
    print(
        "v37.72 solution match:",
        sol_match,
    )
    print(
        "v37.72 cap match     :",
        cap_match,
    )
    print(
        "first-hit Q/Hcap     :",
        f"{enum['first_goal_node']} / "
        f"{hfirst.get('popped')}",
    )

    if (
        enum["first_goal_node"]
        and hfirst.get(
            "popped"
        )
    ):
        first_hit_gain = (
            1.0
            - hfirst["popped"]
            / enum[
                "first_goal_node"
            ]
        )
    else:
        first_hit_gain = 0.0

    print(
        "first-hit node gain  :",
        f"{first_hit_gain:.2%}",
    )
    print()

    passed = (
        cost_match
        and sol_match
        and cap_match
        and hfirst_valid
    )

    return {
        "control": {
            **control,
            "scramble": list(
                scramble
            ),
            "planted_inverse": list(
                control[
                    "planted_inverse"
                ]
            ),
        },
        "exact_enumeration": {
            "nodes": enum[
                "nodes"
            ],
            "q_prunes": enum[
                "q_prunes"
            ],
            "solution_count": len(
                enum[
                    "solutions"
                ]
            ),
            "solutions": [
                list(x)
                for x in enum[
                    "solutions"
                ]
            ],
            "first_goal_node": enum[
                "first_goal_node"
            ],
            "first_goal_word": (
                list(
                    enum[
                        "first_goal_word"
                    ]
                )
                if enum[
                    "first_goal_word"
                ]
                is not None
                else None
            ),
            "wall": enum["wall"],
        },
        "true_minimum": {
            "log10_candidate_volume": (
                min_cost
            ),
            "solution_count": len(
                min_solution_set
            ),
            "solutions": [
                list(x)
                for x in sorted(
                    min_solution_set
                )
            ],
            "cap_vectors": [
                list(x)
                for x in sorted(
                    min_cap_set
                )
            ],
        },
        "v37_72": {
            "minimum_goal_cost": (
                returned_cost
            ),
            "returned_solutions": [
                list(x)
                for x in sorted(
                    returned_solutions
                )
            ],
            "returned_cap_vectors": [
                list(x)
                for x in sorted(
                    returned_caps
                )
            ],
            "cost_match": cost_match,
            "solution_match": (
                sol_match
            ),
            "cap_match": cap_match,
            "popped_nodes": v72row[
                "popped_nodes"
            ],
            "generated_nodes": v72row[
                "generated_nodes"
            ],
            "wall": v72row[
                "wall"
            ],
        },
        "first_hit_ab": {
            "q_first_goal_node": enum[
                "first_goal_node"
            ],
            "hcap_first_goal_popped": (
                hfirst.get(
                    "popped"
                )
            ),
            "hcap_first_goal_word": (
                list(
                    hfirst["word"]
                )
                if hfirst.get(
                    "found"
                )
                else None
            ),
            "hcap_valid": hfirst_valid,
            "node_gain": (
                first_hit_gain
            ),
            "hcap_wall": hfirst[
                "wall"
            ],
        },
        "passed": passed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seed",
        type=int,
        default=20260923,
    )
    ap.add_argument(
        "--max-generation-tries",
        type=int,
        default=20000,
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
        default=1_000_000,
    )
    ap.add_argument(
        "--enum-time-cap",
        type=float,
        default=3.0,
    )
    ap.add_argument(
        "--hcap-time-cap",
        type=float,
        default=3.0,
    )
    ap.add_argument(
        "--output",
        default=(
            "reports/v37/"
            "fresh_exact5_exact8_prefix_capacity_v37_73.json"
        ),
    )
    args = ap.parse_args()

    print(
        "# CubeLab v37.73 - "
        "FRESH EXACT-5 / EXACT-8 PREFIX-CAPACITY HOLDOUT"
    )
    print(
        "control construction :",
        "random reduced planted word + root Q-LB==target depth",
    )
    print(
        "exact-depth proof    :",
        "Q-LB==L and inverse planted witness length L",
    )
    print(
        "validation           :",
        "full exact Q enumeration vs v37.72 minimum portfolio",
    )
    print(
        "ordering A/B         :",
        "Q default first-hit vs Hcap first-hit",
    )
    print(
        "known words in policy:",
        "NO",
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

    rng = random.Random(
        int(args.seed)
    )

    controls = []

    for target in (
        5,
        8,
    ):
        c = make_fresh_exact_control(
            target_depth=target,
            rng=rng,
            qtables=qtables,
            max_tries=int(
                args.max_generation_tries
            ),
        )
        controls.append(c)

    print()

    total_t0 = (
        time.perf_counter()
    )

    rows = []

    for control in controls:
        row = evaluate_control(
            control=control,
            qtables=qtables,
            dr_targets=dr_targets,
            dist_tables=dist_tables,
            node_cap=int(
                args.node_cap
            ),
            enum_time_cap=float(
                args.enum_time_cap
            ),
            hcap_time_cap=float(
                args.hcap_time_cap
            ),
        )
        rows.append(row)

    total_wall = (
        time.perf_counter()
        - total_t0
    )

    print("# DECISION")

    all_pass = all(
        row.get(
            "passed",
            False,
        )
        for row in rows
    )

    gains = [
        row[
            "first_hit_ab"
        ][
            "node_gain"
        ]
        for row in rows
        if row.get(
            "passed"
        )
    ]

    if all_pass:
        if (
            gains
            and sum(gains) / len(gains)
            > 0
        ):
            decision = (
                "FRESH_EXACT5_EXACT8_PREFIX_CAPACITY_PASS_WITH_ORDER_SIGNAL"
            )
            note = (
                "The prefix-capacity objective exactly reproduces the true "
                "minimum candidate-portfolio exact solutions on fresh exact-5 "
                "and exact-8 controls, and also improves first-hit node order "
                "on average."
            )
        else:
            decision = (
                "FRESH_EXACT5_EXACT8_PREFIX_CAPACITY_EXACT_PASS"
            )
            note = (
                "The prefix-capacity objective exactly reproduces the true "
                "minimum candidate-portfolio exact solutions on fresh exact-5 "
                "and exact-8 controls. First-hit ordering shows no node benefit, "
                "so retain it as an optimization/selection signal rather than "
                "a speed heuristic."
            )
    else:
        decision = (
            "FRESH_EXACT5_EXACT8_PREFIX_CAPACITY_FAIL"
        )
        note = (
            "At least one fresh exact-depth control did not match the full "
            "exact enumeration within budget. Do not generalize v37.72 yet."
        )

    print(decision)
    print(note)

    if gains:
        print(
            "mean first-hit gain :",
            f"{sum(gains)/len(gains):.2%}",
        )

    print(
        "total wall           :",
        f"{total_wall:.3f}s",
    )

    payload = {
        "version": "v37.73",
        "mode": (
            "FRESH_EXACT5_EXACT8_PREFIX_CAPACITY_HOLDOUT"
        ),
        "seed": int(
            args.seed
        ),
        "controls": rows,
        "decision": decision,
        "note": note,
        "mean_first_hit_node_gain": (
            sum(gains) / len(gains)
            if gains
            else None
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
