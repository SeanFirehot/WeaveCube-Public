#!/usr/bin/env python3
"""
CubeLab v37.65 — REGULAR-SUPERPOSITION CONFLICT CORE EXTRACTION

Context
-------
v37.64 M11 exact-depth7:

    Q_ONLY      : 34 nodes, 2 exact solutions
    Q_PLUS_GAC  : 33 nodes, 2 exact solutions
    GAC prunes  : 1 Q-surviving state

The v37.64 global "FAIL" label was caused by GAC_ONLY timing out, not by a
solution-set mismatch in Q_PLUS_GAC.

Goal
----
Locate every state in the exact-depth7 Q frontier that:

    phase1_lb <= remaining depth
    BUT
    full 20-piece regular-language GAC is infeasible.

For each such conflict:
  * report prefix / remaining depth / q-LB;
  * recover a minimal or near-minimal subset of piece regular constraints
    that is already GAC-infeasible;
  * distinguish low-order vs distributed coupling.

No solution search beyond the tiny historical Q frontier.
No production changes.
"""

from __future__ import annotations

import argparse
import itertools
import json
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
import search_pdcc_practical_axes as axes

import audit_fast_local_dr_superposition_v37_57 as v57
import audit_fast_regular_superposition_v37_64 as v64


PIECE_NAMES = v64.PIECE_NAMES
NMOVES = len(ts.MOVE_ORDER)
ALL_MOVE_MASK = (1 << NMOVES) - 1


def inverse_face_map(fmap):
    return {dst: src for src, dst in fmap.items()}


def internalize(word, axis):
    return tuple(axes.map_word(tuple(word), axes.AXES[axis]))


def externalize(word, axis):
    inv = inverse_face_map(axes.AXES[axis])
    return tuple(axes.map_word(tuple(word), inv))


def step(poses, q, move, qtables):
    mi = ts.MI[move]
    co, eo, sl = qtables[0], qtables[1], qtables[2]
    nposes = tuple(int(x) for x in pg.move_poses(poses, mi))
    nq = int(ts.q_move(int(q), mi, co, eo, sl))
    return nposes, nq


class SubsetGAC:
    def __init__(self, dr_targets):
        self.target_masks = v64.build_target_masks(dr_targets)
        self.cache = {}

    def propagate(self, poses, rem, previous_face, pieces):
        pieces = tuple(sorted(int(x) for x in pieces))
        key = (
            tuple(int(x) for x in poses),
            int(rem),
            previous_face,
            pieces,
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        if rem == 0:
            feasible = all(
                self.target_masks[i] & v64.pose_bit(poses[i])
                for i in pieces
            )
            result = {
                "feasible": bool(feasible),
                "domains": (),
                "iterations": 0,
                "removed_values": 0,
                "history": [],
            }
            self.cache[key] = result
            return result

        domains = [ALL_MOVE_MASK] * int(rem)
        initial_values = NMOVES * int(rem)
        history = []
        iterations = 0

        while True:
            iterations += 1
            changed = False
            before_sizes = [x.bit_count() for x in domains]

            adj = v64.adjacency_supports(
                tuple(domains),
                previous_face,
            )
            if adj is None:
                result = {
                    "feasible": False,
                    "domains": tuple(domains),
                    "iterations": iterations,
                    "removed_values": (
                        initial_values - sum(x.bit_count() for x in domains)
                    ),
                    "history": history,
                    "failure_stage": "adjacency",
                }
                self.cache[key] = result
                return result

            for k in range(rem):
                nd = domains[k] & adj[k]
                if nd == 0:
                    result = {
                        "feasible": False,
                        "domains": tuple(domains),
                        "iterations": iterations,
                        "removed_values": (
                            initial_values - sum(x.bit_count() for x in domains)
                        ),
                        "history": history,
                        "failure_stage": "adjacency-empty",
                    }
                    self.cache[key] = result
                    return result
                if nd != domains[k]:
                    domains[k] = nd
                    changed = True

            piece_events = []

            for piece_i in pieces:
                before_piece = [x.bit_count() for x in domains]

                supports = v64.piece_regular_supports(
                    piece_i,
                    int(poses[piece_i]),
                    self.target_masks[piece_i],
                    tuple(domains),
                )

                if supports is None:
                    result = {
                        "feasible": False,
                        "domains": tuple(domains),
                        "iterations": iterations,
                        "removed_values": (
                            initial_values - sum(x.bit_count() for x in domains)
                        ),
                        "history": history,
                        "failure_stage": f"piece:{PIECE_NAMES[piece_i]}",
                    }
                    self.cache[key] = result
                    return result

                for k in range(rem):
                    nd = domains[k] & supports[k]
                    if nd == 0:
                        result = {
                            "feasible": False,
                            "domains": tuple(domains),
                            "iterations": iterations,
                            "removed_values": (
                                initial_values - sum(x.bit_count() for x in domains)
                            ),
                            "history": history,
                            "failure_stage": (
                                f"piece-empty:{PIECE_NAMES[piece_i]}:c{k}"
                            ),
                        }
                        self.cache[key] = result
                        return result
                    if nd != domains[k]:
                        domains[k] = nd
                        changed = True

                after_piece = [x.bit_count() for x in domains]
                if after_piece != before_piece:
                    piece_events.append({
                        "piece": PIECE_NAMES[piece_i],
                        "before": before_piece,
                        "after": after_piece,
                    })

            after_sizes = [x.bit_count() for x in domains]
            history.append({
                "iteration": iterations,
                "before": before_sizes,
                "after": after_sizes,
                "piece_events": piece_events,
            })

            if not changed:
                break

            if iterations > initial_values + 5:
                raise RuntimeError("subset GAC did not converge")

        result = {
            "feasible": True,
            "domains": tuple(domains),
            "iterations": iterations,
            "removed_values": (
                initial_values - sum(x.bit_count() for x in domains)
            ),
            "history": history,
            "failure_stage": None,
        }
        self.cache[key] = result
        return result


def q_frontier_conflicts(
    scramble_internal,
    qtables,
    full_gac,
    *,
    depth_limit=7,
):
    start = ts.engine.from_word(" ".join(scramble_internal))
    poses0 = tuple(int(x) for x in start.poses)
    q0 = int(ts.q_of(start))

    cos, eos = qtables[3], qtables[4]

    accepted = 0
    conflicts = []
    solutions = []
    path = []

    def dfs(poses, q, depth, last_face):
        nonlocal accepted
        rem = int(depth_limit) - depth

        h = int(ts.phase1_lb(int(q), cos, eos))
        if h > rem:
            return

        accepted += 1

        gres = full_gac.propagate(
            poses,
            rem,
            last_face,
        )

        if not gres.feasible:
            conflicts.append({
                "prefix_internal": tuple(path),
                "poses": tuple(poses),
                "q": int(q),
                "depth": depth,
                "remaining": rem,
                "last_face": last_face,
                "q_lb": h,
                "gac_iterations": gres.iterations,
                "gac_removed_values": gres.removed_values,
                "gac_domains": tuple(gres.domains),
            })
            # Still continue Q-only traversal: we are auditing the Q frontier.

        if rem == 0:
            if int(q) == int(ts.GOAL_Q):
                solutions.append(tuple(path))
            return

        for move in ts.MOVE_ORDER:
            if last_face is not None and move[0] == last_face:
                continue

            nposes, nq = step(poses, q, move, qtables)
            h2 = int(ts.phase1_lb(nq, cos, eos))
            if h2 > rem - 1:
                continue

            path.append(move)
            dfs(nposes, nq, depth + 1, move[0])
            path.pop()

    dfs(poses0, q0, 0, None)

    return {
        "accepted_q_nodes": accepted,
        "conflicts": conflicts,
        "solutions_internal": tuple(sorted(set(solutions))),
    }


def greedy_core(subset_gac, conflict, all_pieces):
    core = list(all_pieces)

    changed = True
    while changed:
        changed = False
        for piece in list(core):
            trial = tuple(x for x in core if x != piece)
            if not trial:
                continue
            res = subset_gac.propagate(
                conflict["poses"],
                conflict["remaining"],
                conflict["last_face"],
                trial,
            )
            if not res["feasible"]:
                core = list(trial)
                changed = True

    return tuple(core)


def smallest_core_up_to(
    subset_gac,
    conflict,
    *,
    max_size,
    upper_core=None,
):
    all_pieces = tuple(range(20))
    checked = 0

    upper = int(max_size)
    if upper_core is not None:
        upper = min(upper, len(upper_core))

    for size in range(1, upper + 1):
        # If a greedy core is known, test its subsets first at this size.
        candidates = []
        seen = set()

        if upper_core is not None and len(upper_core) >= size:
            for combo in itertools.combinations(upper_core, size):
                candidates.append(combo)
                seen.add(combo)

        for combo in itertools.combinations(all_pieces, size):
            if combo in seen:
                continue
            candidates.append(combo)

        for combo in candidates:
            checked += 1
            res = subset_gac.propagate(
                conflict["poses"],
                conflict["remaining"],
                conflict["last_face"],
                combo,
            )
            if not res["feasible"]:
                return tuple(combo), checked, res

    return None, checked, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache",
        default="reports/pdcc_cache/twist_skeleton_tables_v1.pkl",
    )
    ap.add_argument(
        "--max-exact-core-size",
        type=int,
        default=5,
    )
    ap.add_argument(
        "--output",
        default="reports/v37/regular_superposition_conflict_core_v37_65.json",
    )
    args = ap.parse_args()

    print("# CubeLab v37.65 - REGULAR-SUPERPOSITION CONFLICT CORE")
    print("target               : M11-01 exact depth 7 Q frontier")
    print("goal                 : explain Q-surviving / GAC-infeasible state(s)")
    print("subset core search   :", f"exact up to {args.max_exact_core_size} pieces")
    print("production changes   : NONE")
    print()

    t0 = time.perf_counter()
    qtables, loaded = ts.cache_load_or_build(Path(args.cache))
    print("phase tables         :", "cache" if loaded else "built")
    print("table wall           :", f"{time.perf_counter()-t0:.3f}s")

    dr_targets = v57.local_dr_pose_sets()
    full_gac = v64.RegularSuperposition(dr_targets)
    subset_gac = SubsetGAC(dr_targets)

    m11 = next(
        c for c in v57.CONTROLS
        if c["name"] == "M11-01"
    )
    scramble_internal = internalize(
        m11["scramble"],
        m11["axis"],
    )

    scan_t0 = time.perf_counter()
    scan = q_frontier_conflicts(
        scramble_internal,
        qtables,
        full_gac,
        depth_limit=7,
    )
    scan_wall = time.perf_counter() - scan_t0

    expected_internal = {
        internalize(word, m11["axis"])
        for word in v64.M11_EXACT_ORIGINAL
    }
    found_internal = set(scan["solutions_internal"])

    print("# Q FRONTIER")
    print("accepted Q nodes     :", scan["accepted_q_nodes"])
    print("GAC conflicts        :", len(scan["conflicts"]))
    print("Q exact solutions    :", len(found_internal))
    print("solution-set match   :", found_internal == expected_internal)
    print("scan wall            :", f"{scan_wall:.4f}s")
    print()

    out_conflicts = []

    for idx, conflict in enumerate(scan["conflicts"], 1):
        prefix_original = externalize(
            conflict["prefix_internal"],
            m11["axis"],
        )

        print(f"## CONFLICT {idx}/{len(scan['conflicts'])}")
        print(
            "prefix               :",
            " ".join(prefix_original) if prefix_original else "(root)",
        )
        print("depth / remaining    :", f"{conflict['depth']} / {conflict['remaining']}")
        print("Q lower bound        :", conflict["q_lb"])
        print("last face            :", conflict["last_face"])
        print(
            "full GAC domains     :",
            [x.bit_count() for x in conflict["gac_domains"]],
        )
        print("full GAC removed     :", conflict["gac_removed_values"])

        greedy_t0 = time.perf_counter()
        greedy = greedy_core(
            subset_gac,
            conflict,
            tuple(range(20)),
        )
        greedy_wall = time.perf_counter() - greedy_t0

        print(
            "greedy irreducible core:",
            f"{len(greedy)} pieces",
            " ".join(PIECE_NAMES[i] for i in greedy),
        )
        print("greedy wall          :", f"{greedy_wall:.4f}s")

        exact_t0 = time.perf_counter()
        exact_core, checked, exact_res = smallest_core_up_to(
            subset_gac,
            conflict,
            max_size=int(args.max_exact_core_size),
            upper_core=greedy,
        )
        exact_wall = time.perf_counter() - exact_t0

        if exact_core is not None:
            core = exact_core
            core_type = "EXACT_MIN_WITHIN_SEARCH"
            core_res = exact_res
            print(
                "exact smallest core  :",
                f"{len(core)} pieces",
                " ".join(PIECE_NAMES[i] for i in core),
            )
        else:
            core = greedy
            core_type = "GREEDY_IRREDUCIBLE"
            core_res = subset_gac.propagate(
                conflict["poses"],
                conflict["remaining"],
                conflict["last_face"],
                core,
            )
            print(
                "exact smallest core  :",
                f"not found <= {args.max_exact_core_size}; "
                f"using greedy size {len(core)}",
            )

        print("subset checks        :", f"{checked:,}")
        print("core search wall     :", f"{exact_wall:.4f}s")
        print("core failure stage   :", core_res.get("failure_stage"))

        # Show the propagation chain that actually creates the contradiction.
        history = core_res.get("history", [])
        print("core propagation:")
        if history:
            for step in history:
                print(
                    f"  iter {step['iteration']}: "
                    f"{step['before']} -> {step['after']}"
                )
                for ev in step["piece_events"]:
                    print(
                        f"      {ev['piece']}: "
                        f"{ev['before']} -> {ev['after']}"
                    )
        else:
            print("  (failure before a completed iteration)")

        print()

        out_conflicts.append({
            "prefix_internal": list(conflict["prefix_internal"]),
            "prefix_original": list(prefix_original),
            "depth": conflict["depth"],
            "remaining": conflict["remaining"],
            "q_lb": conflict["q_lb"],
            "last_face": conflict["last_face"],
            "full_gac_domain_sizes": [
                x.bit_count() for x in conflict["gac_domains"]
            ],
            "full_gac_removed_values": conflict["gac_removed_values"],
            "greedy_core": [PIECE_NAMES[i] for i in greedy],
            "greedy_core_size": len(greedy),
            "selected_core": [PIECE_NAMES[i] for i in core],
            "selected_core_indices": list(core),
            "selected_core_size": len(core),
            "core_type": core_type,
            "core_failure_stage": core_res.get("failure_stage"),
            "subset_checks": checked,
            "history": history,
        })

    print("# DECISION")

    if not scan["conflicts"]:
        decision = "NO_GAC_CONFLICT_FOUND"
        note = (
            "The v37.64 telemetry could not be reproduced on the Q frontier."
        )
    else:
        min_size = min(x["selected_core_size"] for x in out_conflicts)
        if min_size <= 3:
            decision = "LOW_ORDER_REGULAR_CONFLICT_FOUND"
            note = (
                "The only Q-surviving GAC contradiction is explained by a "
                f"{min_size}-piece regular-language factor. Build a cheap "
                "specialized factor/lookup instead of running full 20-piece GAC."
            )
        elif min_size <= 6:
            decision = "MID_ORDER_REGULAR_CONFLICT_FOUND"
            note = (
                "The independent GAC signal requires a mid-order joint factor. "
                "Test a compact factor only for this core family before broader "
                "integration."
            )
        else:
            decision = "DISTRIBUTED_REGULAR_CONFLICT"
            note = (
                "The GAC contradiction is genuinely distributed across many "
                "pieces; low-order factorization is unlikely to reproduce it."
            )

    print(decision)
    print(note)
    print("full GAC cache       :", len(full_gac.cache))
    print("subset GAC cache     :", len(subset_gac.cache))

    payload = {
        "version": "v37.65",
        "mode": "REGULAR_SUPERPOSITION_CONFLICT_CORE_EXTRACTION",
        "q_frontier_nodes": scan["accepted_q_nodes"],
        "q_solution_set_match": found_internal == expected_internal,
        "conflict_count": len(scan["conflicts"]),
        "conflicts": out_conflicts,
        "scan_wall": scan_wall,
        "full_gac_cache_entries": len(full_gac.cache),
        "subset_gac_cache_entries": len(subset_gac.cache),
        "decision": decision,
        "note": note,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
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
