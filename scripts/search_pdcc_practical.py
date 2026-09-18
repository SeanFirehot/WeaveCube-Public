#!/usr/bin/env python3
"""CubeLab PDCC v21 practical anytime constructor.

This is deliberately NOT an optimality proof engine.  It complements the exact
``search_pdcc_solver.py`` by spending its time where a good two-phase solution
is most likely to be found instead of exhausting all shallower P1 depths.

Construction policy
-------------------
1. Start from the inverse scramble as a valid incumbent.
2. Compute the exact phase-1 PDB lower bound ``d0``.
3. Probe likely P1 depths around ``d0 + depth_offset`` rather than proving all
   lower depths first.
4. Use PDCC W_i/D_ij child ordering and the exact corner-star full-solution PDB
   as a safe local prune against the current incumbent.
5. At DR, gather a small micro-batch of candidates.  Rank the batch by the exact
   12-double-star K-distance profile and run exact phase2 only for the top few.
6. Every returned full solution is replay verified.  No claim of optimality is
   made unless the separate exact solver later certifies it.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pdcc_double_star as dstar
import pdcc_p1_pose as p1pose
import pdcc_phase2 as p2
import search_pdcc_guided as pg
import search_twist_skeleton as ts
from cubelab.pdcc import PDCCState


@dataclass
class PracticalResult:
    seed_total: int
    best_total: int
    solution: tuple[str, ...]
    nodes: int
    dr_entries: int
    exact_b_calls: int
    phase2_nodes: int
    ranked_terminals: int
    discarded_terminals: int
    tt_pruned: int
    first_improvement_node: int | None
    first_improvement_depth: int | None
    first_improvement_time: float | None
    p1_lb: int
    depth_schedule: tuple[int, ...]
    completed_slices: int
    elapsed: float
    cut: bool


def _make_schedule(lb: int, incumbent: int, offset: int) -> tuple[int, ...]:
    center = max(lb, lb + offset)
    raw = [
        center,
        center + 1,
        center - 1,
        center + 2,
        center - 2,
        center + 3,
        center - 3,
        center + 4,
        center - 4,
        center + 5,
    ]
    out = []
    for d in raw:
        if lb <= d <= incumbent - 1 and d not in out:
            out.append(d)
    return tuple(out)


def solve_practical(
    scramble_word: Sequence[str],
    tables,
    *,
    time_limit: float = 5.0,
    depth_offset: int = 3,
    depth_slice_nodes: int = 20_000,
    p1_policy: str = "relation_first",
    terminal_batch: int = 32,
    terminal_top: int = 4,
    p2_order: str = "auto",
    p2_auto_probe_nodes: int = 128,
    p1_pose_cache: str = "reports/pdcc_cache/p1_corner_star_tables_v1.pkl",
    double_star_cache: str = "reports/pdcc_cache/double_star_k_tables_v1.pkl",
    double_star_index: str = "reports/pdcc_cache/double_star_p2_index_v1.pkl",
    node_cap: int = 10_000_000,
    use_tt: bool = False,
) -> PracticalResult:
    if p1_policy not in {"relation_first", "event_relation", "p1_relation", "star_first"}:
        raise ValueError(p1_policy)
    if p2_order not in {"baseline", "dstar", "auto"}:
        raise ValueError(p2_order)
    terminal_batch = max(1, int(terminal_batch))
    depth_slice_nodes = max(1, int(depth_slice_nodes))
    terminal_top = max(1, min(int(terminal_top), terminal_batch))

    p1pose.configure(p1_pose_cache)
    p1pose.preload()
    if p2_order in {"dstar", "auto"} or terminal_batch > 1:
        p2.configure_double_star_cache(double_star_cache, double_star_index)
        dstar.configure(double_star_cache, double_star_index)

    co, eo, sl, cos, eos, cp, up, sp, cpd, upd, spd = tables
    start = ts.engine.from_word(" ".join(scramble_word))
    q0 = ts.q_of(start)
    seed = ts.normalize(ts.inverse_word(tuple(scramble_word)))
    if not start.apply_word(seed).is_solved():
        raise RuntimeError("inverse seed replay failed")

    incumbent = len(seed)
    best = tuple(seed)
    p1lb = ts.phase1_lb(q0, cos, eos)
    schedule = _make_schedule(p1lb, incumbent, depth_offset)

    nodes = 0
    entries = 0
    bcalls = 0
    phase2_nodes = 0
    ranked = 0
    discarded = 0
    tt_pruned = 0
    first_node = None
    first_depth = None
    first_time = None
    completed_slices = 0
    cut = False
    t0 = time.time()

    for depth in schedule:
        if depth >= incumbent or time.time() - t0 >= time_limit or nodes >= node_cap:
            break

        slice_start_nodes = nodes
        slice_node_limit = min(node_cap, slice_start_nodes + depth_slice_nodes)
        local_slice_stop = False
        path: list[str] = []
        seen: set[tuple[tuple[int, ...], int, str | None]] = set()
        pending: list[tuple[int, tuple[str, ...], str | None]] = []

        def process_pending() -> bool:
            nonlocal incumbent, best, bcalls, phase2_nodes, ranked, discarded, cut
            nonlocal first_node, first_depth, first_time
            if not pending:
                return False

            # Lower profile is better.  Profile values are exact K-projection
            # distances; using them here changes selection only, not correctness
            # of any individual phase2 search.
            pending.sort(key=lambda rec: (dstar.profile(rec[0]), rec[0]))
            ranked += len(pending)
            records = list(pending)
            pending.clear()
            chosen = records[:terminal_top]
            discarded += len(records) - len(chosen)

            for pk, pword, prev_face in chosen:
                # Practical mode is an anytime constructor: do not start another
                # exact-B call after the current depth/global budget expired.
                now = time.time()
                if now - t0 >= time_limit:
                    cut = True
                    break
                max_b = incumbent - len(pword) - 1
                if max_b < 0 or ts.phase2_lb(pk, cpd, upd, spd) > max_b:
                    continue
                stats = {}
                bword, _ = p2.shortest_b_compatible(
                    pk,
                    prev_face,
                    max_b,
                    cp,
                    up,
                    sp,
                    cpd,
                    upd,
                    spd,
                    order=p2_order,
                    stats=stats,
                    auto_probe_nodes=p2_auto_probe_nodes,
                )
                bcalls += 1
                phase2_nodes += stats.get("nodes", 0)
                if bword is None:
                    continue
                sol = tuple(pword) + tuple(bword)
                if len(sol) < incumbent:
                    if not start.apply_word(sol).is_solved():
                        raise RuntimeError("practical candidate replay failed")
                    incumbent = len(sol)
                    best = sol
                    if first_node is None:
                        first_node = nodes
                        first_depth = len(pword)
                        first_time = time.time() - t0
                    # The batch was ranked under the previous incumbent.  A new
                    # incumbent invalidates its remaining caps; return to P1.
                    return True
            return False

        def dfs(q: int, poses: tuple[int, ...], remaining: int, last_face: str | None):
            nonlocal nodes, entries, cut, tt_pruned, local_slice_stop
            if cut or local_slice_stop:
                return
            if time.time() - t0 >= time_limit or nodes >= node_cap:
                cut = True
                return
            if nodes >= slice_node_limit:
                local_slice_stop = True
                return

            nodes += 1
            if ts.phase1_lb(q, cos, eos) > remaining:
                return
            if use_tt:
                tk = (poses, remaining, last_face)
                if tk in seen:
                    tt_pruned += 1
                    return
                seen.add(tk)

            g = depth - remaining
            if g + p1pose.lower_bound(poses) >= incumbent:
                return

            if remaining == 0:
                if q != ts.GOAL_Q:
                    return
                entries += 1
                pk = ts.p_of(PDCCState(poses))
                if pk is None:
                    return
                max_b = incumbent - depth - 1
                if max_b < 0 or ts.phase2_lb(pk, cpd, upd, spd) > max_b:
                    return
                pending.append((pk, tuple(path), path[-1][0] if path else None))
                if len(pending) >= terminal_batch:
                    process_pending()
                return

            if p1_policy == "star_first":
                children = []
                for mi, move in enumerate(ts.MOVE_ORDER):
                    if not ts.allow(last_face, move):
                        continue
                    nq = ts.q_move(q, mi, co, eo, sl)
                    h = ts.phase1_lb(nq, cos, eos)
                    if h > remaining - 1:
                        continue
                    nposes = pg.move_poses(poses, mi)
                    # Exact local full-solution obligations are cheap enough to
                    # serve as a practical ordering profile.  This never prunes.
                    prof = p1pose.profile(nposes)
                    key = (prof, h, 0 if move in ts.T_MOVES else 1, move)
                    children.append((key, mi, move, nq, nposes))
                children.sort(key=lambda x: x[0])
            else:
                children = pg.ordered(
                    p1_policy, q, poses, remaining, last_face, co, eo, sl, cos, eos
                )
            for _key, mi, move, nq, nposes in children:
                if nposes is None:
                    nposes = pg.move_poses(poses, mi)
                path.append(move)
                dfs(nq, nposes, remaining - 1, move[0])
                path.pop()
                if cut or local_slice_stop:
                    return

        dfs(q0, start.poses, depth, None)
        # A partially filled final batch still contains the best-ranked terminals
        # seen in this depth slice.  Spend a little work on them before moving on.
        process_pending()
        completed_slices += 1

        # P1 depth slices are node-count based, never wall-clock based.  This
        # makes the construction order budget-invariant: a longer time budget
        # continues the same node sequence instead of changing depth-switch
        # points because of cache/CPU speed.  Only the global cap can stop the
        # whole search.
        if not (time.time() - t0 >= time_limit or nodes >= node_cap):
            cut = False
        if incumbent <= depth:
            # This depth can no longer beat the new incumbent.
            continue

    elapsed = time.time() - t0
    return PracticalResult(
        seed_total=len(seed),
        best_total=incumbent,
        solution=tuple(best),
        nodes=nodes,
        dr_entries=entries,
        exact_b_calls=bcalls,
        phase2_nodes=phase2_nodes,
        ranked_terminals=ranked,
        discarded_terminals=discarded,
        tt_pruned=tt_pruned,
        first_improvement_node=first_node,
        first_improvement_depth=first_depth,
        first_improvement_time=first_time,
        p1_lb=p1lb,
        depth_schedule=schedule,
        completed_slices=completed_slices,
        elapsed=elapsed,
        cut=(elapsed >= time_limit or nodes >= node_cap),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scramble", required=True)
    ap.add_argument("--time-limit", type=float, default=5.0)
    ap.add_argument("--depth-offset", type=int, default=3)
    ap.add_argument("--depth-slice-nodes", type=int, default=20_000, help="deterministic P1 nodes per scheduled depth slice")
    ap.add_argument(
        "--p1-policy",
        choices=("relation_first", "event_relation", "p1_relation", "star_first"),
        default="relation_first",
    )
    ap.add_argument("--terminal-batch", type=int, default=32)
    ap.add_argument("--terminal-top", type=int, default=4)
    ap.add_argument("--p2-order", choices=("auto", "baseline", "dstar"), default="auto")
    ap.add_argument("--p2-auto-probe-nodes", type=int, default=128)
    ap.add_argument("--node-cap", type=int, default=10_000_000)
    ap.add_argument("--tt", action="store_true", help="practical-only depth-aware full-pose transposition dedup")
    ap.add_argument("--cache", default="reports/pdcc_cache/twist_skeleton_tables_v1.pkl")
    ap.add_argument("--p1-pose-cache", default="reports/pdcc_cache/p1_corner_star_tables_v1.pkl")
    ap.add_argument("--double-star-cache", default="reports/pdcc_cache/double_star_k_tables_v1.pkl")
    ap.add_argument("--double-star-index", default="reports/pdcc_cache/double_star_p2_index_v1.pkl")
    a = ap.parse_args()

    print("loading/building pruning tables...", flush=True)
    tables, loaded = ts.cache_load_or_build(Path(a.cache))
    print(f"tables {'loaded from cache' if loaded else 'built'}", flush=True)

    scramble = tuple(a.scramble.split())
    r = solve_practical(
        scramble,
        tables,
        time_limit=a.time_limit,
        depth_offset=a.depth_offset,
        depth_slice_nodes=a.depth_slice_nodes,
        p1_policy=a.p1_policy,
        terminal_batch=a.terminal_batch,
        terminal_top=a.terminal_top,
        p2_order=a.p2_order,
        p2_auto_probe_nodes=a.p2_auto_probe_nodes,
        p1_pose_cache=a.p1_pose_cache,
        double_star_cache=a.double_star_cache,
        double_star_index=a.double_star_index,
        node_cap=a.node_cap,
        use_tt=a.tt,
    )

    state = ts.engine.from_word(" ".join(scramble))
    replay = state.apply_word(r.solution).is_solved()
    print("\n# PDCC PRACTICAL v21 -- BUDGET-INVARIANT ANYTIME CONSTRUCTION")
    print(f"scramble            : {' '.join(scramble)}")
    print(f"P1 lower bound      : {r.p1_lb}")
    print(f"depth schedule      : {' '.join(map(str, r.depth_schedule))}")
    print(f"completed slices    : {r.completed_slices}")
    print(f"P1 slice nodes      : {a.depth_slice_nodes:,}")
    print(f"seed total          : {r.seed_total}")
    print(f"best total          : {r.best_total}")
    print(f"strict improvement  : {r.best_total < r.seed_total}")
    print(f"certified optimal   : False  (construction mode)")
    print(f"nodes               : {r.nodes:,}")
    print(f"DR terminals        : {r.dr_entries:,}")
    print(f"ranked terminals    : {r.ranked_terminals:,}")
    print(f"discarded terminals : {r.discarded_terminals:,}")
    print(f"TT duplicate prunes : {r.tt_pruned:,}")
    print(f"exact B calls       : {r.exact_b_calls:,}")
    print(f"phase2 nodes        : {r.phase2_nodes:,}")
    print(f"elapsed             : {r.elapsed:.3f}s")
    if r.first_improvement_node is None:
        print("first improvement   : (none)")
    else:
        print(
            "first improvement   : "
            f"node={r.first_improvement_node:,} "
            f"P1={r.first_improvement_depth} "
            f"t={r.first_improvement_time:.3f}s"
        )
    print(f"solution            : {' '.join(r.solution)}")
    print(f"full replay         : {replay}")
    print("\nNOTE: P1 depth switching is node-count based; wall-clock only stops the same deterministic search stream.")
    print("NOTE: practical mode is anytime/heuristic selection; it never certifies optimality.")
    print("NOTE: use search_pdcc_solver.py for exact proof/certification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
