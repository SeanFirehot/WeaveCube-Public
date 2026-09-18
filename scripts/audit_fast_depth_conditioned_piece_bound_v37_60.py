#!/usr/bin/env python3
"""
CubeLab v37.60 — DEPTH-CONDITIONED 20-PIECE LOCAL-DR BOUND

What v37.59 taught us
---------------------
The analytic excursion quantity

    r_p = a_p + d_p(current, localDR) - d_p(start, localDR)

is correct, but the M12-trained fixed envelope
    max<=3, sum<=15, count<=8
does not generalize to M11 exact-7.

The universal depth-conditioned consequence is different.

If total target depth is L, current global depth is t, and R=L-t remains:

    d_p(current) <= R                         for every piece p

because one global move can give a piece at most one active event.

Also every global face turn moves exactly 8 movable pieces, so:

    sum_p d_p(current) <= 8*R

Therefore an admissible independent 20-piece lower bound is

    h_piece =
        max(
            max_p d_p,
            ceil(sum_p d_p / 8)
        )

This audit compares:

    NONE
    PIECE_MAX
    PIECE_CAPACITY
    PIECE_BOTH
    Q_ONLY
    Q_PLUS_PIECE

on:
    M12-02 bounded <=3 SAT
    M11-01 bounded <=4 UNSAT
    M11-01 exact depth 7 positive holdout

No candidate domains are materialized.
No learned envelope is used.
No production changes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, deque
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


MOVE_RANK = {m: i for i, m in enumerate(ts.MOVE_ORDER)}

M11_EXACT_ORIGINAL = (
    ("B'", "U'", "R2", "F2", "D2", "B'", "U"),
    ("B'", "U'", "R2", "F2", "D2", "B'", "U'"),
)


def inverse_face_map(fmap):
    return {dst: src for src, dst in fmap.items()}


def internalize(word, axis):
    return tuple(axes.map_word(tuple(word), axes.AXES[axis]))


def externalize(word, axis):
    inv = inverse_face_map(axes.AXES[axis])
    return tuple(axes.map_word(tuple(word), inv))


def build_piece_distance_tables(dr_targets):
    tables = []

    for piece_i in range(20):
        targets = dr_targets[piece_i]
        rev = [[] for _ in range(24)]

        for pose in range(24):
            for move in ts.MOVE_ORDER:
                nxt = int(
                    pg.POSE_NEXT[
                        ts.MI[move]
                    ][piece_i][pose]
                )
                if nxt == pose:
                    continue
                rev[nxt].append(pose)

        dist = [999] * 24
        q = deque()

        for target in targets:
            dist[int(target)] = 0
            q.append(int(target))

        while q:
            cur = q.popleft()
            nd = dist[cur] + 1

            for pred in rev[cur]:
                if dist[pred] > nd:
                    dist[pred] = nd
                    q.append(pred)

        if any(x >= 999 for x in dist):
            raise RuntimeError(
                f"piece {piece_i}: local-DR distance table incomplete"
            )

        tables.append(tuple(int(x) for x in dist))

    return tuple(tables)


def local_stats(poses, dist_tables):
    ds = tuple(
        int(dist_tables[i][int(poses[i])])
        for i in range(20)
    )
    max_d = max(ds)
    sum_d = sum(ds)
    h_capacity = (sum_d + 7) // 8
    h_piece = max(max_d, h_capacity)

    return {
        "distances": ds,
        "max_d": max_d,
        "sum_d": sum_d,
        "capacity_lb": h_capacity,
        "piece_lb": h_piece,
    }


def step_poses(poses, move):
    return tuple(
        int(x)
        for x in pg.move_poses(
            poses,
            ts.MI[move],
        )
    )


def mode_pass(mode, q, poses, rem, qtables, dist_tables):
    co, eo, sl, cos, eos, *_ = qtables
    stats = local_stats(poses, dist_tables)

    if mode in ("PIECE_MAX", "PIECE_BOTH", "Q_PLUS_PIECE"):
        if stats["max_d"] > rem:
            return False, stats, "piece_max"

    if mode in ("PIECE_CAPACITY", "PIECE_BOTH", "Q_PLUS_PIECE"):
        if stats["sum_d"] > 8 * rem:
            return False, stats, "piece_capacity"

    if mode in ("Q_ONLY", "Q_PLUS_PIECE"):
        if int(ts.phase1_lb(int(q), cos, eos)) > rem:
            return False, stats, "q_lb"

    return True, stats, None


@dataclass
class Census:
    mode: str
    nodes: int
    by_depth: dict
    dr_by_depth: dict
    dr_words: tuple
    prune_counts: dict
    wall: float


def bounded_census(
    *,
    mode,
    scramble_internal,
    max_len,
    qtables,
    dist_tables,
):
    co, eo, sl, *_ = qtables

    start = ts.engine.from_word(" ".join(scramble_internal))
    poses0 = tuple(int(x) for x in start.poses)
    q0 = int(ts.q_of(start))

    nodes = 0
    by_depth = Counter()
    dr_by_depth = Counter()
    dr_words = []
    prunes = Counter()

    def dfs(word, poses, q, depth, last_face):
        nonlocal nodes

        rem = int(max_len) - depth
        ok, _stats, reason = mode_pass(
            mode,
            q,
            poses,
            rem,
            qtables,
            dist_tables,
        )

        if not ok:
            prunes[reason] += 1
            return

        nodes += 1
        by_depth[depth] += 1

        if int(q) == int(ts.GOAL_Q):
            state = start.apply_word(tuple(word))
            if not ts.is_dr(state):
                raise RuntimeError(
                    f"q-goal / explicit DR mismatch mode={mode} word={word}"
                )
            dr_by_depth[depth] += 1
            dr_words.append(tuple(word))

        if depth >= int(max_len):
            return

        for move in ts.MOVE_ORDER:
            if last_face is not None and move[0] == last_face:
                continue

            nposes = step_poses(poses, move)
            nq = int(
                ts.q_move(
                    q,
                    ts.MI[move],
                    co,
                    eo,
                    sl,
                )
            )

            dfs(
                tuple(word) + (move,),
                nposes,
                nq,
                depth + 1,
                move[0],
            )

    t0 = time.perf_counter()
    dfs(
        (),
        poses0,
        q0,
        0,
        None,
    )

    return Census(
        mode=mode,
        nodes=nodes,
        by_depth=dict(sorted(by_depth.items())),
        dr_by_depth=dict(sorted(dr_by_depth.items())),
        dr_words=tuple(sorted(set(dr_words))),
        prune_counts=dict(prunes),
        wall=time.perf_counter() - t0,
    )


def exact_depth_search(
    *,
    mode,
    scramble_internal,
    depth_limit,
    qtables,
    dist_tables,
    node_cap,
    time_cap,
):
    co, eo, sl, *_ = qtables

    start = ts.engine.from_word(" ".join(scramble_internal))
    poses0 = tuple(int(x) for x in start.poses)
    q0 = int(ts.q_of(start))

    deadline = time.perf_counter() + float(time_cap)
    t0 = time.perf_counter()

    nodes = 0
    prunes = Counter()
    solutions = []
    path = []

    class Cut(Exception):
        pass

    def check_cut():
        if nodes >= int(node_cap):
            raise Cut("node_cap")
        if time.perf_counter() >= deadline:
            raise Cut("time_cap")

    def dfs(poses, q, depth, last_face):
        nonlocal nodes

        check_cut()
        rem = int(depth_limit) - depth

        ok, stats, reason = mode_pass(
            mode,
            q,
            poses,
            rem,
            qtables,
            dist_tables,
        )

        if not ok:
            prunes[reason] += 1
            return

        nodes += 1

        if rem == 0:
            if int(q) == int(ts.GOAL_Q):
                word = tuple(path)
                state = start.apply_word(word)
                if not ts.is_dr(state):
                    raise RuntimeError(
                        f"q-goal / explicit DR mismatch mode={mode} word={word}"
                    )
                solutions.append(word)
            return

        children = []

        for move in ts.MOVE_ORDER:
            if last_face is not None and move[0] == last_face:
                continue

            nposes = step_poses(poses, move)
            nq = int(
                ts.q_move(
                    q,
                    ts.MI[move],
                    co,
                    eo,
                    sl,
                )
            )

            child_ok, child_stats, child_reason = mode_pass(
                mode,
                nq,
                nposes,
                rem - 1,
                qtables,
                dist_tables,
            )

            if not child_ok:
                prunes[child_reason] += 1
                continue

            children.append((
                int(ts.phase1_lb(nq, qtables[3], qtables[4])),
                child_stats["piece_lb"],
                child_stats["sum_d"],
                child_stats["max_d"],
                MOVE_RANK[move],
                move,
                nposes,
                nq,
            ))

        children.sort()

        for (
            _hq,
            _hp,
            _sumd,
            _maxd,
            _rank,
            move,
            nposes,
            nq,
        ) in children:
            path.append(move)
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
        cut = str(exc)

    return {
        "mode": mode,
        "nodes": nodes,
        "prune_counts": dict(prunes),
        "solutions_internal": [
            list(x)
            for x in sorted(set(solutions))
        ],
        "wall": time.perf_counter() - t0,
        "cut_reason": cut,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache",
        default="reports/pdcc_cache/twist_skeleton_tables_v1.pkl",
    )
    ap.add_argument("--node-cap", type=int, default=1_000_000)
    ap.add_argument("--time-cap", type=float, default=3.0)
    ap.add_argument(
        "--output",
        default="reports/v37/depth_conditioned_piece_bound_v37_60.json",
    )
    args = ap.parse_args()

    modes = (
        "NONE",
        "PIECE_MAX",
        "PIECE_CAPACITY",
        "PIECE_BOTH",
        "Q_ONLY",
        "Q_PLUS_PIECE",
    )

    print("# CubeLab v37.60 - DEPTH-CONDITIONED 20-PIECE LOCAL-DR BOUND")
    print("candidate domains    : NONE")
    print("learned envelope     : NONE")
    print("piece bound          : max(max d_p, ceil(sum d_p/8))")
    print("modes                :", " / ".join(modes))
    print("production changes   : NONE")
    print()

    t0 = time.perf_counter()
    qtables, loaded = ts.cache_load_or_build(Path(args.cache))
    print("phase tables         :", "cache" if loaded else "built")
    print("table wall           :", f"{time.perf_counter()-t0:.3f}s")

    dr_targets = v57.local_dr_pose_sets()
    dist_tables = build_piece_distance_tables(dr_targets)
    print()

    rows = []

    # --------------------------------------------------------------
    # Bounded historical controls.
    # --------------------------------------------------------------
    for control in v57.CONTROLS:
        scramble_internal = internalize(
            control["scramble"],
            control["axis"],
        )

        start = ts.engine.from_word(
            " ".join(scramble_internal)
        )
        root_stats = local_stats(
            tuple(int(x) for x in start.poses),
            dist_tables,
        )
        root_q = int(ts.q_of(start))
        root_q_lb = int(
            ts.phase1_lb(
                root_q,
                qtables[3],
                qtables[4],
            )
        )

        print(
            f"## {control['name']} "
            f"axis={control['axis']} L<={control['max_len']}"
        )
        print(
            "root bounds          :",
            f"Q={root_q_lb} "
            f"pieceMax={root_stats['max_d']} "
            f"capacity={root_stats['capacity_lb']} "
            f"piece={root_stats['piece_lb']} "
            f"sumD={root_stats['sum_d']}",
        )

        control_modes = []

        for mode in modes:
            res = bounded_census(
                mode=mode,
                scramble_internal=scramble_internal,
                max_len=int(control["max_len"]),
                qtables=qtables,
                dist_tables=dist_tables,
            )

            external = [
                externalize(
                    x,
                    control["axis"],
                )
                for x in res.dr_words
            ]

            print(
                f"{mode:<15}: "
                f"nodes={res.nodes:,} "
                f"byDepth={res.by_depth} "
                f"DR={sum(res.dr_by_depth.values())} "
                f"prunes={res.prune_counts} "
                f"wall={res.wall:.4f}s",
            )

            if external:
                for word in external:
                    print("    ", " ".join(word))

            control_modes.append({
                "mode": mode,
                "nodes": res.nodes,
                "by_depth": res.by_depth,
                "dr_by_depth": res.dr_by_depth,
                "dr_words_original": [
                    list(x)
                    for x in external
                ],
                "prune_counts": res.prune_counts,
                "wall": res.wall,
            })

        print()
        rows.append({
            "control": control["name"],
            "axis": control["axis"],
            "max_len": int(control["max_len"]),
            "root_q_lb": root_q_lb,
            "root_piece_stats": root_stats,
            "modes": control_modes,
        })

    # --------------------------------------------------------------
    # M11 exact-depth7 positive holdout.
    # --------------------------------------------------------------
    m11 = next(
        c for c in v57.CONTROLS
        if c["name"] == "M11-01"
    )
    m11_internal = internalize(
        m11["scramble"],
        m11["axis"],
    )

    print("# M11 EXACT DEPTH-7 HOLDOUT")

    exact_rows = []

    for mode in ("Q_ONLY", "Q_PLUS_PIECE"):
        res = exact_depth_search(
            mode=mode,
            scramble_internal=m11_internal,
            depth_limit=7,
            qtables=qtables,
            dist_tables=dist_tables,
            node_cap=int(args.node_cap),
            time_cap=float(args.time_cap),
        )

        external = [
            list(
                externalize(
                    tuple(x),
                    m11["axis"],
                )
            )
            for x in res["solutions_internal"]
        ]

        expected = {
            tuple(x)
            for x in M11_EXACT_ORIGINAL
        }
        found = {
            tuple(x)
            for x in external
        }

        res["solutions_original"] = external
        res["matches_historical_exact_set"] = (
            found == expected
        )

        print(
            f"{mode:<15}: "
            f"nodes={res['nodes']:,} "
            f"solutions={len(external)} "
            f"prunes={res['prune_counts']} "
            f"wall={res['wall']:.4f}s "
            f"cut={res['cut_reason']}",
        )
        for word in external:
            print("    ", " ".join(word))

        exact_rows.append(res)

    print()
    print("# DECISION")

    qrow = next(x for x in exact_rows if x["mode"] == "Q_ONLY")
    prow = next(x for x in exact_rows if x["mode"] == "Q_PLUS_PIECE")

    bounded_sound = True

    for row in rows:
        qmode = next(x for x in row["modes"] if x["mode"] == "Q_ONLY")
        pmode = next(x for x in row["modes"] if x["mode"] == "Q_PLUS_PIECE")
        if qmode["dr_words_original"] != pmode["dr_words_original"]:
            bounded_sound = False

    exact_sound = (
        qrow["matches_historical_exact_set"]
        and prow["matches_historical_exact_set"]
        and prow["cut_reason"] is None
    )

    gain = (
        1.0 - prow["nodes"] / qrow["nodes"]
        if qrow["nodes"]
        else 0.0
    )

    if bounded_sound and exact_sound and gain > 0:
        decision = "DEPTH_CONDITIONED_PIECE_BOUND_PASS"
        note = (
            "The candidate-free 20-piece local-DR resource bound is sound on "
            "the bounded SAT/UNSAT controls and the M11 exact-7 positive holdout, "
            "while pruning additional nodes beyond phase1_lb."
        )
    elif bounded_sound and exact_sound:
        decision = "PIECE_BOUND_SOUND_BUT_Q_REDUNDANT"
        note = (
            "The candidate-free depth-conditioned bound is sound, but on these "
            "controls phase1_lb already dominates it at the node-expansion level."
        )
    else:
        decision = "DEPTH_CONDITIONED_PIECE_BOUND_FAIL"
        note = (
            "The supposedly admissible piece/resource bound changed a bounded "
            "DR solution set. Stop and fix the local-DR formulation."
        )

    print(decision)
    print(note)
    print(
        "M11 exact7 node reduction:",
        f"{gain:.2%}",
    )

    payload = {
        "version": "v37.60",
        "mode": "DEPTH_CONDITIONED_20_PIECE_LOCAL_DR_BOUND",
        "bounded_controls": rows,
        "m11_exact7": exact_rows,
        "decision": decision,
        "note": note,
        "m11_exact7_node_reduction": gain,
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
