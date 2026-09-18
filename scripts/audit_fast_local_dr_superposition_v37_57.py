#!/usr/bin/env python3
"""
CubeLab v37.57 — INDEPENDENT LOCAL-DR DOMAIN COMPRESSION AUDIT

Motivation
----------
v37.56 proved exact bounded SAT/UNSAT preservation when each piece domain was
built from projections of the COMPLETE bounded global-word universe.

That was a semantic success but not yet a search-compression success:
    DOMAIN rejects = 0
    domain search space = direct search space

This audit removes that circular source.

Candidate domains are now generated INDEPENDENTLY PER PIECE from local DR
geometry only.

For each piece:
  1. Build its exact set of DR-compatible poses as the orbit of the solved
     pose under the standard Phase-2 / K subgroup moves.
  2. From the scrambled pose, compute the shortest ACTIVE-ONLY distance to
     that local DR pose set.
  3. Enumerate every active-only skeleton up to:
         shortest + slack
     but never beyond the bounded global control depth.
  4. No global solution word is used to generate or inject candidates.

Then search global reduced words under the 20 simultaneous domains.

Controls
--------
M12-02, UD, <=3:
    direct DR ground truth contains:
        B L B
        B L B'

M11-01, FB, <=4:
    complete reduced-word universe = 65,089
    direct DR ground truth = UNSAT

Slack ladder
------------
0, 1, 2 by default.

Questions
---------
* Can independent local shortest/near-shortest domains recover M12 SAT?
* How much of the 65,089-word M11 universe do they eliminate?
* What is the smallest slack that preserves all bounded M12 DR witnesses?
* Does EARLY_COLLAPSE still lose compatibility while the joint domains retain it?

A strong result is:
    M12 recall = 100%
    M11 remains UNSAT
    domain search space << direct search space
without any witness injection.

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


CONTROLS = (
    {
        "name": "M12-02",
        "scramble": (
            "F2", "U", "B2", "R2", "D", "L2",
            "D", "L2", "U'", "B'", "L'", "B'",
        ),
        "axis": "UD",
        "max_len": 3,
        "known_direct_original": (
            ("B", "L", "B"),
            ("B", "L", "B'"),
        ),
    },
    {
        "name": "M11-01",
        "scramble": (
            "D", "F'", "R2", "D2", "L2", "U2",
            "B", "R2", "U2", "B'", "D'",
        ),
        "axis": "FB",
        "max_len": 4,
        "known_direct_original": (),
    },
)

MOVE_RANK = {m: i for i, m in enumerate(ts.MOVE_ORDER)}


def inverse_face_map(fmap):
    return {dst: src for src, dst in fmap.items()}


def internalize(control):
    fmap = axes.AXES[control["axis"]]
    inv = inverse_face_map(fmap)
    scramble = tuple(axes.map_word(control["scramble"], fmap))
    known = tuple(
        tuple(axes.map_word(word, fmap))
        for word in control["known_direct_original"]
    )
    return scramble, known, inv


def externalize(word, inv):
    return tuple(axes.map_word(tuple(word), inv))


def expected_word_counts(max_len):
    counts = {0: 1}
    n = 1
    total = 1
    for d in range(1, max_len + 1):
        n = 18 if d == 1 else n * 15
        counts[d] = n
        total += n
    return counts, total


def move_piece_pose(piece_index, pose, move):
    return int(
        pg.POSE_NEXT[
            ts.MI[move]
        ][piece_index][int(pose)]
    )


def local_dr_pose_sets():
    """Exact per-piece DR-compatible pose orbit under K subgroup."""
    out = []

    for piece_i in range(20):
        start = int(ts.SOLVED.poses[piece_i])
        seen = {start}
        q = deque([start])

        while q:
            pose = q.popleft()
            for move in ts.K_MOVES:
                nxt = move_piece_pose(piece_i, pose, move)
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)

        out.append(frozenset(seen))

    return tuple(out)


def reverse_distance_to_targets(piece_i, targets):
    rev = [[] for _ in range(24)]

    for pose in range(24):
        for move in ts.MOVE_ORDER:
            nxt = move_piece_pose(piece_i, pose, move)
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

    return tuple(dist)


def enumerate_local_candidates(
    piece_i,
    start_pose,
    targets,
    *,
    slack,
    max_active_len,
):
    dist = reverse_distance_to_targets(piece_i, targets)
    d0 = int(dist[int(start_pose)])

    if d0 >= 999:
        raise RuntimeError(
            f"piece {piece_i}: no active path to local DR orbit"
        )

    upper = min(
        int(max_active_len),
        d0 + int(slack),
    )

    rows = []

    def dfs(pose, rem, prefix):
        if int(dist[int(pose)]) > rem:
            return

        if rem == 0:
            if int(pose) in targets:
                rows.append(tuple(prefix))
            return

        for move in ts.MOVE_ORDER:
            nxt = move_piece_pose(piece_i, pose, move)
            if nxt == pose:
                continue
            if int(dist[nxt]) > rem - 1:
                continue
            dfs(
                nxt,
                rem - 1,
                prefix + (move,),
            )

    for length in range(d0, upper + 1):
        dfs(
            int(start_pose),
            length,
            (),
        )

    rows = sorted(
        set(rows),
        key=lambda seq: (
            len(seq),
            tuple(MOVE_RANK[m] for m in seq),
        ),
    )

    if not rows:
        raise RuntimeError(
            f"piece {piece_i}: local candidate domain unexpectedly empty"
        )

    return tuple(rows), {
        "shortest": d0,
        "upper": upper,
        "target_pose_count": len(targets),
    }


def build_indexes(domains):
    next_index = []
    complete_masks = []

    for domain in domains:
        max_len = max(len(seq) for seq in domain)

        rows = []
        for k in range(max_len + 1):
            by_move = {
                move: 0
                for move in ts.MOVE_ORDER
            }
            for ci, seq in enumerate(domain):
                if k < len(seq):
                    by_move[seq[k]] |= 1 << ci
            rows.append(by_move)

        cm = {}
        for k in range(max_len + 1):
            mask = 0
            for ci, seq in enumerate(domain):
                if len(seq) == k:
                    mask |= 1 << ci
            cm[k] = mask

        next_index.append(tuple(rows))
        complete_masks.append(cm)

    return tuple(next_index), tuple(complete_masks)


def all_masks(domains):
    return tuple(
        (1 << len(domain)) - 1
        for domain in domains
    )


def early_masks(domains):
    masks = []
    choices = []

    for domain in domains:
        ci = min(
            range(len(domain)),
            key=lambda x: (
                len(domain[x]),
                tuple(MOVE_RANK[m] for m in domain[x]),
            ),
        )
        choices.append(ci)
        masks.append(1 << ci)

    return tuple(masks), tuple(choices)


def complete_filter(complete_masks, masks, ordinals):
    for i, (mask, k) in enumerate(zip(masks, ordinals)):
        cm = complete_masks[i].get(int(k), 0)
        if int(mask) & int(cm) == 0:
            return False
    return True


def propagate(
    poses,
    masks,
    ordinals,
    move,
    next_index,
):
    mi = ts.MI[move]
    nposes = tuple(
        int(x)
        for x in pg.move_poses(poses, mi)
    )
    nmasks = list(masks)
    nord = list(ordinals)

    eliminated = 0

    for i, (before, after) in enumerate(zip(poses, nposes)):
        if int(before) == int(after):
            continue

        k = int(ordinals[i])
        if k >= len(next_index[i]):
            return None

        allowed = int(next_index[i][k][move])
        nm = int(masks[i]) & allowed

        if nm == 0:
            return None

        eliminated += int(masks[i]).bit_count() - nm.bit_count()
        nmasks[i] = nm
        nord[i] = k + 1

    return (
        nposes,
        tuple(nmasks),
        tuple(nord),
        eliminated,
    )


@dataclass
class DirectResult:
    words_by_depth: dict
    dr_by_depth: dict
    dr_words: tuple
    wall: float


def direct_ground_truth(control, tables):
    co, eo, sl, *_ = tables

    scramble, known_internal, inv = internalize(control)
    start = ts.engine.from_word(" ".join(scramble))
    start_poses = tuple(int(x) for x in start.poses)
    q0 = int(ts.q_of(start))

    words = Counter()
    dr = Counter()
    dr_words = []

    def dfs(word, poses, q, depth, last_face):
        words[depth] += 1

        if int(q) == int(ts.GOAL_Q):
            dr[depth] += 1
            dr_words.append(tuple(word))

        if depth >= int(control["max_len"]):
            return

        for move in ts.MOVE_ORDER:
            if last_face is not None and move[0] == last_face:
                continue

            nposes = tuple(
                int(x)
                for x in pg.move_poses(poses, ts.MI[move])
            )
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
        start_poses,
        q0,
        0,
        None,
    )
    wall = time.perf_counter() - t0

    expected_depth, expected_total = expected_word_counts(
        int(control["max_len"])
    )
    if dict(sorted(words.items())) != expected_depth:
        raise RuntimeError(
            f"{control['name']}: direct word counts mismatch"
        )

    known_set = set(known_internal)
    direct_set = set(dr_words)

    if known_set and not known_set.issubset(direct_set):
        raise RuntimeError(
            f"{control['name']}: historical known DR words missing "
            f"from direct search"
        )

    return {
        "scramble": scramble,
        "inverse_map": inv,
        "start_poses": start_poses,
        "q0": q0,
        "known_internal": known_internal,
        "result": DirectResult(
            words_by_depth=dict(sorted(words.items())),
            dr_by_depth=dict(sorted(dr.items())),
            dr_words=tuple(dr_words),
            wall=wall,
        ),
    }


@dataclass
class DomainCensus:
    accepted_by_depth: dict
    dr_by_depth: dict
    dr_words: tuple
    nodes: int
    domain_rejects: int
    candidate_eliminations: int
    wall: float


def domain_census(
    control,
    direct,
    tables,
    domains,
    next_index,
    complete_masks,
    start_masks,
):
    co, eo, sl, *_ = tables

    accepted = Counter()
    dr = Counter()
    dr_words = []

    nodes = 0
    rejects = 0
    eliminations = 0

    def dfs(word, poses, q, masks, ords, depth, last_face):
        nonlocal nodes, rejects, eliminations

        nodes += 1
        accepted[depth] += 1

        if int(q) == int(ts.GOAL_Q):
            if complete_filter(
                complete_masks,
                masks,
                ords,
            ):
                dr[depth] += 1
                dr_words.append(tuple(word))

        if depth >= int(control["max_len"]):
            return

        for move in ts.MOVE_ORDER:
            if last_face is not None and move[0] == last_face:
                continue

            p = propagate(
                poses,
                masks,
                ords,
                move,
                next_index,
            )

            if p is None:
                rejects += 1
                continue

            nposes, nmasks, nord, elim = p
            eliminations += elim

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
                nmasks,
                nord,
                depth + 1,
                move[0],
            )

    t0 = time.perf_counter()
    dfs(
        (),
        direct["start_poses"],
        direct["q0"],
        tuple(start_masks),
        (0,) * 20,
        0,
        None,
    )

    return DomainCensus(
        accepted_by_depth=dict(sorted(accepted.items())),
        dr_by_depth=dict(sorted(dr.items())),
        dr_words=tuple(dr_words),
        nodes=nodes,
        domain_rejects=rejects,
        candidate_eliminations=eliminations,
        wall=time.perf_counter() - t0,
    )


def product_log10(domains):
    return sum(
        math.log10(len(domain))
        for domain in domains
    )


def median(vals):
    vals = sorted(vals)
    n = len(vals)
    if n % 2:
        return float(vals[n // 2])
    return (vals[n // 2 - 1] + vals[n // 2]) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--slacks",
        nargs="+",
        type=int,
        default=[0, 1, 2],
    )
    ap.add_argument(
        "--cache",
        default="reports/pdcc_cache/twist_skeleton_tables_v1.pkl",
    )
    ap.add_argument(
        "--output",
        default="reports/v37/local_dr_domain_compression_v37_57.json",
    )
    args = ap.parse_args()

    slacks = sorted(set(max(0, int(x)) for x in args.slacks))

    print("# CubeLab v37.57 - INDEPENDENT LOCAL-DR DOMAIN COMPRESSION")
    print("controls             : M12-02 SAT<=3 / M11-01 UNSAT<=4")
    print("domain source        : INDEPENDENT per-piece local DR geometry")
    print("global witness inject: NONE")
    print("source correlation   : NONE")
    print("slack ladder         :", slacks)
    print("production changes   : NONE")
    print()

    t0 = time.perf_counter()
    tables, loaded = ts.cache_load_or_build(Path(args.cache))
    print("phase tables         :", "cache" if loaded else "built")
    print("table wall           :", f"{time.perf_counter()-t0:.3f}s")

    dr_targets = local_dr_pose_sets()
    print(
        "local DR pose counts :",
        [len(x) for x in dr_targets],
    )
    print()

    payload_rows = []
    global_best_slack = None
    total_t0 = time.perf_counter()

    for control in CONTROLS:
        print(
            f"## {control['name']} "
            f"axis={control['axis']} maxLen={control['max_len']}"
        )

        direct = direct_ground_truth(
            control,
            tables,
        )
        dres = direct["result"]
        direct_total = sum(dres.words_by_depth.values())
        direct_dr_set = set(dres.dr_words)

        print("direct words/depth  :", dres.words_by_depth)
        print("direct total         :", direct_total)
        print("direct DR/depth      :", dres.dr_by_depth)
        print("direct wall          :", f"{dres.wall:.4f}s")

        if dres.dr_words:
            print("direct DR words:")
            for word in dres.dr_words:
                print(
                    "   ",
                    " ".join(
                        externalize(
                            word,
                            direct["inverse_map"],
                        )
                    ),
                )

        control_rows = []
        first_full_recall = None

        for slack in slacks:
            build_t0 = time.perf_counter()

            domains = []
            meta = []

            for i in range(20):
                rows, m = enumerate_local_candidates(
                    i,
                    int(direct["start_poses"][i]),
                    dr_targets[i],
                    slack=slack,
                    max_active_len=int(control["max_len"]),
                )
                domains.append(rows)
                meta.append(m)

            domains = tuple(domains)
            next_index, complete_masks = build_indexes(domains)
            build_wall = time.perf_counter() - build_t0

            sizes = [len(x) for x in domains]
            root_log = product_log10(domains)

            joint = domain_census(
                control,
                direct,
                tables,
                domains,
                next_index,
                complete_masks,
                all_masks(domains),
            )

            emasks, echoices = early_masks(domains)
            early = domain_census(
                control,
                direct,
                tables,
                domains,
                next_index,
                complete_masks,
                emasks,
            )

            joint_set = set(joint.dr_words)
            early_set = set(early.dr_words)

            recall = (
                1.0
                if not direct_dr_set
                else len(joint_set & direct_dr_set) / len(direct_dr_set)
            )
            extra = joint_set - direct_dr_set
            missed = direct_dr_set - joint_set

            space_ratio = (
                joint.nodes / direct_total
                if direct_total
                else 1.0
            )
            reduction = 1.0 - space_ratio

            if (
                direct_dr_set
                and recall == 1.0
                and first_full_recall is None
            ):
                first_full_recall = slack

            print(
                f"slack={slack}: "
                f"domains min/med/max={min(sizes)}/{median(sizes):.1f}/{max(sizes)} "
                f"sum={sum(sizes)} "
                f"log10|D|={root_log:.2f} "
                f"build={build_wall:.4f}s"
            )
            print(
                f"    JOINT accepted={joint.nodes:,}/{direct_total:,} "
                f"({space_ratio:.2%}) "
                f"pruned={reduction:.2%} "
                f"rejects={joint.domain_rejects:,} "
                f"DR={sum(joint.dr_by_depth.values())} "
                f"recall={recall:.1%} "
                f"extra={len(extra)} "
                f"wall={joint.wall:.4f}s"
            )
            print(
                f"    EARLY accepted={early.nodes:,}/{direct_total:,} "
                f"DR={sum(early.dr_by_depth.values())} "
                f"recall={(
                    1.0 if not direct_dr_set
                    else len(early_set & direct_dr_set) / len(direct_dr_set)
                ):.1%} "
                f"wall={early.wall:.4f}s"
            )

            if missed:
                print("    missed joint DR:")
                for word in sorted(missed):
                    print(
                        "      ",
                        " ".join(
                            externalize(
                                word,
                                direct["inverse_map"],
                            )
                        ),
                    )

            control_rows.append({
                "slack": slack,
                "candidate_build_wall": build_wall,
                "domain_sizes": sizes,
                "domain_sum": sum(sizes),
                "root_log10_product_width": root_log,
                "piece_meta": meta,
                "joint": {
                    "accepted_by_depth": joint.accepted_by_depth,
                    "accepted_total": joint.nodes,
                    "search_space_ratio": space_ratio,
                    "search_space_reduction": reduction,
                    "domain_rejects": joint.domain_rejects,
                    "candidate_eliminations": joint.candidate_eliminations,
                    "dr_by_depth": joint.dr_by_depth,
                    "dr_words_internal": [list(x) for x in joint.dr_words],
                    "dr_words_original": [
                        list(
                            externalize(
                                x,
                                direct["inverse_map"],
                            )
                        )
                        for x in joint.dr_words
                    ],
                    "recall": recall,
                    "extra_dr_words": [list(x) for x in sorted(extra)],
                    "missed_dr_words": [list(x) for x in sorted(missed)],
                    "wall": joint.wall,
                },
                "early": {
                    "accepted_total": early.nodes,
                    "dr_by_depth": early.dr_by_depth,
                    "dr_words_internal": [list(x) for x in early.dr_words],
                    "wall": early.wall,
                },
            })

        print("first full-recall slack:", first_full_recall)
        print()

        if (
            control["name"] == "M12-02"
            and first_full_recall is not None
        ):
            if global_best_slack is None:
                global_best_slack = first_full_recall
            else:
                global_best_slack = max(
                    global_best_slack,
                    first_full_recall,
                )

        payload_rows.append({
            "control": control,
            "direct": {
                "words_by_depth": dres.words_by_depth,
                "total": direct_total,
                "dr_by_depth": dres.dr_by_depth,
                "dr_words_internal": [list(x) for x in dres.dr_words],
                "dr_words_original": [
                    list(
                        externalize(
                            x,
                            direct["inverse_map"],
                        )
                    )
                    for x in dres.dr_words
                ],
                "wall": dres.wall,
            },
            "slack_rows": control_rows,
            "first_full_recall_slack": first_full_recall,
        })

    total_wall = time.perf_counter() - total_t0

    print("# GLOBAL DECISION")

    m12 = next(
        row for row in payload_rows
        if row["control"]["name"] == "M12-02"
    )
    m11 = next(
        row for row in payload_rows
        if row["control"]["name"] == "M11-01"
    )

    successful = []

    for r12 in m12["slack_rows"]:
        if r12["joint"]["recall"] < 1.0:
            continue

        same_slack = next(
            r for r in m11["slack_rows"]
            if r["slack"] == r12["slack"]
        )

        if sum(same_slack["joint"]["dr_by_depth"].values()) != 0:
            continue

        successful.append((
            r12["slack"],
            r12["joint"]["search_space_reduction"],
            same_slack["joint"]["search_space_reduction"],
        ))

    if successful:
        successful.sort(
            key=lambda x: (
                x[0],
                -(x[1] + x[2]),
            )
        )
        best = successful[0]
        decision = "LOCAL_DR_SUPERPOSITION_COMPRESSION_PASS"
        note = (
            f"Independent per-piece DR domains preserve complete M12 SAT recall "
            f"and M11 UNSAT at slack={best[0]}, while pruning the bounded global "
            f"word universe by {best[1]:.1%} / {best[2]:.1%} respectively."
        )
    else:
        decision = "LOCAL_DR_SUPERPOSITION_RECALL_FAIL"
        note = (
            "No tested local slack simultaneously preserves all bounded M12 DR "
            "witnesses and the M11 UNSAT control. The local-domain construction "
            "must be refined before larger composition research."
        )

    print(decision)
    print(note)
    print("total wall           :", f"{total_wall:.3f}s")

    payload = {
        "version": "v37.57",
        "mode": "INDEPENDENT_LOCAL_DR_DOMAIN_COMPRESSION",
        "slacks": slacks,
        "local_dr_pose_counts": [len(x) for x in dr_targets],
        "rows": payload_rows,
        "decision": decision,
        "note": note,
        "total_wall": total_wall,
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
