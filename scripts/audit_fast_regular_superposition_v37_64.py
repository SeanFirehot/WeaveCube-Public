#!/usr/bin/env python3
"""
CubeLab v37.64 — 20-PIECE REGULAR-LANGUAGE SUPERPOSITION (GAC)

Core idea
---------
Do NOT enumerate piece candidate words.
Do NOT truncate candidate length.
Do NOT collapse candidates to face-count vectors.

For a search node with exactly R global columns remaining, introduce variables:

    M_0, M_1, ..., M_{R-1}

Each variable initially ranges over all 18 HTM moves.

For every piece p, impose one REGULAR constraint:

    starting from current pose s_p,
    after the SAME global word M_0..M_{R-1},
    the piece must be in its local-DR pose set.

Inactive moves are ordinary self-loops for that piece.

Therefore each piece constraint implicitly represents ALL possible active
skeleton lengths 0..R, including arbitrary temporary excursions, without
materializing candidate sequences.

Also impose the global reduced-word regular constraint:
    adjacent moves may not use the same face,
and the first future move must differ from the actual previous face.

Generalized arc consistency
---------------------------
For each regular constraint:
  * forward reachable states
  * backward co-reachable states
  * supported move values at every future column

Intersect supported move domains across:
    20 piece automata
    + reduced-word automaton

Repeat until fixpoint.

If any future column domain becomes empty:
    no common global suffix of exactly R columns exists.

At a DFS node, domain(M_0) is therefore a sound higher-order shared-event
branch filter.  It preserves:
    move order
    U/U'/U2 distinctions
    inactive gaps
    all candidate lengths 0..R
    20-piece simultaneous ownership

This is the closest current formulation to the user's original
"20-piece move candidates kept in superposition" idea.

Audit
-----
Root survey:
    v37.44 fixed 5 scrambles x UD/FB/RL
    at R = phase1_lb
    report:
        GAC feasibility
        first-column supported moves
        domain shrink by position

Exact controls:
    M12-02 exact depth 3
    M11-01 exact depth 4 UNSAT
    M11-01 exact depth 7 positive holdout

Compare:
    GAC_ONLY
    Q_ONLY
    Q_PLUS_GAC

Strong signal:
    exact DR word sets preserved
    Q_PLUS_GAC expands fewer nodes than Q_ONLY on M11 exact-7

No learned constants.
No production changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
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


NMOVES = len(ts.MOVE_ORDER)
ALL_MOVE_MASK = (1 << NMOVES) - 1
MOVE_RANK = {m: i for i, m in enumerate(ts.MOVE_ORDER)}
FACE_NAMES = ("U", "D", "R", "L", "F", "B")
FACE_INDEX = {f: i for i, f in enumerate(FACE_NAMES)}
NONE_FACE = 6

MOVE_FACE = tuple(FACE_INDEX[m[0]] for m in ts.MOVE_ORDER)

M11_EXACT_ORIGINAL = tuple(v60.M11_EXACT_ORIGINAL)
M12_EXACT_ORIGINAL = (
    ("B", "L", "B"),
    ("B", "L", "B'"),
)


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
    return tuple(axes.map_word(tuple(word), axes.AXES[axis]))


def externalize(word, axis):
    inv = inverse_face_map(axes.AXES[axis])
    return tuple(axes.map_word(tuple(word), inv))


def iter_move_indices(mask):
    m = int(mask)
    while m:
        low = m & -m
        idx = low.bit_length() - 1
        yield idx
        m ^= low


def mask_moves(mask):
    return tuple(
        ts.MOVE_ORDER[i]
        for i in iter_move_indices(mask)
    )


def pose_bit(pose):
    return 1 << int(pose)


def iter_pose_bits(mask):
    m = int(mask)
    while m:
        low = m & -m
        pose = low.bit_length() - 1
        yield pose
        m ^= low


def build_target_masks(dr_targets):
    masks = []
    for targets in dr_targets:
        mask = 0
        for pose in targets:
            mask |= pose_bit(pose)
        masks.append(mask)
    return tuple(masks)


def piece_forward_images():
    """
    image[piece][move][pose] -> next pose.
    Already available as POSE_NEXT, just normalized to tuples for faster loops.
    """
    return tuple(
        tuple(
            tuple(
                int(pg.POSE_NEXT[mi][p][pose])
                for pose in range(24)
            )
            for mi in range(NMOVES)
        )
        for p in range(20)
    )


POSE_TRANS = piece_forward_images()


def forward_pose_mask(piece_i, state_mask, move_mask):
    out = 0
    for pose in iter_pose_bits(state_mask):
        for mi in iter_move_indices(move_mask):
            out |= pose_bit(POSE_TRANS[piece_i][mi][pose])
    return out


def backward_pose_mask(piece_i, next_good_mask, move_mask):
    out = 0
    for pose in range(24):
        for mi in iter_move_indices(move_mask):
            nxt = POSE_TRANS[piece_i][mi][pose]
            if next_good_mask & pose_bit(nxt):
                out |= pose_bit(pose)
                break
    return out


def supported_moves_piece(
    piece_i,
    forward_mask,
    backward_next_mask,
    domain_mask,
):
    support = 0

    for mi in iter_move_indices(domain_mask):
        ok = False
        trans = POSE_TRANS[piece_i][mi]

        for pose in iter_pose_bits(forward_mask):
            if backward_next_mask & pose_bit(trans[pose]):
                ok = True
                break

        if ok:
            support |= 1 << mi

    return support


def piece_regular_supports(
    piece_i,
    start_pose,
    target_mask,
    domains,
):
    """
    Exact GAC support for one piece's deterministic automaton.
    """
    R = len(domains)

    forward = [0] * (R + 1)
    backward = [0] * (R + 1)

    forward[0] = pose_bit(start_pose)

    for k in range(R):
        forward[k + 1] = forward_pose_mask(
            piece_i,
            forward[k],
            domains[k],
        )
        if forward[k + 1] == 0:
            return None

    backward[R] = int(target_mask)

    for k in range(R - 1, -1, -1):
        backward[k] = backward_pose_mask(
            piece_i,
            backward[k + 1],
            domains[k],
        )
        if backward[k] == 0:
            return None

    if not (backward[0] & pose_bit(start_pose)):
        return None

    supports = []

    for k in range(R):
        sm = supported_moves_piece(
            piece_i,
            forward[k],
            backward[k + 1],
            domains[k],
        )
        if sm == 0:
            return None
        supports.append(sm)

    return tuple(supports)


def adjacency_transition(state, mi):
    """
    state = previous face 0..5 or NONE_FACE.
    Return next face state or -1 if adjacent same-face is illegal.
    """
    face = MOVE_FACE[mi]
    if state != NONE_FACE and face == state:
        return -1
    return face


def adjacency_supports(domains, previous_face):
    R = len(domains)

    forward = [0] * (R + 1)
    backward = [0] * (R + 1)

    start_state = (
        NONE_FACE
        if previous_face is None
        else FACE_INDEX[previous_face]
    )
    forward[0] = 1 << start_state

    for k in range(R):
        out = 0
        for state in range(7):
            if not (forward[k] & (1 << state)):
                continue

            for mi in iter_move_indices(domains[k]):
                nxt = adjacency_transition(state, mi)
                if nxt >= 0:
                    out |= 1 << nxt

        forward[k + 1] = out
        if out == 0:
            return None

    # Any last-face state is acceptable at the end.
    backward[R] = (1 << 6) - 1

    for k in range(R - 1, -1, -1):
        out = 0

        for state in range(7):
            for mi in iter_move_indices(domains[k]):
                nxt = adjacency_transition(state, mi)
                if nxt >= 0 and (backward[k + 1] & (1 << nxt)):
                    out |= 1 << state
                    break

        backward[k] = out
        if out == 0:
            return None

    if not (backward[0] & (1 << start_state)):
        return None

    supports = []

    for k in range(R):
        sm = 0

        for mi in iter_move_indices(domains[k]):
            ok = False

            for state in range(7):
                if not (forward[k] & (1 << state)):
                    continue

                nxt = adjacency_transition(state, mi)
                if nxt >= 0 and (backward[k + 1] & (1 << nxt)):
                    ok = True
                    break

            if ok:
                sm |= 1 << mi

        if sm == 0:
            return None

        supports.append(sm)

    return tuple(supports)


@dataclass(frozen=True)
class GACResult:
    feasible: bool
    domains: tuple[int, ...]
    iterations: int
    removed_values: int


class RegularSuperposition:
    def __init__(self, dr_targets):
        self.target_masks = build_target_masks(dr_targets)
        self.cache = {}

    def propagate(self, poses, rem, previous_face):
        """
        GAC fixpoint on exactly `rem` future global columns.
        """
        key = (
            tuple(int(x) for x in poses),
            int(rem),
            previous_face,
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        if rem == 0:
            feasible = all(
                self.target_masks[i] & pose_bit(poses[i])
                for i in range(20)
            )
            result = GACResult(
                feasible=bool(feasible),
                domains=(),
                iterations=0,
                removed_values=0,
            )
            self.cache[key] = result
            return result

        domains = [ALL_MOVE_MASK] * int(rem)
        initial_values = NMOVES * int(rem)

        iterations = 0

        while True:
            iterations += 1
            changed = False

            # Global reduced-word regular constraint.
            adj = adjacency_supports(
                tuple(domains),
                previous_face,
            )
            if adj is None:
                result = GACResult(
                    False,
                    tuple(domains),
                    iterations,
                    initial_values - sum(x.bit_count() for x in domains),
                )
                self.cache[key] = result
                return result

            for k in range(rem):
                nd = domains[k] & adj[k]
                if nd == 0:
                    result = GACResult(
                        False,
                        tuple(domains),
                        iterations,
                        initial_values - sum(x.bit_count() for x in domains),
                    )
                    self.cache[key] = result
                    return result
                if nd != domains[k]:
                    domains[k] = nd
                    changed = True

            # 20 piece regular-language constraints.
            for piece_i in range(20):
                supports = piece_regular_supports(
                    piece_i,
                    int(poses[piece_i]),
                    self.target_masks[piece_i],
                    tuple(domains),
                )

                if supports is None:
                    result = GACResult(
                        False,
                        tuple(domains),
                        iterations,
                        initial_values - sum(x.bit_count() for x in domains),
                    )
                    self.cache[key] = result
                    return result

                for k in range(rem):
                    nd = domains[k] & supports[k]

                    if nd == 0:
                        result = GACResult(
                            False,
                            tuple(domains),
                            iterations,
                            initial_values - sum(x.bit_count() for x in domains),
                        )
                        self.cache[key] = result
                        return result

                    if nd != domains[k]:
                        domains[k] = nd
                        changed = True

            if not changed:
                break

            # Small finite-domain fixpoint safeguard.
            if iterations > initial_values + 5:
                raise RuntimeError("GAC did not converge")

        result = GACResult(
            True,
            tuple(domains),
            iterations,
            initial_values - sum(x.bit_count() for x in domains),
        )
        self.cache[key] = result
        return result


def step(poses, q, move, qtables):
    mi = ts.MI[move]
    co, eo, sl = qtables[0], qtables[1], qtables[2]

    nposes = tuple(
        int(x)
        for x in pg.move_poses(poses, mi)
    )
    nq = int(
        ts.q_move(
            int(q),
            mi,
            co,
            eo,
            sl,
        )
    )
    return nposes, nq


def q_feasible(q, rem, qtables):
    return int(
        ts.phase1_lb(
            int(q),
            qtables[3],
            qtables[4],
        )
    ) <= int(rem)


def mode_gate(mode, poses, q, rem, last_face, qtables, gac):
    if mode in ("Q_ONLY", "Q_PLUS_GAC"):
        if not q_feasible(q, rem, qtables):
            return False, "q_lb", None

    if mode in ("GAC_ONLY", "Q_PLUS_GAC"):
        gres = gac.propagate(
            poses,
            rem,
            last_face,
        )
        if not gres.feasible:
            return False, "gac", gres
        return True, None, gres

    return True, None, None


@dataclass
class SearchResult:
    mode: str
    nodes: int
    solutions_internal: tuple[tuple[str, ...], ...]
    prune_counts: dict
    branch_rejects: int
    gac_calls: int
    gac_removed_values: int
    gac_iterations: int
    wall: float
    cut_reason: str | None


def exact_depth_search(
    *,
    mode,
    scramble_internal,
    depth_limit,
    qtables,
    gac,
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
    prunes = Counter()
    branch_rejects = 0
    gac_calls = 0
    gac_removed_values = 0
    gac_iterations = 0
    path = []
    solutions = []

    class Cut(Exception):
        pass

    def check_cut():
        if nodes >= int(node_cap):
            raise Cut("node_cap")
        if time.perf_counter() >= deadline:
            raise Cut("time_cap")

    def dfs(poses, q, depth, last_face):
        nonlocal nodes, branch_rejects
        nonlocal gac_calls, gac_removed_values, gac_iterations

        check_cut()
        rem = int(depth_limit) - depth

        ok, reason, gres = mode_gate(
            mode,
            poses,
            q,
            rem,
            last_face,
            qtables,
            gac,
        )

        if mode in ("GAC_ONLY", "Q_PLUS_GAC") and gres is not None:
            gac_calls += 1
            gac_removed_values += int(gres.removed_values)
            gac_iterations += int(gres.iterations)

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
                        f"q-goal / explicit DR mismatch mode={mode}"
                    )

                solutions.append(word)
            return

        if mode in ("GAC_ONLY", "Q_PLUS_GAC"):
            first_mask = int(gres.domains[0])
        else:
            first_mask = ALL_MOVE_MASK

        children = []

        for mi in iter_move_indices(first_mask):
            move = ts.MOVE_ORDER[mi]

            # Q_ONLY has no adjacency GAC, so enforce historical reduction here.
            if last_face is not None and move[0] == last_face:
                branch_rejects += 1
                continue

            nposes, nq = step(
                poses,
                q,
                move,
                qtables,
            )

            # Child feasibility avoids counting a child node that the same
            # constraint would reject immediately.
            ok2, reason2, child_gres = mode_gate(
                mode,
                nposes,
                nq,
                rem - 1,
                move[0],
                qtables,
                gac,
            )

            if mode in ("GAC_ONLY", "Q_PLUS_GAC") and child_gres is not None:
                gac_calls += 1
                gac_removed_values += int(child_gres.removed_values)
                gac_iterations += int(child_gres.iterations)

            if not ok2:
                prunes[reason2] += 1
                continue

            child_domain_size = (
                child_gres.domains[0].bit_count()
                if (
                    child_gres is not None
                    and child_gres.feasible
                    and child_gres.domains
                )
                else NMOVES
            )

            qlb = int(
                ts.phase1_lb(
                    nq,
                    qtables[3],
                    qtables[4],
                )
            )

            children.append((
                qlb,
                child_domain_size,
                MOVE_RANK[move],
                move,
                nposes,
                nq,
            ))

        # Branches excluded by GAC's first variable domain.
        if mode in ("GAC_ONLY", "Q_PLUS_GAC"):
            legal_reduced = sum(
                1
                for m in ts.MOVE_ORDER
                if last_face is None or m[0] != last_face
            )
            branch_rejects += max(
                0,
                legal_reduced - first_mask.bit_count(),
            )

        children.sort()

        for (
            _hq,
            _ds,
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

    return SearchResult(
        mode=mode,
        nodes=nodes,
        solutions_internal=tuple(sorted(set(solutions))),
        prune_counts=dict(prunes),
        branch_rejects=branch_rejects,
        gac_calls=gac_calls,
        gac_removed_values=gac_removed_values,
        gac_iterations=gac_iterations,
        wall=time.perf_counter() - t0,
        cut_reason=cut,
    )


def root_survey(p1_json, qtables, gac, max_horizon):
    path = Path(p1_json)

    if not path.is_file():
        return {
            "available": False,
            "reason": f"missing {path}",
            "rows": [],
        }

    obj = json.loads(
        path.read_text(encoding="utf-8")
    )

    rows = []
    gac_gt_q = 0
    gac_eq_q = 0
    root_infeasible_at_q = 0

    for block in obj.get("rows", [])[:5]:
        scramble = tuple(block["scramble"])
        case = int(block["case"])

        for axis in ("UD", "FB", "RL"):
            sw = internalize(
                scramble,
                axis,
            )
            st = ts.engine.from_word(
                " ".join(sw)
            )
            poses = tuple(int(x) for x in st.poses)
            q = int(ts.q_of(st))

            qlb = int(
                ts.phase1_lb(
                    q,
                    qtables[3],
                    qtables[4],
                )
            )

            # Find smallest exact horizon whose 20 regular constraints reach GAC
            # consistency. Arc consistency is necessary, not sufficient, so this
            # is an admissible lower-bound-like diagnostic only.
            hgac = None

            for r in range(max_horizon + 1):
                gres = gac.propagate(
                    poses,
                    r,
                    None,
                )
                if gres.feasible:
                    hgac = r
                    break

            at_q = (
                gac.propagate(poses, qlb, None)
                if qlb <= max_horizon
                else None
            )

            if hgac is not None and hgac > qlb:
                gac_gt_q += 1
            if hgac == qlb:
                gac_eq_q += 1
            if at_q is not None and not at_q.feasible:
                root_infeasible_at_q += 1

            rows.append({
                "case": case,
                "axis": axis,
                "q_lb": qlb,
                "gac_lb": hgac,
                "gac_gt_q": (
                    hgac is not None and hgac > qlb
                ),
                "gac_feasible_at_q": (
                    None if at_q is None else at_q.feasible
                ),
                "first_moves_at_q": (
                    None
                    if at_q is None or not at_q.feasible or not at_q.domains
                    else list(mask_moves(at_q.domains[0]))
                ),
                "first_move_count_at_q": (
                    None
                    if at_q is None or not at_q.feasible or not at_q.domains
                    else at_q.domains[0].bit_count()
                ),
                "domain_sizes_at_q": (
                    None
                    if at_q is None or not at_q.feasible
                    else [x.bit_count() for x in at_q.domains]
                ),
                "iterations_at_q": (
                    None if at_q is None else at_q.iterations
                ),
                "removed_values_at_q": (
                    None if at_q is None else at_q.removed_values
                ),
            })

    return {
        "available": True,
        "rows": rows,
        "gac_gt_q_count": gac_gt_q,
        "gac_eq_q_count": gac_eq_q,
        "root_infeasible_at_q_count": root_infeasible_at_q,
    }


def external_solution_set(result, axis):
    return {
        externalize(word, axis)
        for word in result.solutions_internal
    }


def run_control(
    name,
    scramble,
    axis,
    depth,
    expected_original,
    qtables,
    gac,
    node_cap,
    time_cap,
):
    sw = internalize(scramble, axis)

    rows = []

    print(f"## {name} axis={axis} exactDepth={depth}")

    for mode in ("GAC_ONLY", "Q_ONLY", "Q_PLUS_GAC"):
        res = exact_depth_search(
            mode=mode,
            scramble_internal=sw,
            depth_limit=int(depth),
            qtables=qtables,
            gac=gac,
            node_cap=int(node_cap),
            time_cap=float(time_cap),
        )

        ext = tuple(
            sorted(
                external_solution_set(
                    res,
                    axis,
                )
            )
        )

        expected = set(expected_original)
        found = set(ext)

        match = (
            found == expected
            and res.cut_reason is None
        )

        print(
            f"{mode:<12}: "
            f"nodes={res.nodes:,} "
            f"solutions={len(ext)} "
            f"prunes={res.prune_counts} "
            f"branchReject={res.branch_rejects:,} "
            f"GACcalls={res.gac_calls:,} "
            f"removed={res.gac_removed_values:,} "
            f"wall={res.wall:.4f}s "
            f"cut={res.cut_reason}"
        )

        for word in ext:
            print("    ", " ".join(word))

        rows.append({
            "mode": mode,
            "nodes": res.nodes,
            "solutions_original": [list(x) for x in ext],
            "matches_expected": match,
            "prune_counts": res.prune_counts,
            "branch_rejects": res.branch_rejects,
            "gac_calls": res.gac_calls,
            "gac_removed_values": res.gac_removed_values,
            "gac_iterations": res.gac_iterations,
            "wall": res.wall,
            "cut_reason": res.cut_reason,
        })

    print()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-horizon",
        type=int,
        default=10,
    )
    ap.add_argument(
        "--cache",
        default="reports/pdcc_cache/twist_skeleton_tables_v1.pkl",
    )
    ap.add_argument(
        "--p1-json",
        default="reports/v37/axis_dr_p1_fast_v37_44.json",
    )
    ap.add_argument("--node-cap", type=int, default=1_000_000)
    ap.add_argument("--time-cap", type=float, default=4.0)
    ap.add_argument(
        "--output",
        default="reports/v37/regular_language_superposition_v37_64.json",
    )
    args = ap.parse_args()

    max_horizon = max(7, min(12, int(args.max_horizon)))

    print("# CubeLab v37.64 - 20-PIECE REGULAR-LANGUAGE SUPERPOSITION")
    print("future horizon       :", f"0..{max_horizon}")
    print("candidate words      : IMPLICIT / ALL")
    print("candidate length cap : ONLY global remaining horizon")
    print("inactive gaps        : SELF-LOOPS")
    print("move order           : PRESERVED")
    print("turn variants        : PRESERVED")
    print("propagation          : GAC fixpoint over 20 regular constraints")
    print("production changes   : NONE")
    print()

    t0 = time.perf_counter()
    qtables, loaded = ts.cache_load_or_build(
        Path(args.cache)
    )
    print("phase tables         :", "cache" if loaded else "built")
    print("table wall           :", f"{time.perf_counter()-t0:.3f}s")

    dr_targets = v57.local_dr_pose_sets()
    gac = RegularSuperposition(dr_targets)
    print()

    survey = root_survey(
        args.p1_json,
        qtables,
        gac,
        max_horizon,
    )

    print("# ROOT GAC COMPLEMENTARITY SURVEY")
    if survey["available"]:
        print(
            "GAC-LB > Q           :",
            f"{survey['gac_gt_q_count']}/15",
        )
        print(
            "GAC-LB == Q          :",
            f"{survey['gac_eq_q_count']}/15",
        )
        print(
            "GAC infeasible @ Q   :",
            f"{survey['root_infeasible_at_q_count']}/15",
        )

        for row in survey["rows"]:
            print(
                f"case={row['case']} {row['axis']}: "
                f"Q={row['q_lb']} "
                f"GAC={row['gac_lb']} "
                f"first={row['first_move_count_at_q']} "
                f"moves={row['first_moves_at_q']} "
                f"domains={row['domain_sizes_at_q']}"
            )
    else:
        print("SKIP:", survey["reason"])
    print()

    # M12 exact depth 3.
    m12 = next(
        c for c in v57.CONTROLS
        if c["name"] == "M12-02"
    )
    m12_rows = run_control(
        "M12-02",
        tuple(m12["scramble"]),
        m12["axis"],
        3,
        M12_EXACT_ORIGINAL,
        qtables,
        gac,
        args.node_cap,
        args.time_cap,
    )

    # M11 exact depth 4 UNSAT.
    m11 = next(
        c for c in v57.CONTROLS
        if c["name"] == "M11-01"
    )
    m11_unsat_rows = run_control(
        "M11-01-UNSAT4",
        tuple(m11["scramble"]),
        m11["axis"],
        4,
        (),
        qtables,
        gac,
        args.node_cap,
        args.time_cap,
    )

    # M11 exact depth 7 positive.
    m11_exact_rows = run_control(
        "M11-01-EXACT7",
        tuple(m11["scramble"]),
        m11["axis"],
        7,
        M11_EXACT_ORIGINAL,
        qtables,
        gac,
        args.node_cap,
        args.time_cap,
    )

    print("# DECISION")

    all_rows = (
        m12_rows
        + m11_unsat_rows
        + m11_exact_rows
    )
    sound = all(
        row["matches_expected"]
        for row in all_rows
    )

    q = next(
        row for row in m11_exact_rows
        if row["mode"] == "Q_ONLY"
    )
    qg = next(
        row for row in m11_exact_rows
        if row["mode"] == "Q_PLUS_GAC"
    )
    g = next(
        row for row in m11_exact_rows
        if row["mode"] == "GAC_ONLY"
    )

    gain = (
        1.0 - qg["nodes"] / q["nodes"]
        if q["nodes"]
        else 0.0
    )

    if not sound:
        decision = "REGULAR_SUPERPOSITION_FAIL"
        note = (
            "The all-length regular-language propagation changed a known exact "
            "DR solution set. Fix the automaton/GAC formulation."
        )
    elif gain > 0:
        decision = "REGULAR_SUPERPOSITION_INDEPENDENT_SIGNAL"
        note = (
            "The 20-piece all-length, order-preserving regular-language "
            "superposition is sound and removes exact-depth nodes beyond "
            "phase1_lb."
        )
    else:
        decision = "REGULAR_SUPERPOSITION_SOUND_BUT_Q_NODE_REDUNDANT"
        note = (
            "The full all-length/order-preserving superposition is sound, but "
            "on this exact-7 control phase1_lb reaches the same expanded-node "
            "frontier. Inspect first-column/domain shrink before deciding whether "
            "to proceed to stronger relational consistency."
        )

    print(decision)
    print(note)
    print("exact7 node gain     :", f"{gain:.2%}")
    print(
        "GAC cache entries    :",
        len(gac.cache),
    )

    payload = {
        "version": "v37.64",
        "mode": "20_PIECE_REGULAR_LANGUAGE_SUPERPOSITION_GAC",
        "max_horizon": max_horizon,
        "root_survey": survey,
        "controls": {
            "M12_exact3": m12_rows,
            "M11_unsat4": m11_unsat_rows,
            "M11_exact7": m11_exact_rows,
        },
        "decision": decision,
        "note": note,
        "exact7_node_gain": gain,
        "gac_cache_entries": len(gac.cache),
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
