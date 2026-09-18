#!/usr/bin/env python3
"""
CubeLab v37.80 — COLUMN-COMPLETION × 20-PIECE SUPERPOSITION PILOT

Principle
---------
Superposition is the STATE REPRESENTATION.
Column completion is the SEARCH OPERATOR.

Never choose one candidate trajectory for a piece.

At a Q-surviving prefix with R future global columns:

    C0, C1, ... C(R-1)

are the only search variables.

Each column initially has all 18 HTM moves.

For every one of the 20 pieces, keep the COMPLETE regular language of all
length-R global words that:

    * start from that piece's current pose,
    * include arbitrary inactive self-loop columns,
    * preserve move order and U/U'/U2 distinctions,
    * finish that piece in its local-DR target set.

Thus all piece candidate active-skeleton lengths 0..R remain implicit.

Propagation
-----------
Apply generalized arc consistency across:

    * global reduced-word adjacency constraint,
    * all 20 piece regular-language constraints.

This is exactly the v37.64 superposition representation.

But DO NOT stop when GAC reaches a fixpoint.

Column completion
-----------------
If GAC leaves multiple values:

    1. choose ONE GLOBAL COLUMN variable;
    2. assign one move value;
    3. rerun 20-piece propagation;
    4. recurse.

No piece candidate is ever collapsed independently.

Two column-choice policies:

    LTR : earliest unresolved column
    MRV : most constrained unresolved column (fewest supported moves)

MRV may assign a future column non-chronologically.  Piece automata preserve
the cross-column ordering correlations during propagation.

At singleton completion:
    replay the complete global suffix on the real cube;
    accept only if explicit DR is reached.

Therefore:
    * CSP UNSAT is a sound DR prune;
    * relaxed local-DR SAT that is not real DR is not accepted; search continues.

Experiment
----------
Historical positive control:
    M11 exact-7 known dead prefix U F D' R' (remaining=3)

Fresh controls:
    automatically generate one fresh exact-7 and one fresh exact-8 control
    that each contain >=3 nonterminal Q-surviving DEAD prefixes with
    remaining depth <=4.

For each control sample:
    * up to 3 Q-DEAD prefixes
    * 1 Q-LIVE prefix

Compare per prefix:

    Q-SUBTREE
        chronological Q search, first solution or complete UNSAT proof

    COLUMN-LTR
        full 20-piece superposition + chronological column completion

    COLUMN-MRV
        full 20-piece superposition + MRV column completion

Also report whether root GAC alone detects the conflict.

This directly tests the hypothesis suggested by the user:

    GAC alone may be weak,
    but COLUMN COMPLETION over the superposed candidate languages may recover
    the missing global incompatibility.

FAST diagnostic only.
No production changes.
"""

from __future__ import annotations

import argparse
import json
import random
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

import audit_fast_local_dr_superposition_v37_57 as v57
import audit_fast_regular_superposition_v37_64 as v64
import audit_fast_requirement_nonuniform_v37_70 as v70
import audit_fast_fresh_exact5_exact8_v37_73 as v73
import audit_fast_broad_gac_opportunity_v37_79 as v79


NMOVES = len(ts.MOVE_ORDER)
ALL_MOVE_MASK = (1 << NMOVES) - 1
MOVE_RANK = {m: i for i, m in enumerate(ts.MOVE_ORDER)}
PIECE_NAMES = tuple(v64.PIECE_NAMES)

HIST_PREFIX_EXTERNAL = ("U", "F", "D'", "R'")


def iter_move_indices(mask):
    m = int(mask)
    while m:
        low = m & -m
        i = low.bit_length() - 1
        yield i
        m ^= low


def singleton_move(mask):
    if int(mask).bit_count() != 1:
        raise ValueError("not singleton")
    i = int(mask).bit_length() - 1
    return ts.MOVE_ORDER[i]


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


@dataclass(frozen=True)
class PropagationResult:
    feasible: bool
    domains: tuple[int, ...]
    iterations: int
    removed_values: int


class ColumnSuperposition:
    def __init__(self, dr_targets):
        self.target_masks = v64.build_target_masks(
            dr_targets
        )

    def propagate(
        self,
        poses,
        previous_face,
        domains,
    ):
        domains = [
            int(x)
            for x in domains
        ]

        R = len(domains)
        initial_values = sum(
            x.bit_count()
            for x in domains
        )
        iterations = 0

        while True:
            iterations += 1
            changed = False

            adj = v64.adjacency_supports(
                tuple(domains),
                previous_face,
            )

            if adj is None:
                return PropagationResult(
                    False,
                    tuple(domains),
                    iterations,
                    initial_values
                    - sum(
                        x.bit_count()
                        for x in domains
                    ),
                )

            for k in range(R):
                nd = domains[k] & int(
                    adj[k]
                )

                if nd == 0:
                    return PropagationResult(
                        False,
                        tuple(domains),
                        iterations,
                        initial_values
                        - sum(
                            x.bit_count()
                            for x in domains
                        ),
                    )

                if nd != domains[k]:
                    domains[k] = nd
                    changed = True

            for piece_i in range(20):
                supports = (
                    v64.piece_regular_supports(
                        piece_i,
                        int(poses[piece_i]),
                        self.target_masks[
                            piece_i
                        ],
                        tuple(domains),
                    )
                )

                if supports is None:
                    return PropagationResult(
                        False,
                        tuple(domains),
                        iterations,
                        initial_values
                        - sum(
                            x.bit_count()
                            for x in domains
                        ),
                    )

                for k in range(R):
                    nd = (
                        domains[k]
                        & int(
                            supports[k]
                        )
                    )

                    if nd == 0:
                        return PropagationResult(
                            False,
                            tuple(domains),
                            iterations,
                            initial_values
                            - sum(
                                x.bit_count()
                                for x in domains
                            ),
                        )

                    if nd != domains[k]:
                        domains[k] = nd
                        changed = True

            if not changed:
                break

            # Every successful iteration removes at least one finite move value.
            if iterations > (
                initial_values + 5
            ):
                raise RuntimeError(
                    "column-superposition GAC did not converge"
                )

        return PropagationResult(
            True,
            tuple(domains),
            iterations,
            initial_values
            - sum(
                x.bit_count()
                for x in domains
            ),
        )


@dataclass
class ColumnSolveResult:
    status: str
    word: tuple[str, ...] | None
    csp_nodes: int
    propagation_calls: int
    propagation_iterations: int
    removed_values: int
    assignments_checked: int
    relaxed_false_sat: int
    memo_hits: int
    wall: float
    cut_reason: str | None


def choose_column(domains, policy):
    unresolved = [
        (k, int(mask).bit_count())
        for k, mask in enumerate(
            domains
        )
        if int(mask).bit_count() > 1
    ]

    if not unresolved:
        return None

    if policy == "LTR":
        return min(
            k
            for k, _size
            in unresolved
        )

    if policy == "MRV":
        # Fewest values first; later column wins a tie only if more constrained
        # by having an already-singleton neighbor.  The primary key is domain
        # size, so this remains a true MRV policy.
        def key(row):
            k, size = row

            neighbor_singletons = 0
            if (
                k > 0
                and int(
                    domains[k - 1]
                ).bit_count()
                == 1
            ):
                neighbor_singletons += 1

            if (
                k + 1
                < len(domains)
                and int(
                    domains[k + 1]
                ).bit_count()
                == 1
            ):
                neighbor_singletons += 1

            return (
                size,
                -neighbor_singletons,
                k,
            )

        return min(
            unresolved,
            key=key,
        )[0]

    raise ValueError(policy)


def solve_columns(
    *,
    state_at_prefix,
    poses,
    previous_face,
    remaining,
    engine,
    policy,
    node_cap,
    time_cap,
):
    """
    Exact column-CSP decision:
        return DR_FOUND if a real DR suffix exists,
        PROVED_UNSAT if all superposed column assignments are exhausted,
        CUT otherwise.
    """
    deadline = (
        time.perf_counter()
        + float(time_cap)
    )
    t0 = time.perf_counter()

    csp_nodes = 0
    prop_calls = 0
    prop_iterations = 0
    removed_values = 0
    assignments_checked = 0
    relaxed_false_sat = 0
    memo_hits = 0

    memo_unsat = set()

    class Cut(Exception):
        pass

    def check_cut():
        if csp_nodes >= int(node_cap):
            raise Cut("node_cap")

        if (
            time.perf_counter()
            >= deadline
        ):
            raise Cut("time_cap")

    def rec(domains):
        nonlocal csp_nodes
        nonlocal prop_calls
        nonlocal prop_iterations
        nonlocal removed_values
        nonlocal assignments_checked
        nonlocal relaxed_false_sat
        nonlocal memo_hits

        check_cut()

        prop_calls += 1

        pres = engine.propagate(
            poses,
            previous_face,
            tuple(domains),
        )

        prop_iterations += int(
            pres.iterations
        )
        removed_values += int(
            pres.removed_values
        )

        if not pres.feasible:
            return None

        pd = tuple(
            pres.domains
        )

        if pd in memo_unsat:
            memo_hits += 1
            return None

        csp_nodes += 1

        if all(
            int(mask).bit_count()
            == 1
            for mask in pd
        ):
            assignments_checked += 1

            word = tuple(
                singleton_move(mask)
                for mask in pd
            )

            final = (
                state_at_prefix.apply_word(
                    word
                )
            )

            if ts.is_dr(final):
                return word

            # The 20 local target constraints can be a relaxation of full DR.
            # Keep searching rather than treating local SAT as real SAT.
            relaxed_false_sat += 1
            memo_unsat.add(pd)
            return None

        k = choose_column(
            pd,
            policy,
        )

        for mi in iter_move_indices(
            pd[k]
        ):
            child = list(pd)
            child[k] = 1 << mi

            witness = rec(
                tuple(child)
            )

            if witness is not None:
                return witness

        memo_unsat.add(pd)
        return None

    initial = tuple(
        ALL_MOVE_MASK
        for _ in range(
            int(remaining)
        )
    )

    try:
        witness = rec(initial)
        cut = None
    except Cut as exc:
        witness = None
        cut = str(exc)

    wall = (
        time.perf_counter()
        - t0
    )

    if cut is not None:
        status = "CUT"
    elif witness is not None:
        status = "DR_FOUND"
    else:
        status = "PROVED_UNSAT"

    return ColumnSolveResult(
        status=status,
        word=witness,
        csp_nodes=csp_nodes,
        propagation_calls=prop_calls,
        propagation_iterations=(
            prop_iterations
        ),
        removed_values=removed_values,
        assignments_checked=(
            assignments_checked
        ),
        relaxed_false_sat=(
            relaxed_false_sat
        ),
        memo_hits=memo_hits,
        wall=wall,
        cut_reason=cut,
    )


@dataclass
class QDecisionResult:
    status: str
    word: tuple[str, ...] | None
    nodes: int
    q_prunes: int
    wall: float
    cut_reason: str | None


def q_decide_from_prefix(
    *,
    state_at_prefix,
    poses,
    q,
    remaining,
    previous_face,
    qtables,
    node_cap,
    time_cap,
):
    deadline = (
        time.perf_counter()
        + float(time_cap)
    )
    t0 = time.perf_counter()

    nodes = 0
    q_prunes = 0
    path = []

    class Cut(Exception):
        pass

    def check_cut():
        if nodes >= int(node_cap):
            raise Cut("node_cap")

        if (
            time.perf_counter()
            >= deadline
        ):
            raise Cut("time_cap")

    def dfs(
        cur_poses,
        cur_q,
        rem,
        last_face,
    ):
        nonlocal nodes, q_prunes

        check_cut()

        h = q_lb(
            cur_q,
            qtables,
        )

        if h > rem:
            q_prunes += 1
            return None

        nodes += 1

        if rem == 0:
            if (
                int(cur_q)
                == int(ts.GOAL_Q)
            ):
                word = tuple(path)
                final = (
                    state_at_prefix
                    .apply_word(word)
                )

                if not ts.is_dr(final):
                    raise RuntimeError(
                        "q-goal / explicit DR mismatch"
                    )

                return word

            return None

        for move in ts.MOVE_ORDER:
            if (
                last_face is not None
                and move[0]
                == last_face
            ):
                continue

            nposes, nq = step_state(
                cur_poses,
                cur_q,
                move,
                qtables,
            )

            h2 = q_lb(
                nq,
                qtables,
            )

            if h2 > rem - 1:
                q_prunes += 1
                continue

            path.append(move)

            witness = dfs(
                nposes,
                nq,
                rem - 1,
                move[0],
            )

            path.pop()

            if witness is not None:
                return witness

        return None

    try:
        witness = dfs(
            poses,
            q,
            int(remaining),
            previous_face,
        )
        cut = None
    except Cut as exc:
        witness = None
        cut = str(exc)

    wall = (
        time.perf_counter()
        - t0
    )

    if cut is not None:
        status = "CUT"
    elif witness is not None:
        status = "DR_FOUND"
    else:
        status = "PROVED_UNSAT"

    return QDecisionResult(
        status=status,
        word=witness,
        nodes=nodes,
        q_prunes=q_prunes,
        wall=wall,
        cut_reason=cut,
    )


def state_for_row(
    scramble_internal,
    row,
):
    root = ts.engine.from_word(
        " ".join(
            scramble_internal
        )
    )

    return root.apply_word(
        tuple(
            row[
                "prefix_internal"
            ]
        )
    )


def row_key(row):
    return tuple(
        row[
            "prefix_internal"
        ]
    )


def pick_dead_rows(
    truth,
    *,
    max_dead,
):
    dead = [
        row
        for row in truth["nodes"]
        if (
            not row["live"]
            and row[
                "remaining"
            ] > 0
            and row[
                "remaining"
            ] <= 4
        )
    ]

    # Prefer nontrivial larger remaining horizons first, then Q-tight prefixes.
    dead.sort(
        key=lambda row: (
            -int(
                row[
                    "remaining"
                ]
            ),
            int(
                row[
                    "remaining"
                ]
            )
            - int(
                row[
                    "q_lb"
                ]
            ),
            -int(
                row[
                    "depth"
                ]
            ),
            tuple(
                MOVE_RANK[m]
                for m in row[
                    "prefix_internal"
                ]
            ),
        )
    )

    return dead[
        : int(max_dead)
    ]


def pick_live_row(
    truth,
    preferred_remaining,
):
    live = [
        row
        for row in truth["nodes"]
        if (
            row["live"]
            and row[
                "remaining"
            ] > 0
            and row[
                "remaining"
            ] <= 4
        )
    ]

    if not live:
        return None

    preferred = set(
        int(x)
        for x in preferred_remaining
    )

    live.sort(
        key=lambda row: (
            0
            if int(
                row[
                    "remaining"
                ]
            )
            in preferred
            else 1,
            -int(
                row[
                    "remaining"
                ]
            ),
            int(
                row[
                    "remaining"
                ]
            )
            - int(
                row[
                    "q_lb"
                ]
            ),
            tuple(
                MOVE_RANK[m]
                for m in row[
                    "prefix_internal"
                ]
            ),
        )
    )

    return live[0]


def fresh_control_with_dead(
    *,
    depth,
    rng,
    qtables,
    min_dead,
    max_control_tries,
    max_generation_tries,
):
    for control_try in range(
        1,
        int(max_control_tries) + 1,
    ):
        control = (
            v73.make_fresh_exact_control(
                target_depth=int(
                    depth
                ),
                rng=rng,
                qtables=qtables,
                max_tries=int(
                    max_generation_tries
                ),
            )
        )

        scramble_internal = tuple(
            control[
                "scramble"
            ]
        )

        truth = (
            v79.q_frontier_with_truth(
                scramble_internal=(
                    scramble_internal
                ),
                depth_limit=int(
                    depth
                ),
                qtables=qtables,
            )
        )

        dead = pick_dead_rows(
            truth,
            max_dead=1000000,
        )

        if len(dead) >= int(
            min_dead
        ):
            return (
                control,
                truth,
                control_try,
            )

    raise RuntimeError(
        f"could not generate fresh exact-{depth} control "
        f"with >= {min_dead} dead prefixes (remaining<=4)"
    )


def analyze_prefix(
    *,
    label,
    scramble_internal,
    axis,
    row,
    qtables,
    engine,
    q_node_cap,
    q_time_cap,
    csp_node_cap,
    csp_time_cap,
):
    prefix_internal = tuple(
        row[
            "prefix_internal"
        ]
    )
    prefix_original = (
        externalize(
            prefix_internal,
            axis,
        )
    )

    state = state_for_row(
        scramble_internal,
        row,
    )

    poses = tuple(
        row["poses"]
    )
    q = int(
        row["q"]
    )
    rem = int(
        row[
            "remaining"
        ]
    )
    last_face = row[
        "last_face"
    ]
    truth_live = bool(
        row["live"]
    )

    # Root GAC only, before column search.
    root_gac = engine.propagate(
        poses,
        last_face,
        tuple(
            ALL_MOVE_MASK
            for _ in range(rem)
        ),
    )

    qres = q_decide_from_prefix(
        state_at_prefix=state,
        poses=poses,
        q=q,
        remaining=rem,
        previous_face=last_face,
        qtables=qtables,
        node_cap=q_node_cap,
        time_cap=q_time_cap,
    )

    ltr = solve_columns(
        state_at_prefix=state,
        poses=poses,
        previous_face=last_face,
        remaining=rem,
        engine=engine,
        policy="LTR",
        node_cap=csp_node_cap,
        time_cap=csp_time_cap,
    )

    mrv = solve_columns(
        state_at_prefix=state,
        poses=poses,
        previous_face=last_face,
        remaining=rem,
        engine=engine,
        policy="MRV",
        node_cap=csp_node_cap,
        time_cap=csp_time_cap,
    )

    expected_status = (
        "DR_FOUND"
        if truth_live
        else "PROVED_UNSAT"
    )

    q_ok = (
        qres.status
        == expected_status
    )
    ltr_ok = (
        ltr.status
        == expected_status
    )
    mrv_ok = (
        mrv.status
        == expected_status
    )

    print(
        f"  {label:<10} "
        f"prefix="
        f"{' '.join(prefix_original) or '(root)'} "
        f"rem={rem} Q={row['q_lb']} "
        f"truth={'LIVE' if truth_live else 'DEAD'} "
        f"rootGAC={'UNSAT' if not root_gac.feasible else 'OPEN'}"
    )
    print(
        "      Q      :",
        f"{qres.status:<12} "
        f"nodes={qres.nodes:5d} "
        f"wall={qres.wall:.4f}s",
    )
    print(
        "      COL-LTR:",
        f"{ltr.status:<12} "
        f"nodes={ltr.csp_nodes:5d} "
        f"props={ltr.propagation_calls:5d} "
        f"assign={ltr.assignments_checked:4d} "
        f"falseSAT={ltr.relaxed_false_sat:4d} "
        f"wall={ltr.wall:.4f}s",
    )
    print(
        "      COL-MRV:",
        f"{mrv.status:<12} "
        f"nodes={mrv.csp_nodes:5d} "
        f"props={mrv.propagation_calls:5d} "
        f"assign={mrv.assignments_checked:4d} "
        f"falseSAT={mrv.relaxed_false_sat:4d} "
        f"wall={mrv.wall:.4f}s",
    )

    return {
        "label": label,
        "prefix_internal": list(
            prefix_internal
        ),
        "prefix_original": list(
            prefix_original
        ),
        "depth": int(
            row["depth"]
        ),
        "remaining": rem,
        "q_lb": int(
            row["q_lb"]
        ),
        "truth_live": truth_live,
        "expected_status": (
            expected_status
        ),
        "root_gac_feasible": bool(
            root_gac.feasible
        ),
        "root_gac_iterations": int(
            root_gac.iterations
        ),
        "root_gac_removed_values": int(
            root_gac.removed_values
        ),
        "root_gac_domain_sizes": [
            int(x).bit_count()
            for x in root_gac.domains
        ],
        "q": {
            "status": qres.status,
            "nodes": qres.nodes,
            "q_prunes": qres.q_prunes,
            "wall": qres.wall,
            "cut_reason": qres.cut_reason,
            "word_internal": (
                None
                if qres.word is None
                else list(
                    qres.word
                )
            ),
            "correct": q_ok,
        },
        "column_ltr": {
            "status": ltr.status,
            "word_internal": (
                None
                if ltr.word is None
                else list(
                    ltr.word
                )
            ),
            "csp_nodes": (
                ltr.csp_nodes
            ),
            "propagation_calls": (
                ltr.propagation_calls
            ),
            "propagation_iterations": (
                ltr.propagation_iterations
            ),
            "removed_values": (
                ltr.removed_values
            ),
            "assignments_checked": (
                ltr.assignments_checked
            ),
            "relaxed_false_sat": (
                ltr.relaxed_false_sat
            ),
            "memo_hits": (
                ltr.memo_hits
            ),
            "wall": ltr.wall,
            "cut_reason": (
                ltr.cut_reason
            ),
            "correct": ltr_ok,
        },
        "column_mrv": {
            "status": mrv.status,
            "word_internal": (
                None
                if mrv.word is None
                else list(
                    mrv.word
                )
            ),
            "csp_nodes": (
                mrv.csp_nodes
            ),
            "propagation_calls": (
                mrv.propagation_calls
            ),
            "propagation_iterations": (
                mrv.propagation_iterations
            ),
            "removed_values": (
                mrv.removed_values
            ),
            "assignments_checked": (
                mrv.assignments_checked
            ),
            "relaxed_false_sat": (
                mrv.relaxed_false_sat
            ),
            "memo_hits": (
                mrv.memo_hits
            ),
            "wall": mrv.wall,
            "cut_reason": (
                mrv.cut_reason
            ),
            "correct": mrv_ok,
        },
    }


def analyze_control(
    *,
    name,
    scramble_original,
    axis,
    depth,
    truth,
    qtables,
    engine,
    max_dead,
    historical_prefix_external=None,
    q_node_cap,
    q_time_cap,
    csp_node_cap,
    csp_time_cap,
):
    scramble_internal = (
        internalize(
            scramble_original,
            axis,
        )
    )

    dead_rows = pick_dead_rows(
        truth,
        max_dead=max_dead,
    )

    # Force historical positive-control prefix into the sample.
    if (
        historical_prefix_external
        is not None
    ):
        hist_internal = (
            internalize(
                historical_prefix_external,
                axis,
            )
        )

        hist_row = next(
            (
                row
                for row
                in truth["nodes"]
                if tuple(
                    row[
                        "prefix_internal"
                    ]
                )
                == tuple(
                    hist_internal
                )
            ),
            None,
        )

        if hist_row is None:
            raise RuntimeError(
                "historical prefix not found in Q frontier"
            )

        dead_rows = [
            hist_row
        ] + [
            row
            for row in dead_rows
            if row_key(row)
            != row_key(
                hist_row
            )
        ]

        dead_rows = dead_rows[
            : int(max_dead)
        ]

    preferred_rem = [
        row[
            "remaining"
        ]
        for row in dead_rows
    ]

    live_row = pick_live_row(
        truth,
        preferred_rem,
    )

    sample = [
        (
            f"DEAD-{i+1}",
            row,
        )
        for i, row in enumerate(
            dead_rows
        )
    ]

    if live_row is not None:
        sample.append(
            (
                "LIVE-1",
                live_row,
            )
        )

    print(
        f"## {name} axis={axis} "
        f"exactDepth={depth}"
    )
    print(
        "scramble             :",
        " ".join(
            scramble_original
        ),
    )
    print(
        "Q frontier / sol     :",
        f"{len(truth['nodes'])} / "
        f"{truth['solution_count']}",
    )
    print(
        "sample dead/live     :",
        f"{len(dead_rows)} / "
        f"{1 if live_row is not None else 0}",
    )

    rows = []

    for label, row in sample:
        rows.append(
            analyze_prefix(
                label=label,
                scramble_internal=(
                    scramble_internal
                ),
                axis=axis,
                row=row,
                qtables=qtables,
                engine=engine,
                q_node_cap=q_node_cap,
                q_time_cap=q_time_cap,
                csp_node_cap=(
                    csp_node_cap
                ),
                csp_time_cap=(
                    csp_time_cap
                ),
            )
        )

    print()

    return {
        "name": name,
        "axis": axis,
        "depth": int(
            depth
        ),
        "scramble_original": list(
            scramble_original
        ),
        "q_frontier_nodes": len(
            truth[
                "nodes"
            ]
        ),
        "q_solution_count": int(
            truth[
                "solution_count"
            ]
        ),
        "sampled_dead_count": len(
            dead_rows
        ),
        "sampled_live_count": (
            1
            if live_row is not None
            else 0
        ),
        "prefixes": rows,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--seed",
        type=int,
        default=20260925,
    )
    ap.add_argument(
        "--fresh-depths",
        nargs="+",
        type=int,
        default=[7, 8],
    )
    ap.add_argument(
        "--min-dead-per-fresh",
        type=int,
        default=3,
    )
    ap.add_argument(
        "--sample-dead-per-control",
        type=int,
        default=3,
    )
    ap.add_argument(
        "--max-control-tries",
        type=int,
        default=30,
    )
    ap.add_argument(
        "--max-generation-tries",
        type=int,
        default=30000,
    )
    ap.add_argument(
        "--cache",
        default=(
            "reports/pdcc_cache/"
            "twist_skeleton_tables_v1.pkl"
        ),
    )
    ap.add_argument(
        "--q-node-cap",
        type=int,
        default=100000,
    )
    ap.add_argument(
        "--q-time-cap",
        type=float,
        default=0.5,
    )
    ap.add_argument(
        "--csp-node-cap",
        type=int,
        default=50000,
    )
    ap.add_argument(
        "--csp-time-cap",
        type=float,
        default=0.5,
    )
    ap.add_argument(
        "--output",
        default=(
            "reports/v37/"
            "column_superposition_pilot_v37_80.json"
        ),
    )

    args = ap.parse_args()

    print(
        "# CubeLab v37.80 - "
        "COLUMN-COMPLETION x 20-PIECE SUPERPOSITION"
    )
    print(
        "state representation :",
        "all 20 piece future languages remain superposed",
    )
    print(
        "search variables     :",
        "GLOBAL COLUMNS ONLY",
    )
    print(
        "piece collapse       :",
        "NEVER",
    )
    print(
        "propagation          :",
        "20-piece GAC after every column assignment",
    )
    print(
        "column policies      :",
        "LTR / MRV",
    )
    print(
        "terminal acceptance  :",
        "explicit real-cube DR only",
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

    engine = ColumnSuperposition(
        dr_targets
    )

    m11 = next(
        c
        for c in v57.CONTROLS
        if c["name"]
        == "M11-01"
    )

    rows = []

    # Historical positive control.
    hist_sw = internalize(
        tuple(
            m11[
                "scramble"
            ]
        ),
        m11["axis"],
    )

    hist_truth = (
        v79.q_frontier_with_truth(
            scramble_internal=hist_sw,
            depth_limit=7,
            qtables=qtables,
        )
    )

    print()

    rows.append(
        analyze_control(
            name="HIST-M11-L7",
            scramble_original=tuple(
                m11[
                    "scramble"
                ]
            ),
            axis=m11["axis"],
            depth=7,
            truth=hist_truth,
            qtables=qtables,
            engine=engine,
            max_dead=int(
                args.sample_dead_per_control
            ),
            historical_prefix_external=(
                HIST_PREFIX_EXTERNAL
            ),
            q_node_cap=int(
                args.q_node_cap
            ),
            q_time_cap=float(
                args.q_time_cap
            ),
            csp_node_cap=int(
                args.csp_node_cap
            ),
            csp_time_cap=float(
                args.csp_time_cap
            ),
        )
    )

    rng = random.Random(
        int(
            args.seed
        )
    )

    fresh_controls = []

    for depth in (
        int(x)
        for x in args.fresh_depths
    ):
        (
            control,
            truth,
            control_try,
        ) = fresh_control_with_dead(
            depth=depth,
            rng=rng,
            qtables=qtables,
            min_dead=int(
                args.min_dead_per_fresh
            ),
            max_control_tries=int(
                args.max_control_tries
            ),
            max_generation_tries=int(
                args.max_generation_tries
            ),
        )

        control = {
            **control,
            "selection_control_try": (
                control_try
            ),
        }

        fresh_controls.append(
            control
        )

        rows.append(
            analyze_control(
                name=f"FRESH-L{depth}",
                scramble_original=tuple(
                    control[
                        "scramble"
                    ]
                ),
                axis="UD",
                depth=depth,
                truth=truth,
                qtables=qtables,
                engine=engine,
                max_dead=int(
                    args.sample_dead_per_control
                ),
                historical_prefix_external=None,
                q_node_cap=int(
                    args.q_node_cap
                ),
                q_time_cap=float(
                    args.q_time_cap
                ),
                csp_node_cap=int(
                    args.csp_node_cap
                ),
                csp_time_cap=float(
                    args.csp_time_cap
                ),
            )
        )

    print("# AGGREGATE")

    prefix_rows = [
        p
        for control in rows
        for p in control[
            "prefixes"
        ]
    ]

    dead_rows = [
        p
        for p in prefix_rows
        if not p[
            "truth_live"
        ]
    ]

    live_rows = [
        p
        for p in prefix_rows
        if p[
            "truth_live"
        ]
    ]

    gac_detected_dead = sum(
        not p[
            "root_gac_feasible"
        ]
        for p in dead_rows
    )

    ltr_dead_proved = sum(
        p[
            "column_ltr"
        ][
            "status"
        ]
        == "PROVED_UNSAT"
        for p in dead_rows
    )

    mrv_dead_proved = sum(
        p[
            "column_mrv"
        ][
            "status"
        ]
        == "PROVED_UNSAT"
        for p in dead_rows
    )

    ltr_all_correct = all(
        p[
            "column_ltr"
        ][
            "correct"
        ]
        for p in prefix_rows
    )

    mrv_all_correct = all(
        p[
            "column_mrv"
        ][
            "correct"
        ]
        for p in prefix_rows
    )

    q_all_correct = all(
        p["q"][
            "correct"
        ]
        for p in prefix_rows
    )

    mrv_resolved_dead = [
        p
        for p in dead_rows
        if p[
            "column_mrv"
        ][
            "status"
        ]
        == "PROVED_UNSAT"
    ]

    mrv_better_than_q_nodes = [
        p
        for p in mrv_resolved_dead
        if (
            p[
                "column_mrv"
            ][
                "csp_nodes"
            ]
            < p["q"][
                "nodes"
            ]
        )
    ]

    print(
        "sampled prefixes     :",
        len(
            prefix_rows
        ),
    )
    print(
        "dead / live          :",
        f"{len(dead_rows)} / "
        f"{len(live_rows)}",
    )
    print(
        "root GAC dead detect :",
        f"{gac_detected_dead}/"
        f"{len(dead_rows)}",
    )
    print(
        "column LTR dead proof:",
        f"{ltr_dead_proved}/"
        f"{len(dead_rows)}",
    )
    print(
        "column MRV dead proof:",
        f"{mrv_dead_proved}/"
        f"{len(dead_rows)}",
    )
    print(
        "MRV cheaper-node dead:",
        f"{len(mrv_better_than_q_nodes)}/"
        f"{len(dead_rows)}",
    )
    print(
        "contracts Q/LTR/MRV  :",
        q_all_correct,
        "/",
        ltr_all_correct,
        "/",
        mrv_all_correct,
    )

    print()
    print("# DECISION")

    if not (
        q_all_correct
        and ltr_all_correct
        and mrv_all_correct
    ):
        decision = (
            "COLUMN_SUPERPOSITION_CONTRACT_FAIL"
        )
        note = (
            "At least one sampled prefix disagrees with the exact Q LIVE/DEAD "
            "ground truth. Fix the column CSP before interpretation."
        )

    elif (
        mrv_dead_proved
        > gac_detected_dead
        and mrv_dead_proved
        == len(
            dead_rows
        )
    ):
        decision = (
            "COLUMN_SUPERPOSITION_STRONG_PILOT_PASS"
        )
        note = (
            "Keeping all piece futures in superposition while actively completing "
            "global columns proves every sampled Q-dead prefix, including fresh "
            "cases that root GAC alone does not detect. Column completion is "
            "therefore the missing operation; superposition and column synthesis "
            "should be treated as one combined model."
        )

    elif (
        mrv_dead_proved
        > gac_detected_dead
    ):
        decision = (
            "COLUMN_SUPERPOSITION_PARTIAL_PILOT_PASS"
        )
        note = (
            "Column completion over the superposed piece languages detects more "
            "dead prefixes than GAC alone, but some sampled cases cut or remain "
            "unresolved within the FAST budget. Continue with targeted search/"
            "propagation engineering rather than discarding the model."
        )

    elif (
        mrv_dead_proved
        == gac_detected_dead
    ):
        decision = (
            "COLUMN_COMPLETION_NO_ADDED_SIGNAL"
        )
        note = (
            "Explicit column completion does not recover additional sampled dead "
            "prefixes beyond root GAC. The current per-piece language representation "
            "is likely too weak even when combined with column search."
        )

    else:
        decision = (
            "COLUMN_SUPERPOSITION_INCONCLUSIVE"
        )
        note = (
            "The pilot is inconclusive under the current FAST caps."
        )

    print(
        decision
    )
    print(
        note
    )

    if mrv_resolved_dead:
        ratios = [
            (
                p[
                    "column_mrv"
                ][
                    "csp_nodes"
                ]
                / p["q"][
                    "nodes"
                ]
            )
            for p in mrv_resolved_dead
            if p["q"][
                "nodes"
            ] > 0
        ]

        if ratios:
            print(
                "median MRV/Q node ratio:",
                f"{sorted(ratios)[len(ratios)//2]:.3f}",
            )

    payload = {
        "version": "v37.80",
        "mode": (
            "COLUMN_COMPLETION_X_20_PIECE_SUPERPOSITION"
        ),
        "seed": int(
            args.seed
        ),
        "fresh_depths": [
            int(x)
            for x in (
                args.fresh_depths
            )
        ],
        "controls": rows,
        "fresh_controls": [
            {
                **c,
                "scramble": list(
                    c[
                        "scramble"
                    ]
                ),
                "planted_inverse": list(
                    c[
                        "planted_inverse"
                    ]
                ),
            }
            for c in fresh_controls
        ],
        "aggregate": {
            "sampled_prefix_count": len(
                prefix_rows
            ),
            "dead_count": len(
                dead_rows
            ),
            "live_count": len(
                live_rows
            ),
            "root_gac_dead_detected": (
                gac_detected_dead
            ),
            "column_ltr_dead_proved": (
                ltr_dead_proved
            ),
            "column_mrv_dead_proved": (
                mrv_dead_proved
            ),
            "mrv_cheaper_node_dead_count": len(
                mrv_better_than_q_nodes
            ),
            "q_all_correct": (
                q_all_correct
            ),
            "ltr_all_correct": (
                ltr_all_correct
            ),
            "mrv_all_correct": (
                mrv_all_correct
            ),
        },
        "decision": decision,
        "note": note,
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
