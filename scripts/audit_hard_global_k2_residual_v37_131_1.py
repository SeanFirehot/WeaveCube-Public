#!/usr/bin/env python3
"""
CubeLab v37.131.1 — DIRECT-Q HARD GLOBAL H-SKELETON K2 PORTFOLIO

Why
---
v37.130 reproduced the natural H6/budget8/+1 K2 near miss, but the entire
available local ancestry was only:

    target H6
    d=1   H7
    d=2   H8

and every level still had:

    min combined K2 LB = 3
    remaining K2 budget = 2
    proxyFit = 0

Therefore the concrete short-corpus conflict is NOT locally repairable inside
the available find ancestry.

Do not invent a nonexistent d=3 local ancestor.
Do not expand blind local mutation.

Instead move the now-grounded K2 residual objective to a genuinely harder
GLOBAL H-skeleton portfolio, while avoiding the v37.23 support-builder
explosion seen on long-H9/H10.

Default hard control
--------------------
The 25-move scramble previously used by the three-axis H factory:

    D' R F2 U D' R L B D R' U D R' D F' B U F' L' D2 B L D' F R'

Procedure
---------
1. Run v37.114 unchanged only to initialize/validate the stable tables/oracle.
2. Build the hard scramble state directly.
3. Compute root phase1 LB Hq.
4. For H = Hq + configured offsets:
       enumerate stable H-prefix decisions for H-4 columns
       synthesize the final exact shared R4 directly to DR
       score every DR endpoint by:
           P2LB
           combined terminal LB
5. Exact-K2 probe only:
       canonical first endpoint
       + top-K combined-LB endpoints
6. Compare:
       canonical residual
       best residual
       exact K2 of the shortlist
       total H + K2

This is a GLOBAL/root H-skeleton portfolio test.
It is not a completed local-NO repair test.

Stable timed search changes: NONE.
Production changes: NONE.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if SRC.is_dir():
    s = str(SRC)
    if s not in sys.path:
        sys.path.insert(0, s)

import search_pdcc_guided as pg
import audit_fast_stable_selective_terminal_macro_v37_114 as v114
import audit_post_dr_k2_deficit_v37_128 as v128


DEFAULT_SCRAMBLE = (
    "D' R F2 U D' R L B D R' U D R' D F' B U "
    "F' L' D2 B L D' F R'"
)

_CAPTURED_SOLVER = None
_BASE_FIND_NO_SHADOW = None

_ORIGINAL_V114_COMBINED_INSTALLER = v114._combined_installer


def _install_capture():
    global _CAPTURED_SOLVER
    global _BASE_FIND_NO_SHADOW

    cls = v114.v24.grc.MemoTotalBudgetSearch

    if getattr(
        cls,
        "_v37131_capture_installed",
        False,
    ):
        return

    orig_find = cls.find
    _BASE_FIND_NO_SHADOW = orig_find

    def find_capture(self, *args, **kwargs):
        global _CAPTURED_SOLVER

        if _CAPTURED_SOLVER is None:
            _CAPTURED_SOLVER = self

        return orig_find(
            self,
            *args,
            **kwargs,
        )

    cls.find = find_capture
    cls._v37131_capture_installed = True


def _v37131_combined_installer():
    _ORIGINAL_V114_COMBINED_INSTALLER()
    _install_capture()


def _combined_lb_metrics(
    *,
    solver,
    end_poses,
):
    pk = v114.v24.grc.ts.p_of(
        pg.PDCCState(
            tuple(
                end_poses
            )
        )
    )

    if pk is None:
        return {
            "status": "P2_COORD_FAIL",
            "p2lb": None,
            "combined_lb": None,
            "combined_available": False,
        }

    p2lb = int(
        v114.v24.grc.ts.phase2_lb(
            int(
                pk
            ),
            solver.cpd,
            solver.upd,
            solver.spd,
        )
    )

    combined_available = hasattr(
        solver,
        "combined_terminal_lb",
    )

    if combined_available:
        try:
            combined_lb = int(
                solver.combined_terminal_lb(
                    int(
                        pk
                    )
                )
            )
        except Exception:
            combined_available = False
            combined_lb = int(
                p2lb
            )
    else:
        combined_lb = int(
            p2lb
        )

    return {
        "status": "OK",
        "p2lb": int(
            p2lb
        ),
        "combined_lb": int(
            combined_lb
        ),
        "combined_available": bool(
            combined_available
        ),
    }


def _scramble_state(
    *,
    solver,
    moves,
):
    """
    Build the hard root through the engine directly.

    v37.131 used _suffix_terminal_state as a generic word applier.  That was
    unnecessary and made the hard-root harness harder to audit.
    """
    state = (
        v114.v24.grc.ts.engine.from_word(
            " ".join(
                moves
            )
        )
    )

    return (
        tuple(
            int(x)
            for x in (
                state.poses
            )
        ),
        int(
            v114.v24.grc.ts.q_of(
                state
            )
        ),
    )


def _iter_bits(
    mask,
):
    value = int(
        mask
    )

    while value:
        low = (
            value
            & -value
        )

        yield (
            low.bit_length()
            - 1
        )

        value ^= low


def _direct_global_h_portfolio(
    *,
    solver,
    oracle,
    poses,
    q,
    h_length,
    r4_suffix_cap,
    prefix_budget,
    deadline,
):
    """
    Enumerate an exact-H GLOBAL root portfolio without fabricating a
    MemoTotalBudgetSearch depth-domain frame.

    Prefix columns:
        canonical reduced move legality
        + exact quotient transition
        + admissible phase1_lb <= remaining-1

    Final four columns:
        exact compiled shared R4 wordset.

    This mirrors the hard H-factory / direct phase1 semantics.  It does NOT use
    solver.oo.allowed_domain(), which is a stable-search depth-domain oracle
    intended for naturally derived find() contexts.
    """
    ts = (
        v114.v24.grc.ts
    )

    skeletons = []
    endpoints = {}
    stats = Counter()

    def recurse(
        cur_poses,
        cur_q,
        remaining_h,
        last_face,
        prefix_moves,
        prefix_ranks,
    ):
        if (
            time.perf_counter()
            >= deadline
            or int(
                prefix_budget[
                    "nodes"
                ]
            )
            >= int(
                prefix_budget[
                    "cap"
                ]
            )
        ):
            return

        # First-entry H semantics: do not reach DR early and leave again.
        if (
            int(
                cur_q
            )
            == int(
                ts.GOAL_Q
            )
            and int(
                remaining_h
            )
            != 0
        ):
            return

        if int(
            remaining_h
        ) == 4:
            stats[
                "r4_leaf_states"
            ] += 1

            qr = (
                oracle.query(
                    poses=tuple(
                        cur_poses
                    ),
                    q=int(
                        cur_q
                    ),
                    R=4,
                    last_face=(
                        last_face
                    ),
                    add_q_first=False,
                )
            )

            if not qr[
                "feasible"
            ]:
                return

            mask = int(
                qr[
                    "word_mask"
                ]
            )

            qcount = int(
                qr.get(
                    "word_count",
                    mask.bit_count(),
                )
            )

            if qcount <= 0:
                return

            stats[
                "r4_oracle_nonempty"
            ] += 1

            stats[
                "r4_word_sum"
            ] += int(
                qcount
            )

            stats[
                "max_r4_words"
            ] = max(
                int(
                    stats[
                        "max_r4_words"
                    ]
                ),
                int(
                    qcount
                ),
            )

            if qcount > int(
                r4_suffix_cap
            ):
                stats[
                    "r4_cap_truncated"
                ] += 1

            words = (
                oracle.words[
                    4
                ]
            )

            for r4_rank, word_idx in enumerate(
                _iter_bits(
                    mask
                ),
                start=1,
            ):
                if r4_rank > int(
                    r4_suffix_cap
                ):
                    break

                if time.perf_counter() >= deadline:
                    break

                stats[
                    "r4_words_considered"
                ] += 1

                row = (
                    words[
                        int(
                            word_idx
                        )
                    ]
                )

                r4_suffix = tuple(
                    ts.MOVE_ORDER[
                        int(
                            mi
                        )
                    ]
                    for mi in row
                )

                (
                    end_poses,
                    end_q,
                    q_before_final,
                    final_face,
                ) = (
                    v114._suffix_terminal_state(
                        solver,
                        tuple(
                            cur_poses
                        ),
                        int(
                            cur_q
                        ),
                        row,
                    )
                )

                if int(
                    end_q
                ) != int(
                    ts.GOAL_Q
                ):
                    continue

                if int(
                    q_before_final
                ) == int(
                    ts.GOAL_Q
                ):
                    continue

                key = (
                    tuple(
                        int(x)
                        for x in end_poses
                    ),
                    final_face,
                )

                if key not in endpoints:
                    endpoints[
                        key
                    ] = {
                        "end_poses": tuple(
                            int(x)
                            for x in end_poses
                        ),
                        "end_q": int(
                            end_q
                        ),
                        "final_face": (
                            final_face
                        ),
                    }

                skeletons.append(
                    {
                        "endpoint_key": key,
                        "prefix_moves": list(
                            prefix_moves
                        ),
                        "prefix_ranks": list(
                            prefix_ranks
                        ),
                        "r4_rank": int(
                            r4_rank
                        ),
                        "r4_suffix": list(
                            r4_suffix
                        ),
                    }
                )

            return

        if int(
            remaining_h
        ) <= 4:
            return

        children = []

        for mi, move in enumerate(
            ts.MOVE_ORDER
        ):
            if not ts.allow(
                last_face,
                move,
            ):
                continue

            nq = int(
                ts.q_move(
                    int(
                        cur_q
                    ),
                    int(
                        mi
                    ),
                    solver.co,
                    solver.eo,
                    solver.sl,
                )
            )

            child_lb = int(
                ts.phase1_lb(
                    int(
                        nq
                    ),
                    solver.cos,
                    solver.eos,
                )
            )

            if child_lb > int(
                remaining_h
            ) - 1:
                continue

            event_rank = (
                0
                if move
                in ts.T_MOVES
                else 1
            )

            children.append(
                (
                    int(
                        child_lb
                    ),
                    int(
                        event_rank
                    ),
                    str(
                        move
                    ),
                    int(
                        mi
                    ),
                    int(
                        nq
                    ),
                )
            )

        children.sort(
            key=lambda x: (
                x[
                    0
                ],
                x[
                    1
                ],
                x[
                    2
                ],
            )
        )

        if not prefix_moves:
            stats[
                "root_q_children"
            ] = len(
                children
            )

        for local_rank, (
            _child_lb,
            _event_rank,
            move,
            mi,
            nq,
        ) in enumerate(
            children,
            start=1,
        ):
            if (
                time.perf_counter()
                >= deadline
                or int(
                    prefix_budget[
                        "nodes"
                    ]
                )
                >= int(
                    prefix_budget[
                        "cap"
                    ]
                )
            ):
                break

            prefix_budget[
                "nodes"
            ] += 1

            stats[
                "prefix_nodes"
            ] += 1

            nposes = (
                v114._pose_step(
                    tuple(
                        cur_poses
                    ),
                    int(
                        mi
                    ),
                )
            )

            recurse(
                tuple(
                    nposes
                ),
                int(
                    nq
                ),
                int(
                    remaining_h
                )
                - 1,
                str(
                    move
                )[
                    0
                ],
                tuple(
                    prefix_moves
                )
                + (
                    str(
                        move
                    ),
                ),
                tuple(
                    prefix_ranks
                )
                + (
                    int(
                        local_rank
                    ),
                ),
            )

    # Diagnostic: the lower bound at the root should agree with the caller.
    stats[
        "root_phase1_lb"
    ] = int(
        v114.v24.grc.ts.phase1_lb(
            int(
                q
            ),
            solver.cos,
            solver.eos,
        )
    )

    recurse(
        tuple(
            poses
        ),
        int(
            q
        ),
        int(
            h_length
        ),
        None,
        (),
        (),
    )

    return {
        "stats": (
            stats
        ),
        "skeletons": (
            skeletons
        ),
        "endpoints": (
            endpoints
        ),
    }


def _endpoint_records(
    *,
    solver,
    portfolio,
):
    first_skeleton = {}

    for skeleton in (
        portfolio[
            "skeletons"
        ]
    ):
        key = skeleton[
            "endpoint_key"
        ]

        if key not in first_skeleton:
            first_skeleton[
                key
            ] = skeleton

    records = []

    for index, (
        key,
        endpoint,
    ) in enumerate(
        portfolio[
            "endpoints"
        ].items()
    ):
        lb = _combined_lb_metrics(
            solver=solver,
            end_poses=(
                endpoint[
                    "end_poses"
                ]
            ),
        )

        records.append(
            {
                "endpoint_index": int(
                    index
                ),
                "endpoint_key": key,
                "endpoint": endpoint,
                "skeleton": (
                    first_skeleton.get(
                        key
                    )
                ),
                "p2lb": (
                    lb[
                        "p2lb"
                    ]
                ),
                "combined_lb": (
                    lb[
                        "combined_lb"
                    ]
                ),
                "combined_available": bool(
                    lb[
                        "combined_available"
                    ]
                ),
            }
        )

    return records


def _record_summary(
    record,
):
    skeleton = record.get(
        "skeleton"
    )

    return {
        "endpoint_index": int(
            record[
                "endpoint_index"
            ]
        ),
        "p2lb": (
            record[
                "p2lb"
            ]
        ),
        "combined_lb": (
            record[
                "combined_lb"
            ]
        ),
        "prefix_moves": (
            []
            if skeleton is None
            else list(
                skeleton[
                    "prefix_moves"
                ]
            )
        ),
        "prefix_ranks": (
            []
            if skeleton is None
            else list(
                skeleton[
                    "prefix_ranks"
                ]
            )
        ),
        "r4_rank": (
            None
            if skeleton is None
            else int(
                skeleton[
                    "r4_rank"
                ]
            )
        ),
        "r4_suffix": (
            []
            if skeleton is None
            else list(
                skeleton[
                    "r4_suffix"
                ]
            )
        ),
    }


def _parse_local_args(
    argv,
):
    ap = argparse.ArgumentParser(
        add_help=False
    )

    ap.add_argument(
        "--hard-scramble",
        default=(
            DEFAULT_SCRAMBLE
        ),
    )

    ap.add_argument(
        "--hard-h-offsets",
        nargs="+",
        type=int,
        default=[
            0,
            1,
            2,
            3,
        ],
    )

    ap.add_argument(
        "--hard-r4-suffix-cap",
        type=int,
        default=64,
    )

    ap.add_argument(
        "--hard-prefix-node-cap",
        type=int,
        default=200000,
    )

    ap.add_argument(
        "--hard-exact-top-k",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--hard-exact-k2-probe-cap",
        type=int,
        default=12,
    )

    ap.add_argument(
        "--hard-exact-k2-call-cap",
        type=int,
        default=2000,
    )

    ap.add_argument(
        "--hard-offline-time-limit",
        type=float,
        default=30.0,
    )

    ap.add_argument(
        "--hard-output",
        default=(
            "reports/v37/"
            "hard_global_h_k2_residual_portfolio_v37_131_1.json"
        ),
    )

    ns, remaining = (
        ap.parse_known_args(
            argv
        )
    )

    ns.hard_h_offsets = tuple(
        sorted(
            {
                max(
                    0,
                    int(
                        x
                    ),
                )
                for x in (
                    ns.hard_h_offsets
                )
            }
        )
    )

    ns.hard_r4_suffix_cap = max(
        1,
        int(
            ns.hard_r4_suffix_cap
        ),
    )

    ns.hard_prefix_node_cap = max(
        1,
        int(
            ns.hard_prefix_node_cap
        ),
    )

    ns.hard_exact_top_k = max(
        1,
        int(
            ns.hard_exact_top_k
        ),
    )

    ns.hard_exact_k2_probe_cap = max(
        0,
        int(
            ns.hard_exact_k2_probe_cap
        ),
    )

    ns.hard_exact_k2_call_cap = max(
        1,
        int(
            ns.hard_exact_k2_call_cap
        ),
    )

    ns.hard_offline_time_limit = max(
        0.1,
        float(
            ns.hard_offline_time_limit
        ),
    )

    return (
        ns,
        remaining,
    )


def main():
    local, remaining = (
        _parse_local_args(
            sys.argv[
                1:
            ]
        )
    )

    v114._combined_installer = (
        _v37131_combined_installer
    )

    sys.argv = [
        sys.argv[
            0
        ],
        *remaining,
    ]

    print(
        "# CubeLab v37.131.1 - "
        "HARD-SCRAMBLE GLOBAL H-SKELETON K2-RESIDUAL PORTFOLIO"
    )

    print(
        "stable bootstrap     :",
        "v37.24.1 + v37.114 unchanged",
    )

    print(
        "hard scramble        :",
        local.hard_scramble,
    )

    print(
        "H offsets            :",
        list(
            local.hard_h_offsets
        ),
    )

    print(
        "future synthesis     :",
        "stable H-prefix + exact shared R4 -> DR",
    )

    print(
        "endpoint score       :",
        "combined terminal K2 LB",
    )

    print(
        "exact K2             :",
        f"canonical + top-{local.hard_exact_top_k} residual endpoints, "
        f"budget<= {local.hard_exact_k2_probe_cap}",
    )

    print(
        "hard prefix engine   :",
        "DIRECT quotient DFS (allow + q_move + phase1_lb), NOT stable depth-domain oracle",
    )

    print(
        "global supportBuild  :",
        "NOT USED for hard portfolio",
    )

    print(
        "production changes   :",
        "NONE",
    )

    print()
    print(
        "# STABLE BOOTSTRAP / REPLAY"
    )

    rc = (
        v114.main()
    )

    solver = (
        _CAPTURED_SOLVER
    )

    if solver is None:
        raise RuntimeError(
            "failed to capture MemoTotalBudgetSearch solver"
        )

    if v114._ORACLE is None:
        raise RuntimeError(
            "v37.114 oracle unavailable"
        )

    if _BASE_FIND_NO_SHADOW is None:
        raise RuntimeError(
            "terminal find capture unavailable"
        )

    v128._BASE_FIND_NO_SHADOW = (
        _BASE_FIND_NO_SHADOW
    )

    moves = tuple(
        token
        for token in (
            local.hard_scramble.split()
        )
        if token
    )

    unknown = [
        move
        for move in moves
        if move not in (
            v114.v24.grc.ts.MI
        )
    ]

    if unknown:
        raise RuntimeError(
            f"unknown scramble moves: {unknown}"
        )

    poses, q = (
        _scramble_state(
            solver=solver,
            moves=moves,
        )
    )

    hq = int(
        v114.v24.grc.ts.phase1_lb(
            int(
                q
            ),
            solver.cos,
            solver.eos,
        )
    )

    print()
    print(
        "# HARD GLOBAL PORTFOLIO"
    )

    print(
        "scramble length      :",
        len(
            moves
        ),
    )

    print(
        "root phase1 LB       :",
        hq,
    )

    deadline = (
        time.perf_counter()
        + float(
            local.hard_offline_time_limit
        )
    )

    prefix_budget = {
        "nodes": 0,
        "cap": int(
            local.hard_prefix_node_cap
        ),
    }

    exact_calls = {
        "calls": 0,
        "cap": int(
            local.hard_exact_k2_call_cap
        ),
        "cache": {},
    }

    rows = []
    truncated = False

    for offset in (
        local.hard_h_offsets
    ):
        if (
            time.perf_counter()
            >= deadline
            or int(
                prefix_budget[
                    "nodes"
                ]
            )
            >= int(
                prefix_budget[
                    "cap"
                ]
            )
            or int(
                exact_calls[
                    "calls"
                ]
            )
            >= int(
                exact_calls[
                    "cap"
                ]
            )
        ):
            truncated = True
            break

        h_length = (
            int(
                hq
            )
            + int(
                offset
            )
        )

        if h_length < 4:
            continue

        before_nodes = int(
            prefix_budget[
                "nodes"
            ]
        )

        portfolio = (
            _direct_global_h_portfolio(
                solver=solver,
                oracle=(
                    v114._ORACLE
                ),
                poses=tuple(
                    poses
                ),
                q=int(
                    q
                ),
                h_length=int(
                    h_length
                ),
                r4_suffix_cap=int(
                    local.hard_r4_suffix_cap
                ),
                prefix_budget=(
                    prefix_budget
                ),
                deadline=(
                    deadline
                ),
            )
        )

        records = (
            _endpoint_records(
                solver=solver,
                portfolio=portfolio,
            )
        )

        records_by_combined = sorted(
            records,
            key=lambda rec: (
                999
                if rec[
                    "combined_lb"
                ]
                is None
                else int(
                    rec[
                        "combined_lb"
                    ]
                ),
                999
                if rec[
                    "p2lb"
                ]
                is None
                else int(
                    rec[
                        "p2lb"
                    ]
                ),
                int(
                    rec[
                        "endpoint_index"
                    ]
                ),
            ),
        )

        canonical = (
            records[
                0
            ]
            if records
            else None
        )

        shortlist = []

        if canonical is not None:
            shortlist.append(
                canonical
            )

        for rec in records_by_combined[
            : int(
                local.hard_exact_top_k
            )
        ]:
            if all(
                int(
                    existing[
                        "endpoint_index"
                    ]
                )
                != int(
                    rec[
                        "endpoint_index"
                    ]
                )
                for existing in shortlist
            ):
                shortlist.append(
                    rec
                )

        exact_rows = []

        for rec in shortlist:
            if (
                time.perf_counter()
                >= deadline
                or int(
                    exact_calls[
                        "calls"
                    ]
                )
                >= int(
                    exact_calls[
                        "cap"
                    ]
                )
            ):
                truncated = True
                break

            cache_key = (
                tuple(
                    rec[
                        "endpoint"
                    ][
                        "end_poses"
                    ]
                ),
                rec[
                    "endpoint"
                ][
                    "final_face"
                ],
                id(
                    solver
                ),
            )

            probe = (
                exact_calls[
                    "cache"
                ].get(
                    cache_key
                )
            )

            if probe is None:
                probe = (
                    v128._probe_min_k2(
                        solver=solver,
                        end_poses=(
                            rec[
                                "endpoint"
                            ][
                                "end_poses"
                            ]
                        ),
                        end_q=int(
                            rec[
                                "endpoint"
                            ][
                                "end_q"
                            ]
                        ),
                        final_face=(
                            rec[
                                "endpoint"
                            ][
                                "final_face"
                            ]
                        ),
                        k2_probe_cap=int(
                            local.hard_exact_k2_probe_cap
                        ),
                        global_calls=(
                            exact_calls
                        ),
                        deadline=(
                            deadline
                        ),
                    )
                )

                exact_calls[
                    "cache"
                ][
                    cache_key
                ] = probe

            exact_rows.append(
                {
                    **_record_summary(
                        rec
                    ),
                    "is_canonical": bool(
                        canonical is not None
                        and int(
                            rec[
                                "endpoint_index"
                            ]
                        )
                        == int(
                            canonical[
                                "endpoint_index"
                            ]
                        )
                    ),
                    "exact_status": str(
                        probe[
                            "status"
                        ]
                    ),
                    "exact_min_k2": (
                        None
                        if probe.get(
                            "min_k2"
                        )
                        is None
                        else int(
                            probe[
                                "min_k2"
                            ]
                        )
                    ),
                    "exact_total": (
                        None
                        if probe.get(
                            "min_k2"
                        )
                        is None
                        else int(
                            h_length
                            + int(
                                probe[
                                    "min_k2"
                                ]
                            )
                        )
                    ),
                }
            )

        combined_values = [
            int(
                rec[
                    "combined_lb"
                ]
            )
            for rec in records
            if rec[
                "combined_lb"
            ]
            is not None
        ]

        best_combined = (
            min(
                combined_values
            )
            if combined_values
            else None
        )

        canonical_combined = (
            None
            if canonical is None
            else canonical[
                "combined_lb"
            ]
        )

        found_exact = [
            row
            for row in exact_rows
            if row[
                "exact_min_k2"
            ]
            is not None
        ]

        best_exact = (
            min(
                int(
                    row[
                        "exact_min_k2"
                    ]
                )
                for row in found_exact
            )
            if found_exact
            else None
        )

        best_exact_total = (
            None
            if best_exact is None
            else int(
                h_length
                + best_exact
            )
        )

        canonical_exact_rows = [
            row
            for row in exact_rows
            if (
                row[
                    "is_canonical"
                ]
                and row[
                    "exact_min_k2"
                ]
                is not None
            )
        ]

        canonical_exact = (
            None
            if not canonical_exact_rows
            else int(
                canonical_exact_rows[
                    0
                ][
                    "exact_min_k2"
                ]
            )
        )

        row = {
            "h_length": int(
                h_length
            ),
            "offset": int(
                offset
            ),
            "prefix_nodes": int(
                prefix_budget[
                    "nodes"
                ]
                - before_nodes
            ),
            "root_q_children": int(
                portfolio[
                    "stats"
                ][
                    "root_q_children"
                ]
            ),
            "portfolio_root_phase1_lb": int(
                portfolio[
                    "stats"
                ][
                    "root_phase1_lb"
                ]
            ),
            "r4_leaf_states": int(
                portfolio[
                    "stats"
                ][
                    "r4_leaf_states"
                ]
            ),
            "r4_oracle_nonempty": int(
                portfolio[
                    "stats"
                ][
                    "r4_oracle_nonempty"
                ]
            ),
            "r4_word_sum": int(
                portfolio[
                    "stats"
                ][
                    "r4_word_sum"
                ]
            ),
            "max_r4_words": int(
                portfolio[
                    "stats"
                ][
                    "max_r4_words"
                ]
            ),
            "r4_cap_truncated": int(
                portfolio[
                    "stats"
                ][
                    "r4_cap_truncated"
                ]
            ),
            "skeleton_count": len(
                portfolio[
                    "skeletons"
                ]
            ),
            "endpoint_count": len(
                records
            ),
            "canonical_combined_lb": (
                canonical_combined
            ),
            "best_combined_lb": (
                best_combined
            ),
            "combined_lb_improvement": (
                None
                if (
                    canonical_combined
                    is None
                    or best_combined
                    is None
                )
                else int(
                    canonical_combined
                )
                - int(
                    best_combined
                )
            ),
            "proxy_best_total": (
                None
                if best_combined is None
                else int(
                    h_length
                    + best_combined
                )
            ),
            "canonical_exact_k2": (
                canonical_exact
            ),
            "best_exact_k2_shortlist": (
                best_exact
            ),
            "best_exact_total_shortlist": (
                best_exact_total
            ),
            "shortlist": (
                exact_rows
            ),
        }

        rows.append(
            row
        )

        print(
            f"H={h_length:<2} "
            f"rootChildren={row['root_q_children']:,} "
            f"prefixNodes={row['prefix_nodes']:,} "
            f"R4leaf={row['r4_leaf_states']:,} "
            f"skeletons={row['skeleton_count']:,} "
            f"endpoints={row['endpoint_count']:,} "
            f"Ccanonical={row['canonical_combined_lb']} "
            f"Cbest={row['best_combined_lb']} "
            f"proxyTotal={row['proxy_best_total']} "
            f"exactBest={row['best_exact_k2_shortlist']} "
            f"exactTotal={row['best_exact_total_shortlist']}"
        )

        if truncated:
            break

    direct_q_nonzero_rows = [
        row
        for row in rows
        if int(
            row[
                "root_q_children"
            ]
        )
        > 0
    ]

    exact_found_rows = [
        row
        for row in rows
        if row[
            "best_exact_total_shortlist"
        ]
        is not None
    ]

    best_global = (
        min(
            exact_found_rows,
            key=lambda row: (
                int(
                    row[
                        "best_exact_total_shortlist"
                    ]
                ),
                int(
                    row[
                        "h_length"
                    ]
                ),
            ),
        )
        if exact_found_rows
        else None
    )

    proxy_improvement_rows = [
        row
        for row in rows
        if (
            row[
                "combined_lb_improvement"
            ]
            is not None
            and int(
                row[
                    "combined_lb_improvement"
                ]
            )
            > 0
        )
    ]

    print()
    print("# SUMMARY")
    print(
        "H rows analyzed      :",
        len(
            rows
        ),
    )
    print(
        "prefix nodes total   :",
        prefix_budget[
            "nodes"
        ],
    )
    print(
        "exact K2 calls       :",
        exact_calls[
            "calls"
        ],
    )
    print(
        "direct-q nonzero rows:",
        f"{len(direct_q_nonzero_rows)}/{len(rows)}",
    )

    print(
        "proxy improves canonical:",
        f"{len(proxy_improvement_rows)}/{len(rows)}",
    )
    print(
        "analysis truncated   :",
        truncated,
    )

    if best_global is not None:
        print(
            "best exact shortlist:",
            f"H={best_global['h_length']} "
            f"K2={best_global['best_exact_k2_shortlist']} "
            f"total={best_global['best_exact_total_shortlist']}",
        )

    if not direct_q_nonzero_rows:
        decision = (
            "HARD_GLOBAL_DIRECT_Q_ENUMERATOR_CONTRACT_FAIL"
        )
        note = (
            "Even the direct quotient DFS produced no legal root child at any "
            "tested H horizon. Treat this as a state-construction/table mismatch, "
            "not as evidence against K2-residual portfolio ranking."
        )

    elif (
        best_global is not None
        and proxy_improvement_rows
    ):
        decision = (
            "HARD_GLOBAL_K2_RESIDUAL_PORTFOLIO_SIGNAL"
        )
        note = (
            "On the hard 25-move root state, combined K2 residual scoring "
            "changes the preferred DR endpoint relative to canonical generation "
            "for at least one H allocation, and the residual shortlist reaches "
            "an exact K2 witness within the probe cap. This is the correct scale "
            "for the next requirement-directed synthesis experiment."
        )

    elif best_global is not None:
        decision = (
            "HARD_GLOBAL_PORTFOLIO_EXACT_HIT_NO_PROXY_REORDER_SIGNAL"
        )
        note = (
            "The hard global H portfolio reaches an exact K2 witness, but the "
            "combined residual score does not improve over the canonical first "
            "endpoint on the tested H allocations. Do not add a residual ranking "
            "layer from this single control."
        )

    elif truncated:
        decision = (
            "HARD_GLOBAL_PORTFOLIO_AUDIT_TRUNCATED"
        )
        note = (
            "The hard global H-skeleton portfolio reached an explicit offline "
            "cap before an exact shortlist conclusion. Increase only that cap, "
            "not the stable/global support-builder budget."
        )

    elif proxy_improvement_rows:
        decision = (
            "HARD_GLOBAL_PROXY_SIGNAL_EXACT_K2_ABOVE_PROBE_CAP"
        )
        note = (
            "Combined residual scoring meaningfully reorders hard-scramble DR "
            "endpoints, but exact K2 did not resolve within the configured probe "
            "cap. Increase only the exact shortlist K2 cap before drawing a "
            "conclusion."
        )

    else:
        decision = (
            "HARD_GLOBAL_K2_RESIDUAL_SIGNAL_NOT_OBSERVED"
        )
        note = (
            "The hard global portfolio exposes no residual advantage over "
            "canonical endpoint generation under the tested H offsets. The "
            "current combined-K2 proxy should remain an offline diagnostic."
        )

    print()
    print("# DECISION")
    print(decision)
    print(note)
    print()
    print("NEXT:")

    if decision == "HARD_GLOBAL_K2_RESIDUAL_PORTFOLIO_SIGNAL":
        print(
            "  v37.132 requirement-directed global H generation"
        )
        print(
            "  compare all H skeletons vs combined-LB shortlist size and exact K2 hit rank"
        )
        print(
            "  then fresh 2-3 hard scrambles before any production hook"
        )
    elif decision == "HARD_GLOBAL_PROXY_SIGNAL_EXACT_K2_ABOVE_PROBE_CAP":
        print(
            "  rerun only with a slightly larger exact-K2 probe cap"
        )
    elif decision == "HARD_GLOBAL_PORTFOLIO_AUDIT_TRUNCATED":
        print(
            "  rerun only with a larger offline prefix/time cap"
        )
    else:
        print(
            "  do not active-integrate combined residual ranking from this control"
        )

    payload = {
        "version": "v37.131.1",
        "mode": "DIRECT_Q_HARD_GLOBAL_H_SKELETON_K2_RESIDUAL_PORTFOLIO",
        "stable_return_code": int(
            rc
        ),
        "scramble": list(
            moves
        ),
        "scramble_length": len(
            moves
        ),
        "root_phase1_lb": int(
            hq
        ),
        "config": {
            "h_offsets": list(
                local.hard_h_offsets
            ),
            "r4_suffix_cap": int(
                local.hard_r4_suffix_cap
            ),
            "prefix_node_cap": int(
                local.hard_prefix_node_cap
            ),
            "exact_top_k": int(
                local.hard_exact_top_k
            ),
            "exact_k2_probe_cap": int(
                local.hard_exact_k2_probe_cap
            ),
            "exact_k2_call_cap": int(
                local.hard_exact_k2_call_cap
            ),
            "offline_time_limit": float(
                local.hard_offline_time_limit
            ),
        },
        "rows": rows,
        "aggregate": {
            "h_rows_analyzed": len(
                rows
            ),
            "prefix_nodes_total": int(
                prefix_budget[
                    "nodes"
                ]
            ),
            "exact_k2_calls": int(
                exact_calls[
                    "calls"
                ]
            ),
            "direct_q_nonzero_rows": len(
                direct_q_nonzero_rows
            ),
            "proxy_improves_canonical_rows": len(
                proxy_improvement_rows
            ),
            "analysis_truncated": bool(
                truncated
            ),
            "best_exact_h": (
                None
                if best_global is None
                else int(
                    best_global[
                        "h_length"
                    ]
                )
            ),
            "best_exact_k2": (
                None
                if best_global is None
                else int(
                    best_global[
                        "best_exact_k2_shortlist"
                    ]
                )
            ),
            "best_exact_total": (
                None
                if best_global is None
                else int(
                    best_global[
                        "best_exact_total_shortlist"
                    ]
                )
            ),
        },
        "decision": decision,
        "note": note,
        "production_changes": "NONE",
    }

    out = Path(
        local.hard_output
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
        "hard JSON            :",
        out,
    )

    return int(
        rc
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
