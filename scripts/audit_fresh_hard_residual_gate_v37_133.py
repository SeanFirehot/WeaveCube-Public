#!/usr/bin/env python3
"""
CubeLab v37.133 — FRESH HARD-SCRAMBLE MIN-RESIDUAL GATE VALIDATION

Background
----------
v37.132 complete control is now a clean PASS:

    hard 25-move control, H=10
    raw skeletons : 174
    raw endpoints : 94
    min combined K2 LB : 8
    gated endpoints : 15
    endpoint compression : 84.0%
    canonical exact K2 : 13
    best gated exact K2 : 11
    exact improvement : 2
    all gated 15/15 exact/replay PASS
    no truncation : True

v37.133 asks whether that mechanism generalizes to FRESH hard scrambles.

No cherry-picking:
    * generate the first N reduced random scrambles from one fixed seed;
    * use H = root phase1 LB + configured offset (default +3);
    * derive each scramble's residual threshold from its OWN minimum
      combined terminal K2 LB;
    * preserve the ENTIRE minimum-LB tie class;
    * exact-K2 compare that gate against the canonical first DR endpoint.

The R4 suffix cap is automatically expanded and the same case is re-enumerated
when the exact oracle reports a larger word count, so a known cap truncation
cannot silently become a scientific result.

This is an OFFLINE validation program.
Stable timed search is unchanged.
Production changes: NONE.
"""

from __future__ import annotations

import argparse
import json
import random
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
import audit_hard_global_k2_residual_v37_131_1 as v1311
import audit_post_dr_k2_deficit_v37_128 as v128
import audit_fast_stable_selective_terminal_macro_v37_114 as v114


DEFAULT_CONTROL = (
    "D' R F2 U D' R L B D R' U D R' D F' B U "
    "F' L' D2 B L D' F R'"
)


def _iter_bits(mask):
    value = int(mask)

    while value:
        low = value & -value
        yield low.bit_length() - 1
        value ^= low


def _generate_reduced_scramble(
    *,
    rng,
    length,
):
    ts = v114.v24.grc.ts

    out = []
    last_face = None

    for _ in range(int(length)):
        legal = [
            move
            for move in ts.MOVE_ORDER
            if ts.allow(
                last_face,
                move,
            )
        ]

        move = rng.choice(
            legal
        )

        out.append(
            str(move)
        )

        last_face = str(
            move
        )[0]

    return tuple(
        out
    )


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
            int(pk),
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
                    int(pk)
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


def _enumerate_direct_h_complete(
    *,
    solver,
    oracle,
    poses,
    q,
    h_length,
    r4_suffix_cap,
    prefix_node_cap,
    time_limit,
):
    """
    Direct quotient H enumeration with explicit cut telemetry.

    Prefix:
        allow -> q_move -> phase1_lb

    Final four H columns:
        exact compiled shared R4 oracle.

    The return value distinguishes:
        prefix node-cap cut
        wall-clock cut
        R4 suffix-cap truncation
    """
    ts = v114.v24.grc.ts

    start = time.perf_counter()
    deadline = start + float(
        time_limit
    )

    stats = Counter()
    skeletons = []
    endpoints = {}

    cut = {
        "node_cap": False,
        "time": False,
    }

    def should_cut():
        if int(
            stats[
                "prefix_nodes"
            ]
        ) >= int(
            prefix_node_cap
        ):
            cut[
                "node_cap"
            ] = True
            return True

        if time.perf_counter() >= deadline:
            cut[
                "time"
            ] = True
            return True

        return False

    def recurse(
        cur_poses,
        cur_q,
        remaining_h,
        last_face,
        prefix_moves,
        prefix_ranks,
    ):
        if should_cut():
            return

        # Preserve first-entry H semantics.
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

            qr = oracle.query(
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

            words = oracle.words[
                4
            ]

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

                if should_cut():
                    break

                stats[
                    "r4_words_considered"
                ] += 1

                row = words[
                    int(
                        word_idx
                    )
                ]

                r4_suffix = tuple(
                    ts.MOVE_ORDER[
                        int(mi)
                    ]
                    for mi in row
                )

                (
                    end_poses,
                    end_q,
                    q_before_final,
                    final_face,
                ) = v114._suffix_terminal_state(
                    solver,
                    tuple(
                        cur_poses
                    ),
                    int(
                        cur_q
                    ),
                    row,
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
                    lb = _combined_lb_metrics(
                        solver=solver,
                        end_poses=(
                            end_poses
                        ),
                    )

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
            if should_cut():
                break

            stats[
                "prefix_nodes"
            ] += 1

            nposes = v114._pose_step(
                tuple(
                    cur_poses
                ),
                int(
                    mi
                ),
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
                )[0],
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

    stats[
        "root_phase1_lb"
    ] = int(
        ts.phase1_lb(
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

    elapsed = (
        time.perf_counter()
        - start
    )

    return {
        "stats": (
            stats
        ),
        "cuts": (
            cut
        ),
        "elapsed": float(
            elapsed
        ),
        "skeletons": (
            skeletons
        ),
        "endpoints": (
            endpoints
        ),
    }


def _verify_solution(
    *,
    scrambled_poses,
    skeleton,
    k2_word,
):
    word = (
        tuple(
            skeleton[
                "prefix_moves"
            ]
        )
        + tuple(
            skeleton[
                "r4_suffix"
            ]
        )
        + tuple(
            k2_word
        )
    )

    poses = tuple(
        scrambled_poses
    )

    for move in word:
        mi = int(
            v114.v24.grc.ts.MI[
                str(
                    move
                )
            ]
        )

        poses = v114._pose_step(
            poses,
            mi,
        )

    solved_poses = tuple(
        int(x)
        for x in (
            v114.v24.grc.ts.SOLVED.poses
        )
    )

    return tuple(
        poses
    ) == solved_poses


def _first_skeleton_by_endpoint(
    portfolio,
):
    out = {}

    for skeleton in portfolio[
        "skeletons"
    ]:
        key = skeleton[
            "endpoint_key"
        ]

        if key not in out:
            out[
                key
            ] = skeleton

    return out


def _probe_endpoint(
    *,
    solver,
    endpoint,
    call_state,
    probe_cap,
    deadline,
):
    key = (
        tuple(
            endpoint[
                "end_poses"
            ]
        ),
        endpoint[
            "final_face"
        ],
        id(
            solver
        ),
    )

    cached = call_state[
        "cache"
    ].get(
        key
    )

    if cached is not None:
        return cached

    probe = v128._probe_min_k2(
        solver=solver,
        end_poses=(
            endpoint[
                "end_poses"
            ]
        ),
        end_q=int(
            endpoint[
                "end_q"
            ]
        ),
        final_face=(
            endpoint[
                "final_face"
            ]
        ),
        k2_probe_cap=int(
            probe_cap
        ),
        global_calls=(
            call_state
        ),
        deadline=(
            deadline
        ),
    )

    call_state[
        "cache"
    ][
        key
    ] = probe

    return probe


def _compare_exact(
    *,
    canonical_probe,
    gated_rows,
    probe_cap,
):
    """
    Return WIN/TIE/LOSS/UNRESOLVED with proof semantics.

    For the gated set, FOUND and ABOVE_CAP are exact-order sufficient once at
    least one FOUND value exists.  Any UNKNOWN/TRUNCATED status makes the gated
    minimum unresolved.
    """
    bad_statuses = {
        row[
            "exact_status"
        ]
        for row in gated_rows
        if row[
            "exact_status"
        ]
        not in (
            "FOUND",
            "ABOVE_CAP",
        )
    }

    if bad_statuses:
        return {
            "status": "UNRESOLVED",
            "reason": (
                "gated_unknown_status:"
                + ",".join(
                    sorted(
                        bad_statuses
                    )
                )
            ),
            "canonical_exact": (
                canonical_probe.get(
                    "min_k2"
                )
            ),
            "best_gated_exact": None,
            "delta": None,
        }

    found_gated = [
        row
        for row in gated_rows
        if row[
            "exact_status"
        ]
        == "FOUND"
    ]

    all_gated_above = bool(
        gated_rows
        and not found_gated
        and all(
            row[
                "exact_status"
            ]
            == "ABOVE_CAP"
            for row in gated_rows
        )
    )

    canonical_status = str(
        canonical_probe[
            "status"
        ]
    )

    canonical_exact = (
        None
        if canonical_probe.get(
            "min_k2"
        )
        is None
        else int(
            canonical_probe[
                "min_k2"
            ]
        )
    )

    if found_gated:
        best_gated_exact = min(
            int(
                row[
                    "exact_k2"
                ]
            )
            for row in found_gated
        )
    else:
        best_gated_exact = None

    if canonical_status == "FOUND":
        if best_gated_exact is not None:
            delta = (
                int(
                    canonical_exact
                )
                - int(
                    best_gated_exact
                )
            )

            if delta > 0:
                result = "WIN"
            elif delta == 0:
                result = "TIE"
            else:
                result = "LOSS"

            return {
                "status": result,
                "reason": (
                    "both_exact"
                ),
                "canonical_exact": int(
                    canonical_exact
                ),
                "best_gated_exact": int(
                    best_gated_exact
                ),
                "delta": int(
                    delta
                ),
            }

        if all_gated_above:
            # canonical <= cap, every gated > cap
            return {
                "status": "LOSS",
                "reason": (
                    "canonical_found_all_gated_above_cap"
                ),
                "canonical_exact": int(
                    canonical_exact
                ),
                "best_gated_exact": None,
                "delta": None,
            }

    if canonical_status == "ABOVE_CAP":
        if best_gated_exact is not None:
            return {
                "status": "WIN",
                "reason": (
                    "canonical_above_cap_gated_found"
                ),
                "canonical_exact": None,
                "best_gated_exact": int(
                    best_gated_exact
                ),
                "delta": (
                    f">{int(probe_cap) - int(best_gated_exact)}"
                ),
            }

        if all_gated_above:
            return {
                "status": "UNRESOLVED",
                "reason": (
                    "both_above_cap"
                ),
                "canonical_exact": None,
                "best_gated_exact": None,
                "delta": None,
            }

    return {
        "status": "UNRESOLVED",
        "reason": (
            "canonical_status:"
            + canonical_status
        ),
        "canonical_exact": (
            canonical_exact
        ),
        "best_gated_exact": (
            best_gated_exact
        ),
        "delta": None,
    }


def _parse_args(
    argv,
):
    ap = argparse.ArgumentParser(
        add_help=False
    )

    ap.add_argument(
        "--fresh-count",
        type=int,
        default=3,
    )

    ap.add_argument(
        "--fresh-seed",
        type=int,
        default=20261044,
    )

    ap.add_argument(
        "--scramble-length",
        type=int,
        default=25,
    )

    ap.add_argument(
        "--h-offset",
        type=int,
        default=3,
    )

    ap.add_argument(
        "--initial-r4-cap",
        type=int,
        default=192,
    )

    ap.add_argument(
        "--prefix-node-cap",
        type=int,
        default=200000,
    )

    ap.add_argument(
        "--case-time-limit",
        type=float,
        default=60.0,
    )

    ap.add_argument(
        "--exact-k2-probe-cap",
        type=int,
        default=18,
    )

    ap.add_argument(
        "--exact-k2-call-cap-per-case",
        type=int,
        default=3000,
    )

    ap.add_argument(
        "--report-examples",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--fresh-output",
        default=(
            "reports/v37/"
            "fresh_hard_min_residual_gate_v37_133.json"
        ),
    )

    ns, remaining = (
        ap.parse_known_args(
            argv
        )
    )

    ns.fresh_count = max(
        1,
        int(
            ns.fresh_count
        ),
    )

    ns.scramble_length = max(
        1,
        int(
            ns.scramble_length
        ),
    )

    ns.h_offset = max(
        0,
        int(
            ns.h_offset
        ),
    )

    ns.initial_r4_cap = max(
        1,
        int(
            ns.initial_r4_cap
        ),
    )

    ns.prefix_node_cap = max(
        1,
        int(
            ns.prefix_node_cap
        ),
    )

    ns.case_time_limit = max(
        1.0,
        float(
            ns.case_time_limit
        ),
    )

    ns.exact_k2_probe_cap = max(
        0,
        int(
            ns.exact_k2_probe_cap
        ),
    )

    ns.exact_k2_call_cap_per_case = max(
        1,
        int(
            ns.exact_k2_call_cap_per_case
        ),
    )

    ns.report_examples = max(
        1,
        int(
            ns.report_examples
        ),
    )

    return (
        ns,
        remaining,
    )


def main():
    local, remaining = _parse_args(
        sys.argv[
            1:
        ]
    )

    # One unchanged bootstrap only.
    v114._combined_installer = (
        v1311._v37131_combined_installer
    )

    sys.argv = [
        sys.argv[
            0
        ],
        *remaining,
    ]

    print(
        "# CubeLab v37.133 - "
        "FRESH HARD-SCRAMBLE MIN-RESIDUAL GATE VALIDATION"
    )

    print(
        "fresh cases          :",
        int(
            local.fresh_count
        ),
    )

    print(
        "fresh seed           :",
        int(
            local.fresh_seed
        ),
    )

    print(
        "scramble length      :",
        int(
            local.scramble_length
        ),
    )

    print(
        "H policy             :",
        f"root phase1 LB + {local.h_offset}",
    )

    print(
        "gate policy          :",
        "KEEP FULL minimum-combined-LB tie class",
    )

    print(
        "initial R4 cap       :",
        int(
            local.initial_r4_cap
        ),
        "(auto-expand + re-enumerate if required)",
    )

    print(
        "exact K2 cap         :",
        int(
            local.exact_k2_probe_cap
        ),
    )

    print(
        "production changes   :",
        "NONE",
    )

    print()
    print(
        "# STABLE BOOTSTRAP / REPLAY"
    )

    rc = v114.main()

    solver = v1311._CAPTURED_SOLVER

    if solver is None:
        raise RuntimeError(
            "failed to capture MemoTotalBudgetSearch solver"
        )

    oracle = v114._ORACLE

    if oracle is None:
        raise RuntimeError(
            "v37.114 oracle unavailable"
        )

    if v1311._BASE_FIND_NO_SHADOW is None:
        raise RuntimeError(
            "terminal finder capture unavailable"
        )

    v128._BASE_FIND_NO_SHADOW = (
        v1311._BASE_FIND_NO_SHADOW
    )

    rng = random.Random(
        int(
            local.fresh_seed
        )
    )

    control_tokens = tuple(
        token
        for token in (
            DEFAULT_CONTROL.split()
        )
        if token
    )

    fresh_scrambles = []

    while len(
        fresh_scrambles
    ) < int(
        local.fresh_count
    ):
        candidate = _generate_reduced_scramble(
            rng=rng,
            length=int(
                local.scramble_length
            ),
        )

        if candidate == control_tokens:
            continue

        if candidate in fresh_scrambles:
            continue

        fresh_scrambles.append(
            candidate
        )

    print()
    print(
        "# FRESH CASES"
    )

    case_rows = []

    for case_index, moves in enumerate(
        fresh_scrambles,
        start=1,
    ):
        case_start = time.perf_counter()

        poses, q = v1311._scramble_state(
            solver=solver,
            moves=moves,
        )

        root_lb = int(
            v114.v24.grc.ts.phase1_lb(
                int(
                    q
                ),
                solver.cos,
                solver.eos,
            )
        )

        h_length = int(
            root_lb
            + int(
                local.h_offset
            )
        )

        print()
        print(
            f"[CASE {case_index}/{local.fresh_count}]"
        )

        print(
            "scramble             :",
            " ".join(
                moves
            ),
        )

        print(
            "root LB / H          :",
            f"{root_lb} / {h_length}",
        )

        r4_cap = int(
            local.initial_r4_cap
        )

        enum_passes = []

        portfolio = None

        while True:
            elapsed_case = (
                time.perf_counter()
                - case_start
            )

            remaining_case_time = (
                float(
                    local.case_time_limit
                )
                - float(
                    elapsed_case
                )
            )

            if remaining_case_time <= 0:
                break

            portfolio = _enumerate_direct_h_complete(
                solver=solver,
                oracle=oracle,
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
                    r4_cap
                ),
                prefix_node_cap=int(
                    local.prefix_node_cap
                ),
                time_limit=float(
                    remaining_case_time
                ),
            )

            enum_passes.append(
                {
                    "r4_cap": int(
                        r4_cap
                    ),
                    "prefix_nodes": int(
                        portfolio[
                            "stats"
                        ][
                            "prefix_nodes"
                        ]
                    ),
                    "r4_leaf_states": int(
                        portfolio[
                            "stats"
                        ][
                            "r4_leaf_states"
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
                    "node_cap_cut": bool(
                        portfolio[
                            "cuts"
                        ][
                            "node_cap"
                        ]
                    ),
                    "time_cut": bool(
                        portfolio[
                            "cuts"
                        ][
                            "time"
                        ]
                    ),
                    "elapsed": float(
                        portfolio[
                            "elapsed"
                        ]
                    ),
                }
            )

            if (
                portfolio[
                    "cuts"
                ][
                    "node_cap"
                ]
                or portfolio[
                    "cuts"
                ][
                    "time"
                ]
            ):
                break

            truncated_count = int(
                portfolio[
                    "stats"
                ][
                    "r4_cap_truncated"
                ]
            )

            max_r4_words = int(
                portfolio[
                    "stats"
                ][
                    "max_r4_words"
                ]
            )

            if (
                truncated_count > 0
                and max_r4_words > int(
                    r4_cap
                )
            ):
                r4_cap = int(
                    max_r4_words
                )

                continue

            break

        if portfolio is None:
            row = {
                "case": int(
                    case_index
                ),
                "scramble": list(
                    moves
                ),
                "root_lb": int(
                    root_lb
                ),
                "h_length": int(
                    h_length
                ),
                "complete": False,
                "reason": (
                    "no_enumeration_pass"
                ),
                "enum_passes": (
                    enum_passes
                ),
                "outcome": (
                    "INCONCLUSIVE"
                ),
            }

            case_rows.append(
                row
            )

            print(
                "OUTCOME              :",
                "INCONCLUSIVE (no enumeration pass)",
            )

            continue

        complete_enum = bool(
            not portfolio[
                "cuts"
            ][
                "node_cap"
            ]
            and not portfolio[
                "cuts"
            ][
                "time"
            ]
            and int(
                portfolio[
                    "stats"
                ][
                    "r4_cap_truncated"
                ]
            )
            == 0
        )

        first_skeleton = (
            _first_skeleton_by_endpoint(
                portfolio
            )
        )

        raw_endpoints = list(
            portfolio[
                "endpoints"
            ].items()
        )

        raw_endpoint_count = len(
            raw_endpoints
        )

        raw_skeleton_count = len(
            portfolio[
                "skeletons"
            ]
        )

        if not raw_endpoints:
            row = {
                "case": int(
                    case_index
                ),
                "scramble": list(
                    moves
                ),
                "root_lb": int(
                    root_lb
                ),
                "h_length": int(
                    h_length
                ),
                "complete": bool(
                    complete_enum
                ),
                "reason": (
                    "no_dr_endpoints"
                ),
                "enum_passes": (
                    enum_passes
                ),
                "raw_endpoint_count": 0,
                "raw_skeleton_count": int(
                    raw_skeleton_count
                ),
                "outcome": (
                    "INCONCLUSIVE"
                ),
            }

            case_rows.append(
                row
            )

            print(
                "raw endpoints        :",
                0,
            )

            print(
                "OUTCOME              :",
                "INCONCLUSIVE (no DR endpoints at chosen H)",
            )

            continue

        canonical_key, canonical_endpoint = raw_endpoints[
            0
        ]

        combined_values = [
            int(
                endpoint[
                    "combined_lb"
                ]
            )
            for _key, endpoint in raw_endpoints
            if endpoint[
                "combined_lb"
            ]
            is not None
        ]

        if not combined_values:
            raise RuntimeError(
                "all DR endpoints missing combined LB"
            )

        min_c = min(
            combined_values
        )

        gated = [
            (
                key,
                endpoint,
            )
            for key, endpoint in raw_endpoints
            if (
                endpoint[
                    "combined_lb"
                ]
                is not None
                and int(
                    endpoint[
                        "combined_lb"
                    ]
                )
                == int(
                    min_c
                )
            )
        ]

        skeleton_mult = Counter(
            skeleton[
                "endpoint_key"
            ]
            for skeleton in (
                portfolio[
                    "skeletons"
                ]
            )
        )

        gated_skeleton_count = sum(
            int(
                skeleton_mult[
                    key
                ]
            )
            for key, _endpoint in gated
        )

        endpoint_compression = (
            1.0
            - (
                float(
                    len(
                        gated
                    )
                )
                / float(
                    raw_endpoint_count
                )
            )
        )

        skeleton_compression = (
            0.0
            if raw_skeleton_count == 0
            else (
                1.0
                - (
                    float(
                        gated_skeleton_count
                    )
                    / float(
                        raw_skeleton_count
                    )
                )
            )
        )

        canonical_c = canonical_endpoint[
            "combined_lb"
        ]

        case_deadline = (
            case_start
            + float(
                local.case_time_limit
            )
        )

        exact_calls = {
            "calls": 0,
            "cap": int(
                local.exact_k2_call_cap_per_case
            ),
            "cache": {},
        }

        canonical_probe = _probe_endpoint(
            solver=solver,
            endpoint=(
                canonical_endpoint
            ),
            call_state=(
                exact_calls
            ),
            probe_cap=int(
                local.exact_k2_probe_cap
            ),
            deadline=(
                case_deadline
            ),
        )

        gated_rows = []

        for gate_rank, (
            key,
            endpoint,
        ) in enumerate(
            gated,
            start=1,
        ):
            if (
                time.perf_counter()
                >= case_deadline
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
                break

            probe = _probe_endpoint(
                solver=solver,
                endpoint=(
                    endpoint
                ),
                call_state=(
                    exact_calls
                ),
                probe_cap=int(
                    local.exact_k2_probe_cap
                ),
                deadline=(
                    case_deadline
                ),
            )

            skeleton = first_skeleton.get(
                key
            )

            exact_k2 = (
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
            )

            solved = False

            if (
                exact_k2 is not None
                and skeleton is not None
            ):
                solved = _verify_solution(
                    scrambled_poses=(
                        poses
                    ),
                    skeleton=(
                        skeleton
                    ),
                    k2_word=tuple(
                        probe.get(
                            "word",
                            (),
                        )
                    ),
                )

            gated_rows.append(
                {
                    "gate_rank": int(
                        gate_rank
                    ),
                    "endpoint_index": int(
                        raw_endpoints.index(
                            (
                                key,
                                endpoint,
                            )
                        )
                    ),
                    "combined_lb": int(
                        endpoint[
                            "combined_lb"
                        ]
                    ),
                    "p2lb": int(
                        endpoint[
                            "p2lb"
                        ]
                    ),
                    "skeleton_multiplicity": int(
                        skeleton_mult[
                            key
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
                    "r4_suffix": (
                        []
                        if skeleton is None
                        else list(
                            skeleton[
                                "r4_suffix"
                            ]
                        )
                    ),
                    "exact_status": str(
                        probe[
                            "status"
                        ]
                    ),
                    "exact_k2": (
                        exact_k2
                    ),
                    "k2_word": list(
                        probe.get(
                            "word",
                            (),
                        )
                    ),
                    "solved_replay": bool(
                        solved
                    ),
                }
            )

        exact_target_coverage = bool(
            len(
                gated_rows
            )
            == len(
                gated
            )
        )

        comparison = _compare_exact(
            canonical_probe=(
                canonical_probe
            ),
            gated_rows=(
                gated_rows
            ),
            probe_cap=int(
                local.exact_k2_probe_cap
            ),
        )

        found_gated = [
            row
            for row in gated_rows
            if (
                row[
                    "exact_status"
                ]
                == "FOUND"
                and row[
                    "solved_replay"
                ]
            )
        ]

        best_gated = (
            min(
                found_gated,
                key=lambda row: (
                    int(
                        row[
                            "exact_k2"
                        ]
                    ),
                    int(
                        row[
                            "gate_rank"
                        ]
                    ),
                ),
            )
            if found_gated
            else None
        )

        complete_case = bool(
            complete_enum
            and exact_target_coverage
            and int(
                exact_calls[
                    "calls"
                ]
            )
            < int(
                exact_calls[
                    "cap"
                ]
            )
            and all(
                row[
                    "exact_status"
                ]
                in (
                    "FOUND",
                    "ABOVE_CAP",
                )
                for row in gated_rows
            )
            and str(
                canonical_probe[
                    "status"
                ]
            )
            in (
                "FOUND",
                "ABOVE_CAP",
            )
        )

        outcome = (
            comparison[
                "status"
            ]
            if complete_case
            else "INCONCLUSIVE"
        )

        row = {
            "case": int(
                case_index
            ),
            "scramble": list(
                moves
            ),
            "root_lb": int(
                root_lb
            ),
            "h_length": int(
                h_length
            ),
            "enum_passes": (
                enum_passes
            ),
            "final_r4_cap": int(
                r4_cap
            ),
            "raw_endpoint_count": int(
                raw_endpoint_count
            ),
            "raw_skeleton_count": int(
                raw_skeleton_count
            ),
            "canonical_combined_lb": (
                canonical_c
            ),
            "min_combined_lb": int(
                min_c
            ),
            "gated_endpoint_count": len(
                gated
            ),
            "gated_skeleton_count": int(
                gated_skeleton_count
            ),
            "endpoint_compression": float(
                endpoint_compression
            ),
            "skeleton_compression": float(
                skeleton_compression
            ),
            "canonical_exact_status": str(
                canonical_probe[
                    "status"
                ]
            ),
            "canonical_exact_k2": (
                None
                if canonical_probe.get(
                    "min_k2"
                )
                is None
                else int(
                    canonical_probe[
                        "min_k2"
                    ]
                )
            ),
            "gated_rows": (
                gated_rows
            ),
            "best_gated": (
                best_gated
            ),
            "comparison": (
                comparison
            ),
            "exact_calls": int(
                exact_calls[
                    "calls"
                ]
            ),
            "complete_enumeration": bool(
                complete_enum
            ),
            "exact_target_coverage": bool(
                exact_target_coverage
            ),
            "complete_case": bool(
                complete_case
            ),
            "outcome": (
                outcome
            ),
            "case_wall": float(
                time.perf_counter()
                - case_start
            ),
        }

        case_rows.append(
            row
        )

        print(
            "raw sk/ep           :",
            f"{raw_skeleton_count}/{raw_endpoint_count}",
        )

        print(
            "min C / canonical C :",
            f"{min_c} / {canonical_c}",
        )

        print(
            "gate sk/ep          :",
            f"{gated_skeleton_count}/{len(gated)}",
        )

        print(
            "compression ep/sk   :",
            f"{endpoint_compression*100:.1f}% / "
            f"{skeleton_compression*100:.1f}%",
        )

        print(
            "R4 cap passes       :",
            " -> ".join(
                str(
                    p[
                        "r4_cap"
                    ]
                )
                for p in (
                    enum_passes
                )
            ),
        )

        print(
            "canonical exact     :",
            f"{canonical_probe['status']}/"
            f"{canonical_probe.get('min_k2')}",
        )

        print(
            "best gated exact    :",
            (
                "None"
                if best_gated is None
                else str(
                    best_gated[
                        "exact_k2"
                    ]
                )
            ),
        )

        print(
            "comparison          :",
            (
                outcome
            ),
            (
                comparison[
                    "reason"
                ]
            ),
        )

        print(
            "complete            :",
            complete_case,
        )

        if best_gated is not None:
            print(
                "winner             :",
                f"gateRank={best_gated['gate_rank']} "
                f"C={best_gated['combined_lb']} "
                f"H={h_length} "
                f"K2={best_gated['exact_k2']} "
                f"total={h_length + int(best_gated['exact_k2'])}",
            )

    resolved = [
        row
        for row in case_rows
        if (
            row.get(
                "complete_case",
                False,
            )
            and row.get(
                "outcome"
            )
            in (
                "WIN",
                "TIE",
                "LOSS",
            )
        )
    ]

    wins = [
        row
        for row in resolved
        if row[
            "outcome"
        ]
        == "WIN"
    ]

    ties = [
        row
        for row in resolved
        if row[
            "outcome"
        ]
        == "TIE"
    ]

    losses = [
        row
        for row in resolved
        if row[
            "outcome"
        ]
        == "LOSS"
    ]

    inconclusive = [
        row
        for row in case_rows
        if row.get(
            "outcome"
        )
        == "INCONCLUSIVE"
    ]

    compressions = [
        float(
            row[
                "endpoint_compression"
            ]
        )
        for row in resolved
        if row.get(
            "endpoint_compression"
        )
        is not None
    ]

    mean_compression = (
        None
        if not compressions
        else (
            sum(
                compressions
            )
            / len(
                compressions
            )
        )
    )

    if (
        len(
            resolved
        )
        >= 2
        and len(
            wins
        )
        >= 2
        and not losses
    ):
        decision = (
            "FRESH_HARD_MIN_RESIDUAL_GATE_GENERALIZATION_SIGNAL"
        )

        note = (
            "At least two fresh hard scrambles independently preserve a strict "
            "exact K2 improvement inside the full minimum-combined-LB tie class, "
            "with no resolved loss. The residual gate now has multi-scramble "
            "generalization evidence, though it is still not a production hook."
        )

    elif losses:
        decision = (
            "FRESH_HARD_MIN_RESIDUAL_GATE_MIXED_WITH_LOSS"
        )

        note = (
            "At least one fully resolved fresh hard scramble is worse under the "
            "minimum-combined-LB gate than the canonical endpoint. The residual "
            "gate is not safe as a standalone global selection rule."
        )

    elif (
        resolved
        and not wins
    ):
        decision = (
            "FRESH_HARD_MIN_RESIDUAL_GATE_NO_STRICT_WIN"
        )

        note = (
            "Fresh resolved controls provide no strict exact K2 improvement over "
            "canonical. Keep the hard H10 control as a real but non-generalized "
            "positive example."
        )

    elif inconclusive:
        decision = (
            "FRESH_HARD_MIN_RESIDUAL_GATE_INCONCLUSIVE"
        )

        note = (
            "Too few fresh controls completed exact enumeration and comparison "
            "under the explicit resource caps. Increase only the binding cap for "
            "the incomplete case(s)."
        )

    else:
        decision = (
            "FRESH_HARD_MIN_RESIDUAL_GATE_WEAK_SIGNAL"
        )

        note = (
            "Fresh cases are resolved but do not yet meet the two-strict-win "
            "generalization threshold. More fresh controls are needed before any "
            "earlier residual propagation experiment."
        )

    print()
    print(
        "# AGGREGATE"
    )

    print(
        "resolved             :",
        f"{len(resolved)}/{len(case_rows)}",
    )

    print(
        "WIN / TIE / LOSS     :",
        f"{len(wins)} / {len(ties)} / {len(losses)}",
    )

    print(
        "inconclusive         :",
        len(
            inconclusive
        ),
    )

    print(
        "mean endpoint compression:",
        (
            "n/a"
            if mean_compression is None
            else f"{mean_compression*100:.1f}%"
        ),
    )

    print()
    print(
        "# DECISION"
    )

    print(
        decision
    )

    print(
        note
    )

    print()
    print(
        "NEXT:"
    )

    if decision == (
        "FRESH_HARD_MIN_RESIDUAL_GATE_GENERALIZATION_SIGNAL"
    ):
        print(
            "  v37.134 push a conservative residual bound one stage earlier than DR completion"
        )

        print(
            "  compare generated prefix count / DR endpoint count / exact K2 against post-DR-only gate"
        )

        print(
            "  retain full min-LB tie classes and exact replay fallback"
        )

    elif decision == (
        "FRESH_HARD_MIN_RESIDUAL_GATE_INCONCLUSIVE"
    ):
        print(
            "  rerun only incomplete case(s) with the binding cap increased"
        )

    else:
        print(
            "  do not push residual propagation earlier yet"
        )

    payload = {
        "version": (
            "v37.133"
        ),
        "mode": (
            "FRESH_HARD_MIN_RESIDUAL_GATE_VALIDATION"
        ),
        "stable_return_code": int(
            rc
        ),
        "config": {
            "fresh_count": int(
                local.fresh_count
            ),
            "fresh_seed": int(
                local.fresh_seed
            ),
            "scramble_length": int(
                local.scramble_length
            ),
            "h_offset": int(
                local.h_offset
            ),
            "initial_r4_cap": int(
                local.initial_r4_cap
            ),
            "prefix_node_cap": int(
                local.prefix_node_cap
            ),
            "case_time_limit": float(
                local.case_time_limit
            ),
            "exact_k2_probe_cap": int(
                local.exact_k2_probe_cap
            ),
            "exact_k2_call_cap_per_case": int(
                local.exact_k2_call_cap_per_case
            ),
        },
        "cases": (
            case_rows
        ),
        "aggregate": {
            "resolved": len(
                resolved
            ),
            "wins": len(
                wins
            ),
            "ties": len(
                ties
            ),
            "losses": len(
                losses
            ),
            "inconclusive": len(
                inconclusive
            ),
            "mean_endpoint_compression": (
                mean_compression
            ),
        },
        "decision": (
            decision
        ),
        "note": (
            note
        ),
        "production_changes": (
            "NONE"
        ),
    }

    out = Path(
        local.fresh_output
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

    return int(
        rc
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
