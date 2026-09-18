#!/usr/bin/env python3
"""
CubeLab v37.153 — ADAPTIVE RANDOM COVERAGE / FALLBACK AUDIT

Frozen experimental hard/global preset
---------------------------------------
    prefix_mode = fused
    r4_cap_mode = fullcap
    cert_mode   = incumbent RETEST

The preset is frozen after v37.152 broad regression.

Purpose
-------
This is NOT another speed micro-benchmark.

It asks the practical solver-coverage question:

    Starting from H=rootLB+3, can the frozen engine solve ordinary fresh
    25-move random scrambles?

Fallback policy
---------------
For each deterministic fresh scramble:

    H = rootLB + start_offset       (default +3)

    complete FUSED+FULLCAP H portfolio

    if NO DR endpoint:
        H <- H + 1
        retry monotonically

    if DR endpoints exist:
        run exact INCUMBENT RETEST certification
        require a replay-verified solved witness
        stop on success

Important scientific distinction
--------------------------------
Only an exact COMPLETE no-DR result advances H.

A node/time cut is NEVER interpreted as "no DR"; it is recorded as a resource
inconclusive case.

Default first coverage batch:
    100 deterministic reduced-random 25-move scrambles
    seed=20261054
    offsets +3 .. +5

This is practical coverage evidence, not a mathematical proof over all legal
cube states and not a uniform-random-state sampler.

PASS:
    solved = count
    replay verified = count
    zero resource-inconclusive cases

On 100/100 PASS:
    next step is a lighter-telemetry 1,000-case coverage audit.

Production/stable changes: NONE.
"""

from __future__ import annotations

import argparse
import json
import statistics
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

import benchmark_selectable_python_hard_global_v37_151 as v151
import audit_isolated_retest_cert_v37_144 as v144
import audit_hard_global_k2_residual_v37_131_1 as v1311
import audit_fast_stable_selective_terminal_macro_v37_114 as v114


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
    )

    ap.add_argument(
        "--fresh-count",
        type=int,
        default=100,
    )

    ap.add_argument(
        "--fresh-seed",
        type=int,
        default=20261054,
    )

    ap.add_argument(
        "--scramble-length",
        type=int,
        default=25,
    )

    ap.add_argument(
        "--start-h-offset",
        type=int,
        default=3,
    )

    ap.add_argument(
        "--max-h-offset",
        type=int,
        default=5,
    )

    ap.add_argument(
        "--portfolio-time-limit",
        type=float,
        default=30.0,
    )

    ap.add_argument(
        "--cert-time-limit",
        type=float,
        default=30.0,
    )

    ap.add_argument(
        "--prefix-node-cap",
        type=int,
        default=1000000,
    )

    ap.add_argument(
        "--k2-probe-cap",
        type=int,
        default=22,
    )

    ap.add_argument(
        "--terminal-call-cap",
        type=int,
        default=16000,
    )

    ap.add_argument(
        "--progress-every",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--analysis-output",
        default=(
            "reports/v37/"
            "adaptive_random_coverage_v37_153.json"
        ),
    )

    ns, remaining = ap.parse_known_args(
        argv
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

    ns.start_h_offset = max(
        0,
        int(
            ns.start_h_offset
        ),
    )

    ns.max_h_offset = max(
        int(
            ns.start_h_offset
        ),
        int(
            ns.max_h_offset
        ),
    )

    ns.portfolio_time_limit = max(
        1.0,
        float(
            ns.portfolio_time_limit
        ),
    )

    ns.cert_time_limit = max(
        1.0,
        float(
            ns.cert_time_limit
        ),
    )

    ns.prefix_node_cap = max(
        1,
        int(
            ns.prefix_node_cap
        ),
    )

    ns.k2_probe_cap = max(
        0,
        int(
            ns.k2_probe_cap
        ),
    )

    ns.terminal_call_cap = max(
        1,
        int(
            ns.terminal_call_cap
        ),
    )

    ns.progress_every = max(
        1,
        int(
            ns.progress_every
        ),
    )

    return (
        ns,
        remaining,
    )


def _portfolio_complete(
    portfolio,
):
    if portfolio is None:
        return False

    return bool(
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


def _percentile(
    values,
    fraction,
):
    if not values:
        return None

    ordered = sorted(
        float(x)
        for x in values
    )

    if len(
        ordered
    ) == 1:
        return ordered[
            0
        ]

    pos = (
        float(
            fraction
        )
        * (
            len(
                ordered
            )
            - 1
        )
    )

    lo = int(
        pos
    )

    hi = min(
        lo + 1,
        len(
            ordered
        )
        - 1,
    )

    frac = (
        pos
        - lo
    )

    return (
        ordered[
            lo
        ]
        * (
            1.0
            - frac
        )
        + ordered[
            hi
        ]
        * frac
    )


def main():
    local, remaining = _parse_args(
        sys.argv[
            1:
        ]
    )

    # Capture the unchanged stable solver / exact terminal finder.
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
        "# CubeLab v37.153 - "
        "ADAPTIVE RANDOM COVERAGE / FALLBACK AUDIT"
    )

    print(
        "frozen preset        :",
        "FUSED + FULLCAP + INCUMBENT RETEST",
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
        "H offsets            :",
        f"+{local.start_h_offset}"
        f"..+{local.max_h_offset}",
    )

    print(
        "portfolio cap        :",
        f"{local.prefix_node_cap:,} prefix nodes / "
        f"{local.portfolio_time_limit:.1f}s per H",
    )

    print(
        "cert cap             :",
        f"K2<={local.k2_probe_cap}, "
        f"{local.terminal_call_cap:,} calls, "
        f"{local.cert_time_limit:.1f}s",
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

    solver = (
        v1311._CAPTURED_SOLVER
    )

    if solver is None:
        raise RuntimeError(
            "failed to capture solver"
        )

    oracle = (
        v114._ORACLE
    )

    if oracle is None:
        raise RuntimeError(
            "v37.114 oracle unavailable"
        )

    if v1311._BASE_FIND_NO_SHADOW is None:
        raise RuntimeError(
            "terminal finder unavailable"
        )

    scrambles = v151._generate_scrambles(
        fresh_count=int(
            local.fresh_count
        ),
        fresh_seed=int(
            local.fresh_seed
        ),
        scramble_length=int(
            local.scramble_length
        ),
    )

    print()
    print(
        "# COVERAGE CASES"
    )

    cases = []

    solved_count = 0
    replay_count = 0

    offset_hist = Counter()
    root_lb_hist = Counter()
    k2_hist = Counter()
    total_length_hist = Counter()

    no_dr_by_offset = Counter()

    resource_inconclusive = []
    complete_no_dr_through_max = []
    replay_failures = []

    case_walls = []
    solved_walls = []

    fallback_cases = []

    total_prefix_nodes = 0
    total_terminal_calls = 0

    for case_index, moves in enumerate(
        scrambles,
        start=1,
    ):
        case_start = (
            time.perf_counter()
        )

        poses, q = (
            v1311._scramble_state(
                solver=solver,
                moves=moves,
            )
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

        root_lb_hist[
            int(
                root_lb
            )
        ] += 1

        attempts = []

        solved = False
        replay_ok = False
        selected_offset = None
        selected_h = None
        exact_k2 = None
        total_length = None

        case_resource_issue = None

        for offset in range(
            int(
                local.start_h_offset
            ),
            int(
                local.max_h_offset
            )
            + 1,
        ):
            h_length = int(
                root_lb
                + int(
                    offset
                )
            )

            attempt_start = (
                time.perf_counter()
            )

            (
                portfolio,
                final_cap,
                passes,
            ) = v151._enumerate_mode(
                prefix_mode="fused",
                r4_cap_mode="fullcap",
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
                initial_r4_cap=192,
                prefix_node_cap=int(
                    local.prefix_node_cap
                ),
                time_limit=float(
                    local.portfolio_time_limit
                ),
            )

            portfolio_wall = (
                time.perf_counter()
                - attempt_start
            )

            if portfolio is None:
                case_resource_issue = (
                    "portfolio_missing"
                )

                attempts.append(
                    {
                        "offset": int(
                            offset
                        ),
                        "h_length": int(
                            h_length
                        ),
                        "outcome": (
                            "RESOURCE_INCONCLUSIVE"
                        ),
                        "reason": (
                            "portfolio_missing"
                        ),
                        "portfolio_wall": float(
                            portfolio_wall
                        ),
                        "passes": passes,
                    }
                )

                break

            stats = (
                portfolio[
                    "stats"
                ]
            )

            prefix_nodes = int(
                stats[
                    "prefix_nodes"
                ]
            )

            raw_endpoints = len(
                portfolio[
                    "endpoints"
                ]
            )

            total_prefix_nodes += int(
                prefix_nodes
            )

            complete = (
                _portfolio_complete(
                    portfolio
                )
            )

            attempt = {
                "offset": int(
                    offset
                ),
                "h_length": int(
                    h_length
                ),
                "portfolio_complete": bool(
                    complete
                ),
                "prefix_nodes": int(
                    prefix_nodes
                ),
                "r4_leaf_states": int(
                    stats[
                        "r4_leaf_states"
                    ]
                ),
                "r4_words_considered": int(
                    stats[
                        "r4_words_considered"
                    ]
                ),
                "raw_endpoints": int(
                    raw_endpoints
                ),
                "final_r4_cap": int(
                    final_cap
                ),
                "portfolio_wall": float(
                    portfolio_wall
                ),
                "passes": passes,
            }

            if not complete:
                case_resource_issue = (
                    "portfolio_cut"
                )

                attempt[
                    "outcome"
                ] = (
                    "RESOURCE_INCONCLUSIVE"
                )

                attempt[
                    "reason"
                ] = {
                    "node_cap": bool(
                        portfolio[
                            "cuts"
                        ][
                            "node_cap"
                        ]
                    ),
                    "time": bool(
                        portfolio[
                            "cuts"
                        ][
                            "time"
                        ]
                    ),
                    "r4_cap_truncated": int(
                        stats[
                            "r4_cap_truncated"
                        ]
                    ),
                }

                attempts.append(
                    attempt
                )

                break

            if raw_endpoints == 0:
                no_dr_by_offset[
                    int(
                        offset
                    )
                ] += 1

                attempt[
                    "outcome"
                ] = (
                    "COMPLETE_NO_DR"
                )

                attempts.append(
                    attempt
                )

                continue

            cert_start = (
                time.perf_counter()
            )

            cert_deadline = (
                cert_start
                + float(
                    local.cert_time_limit
                )
            )

            cert = (
                v144._run_retest(
                    solver=solver,
                    scrambled_poses=tuple(
                        poses
                    ),
                    portfolio=portfolio,
                    probe_cap=int(
                        local.k2_probe_cap
                    ),
                    call_cap=int(
                        local.terminal_call_cap
                    ),
                    deadline=(
                        cert_deadline
                    ),
                )
            )

            cert_wall = (
                time.perf_counter()
                - cert_start
            )

            terminal_calls = int(
                cert.get(
                    "terminal_calls",
                    0,
                )
            )

            total_terminal_calls += int(
                terminal_calls
            )

            certified = bool(
                cert.get(
                    "certified",
                    False,
                )
            )

            witness_replay = bool(
                cert.get(
                    "witness_replay",
                    False,
                )
            )

            attempt[
                "cert_wall"
            ] = float(
                cert_wall
            )

            attempt[
                "terminal_calls"
            ] = int(
                terminal_calls
            )

            attempt[
                "certified"
            ] = bool(
                certified
            )

            attempt[
                "witness_replay"
            ] = bool(
                witness_replay
            )

            attempt[
                "cert_bad_status"
            ] = cert.get(
                "bad_status"
            )

            if not certified:
                case_resource_issue = (
                    "certification_inconclusive"
                )

                attempt[
                    "outcome"
                ] = (
                    "RESOURCE_INCONCLUSIVE"
                )

                attempts.append(
                    attempt
                )

                break

            cert_k2 = cert.get(
                "exact_best_k2"
            )

            if cert_k2 is None:
                case_resource_issue = (
                    "certified_without_k2"
                )

                attempt[
                    "outcome"
                ] = (
                    "CONTRACT_FAIL"
                )

                attempts.append(
                    attempt
                )

                break

            if not witness_replay:
                attempt[
                    "outcome"
                ] = (
                    "REPLAY_FAIL"
                )

                attempts.append(
                    attempt
                )

                replay_failures.append(
                    int(
                        case_index
                    )
                )

                break

            exact_k2 = int(
                cert_k2
            )

            total_length = int(
                h_length
                + exact_k2
            )

            solved = True
            replay_ok = True

            selected_offset = int(
                offset
            )

            selected_h = int(
                h_length
            )

            attempt[
                "exact_k2"
            ] = int(
                exact_k2
            )

            attempt[
                "total_length"
            ] = int(
                total_length
            )

            attempt[
                "outcome"
            ] = (
                "SOLVED_REPLAY_PASS"
            )

            attempts.append(
                attempt
            )

            break

        case_wall = (
            time.perf_counter()
            - case_start
        )

        case_walls.append(
            float(
                case_wall
            )
        )

        if solved:
            solved_count += 1
            replay_count += int(
                replay_ok
            )

            solved_walls.append(
                float(
                    case_wall
                )
            )

            offset_hist[
                int(
                    selected_offset
                )
            ] += 1

            k2_hist[
                int(
                    exact_k2
                )
            ] += 1

            total_length_hist[
                int(
                    total_length
                )
            ] += 1

            if int(
                selected_offset
            ) > int(
                local.start_h_offset
            ):
                fallback_cases.append(
                    {
                        "case": int(
                            case_index
                        ),
                        "scramble": list(
                            moves
                        ),
                        "root_lb": int(
                            root_lb
                        ),
                        "selected_offset": int(
                            selected_offset
                        ),
                        "selected_h": int(
                            selected_h
                        ),
                        "exact_k2": int(
                            exact_k2
                        ),
                        "total_length": int(
                            total_length
                        ),
                        "wall": float(
                            case_wall
                        ),
                    }
                )

        elif case_resource_issue is not None:
            resource_inconclusive.append(
                {
                    "case": int(
                        case_index
                    ),
                    "scramble": list(
                        moves
                    ),
                    "root_lb": int(
                        root_lb
                    ),
                    "reason": str(
                        case_resource_issue
                    ),
                    "attempts": attempts,
                    "wall": float(
                        case_wall
                    ),
                }
            )

        elif (
            attempts
            and all(
                row.get(
                    "outcome"
                )
                == "COMPLETE_NO_DR"
                for row in attempts
            )
            and int(
                attempts[
                    -1
                ][
                    "offset"
                ]
            )
            == int(
                local.max_h_offset
            )
        ):
            complete_no_dr_through_max.append(
                {
                    "case": int(
                        case_index
                    ),
                    "scramble": list(
                        moves
                    ),
                    "root_lb": int(
                        root_lb
                    ),
                    "attempts": attempts,
                    "wall": float(
                        case_wall
                    ),
                }
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
            "attempts": attempts,
            "solved": bool(
                solved
            ),
            "replay_pass": bool(
                replay_ok
            ),
            "selected_offset": (
                selected_offset
            ),
            "selected_h": (
                selected_h
            ),
            "exact_k2": (
                exact_k2
            ),
            "total_length": (
                total_length
            ),
            "resource_issue": (
                case_resource_issue
            ),
            "wall": float(
                case_wall
            ),
        }

        cases.append(
            row
        )

        if solved:
            if int(
                selected_offset
            ) == int(
                local.start_h_offset
            ):
                status = (
                    f"SOLVED +{selected_offset} "
                    f"H={selected_h} K2={exact_k2} "
                    f"T={total_length}"
                )
            else:
                status = (
                    f"FALLBACK_SOLVED +{selected_offset} "
                    f"H={selected_h} K2={exact_k2} "
                    f"T={total_length}"
                )

        elif case_resource_issue:
            status = (
                "RESOURCE_INCONCLUSIVE "
                + str(
                    case_resource_issue
                )
            )

        elif (
            int(
                case_index
            )
            in replay_failures
        ):
            status = (
                "REPLAY_FAIL"
            )

        else:
            status = (
                f"NO_DR_THROUGH_+"
                f"{local.max_h_offset}"
            )

        print(
            f"[{case_index:03d}/"
            f"{local.fresh_count:03d}] "
            f"rootLB={root_lb} "
            f"{status} "
            f"wall={case_wall:.3f}s"
        )

        if (
            int(
                case_index
            )
            % int(
                local.progress_every
            )
            == 0
            or int(
                case_index
            )
            == int(
                local.fresh_count
            )
        ):
            print(
                "  PROGRESS:",
                f"solved={solved_count}/"
                f"{case_index}",
                f"fallback={len(fallback_cases)}",
                f"resource={len(resource_inconclusive)}",
                f"noDRmax={len(complete_no_dr_through_max)}",
                f"replayFail={len(replay_failures)}",
            )

    total_wall = sum(
        case_walls
    )

    median_wall = (
        None
        if not case_walls
        else float(
            statistics.median(
                case_walls
            )
        )
    )

    p90_wall = _percentile(
        case_walls,
        0.90,
    )

    p95_wall = _percentile(
        case_walls,
        0.95,
    )

    max_wall = (
        None
        if not case_walls
        else max(
            case_walls
        )
    )

    max_wall_case = (
        None
        if not cases
        else max(
            cases,
            key=lambda row: float(
                row[
                    "wall"
                ]
            ),
        )[
            "case"
        ]
    )

    start_solved = int(
        offset_hist.get(
            int(
                local.start_h_offset
            ),
            0,
        )
    )

    fallback_solved = int(
        solved_count
        - start_solved
    )

    solved_rate = (
        float(
            solved_count
        )
        / float(
            local.fresh_count
        )
    )

    print()
    print(
        "# COVERAGE SUMMARY"
    )

    print(
        "solved / total       :",
        f"{solved_count}/"
        f"{local.fresh_count} "
        f"({solved_rate*100:.1f}%)",
    )

    print(
        "replay passed        :",
        f"{replay_count}/"
        f"{local.fresh_count}",
    )

    print(
        f"solved at +"
        f"{local.start_h_offset}:",
        start_solved,
    )

    print(
        "fallback solved      :",
        fallback_solved,
    )

    print(
        "offset histogram     :",
        dict(
            sorted(
                offset_hist.items()
            )
        ),
    )

    print(
        "complete no-DR by off:",
        dict(
            sorted(
                no_dr_by_offset.items()
            )
        ),
    )

    print(
        "resource inconclusive:",
        len(
            resource_inconclusive
        ),
    )

    print(
        "no-DR through max    :",
        len(
            complete_no_dr_through_max
        ),
    )

    print(
        "replay failures      :",
        len(
            replay_failures
        ),
    )

    print(
        "root-LB histogram    :",
        dict(
            sorted(
                root_lb_hist.items()
            )
        ),
    )

    print(
        "exact K2 histogram   :",
        dict(
            sorted(
                k2_hist.items()
            )
        ),
    )

    print(
        "total-length hist    :",
        dict(
            sorted(
                total_length_hist.items()
            )
        ),
    )

    print(
        "case wall median/p90 :",
        (
            "n/a"
            if median_wall is None
            else f"{median_wall:.3f}s / "
            f"{p90_wall:.3f}s"
        ),
    )

    print(
        "case wall p95/max    :",
        (
            "n/a"
            if p95_wall is None
            else f"{p95_wall:.3f}s / "
            f"{max_wall:.3f}s "
            f"(case {max_wall_case})"
        ),
    )

    print(
        "case wall total      :",
        f"{total_wall:.3f}s",
    )

    print(
        "prefix nodes total   :",
        f"{total_prefix_nodes:,}",
    )

    print(
        "terminal calls total :",
        f"{total_terminal_calls:,}",
    )

    if replay_failures:
        decision = (
            "ADAPTIVE_RANDOM_COVERAGE_REPLAY_FAIL"
        )

        note = (
            "At least one certified search result failed the required "
            "full-cube replay. This is a correctness failure; isolate it "
            "before any coverage claim."
        )

    elif resource_inconclusive:
        decision = (
            "ADAPTIVE_RANDOM_COVERAGE_RESOURCE_INCONCLUSIVE"
        )

        note = (
            "At least one random case hit an explicit portfolio/certification "
            "resource cap. A bounded cut is not UNSAT and not a no-DR result. "
            "Rerun only the listed case(s) with the binding cap increased."
        )

    elif complete_no_dr_through_max:
        decision = (
            "ADAPTIVE_RANDOM_COVERAGE_HORIZON_LIMIT_FOUND"
        )

        note = (
            "At least one case completed every requested H offset without a "
            "DR endpoint. This does not prove the state unsolvable; it proves "
            "the current adaptive horizon is insufficient. Extend only those "
            "cases beyond the current max offset."
        )

    elif (
        solved_count
        == int(
            local.fresh_count
        )
        and replay_count
        == int(
            local.fresh_count
        )
    ):
        decision = (
            "ADAPTIVE_RANDOM_COVERAGE_100_PASS"
            if int(
                local.fresh_count
            )
            == 100
            else "ADAPTIVE_RANDOM_COVERAGE_PASS"
        )

        note = (
            "Every deterministic reduced-random scramble in this batch was "
            "solved by the frozen FUSED+FULLCAP+RETEST engine under the "
            "adaptive H fallback, and every returned witness passed replay. "
            "This is strong practical random-scramble coverage evidence, "
            "not a proof over all legal cube states."
        )

    else:
        decision = (
            "ADAPTIVE_RANDOM_COVERAGE_UNRESOLVED"
        )

        note = (
            "One or more cases remain unresolved for a reason not classified "
            "as replay failure, resource cut, or complete no-DR through the "
            "maximum H offset. Inspect those case records before proceeding."
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

    if decision in (
        "ADAPTIVE_RANDOM_COVERAGE_100_PASS",
        "ADAPTIVE_RANDOM_COVERAGE_PASS",
    ):
        print(
            "  if this was the 100-case batch, run a lighter-telemetry "
            "1,000-case deterministic coverage audit"
        )

        print(
            "  preserve offset/fallback histogram and replay requirement"
        )

        print(
            "  do not resume micro-optimization before the coverage result"
        )

    elif decision == (
        "ADAPTIVE_RANDOM_COVERAGE_RESOURCE_INCONCLUSIVE"
    ):
        print(
            "  rerun only resource-inconclusive cases with the binding "
            "node/time/call cap increased"
        )

    elif decision == (
        "ADAPTIVE_RANDOM_COVERAGE_HORIZON_LIMIT_FOUND"
    ):
        print(
            "  extend only complete no-DR cases to larger H offsets "
            "(e.g. +6, +7)"
        )

    else:
        print(
            "  isolate correctness/unresolved cases before broader coverage"
        )

    payload = {
        "version": (
            "v37.153"
        ),
        "mode": (
            "ADAPTIVE_RANDOM_COVERAGE_FALLBACK_AUDIT"
        ),
        "stable_return_code": int(
            rc
        ),
        "frozen_preset": {
            "prefix_mode": (
                "fused"
            ),
            "r4_cap_mode": (
                "fullcap"
            ),
            "cert_mode": (
                "retest"
            ),
        },
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
            "start_h_offset": int(
                local.start_h_offset
            ),
            "max_h_offset": int(
                local.max_h_offset
            ),
            "portfolio_time_limit": float(
                local.portfolio_time_limit
            ),
            "cert_time_limit": float(
                local.cert_time_limit
            ),
            "prefix_node_cap": int(
                local.prefix_node_cap
            ),
            "k2_probe_cap": int(
                local.k2_probe_cap
            ),
            "terminal_call_cap": int(
                local.terminal_call_cap
            ),
        },
        "cases": cases,
        "summary": {
            "solved": int(
                solved_count
            ),
            "total": int(
                local.fresh_count
            ),
            "solved_rate": float(
                solved_rate
            ),
            "replay_passed": int(
                replay_count
            ),
            "solved_at_start_offset": int(
                start_solved
            ),
            "fallback_solved": int(
                fallback_solved
            ),
            "offset_histogram": {
                str(k): int(v)
                for k, v in sorted(
                    offset_hist.items()
                )
            },
            "complete_no_dr_by_offset": {
                str(k): int(v)
                for k, v in sorted(
                    no_dr_by_offset.items()
                )
            },
            "resource_inconclusive_count": len(
                resource_inconclusive
            ),
            "complete_no_dr_through_max_count": len(
                complete_no_dr_through_max
            ),
            "replay_failure_count": len(
                replay_failures
            ),
            "root_lb_histogram": {
                str(k): int(v)
                for k, v in sorted(
                    root_lb_hist.items()
                )
            },
            "exact_k2_histogram": {
                str(k): int(v)
                for k, v in sorted(
                    k2_hist.items()
                )
            },
            "total_length_histogram": {
                str(k): int(v)
                for k, v in sorted(
                    total_length_hist.items()
                )
            },
            "case_wall_median": (
                median_wall
            ),
            "case_wall_p90": (
                p90_wall
            ),
            "case_wall_p95": (
                p95_wall
            ),
            "case_wall_max": (
                max_wall
            ),
            "case_wall_max_case": (
                max_wall_case
            ),
            "case_wall_total": float(
                total_wall
            ),
            "prefix_nodes_total": int(
                total_prefix_nodes
            ),
            "terminal_calls_total": int(
                total_terminal_calls
            ),
            "fallback_cases": (
                fallback_cases
            ),
            "resource_inconclusive": (
                resource_inconclusive
            ),
            "complete_no_dr_through_max": (
                complete_no_dr_through_max
            ),
            "replay_failures": (
                replay_failures
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
        local.analysis_output
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
    raise SystemExit(
        main()
    )
