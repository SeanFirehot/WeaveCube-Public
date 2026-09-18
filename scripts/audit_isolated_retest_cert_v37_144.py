#!/usr/bin/env python3
"""
CubeLab v37.144 — ISOLATED ZERO vs INCUMBENT-RETEST EXACT CERTIFICATION A/B

Motivation
----------
v37.143 isolated fresh-process result:

    ZERO calls     : 1785
    LBSTART calls  :  336   (0.188x)
    certification wall LB/ZERO : 0.924x
    pipeline wall LB/ZERO      : 0.963x

So skipping cheap low-budget calls is mathematically clean but does NOT turn
the 81% call reduction into a material end-to-end wall win.

Likely reason:
    the remaining high-budget K2 calls dominate wall.

v37.144 tests a different exact certification schedule that avoids incremental
budget ladders almost entirely.

INCUMBENT_RETEST
----------------
Endpoints are ordered by admissible combined K2 lower bound C.

Start:
    incumbent U = probe_cap + 1
    no witness yet

For each endpoint with C < U:
    test it ONCE at budget U-1

    MISS:
        endpoint is proven unable to beat U

    FOUND with solution total t < U:
        verify replay
        U = t
        restart scan at the tighter incumbent

Previously proven misses are retained:
    if an endpoint missed at budget B, it is also proven miss for every
    smaller budget, so it is not re-run unnecessarily.

Termination:
    a full pass finds no improvement.

Then every endpoint either:
    C >= U
or:
    has a proven MISS at U-1

and we possess a solved witness of length U.

Therefore U is the exact global minimum K2 over the fixed-H endpoint portfolio.

This is branch-and-bound by FEASIBILITY RETEST rather than iterative depth
enumeration.

A/B
---
Every arm run is a FRESH PYTHON PROCESS.

    ZERO:
        existing v37.140 zero-start exact branch-and-bound

    RETEST:
        incumbent-retest algorithm above

Default repeats=2 per arm/case.

Primary metrics:
    exact equality
    terminal call count
    isolated certification wall
    isolated hard-pipeline wall

Production changes: NONE.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if SRC.is_dir():
    s = str(SRC)
    if s not in sys.path:
        sys.path.insert(0, s)

import audit_post_dr_exact_cert_ab_v37_141 as v141
import audit_h_allocation_residual_v37_140 as v140
import audit_fresh_p2plus1_secondary_v37_137 as v137
import audit_fresh_hard_residual_gate_v37_133 as v133
import audit_hard_global_k2_residual_v37_131_1 as v1311
import audit_post_dr_k2_deficit_v37_128 as v128
import audit_fast_stable_selective_terminal_macro_v37_114 as v114


RESULT_MARKER = "__V37144_RESULT__="


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
    )

    ap.add_argument(
        "--source-v142",
        default="reports/v37/fresh_lbstart_exact_cert_v37_142.json",
    )
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--case-time-limit", type=float, default=180.0)
    ap.add_argument("--initial-r4-cap", type=int, default=192)
    ap.add_argument("--prefix-node-cap", type=int, default=200000)
    ap.add_argument("--k2-probe-cap", type=int, default=22)
    ap.add_argument("--terminal-call-cap", type=int, default=16000)
    ap.add_argument(
        "--analysis-output",
        default="reports/v37/isolated_retest_cert_ab_v37_144.json",
    )

    ap.add_argument("--worker", action="store_true")
    ap.add_argument(
        "--worker-arm",
        choices=("ZERO", "RETEST"),
        default=None,
    )
    ap.add_argument("--worker-case", type=int, default=None)

    ns, remaining = ap.parse_known_args(argv)

    ns.repeats = max(1, int(ns.repeats))
    ns.case_time_limit = max(1.0, float(ns.case_time_limit))
    ns.initial_r4_cap = max(1, int(ns.initial_r4_cap))
    ns.prefix_node_cap = max(1, int(ns.prefix_node_cap))
    ns.k2_probe_cap = max(0, int(ns.k2_probe_cap))
    ns.terminal_call_cap = max(1, int(ns.terminal_call_cap))

    return ns, remaining


def _source_case(source, case_index):
    for row in source.get("cases", []):
        if int(row.get("case", -1)) == int(case_index):
            return row

    raise KeyError(
        f"source case {case_index} not found"
    )


def _expected_k2(row):
    value = row.get(
        "zero",
        {},
    ).get(
        "exact_best_k2"
    )

    if value is None:
        value = row.get(
            "lbstart",
            {},
        ).get(
            "exact_best_k2"
        )

    return (
        None
        if value is None
        else int(value)
    )


def _portfolio_complete(portfolio):
    if portfolio is None:
        return False

    return bool(
        not portfolio["cuts"]["node_cap"]
        and not portfolio["cuts"]["time"]
        and int(
            portfolio["stats"]["r4_cap_truncated"]
        )
        == 0
    )


def _first_skeletons(portfolio):
    out = {}

    for skeleton in portfolio["skeletons"]:
        key = skeleton["endpoint_key"]

        if key not in out:
            out[key] = skeleton

    return out


def _one_budget_find(
    *,
    solver,
    endpoint,
    budget,
):
    try:
        result = v1311._BASE_FIND_NO_SHADOW(
            solver,
            tuple(endpoint["end_poses"]),
            int(endpoint["end_q"]),
            (0,),
            endpoint["final_face"],
            False,
            int(budget),
        )

    except Exception as exc:
        return {
            "status": "CUT",
            "error": repr(exc),
        }

    if result is None:
        return {
            "status": "MISS",
        }

    term_h = tuple(
        result.get(
            "h_tail",
            (),
        )
    )

    term_k2 = tuple(
        result.get(
            "k2_word",
            (),
        )
    )

    total = int(
        result.get(
            "total",
            len(term_h) + len(term_k2),
        )
    )

    if term_h:
        return {
            "status": "CONTRACT_H_NONZERO",
        }

    if total > int(budget):
        return {
            "status": "CONTRACT_OVER_BUDGET",
        }

    return {
        "status": "FOUND",
        "exact_k2": int(total),
        "k2_word": list(term_k2),
    }


def _verify(
    *,
    scrambled_poses,
    skeleton,
    probe,
):
    if (
        skeleton is None
        or probe.get("status") != "FOUND"
        or probe.get("exact_k2") is None
    ):
        return False

    return v133._verify_solution(
        scrambled_poses=tuple(scrambled_poses),
        skeleton=skeleton,
        k2_word=tuple(probe.get("k2_word", ())),
    )


def _run_retest(
    *,
    solver,
    scrambled_poses,
    portfolio,
    probe_cap,
    call_cap,
    deadline,
):
    items = sorted(
        portfolio["endpoints"].items(),
        key=lambda item: (
            int(item[1]["combined_lb"]),
            int(item[1]["p2lb"]),
        ),
    )

    first_skeleton = _first_skeletons(
        portfolio
    )

    if not items:
        return {
            "certified": True,
            "exact_best_k2": None,
            "winner_c": None,
            "terminal_calls": 0,
            "endpoint_tests": 0,
            "passes": 0,
            "proof_reuse_skips": 0,
            "bound_skipped": 0,
            "improvements": [],
            "bad_status": None,
        }

    # Initial upper search threshold is probe_cap.
    incumbent = int(probe_cap) + 1
    witness = None
    winner_c = None
    winner_key = None

    calls = 0
    endpoint_tests = 0
    passes = 0
    proof_reuse_skips = 0
    improvements = []
    bad = None

    # Highest budget at which this endpoint is proven MISS.
    miss_bound = {}

    while True:
        if (
            time.perf_counter() >= deadline
            or int(calls) >= int(call_cap)
        ):
            bad = "RESOURCE_CUT"
            break

        passes += 1
        improved = False

        for key, endpoint in items:
            c = int(endpoint["combined_lb"])

            # Sorted by C: all later endpoints are also bound-skipped.
            if c >= int(incumbent):
                break

            target = int(incumbent) - 1

            proven = miss_bound.get(
                key
            )

            if (
                proven is not None
                and int(proven) >= int(target)
            ):
                proof_reuse_skips += 1
                continue

            if (
                time.perf_counter() >= deadline
                or int(calls) >= int(call_cap)
            ):
                bad = "RESOURCE_CUT"
                break

            calls += 1
            endpoint_tests += 1

            probe = _one_budget_find(
                solver=solver,
                endpoint=endpoint,
                budget=int(target),
            )

            status = str(
                probe["status"]
            )

            if status == "MISS":
                miss_bound[key] = max(
                    int(target),
                    int(
                        miss_bound.get(
                            key,
                            -1,
                        )
                    ),
                )
                continue

            if status != "FOUND":
                bad = status
                break

            exact_k2 = int(
                probe["exact_k2"]
            )

            if exact_k2 >= int(incumbent):
                bad = "FOUND_NOT_IMPROVING"
                break

            solved = _verify(
                scrambled_poses=tuple(scrambled_poses),
                skeleton=first_skeleton.get(key),
                probe=probe,
            )

            if not solved:
                bad = "REPLAY_FAIL"
                break

            old = int(incumbent)
            incumbent = int(exact_k2)
            witness = probe
            winner_c = int(c)
            winner_key = key

            improvements.append(
                {
                    "old_incumbent": int(old),
                    "new_incumbent": int(incumbent),
                    "combined_lb": int(c),
                    "target_budget": int(target),
                }
            )

            improved = True
            break

        if bad is not None:
            break

        if improved:
            # Restart at tighter U.  Existing MISS proofs remain valid.
            continue

        # Full scan at current incumbent found no better solution.
        if witness is None:
            bad = "NO_WITNESS_WITHIN_CAP"
            break

        break

    bound_skipped = 0

    if bad is None:
        bound_skipped = sum(
            1
            for _key, endpoint in items
            if int(endpoint["combined_lb"]) >= int(incumbent)
        )

    return {
        "certified": bool(bad is None),
        "exact_best_k2": (
            None
            if bad is not None
            else int(incumbent)
        ),
        "winner_c": (
            None
            if bad is not None
            else int(winner_c)
        ),
        "winner_key_repr": (
            None
            if winner_key is None
            else repr(winner_key)
        ),
        "terminal_calls": int(calls),
        "endpoint_tests": int(endpoint_tests),
        "passes": int(passes),
        "proof_reuse_skips": int(proof_reuse_skips),
        "bound_skipped": int(bound_skipped),
        "improvements": improvements,
        "bad_status": bad,
        "witness_replay": bool(
            bad is None
            and witness is not None
        ),
    }


def _worker_main(local, remaining):
    if local.worker_arm is None or local.worker_case is None:
        raise SystemExit(
            "--worker requires --worker-arm and --worker-case"
        )

    source_path = Path(
        local.source_v142
    )

    source = json.loads(
        source_path.read_text(
            encoding="utf-8",
        )
    )

    source_row = _source_case(
        source,
        int(local.worker_case),
    )

    if not source_row.get(
        "informative",
        False,
    ):
        raise SystemExit(
            f"case {local.worker_case} is not informative"
        )

    moves = tuple(
        source_row["scramble"]
    )

    expected_root_lb = int(
        source_row["root_lb"]
    )

    h_length = int(
        source_row["h_length"]
    )

    expected_raw = int(
        source_row["raw_endpoints"]
    )

    expected_k2 = _expected_k2(
        source_row
    )

    v114._combined_installer = (
        v1311._v37131_combined_installer
    )

    sys.argv = [
        sys.argv[0],
        *remaining,
    ]

    bootstrap_t0 = time.perf_counter()
    rc = v114.main()
    bootstrap_wall = (
        time.perf_counter()
        - bootstrap_t0
    )

    solver = v1311._CAPTURED_SOLVER

    if solver is None:
        raise RuntimeError(
            "failed to capture solver"
        )

    oracle = v114._ORACLE

    if oracle is None:
        raise RuntimeError(
            "v37.114 oracle unavailable"
        )

    if v1311._BASE_FIND_NO_SHADOW is None:
        raise RuntimeError(
            "terminal finder unavailable"
        )

    v128._BASE_FIND_NO_SHADOW = (
        v1311._BASE_FIND_NO_SHADOW
    )

    poses, q = v1311._scramble_state(
        solver=solver,
        moves=moves,
    )

    root_lb = int(
        v114.v24.grc.ts.phase1_lb(
            int(q),
            solver.cos,
            solver.eos,
        )
    )

    if int(root_lb) != int(expected_root_lb):
        raise RuntimeError(
            f"root LB mismatch {root_lb} != {expected_root_lb}"
        )

    hard_t0 = time.perf_counter()
    portfolio_t0 = hard_t0

    portfolio, r4_cap, passes, _extra = (
        v137._auto_complete_portfolio(
            solver=solver,
            oracle=oracle,
            poses=tuple(poses),
            q=int(q),
            h_length=int(h_length),
            initial_r4_cap=int(local.initial_r4_cap),
            prefix_node_cap=int(local.prefix_node_cap),
            time_limit=float(local.case_time_limit),
        )
    )

    portfolio_wall = (
        time.perf_counter()
        - portfolio_t0
    )

    if not _portfolio_complete(
        portfolio
    ):
        result = {
            "arm": local.worker_arm,
            "case": int(local.worker_case),
            "complete": False,
            "reason": "portfolio_incomplete",
            "bootstrap_wall": float(bootstrap_wall),
            "portfolio_wall": float(portfolio_wall),
            "passes": passes,
        }

        print(
            RESULT_MARKER
            + json.dumps(
                result,
                ensure_ascii=False,
                default=repr,
            )
        )

        return 0

    raw_endpoints = len(
        portfolio["endpoints"]
    )

    raw_identity = bool(
        int(raw_endpoints)
        == int(expected_raw)
    )

    deadline = (
        hard_t0
        + float(local.case_time_limit)
    )

    cert_t0 = time.perf_counter()

    if local.worker_arm == "ZERO":
        cert = v140._certify_fixed_h(
            solver=solver,
            scrambled_poses=tuple(poses),
            portfolio=portfolio,
            k2_probe_cap=int(local.k2_probe_cap),
            exact_call_cap=int(local.terminal_call_cap),
            deadline=deadline,
        )

        terminal_calls = int(
            cert.get(
                "exact_budget_calls",
                0,
            )
        )

    else:
        cert = _run_retest(
            solver=solver,
            scrambled_poses=tuple(poses),
            portfolio=portfolio,
            probe_cap=int(local.k2_probe_cap),
            call_cap=int(local.terminal_call_cap),
            deadline=deadline,
        )

        terminal_calls = int(
            cert.get(
                "terminal_calls",
                0,
            )
        )

    cert_wall = (
        time.perf_counter()
        - cert_t0
    )

    pipeline_wall = (
        time.perf_counter()
        - hard_t0
    )

    exact_k2 = cert.get(
        "exact_best_k2"
    )

    certified = bool(
        cert.get(
            "certified",
            False,
        )
    )

    exact_match = bool(
        certified
        and exact_k2 is not None
        and expected_k2 is not None
        and int(exact_k2) == int(expected_k2)
    )

    complete = bool(
        raw_identity
        and exact_match
    )

    result = {
        "arm": local.worker_arm,
        "case": int(local.worker_case),
        "stable_return_code": int(rc),
        "root_lb": int(root_lb),
        "h_length": int(h_length),
        "expected_raw": int(expected_raw),
        "raw_endpoints": int(raw_endpoints),
        "raw_identity": bool(raw_identity),
        "expected_k2": expected_k2,
        "exact_k2": (
            None
            if exact_k2 is None
            else int(exact_k2)
        ),
        "exact_match": bool(exact_match),
        "certified": bool(certified),
        "terminal_calls": int(terminal_calls),
        "endpoint_tests": int(
            cert.get(
                "endpoint_tests",
                cert.get(
                    "endpoint_probes",
                    0,
                ),
            )
        ),
        "passes_retest": int(
            cert.get(
                "passes",
                0,
            )
        ),
        "proof_reuse_skips": int(
            cert.get(
                "proof_reuse_skips",
                0,
            )
        ),
        "bound_skipped": int(
            cert.get(
                "bound_skipped",
                0,
            )
        ),
        "bad_status": cert.get(
            "bad_status"
        ),
        "bootstrap_wall": float(bootstrap_wall),
        "portfolio_wall": float(portfolio_wall),
        "cert_wall": float(cert_wall),
        "pipeline_wall": float(pipeline_wall),
        "r4_cap": int(r4_cap),
        "complete": bool(complete),
    }

    print(
        RESULT_MARKER
        + json.dumps(
            result,
            ensure_ascii=False,
            default=repr,
        )
    )

    return 0


def _extract_result(stdout):
    for line in reversed(
        stdout.splitlines()
    ):
        if line.startswith(
            RESULT_MARKER
        ):
            return json.loads(
                line[len(RESULT_MARKER):]
            )

    raise RuntimeError(
        "worker result marker not found"
    )


def _run_worker(
    *,
    local,
    remaining,
    case_index,
    arm,
):
    script = Path(__file__).resolve()

    cmd = [
        sys.executable,
        str(script),
        "--worker",
        "--worker-arm",
        str(arm),
        "--worker-case",
        str(case_index),
        "--source-v142",
        str(local.source_v142),
        "--case-time-limit",
        str(local.case_time_limit),
        "--initial-r4-cap",
        str(local.initial_r4_cap),
        "--prefix-node-cap",
        str(local.prefix_node_cap),
        "--k2-probe-cap",
        str(local.k2_probe_cap),
        "--terminal-call-cap",
        str(local.terminal_call_cap),
        *remaining,
    ]

    env = dict(
        os.environ
    )

    env[
        "PYTHONUNBUFFERED"
    ] = "1"

    t0 = time.perf_counter()

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    process_wall = (
        time.perf_counter()
        - t0
    )

    if proc.returncode != 0:
        tail = "\n".join(
            proc.stdout.splitlines()[-30:]
        )

        raise RuntimeError(
            f"worker failed arm={arm} case={case_index} "
            f"rc={proc.returncode}\n"
            f"STDOUT TAIL:\n{tail}\n"
            f"STDERR:\n{proc.stderr}"
        )

    result = _extract_result(
        proc.stdout
    )

    result[
        "external_process_wall"
    ] = float(
        process_wall
    )

    return result


def _median(rows, key):
    vals = [
        float(
            row[key]
        )
        for row in rows
        if row.get(key) is not None
    ]

    if not vals:
        return None

    return float(
        statistics.median(
            vals
        )
    )


def _controller_main(local, remaining):
    source_path = Path(
        local.source_v142
    )

    source = json.loads(
        source_path.read_text(
            encoding="utf-8",
        )
    )

    informative = [
        row
        for row in source.get(
            "cases",
            [],
        )
        if (
            bool(
                row.get(
                    "informative",
                    False,
                )
            )
            and bool(
                row.get(
                    "complete",
                    False,
                )
            )
        )
    ]

    if not informative:
        raise RuntimeError(
            "source v37.142 has no informative complete cases"
        )

    print(
        "# CubeLab v37.144 - "
        "ISOLATED ZERO vs INCUMBENT-RETEST EXACT CERTIFICATION A/B"
    )
    print(
        "source v37.142       :",
        source_path,
    )
    print(
        "informative cases    :",
        len(informative),
    )
    print(
        "repeats              :",
        int(local.repeats),
    )
    print(
        "isolation            :",
        "fresh Python process per arm run",
    )
    print(
        "candidate            :",
        "single-bound incumbent RETEST",
    )
    print(
        "production changes   :",
        "NONE",
    )

    print()
    print(
        "# ISOLATED RUNS"
    )

    case_rows = []

    for position, source_row in enumerate(
        informative,
        start=1,
    ):
        case_index = int(
            source_row["case"]
        )

        arm_records = {
            "ZERO": [],
            "RETEST": [],
        }

        print()
        print(
            f"[CASE {case_index}] "
            f"raw={source_row['raw_endpoints']} "
            f"K2={_expected_k2(source_row)}"
        )

        for repeat in range(
            int(local.repeats)
        ):
            zero_first = bool(
                (
                    int(position)
                    + int(repeat)
                )
                % 2
                == 0
            )

            order = (
                ["ZERO", "RETEST"]
                if zero_first
                else ["RETEST", "ZERO"]
            )

            print(
                f"  repeat {repeat+1}: "
                + " -> ".join(order)
            )

            for arm in order:
                result = _run_worker(
                    local=local,
                    remaining=remaining,
                    case_index=int(case_index),
                    arm=arm,
                )

                arm_records[arm].append(
                    result
                )

                print(
                    f"    {arm:<6} "
                    f"calls={result['terminal_calls']:<5} "
                    f"tests={result['endpoint_tests']:<4} "
                    f"passes={result['passes_retest']:<2} "
                    f"portfolio={result['portfolio_wall']:.4f}s "
                    f"cert={result['cert_wall']:.4f}s "
                    f"pipeline={result['pipeline_wall']:.4f}s "
                    f"complete={result['complete']}"
                )

        zero = arm_records[
            "ZERO"
        ]

        retest = arm_records[
            "RETEST"
        ]

        all_complete = bool(
            all(
                row.get(
                    "complete",
                    False,
                )
                for row in (
                    zero
                    + retest
                )
            )
        )

        zero_calls_set = {
            int(row["terminal_calls"])
            for row in zero
        }

        retest_calls_set = {
            int(row["terminal_calls"])
            for row in retest
        }

        call_stable = bool(
            len(zero_calls_set) == 1
            and len(retest_calls_set) == 1
        )

        zero_k2_set = {
            row.get("exact_k2")
            for row in zero
        }

        retest_k2_set = {
            row.get("exact_k2")
            for row in retest
        }

        exact_stable = bool(
            len(zero_k2_set) == 1
            and len(retest_k2_set) == 1
            and zero_k2_set == retest_k2_set
        )

        zero_calls = next(
            iter(zero_calls_set)
        )

        retest_calls = next(
            iter(retest_calls_set)
        )

        zero_pipeline = _median(
            zero,
            "pipeline_wall",
        )

        retest_pipeline = _median(
            retest,
            "pipeline_wall",
        )

        zero_cert = _median(
            zero,
            "cert_wall",
        )

        retest_cert = _median(
            retest,
            "cert_wall",
        )

        call_ratio = (
            float(retest_calls)
            / float(zero_calls)
            if zero_calls > 0
            else None
        )

        pipeline_ratio = (
            float(retest_pipeline)
            / float(zero_pipeline)
            if (
                zero_pipeline is not None
                and zero_pipeline > 0
            )
            else None
        )

        cert_ratio = (
            float(retest_cert)
            / float(zero_cert)
            if (
                zero_cert is not None
                and zero_cert > 0
            )
            else None
        )

        row = {
            "case": int(case_index),
            "source_raw_endpoints": int(
                source_row["raw_endpoints"]
            ),
            "source_exact_k2": _expected_k2(
                source_row
            ),
            "runs": arm_records,
            "contracts": {
                "all_complete": bool(all_complete),
                "call_stable": bool(call_stable),
                "exact_stable": bool(exact_stable),
            },
            "median": {
                "zero_calls": int(zero_calls),
                "retest_calls": int(retest_calls),
                "call_ratio": call_ratio,
                "zero_cert_wall": zero_cert,
                "retest_cert_wall": retest_cert,
                "cert_ratio": cert_ratio,
                "zero_pipeline_wall": zero_pipeline,
                "retest_pipeline_wall": retest_pipeline,
                "pipeline_ratio": pipeline_ratio,
            },
        }

        case_rows.append(row)

        print(
            "  MEDIAN calls       :",
            f"ZERO={zero_calls} RETEST={retest_calls} "
            f"ratio={call_ratio:.3f}x",
        )
        print(
            "  MEDIAN cert        :",
            f"ZERO={zero_cert:.4f}s "
            f"RETEST={retest_cert:.4f}s "
            f"ratio={cert_ratio:.3f}x",
        )
        print(
            "  MEDIAN pipeline    :",
            f"ZERO={zero_pipeline:.4f}s "
            f"RETEST={retest_pipeline:.4f}s "
            f"ratio={pipeline_ratio:.3f}x",
        )
        print(
            "  contracts          :",
            row["contracts"],
        )

    valid = [
        row
        for row in case_rows
        if all(
            row["contracts"].values()
        )
    ]

    total_zero_calls = sum(
        int(row["median"]["zero_calls"])
        for row in valid
    )

    total_retest_calls = sum(
        int(row["median"]["retest_calls"])
        for row in valid
    )

    total_call_ratio = (
        None
        if total_zero_calls <= 0
        else (
            float(total_retest_calls)
            / float(total_zero_calls)
        )
    )

    total_zero_cert = sum(
        float(row["median"]["zero_cert_wall"])
        for row in valid
    )

    total_retest_cert = sum(
        float(row["median"]["retest_cert_wall"])
        for row in valid
    )

    total_cert_ratio = (
        None
        if total_zero_cert <= 0
        else (
            float(total_retest_cert)
            / float(total_zero_cert)
        )
    )

    total_zero_pipeline = sum(
        float(row["median"]["zero_pipeline_wall"])
        for row in valid
    )

    total_retest_pipeline = sum(
        float(row["median"]["retest_pipeline_wall"])
        for row in valid
    )

    total_pipeline_ratio = (
        None
        if total_zero_pipeline <= 0
        else (
            float(total_retest_pipeline)
            / float(total_zero_pipeline)
        )
    )

    pipeline_win_cases = [
        row
        for row in valid
        if (
            row["median"]["pipeline_ratio"] is not None
            and float(
                row["median"]["pipeline_ratio"]
            )
            < 1.0
        )
    ]

    required_wins = max(
        1,
        (3 * len(valid) + 3) // 4,
    )

    print()
    print(
        "# AGGREGATE"
    )
    print(
        "valid cases          :",
        f"{len(valid)}/{len(case_rows)}",
    )
    print(
        "ZERO calls           :",
        total_zero_calls,
    )
    print(
        "RETEST calls         :",
        total_retest_calls,
    )
    print(
        "RETEST / ZERO calls  :",
        (
            "n/a"
            if total_call_ratio is None
            else f"{total_call_ratio:.3f}x"
        ),
    )
    print(
        "cert ZERO / RETEST   :",
        f"{total_zero_cert:.4f}s / "
        f"{total_retest_cert:.4f}s",
    )
    print(
        "RETEST / ZERO cert   :",
        (
            "n/a"
            if total_cert_ratio is None
            else f"{total_cert_ratio:.3f}x"
        ),
    )
    print(
        "pipeline ZERO/RETEST :",
        f"{total_zero_pipeline:.4f}s / "
        f"{total_retest_pipeline:.4f}s",
    )
    print(
        "RETEST / ZERO pipeline:",
        (
            "n/a"
            if total_pipeline_ratio is None
            else f"{total_pipeline_ratio:.3f}x"
        ),
    )
    print(
        "pipeline win cases   :",
        f"{len(pipeline_win_cases)}/{len(valid)} "
        f"(required {required_wins})",
    )

    all_contract = bool(
        len(valid) == len(case_rows)
    )

    if not all_contract:
        decision = (
            "INCUMBENT_RETEST_ISOLATED_CONTRACT_FAIL"
        )
        note = (
            "At least one fresh-process RETEST run failed exact/raw/call "
            "stability. Fix the contract before interpreting performance."
        )

    elif (
        total_call_ratio is not None
        and total_call_ratio <= 0.30
        and total_pipeline_ratio is not None
        and total_pipeline_ratio <= 0.90
        and len(pipeline_win_cases) >= int(required_wins)
    ):
        decision = (
            "INCUMBENT_RETEST_ISOLATED_END_TO_END_WALL_WIN"
        )
        note = (
            "Single-bound incumbent retesting exactly reproduces the global K2 "
            "reference while materially reducing terminal calls and isolated "
            "hard-pipeline wall. This is a stronger runtime candidate than "
            "incremental LBSTART."
        )

    elif (
        total_call_ratio is not None
        and total_call_ratio <= 0.30
        and total_cert_ratio is not None
        and total_cert_ratio <= 0.80
    ):
        decision = (
            "INCUMBENT_RETEST_CERT_WALL_WIN_PIPELINE_LIMITED"
        )
        note = (
            "RETEST materially improves isolated exact-certification wall, but "
            "H-portfolio generation prevents a >=10% aggregate pipeline win."
        )

    elif (
        total_call_ratio is not None
        and total_call_ratio <= 0.30
    ):
        decision = (
            "INCUMBENT_RETEST_CALL_WIN_NO_WALL_WIN"
        )
        note = (
            "RETEST preserves exactness and cuts calls but does not translate "
            "that reduction into isolated wall time. Do not integrate it."
        )

    else:
        decision = (
            "INCUMBENT_RETEST_NOT_USEFUL"
        )
        note = (
            "High-budget feasibility retesting does not provide the required "
            "exact/runtime benefit over the current reference."
        )

    print()
    print(
        "# DECISION"
    )
    print(decision)
    print(note)

    print()
    print(
        "NEXT:"
    )

    if decision == (
        "INCUMBENT_RETEST_ISOLATED_END_TO_END_WALL_WIN"
    ):
        print(
            "  v37.145 hard/global integration candidate using incumbent RETEST"
        )
    elif decision == (
        "INCUMBENT_RETEST_CERT_WALL_WIN_PIPELINE_LIMITED"
    ):
        print(
            "  keep RETEST inside hard/global portfolio tooling only; do not patch stable hot path"
        )
    else:
        print(
            "  close terminal certification scheduling; move bottleneck work to H-portfolio generation"
        )

    payload = {
        "version": "v37.144",
        "mode": "ISOLATED_ZERO_VS_INCUMBENT_RETEST_EXACT_CERTIFICATION_AB",
        "source_v142": str(source_path),
        "config": {
            "repeats": int(local.repeats),
            "case_time_limit": float(local.case_time_limit),
            "initial_r4_cap": int(local.initial_r4_cap),
            "prefix_node_cap": int(local.prefix_node_cap),
            "k2_probe_cap": int(local.k2_probe_cap),
            "terminal_call_cap": int(local.terminal_call_cap),
        },
        "cases": case_rows,
        "aggregate": {
            "valid_cases": len(valid),
            "total_cases": len(case_rows),
            "zero_calls": int(total_zero_calls),
            "retest_calls": int(total_retest_calls),
            "retest_vs_zero_call_ratio": total_call_ratio,
            "zero_cert_wall": float(total_zero_cert),
            "retest_cert_wall": float(total_retest_cert),
            "retest_vs_zero_cert_ratio": total_cert_ratio,
            "zero_pipeline_wall": float(total_zero_pipeline),
            "retest_pipeline_wall": float(total_retest_pipeline),
            "retest_vs_zero_pipeline_ratio": total_pipeline_ratio,
            "pipeline_win_cases": len(pipeline_win_cases),
            "required_pipeline_win_cases": int(required_wins),
        },
        "decision": decision,
        "note": note,
        "production_changes": "NONE",
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


def main():
    local, remaining = _parse_args(
        sys.argv[1:]
    )

    if local.worker:
        return _worker_main(
            local,
            remaining,
        )

    return _controller_main(
        local,
        remaining,
    )


if __name__ == "__main__":
    raise SystemExit(main())
