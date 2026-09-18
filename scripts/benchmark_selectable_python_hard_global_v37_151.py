#!/usr/bin/env python3
"""
CubeLab v37.151 — SELECTABLE PYTHON HARD/GLOBAL MODES + MIXED FRESH BENCHMARK

Frozen candidate after v37.150
------------------------------
Reference preset:
    prefix_mode=current
    r4_cap_mode=auto192
    cert_mode=zero

Candidate preset:
    prefix_mode=fused
    r4_cap_mode=fullcap
    cert_mode=retest

v37.150 fresh isolated result:
    exact contracts     : 5/5
    calls               : 0.092x
    H-portfolio wall    : 0.783x
    hard pipeline wall  : 0.829x
    case wins           : 5/5

v37.151 freezes those semantics behind explicit selectable modes and validates
them on a NEW deterministic mixed hard batch.

Selectable axes
---------------
prefix_mode:
    current
    fused

r4_cap_mode:
    auto192
    fullcap

cert_mode:
    zero
    retest

All combinations are supported in worker mode.

Mixed benchmark
---------------
Default:
    6 deterministic reduced random 25-move scrambles
    seed=20261052
    H=root phase1 LB + 3
    repeats=2
    each arm run in a fresh Python process

The natural endpoint-count spread is reported as:
    small  : 1..32 raw endpoints
    medium : 33..128
    large  : 129+

Contracts
---------
For every case:
    same scramble
    same root LB / H
    complete non-truncated H portfolio
    same prefix/R4 counts
    same endpoint fingerprint
    same skeleton fingerprint
    same exact global K2
    stable terminal-call count per arm

Primary metric:
    hard pipeline wall = H portfolio + exact certification

PASS:
    all case contracts
    >=4 informative cases
    candidate/reference calls <=0.25x
    candidate/reference pipeline <=0.90x
    candidate faster in >=75% informative cases

Production changes: NONE.
Stable v37.24.1 / v37.114 remain untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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

import audit_fused_q_lb_prefix_v37_149 as v149
import audit_isolated_retest_cert_v37_144 as v144
import audit_h_allocation_residual_v37_140 as v140
import audit_fresh_hard_residual_gate_v37_133 as v133
import audit_hard_global_k2_residual_v37_131_1 as v1311
import audit_post_dr_k2_deficit_v37_128 as v128
import audit_fast_stable_selective_terminal_macro_v37_114 as v114


RESULT_MARKER = "__V37151_RESULT__="

CONTROL_TOKENS = tuple(
    "D' R F2 U D' R L B D R' U D R' D F' B U F' L' D2 B L D' F R'".split()
)


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
    )

    ap.add_argument("--fresh-count", type=int, default=6)
    ap.add_argument("--fresh-seed", type=int, default=20261052)
    ap.add_argument("--scramble-length", type=int, default=25)
    ap.add_argument("--h-offset", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=2)

    ap.add_argument("--case-time-limit", type=float, default=180.0)
    ap.add_argument("--initial-r4-cap", type=int, default=192)
    ap.add_argument("--prefix-node-cap", type=int, default=200000)
    ap.add_argument("--k2-probe-cap", type=int, default=22)
    ap.add_argument("--terminal-call-cap", type=int, default=16000)

    ap.add_argument(
        "--reference-prefix-mode",
        choices=("current", "fused"),
        default="current",
    )
    ap.add_argument(
        "--reference-r4-cap-mode",
        choices=("auto192", "fullcap"),
        default="auto192",
    )
    ap.add_argument(
        "--reference-cert-mode",
        choices=("zero", "retest"),
        default="zero",
    )

    ap.add_argument(
        "--candidate-prefix-mode",
        choices=("current", "fused"),
        default="fused",
    )
    ap.add_argument(
        "--candidate-r4-cap-mode",
        choices=("auto192", "fullcap"),
        default="fullcap",
    )
    ap.add_argument(
        "--candidate-cert-mode",
        choices=("zero", "retest"),
        default="retest",
    )

    ap.add_argument(
        "--analysis-output",
        default="reports/v37/selectable_python_hard_global_v37_151.json",
    )

    # Internal worker args.
    ap.add_argument("--worker", action="store_true")
    ap.add_argument(
        "--worker-side",
        choices=("REFERENCE", "CANDIDATE"),
        default=None,
    )
    ap.add_argument("--worker-case", type=int, default=None)

    ns, remaining = ap.parse_known_args(argv)

    ns.fresh_count = max(1, int(ns.fresh_count))
    ns.scramble_length = max(1, int(ns.scramble_length))
    ns.h_offset = max(0, int(ns.h_offset))
    ns.repeats = max(1, int(ns.repeats))
    ns.case_time_limit = max(1.0, float(ns.case_time_limit))
    ns.initial_r4_cap = max(1, int(ns.initial_r4_cap))
    ns.prefix_node_cap = max(1, int(ns.prefix_node_cap))
    ns.k2_probe_cap = max(0, int(ns.k2_probe_cap))
    ns.terminal_call_cap = max(1, int(ns.terminal_call_cap))

    return ns, remaining


def _generate_scrambles(
    *,
    fresh_count,
    fresh_seed,
    scramble_length,
):
    rng = random.Random(int(fresh_seed))
    out = []

    while len(out) < int(fresh_count):
        candidate = v133._generate_reduced_scramble(
            rng=rng,
            length=int(scramble_length),
        )

        if candidate == CONTROL_TOKENS:
            continue

        if candidate in out:
            continue

        out.append(candidate)

    return out


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


def _endpoint_fingerprint(portfolio):
    rows = sorted(
        repr(key)
        for key in portfolio["endpoints"].keys()
    )

    h = hashlib.sha256()

    for row in rows:
        h.update(row.encode("utf-8"))
        h.update(b"\n")

    return h.hexdigest()


def _skeleton_fingerprint(portfolio):
    rows = []

    for skeleton in portfolio["skeletons"]:
        rows.append(
            (
                repr(skeleton["endpoint_key"]),
                tuple(skeleton["prefix_moves"]),
                tuple(skeleton["prefix_ranks"]),
                int(skeleton["r4_rank"]),
                tuple(skeleton["r4_suffix"]),
            )
        )

    rows.sort(key=repr)

    h = hashlib.sha256()

    for row in rows:
        h.update(repr(row).encode("utf-8"))
        h.update(b"\n")

    return h.hexdigest()


def _enumerate_once(
    *,
    prefix_mode,
    solver,
    oracle,
    poses,
    q,
    h_length,
    r4_cap,
    prefix_node_cap,
    time_limit,
):
    if prefix_mode == "current":
        return v133._enumerate_direct_h_complete(
            solver=solver,
            oracle=oracle,
            poses=tuple(poses),
            q=int(q),
            h_length=int(h_length),
            r4_suffix_cap=int(r4_cap),
            prefix_node_cap=int(prefix_node_cap),
            time_limit=float(time_limit),
        )

    return v149._enumerate_fused(
        solver=solver,
        oracle=oracle,
        poses=tuple(poses),
        q=int(q),
        h_length=int(h_length),
        r4_suffix_cap=int(r4_cap),
        prefix_node_cap=int(prefix_node_cap),
        time_limit=float(time_limit),
    )


def _enumerate_mode(
    *,
    prefix_mode,
    r4_cap_mode,
    solver,
    oracle,
    poses,
    q,
    h_length,
    initial_r4_cap,
    prefix_node_cap,
    time_limit,
):
    start = time.perf_counter()
    deadline = start + float(time_limit)

    if r4_cap_mode == "fullcap":
        cap = int(
            len(oracle.words[4])
        )

        portfolio = _enumerate_once(
            prefix_mode=prefix_mode,
            solver=solver,
            oracle=oracle,
            poses=tuple(poses),
            q=int(q),
            h_length=int(h_length),
            r4_cap=int(cap),
            prefix_node_cap=int(prefix_node_cap),
            time_limit=float(
                max(
                    0.0,
                    deadline
                    - time.perf_counter(),
                )
            ),
        )

        passes = [
            {
                "r4_cap": int(cap),
                "prefix_nodes": int(
                    portfolio["stats"]["prefix_nodes"]
                ),
                "r4_leaf_states": int(
                    portfolio["stats"]["r4_leaf_states"]
                ),
                "max_r4_words": int(
                    portfolio["stats"]["max_r4_words"]
                ),
                "r4_cap_truncated": int(
                    portfolio["stats"]["r4_cap_truncated"]
                ),
                "node_cap_cut": bool(
                    portfolio["cuts"]["node_cap"]
                ),
                "time_cut": bool(
                    portfolio["cuts"]["time"]
                ),
                "elapsed": float(
                    portfolio["elapsed"]
                ),
            }
        ]

        return portfolio, int(cap), passes

    cap = int(initial_r4_cap)
    passes = []

    while True:
        remaining = (
            deadline
            - time.perf_counter()
        )

        if remaining <= 0:
            return None, int(cap), passes

        portfolio = _enumerate_once(
            prefix_mode=prefix_mode,
            solver=solver,
            oracle=oracle,
            poses=tuple(poses),
            q=int(q),
            h_length=int(h_length),
            r4_cap=int(cap),
            prefix_node_cap=int(prefix_node_cap),
            time_limit=float(remaining),
        )

        passes.append(
            {
                "r4_cap": int(cap),
                "prefix_nodes": int(
                    portfolio["stats"]["prefix_nodes"]
                ),
                "r4_leaf_states": int(
                    portfolio["stats"]["r4_leaf_states"]
                ),
                "max_r4_words": int(
                    portfolio["stats"]["max_r4_words"]
                ),
                "r4_cap_truncated": int(
                    portfolio["stats"]["r4_cap_truncated"]
                ),
                "node_cap_cut": bool(
                    portfolio["cuts"]["node_cap"]
                ),
                "time_cut": bool(
                    portfolio["cuts"]["time"]
                ),
                "elapsed": float(
                    portfolio["elapsed"]
                ),
            }
        )

        if (
            portfolio["cuts"]["node_cap"]
            or portfolio["cuts"]["time"]
        ):
            return portfolio, int(cap), passes

        truncated = int(
            portfolio["stats"]["r4_cap_truncated"]
        )

        max_words = int(
            portfolio["stats"]["max_r4_words"]
        )

        if (
            truncated > 0
            and max_words > int(cap)
        ):
            cap = int(max_words)
            continue

        return portfolio, int(cap), passes


def _certify_mode(
    *,
    cert_mode,
    solver,
    poses,
    portfolio,
    k2_probe_cap,
    terminal_call_cap,
    deadline,
):
    raw_endpoints = len(
        portfolio["endpoints"]
    )

    if raw_endpoints == 0:
        return {
            "certified": True,
            "exact_best_k2": None,
            "terminal_calls": 0,
            "endpoint_tests": 0,
            "bad_status": None,
            "bound_skipped": 0,
        }

    if cert_mode == "zero":
        zero = v140._certify_fixed_h(
            solver=solver,
            scrambled_poses=tuple(poses),
            portfolio=portfolio,
            k2_probe_cap=int(k2_probe_cap),
            exact_call_cap=int(
                terminal_call_cap
            ),
            deadline=deadline,
        )

        return {
            "certified": bool(
                zero.get(
                    "certified",
                    False,
                )
            ),
            "exact_best_k2": zero.get(
                "exact_best_k2"
            ),
            "terminal_calls": int(
                zero.get(
                    "exact_budget_calls",
                    0,
                )
            ),
            "endpoint_tests": int(
                zero.get(
                    "endpoint_probes",
                    0,
                )
            ),
            "bad_status": zero.get(
                "bad_status"
            ),
            "bound_skipped": int(
                zero.get(
                    "bound_skipped",
                    0,
                )
            ),
        }

    retest = v144._run_retest(
        solver=solver,
        scrambled_poses=tuple(poses),
        portfolio=portfolio,
        probe_cap=int(k2_probe_cap),
        call_cap=int(
            terminal_call_cap
        ),
        deadline=deadline,
    )

    return {
        "certified": bool(
            retest.get(
                "certified",
                False,
            )
        ),
        "exact_best_k2": retest.get(
            "exact_best_k2"
        ),
        "terminal_calls": int(
            retest.get(
                "terminal_calls",
                0,
            )
        ),
        "endpoint_tests": int(
            retest.get(
                "endpoint_tests",
                0,
            )
        ),
        "retest_passes": int(
            retest.get(
                "passes",
                0,
            )
        ),
        "proof_reuse_skips": int(
            retest.get(
                "proof_reuse_skips",
                0,
            )
        ),
        "bad_status": retest.get(
            "bad_status"
        ),
        "bound_skipped": int(
            retest.get(
                "bound_skipped",
                0,
            )
        ),
    }


def _modes_for_side(local, side):
    if side == "REFERENCE":
        return {
            "prefix_mode": str(
                local.reference_prefix_mode
            ),
            "r4_cap_mode": str(
                local.reference_r4_cap_mode
            ),
            "cert_mode": str(
                local.reference_cert_mode
            ),
        }

    return {
        "prefix_mode": str(
            local.candidate_prefix_mode
        ),
        "r4_cap_mode": str(
            local.candidate_r4_cap_mode
        ),
        "cert_mode": str(
            local.candidate_cert_mode
        ),
    }


def _worker_main(local, remaining):
    if (
        local.worker_side is None
        or local.worker_case is None
    ):
        raise SystemExit(
            "--worker requires --worker-side and --worker-case"
        )

    scrambles = _generate_scrambles(
        fresh_count=int(local.fresh_count),
        fresh_seed=int(local.fresh_seed),
        scramble_length=int(local.scramble_length),
    )

    idx = int(local.worker_case) - 1

    if idx < 0 or idx >= len(scrambles):
        raise SystemExit(
            f"worker case out of range: {local.worker_case}"
        )

    moves = tuple(
        scrambles[idx]
    )

    modes = _modes_for_side(
        local,
        str(local.worker_side),
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

    h_length = int(
        root_lb
        + int(local.h_offset)
    )

    hard_t0 = time.perf_counter()
    portfolio_t0 = hard_t0

    (
        portfolio,
        final_cap,
        passes,
    ) = _enumerate_mode(
        prefix_mode=modes[
            "prefix_mode"
        ],
        r4_cap_mode=modes[
            "r4_cap_mode"
        ],
        solver=solver,
        oracle=oracle,
        poses=tuple(poses),
        q=int(q),
        h_length=int(h_length),
        initial_r4_cap=int(
            local.initial_r4_cap
        ),
        prefix_node_cap=int(
            local.prefix_node_cap
        ),
        time_limit=float(
            local.case_time_limit
        ),
    )

    portfolio_wall = (
        time.perf_counter()
        - portfolio_t0
    )

    if not _portfolio_complete(
        portfolio
    ):
        result = {
            "side": str(
                local.worker_side
            ),
            "modes": modes,
            "case": int(
                local.worker_case
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
                "portfolio_incomplete"
            ),
            "stable_return_code": int(
                rc
            ),
            "bootstrap_wall": float(
                bootstrap_wall
            ),
            "portfolio_wall": float(
                portfolio_wall
            ),
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

    stats = portfolio["stats"]

    raw_endpoints = len(
        portfolio["endpoints"]
    )

    skeletons = len(
        portfolio["skeletons"]
    )

    endpoint_fp = (
        _endpoint_fingerprint(
            portfolio
        )
    )

    skeleton_fp = (
        _skeleton_fingerprint(
            portfolio
        )
    )

    deadline = (
        hard_t0
        + float(
            local.case_time_limit
        )
    )

    cert_t0 = time.perf_counter()

    cert = _certify_mode(
        cert_mode=modes[
            "cert_mode"
        ],
        solver=solver,
        poses=tuple(
            poses
        ),
        portfolio=portfolio,
        k2_probe_cap=int(
            local.k2_probe_cap
        ),
        terminal_call_cap=int(
            local.terminal_call_cap
        ),
        deadline=deadline,
    )

    cert_wall = (
        time.perf_counter()
        - cert_t0
    )

    pipeline_wall = (
        time.perf_counter()
        - hard_t0
    )

    result = {
        "side": str(
            local.worker_side
        ),
        "modes": modes,
        "case": int(
            local.worker_case
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
        "prefix_nodes": int(
            stats[
                "prefix_nodes"
            ]
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
        "skeletons": int(
            skeletons
        ),
        "endpoint_fingerprint": str(
            endpoint_fp
        ),
        "skeleton_fingerprint": str(
            skeleton_fp
        ),
        "r4_cap_truncated": int(
            stats[
                "r4_cap_truncated"
            ]
        ),
        "max_r4_words": int(
            stats[
                "max_r4_words"
            ]
        ),
        "final_r4_cap": int(
            final_cap
        ),
        "pass_count": int(
            len(
                passes
            )
        ),
        "passes": passes,
        "informative": bool(
            raw_endpoints > 0
        ),
        "exact_k2": (
            None
            if cert.get(
                "exact_best_k2"
            )
            is None
            else int(
                cert[
                    "exact_best_k2"
                ]
            )
        ),
        "terminal_calls": int(
            cert.get(
                "terminal_calls",
                0,
            )
        ),
        "endpoint_tests": int(
            cert.get(
                "endpoint_tests",
                0,
            )
        ),
        "retest_passes": int(
            cert.get(
                "retest_passes",
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
        "certified": bool(
            cert.get(
                "certified",
                False,
            )
        ),
        "bad_status": cert.get(
            "bad_status"
        ),
        "stable_return_code": int(
            rc
        ),
        "bootstrap_wall": float(
            bootstrap_wall
        ),
        "portfolio_wall": float(
            portfolio_wall
        ),
        "cert_wall": float(
            cert_wall
        ),
        "pipeline_wall": float(
            pipeline_wall
        ),
        "complete": bool(
            cert.get(
                "certified",
                False,
            )
        ),
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
                line[
                    len(
                        RESULT_MARKER
                    ):
                ]
            )

    raise RuntimeError(
        "worker result marker not found"
    )


def _run_worker(
    *,
    local,
    remaining,
    case_index,
    side,
):
    script = Path(
        __file__
    ).resolve()

    cmd = [
        sys.executable,
        str(script),
        "--worker",
        "--worker-side",
        str(side),
        "--worker-case",
        str(case_index),

        "--fresh-count",
        str(local.fresh_count),
        "--fresh-seed",
        str(local.fresh_seed),
        "--scramble-length",
        str(local.scramble_length),
        "--h-offset",
        str(local.h_offset),

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

        "--reference-prefix-mode",
        str(local.reference_prefix_mode),
        "--reference-r4-cap-mode",
        str(local.reference_r4_cap_mode),
        "--reference-cert-mode",
        str(local.reference_cert_mode),

        "--candidate-prefix-mode",
        str(local.candidate_prefix_mode),
        "--candidate-r4-cap-mode",
        str(local.candidate_r4_cap_mode),
        "--candidate-cert-mode",
        str(local.candidate_cert_mode),

        *remaining,
    ]

    env = dict(
        os.environ
    )

    env[
        "PYTHONUNBUFFERED"
    ] = "1"

    t0 = (
        time.perf_counter()
    )

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
            proc.stdout.splitlines()[
                -40:
            ]
        )

        raise RuntimeError(
            f"worker failed side={side} "
            f"case={case_index} "
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


def _median(
    rows,
    key,
):
    vals = [
        float(
            row[
                key
            ]
        )
        for row in rows
        if row.get(
            key
        )
        is not None
    ]

    if not vals:
        return None

    return float(
        statistics.median(
            vals
        )
    )


def _stable_set(
    rows,
    key,
):
    return {
        json.dumps(
            row.get(
                key
            ),
            sort_keys=True,
            default=repr,
        )
        for row in rows
    }


def _bucket(
    n,
):
    n = int(n)

    if n <= 0:
        return "none"

    if n <= 32:
        return "small"

    if n <= 128:
        return "medium"

    return "large"


def _controller_main(
    local,
    remaining,
):
    scrambles = _generate_scrambles(
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

    ref_modes = _modes_for_side(
        local,
        "REFERENCE",
    )

    cand_modes = _modes_for_side(
        local,
        "CANDIDATE",
    )

    print(
        "# CubeLab v37.151 - "
        "SELECTABLE PYTHON HARD/GLOBAL MODES + MIXED FRESH BENCHMARK"
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
        "H policy             :",
        f"root LB + {local.h_offset}",
    )

    print(
        "repeats / side / case:",
        int(
            local.repeats
        ),
    )

    print(
        "REFERENCE modes      :",
        ref_modes,
    )

    print(
        "CANDIDATE modes      :",
        cand_modes,
    )

    print(
        "isolation            :",
        "fresh Python process per side run",
    )

    print(
        "stable changes       :",
        "NONE",
    )

    print()
    print(
        "# MIXED FRESH CASES"
    )

    case_rows = []

    for case_index, moves in enumerate(
        scrambles,
        start=1,
    ):
        print()
        print(
            f"[CASE {case_index}/"
            f"{local.fresh_count}]"
        )

        print(
            "scramble             :",
            " ".join(
                moves
            ),
        )

        arms = {
            "REFERENCE": [],
            "CANDIDATE": [],
        }

        for repeat in range(
            int(
                local.repeats
            )
        ):
            reference_first = bool(
                (
                    int(
                        case_index
                    )
                    + int(
                        repeat
                    )
                )
                % 2
                == 0
            )

            order = (
                [
                    "REFERENCE",
                    "CANDIDATE",
                ]
                if reference_first
                else [
                    "CANDIDATE",
                    "REFERENCE",
                ]
            )

            print(
                f"  repeat {repeat+1}: "
                + " -> ".join(
                    order
                )
            )

            for side in order:
                result = _run_worker(
                    local=local,
                    remaining=remaining,
                    case_index=int(
                        case_index
                    ),
                    side=side,
                )

                arms[
                    side
                ].append(
                    result
                )

                print(
                    f"    {side:<9} "
                    f"passes={result['pass_count']} "
                    f"prefix={result['prefix_nodes']:<7} "
                    f"raw={result['raw_endpoints']:<4} "
                    f"K2={str(result['exact_k2']):<4} "
                    f"calls={result['terminal_calls']:<5} "
                    f"portfolio={result['portfolio_wall']:.4f}s "
                    f"cert={result['cert_wall']:.4f}s "
                    f"pipeline={result['pipeline_wall']:.4f}s "
                    f"complete={result['complete']}"
                )

        ref = arms[
            "REFERENCE"
        ]

        cand = arms[
            "CANDIDATE"
        ]

        contracts = {
            "all_complete": bool(
                all(
                    row.get(
                        "complete",
                        False,
                    )
                    for row in (
                        ref
                        + cand
                    )
                )
            ),

            "scramble_identity": bool(
                _stable_set(
                    ref,
                    "scramble",
                )
                == _stable_set(
                    cand,
                    "scramble",
                )
                and len(
                    _stable_set(
                        ref,
                        "scramble",
                    )
                )
                == 1
            ),

            "root_h_identity": bool(
                _stable_set(
                    ref,
                    "root_lb",
                )
                == _stable_set(
                    cand,
                    "root_lb",
                )
                and _stable_set(
                    ref,
                    "h_length",
                )
                == _stable_set(
                    cand,
                    "h_length",
                )
            ),

            "prefix_r4_count_identity": bool(
                _stable_set(
                    ref,
                    "prefix_nodes",
                )
                == _stable_set(
                    cand,
                    "prefix_nodes",
                )
                and _stable_set(
                    ref,
                    "r4_leaf_states",
                )
                == _stable_set(
                    cand,
                    "r4_leaf_states",
                )
                and _stable_set(
                    ref,
                    "r4_words_considered",
                )
                == _stable_set(
                    cand,
                    "r4_words_considered",
                )
            ),

            "endpoint_identity": bool(
                _stable_set(
                    ref,
                    "endpoint_fingerprint",
                )
                == _stable_set(
                    cand,
                    "endpoint_fingerprint",
                )
            ),

            "skeleton_identity": bool(
                _stable_set(
                    ref,
                    "skeleton_fingerprint",
                )
                == _stable_set(
                    cand,
                    "skeleton_fingerprint",
                )
            ),

            "exact_identity": bool(
                _stable_set(
                    ref,
                    "exact_k2",
                )
                == _stable_set(
                    cand,
                    "exact_k2",
                )
            ),

            "reference_call_stable": bool(
                len(
                    _stable_set(
                        ref,
                        "terminal_calls",
                    )
                )
                == 1
            ),

            "candidate_call_stable": bool(
                len(
                    _stable_set(
                        cand,
                        "terminal_calls",
                    )
                )
                == 1
            ),

            "no_final_truncation": bool(
                all(
                    int(
                        row.get(
                            "r4_cap_truncated",
                            1,
                        )
                    )
                    == 0
                    for row in (
                        ref
                        + cand
                    )
                )
            ),
        }

        raw_endpoints = int(
            ref[0][
                "raw_endpoints"
            ]
        )

        informative = bool(
            raw_endpoints > 0
        )

        bucket = _bucket(
            raw_endpoints
        )

        ref_calls = int(
            ref[0][
                "terminal_calls"
            ]
        )

        cand_calls = int(
            cand[0][
                "terminal_calls"
            ]
        )

        ref_portfolio = _median(
            ref,
            "portfolio_wall",
        )

        cand_portfolio = _median(
            cand,
            "portfolio_wall",
        )

        ref_cert = _median(
            ref,
            "cert_wall",
        )

        cand_cert = _median(
            cand,
            "cert_wall",
        )

        ref_pipeline = _median(
            ref,
            "pipeline_wall",
        )

        cand_pipeline = _median(
            cand,
            "pipeline_wall",
        )

        call_ratio = (
            None
            if ref_calls <= 0
            else float(
                cand_calls
            )
            / float(
                ref_calls
            )
        )

        portfolio_ratio = (
            None
            if (
                ref_portfolio is None
                or ref_portfolio <= 0
            )
            else float(
                cand_portfolio
            )
            / float(
                ref_portfolio
            )
        )

        cert_ratio = (
            None
            if (
                ref_cert is None
                or ref_cert <= 0
            )
            else float(
                cand_cert
            )
            / float(
                ref_cert
            )
        )

        pipeline_ratio = (
            None
            if (
                ref_pipeline is None
                or ref_pipeline <= 0
            )
            else float(
                cand_pipeline
            )
            / float(
                ref_pipeline
            )
        )

        row = {
            "case": int(
                case_index
            ),
            "scramble": list(
                moves
            ),
            "raw_endpoints": int(
                raw_endpoints
            ),
            "size_bucket": str(
                bucket
            ),
            "informative": bool(
                informative
            ),
            "runs": arms,
            "contracts": contracts,
            "reference_pass_count": int(
                ref[0][
                    "pass_count"
                ]
            ),
            "candidate_pass_count": int(
                cand[0][
                    "pass_count"
                ]
            ),
            "median": {
                "reference_calls": int(
                    ref_calls
                ),
                "candidate_calls": int(
                    cand_calls
                ),
                "call_ratio": (
                    call_ratio
                ),

                "reference_portfolio_wall": (
                    ref_portfolio
                ),
                "candidate_portfolio_wall": (
                    cand_portfolio
                ),
                "portfolio_ratio": (
                    portfolio_ratio
                ),

                "reference_cert_wall": (
                    ref_cert
                ),
                "candidate_cert_wall": (
                    cand_cert
                ),
                "cert_ratio": (
                    cert_ratio
                ),

                "reference_pipeline_wall": (
                    ref_pipeline
                ),
                "candidate_pipeline_wall": (
                    cand_pipeline
                ),
                "pipeline_ratio": (
                    pipeline_ratio
                ),
            },
        }

        case_rows.append(
            row
        )

        print(
            "  bucket             :",
            bucket,
        )

        print(
            "  contracts          :",
            contracts,
        )

        if informative:
            print(
                "  MEDIAN calls       :",
                f"REF={ref_calls} "
                f"CAND={cand_calls} "
                f"ratio={call_ratio:.3f}x",
            )

            print(
                "  MEDIAN portfolio   :",
                f"REF={ref_portfolio:.4f}s "
                f"CAND={cand_portfolio:.4f}s "
                f"ratio={portfolio_ratio:.3f}x",
            )

            print(
                "  MEDIAN cert        :",
                f"REF={ref_cert:.4f}s "
                f"CAND={cand_cert:.4f}s "
                f"ratio={cert_ratio:.3f}x",
            )

            print(
                "  MEDIAN pipeline    :",
                f"REF={ref_pipeline:.4f}s "
                f"CAND={cand_pipeline:.4f}s "
                f"ratio={pipeline_ratio:.3f}x",
            )

        else:
            print(
                "  outcome            :",
                "NO_DR_ENDPOINTS / exact-equivalent",
            )

    valid = [
        row
        for row in case_rows
        if all(
            row[
                "contracts"
            ].values()
        )
    ]

    informative = [
        row
        for row in valid
        if row[
            "informative"
        ]
    ]

    no_dr = [
        row
        for row in valid
        if not row[
            "informative"
        ]
    ]

    bucket_hist = {
        name: sum(
            1
            for row in valid
            if row[
                "size_bucket"
            ]
            == name
        )
        for name in (
            "none",
            "small",
            "medium",
            "large",
        )
    }

    rerun_cases = [
        row
        for row in valid
        if int(
            row[
                "reference_pass_count"
            ]
        )
        > 1
    ]

    total_ref_calls = sum(
        int(
            row[
                "median"
            ][
                "reference_calls"
            ]
        )
        for row in informative
    )

    total_cand_calls = sum(
        int(
            row[
                "median"
            ][
                "candidate_calls"
            ]
        )
        for row in informative
    )

    total_call_ratio = (
        None
        if total_ref_calls <= 0
        else float(
            total_cand_calls
        )
        / float(
            total_ref_calls
        )
    )

    total_ref_portfolio = sum(
        float(
            row[
                "median"
            ][
                "reference_portfolio_wall"
            ]
        )
        for row in informative
    )

    total_cand_portfolio = sum(
        float(
            row[
                "median"
            ][
                "candidate_portfolio_wall"
            ]
        )
        for row in informative
    )

    total_portfolio_ratio = (
        None
        if total_ref_portfolio <= 0
        else float(
            total_cand_portfolio
        )
        / float(
            total_ref_portfolio
        )
    )

    total_ref_cert = sum(
        float(
            row[
                "median"
            ][
                "reference_cert_wall"
            ]
        )
        for row in informative
    )

    total_cand_cert = sum(
        float(
            row[
                "median"
            ][
                "candidate_cert_wall"
            ]
        )
        for row in informative
    )

    total_cert_ratio = (
        None
        if total_ref_cert <= 0
        else float(
            total_cand_cert
        )
        / float(
            total_ref_cert
        )
    )

    total_ref_pipeline = sum(
        float(
            row[
                "median"
            ][
                "reference_pipeline_wall"
            ]
        )
        for row in informative
    )

    total_cand_pipeline = sum(
        float(
            row[
                "median"
            ][
                "candidate_pipeline_wall"
            ]
        )
        for row in informative
    )

    total_pipeline_ratio = (
        None
        if total_ref_pipeline <= 0
        else float(
            total_cand_pipeline
        )
        / float(
            total_ref_pipeline
        )
    )

    pipeline_wins = [
        row
        for row in informative
        if (
            row[
                "median"
            ][
                "pipeline_ratio"
            ]
            is not None
            and float(
                row[
                    "median"
                ][
                    "pipeline_ratio"
                ]
            )
            < 1.0
        )
    ]

    portfolio_wins = [
        row
        for row in informative
        if (
            row[
                "median"
            ][
                "portfolio_ratio"
            ]
            is not None
            and float(
                row[
                    "median"
                ][
                    "portfolio_ratio"
                ]
            )
            < 1.0
        )
    ]

    required_wins = max(
        1,
        (
            3
            * len(
                informative
            )
            + 3
        )
        // 4,
    )

    print()
    print(
        "# AGGREGATE"
    )

    print(
        "contract-valid cases :",
        f"{len(valid)}/{len(case_rows)}",
    )

    print(
        "informative cases    :",
        len(
            informative
        ),
    )

    print(
        "no-DR cases          :",
        len(
            no_dr
        ),
    )

    print(
        "endpoint-size buckets:",
        bucket_hist,
    )

    print(
        "REFERENCE rerun cases:",
        [
            row[
                "case"
            ]
            for row in rerun_cases
        ],
    )

    print(
        "terminal calls REF/CAND:",
        f"{total_ref_calls} / "
        f"{total_cand_calls}",
    )

    print(
        "CAND / REF calls     :",
        (
            "n/a"
            if total_call_ratio is None
            else f"{total_call_ratio:.3f}x"
        ),
    )

    print(
        "portfolio REF/CAND   :",
        f"{total_ref_portfolio:.4f}s / "
        f"{total_cand_portfolio:.4f}s",
    )

    print(
        "CAND / REF portfolio :",
        (
            "n/a"
            if total_portfolio_ratio is None
            else f"{total_portfolio_ratio:.3f}x"
        ),
    )

    print(
        "cert REF/CAND        :",
        f"{total_ref_cert:.4f}s / "
        f"{total_cand_cert:.4f}s",
    )

    print(
        "CAND / REF cert      :",
        (
            "n/a"
            if total_cert_ratio is None
            else f"{total_cert_ratio:.3f}x"
        ),
    )

    print(
        "pipeline REF/CAND    :",
        f"{total_ref_pipeline:.4f}s / "
        f"{total_cand_pipeline:.4f}s",
    )

    print(
        "CAND / REF pipeline  :",
        (
            "n/a"
            if total_pipeline_ratio is None
            else f"{total_pipeline_ratio:.3f}x"
        ),
    )

    print(
        "portfolio win cases  :",
        f"{len(portfolio_wins)}/"
        f"{len(informative)}",
    )

    print(
        "pipeline win cases   :",
        f"{len(pipeline_wins)}/"
        f"{len(informative)} "
        f"(required {required_wins})",
    )

    all_contracts = bool(
        len(
            valid
        )
        == len(
            case_rows
        )
    )

    if not all_contracts:
        decision = (
            "SELECTABLE_PYTHON_HARD_GLOBAL_CONTRACT_FAIL"
        )

        note = (
            "At least one selectable-mode case changed exact H-portfolio or "
            "global-K2 semantics. Fix only that mode contract before integration."
        )

    elif len(
        informative
    ) < 4:
        decision = (
            "SELECTABLE_PYTHON_HARD_GLOBAL_TOO_FEW_INFORMATIVE"
        )

        note = (
            "Too few predetermined cases produced DR endpoints at rootLB+3. "
            "Run one additional fixed batch without cherry-picking."
        )

    elif (
        total_call_ratio is not None
        and total_call_ratio <= 0.25
        and total_pipeline_ratio is not None
        and total_pipeline_ratio <= 0.90
        and len(
            pipeline_wins
        )
        >= int(
            required_wins
        )
    ):
        decision = (
            "SELECTABLE_PYTHON_HARD_GLOBAL_CANDIDATE_PASS"
        )

        note = (
            "The frozen selectable Python candidate preserves exact portfolio/"
            "skeleton/global-K2 identity on the new mixed hard batch and again "
            "clears the >=10% hard-pipeline gate with consistent per-case wins. "
            "The FUSED+FULLCAP+RETEST preset is ready to become the default "
            "experimental hard/global engine mode, while stable v37.24.1 and "
            "v37.114 remain unchanged fallback references."
        )

    elif (
        total_pipeline_ratio is not None
        and total_pipeline_ratio < 1.0
    ):
        decision = (
            "SELECTABLE_PYTHON_HARD_GLOBAL_WALL_SIGNAL_BELOW_GATE"
        )

        note = (
            "The selectable candidate remains exact and faster in aggregate, "
            "but misses the predeclared >=10% mixed-batch pipeline threshold. "
            "Keep it experimental rather than default."
        )

    else:
        decision = (
            "SELECTABLE_PYTHON_HARD_GLOBAL_NOT_CONFIRMED"
        )

        note = (
            "The mixed fresh benchmark does not confirm a useful runtime win "
            "for the frozen Python candidate."
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
        "SELECTABLE_PYTHON_HARD_GLOBAL_CANDIDATE_PASS"
    ):
        print(
            "  v37.152 make FUSED+FULLCAP+RETEST the default EXPERIMENTAL "
            "hard/global preset"
        )

        print(
            "  retain explicit reference preset current+auto192+zero"
        )

        print(
            "  run broader mixed regression; stable v37.24.1/v37.114 untouched"
        )

    elif decision == (
        "SELECTABLE_PYTHON_HARD_GLOBAL_CONTRACT_FAIL"
    ):
        print(
            "  fix only the failed mode identity contract"
        )

    else:
        print(
            "  keep candidate selectable but non-default; profile remaining "
            "Python recurse/pose/list-sort overhead"
        )

    payload = {
        "version": "v37.151",
        "mode": (
            "SELECTABLE_PYTHON_HARD_GLOBAL_MODES_MIXED_FRESH_BENCHMARK"
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
            "repeats": int(
                local.repeats
            ),
            "case_time_limit": float(
                local.case_time_limit
            ),
            "initial_r4_cap": int(
                local.initial_r4_cap
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
            "reference_modes": (
                ref_modes
            ),
            "candidate_modes": (
                cand_modes
            ),
        },
        "cases": case_rows,
        "aggregate": {
            "contract_valid_cases": len(
                valid
            ),
            "total_cases": len(
                case_rows
            ),
            "informative_cases": len(
                informative
            ),
            "no_dr_cases": len(
                no_dr
            ),
            "endpoint_size_buckets": (
                bucket_hist
            ),
            "reference_rerun_cases": [
                int(
                    row[
                        "case"
                    ]
                )
                for row in rerun_cases
            ],
            "reference_calls": int(
                total_ref_calls
            ),
            "candidate_calls": int(
                total_cand_calls
            ),
            "candidate_vs_reference_call_ratio": (
                total_call_ratio
            ),
            "reference_portfolio_wall": float(
                total_ref_portfolio
            ),
            "candidate_portfolio_wall": float(
                total_cand_portfolio
            ),
            "candidate_vs_reference_portfolio_ratio": (
                total_portfolio_ratio
            ),
            "reference_cert_wall": float(
                total_ref_cert
            ),
            "candidate_cert_wall": float(
                total_cand_cert
            ),
            "candidate_vs_reference_cert_ratio": (
                total_cert_ratio
            ),
            "reference_pipeline_wall": float(
                total_ref_pipeline
            ),
            "candidate_pipeline_wall": float(
                total_cand_pipeline
            ),
            "candidate_vs_reference_pipeline_ratio": (
                total_pipeline_ratio
            ),
            "portfolio_win_cases": len(
                portfolio_wins
            ),
            "pipeline_win_cases": len(
                pipeline_wins
            ),
            "required_pipeline_win_cases": int(
                required_wins
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
