#!/usr/bin/env python3
"""
CubeLab v37.149 — ISOLATED CURRENT vs FUSED q+LB H-PREFIX A/B

Background
----------
v37.148 on five new hard cases:

    exact contracts       : 5/5
    RETEST calls          : 0.093x reference
    certification wall    : 0.807x
    hard pipeline wall    : 0.939x
    pipeline wins         : 5/5

The remaining dominant cost is H-portfolio generation.

v37.146 cProfile showed the dominant Python self-time:

    recurse
    q_move
    phase1_lb
    pose_step

The current child loop does, for EVERY legal move:

    nq = q_move(parent_q, mi, ...)
        -> unpack_q(parent_q)
        -> transition co/eo/sl
        -> pack_q(child)

    child_lb = phase1_lb(nq, ...)
        -> unpack_q(child_q) AGAIN
        -> two PDB lookups

This repeats integer quotient unpacking and Python function dispatch millions
of times.

FUSED arm
---------
Unpack the parent q ONCE per recurse node:

    parent q -> co, eo, sl

Then each child is computed directly from the existing transition arrays:

    nco = CO[co,mi]
    neo = EO[eo,mi]
    nsl = SL[sl,mi]

    nq = pack(nco,neo,nsl)

    child_lb = max(
        COS[nco,nsl],
        EOS[neo,nsl]
    )

No q_move() call.
No child phase1_lb() call.
No second unpack_q().

Everything else is intentionally identical:
    same allow()
    same child ordering
    same pose stepping
    same exact compiled R4 oracle
    same FULLCAP R4 suffix policy
    same endpoint / skeleton materialization
    same INCUMBENT RETEST certification

A/B
---
Uses the same completed v37.148 five-scramble batch by default.

Every arm run is a FRESH Python process.

CURRENT:
    existing v37.133 FULLCAP H enumeration
    + RETEST

FUSED:
    exact fused q-transition + phase1-LB child generation
    + same FULLCAP
    + same RETEST

Contracts
---------
* same scramble / root LB / H
* complete non-truncated portfolio
* same prefix-node count
* same R4 leaf / word counts
* same endpoint fingerprint
* same skeleton fingerprint
* same exact global K2
* same RETEST terminal-call count

Primary metric:
    isolated H-portfolio wall

Secondary:
    full hard pipeline wall

Decision:
    FUSED exact portfolio candidate is a strong Python win if:
        all contracts pass
        portfolio wall <= 0.90x CURRENT
        portfolio faster in >=75% of cases

Production changes: NONE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
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

import audit_isolated_retest_cert_v37_144 as v144
import audit_fresh_hard_residual_gate_v37_133 as v133
import audit_hard_global_k2_residual_v37_131_1 as v1311
import audit_post_dr_k2_deficit_v37_128 as v128
import audit_fast_stable_selective_terminal_macro_v37_114 as v114


RESULT_MARKER = "__V37149_RESULT__="


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
    )

    ap.add_argument(
        "--source-v148",
        default="reports/v37/fresh_combined_fullcap_retest_v37_148.json",
    )
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--case-time-limit", type=float, default=180.0)
    ap.add_argument("--prefix-node-cap", type=int, default=200000)
    ap.add_argument("--k2-probe-cap", type=int, default=22)
    ap.add_argument("--terminal-call-cap", type=int, default=16000)
    ap.add_argument(
        "--analysis-output",
        default="reports/v37/fused_q_lb_prefix_ab_v37_149.json",
    )

    ap.add_argument("--worker", action="store_true")
    ap.add_argument(
        "--worker-arm",
        choices=("CURRENT", "FUSED"),
        default=None,
    )
    ap.add_argument("--worker-case", type=int, default=None)

    ns, remaining = ap.parse_known_args(argv)

    ns.repeats = max(1, int(ns.repeats))
    ns.case_time_limit = max(1.0, float(ns.case_time_limit))
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


def _enumerate_fused(
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
    Exact clone of v37.133 direct H enumeration except for one optimization:
    parent q is unpacked once, and child q + phase1 LB are fused inline.
    """
    ts = v114.v24.grc.ts

    start = time.perf_counter()
    deadline = start + float(time_limit)

    stats = Counter()
    skeletons = []
    endpoints = {}

    cut = {
        "node_cap": False,
        "time": False,
    }

    # Local aliases are deliberate: reduce Python global/attribute dispatch
    # without changing the search semantics.
    move_order = ts.MOVE_ORDER
    t_moves = ts.T_MOVES
    allow = ts.allow

    co_table = solver.co
    eo_table = solver.eo
    sl_table = solver.sl
    cos_pdb = solver.cos
    eos_pdb = solver.eos

    NM = int(ts.NM)
    NEO = int(ts.NEO)
    NS = int(ts.NS)

    suffix_terminal_state = v114._suffix_terminal_state
    pose_step = v114._pose_step
    combined_lb_metrics = v133._combined_lb_metrics
    iter_bits = v133._iter_bits

    def should_cut():
        if int(stats["prefix_nodes"]) >= int(prefix_node_cap):
            cut["node_cap"] = True
            return True

        if time.perf_counter() >= deadline:
            cut["time"] = True
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

        if (
            int(cur_q) == int(ts.GOAL_Q)
            and int(remaining_h) != 0
        ):
            return

        if int(remaining_h) == 4:
            stats["r4_leaf_states"] += 1

            qr = oracle.query(
                poses=tuple(cur_poses),
                q=int(cur_q),
                R=4,
                last_face=last_face,
                add_q_first=False,
            )

            if not qr["feasible"]:
                return

            mask = int(qr["word_mask"])
            qcount = int(
                qr.get(
                    "word_count",
                    mask.bit_count(),
                )
            )

            if qcount <= 0:
                return

            stats["r4_oracle_nonempty"] += 1
            stats["r4_word_sum"] += int(qcount)
            stats["max_r4_words"] = max(
                int(stats["max_r4_words"]),
                int(qcount),
            )

            if qcount > int(r4_suffix_cap):
                stats["r4_cap_truncated"] += 1

            words = oracle.words[4]

            for r4_rank, word_idx in enumerate(
                iter_bits(mask),
                start=1,
            ):
                if r4_rank > int(r4_suffix_cap):
                    break

                if should_cut():
                    break

                stats["r4_words_considered"] += 1

                row = words[int(word_idx)]

                r4_suffix = tuple(
                    move_order[int(mi)]
                    for mi in row
                )

                (
                    end_poses,
                    end_q,
                    q_before_final,
                    final_face,
                ) = suffix_terminal_state(
                    solver,
                    tuple(cur_poses),
                    int(cur_q),
                    row,
                )

                if int(end_q) != int(ts.GOAL_Q):
                    continue

                if int(q_before_final) == int(ts.GOAL_Q):
                    continue

                key = (
                    tuple(
                        int(x)
                        for x in end_poses
                    ),
                    final_face,
                )

                if key not in endpoints:
                    lb = combined_lb_metrics(
                        solver=solver,
                        end_poses=end_poses,
                    )

                    endpoints[key] = {
                        "end_poses": tuple(
                            int(x)
                            for x in end_poses
                        ),
                        "end_q": int(end_q),
                        "final_face": final_face,
                        "p2lb": lb["p2lb"],
                        "combined_lb": lb[
                            "combined_lb"
                        ],
                        "combined_available": bool(
                            lb["combined_available"]
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
                        "r4_rank": int(r4_rank),
                        "r4_suffix": list(
                            r4_suffix
                        ),
                    }
                )

            return

        if int(remaining_h) <= 4:
            return

        # --------- FUSED q transition + phase1 LB ----------
        # Unpack parent q exactly once for this recurse node.
        packed = int(cur_q)

        parent_sl = packed % NS
        packed //= NS
        parent_eo = packed % NEO
        parent_co = packed // NEO

        co_base = int(parent_co) * NM
        eo_base = int(parent_eo) * NM
        sl_base = int(parent_sl) * NM

        children = []

        next_remaining = int(remaining_h) - 1

        for mi, move in enumerate(move_order):
            if not allow(
                last_face,
                move,
            ):
                continue

            nco = int(
                co_table[
                    co_base + int(mi)
                ]
            )

            neo = int(
                eo_table[
                    eo_base + int(mi)
                ]
            )

            nsl = int(
                sl_table[
                    sl_base + int(mi)
                ]
            )

            nq = (
                (
                    int(nco)
                    * NEO
                    + int(neo)
                )
                * NS
                + int(nsl)
            )

            child_lb = max(
                int(
                    cos_pdb[
                        int(nco) * NS
                        + int(nsl)
                    ]
                ),
                int(
                    eos_pdb[
                        int(neo) * NS
                        + int(nsl)
                    ]
                ),
            )

            if int(child_lb) > int(next_remaining):
                continue

            event_rank = (
                0
                if move in t_moves
                else 1
            )

            children.append(
                (
                    int(child_lb),
                    int(event_rank),
                    str(move),
                    int(mi),
                    int(nq),
                )
            )

        children.sort(
            key=lambda x: (
                x[0],
                x[1],
                x[2],
            )
        )

        if not prefix_moves:
            stats["root_q_children"] = len(
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

            stats["prefix_nodes"] += 1

            nposes = pose_step(
                tuple(cur_poses),
                int(mi),
            )

            recurse(
                tuple(nposes),
                int(nq),
                int(next_remaining),
                str(move)[0],
                tuple(prefix_moves)
                + (str(move),),
                tuple(prefix_ranks)
                + (int(local_rank),),
            )

    stats["root_phase1_lb"] = int(
        ts.phase1_lb(
            int(q),
            solver.cos,
            solver.eos,
        )
    )

    recurse(
        tuple(poses),
        int(q),
        int(h_length),
        None,
        (),
        (),
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return {
        "stats": stats,
        "cuts": cut,
        "elapsed": float(elapsed),
        "skeletons": skeletons,
        "endpoints": endpoints,
    }


def _worker_main(local, remaining):
    if (
        local.worker_arm is None
        or local.worker_case is None
    ):
        raise SystemExit(
            "--worker requires --worker-arm and --worker-case"
        )

    source_path = Path(
        local.source_v148
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

    moves = tuple(
        source_row["scramble"]
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

    expected_root = int(
        source_row[
            "runs"
        ][
            "REFERENCE"
        ][0][
            "root_lb"
        ]
    )

    expected_h = int(
        source_row[
            "runs"
        ][
            "REFERENCE"
        ][0][
            "h_length"
        ]
    )

    if int(root_lb) != int(expected_root):
        raise RuntimeError(
            f"root LB mismatch: "
            f"{root_lb} != {expected_root}"
        )

    h_length = int(expected_h)

    full_cap = int(
        len(
            oracle.words[4]
        )
    )

    hard_t0 = time.perf_counter()
    portfolio_t0 = hard_t0

    if local.worker_arm == "CURRENT":
        portfolio = (
            v133._enumerate_direct_h_complete(
                solver=solver,
                oracle=oracle,
                poses=tuple(poses),
                q=int(q),
                h_length=int(h_length),
                r4_suffix_cap=int(full_cap),
                prefix_node_cap=int(
                    local.prefix_node_cap
                ),
                time_limit=float(
                    local.case_time_limit
                ),
            )
        )

    else:
        portfolio = _enumerate_fused(
            solver=solver,
            oracle=oracle,
            poses=tuple(poses),
            q=int(q),
            h_length=int(h_length),
            r4_suffix_cap=int(full_cap),
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
            "arm": local.worker_arm,
            "case": int(local.worker_case),
            "complete": False,
            "reason": "portfolio_incomplete",
            "stable_return_code": int(rc),
            "bootstrap_wall": float(
                bootstrap_wall
            ),
            "portfolio_wall": float(
                portfolio_wall
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

    stats = portfolio["stats"]

    endpoint_fp = _endpoint_fingerprint(
        portfolio
    )

    skeleton_fp = _skeleton_fingerprint(
        portfolio
    )

    raw_endpoints = len(
        portfolio["endpoints"]
    )

    deadline = (
        hard_t0
        + float(local.case_time_limit)
    )

    cert_t0 = time.perf_counter()

    if raw_endpoints == 0:
        cert = {
            "certified": True,
            "exact_best_k2": None,
            "terminal_calls": 0,
            "bad_status": None,
        }

    else:
        cert = v144._run_retest(
            solver=solver,
            scrambled_poses=tuple(poses),
            portfolio=portfolio,
            probe_cap=int(local.k2_probe_cap),
            call_cap=int(
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
        "arm": local.worker_arm,
        "case": int(local.worker_case),
        "scramble": list(moves),
        "root_lb": int(root_lb),
        "h_length": int(h_length),
        "full_r4_cap": int(full_cap),
        "prefix_nodes": int(
            stats["prefix_nodes"]
        ),
        "r4_leaf_states": int(
            stats["r4_leaf_states"]
        ),
        "r4_words_considered": int(
            stats["r4_words_considered"]
        ),
        "raw_endpoints": int(
            raw_endpoints
        ),
        "skeletons": int(
            len(
                portfolio["skeletons"]
            )
        ),
        "endpoint_fingerprint": str(
            endpoint_fp
        ),
        "skeleton_fingerprint": str(
            skeleton_fp
        ),
        "exact_k2": (
            None
            if cert.get(
                "exact_best_k2"
            )
            is None
            else int(
                cert["exact_best_k2"]
            )
        ),
        "terminal_calls": int(
            cert.get(
                "terminal_calls",
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
    arm,
):
    script = Path(
        __file__
    ).resolve()

    cmd = [
        sys.executable,
        str(script),
        "--worker",
        "--worker-arm",
        str(arm),
        "--worker-case",
        str(case_index),
        "--source-v148",
        str(local.source_v148),
        "--case-time-limit",
        str(local.case_time_limit),
        "--prefix-node-cap",
        str(local.prefix_node_cap),
        "--k2-probe-cap",
        str(local.k2_probe_cap),
        "--terminal-call-cap",
        str(local.terminal_call_cap),
        *remaining,
    ]

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

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
            proc.stdout.splitlines()[-40:]
        )

        raise RuntimeError(
            f"worker failed arm={arm} "
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


def _median(rows, key):
    vals = [
        float(row[key])
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


def _stable_set(rows, key):
    return {
        row.get(key)
        for row in rows
    }


def _controller_main(local, remaining):
    source_path = Path(
        local.source_v148
    )

    source = json.loads(
        source_path.read_text(
            encoding="utf-8",
        )
    )

    source_cases = list(
        source.get(
            "cases",
            [],
        )
    )

    if not source_cases:
        raise RuntimeError(
            "source v37.148 has no cases"
        )

    print(
        "# CubeLab v37.149 - "
        "ISOLATED CURRENT vs FUSED q+LB H-PREFIX A/B"
    )
    print(
        "source v37.148       :",
        source_path,
    )
    print(
        "cases                :",
        len(source_cases),
    )
    print(
        "repeats              :",
        int(local.repeats),
    )
    print(
        "R4 policy            :",
        "FULLCAP in both arms",
    )
    print(
        "certification        :",
        "INCUMBENT RETEST in both arms",
    )
    print(
        "CURRENT child loop   :",
        "q_move + phase1_lb (two unpack_q paths)",
    )
    print(
        "FUSED child loop     :",
        "one parent unpack -> direct child q + PDB LB",
    )
    print(
        "isolation            :",
        "fresh Python process per arm run",
    )
    print(
        "production changes   :",
        "NONE",
    )

    print()
    print(
        "# ISOLATED CASES"
    )

    case_rows = []

    for position, source_row in enumerate(
        source_cases,
        start=1,
    ):
        case_index = int(
            source_row["case"]
        )

        print()
        print(
            f"[CASE {case_index}] "
            f"source raw={source_row['raw_endpoints']}"
        )

        arms = {
            "CURRENT": [],
            "FUSED": [],
        }

        for repeat in range(
            int(local.repeats)
        ):
            current_first = bool(
                (
                    int(position)
                    + int(repeat)
                )
                % 2
                == 0
            )

            order = (
                ["CURRENT", "FUSED"]
                if current_first
                else ["FUSED", "CURRENT"]
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

                arms[arm].append(
                    result
                )

                print(
                    f"    {arm:<7} "
                    f"prefix={result['prefix_nodes']:<7} "
                    f"raw={result['raw_endpoints']:<4} "
                    f"K2={str(result['exact_k2']):<4} "
                    f"calls={result['terminal_calls']:<4} "
                    f"portfolio={result['portfolio_wall']:.4f}s "
                    f"pipeline={result['pipeline_wall']:.4f}s "
                    f"complete={result['complete']}"
                )

        current = arms["CURRENT"]
        fused = arms["FUSED"]

        all_complete = bool(
            all(
                row.get(
                    "complete",
                    False,
                )
                for row in (
                    current
                    + fused
                )
            )
        )

        contracts = {
            "all_complete": bool(
                all_complete
            ),
            "root_h_identity": bool(
                _stable_set(
                    current,
                    "root_lb",
                )
                == _stable_set(
                    fused,
                    "root_lb",
                )
                and _stable_set(
                    current,
                    "h_length",
                )
                == _stable_set(
                    fused,
                    "h_length",
                )
            ),
            "prefix_count_identity": bool(
                _stable_set(
                    current,
                    "prefix_nodes",
                )
                == _stable_set(
                    fused,
                    "prefix_nodes",
                )
                and _stable_set(
                    current,
                    "r4_leaf_states",
                )
                == _stable_set(
                    fused,
                    "r4_leaf_states",
                )
            ),
            "r4_word_identity": bool(
                _stable_set(
                    current,
                    "r4_words_considered",
                )
                == _stable_set(
                    fused,
                    "r4_words_considered",
                )
            ),
            "endpoint_identity": bool(
                _stable_set(
                    current,
                    "endpoint_fingerprint",
                )
                == _stable_set(
                    fused,
                    "endpoint_fingerprint",
                )
            ),
            "skeleton_identity": bool(
                _stable_set(
                    current,
                    "skeleton_fingerprint",
                )
                == _stable_set(
                    fused,
                    "skeleton_fingerprint",
                )
            ),
            "exact_identity": bool(
                _stable_set(
                    current,
                    "exact_k2",
                )
                == _stable_set(
                    fused,
                    "exact_k2",
                )
            ),
            "retest_call_identity": bool(
                _stable_set(
                    current,
                    "terminal_calls",
                )
                == _stable_set(
                    fused,
                    "terminal_calls",
                )
            ),
        }

        current_portfolio = _median(
            current,
            "portfolio_wall",
        )

        fused_portfolio = _median(
            fused,
            "portfolio_wall",
        )

        current_pipeline = _median(
            current,
            "pipeline_wall",
        )

        fused_pipeline = _median(
            fused,
            "pipeline_wall",
        )

        current_cert = _median(
            current,
            "cert_wall",
        )

        fused_cert = _median(
            fused,
            "cert_wall",
        )

        portfolio_ratio = (
            None
            if (
                current_portfolio is None
                or current_portfolio <= 0
            )
            else (
                float(fused_portfolio)
                / float(current_portfolio)
            )
        )

        pipeline_ratio = (
            None
            if (
                current_pipeline is None
                or current_pipeline <= 0
            )
            else (
                float(fused_pipeline)
                / float(current_pipeline)
            )
        )

        row = {
            "case": int(case_index),
            "runs": arms,
            "contracts": contracts,
            "median": {
                "current_portfolio_wall": (
                    current_portfolio
                ),
                "fused_portfolio_wall": (
                    fused_portfolio
                ),
                "portfolio_ratio": (
                    portfolio_ratio
                ),
                "current_cert_wall": (
                    current_cert
                ),
                "fused_cert_wall": (
                    fused_cert
                ),
                "current_pipeline_wall": (
                    current_pipeline
                ),
                "fused_pipeline_wall": (
                    fused_pipeline
                ),
                "pipeline_ratio": (
                    pipeline_ratio
                ),
            },
        }

        case_rows.append(row)

        print(
            "  contracts          :",
            contracts,
        )
        print(
            "  MEDIAN portfolio   :",
            f"CURRENT={current_portfolio:.4f}s "
            f"FUSED={fused_portfolio:.4f}s "
            f"ratio={portfolio_ratio:.3f}x",
        )
        print(
            "  MEDIAN pipeline    :",
            f"CURRENT={current_pipeline:.4f}s "
            f"FUSED={fused_pipeline:.4f}s "
            f"ratio={pipeline_ratio:.3f}x",
        )

    valid = [
        row
        for row in case_rows
        if all(
            row["contracts"].values()
        )
    ]

    total_current_portfolio = sum(
        float(
            row[
                "median"
            ][
                "current_portfolio_wall"
            ]
        )
        for row in valid
    )

    total_fused_portfolio = sum(
        float(
            row[
                "median"
            ][
                "fused_portfolio_wall"
            ]
        )
        for row in valid
    )

    portfolio_ratio = (
        None
        if total_current_portfolio <= 0
        else (
            float(total_fused_portfolio)
            / float(total_current_portfolio)
        )
    )

    total_current_pipeline = sum(
        float(
            row[
                "median"
            ][
                "current_pipeline_wall"
            ]
        )
        for row in valid
    )

    total_fused_pipeline = sum(
        float(
            row[
                "median"
            ][
                "fused_pipeline_wall"
            ]
        )
        for row in valid
    )

    pipeline_ratio = (
        None
        if total_current_pipeline <= 0
        else (
            float(total_fused_pipeline)
            / float(total_current_pipeline)
        )
    )

    portfolio_wins = [
        row
        for row in valid
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
            * len(valid)
            + 3
        )
        // 4,
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
        "portfolio CURRENT/FUSED:",
        f"{total_current_portfolio:.4f}s / "
        f"{total_fused_portfolio:.4f}s",
    )
    print(
        "FUSED / CURRENT portfolio:",
        (
            "n/a"
            if portfolio_ratio is None
            else f"{portfolio_ratio:.3f}x"
        ),
    )
    print(
        "pipeline CURRENT/FUSED:",
        f"{total_current_pipeline:.4f}s / "
        f"{total_fused_pipeline:.4f}s",
    )
    print(
        "FUSED / CURRENT pipeline:",
        (
            "n/a"
            if pipeline_ratio is None
            else f"{pipeline_ratio:.3f}x"
        ),
    )
    print(
        "portfolio win cases  :",
        f"{len(portfolio_wins)}/{len(valid)} "
        f"(required {required_wins})",
    )

    all_contracts = bool(
        len(valid) == len(case_rows)
    )

    if not all_contracts:
        decision = (
            "FUSED_Q_LB_PREFIX_CONTRACT_FAIL"
        )
        note = (
            "At least one fused H-prefix run changed the exact portfolio, "
            "skeleton ordering, or RETEST result. Fix the fused coordinate "
            "contract before interpreting performance."
        )

    elif (
        portfolio_ratio is not None
        and portfolio_ratio <= 0.90
        and len(
            portfolio_wins
        )
        >= int(
            required_wins
        )
    ):
        decision = (
            "FUSED_Q_LB_PREFIX_PYTHON_WALL_WIN"
        )
        note = (
            "Fusing parent quotient unpack, child quotient transition, and "
            "phase1 PDB lower-bound lookup preserves exact H-portfolio identity "
            "while reducing isolated H-portfolio wall by at least 10% with "
            "consistent case wins. This is a real Python-kernel optimization "
            "and should be validated in a fresh combined reference A/B."
        )

    elif (
        portfolio_ratio is not None
        and portfolio_ratio <= 0.95
    ):
        decision = (
            "FUSED_Q_LB_PREFIX_PYTHON_WALL_SIGNAL"
        )
        note = (
            "The fused quotient/LB child loop preserves exact identity and "
            "reduces H-portfolio wall by at least 5%, but does not yet clear the "
            "strong 10% kernel threshold."
        )

    elif (
        portfolio_ratio is not None
        and portfolio_ratio < 1.0
    ):
        decision = (
            "FUSED_Q_LB_PREFIX_PYTHON_GAIN_WEAK"
        )
        note = (
            "The exact fused child loop is slightly faster, but the remaining "
            "Python recurse/pose overhead dominates. Do not over-tune this path."
        )

    else:
        decision = (
            "FUSED_Q_LB_PREFIX_NO_WALL_WIN"
        )
        note = (
            "Removing duplicate q unpack/function-dispatch work does not reduce "
            "isolated H-portfolio wall on this batch. The remaining bottleneck "
            "is deeper Python recursion/pose stepping rather than quotient/LB "
            "function overhead."
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
        "FUSED_Q_LB_PREFIX_PYTHON_WALL_WIN"
    ):
        print(
            "  v37.150 new-batch contemporaneous A/B:"
        )
        print(
            "    reference=AUTO192+ZERO"
        )
        print(
            "    candidate=FUSED-FULLCAP+RETEST"
        )
    elif decision == (
        "FUSED_Q_LB_PREFIX_CONTRACT_FAIL"
    ):
        print(
            "  fix only the fused coordinate contract; do not benchmark yet"
        )
    else:
        print(
            "  profile remaining recurse/pose cost before another Python micro-optimization"
        )

    payload = {
        "version": "v37.149",
        "mode": "ISOLATED_CURRENT_VS_FUSED_Q_LB_H_PREFIX_AB",
        "source_v148": str(source_path),
        "config": {
            "repeats": int(local.repeats),
            "case_time_limit": float(
                local.case_time_limit
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
        "cases": case_rows,
        "aggregate": {
            "valid_cases": len(valid),
            "total_cases": len(case_rows),
            "current_portfolio_wall": float(
                total_current_portfolio
            ),
            "fused_portfolio_wall": float(
                total_fused_portfolio
            ),
            "fused_vs_current_portfolio_ratio": (
                portfolio_ratio
            ),
            "current_pipeline_wall": float(
                total_current_pipeline
            ),
            "fused_pipeline_wall": float(
                total_fused_pipeline
            ),
            "fused_vs_current_pipeline_ratio": (
                pipeline_ratio
            ),
            "portfolio_win_cases": len(
                portfolio_wins
            ),
            "required_portfolio_win_cases": int(
                required_wins
            ),
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
