#!/usr/bin/env python3
"""
CubeLab v37.140 — RESIDUAL-AWARE H-ALLOCATION FRONTIER

For each fresh hard scramble and each H in rootLB + offsets (default 0..3):

    complete fixed-H raw DR portfolio
    min combined K2 LB (minC)
    proxy total = H + minC

Then certify the exact global-best K2 over that fixed-H raw endpoint portfolio
with admissible combined-LB branch-and-bound:

    endpoints ordered by C = combined K2 LB

    acquire first exact witness
    incumbent = exact K2

    for each remaining endpoint:
        if C >= incumbent:
            cannot strictly improve -> bound skip
        else:
            exact probe only through incumbent-1

Per case compare:

    proxy-selected H = argmin(H + minC)
    exact-selected H = argmin(H + exactBestK2)
    fixed rootLB+3 exact total versus exact-selected total

A complete H with no DR endpoint is +infinity, not an audit failure.

OFFLINE audit only.
Production changes: NONE.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if SRC.is_dir():
    s = str(SRC)
    if s not in sys.path:
        sys.path.insert(0, s)

import audit_fresh_p2plus1_secondary_v37_137 as v137
import audit_fresh_hard_residual_gate_v37_133 as v133
import audit_hard_global_k2_residual_v37_131_1 as v1311
import audit_post_dr_k2_deficit_v37_128 as v128
import audit_fast_stable_selective_terminal_macro_v37_114 as v114


CONTROL_TOKENS = tuple(
    "D' R F2 U D' R L B D R' U D R' D F' B U F' L' D2 B L D' F R'".split()
)


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
    )
    ap.add_argument("--fresh-count", type=int, default=3)
    ap.add_argument("--fresh-seed", type=int, default=20261047)
    ap.add_argument("--scramble-length", type=int, default=25)
    ap.add_argument(
        "--h-offsets",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3],
    )
    ap.add_argument("--initial-r4-cap", type=int, default=192)
    ap.add_argument("--prefix-node-cap", type=int, default=200000)
    ap.add_argument("--case-time-limit", type=float, default=180.0)
    ap.add_argument("--k2-probe-cap", type=int, default=22)
    ap.add_argument("--exact-call-cap-per-h", type=int, default=12000)
    ap.add_argument(
        "--analysis-output",
        default="reports/v37/residual_aware_h_allocation_v37_140.json",
    )

    ns, remaining = ap.parse_known_args(argv)

    ns.fresh_count = max(1, int(ns.fresh_count))
    ns.scramble_length = max(1, int(ns.scramble_length))
    ns.h_offsets = tuple(sorted({max(0, int(x)) for x in ns.h_offsets}))
    ns.initial_r4_cap = max(1, int(ns.initial_r4_cap))
    ns.prefix_node_cap = max(1, int(ns.prefix_node_cap))
    ns.case_time_limit = max(1.0, float(ns.case_time_limit))
    ns.k2_probe_cap = max(0, int(ns.k2_probe_cap))
    ns.exact_call_cap_per_h = max(1, int(ns.exact_call_cap_per_h))

    return ns, remaining


def _portfolio_complete(portfolio):
    if portfolio is None:
        return False

    return bool(
        not portfolio["cuts"]["node_cap"]
        and not portfolio["cuts"]["time"]
        and int(portfolio["stats"]["r4_cap_truncated"]) == 0
    )


def _first_skeletons(portfolio):
    out = {}

    for skeleton in portfolio["skeletons"]:
        key = skeleton["endpoint_key"]
        if key not in out:
            out[key] = skeleton

    return out


def _probe(*, solver, endpoint, cap, calls, deadline):
    return v128._probe_min_k2(
        solver=solver,
        end_poses=endpoint["end_poses"],
        end_q=int(endpoint["end_q"]),
        final_face=endpoint["final_face"],
        k2_probe_cap=int(cap),
        global_calls=calls,
        deadline=deadline,
    )


def _verify(*, scrambled_poses, skeleton, probe):
    if (
        skeleton is None
        or probe.get("status") != "FOUND"
        or probe.get("min_k2") is None
    ):
        return False

    return v133._verify_solution(
        scrambled_poses=tuple(scrambled_poses),
        skeleton=skeleton,
        k2_word=tuple(probe.get("word", ())),
    )


def _certify_fixed_h(
    *,
    solver,
    scrambled_poses,
    portfolio,
    k2_probe_cap,
    exact_call_cap,
    deadline,
):
    raw_items = list(portfolio["endpoints"].items())

    if not raw_items:
        return {
            "certified": True,
            "no_dr_endpoints": True,
            "raw_endpoints": 0,
            "min_c": None,
            "exact_best_k2": None,
            "exact_winner_c": None,
            "winner_delta_from_min_c": None,
            "endpoint_probes": 0,
            "exact_budget_calls": 0,
            "bound_skipped": 0,
            "improvements": [],
            "bad_status": None,
        }

    raw_items.sort(
        key=lambda item: (
            int(item[1]["combined_lb"]),
            int(item[1]["p2lb"]),
        )
    )

    min_c = int(raw_items[0][1]["combined_lb"])
    first_skeleton = _first_skeletons(portfolio)

    calls = {
        "calls": 0,
        "cap": int(exact_call_cap),
        "cache": {},
    }

    incumbent = None
    winner_c = None
    endpoint_probes = 0
    bound_skipped = 0
    bad_status = None
    improvements = []
    first_witness_index = None

    # Acquire first exact witness in increasing combined-LB order.
    for idx, (key, endpoint) in enumerate(raw_items):
        if (
            time.perf_counter() >= deadline
            or int(calls["calls"]) >= int(calls["cap"])
        ):
            bad_status = "RESOURCE_CUT_BEFORE_WITNESS"
            break

        endpoint_probes += 1

        probe = _probe(
            solver=solver,
            endpoint=endpoint,
            cap=int(k2_probe_cap),
            calls=calls,
            deadline=deadline,
        )

        status = str(probe["status"])

        if status == "ABOVE_CAP":
            continue

        if status != "FOUND":
            bad_status = status
            break

        solved = _verify(
            scrambled_poses=tuple(scrambled_poses),
            skeleton=first_skeleton.get(key),
            probe=probe,
        )

        if not solved:
            bad_status = "REPLAY_FAIL_INITIAL"
            break

        incumbent = int(probe["min_k2"])
        winner_c = int(endpoint["combined_lb"])
        first_witness_index = int(idx)

        improvements.append(
            {
                "phase": "initial",
                "combined_lb": int(winner_c),
                "new_incumbent": int(incumbent),
                "solved_replay": True,
            }
        )
        break

    if bad_status is not None:
        return {
            "certified": False,
            "no_dr_endpoints": False,
            "raw_endpoints": len(raw_items),
            "min_c": int(min_c),
            "exact_best_k2": None,
            "exact_winner_c": None,
            "winner_delta_from_min_c": None,
            "endpoint_probes": int(endpoint_probes),
            "exact_budget_calls": int(calls["calls"]),
            "bound_skipped": int(bound_skipped),
            "improvements": improvements,
            "bad_status": bad_status,
        }

    if incumbent is None:
        return {
            "certified": False,
            "no_dr_endpoints": False,
            "raw_endpoints": len(raw_items),
            "min_c": int(min_c),
            "exact_best_k2": None,
            "exact_winner_c": None,
            "winner_delta_from_min_c": None,
            "endpoint_probes": int(endpoint_probes),
            "exact_budget_calls": int(calls["calls"]),
            "bound_skipped": int(bound_skipped),
            "improvements": improvements,
            "bad_status": "NO_WITNESS_WITHIN_PROBE_CAP",
        }

    # Certify remaining endpoints.
    start = int(first_witness_index) + 1

    for pos in range(start, len(raw_items)):
        key, endpoint = raw_items[pos]

        if (
            time.perf_counter() >= deadline
            or int(calls["calls"]) >= int(calls["cap"])
        ):
            bad_status = "RESOURCE_CUT_CERTIFICATION"
            break

        c = int(endpoint["combined_lb"])

        if c >= int(incumbent):
            bound_skipped = len(raw_items) - int(pos)
            break

        endpoint_probes += 1

        probe = _probe(
            solver=solver,
            endpoint=endpoint,
            cap=int(incumbent) - 1,
            calls=calls,
            deadline=deadline,
        )

        status = str(probe["status"])

        if status == "ABOVE_CAP":
            continue

        if status != "FOUND":
            bad_status = status
            break

        exact_k2 = int(probe["min_k2"])

        if exact_k2 >= int(incumbent):
            continue

        solved = _verify(
            scrambled_poses=tuple(scrambled_poses),
            skeleton=first_skeleton.get(key),
            probe=probe,
        )

        if not solved:
            bad_status = "REPLAY_FAIL_CHALLENGER"
            break

        old = int(incumbent)
        incumbent = int(exact_k2)
        winner_c = int(c)

        improvements.append(
            {
                "phase": "challenger",
                "combined_lb": int(c),
                "old_incumbent": int(old),
                "new_incumbent": int(incumbent),
                "solved_replay": True,
            }
        )

    certified = bool(bad_status is None)

    return {
        "certified": bool(certified),
        "no_dr_endpoints": False,
        "raw_endpoints": len(raw_items),
        "min_c": int(min_c),
        "exact_best_k2": (
            int(incumbent)
            if certified
            else None
        ),
        "exact_winner_c": (
            int(winner_c)
            if certified
            else None
        ),
        "winner_delta_from_min_c": (
            int(winner_c - min_c)
            if certified
            else None
        ),
        "endpoint_probes": int(endpoint_probes),
        "exact_budget_calls": int(calls["calls"]),
        "bound_skipped": int(bound_skipped),
        "improvements": improvements,
        "bad_status": bad_status,
    }


def main():
    local, remaining = _parse_args(sys.argv[1:])

    v114._combined_installer = (
        v1311._v37131_combined_installer
    )

    sys.argv = [
        sys.argv[0],
        *remaining,
    ]

    print(
        "# CubeLab v37.140 - "
        "RESIDUAL-AWARE H-ALLOCATION FRONTIER"
    )
    print("fresh cases          :", int(local.fresh_count))
    print("fresh seed           :", int(local.fresh_seed))
    print("scramble length      :", int(local.scramble_length))
    print("H offsets            :", list(local.h_offsets))
    print("proxy allocation     :", "minimize H + minC")
    print(
        "exact allocation     :",
        "minimize H + branch-and-bound-certified best K2",
    )
    print("production changes   :", "NONE")

    print()
    print("# STABLE BOOTSTRAP / REPLAY")

    rc = v114.main()

    solver = v1311._CAPTURED_SOLVER

    if solver is None:
        raise RuntimeError("failed to capture solver")

    oracle = v114._ORACLE

    if oracle is None:
        raise RuntimeError("v37.114 oracle unavailable")

    if v1311._BASE_FIND_NO_SHADOW is None:
        raise RuntimeError("terminal finder capture unavailable")

    v128._BASE_FIND_NO_SHADOW = (
        v1311._BASE_FIND_NO_SHADOW
    )

    rng = random.Random(int(local.fresh_seed))
    scrambles = []

    while len(scrambles) < int(local.fresh_count):
        candidate = v133._generate_reduced_scramble(
            rng=rng,
            length=int(local.scramble_length),
        )

        if candidate == CONTROL_TOKENS:
            continue

        if candidate in scrambles:
            continue

        scrambles.append(candidate)

    print()
    print("# ALLOCATION CASES")

    case_rows = []

    for case_index, moves in enumerate(scrambles, start=1):
        case_start = time.perf_counter()
        deadline = case_start + float(local.case_time_limit)

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

        print()
        print(
            f"[CASE {case_index}/{local.fresh_count}] rootLB={root_lb}"
        )
        print("scramble             :", " ".join(moves))

        h_rows = []
        case_bad = None

        for offset in local.h_offsets:
            if time.perf_counter() >= deadline:
                case_bad = "CASE_TIME_EXHAUSTED"
                break

            h_length = int(root_lb + int(offset))
            remaining = deadline - time.perf_counter()

            portfolio, r4_cap, passes, _extra = (
                v137._auto_complete_portfolio(
                    solver=solver,
                    oracle=oracle,
                    poses=tuple(poses),
                    q=int(q),
                    h_length=int(h_length),
                    initial_r4_cap=int(local.initial_r4_cap),
                    prefix_node_cap=int(local.prefix_node_cap),
                    time_limit=float(max(0.0, remaining)),
                )
            )

            if not _portfolio_complete(portfolio):
                h_rows.append(
                    {
                        "offset": int(offset),
                        "h_length": int(h_length),
                        "complete_enumeration": False,
                        "passes": passes,
                        "certified": False,
                        "bad_status": "PORTFOLIO_INCOMPLETE",
                    }
                )

                case_bad = f"H{h_length}_PORTFOLIO_INCOMPLETE"
                break

            cert = _certify_fixed_h(
                solver=solver,
                scrambled_poses=tuple(poses),
                portfolio=portfolio,
                k2_probe_cap=int(local.k2_probe_cap),
                exact_call_cap=int(local.exact_call_cap_per_h),
                deadline=deadline,
            )

            if cert["no_dr_endpoints"]:
                proxy_total = math.inf
                exact_total = math.inf
            else:
                proxy_total = (
                    int(h_length)
                    + int(cert["min_c"])
                )

                exact_total = (
                    math.inf
                    if not cert["certified"]
                    else (
                        int(h_length)
                        + int(cert["exact_best_k2"])
                    )
                )

            row = {
                "offset": int(offset),
                "h_length": int(h_length),
                "complete_enumeration": True,
                "passes": passes,
                "r4_cap": int(r4_cap),
                "prefix_nodes": int(
                    portfolio["stats"]["prefix_nodes"]
                ),
                "raw_skeletons": len(
                    portfolio["skeletons"]
                ),
                **cert,
                "proxy_total_h_plus_min_c": (
                    None
                    if math.isinf(proxy_total)
                    else int(proxy_total)
                ),
                "exact_total_h_plus_k2": (
                    None
                    if math.isinf(exact_total)
                    else int(exact_total)
                ),
            }

            h_rows.append(row)

            print(
                f"  H={h_length:<2} "
                f"ep={cert['raw_endpoints']:<4} "
                f"minC={cert['min_c']} "
                f"proxyTotal={row['proxy_total_h_plus_min_c']} "
                f"exactK2={cert['exact_best_k2']} "
                f"exactTotal={row['exact_total_h_plus_k2']} "
                f"winnerC={cert['exact_winner_c']} "
                f"probeEP={cert['endpoint_probes']} "
                f"skip={cert['bound_skipped']} "
                f"cert={cert['certified']}"
            )

            if not cert["certified"]:
                case_bad = (
                    f"H{h_length}_EXACT_INCONCLUSIVE:"
                    + str(cert["bad_status"])
                )
                break

        finite_exact = [
            row
            for row in h_rows
            if (
                row.get("certified", False)
                and row.get("exact_total_h_plus_k2") is not None
            )
        ]

        finite_proxy = [
            row
            for row in h_rows
            if row.get("proxy_total_h_plus_min_c") is not None
        ]

        all_offsets_completed = bool(
            len(h_rows) == len(local.h_offsets)
            and all(
                row.get("complete_enumeration", False)
                for row in h_rows
            )
            and all(
                (
                    row.get("no_dr_endpoints", False)
                    or row.get("certified", False)
                )
                for row in h_rows
            )
        )

        if not finite_exact:
            case_complete = False
            proxy_best = None
            exact_best = None
        else:
            case_complete = bool(all_offsets_completed)

            proxy_best = (
                min(
                    finite_proxy,
                    key=lambda row: (
                        int(row["proxy_total_h_plus_min_c"]),
                        int(row["h_length"]),
                    ),
                )
                if finite_proxy
                else None
            )

            exact_best = min(
                finite_exact,
                key=lambda row: (
                    int(row["exact_total_h_plus_k2"]),
                    int(row["h_length"]),
                ),
            )

        fixed_plus3_h = root_lb + 3

        fixed_plus3_row = next(
            (
                row
                for row in h_rows
                if int(row["h_length"]) == int(fixed_plus3_h)
            ),
            None,
        )

        fixed_plus3_total = (
            None
            if (
                fixed_plus3_row is None
                or fixed_plus3_row.get(
                    "exact_total_h_plus_k2"
                )
                is None
            )
            else int(
                fixed_plus3_row["exact_total_h_plus_k2"]
            )
        )

        best_exact_total = (
            None
            if exact_best is None
            else int(exact_best["exact_total_h_plus_k2"])
        )

        allocation_gain_vs_plus3 = (
            None
            if (
                fixed_plus3_total is None
                or best_exact_total is None
            )
            else int(
                fixed_plus3_total
                - best_exact_total
            )
        )

        proxy_matches_exact = bool(
            proxy_best is not None
            and exact_best is not None
            and int(proxy_best["h_length"])
            == int(exact_best["h_length"])
        )

        row = {
            "case": int(case_index),
            "scramble": list(moves),
            "root_lb": int(root_lb),
            "h_rows": h_rows,
            "case_bad_status": case_bad,
            "complete": bool(case_complete),
            "proxy_best_h": (
                None
                if proxy_best is None
                else int(proxy_best["h_length"])
            ),
            "proxy_best_total": (
                None
                if proxy_best is None
                else int(
                    proxy_best["proxy_total_h_plus_min_c"]
                )
            ),
            "exact_best_h": (
                None
                if exact_best is None
                else int(exact_best["h_length"])
            ),
            "exact_best_k2": (
                None
                if exact_best is None
                else int(exact_best["exact_best_k2"])
            ),
            "exact_best_total": best_exact_total,
            "fixed_plus3_h": int(fixed_plus3_h),
            "fixed_plus3_exact_total": fixed_plus3_total,
            "allocation_gain_vs_plus3": allocation_gain_vs_plus3,
            "proxy_matches_exact_h": bool(proxy_matches_exact),
            "case_wall": float(
                time.perf_counter()
                - case_start
            ),
        }

        case_rows.append(row)

        print(
            "  proxy best H       :",
            row["proxy_best_h"],
            "proxyTotal=",
            row["proxy_best_total"],
        )
        print(
            "  exact best H       :",
            row["exact_best_h"],
            "K2=",
            row["exact_best_k2"],
            "total=",
            row["exact_best_total"],
        )
        print(
            "  fixed +3 total     :",
            row["fixed_plus3_exact_total"],
        )
        print(
            "  allocation gain    :",
            row["allocation_gain_vs_plus3"],
        )
        print(
            "  proxy matches exact:",
            row["proxy_matches_exact_h"],
        )
        print(
            "  complete           :",
            row["complete"],
        )

    complete_rows = [
        row
        for row in case_rows
        if row.get("complete", False)
    ]

    improved_vs_plus3 = [
        row
        for row in complete_rows
        if (
            row["allocation_gain_vs_plus3"] is not None
            and int(row["allocation_gain_vs_plus3"]) > 0
        )
    ]

    proxy_match_rows = [
        row
        for row in complete_rows
        if row["proxy_matches_exact_h"]
    ]

    exact_best_h_hist = {}

    for row in complete_rows:
        h = str(row["exact_best_h"])
        exact_best_h_hist[h] = (
            exact_best_h_hist.get(h, 0) + 1
        )

    gains = [
        int(row["allocation_gain_vs_plus3"])
        for row in complete_rows
        if row["allocation_gain_vs_plus3"] is not None
    ]

    mean_gain = (
        None
        if not gains
        else sum(gains) / len(gains)
    )

    print()
    print("# AGGREGATE")
    print(
        "complete cases       :",
        f"{len(complete_rows)}/{len(case_rows)}",
    )
    print("exact best H hist    :", exact_best_h_hist)
    print(
        "improves fixed +3    :",
        f"{len(improved_vs_plus3)}/{len(complete_rows)}",
    )
    print(
        "proxy H matches exact:",
        f"{len(proxy_match_rows)}/{len(complete_rows)}",
    )
    print(
        "mean gain vs +3      :",
        (
            "n/a"
            if mean_gain is None
            else f"{mean_gain:.2f} moves"
        ),
    )

    if len(complete_rows) < int(local.fresh_count):
        decision = "RESIDUAL_AWARE_H_ALLOCATION_INCONCLUSIVE"
        note = (
            "At least one fresh case did not complete exact certification for "
            "every requested H allocation. Increase only the binding cap for "
            "that case/H."
        )

    elif len(improved_vs_plus3) >= 2:
        decision = "RESIDUAL_AWARE_H_ALLOCATION_EXACT_SIGNAL"
        note = (
            "Across the fresh hard controls, branch-and-bound-certified exact "
            "H+K2 totals show that a fixed rootLB+3 allocation is suboptimal in "
            "at least two cases. H allocation itself is now a real optimization "
            "dimension, not merely endpoint selection."
        )

    elif len(proxy_match_rows) == len(complete_rows):
        decision = "H_PLUS_MINC_PROXY_MATCHES_EXACT_ALLOCATION"
        note = (
            "The cheap allocation proxy H+minC selects the same H as the exact "
            "branch-and-bound-certified H+K2 optimum on every completed fresh "
            "case. Validate on additional cases before using it to order H "
            "allocations."
        )

    else:
        decision = "H_ALLOCATION_FIXED_PLUS3_NOT_DISPROVEN"
        note = (
            "The tested fresh controls do not provide enough exact evidence "
            "that residual-aware H allocation improves over the existing +3 "
            "policy. Keep endpoint residual search, but do not add allocation "
            "logic yet."
        )

    print()
    print("# DECISION")
    print(decision)
    print(note)

    print()
    print("NEXT:")

    if decision == "RESIDUAL_AWARE_H_ALLOCATION_EXACT_SIGNAL":
        print(
            "  v37.141 learn/validate cheap H-allocation ordering from H+minC"
        )
        print(
            "  exact branch-and-bound remains the reference; no production hook yet"
        )

    elif decision == "H_PLUS_MINC_PROXY_MATCHES_EXACT_ALLOCATION":
        print(
            "  validate H+minC allocation ordering on 5 additional fresh hard scrambles"
        )

    elif decision == "RESIDUAL_AWARE_H_ALLOCATION_INCONCLUSIVE":
        print(
            "  rerun only the incomplete case/H with larger exact/time caps"
        )

    else:
        print(
            "  retain rootLB+3 for now; focus on exact post-DR branch-and-bound efficiency"
        )

    payload = {
        "version": "v37.140",
        "mode": "RESIDUAL_AWARE_H_ALLOCATION_FRONTIER",
        "stable_return_code": int(rc),
        "config": {
            "fresh_count": int(local.fresh_count),
            "fresh_seed": int(local.fresh_seed),
            "scramble_length": int(local.scramble_length),
            "h_offsets": list(local.h_offsets),
            "initial_r4_cap": int(local.initial_r4_cap),
            "prefix_node_cap": int(local.prefix_node_cap),
            "case_time_limit": float(local.case_time_limit),
            "k2_probe_cap": int(local.k2_probe_cap),
            "exact_call_cap_per_h": int(
                local.exact_call_cap_per_h
            ),
        },
        "cases": case_rows,
        "aggregate": {
            "complete_cases": len(complete_rows),
            "total_cases": len(case_rows),
            "exact_best_h_histogram": exact_best_h_hist,
            "improves_fixed_plus3": len(improved_vs_plus3),
            "proxy_matches_exact_h": len(proxy_match_rows),
            "mean_gain_vs_plus3": mean_gain,
        },
        "decision": decision,
        "note": note,
        "production_changes": "NONE",
    }

    out = Path(local.analysis_output)
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

    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
