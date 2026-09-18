#!/usr/bin/env python3
"""
CubeLab v37.141 — POST-DR EXACT CERTIFICATION SEARCH A/B

Grounded direction after v37.140
--------------------------------
* rootLB+3 remained exact-best H on 3/3 fresh hard controls.
* H+minC allocation proxy matched only 2/3.
* fixed residual bands are not quality-complete:
      one new H10 control had minC=8 but exact winner C=11.

So the practical quality-complete mechanism is:

    complete rootLB+3 H portfolio
    -> raw DR endpoints
    -> admissible combined-LB exact certification

v37.141 optimizes ONLY that exact certification stage.

Reference
---------
The existing v37.140 JSON supplies, for the same 3 scrambles at H=rootLB+3:

    source global exact-best K2
    source ZERO-START branch-and-bound budget-call count

Arms
----
1) LBSTART_BNB

   Same endpoint order as current branch-and-bound, but never exact-probe a
   budget below the endpoint's admissible combined lower bound C.

   Initial witness:
       budgets C .. probe_cap

   Challenger:
       budgets C .. incumbent-1

2) C_LAYERED

   Global iterative deepening over K2 budget k:

       for k = minC .. probe_cap:
           test every endpoint with C <= k once at budget k
           first hit at the first successful k certifies global exact K2 = k

   Why exact:
       * all smaller global budgets were exhausted;
       * endpoints with C > k cannot solve in <= k by admissibility.

Scientific contracts
--------------------
* Re-enumerated H+3 raw endpoint count must match v37.140.
* LBSTART_BNB exact-best K2 must equal v37.140 source exact-best.
* C_LAYERED exact-best K2 must equal v37.140 source exact-best.
* returned witness must replay solved.
* no H enumeration truncation.

Primary performance metric:
    terminal budget-call count

Wall times are reported but same-process cache effects make call count the more
reliable A/B metric.

OFFLINE only.
Production changes: NONE.
"""

from __future__ import annotations

import argparse
import json
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
import audit_fast_stable_selective_terminal_macro_v37_114 as v114


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
    )

    ap.add_argument(
        "--source-v140",
        default="reports/v37/residual_aware_h_allocation_v37_140.json",
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
        default=120.0,
    )

    ap.add_argument(
        "--k2-probe-cap",
        type=int,
        default=22,
    )

    ap.add_argument(
        "--terminal-call-cap-per-arm",
        type=int,
        default=12000,
    )

    ap.add_argument(
        "--analysis-output",
        default="reports/v37/post_dr_exact_cert_ab_v37_141.json",
    )

    ns, remaining = ap.parse_known_args(argv)

    ns.initial_r4_cap = max(1, int(ns.initial_r4_cap))
    ns.prefix_node_cap = max(1, int(ns.prefix_node_cap))
    ns.case_time_limit = max(1.0, float(ns.case_time_limit))
    ns.k2_probe_cap = max(0, int(ns.k2_probe_cap))
    ns.terminal_call_cap_per_arm = max(
        1,
        int(ns.terminal_call_cap_per_arm),
    )

    return ns, remaining


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
    """
    Exactly one H=0/K2 terminal call at one total K2 budget.

    Exceptions are UNKNOWN/CUT, never UNSAT.
    """
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
            "result": None,
            "error": repr(exc),
        }

    if result is None:
        return {
            "status": "MISS",
            "result": None,
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
            "result": result,
        }

    if total > int(budget):
        return {
            "status": "CONTRACT_OVER_BUDGET",
            "result": result,
        }

    return {
        "status": "FOUND",
        "result": result,
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


def _sorted_endpoints(portfolio):
    return sorted(
        portfolio["endpoints"].items(),
        key=lambda item: (
            int(item[1]["combined_lb"]),
            int(item[1]["p2lb"]),
        ),
    )


def _lbstart_probe_range(
    *,
    solver,
    endpoint,
    start_budget,
    end_budget,
    counter,
    deadline,
):
    for budget in range(
        int(start_budget),
        int(end_budget) + 1,
    ):
        if (
            time.perf_counter() >= deadline
            or int(counter["calls"]) >= int(counter["cap"])
        ):
            return {
                "status": "RESOURCE_CUT",
            }

        counter["calls"] += 1

        probe = _one_budget_find(
            solver=solver,
            endpoint=endpoint,
            budget=int(budget),
        )

        status = str(probe["status"])

        if status == "MISS":
            continue

        return probe

    return {
        "status": "ABOVE_RANGE",
    }


def _run_lbstart_bnb(
    *,
    solver,
    scrambled_poses,
    portfolio,
    probe_cap,
    call_cap,
    deadline,
):
    items = _sorted_endpoints(portfolio)
    first_skeleton = _first_skeletons(portfolio)

    calls = {
        "calls": 0,
        "cap": int(call_cap),
    }

    incumbent = None
    winner_c = None
    winner_key = None
    witness = None
    bad = None
    bound_skipped = 0
    endpoint_probes = 0
    improvements = []
    first_witness_index = None

    t0 = time.perf_counter()

    # Initial exact witness.
    for idx, (key, endpoint) in enumerate(items):
        c = int(endpoint["combined_lb"])

        if c > int(probe_cap):
            # All later endpoints have >= C and cannot resolve within cap.
            break

        endpoint_probes += 1

        probe = _lbstart_probe_range(
            solver=solver,
            endpoint=endpoint,
            start_budget=int(c),
            end_budget=int(probe_cap),
            counter=calls,
            deadline=deadline,
        )

        status = str(probe["status"])

        if status == "ABOVE_RANGE":
            continue

        if status != "FOUND":
            bad = status
            break

        solved = _verify(
            scrambled_poses=tuple(scrambled_poses),
            skeleton=first_skeleton.get(key),
            probe=probe,
        )

        if not solved:
            bad = "REPLAY_FAIL_INITIAL"
            break

        incumbent = int(probe["exact_k2"])
        winner_c = int(c)
        winner_key = key
        witness = probe
        first_witness_index = int(idx)

        improvements.append(
            {
                "phase": "initial",
                "combined_lb": int(c),
                "new_incumbent": int(incumbent),
            }
        )
        break

    if bad is None and incumbent is None:
        bad = "NO_WITNESS_WITHIN_CAP"

    # Certify challengers after the first witness.
    if bad is None:
        start = int(first_witness_index) + 1

        for pos in range(start, len(items)):
            key, endpoint = items[pos]
            c = int(endpoint["combined_lb"])

            if (
                time.perf_counter() >= deadline
                or int(calls["calls"]) >= int(calls["cap"])
            ):
                bad = "RESOURCE_CUT_CERTIFICATION"
                break

            if c >= int(incumbent):
                bound_skipped = len(items) - int(pos)
                break

            endpoint_probes += 1

            probe = _lbstart_probe_range(
                solver=solver,
                endpoint=endpoint,
                start_budget=int(c),
                end_budget=int(incumbent) - 1,
                counter=calls,
                deadline=deadline,
            )

            status = str(probe["status"])

            if status == "ABOVE_RANGE":
                continue

            if status != "FOUND":
                bad = status
                break

            exact_k2 = int(probe["exact_k2"])

            if exact_k2 >= int(incumbent):
                continue

            solved = _verify(
                scrambled_poses=tuple(scrambled_poses),
                skeleton=first_skeleton.get(key),
                probe=probe,
            )

            if not solved:
                bad = "REPLAY_FAIL_CHALLENGER"
                break

            old = int(incumbent)
            incumbent = int(exact_k2)
            winner_c = int(c)
            winner_key = key
            witness = probe

            improvements.append(
                {
                    "phase": "challenger",
                    "combined_lb": int(c),
                    "old_incumbent": int(old),
                    "new_incumbent": int(incumbent),
                }
            )

    wall = time.perf_counter() - t0

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
        "terminal_calls": int(calls["calls"]),
        "endpoint_probes": int(endpoint_probes),
        "bound_skipped": int(bound_skipped),
        "improvements": improvements,
        "bad_status": bad,
        "wall": float(wall),
        "witness_replay": bool(
            bad is None
            and witness is not None
        ),
    }


def _run_c_layered(
    *,
    solver,
    scrambled_poses,
    portfolio,
    probe_cap,
    call_cap,
    deadline,
):
    items = _sorted_endpoints(portfolio)
    first_skeleton = _first_skeletons(portfolio)

    calls = 0
    bad = None
    winner = None
    levels_completed = 0
    eligible_tests = 0

    if not items:
        return {
            "certified": True,
            "exact_best_k2": None,
            "winner_c": None,
            "terminal_calls": 0,
            "levels_completed": 0,
            "eligible_tests": 0,
            "bad_status": None,
            "wall": 0.0,
            "witness_replay": False,
        }

    min_c = int(items[0][1]["combined_lb"])
    t0 = time.perf_counter()

    for budget in range(
        int(min_c),
        int(probe_cap) + 1,
    ):
        eligible = [
            (key, endpoint)
            for key, endpoint in items
            if int(endpoint["combined_lb"]) <= int(budget)
        ]

        level_hit = None

        for key, endpoint in eligible:
            if (
                time.perf_counter() >= deadline
                or int(calls) >= int(call_cap)
            ):
                bad = "RESOURCE_CUT"
                break

            calls += 1
            eligible_tests += 1

            probe = _one_budget_find(
                solver=solver,
                endpoint=endpoint,
                budget=int(budget),
            )

            status = str(probe["status"])

            if status == "MISS":
                continue

            if status != "FOUND":
                bad = status
                break

            solved = _verify(
                scrambled_poses=tuple(scrambled_poses),
                skeleton=first_skeleton.get(key),
                probe=probe,
            )

            if not solved:
                bad = "REPLAY_FAIL"
                break

            exact_k2 = int(probe["exact_k2"])

            # By global iterative deepening + admissible C, the first successful
            # budget is the global optimum.  Defensive check:
            if exact_k2 > int(budget):
                bad = "CONTRACT_OVER_BUDGET"
                break

            level_hit = {
                "key": key,
                "probe": probe,
                "combined_lb": int(endpoint["combined_lb"]),
                "budget": int(budget),
                "exact_k2": int(exact_k2),
            }
            break

        if bad is not None:
            break

        if level_hit is not None:
            winner = level_hit
            break

        levels_completed += 1

    wall = time.perf_counter() - t0

    if bad is None and winner is None:
        bad = "NO_WITNESS_WITHIN_CAP"

    return {
        "certified": bool(bad is None),
        "exact_best_k2": (
            None
            if winner is None
            else int(winner["exact_k2"])
        ),
        "winner_c": (
            None
            if winner is None
            else int(winner["combined_lb"])
        ),
        "winner_key_repr": (
            None
            if winner is None
            else repr(winner["key"])
        ),
        "terminal_calls": int(calls),
        "levels_completed": int(levels_completed),
        "eligible_tests": int(eligible_tests),
        "bad_status": bad,
        "wall": float(wall),
        "witness_replay": bool(
            winner is not None
            and bad is None
        ),
    }


def main():
    local, remaining = _parse_args(
        sys.argv[1:]
    )

    source_path = Path(local.source_v140)

    if not source_path.is_file():
        raise SystemExit(
            f"missing source JSON: {source_path}"
        )

    source = json.loads(
        source_path.read_text(
            encoding="utf-8",
        )
    )

    source_cases = [
        row
        for row in source.get("cases", [])
        if bool(row.get("complete", False))
    ]

    if not source_cases:
        raise RuntimeError(
            "source v37.140 has no complete cases"
        )

    v114._combined_installer = (
        v1311._v37131_combined_installer
    )

    sys.argv = [
        sys.argv[0],
        *remaining,
    ]

    print(
        "# CubeLab v37.141 - "
        "POST-DR EXACT CERTIFICATION SEARCH A/B"
    )
    print("source v37.140       :", source_path)
    print("cases                :", len(source_cases))
    print("H                    :", "source rootLB+3 only")
    print("reference            :", "v37.140 ZERO-START BnB")
    print("arm 1                :", "LBSTART_BNB")
    print("arm 2                :", "C_LAYERED global iterative deepening")
    print("production changes   :", "NONE")

    print()
    print("# STABLE BOOTSTRAP / REPLAY")

    rc = v114.main()

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
            "terminal finder capture unavailable"
        )

    print()
    print("# A/B CASES")

    rows = []

    for source_row in source_cases:
        case_index = int(source_row["case"])
        moves = tuple(source_row["scramble"])
        root_lb = int(source_row["root_lb"])
        h_length = int(source_row["fixed_plus3_h"])

        source_h = next(
            row
            for row in source_row["h_rows"]
            if int(row["h_length"]) == int(h_length)
        )

        source_exact = int(source_h["exact_best_k2"])
        source_calls = int(source_h["exact_budget_calls"])
        source_raw = int(source_h["raw_endpoints"])

        case_start = time.perf_counter()
        deadline = (
            case_start
            + float(local.case_time_limit)
        )

        poses, q = v1311._scramble_state(
            solver=solver,
            moves=moves,
        )

        print()
        print(
            f"[CASE {case_index}] rootLB={root_lb} H={h_length} "
            f"sourceK2={source_exact}"
        )

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

        complete = _portfolio_complete(
            portfolio
        )

        if not complete:
            row = {
                "case": int(case_index),
                "complete": False,
                "reason": "portfolio_incomplete",
                "passes": passes,
            }

            rows.append(row)
            print("OUTCOME              : INCONCLUSIVE portfolio_incomplete")
            continue

        raw_count = len(
            portfolio["endpoints"]
        )

        identity = bool(
            raw_count == int(source_raw)
        )

        lb = _run_lbstart_bnb(
            solver=solver,
            scrambled_poses=tuple(poses),
            portfolio=portfolio,
            probe_cap=int(local.k2_probe_cap),
            call_cap=int(local.terminal_call_cap_per_arm),
            deadline=deadline,
        )

        layered = _run_c_layered(
            solver=solver,
            scrambled_poses=tuple(poses),
            portfolio=portfolio,
            probe_cap=int(local.k2_probe_cap),
            call_cap=int(local.terminal_call_cap_per_arm),
            deadline=deadline,
        )

        lb_match = bool(
            lb["certified"]
            and int(lb["exact_best_k2"]) == int(source_exact)
        )

        layered_match = bool(
            layered["certified"]
            and int(layered["exact_best_k2"]) == int(source_exact)
        )

        lb_call_ratio = (
            float(lb["terminal_calls"])
            / float(source_calls)
            if source_calls > 0
            else None
        )

        layered_call_ratio = (
            float(layered["terminal_calls"])
            / float(source_calls)
            if source_calls > 0
            else None
        )

        layered_vs_lb = (
            float(layered["terminal_calls"])
            / float(lb["terminal_calls"])
            if lb["terminal_calls"] > 0
            else None
        )

        case_complete = bool(
            identity
            and lb_match
            and layered_match
        )

        row = {
            "case": int(case_index),
            "scramble": list(moves),
            "root_lb": int(root_lb),
            "h_length": int(h_length),
            "source_zero_start": {
                "exact_best_k2": int(source_exact),
                "terminal_calls": int(source_calls),
                "raw_endpoints": int(source_raw),
            },
            "portfolio_identity": bool(identity),
            "recomputed_raw_endpoints": int(raw_count),
            "r4_cap": int(r4_cap),
            "lbstart_bnb": lb,
            "c_layered": layered,
            "comparison": {
                "lbstart_matches_source": bool(lb_match),
                "layered_matches_source": bool(layered_match),
                "lbstart_call_ratio_vs_source": lb_call_ratio,
                "layered_call_ratio_vs_source": layered_call_ratio,
                "layered_call_ratio_vs_lbstart": layered_vs_lb,
            },
            "complete": bool(case_complete),
            "case_wall": float(
                time.perf_counter()
                - case_start
            ),
        }

        rows.append(row)

        print(
            "raw identity        :",
            f"{raw_count}/{source_raw} -> {identity}",
        )
        print(
            "ZERO calls / K2     :",
            f"{source_calls} / {source_exact}",
        )
        print(
            "LBSTART calls / K2  :",
            f"{lb['terminal_calls']} / {lb['exact_best_k2']} "
            f"ratio={lb_call_ratio:.3f}",
        )
        print(
            "LAYERED calls / K2  :",
            f"{layered['terminal_calls']} / "
            f"{layered['exact_best_k2']} "
            f"ratio={layered_call_ratio:.3f}",
        )
        print(
            "LAYERED / LBSTART   :",
            f"{layered_vs_lb:.3f}",
        )
        print(
            "match source        :",
            f"LB={lb_match} LAYERED={layered_match}",
        )
        print(
            "wall LB / LAYERED   :",
            f"{lb['wall']:.4f}s / {layered['wall']:.4f}s",
        )
        print(
            "complete            :",
            case_complete,
        )

    valid = [
        row
        for row in rows
        if row.get("complete", False)
    ]

    total_source_calls = sum(
        int(row["source_zero_start"]["terminal_calls"])
        for row in valid
    )

    total_lb_calls = sum(
        int(row["lbstart_bnb"]["terminal_calls"])
        for row in valid
    )

    total_layered_calls = sum(
        int(row["c_layered"]["terminal_calls"])
        for row in valid
    )

    total_lb_ratio = (
        None
        if total_source_calls == 0
        else float(total_lb_calls) / float(total_source_calls)
    )

    total_layered_ratio = (
        None
        if total_source_calls == 0
        else float(total_layered_calls) / float(total_source_calls)
    )

    total_layered_vs_lb = (
        None
        if total_lb_calls == 0
        else float(total_layered_calls) / float(total_lb_calls)
    )

    print()
    print("# AGGREGATE")
    print(
        "valid cases          :",
        f"{len(valid)}/{len(rows)}",
    )
    print(
        "terminal calls ZERO  :",
        total_source_calls,
    )
    print(
        "terminal calls LB    :",
        total_lb_calls,
    )
    print(
        "terminal calls LAYER :",
        total_layered_calls,
    )
    print(
        "LB / ZERO            :",
        (
            "n/a"
            if total_lb_ratio is None
            else f"{total_lb_ratio:.3f}x"
        ),
    )
    print(
        "LAYER / ZERO         :",
        (
            "n/a"
            if total_layered_ratio is None
            else f"{total_layered_ratio:.3f}x"
        ),
    )
    print(
        "LAYER / LB           :",
        (
            "n/a"
            if total_layered_vs_lb is None
            else f"{total_layered_vs_lb:.3f}x"
        ),
    )

    if len(valid) != len(rows):
        decision = "POST_DR_CERTIFICATION_AB_INCONCLUSIVE"
        note = (
            "At least one case failed portfolio identity or exact-best matching. "
            "Fix only that contract before interpreting call-count savings."
        )

    elif (
        total_layered_ratio is not None
        and total_layered_ratio <= 0.50
    ):
        decision = "C_LAYERED_EXACT_CERTIFICATION_CALL_WIN"
        note = (
            "Global C-layered iterative deepening reproduces the branch-and-bound "
            "exact K2 reference on every case while cutting terminal budget calls "
            "by at least half versus the zero-start reference. This is the "
            "preferred next exact post-DR certification engine."
        )

    elif (
        total_lb_ratio is not None
        and total_lb_ratio <= 0.70
    ):
        decision = "LBSTART_EXACT_CERTIFICATION_CALL_WIN"
        note = (
            "Starting each endpoint's exact probe at its admissible combined "
            "lower bound materially reduces terminal budget calls while preserving "
            "the exact reference. Adopt this simpler optimization before more "
            "complex portfolio scheduling."
        )

    else:
        decision = "POST_DR_CERTIFICATION_CALL_SAVINGS_WEAK"
        note = (
            "Neither admissible LB-start probing nor global C-layered iterative "
            "deepening cuts enough terminal calls on the fixed fresh controls. "
            "Keep the current branch-and-bound reference and stop micro-tuning."
        )

    print()
    print("# DECISION")
    print(decision)
    print(note)

    print()
    print("NEXT:")

    if decision == "C_LAYERED_EXACT_CERTIFICATION_CALL_WIN":
        print(
            "  v37.142 validate C-layered certification on 5 new hard scrambles"
        )
        print(
            "  then compare end-to-end rootLB+3 portfolio wall against current reference"
        )

    elif decision == "LBSTART_EXACT_CERTIFICATION_CALL_WIN":
        print(
            "  v37.142 validate LB-start certification on 5 new hard scrambles"
        )

    elif decision == "POST_DR_CERTIFICATION_AB_INCONCLUSIVE":
        print(
            "  rerun only the failed case with the binding cap/contract fixed"
        )

    else:
        print(
            "  retain current exact branch-and-bound; no further certification micro-optimization"
        )

    payload = {
        "version": "v37.141",
        "mode": "POST_DR_EXACT_CERTIFICATION_SEARCH_AB",
        "source_v140": str(source_path),
        "stable_return_code": int(rc),
        "cases": rows,
        "aggregate": {
            "valid_cases": len(valid),
            "total_cases": len(rows),
            "source_zero_start_calls": int(total_source_calls),
            "lbstart_calls": int(total_lb_calls),
            "c_layered_calls": int(total_layered_calls),
            "lbstart_vs_source_ratio": total_lb_ratio,
            "c_layered_vs_source_ratio": total_layered_ratio,
            "c_layered_vs_lbstart_ratio": total_layered_vs_lb,
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
