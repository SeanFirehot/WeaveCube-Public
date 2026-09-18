#!/usr/bin/env python3
"""
CubeLab v37.137 — FRESH MIN-C / P2+1 SECONDARY-GATE VALIDATION

Grounding
---------
v37.136 used four completed hard controls and found:

    min-P2 gate:
        exact-best preserved 2/4 only -> REJECT

    P2LB <= minP2 + 1 inside the grounded min-C tie class:
        exact-best preserved 4/4
        retained 31/47 endpoints
        additional compression 34.0%

This is NOT yet a sound production prune.

v37.137 validates the exact same secondary rule on FIVE NEW deterministic hard
scrambles, with no cherry-picking.

Per fresh case
--------------
1. Generate reduced random 25-move scramble from fixed seed.
2. Set H = root phase1 LB + configured offset (default +3).
3. Enumerate a COMPLETE direct-q H portfolio.
   - auto-expand R4 cap if exact R4 word count exceeds current cap.
4. Build the grounded primary gate:
       C = combined terminal K2 LB
       keep FULL min-C tie class
5. Inside min-C, build secondary candidate set:
       P2LB <= minP2 + 1
6. Exact-K2 probe the ENTIRE min-C class.
7. Check whether the secondary subset contains at least one endpoint attaining
   the exact-best K2 of the full min-C class.
8. Replay every exact-best candidate used for the contract.

Important
---------
The full min-C class is always retained as the audit fallback/reference.
No production pruning is enabled.

PASS signal
-----------
A strong generalization signal requires:
    * all 5 cases complete;
    * P2+1 preserves a full-min-C exact-best endpoint in all 5;
    * no secondary exact loss;
    * mean additional compression >= 20%.

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

import audit_fresh_hard_residual_gate_v37_133 as v133
import audit_hard_global_k2_residual_v37_131_1 as v1311
import audit_post_dr_k2_deficit_v37_128 as v128
import audit_fast_stable_selective_terminal_macro_v37_114 as v114


CONTROL_TOKENS = tuple(
    "D' R F2 U D' R L B D R' U D R' D F' B U F' L' D2 B L D' F R'".split()
)


def _auto_complete_portfolio(
    *,
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

    r4_cap = int(initial_r4_cap)
    passes = []

    while True:
        remaining = deadline - time.perf_counter()

        if remaining <= 0:
            return None, r4_cap, passes, {
                "time_exhausted_before_pass": True,
            }

        portfolio = v133._enumerate_direct_h_complete(
            solver=solver,
            oracle=oracle,
            poses=tuple(poses),
            q=int(q),
            h_length=int(h_length),
            r4_suffix_cap=int(r4_cap),
            prefix_node_cap=int(prefix_node_cap),
            time_limit=float(remaining),
        )

        p = {
            "r4_cap": int(r4_cap),
            "prefix_nodes": int(portfolio["stats"]["prefix_nodes"]),
            "r4_leaf_states": int(portfolio["stats"]["r4_leaf_states"]),
            "max_r4_words": int(portfolio["stats"]["max_r4_words"]),
            "r4_cap_truncated": int(portfolio["stats"]["r4_cap_truncated"]),
            "node_cap_cut": bool(portfolio["cuts"]["node_cap"]),
            "time_cut": bool(portfolio["cuts"]["time"]),
            "elapsed": float(portfolio["elapsed"]),
        }

        passes.append(p)

        if portfolio["cuts"]["node_cap"] or portfolio["cuts"]["time"]:
            return portfolio, r4_cap, passes, {
                "time_exhausted_before_pass": False,
            }

        truncated = int(portfolio["stats"]["r4_cap_truncated"])
        max_words = int(portfolio["stats"]["max_r4_words"])

        if truncated > 0 and max_words > int(r4_cap):
            r4_cap = int(max_words)
            continue

        return portfolio, r4_cap, passes, {
            "time_exhausted_before_pass": False,
        }


def _first_skeletons(portfolio):
    out = {}

    for skeleton in portfolio["skeletons"]:
        key = skeleton["endpoint_key"]

        if key not in out:
            out[key] = skeleton

    return out


def _probe_all_min_c(
    *,
    solver,
    scrambled_poses,
    min_c_items,
    canonical_item,
    probe_cap,
    call_cap,
    deadline,
):
    v128._BASE_FIND_NO_SHADOW = v1311._BASE_FIND_NO_SHADOW

    calls = {
        "calls": 0,
        "cap": int(call_cap),
        "cache": {},
    }

    all_items = list(min_c_items)

    # Probe canonical too for telemetry, but the scientific preservation
    # contract compares secondary vs full min-C.
    if canonical_item[0] not in {
        key
        for key, _endpoint in all_items
    }:
        probe_order = [canonical_item, *all_items]
    else:
        probe_order = list(all_items)

    results = {}

    for key, endpoint in probe_order:
        if (
            time.perf_counter() >= deadline
            or int(calls["calls"]) >= int(calls["cap"])
        ):
            break

        probe = v133._probe_endpoint(
            solver=solver,
            endpoint=endpoint,
            call_state=calls,
            probe_cap=int(probe_cap),
            deadline=deadline,
        )

        results[key] = probe

    canonical_probe = results.get(
        canonical_item[0]
    )

    rows = []

    for gate_rank, (key, endpoint) in enumerate(
        all_items,
        start=1,
    ):
        probe = results.get(key)

        if probe is None:
            rows.append(
                {
                    "gate_rank": int(gate_rank),
                    "endpoint_key_repr": repr(key),
                    "combined_lb": int(endpoint["combined_lb"]),
                    "p2lb": int(endpoint["p2lb"]),
                    "exact_status": "NOT_PROBED",
                    "exact_k2": None,
                }
            )

            continue

        rows.append(
            {
                "gate_rank": int(gate_rank),
                "endpoint_key_repr": repr(key),
                "combined_lb": int(endpoint["combined_lb"]),
                "p2lb": int(endpoint["p2lb"]),
                "exact_status": str(probe["status"]),
                "exact_k2": (
                    None
                    if probe.get("min_k2") is None
                    else int(probe["min_k2"])
                ),
                "k2_word": list(probe.get("word", ())),
            }
        )

    return {
        "calls": int(calls["calls"]),
        "canonical_probe": canonical_probe,
        "rows": rows,
        "coverage_complete": bool(
            len(results)
            >= len(
                {
                    key
                    for key, _endpoint in probe_order
                }
            )
        ),
    }


def _verify_exact_row(
    *,
    scrambled_poses,
    skeleton,
    row,
):
    if (
        row.get("exact_k2") is None
        or row.get("exact_status") != "FOUND"
        or skeleton is None
    ):
        return False

    return v133._verify_solution(
        scrambled_poses=tuple(scrambled_poses),
        skeleton=skeleton,
        k2_word=tuple(row.get("k2_word", ())),
    )


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
    )

    ap.add_argument(
        "--fresh-count",
        type=int,
        default=5,
    )

    ap.add_argument(
        "--fresh-seed",
        type=int,
        default=20261045,
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
        default=90.0,
    )

    ap.add_argument(
        "--exact-k2-probe-cap",
        type=int,
        default=18,
    )

    ap.add_argument(
        "--exact-k2-call-cap-per-case",
        type=int,
        default=5000,
    )

    ap.add_argument(
        "--analysis-output",
        default=(
            "reports/v37/"
            "fresh_min_c_p2plus1_validation_v37_137.json"
        ),
    )

    ns, remaining = ap.parse_known_args(argv)

    ns.fresh_count = max(1, int(ns.fresh_count))
    ns.scramble_length = max(1, int(ns.scramble_length))
    ns.h_offset = max(0, int(ns.h_offset))
    ns.initial_r4_cap = max(1, int(ns.initial_r4_cap))
    ns.prefix_node_cap = max(1, int(ns.prefix_node_cap))
    ns.case_time_limit = max(1.0, float(ns.case_time_limit))
    ns.exact_k2_probe_cap = max(0, int(ns.exact_k2_probe_cap))
    ns.exact_k2_call_cap_per_case = max(
        1,
        int(ns.exact_k2_call_cap_per_case),
    )

    return ns, remaining


def main():
    local, remaining = _parse_args(
        sys.argv[1:]
    )

    v114._combined_installer = (
        v1311._v37131_combined_installer
    )

    sys.argv = [
        sys.argv[0],
        *remaining,
    ]

    print(
        "# CubeLab v37.137 - "
        "FRESH MIN-C / P2+1 SECONDARY-GATE VALIDATION"
    )

    print(
        "fresh cases          :",
        int(local.fresh_count),
    )

    print(
        "fresh seed           :",
        int(local.fresh_seed),
    )

    print(
        "scramble length      :",
        int(local.scramble_length),
    )

    print(
        "H policy             :",
        f"root LB + {local.h_offset}",
    )

    print(
        "primary gate         :",
        "full min combined-K2-LB tie class",
    )

    print(
        "secondary gate       :",
        "P2LB <= minP2 + 1 within min-C",
    )

    print(
        "full min-C fallback  :",
        "ALWAYS retained in audit",
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
        int(local.fresh_seed)
    )

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
    print(
        "# FRESH VALIDATION CASES"
    )

    rows = []

    for case_index, moves in enumerate(
        scrambles,
        start=1,
    ):
        case_start = time.perf_counter()
        case_deadline = (
            case_start
            + float(local.case_time_limit)
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

        print()
        print(
            f"[CASE {case_index}/{local.fresh_count}] "
            f"rootLB={root_lb} H={h_length}"
        )

        print(
            "scramble             :",
            " ".join(moves),
        )

        remaining_for_enum = (
            case_deadline
            - time.perf_counter()
        )

        portfolio, r4_cap, enum_passes, enum_extra = (
            _auto_complete_portfolio(
                solver=solver,
                oracle=oracle,
                poses=tuple(poses),
                q=int(q),
                h_length=int(h_length),
                initial_r4_cap=int(local.initial_r4_cap),
                prefix_node_cap=int(local.prefix_node_cap),
                time_limit=float(
                    max(
                        0.0,
                        remaining_for_enum,
                    )
                ),
            )
        )

        if portfolio is None:
            row = {
                "case": int(case_index),
                "scramble": list(moves),
                "root_lb": int(root_lb),
                "h_length": int(h_length),
                "complete": False,
                "outcome": "INCONCLUSIVE",
                "reason": "enumeration_missing",
                "enum_passes": enum_passes,
                "enum_extra": enum_extra,
            }

            rows.append(row)

            print(
                "OUTCOME              :",
                "INCONCLUSIVE enumeration_missing",
            )

            continue

        complete_enum = bool(
            not portfolio["cuts"]["node_cap"]
            and not portfolio["cuts"]["time"]
            and int(
                portfolio["stats"]["r4_cap_truncated"]
            )
            == 0
        )

        raw_items = list(
            portfolio["endpoints"].items()
        )

        if not raw_items:
            row = {
                "case": int(case_index),
                "scramble": list(moves),
                "root_lb": int(root_lb),
                "h_length": int(h_length),
                "complete": False,
                "outcome": "INCONCLUSIVE",
                "reason": "no_dr_endpoints",
                "enum_passes": enum_passes,
            }

            rows.append(row)

            print(
                "OUTCOME              :",
                "INCONCLUSIVE no_dr_endpoints",
            )

            continue

        canonical_item = raw_items[0]

        min_c = min(
            int(endpoint["combined_lb"])
            for _key, endpoint in raw_items
            if endpoint["combined_lb"] is not None
        )

        min_c_items = [
            (key, endpoint)
            for key, endpoint in raw_items
            if (
                endpoint["combined_lb"] is not None
                and int(endpoint["combined_lb"])
                == int(min_c)
            )
        ]

        min_p2 = min(
            int(endpoint["p2lb"])
            for _key, endpoint in min_c_items
        )

        secondary_items = [
            (key, endpoint)
            for key, endpoint in min_c_items
            if int(endpoint["p2lb"])
            <= int(min_p2) + 1
        ]

        first_skeleton = _first_skeletons(
            portfolio
        )

        remaining_for_exact = (
            case_deadline
            - time.perf_counter()
        )

        if remaining_for_exact <= 0:
            row = {
                "case": int(case_index),
                "scramble": list(moves),
                "root_lb": int(root_lb),
                "h_length": int(h_length),
                "complete": False,
                "outcome": "INCONCLUSIVE",
                "reason": "no_time_for_exact",
                "enum_passes": enum_passes,
            }

            rows.append(row)

            print(
                "OUTCOME              :",
                "INCONCLUSIVE no_time_for_exact",
            )

            continue

        exact = _probe_all_min_c(
            solver=solver,
            scrambled_poses=tuple(poses),
            min_c_items=min_c_items,
            canonical_item=canonical_item,
            probe_cap=int(local.exact_k2_probe_cap),
            call_cap=int(
                local.exact_k2_call_cap_per_case
            ),
            deadline=case_deadline,
        )

        key_by_repr = {
            repr(key): key
            for key, _endpoint in min_c_items
        }

        secondary_key_set = {
            key
            for key, _endpoint in secondary_items
        }

        full_found = []

        for row_exact in exact["rows"]:
            if (
                row_exact["exact_status"] == "FOUND"
                and row_exact["exact_k2"] is not None
            ):
                key = key_by_repr[
                    row_exact["endpoint_key_repr"]
                ]

                skeleton = first_skeleton.get(key)

                solved = _verify_exact_row(
                    scrambled_poses=tuple(poses),
                    skeleton=skeleton,
                    row=row_exact,
                )

                row_exact["solved_replay"] = bool(
                    solved
                )

                if solved:
                    full_found.append(
                        (
                            key,
                            row_exact,
                        )
                    )

            elif (
                row_exact["exact_status"]
                == "ABOVE_CAP"
            ):
                row_exact["solved_replay"] = None

            else:
                row_exact["solved_replay"] = False

        bad_statuses = {
            row_exact["exact_status"]
            for row_exact in exact["rows"]
            if row_exact["exact_status"]
            not in (
                "FOUND",
                "ABOVE_CAP",
            )
        }

        if full_found:
            full_best_k2 = min(
                int(row_exact["exact_k2"])
                for _key, row_exact in full_found
            )

            full_best_keys = {
                key
                for key, row_exact in full_found
                if int(row_exact["exact_k2"])
                == int(full_best_k2)
            }

        else:
            full_best_k2 = None
            full_best_keys = set()

        secondary_found = [
            (key, row_exact)
            for key, row_exact in full_found
            if key in secondary_key_set
        ]

        secondary_best_k2 = (
            None
            if not secondary_found
            else min(
                int(row_exact["exact_k2"])
                for _key, row_exact in secondary_found
            )
        )

        preserves_exact_best = bool(
            full_best_keys
            and (
                full_best_keys
                & secondary_key_set
            )
        )

        secondary_loss = (
            None
            if (
                full_best_k2 is None
                or secondary_best_k2 is None
            )
            else int(
                secondary_best_k2
                - full_best_k2
            )
        )

        secondary_compression = (
            0.0
            if not min_c_items
            else (
                1.0
                - (
                    float(len(secondary_items))
                    / float(len(min_c_items))
                )
            )
        )

        canonical_probe = exact.get(
            "canonical_probe"
        )

        canonical_exact = (
            None
            if (
                canonical_probe is None
                or canonical_probe.get("min_k2") is None
            )
            else int(canonical_probe["min_k2"])
        )

        full_min_c_vs_canonical = (
            None
            if (
                canonical_exact is None
                or full_best_k2 is None
            )
            else int(
                canonical_exact
                - full_best_k2
            )
        )

        exact_complete = bool(
            exact["coverage_complete"]
            and not bad_statuses
            and len(exact["rows"])
            == len(min_c_items)
        )

        case_complete = bool(
            complete_enum
            and exact_complete
            and full_best_k2 is not None
        )

        if case_complete:
            if preserves_exact_best:
                outcome = "PRESERVE"
            else:
                outcome = "LOSS"
        else:
            outcome = "INCONCLUSIVE"

        row = {
            "case": int(case_index),
            "scramble": list(moves),
            "root_lb": int(root_lb),
            "h_length": int(h_length),
            "enum_passes": enum_passes,
            "final_r4_cap": int(r4_cap),
            "raw_endpoints": len(raw_items),
            "raw_skeletons": len(portfolio["skeletons"]),
            "min_combined_lb": int(min_c),
            "min_c_endpoints": len(min_c_items),
            "min_p2lb_within_min_c": int(min_p2),
            "secondary_endpoints": len(secondary_items),
            "secondary_compression": float(
                secondary_compression
            ),
            "canonical_exact_k2": canonical_exact,
            "full_min_c_best_exact_k2": full_best_k2,
            "secondary_best_exact_k2": secondary_best_k2,
            "full_min_c_vs_canonical_improvement": (
                full_min_c_vs_canonical
            ),
            "secondary_loss_vs_full_min_c": (
                secondary_loss
            ),
            "preserves_exact_best": bool(
                preserves_exact_best
            ),
            "full_best_endpoint_count": len(
                full_best_keys
            ),
            "full_best_secondary_intersection": len(
                full_best_keys
                & secondary_key_set
            ),
            "exact_calls": int(exact["calls"]),
            "bad_exact_statuses": sorted(
                bad_statuses
            ),
            "complete_enumeration": bool(
                complete_enum
            ),
            "exact_complete": bool(
                exact_complete
            ),
            "complete": bool(
                case_complete
            ),
            "outcome": outcome,
            "case_wall": float(
                time.perf_counter()
                - case_start
            ),
        }

        rows.append(row)

        print(
            "raw/minC/sec ep     :",
            f"{len(raw_items)} / "
            f"{len(min_c_items)} / "
            f"{len(secondary_items)}",
        )

        print(
            "Cmin / P2min        :",
            f"{min_c} / {min_p2}",
        )

        print(
            "secondary compression:",
            f"{secondary_compression*100:.1f}%",
        )

        print(
            "canonical/full/sec K2:",
            f"{canonical_exact} / "
            f"{full_best_k2} / "
            f"{secondary_best_k2}",
        )

        print(
            "preserve exact best :",
            preserves_exact_best,
        )

        print(
            "secondary loss      :",
            secondary_loss,
        )

        print(
            "exact calls         :",
            exact["calls"],
        )

        print(
            "complete/outcome    :",
            f"{case_complete} / {outcome}",
        )

    complete_rows = [
        row
        for row in rows
        if row.get("complete", False)
    ]

    preserve_rows = [
        row
        for row in complete_rows
        if row["outcome"] == "PRESERVE"
    ]

    loss_rows = [
        row
        for row in complete_rows
        if row["outcome"] == "LOSS"
    ]

    inconclusive_rows = [
        row
        for row in rows
        if row["outcome"] == "INCONCLUSIVE"
    ]

    compressions = [
        float(row["secondary_compression"])
        for row in complete_rows
    ]

    mean_compression = (
        None
        if not compressions
        else sum(compressions) / len(compressions)
    )

    min_compression = (
        None
        if not compressions
        else min(compressions)
    )

    strict_primary_wins = [
        row
        for row in complete_rows
        if (
            row["full_min_c_vs_canonical_improvement"]
            is not None
            and int(
                row["full_min_c_vs_canonical_improvement"]
            )
            > 0
        )
    ]

    print()
    print(
        "# AGGREGATE"
    )

    print(
        "complete             :",
        f"{len(complete_rows)}/{len(rows)}",
    )

    print(
        "secondary preserve   :",
        f"{len(preserve_rows)}/{len(rows)}",
    )

    print(
        "secondary losses     :",
        len(loss_rows),
    )

    print(
        "inconclusive         :",
        len(inconclusive_rows),
    )

    print(
        "primary min-C strict wins:",
        f"{len(strict_primary_wins)}/{len(complete_rows)}",
    )

    print(
        "mean secondary compression:",
        (
            "n/a"
            if mean_compression is None
            else f"{mean_compression*100:.1f}%"
        ),
    )

    print(
        "min case compression :",
        (
            "n/a"
            if min_compression is None
            else f"{min_compression*100:.1f}%"
        ),
    )

    if (
        len(complete_rows) == int(local.fresh_count)
        and len(preserve_rows) == int(local.fresh_count)
        and not loss_rows
        and mean_compression is not None
        and mean_compression >= 0.20
    ):
        decision = (
            "P2_PLUS1_SECONDARY_GATE_FRESH_GENERALIZATION_SIGNAL"
        )

        note = (
            "On all five new deterministic hard scrambles, the P2LB<=minP2+1 "
            "subset inside the grounded full min-C tie class preserves at least "
            "one endpoint attaining the full min-C exact-best K2, with no exact "
            "loss and at least 20% mean additional candidate compression. This "
            "supports a two-stage post-DR portfolio policy, but the full min-C "
            "fallback should remain available until a larger validation set."
        )

    elif loss_rows:
        decision = (
            "P2_PLUS1_SECONDARY_GATE_FRESH_LOSS"
        )

        note = (
            "At least one complete fresh hard scramble loses the full min-C "
            "exact-best endpoint under P2LB<=minP2+1. Reject the secondary gate "
            "as a standalone prune and keep the full min-C tie class."
        )

    elif len(complete_rows) == int(local.fresh_count):
        decision = (
            "P2_PLUS1_SECONDARY_GATE_PRESERVES_BUT_LOW_COMPRESSION"
        )

        note = (
            "The secondary subset preserves all fresh exact winners but does "
            "not provide enough additional compression to justify another gate "
            "layer. Keep the full min-C tie class as the practical mechanism."
        )

    else:
        decision = (
            "P2_PLUS1_SECONDARY_GATE_FRESH_INCONCLUSIVE"
        )

        note = (
            "One or more fresh cases did not complete the full-min-C exact "
            "reference under the explicit resource caps. Increase only the "
            "binding cap for incomplete cases."
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
        "P2_PLUS1_SECONDARY_GATE_FRESH_GENERALIZATION_SIGNAL"
    ):
        print(
            "  v37.138 measure two-stage post-DR portfolio runtime / exact-call savings"
        )

        print(
            "  policy: min-C full fallback, try P2<=minP2+1 first"
        )

        print(
            "  do not push residual propagation earlier than DR; v37.135 closed that runtime path"
        )

    elif decision == (
        "P2_PLUS1_SECONDARY_GATE_FRESH_LOSS"
    ):
        print(
            "  close secondary P2 gate and retain full min-C tie class only"
        )

    elif decision == (
        "P2_PLUS1_SECONDARY_GATE_FRESH_INCONCLUSIVE"
    ):
        print(
            "  rerun only incomplete cases with the binding cap increased"
        )

    else:
        print(
            "  stop secondary-gate tuning; retain full min-C tie class"
        )

    payload = {
        "version": "v37.137",
        "mode": "FRESH_MIN_C_P2PLUS1_SECONDARY_GATE_VALIDATION",
        "stable_return_code": int(rc),
        "config": {
            "fresh_count": int(local.fresh_count),
            "fresh_seed": int(local.fresh_seed),
            "scramble_length": int(local.scramble_length),
            "h_offset": int(local.h_offset),
            "initial_r4_cap": int(local.initial_r4_cap),
            "prefix_node_cap": int(local.prefix_node_cap),
            "case_time_limit": float(local.case_time_limit),
            "exact_k2_probe_cap": int(
                local.exact_k2_probe_cap
            ),
            "exact_k2_call_cap_per_case": int(
                local.exact_k2_call_cap_per_case
            ),
        },
        "cases": rows,
        "aggregate": {
            "complete": len(complete_rows),
            "preserve": len(preserve_rows),
            "losses": len(loss_rows),
            "inconclusive": len(inconclusive_rows),
            "primary_min_c_strict_wins": len(
                strict_primary_wins
            ),
            "mean_secondary_compression": (
                mean_compression
            ),
            "min_case_secondary_compression": (
                min_compression
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

    return int(rc)


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
