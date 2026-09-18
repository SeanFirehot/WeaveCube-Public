#!/usr/bin/env python3
"""CubeLab v37.23 - budget-derived Global Requirement Closure audit.

Recreates the same deterministic H5/H8/H9/H10 planted cases used by v37.22.x.

Three roles:
UPPER
    planted valid H+K2 word. Only an upper bound.

FIXED-REF
    v37.16 exact total-budget with a preselected reference H horizon equal to
    the planted H length. This is comparison/audit only.

GLOBAL-REQ
    no preselected H horizon.
    Derive every possible improving H depth directly from:
        strict total budget = upper_total - 1

Require exact total parity and full replay.

The goal is to replace the 30-second Global DFBnB proof with direct
requirement propagation, not to raise search caps.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
for p in (str(SRC), str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pdcc_double_star as dstar
import pdcc_p1_pose as p1pose
import pdcc_phase2 as p2
import search_twist_skeleton as ts

from p1_orientation_domains_v37_0 import D6MoveDomainOracle
from audit_memoized_total_budget_v37_16 import MemoTotalBudgetSearch
from audit_elastic_h8_macro_frontier_v37_21 import (
    ExtendedOrientationOracle,
    build_planted_case,
)
from global_requirement_closure_v37_23 import (
    GlobalRequirementClosure,
)
from memory_telemetry_v37_22_2 import (
    current_process_rss_mb,
    format_mb,
)


def clear_p2():
    fn = getattr(p2, "clear_double_star_query_cache", None)
    if callable(fn):
        fn()


def replay(start, w):
    return (
        w is not None
        and start.apply_word(
            tuple(w["h_tail"])
            + tuple(w["k2_word"])
        ).is_solved()
    )


def upper_from_case(case):
    return {
        "total": int(case["known_total"]),
        "h_tail": tuple(case["h_word"]),
        "k2_word": tuple(case["k2_word"]),
        "k2_exact": len(case["k2_word"]),
    }


def fixed_reference(
    start,
    q0,
    h,
    upper_total,
    tables,
    d6,
    args,
):
    oracle = ExtendedOrientationOracle(
        d6,
        max_depth=max(
            int(d6.table.max_depth),
            int(h),
        ),
    )

    solver = MemoTotalBudgetSearch(
        tables,
        oracle,
        p2_order=args.p2_order,
        p2_probe=args.p2_auto_probe_nodes,
    )

    t0 = time.perf_counter()
    res = solver.exact_min_total(
        start,
        int(q0),
        None,
        tuple(range(1, int(h) + 1)),
        int(upper_total),
    )
    wall = time.perf_counter() - t0

    return res, wall, {
        "hits": oracle.hits,
        "builds": oracle.builds,
        "cache_size": len(oracle.cache),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--d6-table",
        default="reports/pdcc_cache/p1_tail_nosuffix_v36_6i13_d6.pkl",
    )
    ap.add_argument(
        "--cache",
        default="reports/pdcc_cache/twist_skeleton_tables_v1.pkl",
    )
    ap.add_argument(
        "--long-horizons",
        type=int,
        nargs="+",
        default=[9, 10],
    )
    ap.add_argument("--planted-k2-length", type=int, default=4)
    ap.add_argument("--max-attempts", type=int, default=20000)
    ap.add_argument("--p2-order", default="auto")
    ap.add_argument("--p2-auto-probe-nodes", type=int, default=128)
    ap.add_argument(
        "--p1-pose-cache",
        default="reports/pdcc_cache/p1_corner_star_tables_v1.pkl",
    )
    ap.add_argument(
        "--double-star-cache",
        default="reports/pdcc_cache/double_star_k_tables_v1.pkl",
    )
    ap.add_argument(
        "--double-star-index",
        default="reports/pdcc_cache/double_star_p2_index_v1.pkl",
    )
    ap.add_argument(
        "--support-cache-max",
        type=int,
        default=200000,
    )
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument(
        "--output",
        default="reports/v37/global_requirement_closure_v37_23.json",
    )
    args = ap.parse_args()

    tables, _loaded = ts.cache_load_or_build(Path(args.cache))
    co, eo, sl, cos, eos, *_ = tables

    d6 = D6MoveDomainOracle.load(
        Path(args.d6_table),
        co, eo, sl, cos, eos,
    )

    max_h = max(max(args.long_horizons), 8)
    construction_oracle = ExtendedOrientationOracle(
        d6,
        max_depth=max_h,
    )

    p2.configure_double_star_cache(
        args.double_star_cache,
        args.double_star_index,
    )
    dstar.configure(
        args.double_star_cache,
        args.double_star_index,
    )
    p1pose.configure(args.p1_pose_cache)
    p1pose.preload()

    rng = random.Random(int(args.seed))

    baseline_rss = current_process_rss_mb()

    print("# CubeLab v37.23 - GLOBAL REQUIREMENT CLOSURE")
    print()
    print("RNG seed             :", args.seed)
    print("outer H depth        : NOT PRESELECTED in GLOBAL-REQ")
    print("H domain source      : total budget U-1")
    print("base exact oracle    : d6")
    print(
        "support cache       :",
        f"bounded <= {args.support_cache_max:,}",
    )
    print(
        "baseline RSS        :",
        format_mb(baseline_rss),
    )
    print("global OPEN heap     : NONE")
    print("multiprocessing      : NONE")
    print("production changes   : NONE")
    print()

    specs = [
        ("short-H5", 5, 3, 1),
        ("short-H8", 8, 3, 1),
    ]
    specs += [
        (
            f"long-H{h}",
            int(h),
            int(args.planted_k2_length),
            2,
        )
        for h in args.long_horizons
    ]

    rows = []
    failures = []
    t0 = time.perf_counter()

    for idx, (
        name,
        h,
        k2len,
        min_root,
    ) in enumerate(specs, 1):
        case = build_planted_case(
            rng,
            h_length=int(h),
            k2_length=int(k2len),
            oracle=construction_oracle,
            max_attempts=int(args.max_attempts),
            require_root_width=int(min_root),
        )

        start = case["start"]
        q0 = int(case["q0"])
        upper = upper_from_case(case)

        clear_p2()
        ref, ref_wall, ref_oracle_stats = fixed_reference(
            start,
            q0,
            int(h),
            int(upper["total"]),
            tables,
            d6,
            args,
        )
        rw = ref["witness"]

        clear_p2()
        before_rss = current_process_rss_mb()

        global_solver = GlobalRequirementClosure(
            tables,
            d6,
            p2_order=args.p2_order,
            p2_probe=args.p2_auto_probe_nodes,
            support_cache_max=args.support_cache_max,
        )

        gres = global_solver.solve(
            start,
            upper_witness=upper,
        )

        after_rss = current_process_rss_mb()
        gw = gres["witness"]

        parity = (
            gres["certified"]
            and rw is not None
            and gw is not None
            and int(rw["total"]) == int(gw["total"])
            and replay(start, rw)
            and replay(start, gw)
        )
        if not parity:
            failures.append(name)

        row = {
            "name": name,
            "horizon_witness": int(h),
            "attempt": int(case["attempt"]),
            "root_size": int(case["root_size"]),
            "upper_total": int(upper["total"]),
            "fixed_reference": {
                "total": None if rw is None else int(rw["total"]),
                "h": None if rw is None else len(rw["h_tail"]),
                "k2": None if rw is None else int(rw["k2_exact"]),
                "nodes": int(ref["total_nodes"]),
                "k2_calls": int(ref["k2_calls"]),
                "wall": ref_wall,
                "oracle": ref_oracle_stats,
            },
            "global_requirement": {
                "total": int(gw["total"]),
                "h": len(gw["h_tail"]),
                "k2": int(gw["k2_exact"]),
                "upper_total": int(gres["upper_total"]),
                "strict_budget": int(gres["strict_budget"]),
                "derived_h_max": gres["derived_h_max"],
                "derived_h_count": len(gres["derived_h_depths"]),
                "improvement_found": bool(
                    gres["improvement_found"]
                ),
                "predicate": gres["predicate"],
                "oracle": gres["oracle_stats"],
                "wall": float(gres["wall"]),
                "rss_before_mb": before_rss,
                "rss_after_mb": after_rss,
            },
            "parity": parity,
        }
        rows.append(row)

        pred = gres["predicate"] or {}

        print(
            f"[{idx}/{len(specs)}] {name} "
            f"upper={upper['total']} root={case['root_size']} "
            f"REF={row['fixed_reference']['h']}+"
            f"{row['fixed_reference']['k2']}="
            f"{row['fixed_reference']['total']} "
            f"GR={row['global_requirement']['h']}+"
            f"{row['global_requirement']['k2']}="
            f"{row['global_requirement']['total']} "
            f"Hmax={row['global_requirement']['derived_h_max']} "
            f"budgets={pred.get('attempt_count', 0)} "
            f"nodes={pred.get('total_nodes', 0):,} "
            f"K2q={pred.get('total_threshold_k2_calls', 0)} "
            f"supportBuild={gres['oracle_stats']['builds']:,} "
            f"cache={gres['oracle_stats']['max_cache_size']:,} "
            f"RSS={format_mb(before_rss, decimals=0)}"
            f"->{format_mb(after_rss, decimals=0)} "
            f"wall={ref_wall:.3f}/{gres['wall']:.3f}s "
            f"parity={'PASS' if parity else 'FAIL'}"
        )

    good = [r for r in rows if r["parity"]]

    ref_wall_sum = sum(
        r["fixed_reference"]["wall"]
        for r in good
    )
    global_wall_sum = sum(
        r["global_requirement"]["wall"]
        for r in good
    )

    builds = [
        r["global_requirement"]["oracle"]["builds"]
        for r in good
    ]
    cache_peaks = [
        r["global_requirement"]["oracle"]["max_cache_size"]
        for r in good
    ]

    elapsed = time.perf_counter() - t0
    overall = not failures

    ratio = (
        None
        if not ref_wall_sum
        else global_wall_sum / ref_wall_sum
    )

    print()
    print("# SUMMARY")
    print("exact parity         :", f"{len(good)}/{len(rows)}")
    print("failures             :", failures or "-")
    print("global heap nodes    : 0")
    print("outer H fixed        : NO")
    print(
        "GLOBAL/FIXED wall   :",
        "n/a" if ratio is None else f"{ratio:.3f}x",
    )
    print(
        "max support builds  :",
        max(builds) if builds else 0,
    )
    print(
        "max support cache   :",
        max(cache_peaks) if cache_peaks else 0,
    )
    print(
        "final RSS           :",
        format_mb(current_process_rss_mb()),
    )
    print("audit wall           :", f"{elapsed:.3f}s")

    if overall:
        conclusion = (
            "GLOBAL_TOTAL_REQUIREMENT_CLOSURE_EXACT_PASS"
        )
    else:
        conclusion = (
            "GLOBAL_TOTAL_REQUIREMENT_CLOSURE_PARITY_FAIL"
        )

    print("conclusion           :", conclusion)

    payload = {
        "version": "v37.23",
        "seed": int(args.seed),
        "rows": rows,
        "failures": failures,
        "baseline_rss_mb": baseline_rss,
        "final_rss_mb": current_process_rss_mb(),
        "global_heap_nodes": 0,
        "outer_h_fixed": False,
        "weighted_global_over_fixed_wall": ratio,
        "max_support_builds": (
            max(builds) if builds else 0
        ),
        "max_support_cache_size": (
            max(cache_peaks) if cache_peaks else 0
        ),
        "elapsed": elapsed,
        "overall_pass": overall,
        "conclusion": conclusion,
        "scope_note": (
            "GLOBAL-REQ derives the complete improving H-depth set from the "
            "strict total budget U-1. No planted/reference H horizon is passed "
            "into the GlobalRequirementClosure. The fixed-horizon solver is "
            "audit-only."
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("JSON                 :", out)
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
