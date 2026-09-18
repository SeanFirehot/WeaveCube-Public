#!/usr/bin/env python3
"""CubeLab v37.2 - permutation-secondary-channel audit.

Research-only. Production changes: NONE.

Purpose
=======
v37.1 established that the final <=6 H columns can be synthesized exactly in
the Orientation/DR quotient with zero backtracking.

v37.2 asks the next necessary question:

    Among Orientation-valid H suffixes, how much does full permutation/K2
    quality vary?

This script therefore treats Orientation as the exact primary feasibility
channel and full Phase-2 distance as a LABEL ONLY.  No permutation-derived
quantity prunes anything.

For each case:
- enumerate every Orientation-valid suffix represented by the exact d6 policy;
- replay each suffix on the full cube state;
- compute terminal Phase-2 lower bound;
- compute exact boundary-compatible K2 distance;
- measure how much the exact K2 label contracts the Orientation root domain;
- compare a naive Orientation-only greedy choice with the best K2 continuation;
- compare terminal P2LB ranking with exact K2 ranking.

Validation scopes:
1. fresh full-domain 5-column cases;
2. fresh full-domain 6-column cases;
3. one restricted 8-column exact MDD multi-solution fixture;
4. the frozen hard20 prefix-12 calibration from v36.6 research.

The 8-column fixture is restricted-domain because the compiled exact policy is
currently complete only through depth 6.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
for p in (str(SRC), str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pdcc_phase2 as p2
import search_twist_skeleton as ts

from p1_orientation_domains_v37_0 import (
    ColumnDomain,
    D6MoveDomainOracle,
    MOVE_INDEX,
    MOVE_ORDER,
    build_restricted_orientation_mdd,
    enumerate_supported_words,
)
from p1_orientation_synthesis_v37_1 import synthesize_orientation_suffix


HARD20 = (
    "L", "R2", "U2", "B'", "D2", "L", "D2", "F'", "U'", "R",
    "U2", "L'", "F'", "D'", "R'", "B2", "D2", "R'", "U'", "F2",
)
HARD20_PREFIX12 = (
    "F2", "U", "R", "D2", "B2", "R", "D", "F", "L", "U2", "R'", "U",
)


def make_true_dr_case(rng, horizon, co, eo, sl):
    for _ in range(100_000):
        word = []
        last = None
        for _i in range(horizon):
            opts = [m for m in MOVE_ORDER if ts.allow(last, m)]
            move = rng.choice(opts)
            word.append(move)
            last = move[0]

        witness = tuple(word)
        scramble = ts.normalize(ts.inverse_word(witness))
        start = ts.engine.from_word(" ".join(scramble))
        q0 = ts.q_of(start)
        if q0 == ts.GOAL_Q:
            continue

        q = q0
        parent = None
        for move in witness:
            parent = q
            q = ts.q_move(q, MOVE_INDEX[move], co, eo, sl)

        if q == ts.GOAL_Q and parent != ts.GOAL_Q:
            return start, scramble, witness, q0
    raise RuntimeError(f"failed to create horizon={horizon} true-DR case")


def enumerate_policy_words(q0, remaining, previous_face, oracle, cap):
    out = []
    path = []
    truncated = False

    def rec(q, rem, last):
        nonlocal truncated
        if truncated:
            return
        if rem == 0:
            out.append(tuple(path))
            if len(out) >= cap:
                truncated = True
            return

        domain = oracle.allowed_domain(q, rem, last)
        for move in domain.moves():
            nq = ts.q_move(
                q, MOVE_INDEX[move], oracle.co, oracle.eo, oracle.sl
            )
            path.append(move)
            rec(nq, rem - 1, move[0])
            path.pop()
            if truncated:
                return

    rec(int(q0), int(remaining), previous_face)
    return tuple(out), truncated


def exact_k2_labeler(tables, order, auto_probe_nodes, max_depth):
    _co, _eo, _sl, _cos, _eos, cp, up, sp, cpd, upd, spd = tables
    cache = {}

    def label(state, hword):
        pk = ts.p_of(state)
        if pk is None:
            raise RuntimeError("orientation-valid terminal is not Phase-2 addressable")
        prev = hword[-1][0] if hword else None
        p2lb = int(ts.phase2_lb(pk, cpd, upd, spd))
        key = (int(pk), prev)
        hit = cache.get(key)
        if hit is None:
            stats = {}
            bword, dist = p2.shortest_b_compatible(
                pk,
                prev,
                max_depth,
                cp, up, sp, cpd, upd, spd,
                order=order,
                stats=stats,
                auto_probe_nodes=auto_probe_nodes,
            )
            if dist is None or bword is None:
                raise RuntimeError(
                    f"no exact K2 within max_depth={max_depth}; p2lb={p2lb}"
                )
            hit = (tuple(bword), int(dist), int(stats.get("nodes", 0)))
            cache[key] = hit
        bword, dist, nodes = hit
        return {
            "pk": int(pk),
            "p2lb": p2lb,
            "k2_word": list(bword),
            "k2_exact": dist,
            "k2_nodes": nodes,
        }

    return label, cache


def evaluate_word_set(
    name,
    start,
    q0,
    words,
    tables,
    k2_label,
    greedy_word=None,
):
    if not words:
        raise RuntimeError(f"{name}: empty Orientation-valid word set")

    records = []
    for word in sorted(set(words)):
        terminal = start.apply_word(word)
        if not ts.is_dr(terminal):
            raise RuntimeError(f"{name}: quotient-valid word failed full DR replay: {word}")
        lab = k2_label(terminal, word)
        records.append({
            "word": list(word),
            "first_move": word[0] if word else None,
            **lab,
        })

    k2s = [r["k2_exact"] for r in records]
    best_k2 = min(k2s)
    worst_k2 = max(k2s)
    best_records = [r for r in records if r["k2_exact"] == best_k2]
    best_root_moves = sorted({r["first_move"] for r in best_records})

    root_moves = sorted({r["first_move"] for r in records})
    by_root = {}
    for move in root_moves:
        rr = [r for r in records if r["first_move"] == move]
        by_root[move] = {
            "suffixes": len(rr),
            "min_k2": min(r["k2_exact"] for r in rr),
            "max_k2": max(r["k2_exact"] for r in rr),
            "min_p2lb": min(r["p2lb"] for r in rr),
        }

    p2lb_choice = min(
        records,
        key=lambda r: (r["p2lb"], r["k2_exact"], tuple(r["word"])),
    )
    greedy_rec = None
    if greedy_word is not None:
        gw = list(greedy_word)
        greedy_rec = next((r for r in records if r["word"] == gw), None)
        if greedy_rec is None:
            raise RuntimeError(f"{name}: greedy synthesized word missing from universe")

    distinct_pk = len({r["pk"] for r in records})
    distinct_k2 = sorted(set(k2s))

    return {
        "name": name,
        "orientation_suffixes": len(records),
        "root_orientation_domain": root_moves,
        "root_orientation_domain_size": len(root_moves),
        "root_best_k2_domain": best_root_moves,
        "root_best_k2_domain_size": len(best_root_moves),
        "root_domain_contracted": len(best_root_moves) < len(root_moves),
        "distinct_terminal_p": distinct_pk,
        "distinct_exact_k2": distinct_k2,
        "k2_min": best_k2,
        "k2_max": worst_k2,
        "k2_spread": worst_k2 - best_k2,
        "permutation_divergence": len(distinct_k2) > 1,
        "best_records": best_records[:8],
        "by_root": by_root,
        "p2lb_choice": p2lb_choice,
        "p2lb_choice_optimal": p2lb_choice["k2_exact"] == best_k2,
        "p2lb_choice_loss": p2lb_choice["k2_exact"] - best_k2,
        "greedy_record": greedy_rec,
        "greedy_optimal": (
            None if greedy_rec is None else greedy_rec["k2_exact"] == best_k2
        ),
        "greedy_loss": (
            None if greedy_rec is None else greedy_rec["k2_exact"] - best_k2
        ),
        "records": records,
    }


def full_domain_case(
    case_id,
    horizon,
    rng,
    oracle,
    tables,
    k2_label,
    co, eo, sl,
    word_cap,
):
    start, scramble, known, q0 = make_true_dr_case(rng, horizon, co, eo, sl)
    words, truncated = enumerate_policy_words(q0, horizon, None, oracle, word_cap)
    if truncated:
        return {
            "case": case_id,
            "kind": "full-domain",
            "horizon": horizon,
            "truncated": True,
            "orientation_suffixes_at_cap": len(words),
        }

    greedy = synthesize_orientation_suffix(q0, horizon, None, oracle)
    if not greedy.success:
        raise RuntimeError(f"case {case_id}: v37.1 greedy synthesis failed")

    ev = evaluate_word_set(
        f"full-h{horizon}-case{case_id}",
        start,
        q0,
        words,
        tables,
        k2_label,
        greedy_word=greedy.word,
    )
    return {
        "case": case_id,
        "kind": "full-domain",
        "horizon": horizon,
        "scramble": list(scramble),
        "known_witness": list(known),
        "q0": int(q0),
        "truncated": False,
        "greedy_word": list(greedy.word),
        **ev,
    }


def make_restricted_8_fixture(
    rng,
    co, eo, sl,
    tables,
    k2_label,
    domain_width,
    word_cap,
):
    for attempt in range(1, 4000):
        start, scramble, witness, q0 = make_true_dr_case(rng, 8, co, eo, sl)
        domains = []
        for w in witness:
            choices = {w}
            pool = [m for m in MOVE_ORDER if m != w]
            rng.shuffle(pool)
            choices.update(pool[: domain_width - 1])
            domains.append(ColumnDomain.from_moves(choices))

        mdd = build_restricted_orientation_mdd(q0, domains, co, eo, sl)
        if mdd.impossible or mdd.accepted_word_count < 2:
            continue
        if mdd.accepted_word_count > word_cap:
            continue

        words = enumerate_supported_words(mdd, q0)
        ev = evaluate_word_set(
            f"restricted-h8-attempt{attempt}",
            start,
            q0,
            words,
            tables,
            k2_label,
            greedy_word=None,
        )
        if ev["permutation_divergence"]:
            return {
                "kind": "restricted-8",
                "attempt": attempt,
                "scramble": list(scramble),
                "known_witness": list(witness),
                "q0": int(q0),
                "initial_domains": [list(d.moves()) for d in domains],
                "reduced_domains": [list(d.moves()) for d in mdd.reduced_domains],
                "mdd_valid_words": mdd.accepted_word_count,
                **ev,
            }

    raise RuntimeError("could not find 8-column multi-solution permutation-divergence fixture")


def hard20_case(oracle, tables, k2_label, word_cap):
    start = ts.engine.from_word(" ".join(HARD20)).apply_word(HARD20_PREFIX12)
    q0 = ts.q_of(start)
    remaining = 5
    words, truncated = enumerate_policy_words(q0, remaining, HARD20_PREFIX12[-1][0], oracle, word_cap)
    if truncated:
        raise RuntimeError("hard20 orientation universe exceeded word cap")
    ev = evaluate_word_set(
        "hard20-prefix12-rem5",
        start,
        q0,
        words,
        tables,
        k2_label,
        greedy_word=None,
    )
    return {
        "kind": "hard20-prefix12",
        "scramble": list(HARD20),
        "prefix": list(HARD20_PREFIX12),
        "remaining": remaining,
        "q0": int(q0),
        **ev,
    }


def summarize(rows):
    usable = [r for r in rows if not r.get("truncated")]
    spreads = [r["k2_spread"] for r in usable]
    greedy = [r for r in usable if r.get("greedy_optimal") is not None]
    return {
        "cases": len(rows),
        "usable": len(usable),
        "truncated": len(rows) - len(usable),
        "permutation_divergence_cases": sum(r["permutation_divergence"] for r in usable),
        "root_contraction_cases": sum(r["root_domain_contracted"] for r in usable),
        "k2_spread_hist": dict(sorted(Counter(spreads).items())),
        "k2_spread_mean": statistics.mean(spreads) if spreads else None,
        "k2_spread_median": statistics.median(spreads) if spreads else None,
        "orientation_greedy_optimal": sum(r["greedy_optimal"] for r in greedy),
        "orientation_greedy_cases": len(greedy),
        "orientation_greedy_loss_mean": (
            statistics.mean(r["greedy_loss"] for r in greedy) if greedy else None
        ),
        "p2lb_choice_optimal": sum(r["p2lb_choice_optimal"] for r in usable),
        "p2lb_choice_cases": len(usable),
        "p2lb_choice_loss_mean": (
            statistics.mean(r["p2lb_choice_loss"] for r in usable) if usable else None
        ),
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
        "--double-star-cache",
        default="reports/pdcc_cache/double_star_k_tables_v1.pkl",
    )
    ap.add_argument(
        "--double-star-index",
        default="reports/pdcc_cache/double_star_p2_index_v1.pkl",
    )
    ap.add_argument("--count-per-horizon", type=int, default=30)
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 6])
    ap.add_argument("--word-cap", type=int, default=4096)
    ap.add_argument("--k2-max-depth", type=int, default=18)
    ap.add_argument("--p2-order", default="auto")
    ap.add_argument("--p2-auto-probe-nodes", type=int, default=128)
    ap.add_argument("--domain-width-8", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument(
        "--output",
        default="reports/v37/permutation_secondary_audit_v37_2.json",
    )
    args = ap.parse_args()

    tables, loaded = ts.cache_load_or_build(Path(args.cache))
    co, eo, sl, cos, eos, cp, up, sp, cpd, upd, spd = tables

    oracle = D6MoveDomainOracle.load(
        Path(args.d6_table), co, eo, sl, cos, eos
    )
    p2.configure_double_star_cache(
        args.double_star_cache, args.double_star_index
    )
    preload = getattr(p2, "preload_double_star", None)
    if preload is not None:
        preload()
    clear = getattr(p2, "clear_double_star_query_cache", None)
    if clear is not None:
        clear()

    k2_label, k2_cache = exact_k2_labeler(
        tables, args.p2_order, args.p2_auto_probe_nodes, args.k2_max_depth
    )

    rng = random.Random(args.seed)

    print("# CubeLab v37.2 - PERMUTATION SECONDARY CHANNEL AUDIT")
    print()
    print("tables               :", "cache" if loaded else "built")
    print("orientation horizon  :", int(oracle.table.max_depth))
    print("fresh horizons       :", args.horizons)
    print("count/horizon        :", args.count_per_horizon)
    print("word cap/case        :", args.word_cap)
    print("K2 max depth         :", args.k2_max_depth)
    print("production changes   : NONE")
    print()

    t0 = time.perf_counter()
    rows = []
    cid = 0

    for horizon in args.horizons:
        print(f"# FULL-DOMAIN HORIZON {horizon}")
        for local in range(1, args.count_per_horizon + 1):
            cid += 1
            row = full_domain_case(
                cid, horizon, rng, oracle, tables, k2_label,
                co, eo, sl, args.word_cap
            )
            rows.append(row)
            if row.get("truncated"):
                print(f"[{local:2d}/{args.count_per_horizon}] TRUNCATED at {row['orientation_suffixes_at_cap']}")
                continue
            if local <= 5:
                print(
                    f"[{local:2d}/{args.count_per_horizon}] "
                    f"Hwords={row['orientation_suffixes']:3d} "
                    f"root={row['root_orientation_domain_size']}→"
                    f"{row['root_best_k2_domain_size']} "
                    f"K2={row['k2_min']}..{row['k2_max']} "
                    f"spread={row['k2_spread']} "
                    f"greedyLoss={row['greedy_loss']} "
                    f"P2LBLoss={row['p2lb_choice_loss']}"
                )
        group = [r for r in rows if r.get("horizon") == horizon]
        s = summarize(group)
        print(
            "summary              : "
            f"divergence={s['permutation_divergence_cases']}/{s['usable']} "
            f"rootContract={s['root_contraction_cases']}/{s['usable']} "
            f"greedyOptimal={s['orientation_greedy_optimal']}/{s['orientation_greedy_cases']} "
            f"P2LBOptimal={s['p2lb_choice_optimal']}/{s['p2lb_choice_cases']} "
            f"spreadMedian={s['k2_spread_median']}"
        )
        print()

    print("# RESTRICTED 8-COLUMN FIXTURE")
    case8 = make_restricted_8_fixture(
        rng, co, eo, sl, tables, k2_label,
        args.domain_width_8, args.word_cap
    )
    print("orientation suffixes :", case8["orientation_suffixes"])
    print("root domain          :", case8["root_orientation_domain"])
    print("best-K2 root domain  :", case8["root_best_k2_domain"])
    print("exact K2 values      :", case8["distinct_exact_k2"])
    print("spread               :", case8["k2_spread"])
    print("root contracted      :", case8["root_domain_contracted"])
    print()

    print("# HARD20 PREFIX12 / REM5 CALIBRATION")
    hard = hard20_case(oracle, tables, k2_label, args.word_cap)
    print("orientation suffixes :", hard["orientation_suffixes"])
    print("root domain          :", hard["root_orientation_domain"])
    print("best-K2 root domain  :", hard["root_best_k2_domain"])
    print("exact K2 values      :", hard["distinct_exact_k2"])
    print("spread               :", hard["k2_spread"])
    print("best records         :")
    for rec in hard["best_records"]:
        print(
            f"  Htail={' '.join(rec['word'])} "
            f"K2={rec['k2_exact']} P2LB={rec['p2lb']} "
            f"K2word={' '.join(rec['k2_word'])}"
        )
    print()

    by_h = {}
    for h in args.horizons:
        by_h[str(h)] = summarize([r for r in rows if r.get("horizon") == h])

    overall_usable = [r for r in rows if not r.get("truncated")]
    divergence = sum(r["permutation_divergence"] for r in overall_usable)
    contraction = sum(r["root_domain_contracted"] for r in overall_usable)
    overall = (
        len(overall_usable) > 0
        and divergence > 0
        and contraction > 0
        and case8["permutation_divergence"]
        and hard["permutation_divergence"]
    )

    elapsed = time.perf_counter() - t0

    print("# SUMMARY")
    print("fresh usable cases   :", len(overall_usable))
    print("fresh divergence     :", f"{divergence}/{len(overall_usable)}")
    print("fresh root contraction:", f"{contraction}/{len(overall_usable)}")
    print("8-column divergence  :", case8["permutation_divergence"])
    print("hard20 divergence    :", hard["permutation_divergence"])
    print("exact K2 cache states:", len(k2_cache))
    print("overall              :", "PASS" if overall else "FAIL")
    print("wall                 :", f"{elapsed:.3f}s")
    print(
        "conclusion           :",
        "PERMUTATION_SECONDARY_CHANNEL_REQUIRED"
        if overall else
        "PERMUTATION_SECONDARY_SIGNAL_NOT_ESTABLISHED",
    )

    payload = {
        "version": "v37.2",
        "seed": args.seed,
        "horizons": args.horizons,
        "count_per_horizon": args.count_per_horizon,
        "rows": rows,
        "by_horizon": by_h,
        "restricted_8": case8,
        "hard20": hard,
        "fresh_divergence_cases": divergence,
        "fresh_root_contraction_cases": contraction,
        "k2_cache_states": len(k2_cache),
        "overall_pass": overall,
        "elapsed": elapsed,
        "conclusion": (
            "PERMUTATION_SECONDARY_CHANNEL_REQUIRED"
            if overall else
            "PERMUTATION_SECONDARY_SIGNAL_NOT_ESTABLISHED"
        ),
        "scope_note": (
            "Permutation/K2 is evaluation-only in v37.2. No permutation score "
            "prunes Orientation-valid suffixes. Exact K2 labels are used to "
            "measure how much information the Orientation quotient discards."
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("JSON                 :", out)
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
