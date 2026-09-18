#!/usr/bin/env python3
"""CubeLab v37.2.1 - unbiased permutation signal + RL hard20 repair.

Research-only. Production changes: NONE.

Why
===
v37.2 fresh cases were useful positive controls, but their generator used:
    scramble = inverse(H_witness)
so the planted witness solved the whole cube and guaranteed one K2=0 terminal.

v37.2 also replayed the historical hard20 calibration in the native UD frame,
while that H=17 / K2=3 calibration belongs to the RL-axis conjugated frame.

v37.2.1 fixes both issues:

A. RANDOM-ROOT CORPUS
   Generate ordinary canonical random full-cube scrambles. Keep roots only when
   the exact d6 orientation policy has at least one exact H suffix at the
   requested horizon. No solving H witness is planted.

B. HARD20 RL CALIBRATION
   Conjugate the historical scramble, prefix12, and known good H-tail into the
   canonical UD machinery using the same RL face map used by CubeLab's
   three-axis search.

Permutation/K2 remains LABEL ONLY. It never prunes.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
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
    D6MoveDomainOracle,
    MOVE_ORDER,
)
from p1_orientation_synthesis_v37_1 import synthesize_orientation_suffix
from audit_permutation_secondary_v37_2 import (
    HARD20,
    HARD20_PREFIX12,
    enumerate_policy_words,
    evaluate_word_set,
    exact_k2_labeler,
)

RL_MAP = {
    "R": "U",
    "U": "L",
    "L": "D",
    "D": "R",
    "F": "F",
    "B": "B",
}
RL_INV = {v: k for k, v in RL_MAP.items()}

HARD20_GOOD_TAIL_ORIGINAL = ("F", "D2", "L'", "D2", "B")


def map_move(move, fmap):
    return fmap[move[0]] + move[1:]


def map_word(word, fmap):
    return tuple(map_move(m, fmap) for m in word)


def random_scramble(rng, length):
    out = []
    last = None
    for _ in range(length):
        opts = [m for m in MOVE_ORDER if ts.allow(last, m)]
        move = rng.choice(opts)
        out.append(move)
        last = move[0]
    return tuple(out)


def find_random_root_case(
    rng,
    horizon,
    scramble_length,
    oracle,
    tables,
    k2_label,
    word_cap,
    max_attempts,
):
    for attempt in range(1, max_attempts + 1):
        scramble = random_scramble(rng, scramble_length)
        start = ts.engine.from_word(" ".join(scramble))
        q0 = ts.q_of(start)
        if q0 == ts.GOAL_Q:
            continue

        root_domain = oracle.allowed_domain(q0, horizon, None)
        if root_domain.size() == 0:
            continue

        words, truncated = enumerate_policy_words(
            q0, horizon, None, oracle, word_cap
        )
        if truncated or not words:
            continue

        greedy = synthesize_orientation_suffix(
            q0, horizon, None, oracle
        )
        if not greedy.success:
            raise RuntimeError(
                f"random-root attempt {attempt}: exact policy greedy failed"
            )

        ev = evaluate_word_set(
            f"random-root-h{horizon}-attempt{attempt}",
            start,
            q0,
            words,
            tables,
            k2_label,
            greedy_word=greedy.word,
        )

        return {
            "attempt": attempt,
            "kind": "random-root",
            "horizon": horizon,
            "scramble_length": scramble_length,
            "scramble": list(scramble),
            "q0": int(q0),
            "greedy_word": list(greedy.word),
            **ev,
        }

    raise RuntimeError(
        f"no usable random root for horizon={horizon} "
        f"within {max_attempts} attempts"
    )


def hard20_rl_case(oracle, tables, k2_label, word_cap):
    mapped_scramble = map_word(HARD20, RL_MAP)
    mapped_prefix = map_word(HARD20_PREFIX12, RL_MAP)
    mapped_good_tail = map_word(HARD20_GOOD_TAIL_ORIGINAL, RL_MAP)

    start = ts.engine.from_word(" ".join(mapped_scramble)).apply_word(
        mapped_prefix
    )
    q0 = ts.q_of(start)
    remaining = 5
    previous_face = mapped_prefix[-1][0]

    words, truncated = enumerate_policy_words(
        q0, remaining, previous_face, oracle, word_cap
    )
    if truncated:
        raise RuntimeError("RL hard20 orientation universe exceeded word cap")
    if not words:
        raise RuntimeError(
            "RL hard20 still has no Orientation-valid rem5 suffixes"
        )

    known_tail_present = mapped_good_tail in set(words)

    ev = evaluate_word_set(
        "hard20-RL-prefix12-rem5",
        start,
        q0,
        words,
        tables,
        k2_label,
        greedy_word=None,
    )

    known_rec = next(
        (
            rec for rec in ev["records"]
            if tuple(rec["word"]) == mapped_good_tail
        ),
        None,
    )

    mapped_records = []
    for rec in ev["records"]:
        mapped_records.append({
            **rec,
            "word_original_frame": list(
                map_word(tuple(rec["word"]), RL_INV)
            ),
            "k2_word_original_frame": list(
                map_word(tuple(rec["k2_word"]), RL_INV)
            ),
        })

    ev["records"] = mapped_records

    return {
        "kind": "hard20-RL-prefix12",
        "axis": "RL",
        "face_map_to_canonical_UD": RL_MAP,
        "scramble_original": list(HARD20),
        "scramble_mapped": list(mapped_scramble),
        "prefix_original": list(HARD20_PREFIX12),
        "prefix_mapped": list(mapped_prefix),
        "remaining": remaining,
        "q0": int(q0),
        "known_good_tail_original": list(HARD20_GOOD_TAIL_ORIGINAL),
        "known_good_tail_mapped": list(mapped_good_tail),
        "known_good_tail_present": known_tail_present,
        "known_good_tail_exact_k2": (
            None if known_rec is None else known_rec["k2_exact"]
        ),
        "known_good_tail_p2lb": (
            None if known_rec is None else known_rec["p2lb"]
        ),
        **ev,
    }


def summarize(rows):
    spreads = [r["k2_spread"] for r in rows]
    return {
        "cases": len(rows),
        "permutation_divergence_cases": sum(
            r["permutation_divergence"] for r in rows
        ),
        "root_contraction_cases": sum(
            r["root_domain_contracted"] for r in rows
        ),
        "k2_min_zero_cases": sum(r["k2_min"] == 0 for r in rows),
        "k2_spread_hist": dict(sorted(Counter(spreads).items())),
        "k2_spread_mean": statistics.mean(spreads) if spreads else None,
        "k2_spread_median": statistics.median(spreads) if spreads else None,
        "orientation_greedy_optimal": sum(
            r["greedy_optimal"] for r in rows
        ),
        "orientation_greedy_loss_mean": statistics.mean(
            r["greedy_loss"] for r in rows
        ) if rows else None,
        "p2lb_choice_optimal": sum(
            r["p2lb_choice_optimal"] for r in rows
        ),
        "p2lb_choice_loss_mean": statistics.mean(
            r["p2lb_choice_loss"] for r in rows
        ) if rows else None,
        "orientation_suffix_count_median": statistics.median(
            r["orientation_suffixes"] for r in rows
        ) if rows else None,
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
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 6])
    ap.add_argument("--count-per-horizon", type=int, default=20)
    ap.add_argument("--random-scramble-length", type=int, default=15)
    ap.add_argument("--max-root-attempts", type=int, default=20000)
    ap.add_argument("--word-cap", type=int, default=4096)
    ap.add_argument("--k2-max-depth", type=int, default=18)
    ap.add_argument("--p2-order", default="auto")
    ap.add_argument("--p2-auto-probe-nodes", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument(
        "--output",
        default="reports/v37/permutation_secondary_unbiased_v37_2_1.json",
    )
    args = ap.parse_args()

    tables, loaded = ts.cache_load_or_build(Path(args.cache))
    co, eo, sl, cos, eos, *_ = tables

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
        tables,
        args.p2_order,
        args.p2_auto_probe_nodes,
        args.k2_max_depth,
    )

    rng = random.Random(args.seed)

    print("# CubeLab v37.2.1 - UNBIASED PERMUTATION SIGNAL + RL HARD20")
    print()
    print("tables               :", "cache" if loaded else "built")
    print("horizons             :", args.horizons)
    print("count/horizon        :", args.count_per_horizon)
    print("random scramble len  :", args.random_scramble_length)
    print("word cap             :", args.word_cap)
    print("seed                 :", args.seed)
    print("production changes   : NONE")
    print()

    t0 = time.perf_counter()
    rows = []
    cid = 0

    for horizon in args.horizons:
        print(f"# RANDOM-ROOT HORIZON {horizon}")
        group = []
        for local in range(1, args.count_per_horizon + 1):
            cid += 1
            row = find_random_root_case(
                rng,
                horizon,
                args.random_scramble_length,
                oracle,
                tables,
                k2_label,
                args.word_cap,
                args.max_root_attempts,
            )
            row["case"] = cid
            rows.append(row)
            group.append(row)

            if local <= 5:
                print(
                    f"[{local:2d}/{args.count_per_horizon}] "
                    f"attempt={row['attempt']:4d} "
                    f"Hwords={row['orientation_suffixes']:3d} "
                    f"root={row['root_orientation_domain_size']}→"
                    f"{row['root_best_k2_domain_size']} "
                    f"K2={row['k2_min']}..{row['k2_max']} "
                    f"spread={row['k2_spread']} "
                    f"greedyLoss={row['greedy_loss']} "
                    f"P2LBLoss={row['p2lb_choice_loss']}"
                )

        s = summarize(group)
        print(
            "summary              : "
            f"divergence={s['permutation_divergence_cases']}/{s['cases']} "
            f"rootContract={s['root_contraction_cases']}/{s['cases']} "
            f"K2min0={s['k2_min_zero_cases']}/{s['cases']} "
            f"greedyOptimal={s['orientation_greedy_optimal']}/{s['cases']} "
            f"P2LBOptimal={s['p2lb_choice_optimal']}/{s['cases']} "
            f"spreadMedian={s['k2_spread_median']}"
        )
        print()

    print("# HARD20 RL PREFIX12 / REM5 CALIBRATION")
    hard = hard20_rl_case(
        oracle, tables, k2_label, args.word_cap
    )
    print("axis                 :", hard["axis"])
    print("mapped prefix        :", " ".join(hard["prefix_mapped"]))
    print("known good tail orig :", " ".join(hard["known_good_tail_original"]))
    print("known good tail map  :", " ".join(hard["known_good_tail_mapped"]))
    print("known tail present   :", hard["known_good_tail_present"])
    print("known tail exact K2  :", hard["known_good_tail_exact_k2"])
    print("orientation suffixes :", hard["orientation_suffixes"])
    print("root domain          :", hard["root_orientation_domain"])
    print("best-K2 root domain  :", hard["root_best_k2_domain"])
    print("exact K2 values      :", hard["distinct_exact_k2"])
    print("spread               :", hard["k2_spread"])
    print("root contracted      :", hard["root_domain_contracted"])
    print("P2LB choice loss     :", hard["p2lb_choice_loss"])
    print("best records:")
    for rec in hard["best_records"]:
        orig = map_word(tuple(rec["word"]), RL_INV)
        print(
            f"  Htail(mapped)={' '.join(rec['word'])} "
            f"Htail(orig)={' '.join(orig)} "
            f"K2={rec['k2_exact']} P2LB={rec['p2lb']}"
        )
    print()

    by_h = {
        str(h): summarize([r for r in rows if r["horizon"] == h])
        for h in args.horizons
    }
    total_summary = summarize(rows)

    hard_gate = (
        hard["known_good_tail_present"]
        and hard["known_good_tail_exact_k2"] == 3
        and hard["permutation_divergence"]
    )

    fresh_gate = (
        total_summary["cases"] > 0
        and total_summary["permutation_divergence_cases"] > 0
        and total_summary["root_contraction_cases"] > 0
    )

    overall = fresh_gate and hard_gate
    elapsed = time.perf_counter() - t0

    print("# SUMMARY")
    print("fresh cases          :", total_summary["cases"])
    print(
        "fresh divergence     :",
        f"{total_summary['permutation_divergence_cases']}/"
        f"{total_summary['cases']}",
    )
    print(
        "fresh root contraction:",
        f"{total_summary['root_contraction_cases']}/"
        f"{total_summary['cases']}",
    )
    print(
        "fresh K2min=0        :",
        f"{total_summary['k2_min_zero_cases']}/"
        f"{total_summary['cases']}",
    )
    print("hard20 RL gate       :", "PASS" if hard_gate else "FAIL")
    print("exact K2 cache states:", len(k2_cache))
    print("overall              :", "PASS" if overall else "FAIL")
    print("wall                 :", f"{elapsed:.3f}s")
    print(
        "conclusion           :",
        "PERMUTATION_SECONDARY_CHANNEL_CONFIRMED"
        if overall else
        "PERMUTATION_SECONDARY_CHANNEL_NEEDS_MORE_EVIDENCE",
    )

    payload = {
        "version": "v37.2.1",
        "seed": args.seed,
        "horizons": args.horizons,
        "count_per_horizon": args.count_per_horizon,
        "random_scramble_length": args.random_scramble_length,
        "rows": rows,
        "by_horizon": by_h,
        "total_summary": total_summary,
        "hard20_RL": hard,
        "fresh_gate": fresh_gate,
        "hard20_gate": hard_gate,
        "k2_cache_states": len(k2_cache),
        "overall_pass": overall,
        "elapsed": elapsed,
        "conclusion": (
            "PERMUTATION_SECONDARY_CHANNEL_CONFIRMED"
            if overall else
            "PERMUTATION_SECONDARY_CHANNEL_NEEDS_MORE_EVIDENCE"
        ),
        "scope_note": (
            "Permutation is still label-only. Random-root cases are conditioned "
            "only on existence of an exact H suffix at the requested horizon; "
            "no solving H witness is planted. Hard20 is evaluated in the RL "
            "conjugated frame used by the historical calibration."
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
