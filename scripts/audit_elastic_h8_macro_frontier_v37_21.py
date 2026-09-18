#!/usr/bin/env python3
"""CubeLab v37.21 - elastic H7/H8 macro-frontier admission.

Research-only. Production changes: NONE.

v37.20.4 rejected d6 multiprocessing on an untouched holdout.  v37.21 stops
retuning d6 and moves to naturally coarser H7/H8 work.

Exact extension
===============
For remaining H depth <= 6:
    use the compiled exact d6 oracle.

For remaining H depth 7 or 8:
    enumerate one canonical HTM move and ask whether the child has an exact
    completion at depth-1.

The d6 oracle is the exact base case.  The extension is memoized by
(q, remaining_depth, previous_face).

Validation
==========
Before performance cases, construct concrete planted H5+K2 and H8+K2 examples
with no intermediate full-DR visit during H.  The planted H word must remain
supported move-by-move.

Longer-frontier cases
=====================
Construct fresh planted H7/H8 + random K2 roots.  The planted word is only an
admission witness, not assumed optimal.

Compare exact:
MEMO = v37.16 unified total-budget predicate
PORT = sound no-closed-cache BnB + bounded multi-H seed

Require equal exact total and full replay.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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

from p1_orientation_domains_v37_0 import (
    D6MoveDomainOracle,
    MOVE_INDEX,
)
from audit_memoized_total_budget_v37_16 import MemoTotalBudgetSearch
from audit_perm_aware_seed_portfolio_v37_18_4 import PortfolioSeedBnB


P2_MOVES = (
    "U", "U'", "U2",
    "D", "D'", "D2",
    "L2", "R2", "F2", "B2",
)


class MoveDomain:
    def __init__(self, moves):
        self._moves = tuple(moves)
        self._set = frozenset(self._moves)

    def moves(self):
        return self._moves

    def contains(self, move):
        return move in self._set

    def size(self):
        return len(self._moves)


class ExtendedOrientationOracle:
    """Exact d6 oracle extended on demand to a small larger horizon."""

    def __init__(self, base, max_depth=8):
        if int(max_depth) < int(base.table.max_depth):
            raise ValueError("extended max depth below base depth")
        self.base = base
        self.co = base.co
        self.eo = base.eo
        self.sl = base.sl
        self.table = SimpleNamespace(max_depth=int(max_depth))
        self.cache = {}
        self.hits = 0
        self.builds = 0

    def allowed_domain(self, q, remaining, previous_face):
        q = int(q)
        remaining = int(remaining)

        if remaining <= int(self.base.table.max_depth):
            return self.base.allowed_domain(q, remaining, previous_face)

        if remaining < 1 or remaining > int(self.table.max_depth):
            return MoveDomain(())

        key = (q, remaining, previous_face)
        hit = self.cache.get(key)
        if hit is not None:
            self.hits += 1
            return hit

        moves = []
        for move in ts.MOVE_ORDER:
            if not ts.allow(previous_face, move):
                continue

            mi = MOVE_INDEX[move]
            nq = ts.q_move(q, mi, self.co, self.eo, self.sl)

            child = self.allowed_domain(
                int(nq),
                remaining - 1,
                move[0],
            )
            if child.size() > 0:
                moves.append(move)

        val = MoveDomain(moves)
        self.cache[key] = val
        self.builds += 1
        return val


def inv_move(move):
    if move.endswith("2"):
        return move
    if move.endswith("'"):
        return move[:-1]
    return move + "'"


def inverse_word(word):
    return tuple(inv_move(m) for m in reversed(tuple(word)))


def canonical_random_word(rng, moves, length):
    out = []
    last = None
    for _ in range(int(length)):
        opts = [m for m in moves if ts.allow(last, m)]
        if not opts:
            raise RuntimeError("no canonical random move options")
        m = rng.choice(opts)
        out.append(m)
        last = m[0]
    return tuple(out)


def build_planted_case(
    rng,
    *,
    h_length,
    k2_length,
    oracle,
    max_attempts,
    require_root_width=1,
):
    """Construct start whose known solution is Hword + K2word."""
    for attempt in range(1, int(max_attempts) + 1):
        k2word = canonical_random_word(rng, P2_MOVES, k2_length)

        terminal = ts.engine.from_word(
            " ".join(inverse_word(k2word))
        )
        if not ts.is_dr(terminal) or terminal.is_solved():
            continue

        hword = canonical_random_word(
            rng, ts.MOVE_ORDER, h_length
        )

        scramble = inverse_word(k2word) + inverse_word(hword)
        start = ts.engine.from_word(" ".join(scramble))

        if not start.apply_word(hword + k2word).is_solved():
            continue

        state = start
        good = True
        for i, move in enumerate(hword, 1):
            state = state.apply_word((move,))
            if i < len(hword) and ts.is_dr(state):
                good = False
                break
        if not good or not ts.is_dr(state):
            continue

        q = ts.q_of(start)
        domain = oracle.allowed_domain(q, int(h_length), None)
        if domain.size() < int(require_root_width):
            continue
        if not domain.contains(hword[0]):
            continue

        trace = []
        qcur = int(q)
        prev = None
        support_ok = True

        for i, move in enumerate(hword):
            rem = int(h_length) - i
            dom = oracle.allowed_domain(qcur, rem, prev)
            supported = bool(dom.contains(move))
            trace.append({
                "remaining": rem,
                "domain_size": int(dom.size()),
                "domain": list(dom.moves()),
                "move": move,
                "supported": supported,
            })
            if not supported:
                support_ok = False
                break

            mi = MOVE_INDEX[move]
            qcur = int(ts.q_move(
                qcur, mi, oracle.co, oracle.eo, oracle.sl
            ))
            prev = move[0]

        if not support_ok:
            continue

        return {
            "attempt": attempt,
            "scramble": scramble,
            "start": start,
            "q0": int(q),
            "h_word": hword,
            "k2_word": k2word,
            "known_total": len(hword) + len(k2word),
            "root_domain": tuple(domain.moves()),
            "root_size": int(domain.size()),
            "support_trace": trace,
        }

    raise RuntimeError(
        f"failed to build planted H{h_length} case "
        f"within {max_attempts} attempts"
    )


def clear_p2():
    fn = getattr(p2, "clear_double_star_query_cache", None)
    if callable(fn):
        fn()


def replay(start, witness):
    return (
        witness is not None
        and start.apply_word(
            tuple(witness["h_tail"]) + tuple(witness["k2_word"])
        ).is_solved()
    )


def run_memo(start, q0, depths, max_total, tables, oracle, args):
    clear_p2()
    solver = MemoTotalBudgetSearch(
        tables,
        oracle,
        p2_order=args.p2_order,
        p2_probe=args.p2_auto_probe_nodes,
    )
    t0 = time.perf_counter()
    res = solver.exact_min_total(
        start, int(q0), None, depths, int(max_total)
    )
    return res, time.perf_counter() - t0


def run_port(start, q0, depths, tables, oracle, args):
    clear_p2()
    solver = PortfolioSeedBnB(
        tables,
        oracle,
        p2_order=args.p2_order,
        p2_probe=args.p2_auto_probe_nodes,
        max_k2=args.max_k2,
        p1_pose_cache=args.p1_pose_cache,
        seed_depths=depths,
        seed_per_depth=args.seed_per_depth,
    )
    t0 = time.perf_counter()
    res = solver.solve_portfolio(
        start, int(q0), None, depths
    )
    return res, time.perf_counter() - t0


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
        "--depths",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    ap.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[7, 8],
    )
    ap.add_argument("--count-per-horizon", type=int, default=3)
    ap.add_argument("--planted-k2-length", type=int, default=6)
    ap.add_argument("--seed-per-depth", type=int, default=8)
    ap.add_argument("--min-root-width", type=int, default=2)
    ap.add_argument("--max-attempts", type=int, default=10000)
    ap.add_argument("--max-k2", type=int, default=18)
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
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument(
        "--output",
        default="reports/v37/elastic_h8_macro_frontier_v37_21.json",
    )
    args = ap.parse_args()

    tables, _loaded = ts.cache_load_or_build(Path(args.cache))
    co, eo, sl, cos, eos, *_ = tables

    base = D6MoveDomainOracle.load(
        Path(args.d6_table), co, eo, sl, cos, eos
    )
    oracle = ExtendedOrientationOracle(
        base, max_depth=max(args.depths)
    )

    p2.configure_double_star_cache(
        args.double_star_cache, args.double_star_index
    )
    dstar.configure(
        args.double_star_cache, args.double_star_index
    )
    p1pose.configure(args.p1_pose_cache)
    p1pose.preload()

    depths = tuple(sorted(set(
        int(d) for d in args.depths
        if 1 <= int(d) <= int(oracle.table.max_depth)
    )))
    if not depths:
        raise SystemExit("no valid elastic H depths")

    rng = random.Random(int(args.seed))

    print("# CubeLab v37.21 - ELASTIC H7/H8 MACRO-FRONTIER ADMISSION")
    print()
    print("RNG seed             :", args.seed)
    print("base exact oracle    : d6")
    print("extended horizon     :", oracle.table.max_depth)
    print("elastic H depths     :", depths)
    print("target horizons      :", args.horizons)
    print("cases/horizon        :", args.count_per_horizon)
    print("planted K2 length    :", args.planted_k2_length)
    print("seed terminals/depth :", args.seed_per_depth)
    print("multiprocessing      : NONE")
    print("production changes   : NONE")
    print()

    print("# CONCRETE SUPPORT VALIDATION")
    validations = []
    validation_failures = []

    for h in (5, 8):
        case = build_planted_case(
            rng,
            h_length=h,
            k2_length=max(3, min(args.planted_k2_length, 6)),
            oracle=oracle,
            max_attempts=args.max_attempts,
            require_root_width=1,
        )
        ok = (
            all(x["supported"] for x in case["support_trace"])
            and case["start"].apply_word(
                case["h_word"] + case["k2_word"]
            ).is_solved()
        )
        if not ok:
            validation_failures.append(h)

        validations.append({
            "h": h,
            "attempt": int(case["attempt"]),
            "scramble": list(case["scramble"]),
            "h_word": list(case["h_word"]),
            "k2_word": list(case["k2_word"]),
            "known_total": int(case["known_total"]),
            "root_size": int(case["root_size"]),
            "root_domain": list(case["root_domain"]),
            "support_trace": case["support_trace"],
            "pass": ok,
        })

        print(
            f"H{h}: attempt={case['attempt']} "
            f"root={case['root_size']} "
            f"H={' '.join(case['h_word'])} "
            f"K2={' '.join(case['k2_word'])} "
            f"support={'PASS' if ok else 'FAIL'}"
        )

    if validation_failures:
        raise SystemExit(
            f"concrete support validation failed: {validation_failures}"
        )

    print()

    rows = []
    failures = []
    case_no = 0
    t0 = time.perf_counter()
    max_total = max(depths) + int(args.max_k2)

    for h in args.horizons:
        print(f"# FRESH PLANTED H={h}")

        for local_i in range(1, int(args.count_per_horizon) + 1):
            case_no += 1
            planted = build_planted_case(
                rng,
                h_length=int(h),
                k2_length=int(args.planted_k2_length),
                oracle=oracle,
                max_attempts=args.max_attempts,
                require_root_width=int(args.min_root_width),
            )

            start = planted["start"]
            q0 = planted["q0"]

            order = (
                ("MEMO", "PORT")
                if case_no % 2
                else ("PORT", "MEMO")
            )
            got = {}

            for arm in order:
                if arm == "MEMO":
                    got["MEMO"] = run_memo(
                        start, q0, depths, max_total,
                        tables, oracle, args,
                    )
                else:
                    got["PORT"] = run_port(
                        start, q0, depths,
                        tables, oracle, args,
                    )

            memo, mwall = got["MEMO"]
            port, pwall = got["PORT"]

            mw = memo["witness"]
            pw = port["witness"]

            mreplay = replay(start, mw)
            preplay = replay(start, pw)

            parity = (
                mw is not None
                and pw is not None
                and int(mw["total"]) == int(pw["total"])
                and mreplay
                and preplay
            )
            if not parity:
                failures.append(case_no)

            row = {
                "case": case_no,
                "target_h": int(h),
                "attempt": int(planted["attempt"]),
                "scramble": list(planted["scramble"]),
                "planted_h_word": list(planted["h_word"]),
                "planted_k2_word": list(planted["k2_word"]),
                "planted_total": int(planted["known_total"]),
                "root_size": int(planted["root_size"]),
                "root_domain": list(planted["root_domain"]),
                "order": list(order),
                "parity": parity,
                "memo": {
                    "total": None if mw is None else int(mw["total"]),
                    "h": None if mw is None else len(mw["h_tail"]),
                    "k2": None if mw is None else int(mw["k2_exact"]),
                    "nodes": int(memo["total_nodes"]),
                    "k2_calls": int(memo["k2_calls"]),
                    "wall": mwall,
                    "replay": mreplay,
                },
                "port": {
                    "total": None if pw is None else int(pw["total"]),
                    "h": None if pw is None else len(pw["h_tail"]),
                    "k2": None if pw is None else int(pw["k2_exact"]),
                    "seed_total": int(port["seed_total"]),
                    "seed_h": len(port["seed_witness"]["h_tail"]),
                    "seed_k2": int(port["seed_witness"]["k2_exact"]),
                    "seed_pool_size": int(port["seed_pool_size"]),
                    "seed_pool_by_depth": port["seed_pool_by_depth"],
                    "seed_total_wall": float(port["seed_total_wall"]),
                    "proof_wall_after_seed": float(
                        port["proof_wall_after_seed"]
                    ),
                    "nodes": int(port["nodes"]),
                    "k2_calls": int(port["k2_calls"]),
                    "wall": pwall,
                    "replay": preplay,
                },
                "port_over_memo_wall": (
                    pwall / mwall if mwall else None
                ),
            }
            rows.append(row)

            print(
                f"[{local_i}/{args.count_per_horizon}] "
                f"case={case_no:2d} attempt={planted['attempt']:4d} "
                f"root={planted['root_size']} "
                f"order={'->'.join(order)} "
                f"planted={planted['known_total']} "
                f"M={row['memo']['h']}+{row['memo']['k2']}="
                f"{row['memo']['total']} "
                f"P={row['port']['h']}+{row['port']['k2']}="
                f"{row['port']['total']} "
                f"Pseed={row['port']['seed_h']}+{row['port']['seed_k2']}="
                f"{row['port']['seed_total']} "
                f"pool={row['port']['seed_pool_size']} "
                f"proof={row['port']['proof_wall_after_seed']:.3f}s "
                f"wall={mwall:.3f}->{pwall:.3f}s "
                f"x{(pwall/mwall if mwall else 0):.3f} "
                f"parity={'PASS' if parity else 'FAIL'}"
            )
        print()

    good = [r for r in rows if r["parity"]]

    memo_wall = sum(r["memo"]["wall"] for r in good)
    port_wall = sum(r["port"]["wall"] for r in good)
    memo_nodes = sum(r["memo"]["nodes"] for r in good)
    port_nodes = sum(r["port"]["nodes"] for r in good)
    memo_k2 = sum(r["memo"]["k2_calls"] for r in good)
    port_k2 = sum(r["port"]["k2_calls"] for r in good)

    roots = [r["root_size"] for r in good]
    pools = [r["port"]["seed_pool_size"] for r in good]
    proofs = [r["port"]["proof_wall_after_seed"] for r in good]
    totals = [r["port"]["wall"] for r in good]

    port_faster = sum(
        r["port"]["wall"] < r["memo"]["wall"]
        for r in good
    )
    seed_opt = sum(
        r["port"]["seed_total"] == r["port"]["total"]
        for r in good
    )

    elapsed = time.perf_counter() - t0
    overall = not failures and not validation_failures

    wall_ratio = (
        None if not memo_wall else port_wall / memo_wall
    )

    print("# SUMMARY")
    print("support validation   : 2/2")
    print("long exact parity    :", f"{len(good)}/{len(rows)}")
    print("failures             :", failures or "-")
    print(
        "PORT/MEMO wall      :",
        "n/a" if wall_ratio is None else f"{wall_ratio:.3f}x",
    )
    print(
        "PORT/MEMO nodes     :",
        "n/a" if not memo_nodes else f"{port_nodes/memo_nodes:.3f}x",
    )
    print(
        "PORT/MEMO K2 calls  :",
        "n/a" if not memo_k2 else f"{port_k2/memo_k2:.3f}x",
    )
    print("PORT faster cases    :", f"{port_faster}/{len(good)}")
    print("PORT seed optimum    :", f"{seed_opt}/{len(good)}")
    print(
        "mean root width     :",
        f"{statistics.mean(roots):.3f}" if roots else "n/a",
    )
    print(
        "mean seed pool      :",
        f"{statistics.mean(pools):.3f}" if pools else "n/a",
    )
    print(
        "median proof wall   :",
        f"{statistics.median(proofs):.3f}s" if proofs else "n/a",
    )
    print(
        "max proof wall      :",
        f"{max(proofs):.3f}s" if proofs else "n/a",
    )
    print(
        "median total wall   :",
        f"{statistics.median(totals):.3f}s" if totals else "n/a",
    )
    print(
        "extended cache hit/build:",
        f"{oracle.hits}/{oracle.builds}",
    )
    print("audit wall           :", f"{elapsed:.3f}s")

    if overall:
        coarse = (
            bool(proofs)
            and statistics.median(proofs) >= 0.05
        )
        if coarse:
            conclusion = "ELASTIC_H8_EXACT_AND_COARSE_FRONTIER_CONFIRMED"
        else:
            conclusion = "ELASTIC_H8_EXACT_BUT_FRONTIER_STILL_FINE"
    else:
        conclusion = "ELASTIC_H8_MACRO_FRONTIER_PARITY_FAIL"

    print("conclusion           :", conclusion)

    payload = {
        "version": "v37.21",
        "seed": int(args.seed),
        "base_depth": int(base.table.max_depth),
        "extended_depth": int(oracle.table.max_depth),
        "depths": depths,
        "horizons": args.horizons,
        "count_per_horizon": int(args.count_per_horizon),
        "planted_k2_length": int(args.planted_k2_length),
        "seed_per_depth": int(args.seed_per_depth),
        "validations": validations,
        "validation_failures": validation_failures,
        "rows": rows,
        "failures": failures,
        "weighted_port_over_memo_wall": wall_ratio,
        "weighted_port_over_memo_nodes": (
            None if not memo_nodes else port_nodes / memo_nodes
        ),
        "weighted_port_over_memo_k2_calls": (
            None if not memo_k2 else port_k2 / memo_k2
        ),
        "port_faster_cases": port_faster,
        "port_seed_optimal_cases": seed_opt,
        "mean_root_width": (
            None if not roots else statistics.mean(roots)
        ),
        "mean_seed_pool_size": (
            None if not pools else statistics.mean(pools)
        ),
        "median_proof_wall": (
            None if not proofs else statistics.median(proofs)
        ),
        "max_proof_wall": (
            None if not proofs else max(proofs)
        ),
        "median_total_wall": (
            None if not totals else statistics.median(totals)
        ),
        "extended_oracle_cache_hits": oracle.hits,
        "extended_oracle_cache_builds": oracle.builds,
        "elapsed": elapsed,
        "overall_pass": overall,
        "conclusion": conclusion,
        "scope_note": (
            "Exact d7/d8 next-move support is compiled on demand from the "
            "existing exact d6 oracle. No d7/d8 perimeter table is built. "
            "Fresh longer cases are planted H7/H8 + random Phase2 states; "
            "the planted total is an admission witness, not ground truth. "
            "Exact ground truth is MEMO/PORT agreement plus full replay."
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
