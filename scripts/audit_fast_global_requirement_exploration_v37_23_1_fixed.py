#!/usr/bin/env python3
"""
CubeLab v37.23.1 — FAST GLOBAL REQUIREMENT EXPLORATION OVERLAY

Purpose
-------
Temporary research mode for v37.23 Global Requirement Closure.

The exact v37.23 solver is intentionally left untouched.  This overlay:
  * replaces iterative exact-minimum proof with ONE bounded "is there any
    solution <= U-1?" probe;
  * repeats only after a strict improvement, at most N times;
  * never treats a bounded miss as UNSAT or as an optimality certificate;
  * enforces per-probe wall/node/support-build budgets;
  * keeps the existing v37.23 requirement-domain construction and exact d6/K2
    machinery.

This is an exploration tool, NOT a production solver and NOT an exact
shortest-solution certificate.

Place this file in CubeLab/scripts and run it from the repository root.

Recommended first run:
    python scripts/audit_fast_global_requirement_exploration_v37_23_1.py ^
      --d6-table reports/pdcc_cache/p1_tail_nosuffix_v36_6i13_d6.pkl ^
      --long-horizons 6 7 ^
      --planted-k2-length 4 ^
      --p2-order auto ^
      --p2-auto-probe-nodes 128 ^
      --seed 20260919 ^
      --support-cache-max 100000 ^
      --fast-max-witnesses 5 ^
      --fast-probe-time-limit 3 ^
      --fast-probe-node-cap 50000 ^
      --fast-support-build-cap 100000 ^
      --output reports/v37/global_requirement_fast_v37_23_1.json

PowerShell uses backticks (`) instead of ^ for line continuation.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import sys
import time
from pathlib import Path
from typing import Any

# Make CubeLab/src importable without requiring the caller to set
# $env:PYTHONPATH=src manually.  This script is expected to live in
# <repo>/scripts, so parents[1] is the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir():
    _src = str(_SRC_DIR)
    if _src not in sys.path:
        sys.path.insert(0, _src)


class _FastCut(RuntimeError):
    pass


_FAST = None
_LAST_PROBE: dict[str, Any] = {}


def _metric_snapshot(solver) -> dict[str, int]:
    k2 = getattr(solver, "k2", None)
    return {
        "nodes": int(getattr(solver, "nodes", 0)),
        "terminals": int(getattr(solver, "terminals", 0)),
        "success_hits": int(getattr(solver, "success_cache_hits", 0)),
        "fail_hits": int(getattr(solver, "fail_cache_hits", 0)),
        "transition_hits": int(getattr(solver, "transition_hits", 0)),
        "transition_builds": int(getattr(solver, "transition_builds", 0)),
        "domain_queries": int(getattr(solver, "domain_queries", 0)),
        "k2_calls": int(getattr(k2, "calls", 0)),
        "k2_success_hits": int(getattr(k2, "success_hits", 0)),
        "k2_fail_hits": int(getattr(k2, "fail_hits", 0)),
    }


def _delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {k: int(after.get(k, 0) - before.get(k, 0)) for k in before}


def _fast_exact_min_total(self, start, q0, previous_face, depths, max_total):
    """Bounded existence probe at the current strict budget U-1.

    Deliberately does NOT iterate from the lower bound upward.  A miss after a
    cut is UNKNOWN.  Even a fully exhausted miss is not promoted to a global
    optimality certificate by this exploration overlay.
    """
    global _LAST_PROBE

    cfg = _FAST
    depths = tuple(sorted(set(int(d) for d in depths)))
    max_total = int(max_total)

    # Keep the original v37.16 lower-bound telemetry meaning.
    if depths:
        p1lb = int(cfg.grc.ts.phase1_lb(int(q0), self.cos, self.eos))
        low = max(min(depths), p1lb)
    else:
        low = max_total

    before = _metric_snapshot(self)
    support0 = int(getattr(self.oo, "builds", 0))
    t0 = time.perf_counter()
    deadline = t0 + max(0.001, float(cfg.probe_time_limit))

    orig_find = self.find

    def check_limits() -> None:
        if time.perf_counter() >= deadline:
            raise _FastCut("time_limit")
        if int(getattr(self, "nodes", 0)) - before["nodes"] >= cfg.probe_node_cap:
            raise _FastCut("node_cap")
        if (
            int(getattr(self.oo, "builds", 0)) - support0
            >= cfg.support_build_cap
        ):
            raise _FastCut("support_build_cap")

    @functools.wraps(orig_find)
    def guarded_find(*args, **kwargs):
        check_limits()
        ans = orig_find(*args, **kwargs)
        check_limits()
        return ans

    # Recursive calls inside MemoTotalBudgetSearch.find use self.find, so an
    # instance-level guard intercepts the entire recursion without modifying
    # the v37.16 source file.
    self.find = guarded_find

    hit = None
    cut = False
    cut_reason = None

    try:
        hit = self.find(
            start.poses,
            int(q0),
            depths,
            previous_face,
            None,
            max_total,
        )
    except _FastCut as exc:
        cut = True
        cut_reason = str(exc)
    finally:
        self.find = orig_find

    wall = time.perf_counter() - t0
    after = _metric_snapshot(self)
    d = _delta(after, before)
    support_builds = int(getattr(self.oo, "builds", 0)) - support0

    witness = None
    if hit is not None:
        witness = dict(hit)
        witness.setdefault("budget", max_total)

    attempt = {
        "budget": max_total,
        "found": witness is not None,
        "cut": cut,
        "cut_reason": cut_reason,
        "wall": wall,
        **d,
        "support_builds": support_builds,
    }

    _LAST_PROBE = {
        "budget": max_total,
        "found": witness is not None,
        "cut": cut,
        "cut_reason": cut_reason,
        "nodes": d["nodes"],
        "terminals": d["terminals"],
        "domain_queries": d["domain_queries"],
        "k2_calls": d["k2_calls"],
        "support_builds": support_builds,
        "wall": wall,
    }

    # Return both v37.16 native field names and v37.15/v37.23 aliases.
    # This also tolerates either telemetry adapter version locally.
    return {
        "lower_budget": low,
        "attempts": [attempt],
        "witness": witness,
        "total_nodes": d["nodes"],
        "total_terminals": d["terminals"],
        "success_cache_hits": d["success_hits"],
        "fail_cache_hits": d["fail_hits"],
        "transition_hits": d["transition_hits"],
        "transition_builds": d["transition_builds"],
        "domain_queries": d["domain_queries"],
        "total_domain_queries": d["domain_queries"],
        "success_cache_size": len(getattr(self, "success_cache", {})),
        "fail_cache_size": len(getattr(self, "fail_upto", {})),
        "transition_cache_size": len(getattr(self, "transition_cache", {})),
        "k2_calls": d["k2_calls"],
        "total_threshold_k2_calls": d["k2_calls"],
        "k2_nodes": int(getattr(getattr(self, "k2", None), "nodes", 0)),
        "k2_success_hits": d["k2_success_hits"],
        "k2_fail_hits": d["k2_fail_hits"],
        "k2_wall": float(getattr(getattr(self, "k2", None), "wall", 0.0)),
        "total_wall": wall,
        "cut": cut,
        "cut_reason": cut_reason,
        "support_builds": support_builds,
        "exploration_mode": True,
    }


def _aggregate_predicates(round_results):
    ps = [r.get("predicate") for r in round_results if r.get("predicate")]
    if not ps:
        return None
    return {
        "lower_budget": min(int(p.get("lower_budget", 0)) for p in ps),
        "attempt_count": sum(int(p.get("attempt_count", 0)) for p in ps),
        "total_nodes": sum(int(p.get("total_nodes", 0)) for p in ps),
        "total_terminals": sum(int(p.get("total_terminals", 0)) for p in ps),
        "total_domain_queries": sum(
            int(p.get("total_domain_queries", 0)) for p in ps
        ),
        "total_threshold_k2_calls": sum(
            int(p.get("total_threshold_k2_calls", 0)) for p in ps
        ),
        "total_wall_internal": sum(
            float(p.get("total_wall_internal", 0.0)) for p in ps
        ),
    }


def _aggregate_oracles(round_results):
    stats = [r.get("oracle_stats") for r in round_results if r.get("oracle_stats")]
    if not stats:
        return None
    sum_keys = ("hits", "misses", "builds", "evictions")
    out = {k: sum(int(s.get(k, 0)) for s in stats) for k in sum_keys}
    out["extended_max_depth"] = max(
        int(s.get("extended_max_depth", 0)) for s in stats
    )
    out["cache_size"] = int(stats[-1].get("cache_size", 0))
    out["max_cache_size"] = max(
        int(s.get("max_cache_size", 0)) for s in stats
    )
    out["cache_max_entries"] = max(
        int(s.get("cache_max_entries", 0)) for s in stats
    )
    return out


def _make_fast_solve(orig_solve):
    @functools.wraps(orig_solve)
    def fast_solve(self, *args, **kwargs):
        global _LAST_PROBE

        # v37.23 audit calls solve(..., upper_witness=upper).  Leave any other
        # use untouched rather than guessing its calling contract.
        if "upper_witness" not in kwargs:
            return orig_solve(self, *args, **kwargs)

        current = dict(kwargs["upper_witness"])
        initial_upper = int(current["total"])

        round_results = []
        round_meta = []
        improving_witnesses = []
        stop_reason = "max_witnesses"

        for round_index in range(1, _FAST.max_witnesses + 1):
            _LAST_PROBE = {}

            call_kwargs = dict(kwargs)
            call_kwargs["upper_witness"] = current
            result = orig_solve(self, *args, **call_kwargs)
            round_results.append(result)

            probe = dict(_LAST_PROBE)
            before_total = int(current["total"])
            candidate = result.get("witness")
            candidate_total = (
                int(candidate["total"])
                if candidate is not None and "total" in candidate
                else before_total
            )
            improved = bool(
                result.get("improvement_found")
                and candidate is not None
                and candidate_total < before_total
            )

            round_meta.append({
                "round": round_index,
                "upper_before": before_total,
                "strict_budget": before_total - 1,
                "candidate_total": candidate_total,
                "improved": improved,
                "probe": probe,
            })

            if improved:
                current = dict(candidate)
                improving_witnesses.append(dict(candidate))

                if int(current["total"]) <= 0:
                    stop_reason = "solved_zero"
                    break

                # Descend immediately with the tighter U-1 domain.
                continue

            if probe.get("cut"):
                stop_reason = f"cut:{probe.get('cut_reason')}"
            else:
                stop_reason = "no_better_witness_in_bounded_probe"
            break

        if not round_results:
            return orig_solve(self, *args, **kwargs)

        final = dict(round_results[-1])

        # Never let this overlay claim exact optimality.  The point is rapid
        # mathematical exploration; exact certification returns later.
        final["witness"] = current
        final["certified"] = False
        final["improvement_found"] = bool(improving_witnesses)
        final["exploration_mode"] = True
        final["initial_upper_total"] = initial_upper
        final["final_best_total"] = int(current["total"])
        final["witness_count"] = len(improving_witnesses)
        final["witnesses"] = improving_witnesses
        final["exploration_rounds"] = round_meta
        final["exploration_stop_reason"] = stop_reason
        final["fast_cut"] = stop_reason.startswith("cut:")
        final["fast_cut_reason"] = (
            stop_reason.split(":", 1)[1]
            if stop_reason.startswith("cut:")
            else None
        )
        final["wall"] = sum(float(r.get("wall", 0.0)) for r in round_results)
        final["predicate"] = _aggregate_predicates(round_results)
        final["oracle_stats"] = _aggregate_oracles(round_results)

        # Preserve the latest derived-domain telemetry, but never preserve an
        # exact-proof source label from v37.23 on a bounded miss.
        if not improving_witnesses and isinstance(final.get("witness"), dict):
            w = dict(final["witness"])
            if "source" in w:
                w["source"] = "UPPER_RETAINED_AFTER_BOUNDED_EXPLORATION"
            final["witness"] = w

        return final

    return fast_solve


def _install_overlay(grc):
    # 1) Replace only the exact-minimum entry point.  find(), transitions(),
    #    requirement domains, d6 oracle, K2 oracle, etc. stay unchanged.
    memo_cls = grc.MemoTotalBudgetSearch
    memo_cls.exact_min_total = _fast_exact_min_total

    # 2) Wrap the v37.23 solver class(es) that own solve(), so every strict
    #    improvement causes a new U-1 requirement domain, up to max_witnesses.
    wrapped = []
    for name, cls in inspect.getmembers(grc, inspect.isclass):
        if cls.__module__ != grc.__name__:
            continue
        if "solve" not in cls.__dict__:
            continue
        orig = cls.__dict__["solve"]
        cls.solve = _make_fast_solve(orig)
        wrapped.append(name)

    if not wrapped:
        raise RuntimeError(
            "v37.23 overlay could not find a local solver class with solve(); "
            "the source layout may have changed."
        )
    return wrapped


def _parse_fast_args(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--fast-max-witnesses", type=int, default=5)
    ap.add_argument("--fast-probe-time-limit", type=float, default=3.0)
    ap.add_argument("--fast-probe-node-cap", type=int, default=50_000)
    ap.add_argument("--fast-support-build-cap", type=int, default=100_000)
    ns, remaining = ap.parse_known_args(argv)

    ns.fast_max_witnesses = max(1, min(int(ns.fast_max_witnesses), 5))
    ns.fast_probe_time_limit = max(0.05, float(ns.fast_probe_time_limit))
    ns.fast_probe_node_cap = max(1, int(ns.fast_probe_node_cap))
    ns.fast_support_build_cap = max(1, int(ns.fast_support_build_cap))
    return ns, remaining


class _Config:
    pass


def main() -> int:
    global _FAST

    fast, remaining = _parse_fast_args(sys.argv[1:])

    # These imports resolve from the same scripts directory when this file is
    # placed in CubeLab/scripts.
    import global_requirement_closure_v37_23 as grc
    import audit_global_requirement_closure_v37_23 as audit23

    cfg = _Config()
    cfg.grc = grc
    cfg.max_witnesses = fast.fast_max_witnesses
    cfg.probe_time_limit = fast.fast_probe_time_limit
    cfg.probe_node_cap = fast.fast_probe_node_cap
    cfg.support_build_cap = fast.fast_support_build_cap
    _FAST = cfg

    wrapped = _install_overlay(grc)

    # Hand only the original v37.23 arguments to its audit parser.
    sys.argv = [sys.argv[0], *remaining]

    print("# CubeLab v37.23.1 - FAST GLOBAL REQUIREMENT EXPLORATION")
    print("mode                 : ANYTIME / NO OPTIMALITY CLAIM")
    print("max improving witness:", cfg.max_witnesses)
    print("time/probe           :", f"{cfg.probe_time_limit:.3f}s")
    print("node cap/probe       :", f"{cfg.probe_node_cap:,}")
    print("support build cap    :", f"{cfg.support_build_cap:,}")
    print("wrapped solver       :", ",".join(wrapped))
    print("exact v37.23 files   : UNCHANGED")
    print()

    rc = audit23.main()

    print()
    print("# FAST-MODE NOTE")
    print(
        "A parity mismatch or nonzero audit status can mean only that the "
        "bounded exploration did not reach the exact reference."
    )
    print(
        "This overlay never upgrades a bounded miss to UNSAT and never claims "
        "shortest-solution certification."
    )
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
