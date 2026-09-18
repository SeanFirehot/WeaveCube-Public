#!/usr/bin/env python3
"""Exact phase-2 helpers with adaptive double-star column PDB.

The 12 retained double-star projections are exact K-subproblem distances, so

    max(old_P2_PDB, max_double_star_distance)

is admissible.  The full 12-distance profile is ordering-only.

v16 changes ``auto`` from a root-threshold guess to a small *legacy probe*:
legacy phase-2 is allowed to search a bounded number of nodes first.  If it
solves inside that budget, no double-star query is paid.  If the budget is
exhausted, search restarts exactly with the stronger double-star bound/profile.
Correctness is unchanged because the probe never prunes the exact follow-up.
"""
from __future__ import annotations

import search_twist_skeleton as ts
import pdcc_double_star as ds

AUTO_PROBE_NODES = 128
_ABORT = object()


def configure_double_star_cache(cache_path=None, index_path=None):
    ds.configure(cache_path, index_path)


def preload_double_star():
    """Load retained static tables/index without retaining state-query results."""
    ds._ensure_loaded()


def clear_double_star_query_cache():
    """Clear only the per-state LRU profile cache (static tables stay loaded)."""
    ds.profile.cache_clear()


def lower_bound(key, cpd, upd, spd, *, use_double_star: bool = False):
    old = ts.phase2_lb(key, cpd, upd, spd)
    if not use_double_star or key == ts.GOAL_P:
        return old
    return max(old, ds.lower_bound(key))


def _exact_search(
    key, prev_face, max_depth, cp, up, sp, cpd, upd, spd,
    *, use_double_star: bool, stats: dict | None = None,
):
    """Exact shortest K suffix.  No activation policy is decided here."""
    def hit_node():
        if stats is not None:
            stats["nodes"] = stats.get("nodes", 0) + 1

    if key == ts.GOAL_P:
        return (), 0
    lb = lower_bound(key, cpd, upd, spd, use_double_star=use_double_star)
    if lb > max_depth:
        return None, None

    path: list[str] = []
    for limit in range(lb, max_depth + 1):
        failed: set[tuple[int, int, str | None]] = set()

        def dfs(x: int, remaining: int, last_face: str | None):
            hit_node()
            h = lower_bound(x, cpd, upd, spd, use_double_star=use_double_star)
            if h > remaining:
                return None
            if x == ts.GOAL_P:
                return tuple(path)
            if remaining == 0:
                return None
            sig = (x, remaining, last_face)
            if sig in failed:
                return None

            children = []
            for mi, move in enumerate(ts.K_MOVES):
                if not ts.allow(last_face, move):
                    continue
                y = ts.p_move(x, mi, cp, up, sp)
                hy = lower_bound(y, cpd, upd, spd, use_double_star=use_double_star)
                if hy > remaining - 1:
                    continue
                # The full profile is an ordering key only.  hy remains the
                # admissible prune key in either arm.
                key2 = (ds.profile(y), hy, move) if use_double_star else (hy, move)
                children.append((key2, move, y))
            children.sort(key=lambda z: z[0])
            for _, move, y in children:
                path.append(move)
                ans = dfs(y, remaining - 1, move[0])
                if ans is not None:
                    return ans
                path.pop()
            failed.add(sig)
            return None

        ans = dfs(key, limit, prev_face)
        if ans is not None:
            return ans, len(ans)
    return None, None


def _legacy_probe(
    key, prev_face, max_depth, cp, up, sp, cpd, upd, spd,
    *, node_budget: int,
):
    """Run the exact legacy IDA order, but abort after ``node_budget`` nodes.

    Returns ``(word, dist, nodes, aborted)``.  If ``aborted`` is False the
    result is already exact (solution or exhaustive no-solution within cap).
    """
    nodes = 0
    if key == ts.GOAL_P:
        return (), 0, 0, False
    lb = ts.phase2_lb(key, cpd, upd, spd)
    if lb > max_depth:
        return None, None, 0, False

    path: list[str] = []
    for limit in range(lb, max_depth + 1):
        failed: set[tuple[int, int, str | None]] = set()

        def dfs(x: int, remaining: int, last_face: str | None):
            nonlocal nodes
            if nodes >= node_budget:
                return _ABORT
            nodes += 1

            h = ts.phase2_lb(x, cpd, upd, spd)
            if h > remaining:
                return None
            if x == ts.GOAL_P:
                return tuple(path)
            if remaining == 0:
                return None
            sig = (x, remaining, last_face)
            if sig in failed:
                return None

            children = []
            for mi, move in enumerate(ts.K_MOVES):
                if not ts.allow(last_face, move):
                    continue
                y = ts.p_move(x, mi, cp, up, sp)
                hy = ts.phase2_lb(y, cpd, upd, spd)
                if hy > remaining - 1:
                    continue
                children.append(((hy, move), move, y))
            children.sort(key=lambda z: z[0])
            for _, move, y in children:
                path.append(move)
                ans = dfs(y, remaining - 1, move[0])
                path.pop()
                if ans is _ABORT:
                    return _ABORT
                if ans is not None:
                    return ans
            failed.add(sig)
            return None

        ans = dfs(key, limit, prev_face)
        if ans is _ABORT:
            return None, None, nodes, True
        if ans is not None:
            return ans, len(ans), nodes, False
    return None, None, nodes, False


def shortest_b_compatible(
    key, prev_face, max_depth, cp, up, sp, cpd, upd, spd,
    *, order: str = "auto", stats: dict | None = None,
    auto_probe_nodes: int = AUTO_PROBE_NODES,
):
    """Exact shortest K suffix with HTM boundary compatibility.

    baseline
        Legacy P2 PDB/order only.
    dstar
        Always use ``h+=max(old,dstar)`` plus double-star profile ordering.
    auto
        First run up to ``auto_probe_nodes`` nodes of the legacy exact search.
        If it solves/exhausts inside the probe, return that exact result without
        loading/querying double-star.  Otherwise restart exactly with dstar.

    The auto restart is deliberate: it buys a cheap empirical hardness test and
    never changes the admissible target or proof semantics.
    """
    if order not in {"baseline", "dstar", "auto"}:
        raise ValueError(order)

    if stats is not None:
        stats.clear()
        stats["used_double_star"] = False
        stats["probe_nodes"] = 0
        stats["dstar_nodes"] = 0
        stats["auto_switched"] = False

    if order == "baseline":
        local = {}
        ans = _exact_search(
            key, prev_face, max_depth, cp, up, sp, cpd, upd, spd,
            use_double_star=False, stats=local,
        )
        if stats is not None:
            stats["nodes"] = local.get("nodes", 0)
        return ans

    if order == "dstar":
        local = {}
        ans = _exact_search(
            key, prev_face, max_depth, cp, up, sp, cpd, upd, spd,
            use_double_star=True, stats=local,
        )
        if stats is not None:
            stats["used_double_star"] = True
            stats["dstar_nodes"] = local.get("nodes", 0)
            stats["nodes"] = local.get("nodes", 0)
        return ans

    # auto: a bounded legacy probe is the activation test.
    probe_budget = max(1, int(auto_probe_nodes))
    word, dist, pnodes, aborted = _legacy_probe(
        key, prev_face, max_depth, cp, up, sp, cpd, upd, spd,
        node_budget=probe_budget,
    )
    if stats is not None:
        stats["probe_nodes"] = pnodes

    if not aborted:
        if stats is not None:
            stats["nodes"] = pnodes
        return word, dist

    local = {}
    ans = _exact_search(
        key, prev_face, max_depth, cp, up, sp, cpd, upd, spd,
        use_double_star=True, stats=local,
    )
    dnodes = local.get("nodes", 0)
    if stats is not None:
        stats["used_double_star"] = True
        stats["auto_switched"] = True
        stats["dstar_nodes"] = dnodes
        stats["nodes"] = pnodes + dnodes
    return ans
