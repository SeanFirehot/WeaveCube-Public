#!/usr/bin/env python3
"""CubeLab v37.23 - budget-derived Global Requirement Closure.

Research-only. Production changes: NONE.

Core idea
=========
Do not select an outer H depth.

Given ANY valid upper solution of total cost U:

    any strictly better solution has total <= U-1

and therefore necessarily:

    H_length <= U-1

So the total-cost requirement itself derives the COMPLETE exact H-depth domain:

    D(U) = {1,2,...,U-1}

Then ask one exact question:

    Exists / min total H+K2 within U-1 ?

using:
- compiled exact d6 Orientation support as the base;
- on-demand exact support recursion above d6;
- v37.16 exact unified total-budget predicate;
- exact boundary-compatible K2.

If no witness exists below U:
    U is certified exact.

If a witness exists:
    v37.16 searches budgets from its admissible lower bound upward, so the
    first success is already the exact global optimum below U.

There is no:
- fixed outer H depth;
- global OPEN heap;
- Global DFBnB;
- fixed H_N.

The total-cost requirement directly generates the needed H search domain.
"""
from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace
import time

import search_twist_skeleton as ts

from p1_orientation_domains_v37_0 import MOVE_INDEX
from audit_memoized_total_budget_v37_16 import MemoTotalBudgetSearch


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


class BudgetExtendedOrientationOracle:
    """Exact d6 support extended on-demand to a budget-derived horizon.

    Cache eviction is exactness-neutral; an evicted support relation is simply
    recomputed later.
    """

    def __init__(
        self,
        base_d6,
        *,
        max_depth,
        cache_max_entries=200000,
    ):
        self.base = base_d6
        self.co = base_d6.co
        self.eo = base_d6.eo
        self.sl = base_d6.sl
        self.table = SimpleNamespace(max_depth=int(max_depth))

        self.cache_max_entries = max(
            0, int(cache_max_entries)
        )
        self.cache = OrderedDict()

        self.hits = 0
        self.misses = 0
        self.builds = 0
        self.evictions = 0
        self.max_cache_size = 0

    def _cache_get(self, key):
        if self.cache_max_entries <= 0:
            return None
        hit = self.cache.get(key)
        if hit is None:
            return None
        self.hits += 1
        self.cache.move_to_end(key)
        return hit

    def _cache_put(self, key, value):
        if self.cache_max_entries <= 0:
            return

        self.cache[key] = value
        self.cache.move_to_end(key)

        while len(self.cache) > self.cache_max_entries:
            self.cache.popitem(last=False)
            self.evictions += 1

        self.max_cache_size = max(
            self.max_cache_size,
            len(self.cache),
        )

    def allowed_domain(
        self,
        q,
        remaining,
        previous_face,
    ):
        q = int(q)
        remaining = int(remaining)

        if remaining <= int(self.base.table.max_depth):
            return self.base.allowed_domain(
                q,
                remaining,
                previous_face,
            )

        if remaining < 1 or remaining > int(self.table.max_depth):
            return MoveDomain(())

        key = (
            q,
            remaining,
            previous_face,
        )

        hit = self._cache_get(key)
        if hit is not None:
            return hit

        self.misses += 1

        moves = []
        for move in ts.MOVE_ORDER:
            if not ts.allow(previous_face, move):
                continue

            mi = MOVE_INDEX[move]
            nq = int(ts.q_move(
                q,
                mi,
                self.co,
                self.eo,
                self.sl,
            ))

            child = self.allowed_domain(
                nq,
                remaining - 1,
                move[0],
            )

            if child.size() > 0:
                moves.append(move)

        value = MoveDomain(moves)
        self._cache_put(key, value)
        self.builds += 1
        return value

    def stats(self):
        return {
            "extended_max_depth": int(self.table.max_depth),
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "cache_max_entries": self.cache_max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "builds": self.builds,
            "evictions": self.evictions,
        }


class GlobalRequirementClosure:
    """Exact total-budget-derived Global H/K2 closure."""

    def __init__(
        self,
        tables,
        base_d6,
        *,
        p2_order,
        p2_probe,
        support_cache_max,
    ):
        self.tables = tables
        self.base_d6 = base_d6
        self.p2_order = p2_order
        self.p2_probe = int(p2_probe)
        self.support_cache_max = int(support_cache_max)

    @staticmethod
    def _valid_upper(start, upper):
        if upper is None:
            return False
        word = (
            tuple(upper["h_tail"])
            + tuple(upper["k2_word"])
        )
        return (
            int(upper["total"]) == len(word)
            and start.apply_word(word).is_solved()
        )

    def solve(
        self,
        start,
        *,
        upper_witness,
    ):
        if not self._valid_upper(start, upper_witness):
            raise RuntimeError(
                "GlobalRequirementClosure requires a valid upper witness "
                "whose total equals its word length"
            )

        upper_total = int(upper_witness["total"])

        # A solved cube has no improving positive-length solution.
        if upper_total <= 0:
            return {
                "witness": dict(upper_witness),
                "upper_total": upper_total,
                "strict_budget": -1,
                "derived_h_depths": [],
                "improvement_found": False,
                "certified": True,
                "predicate": None,
                "oracle_stats": None,
                "wall": 0.0,
            }

        strict_budget = upper_total - 1

        # Any strictly better solution has H <= total <= strict_budget.
        max_h = max(
            int(self.base_d6.table.max_depth),
            strict_budget,
        )

        oracle = BudgetExtendedOrientationOracle(
            self.base_d6,
            max_depth=max_h,
            cache_max_entries=self.support_cache_max,
        )

        # H must consume at least one move for this H->K2 factorization.
        # If strict_budget==0, no positive H can improve the upper witness.
        depths = tuple(range(
            1,
            strict_budget + 1,
        ))

        t0 = time.perf_counter()

        if not depths:
            predicate_result = None
            found = None
        else:
            solver = MemoTotalBudgetSearch(
                self.tables,
                oracle,
                p2_order=self.p2_order,
                p2_probe=self.p2_probe,
            )

            predicate_result = solver.exact_min_total(
                start,
                int(ts.q_of(start)),
                None,
                depths,
                strict_budget,
            )
            found = predicate_result["witness"]

        wall = time.perf_counter() - t0

        if found is None:
            best = {
                "total": upper_total,
                "h_tail": tuple(upper_witness["h_tail"]),
                "k2_word": tuple(upper_witness["k2_word"]),
                "k2_exact": int(upper_witness["k2_exact"]),
                "source": "UPPER_CERTIFIED_BY_GLOBAL_REQUIREMENT_NO",
            }
            improvement_found = False
        else:
            best = {
                "total": int(found["total"]),
                "h_tail": tuple(found["h_tail"]),
                "k2_word": tuple(found["k2_word"]),
                "k2_exact": int(found["k2_exact"]),
                "source": "GLOBAL_REQUIREMENT_EXACT_FIRST_SUCCESS",
            }
            improvement_found = True

        word = (
            tuple(best["h_tail"])
            + tuple(best["k2_word"])
        )
        if not start.apply_word(word).is_solved():
            raise RuntimeError(
                "GlobalRequirementClosure final witness replay failed"
            )

        # Because exact_min_total tests total budgets in ascending order and
        # depths contains every possible H length <= strict budget:
        # - found => first success is exact optimum;
        # - not found => upper witness is exact optimum.
        certified = True

        pred_summary = None
        if predicate_result is not None:
            pred_summary = {
                "lower_budget": int(
                    predicate_result["lower_budget"]
                ),
                "attempt_count": len(
                    predicate_result["attempts"]
                ),
                "total_nodes": int(
                    predicate_result["total_nodes"]
                ),
                "total_terminals": int(
                    predicate_result["total_terminals"]
                ),
                "total_domain_queries": int(
                    predicate_result["domain_queries"]
                ),
                "total_threshold_k2_calls": int(
                    predicate_result["k2_calls"]
                ),
                "total_wall_internal": float(
                    predicate_result["total_wall"]
                ),
            }

        return {
            "witness": best,
            "upper_total": upper_total,
            "strict_budget": strict_budget,
            "derived_h_depths": list(depths),
            "derived_h_max": (
                None if not depths else max(depths)
            ),
            "improvement_found": improvement_found,
            "certified": certified,
            "predicate": pred_summary,
            "oracle_stats": oracle.stats(),
            "wall": wall,
        }
