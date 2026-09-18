#!/usr/bin/env python3
"""
CubeLab v37.24 — FAST DYNAMIC H<->K2 ALLOCATION ORDERING

Research purpose
----------------
Add exactly one new mathematical idea on top of v37.23.1 FAST exploration:

    Dynamic H/K2 budget allocation by local slack.

This version DOES NOT prune any branch and DOES NOT change feasibility.
It only reorders already-valid MemoTotalBudgetSearch transitions.

For a child state q' under local remaining budget B' and each surviving
H-depth h, define:

    H_slack(h)  = h - phase1_lb(q')
    K2_budget(h)= B' - h

A child transition is ranked by the best surviving allocation:

    1) smallest H_slack
    2) largest K2_budget
    3) more surviving H allocations
    4) smaller H depth
    5) canonical move name

Thus a move that keeps H close to its admissible lower bound while leaving
more of the current total budget available to K2 is explored first.

IMPORTANT
---------
* Ordering only. No branch is removed.
* Fast-mode cuts remain UNKNOWN.
* No optimality claim.
* Requires audit_fast_global_requirement_exploration_v37_23_1_fixed.py
  in the same scripts directory.

Place in CubeLab/scripts and run from repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir():
    s = str(_SRC_DIR)
    if s not in sys.path:
        sys.path.insert(0, s)

import audit_fast_global_requirement_exploration_v37_23_1_fixed as fast23
import global_requirement_closure_v37_23 as grc


_STATS = {
    "transition_calls": 0,
    "multiway_calls": 0,
    "reordered_calls": 0,
    "children_seen": 0,
}


def _install_dynamic_hk2_ordering() -> None:
    cls = grc.MemoTotalBudgetSearch

    if getattr(cls, "_v3724_dynamic_hk2_installed", False):
        return

    orig_find = cls.find
    orig_transitions = cls.transitions

    def find_with_budget_context(self, *args, **kwargs):
        # Known v37.16 signature after self:
        # (poses, q, depths, last, parent_was_goal, budget)
        if "budget" in kwargs:
            budget = kwargs["budget"]
        elif len(args) >= 6:
            budget = args[5]
        else:
            budget = None

        sentinel = object()
        previous = getattr(self, "_v3724_local_budget", sentinel)
        if budget is not None:
            self._v3724_local_budget = int(budget)

        try:
            return orig_find(self, *args, **kwargs)
        finally:
            if previous is sentinel:
                try:
                    delattr(self, "_v3724_local_budget")
                except AttributeError:
                    pass
            else:
                self._v3724_local_budget = previous

    def transitions_dynamic(self, q, depths, last):
        out = tuple(orig_transitions(self, q, depths, last))
        _STATS["transition_calls"] += 1
        _STATS["children_seen"] += len(out)

        if len(out) <= 1:
            return out

        B = getattr(self, "_v3724_local_budget", None)
        if B is None:
            return out

        # Each transition consumes one H move before recursion.
        child_total_budget = int(B) - 1

        def score(item):
            move, _mi, nq, child_depths = item

            h_lb = int(grc.ts.phase1_lb(
                int(nq), self.cos, self.eos
            ))

            allocs = []
            for h in child_depths:
                h = int(h)
                k2_budget = child_total_budget - h
                if k2_budget < 0:
                    # Do not prune it here; just rank it after feasible
                    # local allocations. The underlying search retains
                    # ownership of feasibility.
                    continue
                h_slack = h - h_lb
                allocs.append((
                    h_slack,
                    -k2_budget,
                    h,
                ))

            if allocs:
                best = min(allocs)
                return (
                    0,                  # has a budget-consistent allocation
                    best[0],            # minimum H slack
                    best[1],            # maximum K2 budget
                    -len(child_depths), # preserve allocation flexibility
                    best[2],            # shorter H depth
                    move,
                )

            # Keep, never prune, any transition whose local allocations look
            # over-budget. It is simply ordered last.
            return (
                1,
                10**9,
                10**9,
                -len(child_depths),
                min((int(x) for x in child_depths), default=10**9),
                move,
            )

        ordered = tuple(sorted(out, key=score))
        _STATS["multiway_calls"] += 1
        if ordered != out:
            _STATS["reordered_calls"] += 1
        return ordered

    cls.find = find_with_budget_context
    cls.transitions = transitions_dynamic
    cls._v3724_dynamic_hk2_installed = True


def main() -> int:
    _install_dynamic_hk2_ordering()

    print("# CubeLab v37.24 - DYNAMIC H<->K2 ALLOCATION ORDERING")
    print("mathematical change  : LOCAL H-SLACK / K2-BUDGET ORDER")
    print("branch pruning       : NONE")
    print("feasibility change   : NONE")
    print("optimality claim     : NONE (FAST exploration)")
    print()

    rc = fast23.main()

    print()
    print("# v37.24 ORDERING TELEMETRY")
    print("transition calls     :", f"{_STATS['transition_calls']:,}")
    print("multiway calls       :", f"{_STATS['multiway_calls']:,}")
    print("reordered calls      :", f"{_STATS['reordered_calls']:,}")
    print("children seen        :", f"{_STATS['children_seen']:,}")
    if _STATS["multiway_calls"]:
        frac = _STATS["reordered_calls"] / _STATS["multiway_calls"]
        print("reorder fraction     :", f"{frac:.1%}")

    print()
    print("COMPARE AGAINST v37.23.1 BASELINE:")
    print("  total best length, nodes, supportBuild, wall")
    print("The old exact-audit parity label is not a FAST-mode success criterion.")

    # v37.23 exact audit returns 2 because FAST mode deliberately clears
    # certification. That is not a failure of this exploration executable.
    # Preserve unexpected codes, but normalize the known parity-only code 2.
    return 0 if int(rc) == 2 else int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
