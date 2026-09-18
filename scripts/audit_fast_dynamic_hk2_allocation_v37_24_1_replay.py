#!/usr/bin/env python3
"""
CubeLab v37.24.1 — FAST DYNAMIC H<->K2 ALLOCATION + WITNESS REPLAY GUARD

This is a verification wrapper for v37.24.

It keeps the v37.24 dynamic H/K2 ordering unchanged, but every final witness
returned by FAST Global Requirement Exploration is replayed on the actual
start state before the audit can continue.

If a witness:
  * does not solve the cube, or
  * reports a total different from len(h_tail)+len(k2_word)

the run aborts immediately.

No search semantics are changed.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir():
    s = str(_SRC_DIR)
    if s not in sys.path:
        sys.path.insert(0, s)

import audit_fast_dynamic_hk2_allocation_v37_24 as v24


_REPLAY_STATS = {
    "checked": 0,
    "passed": 0,
}


_original_make_fast_solve = v24.fast23._make_fast_solve


def _verified_make_fast_solve(orig_solve):
    fast_solve = _original_make_fast_solve(orig_solve)

    @functools.wraps(fast_solve)
    def verified(self, *args, **kwargs):
        result = fast_solve(self, *args, **kwargs)

        if "upper_witness" not in kwargs or not args:
            return result

        start = args[0]
        witness = result.get("witness")
        if not isinstance(witness, dict):
            return result

        h_tail = tuple(witness.get("h_tail", ()))
        k2_word = tuple(witness.get("k2_word", ()))
        word = h_tail + k2_word

        reported_total = int(witness.get("total", len(word)))
        actual_total = len(word)

        solved = bool(start.apply_word(word).is_solved())

        _REPLAY_STATS["checked"] += 1

        print(
            "[REPLAY] "
            f"reported={reported_total} "
            f"actual={actual_total} "
            f"H={len(h_tail)} "
            f"K2={len(k2_word)} "
            f"solved={solved} "
            f"word={' '.join(word) if word else '<empty>'}"
        )

        if reported_total != actual_total:
            raise RuntimeError(
                "FAST witness total mismatch: "
                f"reported={reported_total}, actual={actual_total}, "
                f"word={word}"
            )

        if not solved:
            raise RuntimeError(
                "FAST witness replay FAILED: "
                f"reported_total={reported_total}, word={word}"
            )

        _REPLAY_STATS["passed"] += 1

        # Add explicit verification telemetry to the returned result.
        result = dict(result)
        result["replay_verified"] = True
        result["replay_total"] = actual_total
        result["replay_word"] = list(word)
        return result

    return verified


def main() -> int:
    # fast23.main() installs the solve wrapper later. Replace only its wrapper
    # factory now so the installed FAST solve gains replay validation.
    v24.fast23._make_fast_solve = _verified_make_fast_solve

    print("# CubeLab v37.24.1 - WITNESS REPLAY GUARD")
    print("search change         : NONE")
    print("v37.24 ordering       : UNCHANGED")
    print("witness replay        : REQUIRED")
    print()

    rc = v24.main()

    print()
    print("# v37.24.1 REPLAY SUMMARY")
    print("witnesses checked     :", _REPLAY_STATS["checked"])
    print("witnesses passed      :", _REPLAY_STATS["passed"])
    if _REPLAY_STATS["checked"] != _REPLAY_STATS["passed"]:
        return 3

    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
