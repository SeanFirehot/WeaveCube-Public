#!/usr/bin/env python3
"""
CubeLab v37.114 — SELECTIVE H3/H4 CONSTRUCTIVE COLUMN TERMINAL MACRO

Motivation
----------
v37.110:
    exact q-tight shared-tail feasibility removed NOTHING.

v37.111:
    exact shared-tail multiplicity changed the retained v37.24 order in
    essentially no contexts.

Therefore stop using the R<=4 compiled shared-word oracle as:
    * a feasibility filter, or
    * a branch-order score.

Use its strongest proven capability instead:

    construct the COMPLETE q-tight H suffix in one operation.

This is a speculative fast path only.

At find(poses, q, depths, last, parent_was_goal, budget):

    R = phase1_lb(q)

Only when:
    1 <= R <= 4
    R is one of the surviving H depths
    budget >= R

query the exact 20-piece shared-word oracle for complete reduced words of
length R.

For each exact shared suffix (up to --macro-max-suffixes):
    1. replay q exactly through the suffix;
    2. require final q == GOAL_Q;
    3. require the final column to be a fresh DR entry
       (q before the final move != GOAL_Q);
    4. call the EXISTING MemoTotalBudgetSearch.find at H-depth 0 with the
       remaining total budget;
    5. if the existing terminal/K2 machinery returns a witness, prepend the
       whole H suffix and return immediately.

If no speculative macro succeeds:
    FALL BACK TO THE ORIGINAL v37.24 find UNCHANGED.

Safety
------
* NO pruning.
* NO feasibility rule is strengthened.
* NO H-depth slot is removed.
* NO transition is removed.
* A macro miss is NOT learned as UNSAT.
* Slack-H allocations remain available through fallback.
* Existing FAST cuts propagate normally.
* v37.24 Dynamic H/K2 ordering remains the fallback search.
* v37.24.1 full witness replay remains mandatory.
* Production changes: NONE.

This experiment tests whether the compiled column+20-piece superposition is
useful as a constructive subtree replacement in the actual stable variable-H
search — the role in which v37.87 was successful in first-witness microbenchmarks.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if SRC.is_dir():
    s = str(SRC)
    if s not in sys.path:
        sys.path.insert(0, s)


import audit_fast_dynamic_hk2_allocation_v37_24 as v24
import audit_fast_dynamic_hk2_allocation_v37_24_1_replay as replay24
import audit_fast_broad_compiled_column_v37_84 as v84
import audit_fast_local_dr_superposition_v37_57 as v57
import search_pdcc_guided as pg


_ORACLE = None
_ORACLE_SOURCE = None
_ORACLE_WALL = 0.0

_MACRO_MAX_SUFFIXES = 2
_MACRO_MIN_HQ = 3
_MACRO_MAX_HQ = 4

_STATS = {
    "find_calls": 0,
    "eligible_calls": 0,
    "in_range_qtight_calls": 0,
    "below_range_skips": 0,
    "above_range_skips": 0,
    "non_qtight_or_budget_skips": 0,
    "oracle_calls": 0,
    "oracle_empty": 0,
    "oracle_nonempty": 0,
    "oracle_word_count_sum": 0,
    "oracle_word_count_max": 0,

    "suffix_words_considered": 0,
    "suffix_q_rejects": 0,
    "suffix_parent_goal_skips": 0,

    "terminal_calls": 0,
    "terminal_hits": 0,
    "terminal_misses": 0,

    "macro_hits": 0,
    "fallback_calls": 0,

    "contract_term_h_nonempty": 0,
    "contract_term_total_mismatch": 0,
    "contract_over_budget": 0,
}

_H_HIST = Counter()
_H_SKIP_HIST = Counter()
_H_HIT_HIST = Counter()
_H_TERMINAL_CALL_HIST = Counter()
_WORDCOUNT_HIST = Counter()
_TERMINAL_K2_HIST = Counter()
_TERMINAL_CALL_RANK_HIST = Counter()
_TERMINAL_HIT_RANK_HIST = Counter()
_MACRO_HIT_RANK_HIST = Counter()
_MACRO_HIT_EVENTS = []


def _iter_set_bits(mask):
    m = int(mask)
    while m:
        low = m & -m
        idx = low.bit_length() - 1
        yield idx
        m ^= low


def _extract_find_args(args, kwargs):
    """
    Known MemoTotalBudgetSearch signature after self:
        find(poses, q, depths, last, parent_was_goal, budget)
    """
    poses = kwargs.get("poses")
    q = kwargs.get("q")
    depths = kwargs.get("depths")

    last = kwargs.get("last")
    if last is None:
        last = kwargs.get("last_face")

    parent_was_goal = kwargs.get("parent_was_goal")
    budget = kwargs.get("budget")

    if poses is None and len(args) >= 1:
        poses = args[0]
    if q is None and len(args) >= 2:
        q = args[1]
    if depths is None and len(args) >= 3:
        depths = args[2]
    if last is None and len(args) >= 4:
        last = args[3]
    if parent_was_goal is None and len(args) >= 5:
        parent_was_goal = args[4]
    if budget is None and len(args) >= 6:
        budget = args[5]

    if poses is None or q is None or depths is None or budget is None:
        return None

    return (
        tuple(int(x) for x in poses),
        int(q),
        tuple(sorted(set(int(x) for x in depths))),
        last,
        parent_was_goal,
        int(budget),
    )


def _q_step(self, q, mi):
    return int(
        v24.grc.ts.q_move(
            int(q),
            int(mi),
            self.co,
            self.eo,
            self.sl,
        )
    )


def _pose_step(poses, mi):
    return tuple(
        int(x)
        for x in pg.move_poses(
            tuple(poses),
            int(mi),
        )
    )


def _suffix_terminal_state(self, poses, q, row):
    """
    Replay one complete oracle word.

    Returns:
        final_poses, final_q, q_before_final, final_face
    """
    cur_poses = tuple(poses)
    cur_q = int(q)
    q_before_final = int(q)
    final_face = None

    for j, raw_mi in enumerate(row):
        mi = int(raw_mi)

        if j == len(row) - 1:
            q_before_final = int(cur_q)

        cur_poses = _pose_step(
            cur_poses,
            mi,
        )

        cur_q = _q_step(
            self,
            cur_q,
            mi,
        )

        final_face = (
            v24.grc.ts.MOVE_ORDER[
                mi
            ][0]
        )

    return (
        cur_poses,
        int(cur_q),
        int(q_before_final),
        final_face,
    )


def _combine_macro_witness(
    *,
    suffix,
    terminal,
    budget,
):
    term_h = tuple(
        terminal.get(
            "h_tail",
            (),
        )
    )

    term_k2 = tuple(
        terminal.get(
            "k2_word",
            (),
        )
    )

    term_total = int(
        terminal.get(
            "total",
            len(term_h)
            + len(term_k2),
        )
    )

    actual_term_total = (
        len(term_h)
        + len(term_k2)
    )

    if term_h:
        _STATS[
            "contract_term_h_nonempty"
        ] += 1

        raise RuntimeError(
            "v37.112 terminal H=0 call returned nonempty h_tail: "
            f"{term_h}"
        )

    if term_total != actual_term_total:
        _STATS[
            "contract_term_total_mismatch"
        ] += 1

        raise RuntimeError(
            "v37.112 terminal witness total mismatch: "
            f"reported={term_total}, "
            f"actual={actual_term_total}"
        )

    total = (
        len(suffix)
        + term_total
    )

    if total > int(
        budget
    ):
        _STATS[
            "contract_over_budget"
        ] += 1

        raise RuntimeError(
            "v37.112 macro witness exceeds local budget: "
            f"total={total}, budget={budget}"
        )

    _TERMINAL_K2_HIST[
        len(term_k2)
    ] += 1

    out = dict(
        terminal
    )

    out[
        "total"
    ] = int(
        total
    )

    out[
        "h_tail"
    ] = tuple(
        suffix
    )

    out[
        "k2_word"
    ] = term_k2

    out[
        "k2_exact"
    ] = int(
        terminal.get(
            "k2_exact",
            len(term_k2),
        )
    )

    out[
        "source"
    ] = (
        "V37_112_QTIGHT_COLUMN_TERMINAL_MACRO"
    )

    out[
        "macro_h"
    ] = int(
        len(suffix)
    )

    out[
        "macro_terminal_total"
    ] = int(
        term_total
    )

    return out


def _install_terminal_macro():
    cls = (
        v24.grc.MemoTotalBudgetSearch
    )

    if getattr(
        cls,
        "_v37112_terminal_macro_installed",
        False,
    ):
        return

    # v37.24 has already wrapped find with its local-budget context.
    orig_find = cls.find

    def find_with_terminal_macro(
        self,
        *args,
        **kwargs,
    ):
        _STATS[
            "find_calls"
        ] += 1

        parsed = _extract_find_args(
            args,
            kwargs,
        )

        if parsed is None:
            _STATS[
                "fallback_calls"
            ] += 1

            return orig_find(
                self,
                *args,
                **kwargs,
            )

        (
            poses,
            q,
            depths,
            last,
            _parent_was_goal,
            budget,
        ) = parsed

        positive_depths = tuple(
            d
            for d in depths
            if d > 0
        )

        if not positive_depths:
            _STATS[
                "fallback_calls"
            ] += 1

            return orig_find(
                self,
                *args,
                **kwargs,
            )

        hq = int(
            v24.grc.ts.phase1_lb(
                int(q),
                self.cos,
                self.eos,
            )
        )

        # First preserve the exact v37.112 eligibility contract.
        if (
            hq < 1
            or hq > 4
            or hq not in positive_depths
            or budget < hq
        ):
            _STATS[
                "non_qtight_or_budget_skips"
            ] += 1

            _STATS[
                "fallback_calls"
            ] += 1

            return orig_find(
                self,
                *args,
                **kwargs,
            )

        _STATS[
            "in_range_qtight_calls"
        ] += 1

        # v37.114 single experimental change:
        # pay for the constructive macro only inside the configured Hq band.
        # Outside the band, immediately return ownership to stable v37.24.
        if hq < int(
            _MACRO_MIN_HQ
        ):
            _STATS[
                "below_range_skips"
            ] += 1

            _H_SKIP_HIST[
                hq
            ] += 1

            _STATS[
                "fallback_calls"
            ] += 1

            return orig_find(
                self,
                *args,
                **kwargs,
            )

        if hq > int(
            _MACRO_MAX_HQ
        ):
            _STATS[
                "above_range_skips"
            ] += 1

            _H_SKIP_HIST[
                hq
            ] += 1

            _STATS[
                "fallback_calls"
            ] += 1

            return orig_find(
                self,
                *args,
                **kwargs,
            )

        _STATS[
            "eligible_calls"
        ] += 1

        _H_HIST[
            hq
        ] += 1

        _STATS[
            "oracle_calls"
        ] += 1

        qr = _ORACLE.query(
            poses=poses,
            q=int(q),
            R=int(hq),
            last_face=last,
            add_q_first=False,
        )

        if not qr[
            "feasible"
        ]:
            # Speculative macro only.
            # NEVER turn this into a prune; fall back to stable search.
            _STATS[
                "oracle_empty"
            ] += 1

            _STATS[
                "fallback_calls"
            ] += 1

            return orig_find(
                self,
                *args,
                **kwargs,
            )

        _STATS[
            "oracle_nonempty"
        ] += 1

        word_count = int(
            qr.get(
                "word_count",
                int(
                    qr[
                        "word_mask"
                    ]
                ).bit_count(),
            )
        )

        _STATS[
            "oracle_word_count_sum"
        ] += word_count

        if (
            word_count
            > _STATS[
                "oracle_word_count_max"
            ]
        ):
            _STATS[
                "oracle_word_count_max"
            ] = word_count

        _WORDCOUNT_HIST[
            word_count
        ] += 1

        words = _ORACLE.words[
            int(hq)
        ]

        tested = 0

        for word_idx in _iter_set_bits(
            qr[
                "word_mask"
            ]
        ):
            if (
                tested
                >= int(
                    _MACRO_MAX_SUFFIXES
                )
            ):
                break

            tested += 1

            _STATS[
                "suffix_words_considered"
            ] += 1

            row = words[
                int(
                    word_idx
                )
            ]

            suffix = tuple(
                v24.grc.ts.MOVE_ORDER[
                    int(mi)
                ]
                for mi in row
            )

            (
                end_poses,
                end_q,
                q_before_final,
                final_face,
            ) = _suffix_terminal_state(
                self,
                poses,
                q,
                row,
            )

            if (
                int(end_q)
                != int(
                    v24.grc.ts.GOAL_Q
                )
            ):
                _STATS[
                    "suffix_q_rejects"
                ] += 1
                continue

            # Only use a genuine final DR-entry boundary in the macro.
            # If the parent of the final column is already DR, fallback search
            # retains the earlier-entry possibility exactly as before.
            if (
                int(q_before_final)
                == int(
                    v24.grc.ts.GOAL_Q
                )
            ):
                _STATS[
                    "suffix_parent_goal_skips"
                ] += 1
                continue

            remaining_budget = (
                int(budget)
                - int(hq)
            )

            if remaining_budget < 0:
                continue

            _STATS[
                "terminal_calls"
            ] += 1

            _H_TERMINAL_CALL_HIST[
                hq
            ] += 1

            _TERMINAL_CALL_RANK_HIST[
                tested
            ] += 1

            # IMPORTANT:
            # Call self.find rather than captured orig_find.
            #
            # During FAST probes, self.find is temporarily replaced by the
            # guard wrapper.  Calling self.find keeps time/node/support caps
            # active.  This re-enters this macro wrapper, but depths=(0,) makes
            # the macro ineligible and it immediately falls through to the
            # stable v37.24/base find.
            terminal = self.find(
                end_poses,
                int(end_q),
                (0,),
                final_face,
                False,
                int(
                    remaining_budget
                ),
            )

            if terminal is None:
                _STATS[
                    "terminal_misses"
                ] += 1
                continue

            _STATS[
                "terminal_hits"
            ] += 1

            _TERMINAL_HIT_RANK_HIST[
                tested
            ] += 1

            witness = (
                _combine_macro_witness(
                    suffix=suffix,
                    terminal=terminal,
                    budget=budget,
                )
            )

            _STATS[
                "macro_hits"
            ] += 1

            _H_HIT_HIST[
                hq
            ] += 1

            _MACRO_HIT_RANK_HIST[
                tested
            ] += 1

            _MACRO_HIT_EVENTS.append({
                "suffix_rank": int(
                    tested
                ),
                "hq": int(
                    hq
                ),
                "word_count": int(
                    word_count
                ),
                "local_budget": int(
                    budget
                ),
                "remaining_budget": int(
                    remaining_budget
                ),
                "terminal_total": int(
                    witness.get(
                        "macro_terminal_total",
                        0,
                    )
                ),
                "terminal_k2_len": len(
                    tuple(
                        witness.get(
                            "k2_word",
                            (),
                        )
                    )
                ),
                "suffix": list(
                    suffix
                ),
            })

            return witness

        # Macro is only a speculative constructor.
        # Any miss/cap/unsupported suffix returns ownership to the unchanged
        # stable search.
        _STATS[
            "fallback_calls"
        ] += 1

        return orig_find(
            self,
            *args,
            **kwargs,
        )

    cls.find = (
        find_with_terminal_macro
    )

    cls._v37112_terminal_macro_installed = (
        True
    )


_ORIGINAL_V24_INSTALLER = (
    v24._install_dynamic_hk2_ordering
)


def _combined_installer():
    _ORIGINAL_V24_INSTALLER()
    _install_terminal_macro()


def _parse_local_args(argv):
    ap = argparse.ArgumentParser(
        add_help=False
    )

    ap.add_argument(
        "--column-word-cache",
        default=(
            "reports/pdcc_cache/"
            "compiled_column_wordset_r4_v37_84.pkl"
        ),
    )

    ap.add_argument(
        "--macro-max-suffixes",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--macro-min-hq",
        type=int,
        default=3,
    )

    ap.add_argument(
        "--macro-max-hq",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--macro-telemetry-output",
        default=None,
    )

    ns, remaining = (
        ap.parse_known_args(
            argv
        )
    )

    ns.macro_max_suffixes = max(
        1,
        int(
            ns.macro_max_suffixes
        ),
    )

    ns.macro_min_hq = max(
        1,
        int(
            ns.macro_min_hq
        ),
    )

    ns.macro_max_hq = min(
        4,
        int(
            ns.macro_max_hq
        ),
    )

    if ns.macro_min_hq > ns.macro_max_hq:
        raise SystemExit(
            "--macro-min-hq must be <= --macro-max-hq"
        )

    return (
        ns,
        remaining,
    )


def _write_telemetry(
    path,
    *,
    rc,
):
    if path is None:
        path = (
            "reports/v37/"
            "stable_selective_terminal_macro_v37_114_telemetry.json"
        )

    payload = {
        "version": "v37.114",
        "mode": (
            "SELECTIVE_QTIGHT_COLUMN_TERMINAL_MACRO"
        ),
        "oracle_source": (
            _ORACLE_SOURCE
        ),
        "oracle_load_or_build_wall": (
            _ORACLE_WALL
        ),
        "macro_max_suffixes": int(
            _MACRO_MAX_SUFFIXES
        ),
        "macro_min_hq": int(
            _MACRO_MIN_HQ
        ),
        "macro_max_hq": int(
            _MACRO_MAX_HQ
        ),
        "stats": dict(
            _STATS
        ),
        "h_histogram": {
            str(k): int(v)
            for k, v in sorted(
                _H_HIST.items()
            )
        },
        "h_skip_histogram": {
            str(k): int(v)
            for k, v in sorted(
                _H_SKIP_HIST.items()
            )
        },
        "h_terminal_call_histogram": {
            str(k): int(v)
            for k, v in sorted(
                _H_TERMINAL_CALL_HIST.items()
            )
        },
        "h_hit_histogram": {
            str(k): int(v)
            for k, v in sorted(
                _H_HIT_HIST.items()
            )
        },
        "word_count_histogram": {
            str(k): int(v)
            for k, v in sorted(
                _WORDCOUNT_HIST.items()
            )
        },
        "terminal_k2_histogram": {
            str(k): int(v)
            for k, v in sorted(
                _TERMINAL_K2_HIST.items()
            )
        },
        "terminal_call_rank_histogram": {
            str(k): int(v)
            for k, v in sorted(
                _TERMINAL_CALL_RANK_HIST.items()
            )
        },
        "terminal_hit_rank_histogram": {
            str(k): int(v)
            for k, v in sorted(
                _TERMINAL_HIT_RANK_HIST.items()
            )
        },
        "macro_hit_rank_histogram": {
            str(k): int(v)
            for k, v in sorted(
                _MACRO_HIT_RANK_HIST.items()
            )
        },
        "macro_hit_events": list(
            _MACRO_HIT_EVENTS
        ),
        "return_code": int(
            rc
        ),
        "production_changes": (
            "NONE"
        ),
    }

    out = Path(
        path
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
        ),
        encoding="utf-8",
    )

    print(
        "v37.114 telemetry JSON:",
        out,
    )


def main():
    global _ORACLE
    global _ORACLE_SOURCE
    global _ORACLE_WALL
    global _MACRO_MAX_SUFFIXES
    global _MACRO_MIN_HQ
    global _MACRO_MAX_HQ

    local, remaining = (
        _parse_local_args(
            sys.argv[
                1:
            ]
        )
    )

    _MACRO_MAX_SUFFIXES = int(
        local.macro_max_suffixes
    )

    _MACRO_MIN_HQ = int(
        local.macro_min_hq
    )

    _MACRO_MAX_HQ = int(
        local.macro_max_hq
    )

    dr_targets = (
        v57.local_dr_pose_sets()
    )

    (
        _ORACLE,
        _ORACLE_SOURCE,
        _ORACLE_WALL,
    ) = v84.load_or_build_oracle(
        cache_path=Path(
            local.column_word_cache
        ),
        dr_targets=dr_targets,
        qtables=None,
    )

    v24._install_dynamic_hk2_ordering = (
        _combined_installer
    )

    sys.argv = [
        sys.argv[
            0
        ],
        *remaining,
    ]

    print(
        "# CubeLab v37.114 - "
        "STABLE Q-TIGHT COLUMN TERMINAL MACRO"
    )

    print(
        "base                 :",
        "v37.24 Dynamic H/K2 + v37.24.1 replay",
    )

    print(
        "macro horizon        :",
        (
            f"q-tight exact H, "
            f"{_MACRO_MIN_HQ}<=Hq<={_MACRO_MAX_HQ}"
        ),
    )

    print(
        "macro operation      :",
        "complete shared H suffix -> existing H=0 terminal/K2 find",
    )

    print(
        "macro suffix cap     :",
        _MACRO_MAX_SUFFIXES,
    )

    print(
        "outside Hq band      :",
        "immediate unchanged stable-find fallback",
    )

    print(
        "terminal front       :",
        "CANONICAL / no P2LB prefront",
    )

    print(
        "macro miss           :",
        "UNCHANGED stable find fallback",
    )

    print(
        "pruning              :",
        "NONE",
    )

    print(
        "feasibility change   :",
        "NONE",
    )

    print(
        "oracle cache         :",
        _ORACLE_SOURCE,
    )

    print(
        "oracle load/build    :",
        f"{_ORACLE_WALL:.3f}s",
    )

    print(
        "witness replay       :",
        "REQUIRED",
    )

    print(
        "production changes   :",
        "NONE",
    )

    print()

    rc = replay24.main()

    print()
    print(
        "# v37.112 TERMINAL MACRO TELEMETRY"
    )

    for key in (
        "find_calls",
        "in_range_qtight_calls",
        "eligible_calls",
        "below_range_skips",
        "above_range_skips",
        "non_qtight_or_budget_skips",
        "oracle_calls",
        "oracle_empty",
        "oracle_nonempty",
        "oracle_word_count_sum",
        "oracle_word_count_max",
        "suffix_words_considered",
        "suffix_q_rejects",
        "suffix_parent_goal_skips",
        "terminal_calls",
        "terminal_hits",
        "terminal_misses",
        "macro_hits",
        "fallback_calls",
        "contract_term_h_nonempty",
        "contract_term_total_mismatch",
        "contract_over_budget",
    ):
        print(
            f"{key:<30}:",
            f"{_STATS[key]:,}",
        )

    if _STATS[
        "oracle_nonempty"
    ]:
        print(
            "mean exact suffix words       :",
            f"{_STATS['oracle_word_count_sum']/_STATS['oracle_nonempty']:.3f}",
        )

    if _STATS[
        "eligible_calls"
    ]:
        print(
            "macro hit / eligible          :",
            f"{_STATS['macro_hits']/_STATS['eligible_calls']:.2%}",
        )

    if _STATS[
        "terminal_calls"
    ]:
        print(
            "terminal success rate         :",
            f"{_STATS['terminal_hits']/_STATS['terminal_calls']:.2%}",
        )

    print(
        "active Hq histogram           :",
        dict(
            sorted(
                _H_HIST.items()
            )
        ),
    )

    print(
        "skipped Hq histogram          :",
        dict(
            sorted(
                _H_SKIP_HIST.items()
            )
        ),
    )

    print(
        "terminal calls by Hq          :",
        dict(
            sorted(
                _H_TERMINAL_CALL_HIST.items()
            )
        ),
    )

    print(
        "macro hits by Hq              :",
        dict(
            sorted(
                _H_HIT_HIST.items()
            )
        ),
    )

    print(
        "word-count histogram          :",
        dict(
            sorted(
                _WORDCOUNT_HIST.items()
            )
        ),
    )

    print(
        "terminal K2 histogram         :",
        dict(
            sorted(
                _TERMINAL_K2_HIST.items()
            )
        ),
    )

    print(
        "terminal call rank histogram  :",
        dict(
            sorted(
                _TERMINAL_CALL_RANK_HIST.items()
            )
        ),
    )

    print(
        "terminal hit rank histogram   :",
        dict(
            sorted(
                _TERMINAL_HIT_RANK_HIST.items()
            )
        ),
    )

    print(
        "macro hit rank histogram      :",
        dict(
            sorted(
                _MACRO_HIT_RANK_HIST.items()
            )
        ),
    )

    if _MACRO_HIT_EVENTS:
        print(
            "macro hit events:"
        )
        for event in _MACRO_HIT_EVENTS:
            print(
                "   ",
                f"rank={event['suffix_rank']} "
                f"H={event['hq']} "
                f"words={event['word_count']} "
                f"budget={event['local_budget']} "
                f"K2={event['terminal_k2_len']} "
                f"suffix={' '.join(event['suffix'])}"
            )

    print()
    print(
        "Interpretation:"
    )
    print(
        "  Preserve H3/H4 macro hits and H6/H7 collapse -> selective activation is sound operationally."
    )
    print(
        "  H5/H8 nodes/wall moving toward OFF -> H1/H2 speculative handoffs were pure overhead."
    )
    print(
        "  many terminal calls but no hits -> K2 boundary dominates; close this macro form."
    )
    print(
        "  macro hits but slower wall -> retain semantics, optimize terminal handoff only."
    )
    print(
        "  macro miss is never UNSAT; every miss falls back to stable v37.24."
    )

    _write_telemetry(
        local.macro_telemetry_output,
        rc=rc,
    )

    return int(
        rc
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
