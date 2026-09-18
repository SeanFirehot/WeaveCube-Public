#!/usr/bin/env python3
"""
CubeLab v37.128 — POST-DR K2 DEFICIT LANDSCAPE AUDIT

Motivation
----------
Repeated natural-search portfolio audits now show the same pattern:

    exact H suffix / H-skeleton synthesis works,
    but successful terminal continuations are sparse
    and successful branches are already early/stable choices.

So before moving farther upward in H horizon, diagnose the actual boundary:

    POST-DR K2 feasibility.

For natural q-tight H6/H7 contexts:

    enumerate the same dynamic H-prefix + exact R4-to-DR portfolio
        ↓
    deduplicate resulting DR endpoints
        ↓
    probe H=0/K2 terminal with budgets 0..K2_PROBE_CAP
        ↓
    estimate minimal K2 budget for each exact endpoint
        ↓
    compare against the remaining K2 budget of the original H frontier

Define:

    K2 deficit = minimal_k2_budget - remaining_budget

Interpretation
--------------
    deficit <= 0 : should fit current budget
    deficit = +1 : one-move near miss
    deficit = +2 : two-move near miss
    deficit >= 3 : structurally farther away
    UNKNOWN      : no terminal witness within probe cap / probe cut

This tells us whether the next directed-repair requirement should be:

    "reach DR"

or the stronger:

    "reach a DR endpoint whose K2 residual fits the remaining total budget."

Stable timed search is unchanged.
All K2 probing is OFFLINE.

Production changes: NONE.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if SRC.is_dir():
    s = str(SRC)
    if s not in sys.path:
        sys.path.insert(0, s)

import audit_fast_stable_selective_terminal_macro_v37_114 as v114

_SHADOW_MAX_SAMPLES = 4096
_SHADOW_SEED = 20261042
_HORIZONS = (6, 7)
_R4_SUFFIX_CAP = 64
_PREFIX_NODE_CAP = 100000
_K2_PROBE_CAP = 10
_K2_CALL_CAP = 10000
_OFFLINE_TIME_LIMIT = 30.0
_REPORT_EXAMPLES = 12
_OUTPUT = "reports/v37/post_dr_k2_deficit_landscape_v37_128.json"

_RNG = random.Random(_SHADOW_SEED)
_STATS = Counter()
_SAMPLES = []
_BASE_FIND_NO_SHADOW = None


def _extract_frame(args, kwargs):
    poses = kwargs.get("poses")
    q = kwargs.get("q")
    depths = kwargs.get("depths")
    last = kwargs.get("last")
    if last is None:
        last = kwargs.get("last_face")
    parent = kwargs.get("parent_was_goal")
    budget = kwargs.get("budget")

    if poses is None and len(args) >= 1:
        poses = args[0]
    if q is None and len(args) >= 2:
        q = args[1]
    if depths is None and len(args) >= 3:
        depths = args[2]
    if last is None and len(args) >= 4:
        last = args[3]
    if parent is None and len(args) >= 5:
        parent = args[4]
    if budget is None and len(args) >= 6:
        budget = args[5]

    if poses is None or q is None or depths is None or budget is None:
        return None

    return {
        "poses": tuple(int(x) for x in poses),
        "q": int(q),
        "depths": tuple(int(x) for x in depths),
        "last": last,
        "parent_was_goal": parent,
        "budget": int(budget),
    }


def _stable_result_summary(result):
    if result is None:
        return {"found": False, "total": None}

    if not isinstance(result, dict):
        return {"found": True, "total": None}

    return {
        "found": True,
        "total": (
            None
            if result.get("total") is None
            else int(result["total"])
        ),
    }


def _reservoir_add(sample):
    _STATS["positive_find_returns_observed"] += 1
    seen = int(_STATS["positive_find_returns_observed"])

    if len(_SAMPLES) < int(_SHADOW_MAX_SAMPLES):
        _SAMPLES.append(sample)
        return

    j = _RNG.randrange(seen)

    if j < int(_SHADOW_MAX_SAMPLES):
        _SAMPLES[j] = sample


def _install_shadow_observer():
    global _BASE_FIND_NO_SHADOW

    cls = v114.v24.grc.MemoTotalBudgetSearch

    if getattr(cls, "_v37128_shadow_installed", False):
        return

    orig_find = cls.find
    _BASE_FIND_NO_SHADOW = orig_find

    def find_with_shadow(self, *args, **kwargs):
        _STATS["find_calls"] += 1
        frame = _extract_frame(args, kwargs)

        result = orig_find(self, *args, **kwargs)

        _STATS["normal_find_returns"] += 1
        if result is None:
            _STATS["normal_no_returns"] += 1

        if frame is not None:
            positive = tuple(
                int(d)
                for d in frame["depths"]
                if int(d) > 0
            )

            if positive:
                _reservoir_add(
                    {
                        **frame,
                        "solver": self,
                        "stable_result": _stable_result_summary(result),
                    }
                )

        return result

    cls.find = find_with_shadow
    cls._v37128_shadow_installed = True


_ORIGINAL_V114_COMBINED_INSTALLER = v114._combined_installer


def _v37128_combined_installer():
    _ORIGINAL_V114_COMBINED_INSTALLER()
    _install_shadow_observer()


def _frame_key(frame):
    return (
        tuple(frame["poses"]),
        int(frame["q"]),
        tuple(frame["depths"]),
        frame["last"],
        int(frame["budget"]),
    )


def _with_budget_transitions(solver, *, q, depths, last, budget):
    sentinel = object()
    previous = getattr(solver, "_v3724_local_budget", sentinel)

    try:
        solver._v3724_local_budget = int(budget)

        return tuple(
            solver.transitions(
                int(q),
                tuple(int(x) for x in depths),
                last,
            )
        )
    finally:
        if previous is sentinel:
            try:
                delattr(solver, "_v3724_local_budget")
            except AttributeError:
                pass
        else:
            solver._v3724_local_budget = previous


def _oracle_query_r4(oracle, *, poses, q, last_face):
    qr = oracle.query(
        poses=tuple(poses),
        q=int(q),
        R=4,
        last_face=last_face,
        add_q_first=False,
    )

    if not qr["feasible"]:
        return {"mask": 0, "count": 0}

    mask = int(qr["word_mask"])

    return {
        "mask": mask,
        "count": int(qr.get("word_count", mask.bit_count())),
    }


def _iter_bits(mask):
    value = int(mask)

    while value:
        low = value & -value
        yield low.bit_length() - 1
        value ^= low


def _endpoint_key(end_poses, final_face):
    return (
        tuple(int(x) for x in end_poses),
        final_face,
    )


def _probe_min_k2(
    *,
    solver,
    end_poses,
    end_q,
    final_face,
    k2_probe_cap,
    global_calls,
    deadline,
):
    """
    Increasing-budget H=0 terminal probe.

    A normal witness at budget b after all smaller normal misses is an exact
    upper/minimum-under-this-terminal-search observation for this audit.
    Exceptions are treated as UNKNOWN, never as UNSAT.
    """
    calls = 0
    exceptions = 0

    for budget in range(
        0,
        int(k2_probe_cap) + 1,
    ):
        if (
            time.perf_counter() >= deadline
            or int(global_calls["calls"]) >= int(global_calls["cap"])
        ):
            return {
                "status": "TRUNCATED",
                "min_k2": None,
                "calls": int(calls),
                "exceptions": int(exceptions),
            }

        calls += 1
        global_calls["calls"] += 1

        try:
            result = _BASE_FIND_NO_SHADOW(
                solver,
                tuple(end_poses),
                int(end_q),
                (0,),
                final_face,
                False,
                int(budget),
            )
        except Exception:
            exceptions += 1
            return {
                "status": "UNKNOWN_CUT",
                "min_k2": None,
                "calls": int(calls),
                "exceptions": int(exceptions),
            }

        if result is None:
            continue

        term_h = tuple(result.get("h_tail", ()))
        term_k2 = tuple(result.get("k2_word", ()))

        if term_h:
            return {
                "status": "UNKNOWN_NONZERO_H",
                "min_k2": None,
                "calls": int(calls),
                "exceptions": int(exceptions),
            }

        terminal_total = int(
            result.get(
                "total",
                len(term_k2),
            )
        )

        if terminal_total > int(budget):
            return {
                "status": "UNKNOWN_BUDGET_MISMATCH",
                "min_k2": None,
                "calls": int(calls),
                "exceptions": int(exceptions),
            }

        return {
            "status": "FOUND",
            "min_k2": int(terminal_total),
            "calls": int(calls),
            "exceptions": int(exceptions),
            "word": list(term_k2),
        }

    return {
        "status": "ABOVE_CAP",
        "min_k2": None,
        "calls": int(calls),
        "exceptions": int(exceptions),
    }


def _enumerate_context_endpoints(
    *,
    frame,
    oracle,
    hq,
    r4_suffix_cap,
    prefix_budget,
    deadline,
):
    """
    Enumerate exact DR endpoints for one natural q-tight H6/H7 context.

    Returns skeleton records plus dedup endpoint metadata.
    """
    solver = frame["solver"]

    skeletons = []
    endpoints = {}

    stats = Counter()

    def recurse(
        state,
        remaining_h,
        prefix_moves,
        prefix_ranks,
    ):
        if time.perf_counter() >= deadline:
            return

        if int(prefix_budget["nodes"]) >= int(prefix_budget["cap"]):
            return

        if remaining_h == 4:
            stats["r4_leaf_states"] += 1

            qinfo = _oracle_query_r4(
                oracle,
                poses=state["poses"],
                q=int(state["q"]),
                last_face=state["last"],
            )

            qcount = int(qinfo["count"])

            if qcount <= 0:
                return

            stats["r4_oracle_nonempty"] += 1
            stats["r4_word_sum"] += qcount
            stats["max_r4_words"] = max(
                int(stats["max_r4_words"]),
                qcount,
            )

            if qcount > int(r4_suffix_cap):
                stats["r4_cap_truncated"] += 1

            words = oracle.words[4]

            for r4_rank, word_idx in enumerate(
                _iter_bits(qinfo["mask"]),
                start=1,
            ):
                if r4_rank > int(r4_suffix_cap):
                    break

                if time.perf_counter() >= deadline:
                    break

                stats["r4_words_considered"] += 1

                row = words[int(word_idx)]

                r4_suffix = tuple(
                    v114.v24.grc.ts.MOVE_ORDER[int(mi)]
                    for mi in row
                )

                end_poses, end_q, q_before_final, final_face = (
                    v114._suffix_terminal_state(
                        solver,
                        tuple(state["poses"]),
                        int(state["q"]),
                        row,
                    )
                )

                if int(end_q) != int(v114.v24.grc.ts.GOAL_Q):
                    continue

                if int(q_before_final) == int(v114.v24.grc.ts.GOAL_Q):
                    continue

                key = _endpoint_key(
                    end_poses,
                    final_face,
                )

                if key not in endpoints:
                    endpoints[key] = {
                        "end_poses": tuple(
                            int(x)
                            for x in end_poses
                        ),
                        "end_q": int(end_q),
                        "final_face": final_face,
                    }

                skeletons.append(
                    {
                        "endpoint_key": key,
                        "prefix_moves": list(prefix_moves),
                        "prefix_ranks": list(prefix_ranks),
                        "r4_rank": int(r4_rank),
                        "r4_suffix": list(r4_suffix),
                    }
                )

            return

        transitions = _with_budget_transitions(
            solver,
            q=int(state["q"]),
            depths=tuple(state["depths"]),
            last=state["last"],
            budget=int(state["budget"]),
        )

        required_child_h = int(remaining_h) - 1

        for local_rank, (
            move,
            mi,
            nq,
            child_depths,
        ) in enumerate(
            transitions,
            start=1,
        ):
            if (
                time.perf_counter() >= deadline
                or int(prefix_budget["nodes"]) >= int(prefix_budget["cap"])
            ):
                break

            positive_child_depths = {
                int(d)
                for d in child_depths
                if int(d) > 0
            }

            if required_child_h not in positive_child_depths:
                continue

            prefix_budget["nodes"] += 1
            stats["prefix_nodes"] += 1

            child_poses = v114._pose_step(
                tuple(state["poses"]),
                int(mi),
            )

            recurse(
                {
                    "poses": tuple(child_poses),
                    "q": int(nq),
                    "depths": tuple(int(x) for x in child_depths),
                    "last": str(move)[0],
                    "budget": int(state["budget"]) - 1,
                },
                required_child_h,
                tuple(prefix_moves) + (str(move),),
                tuple(prefix_ranks) + (int(local_rank),),
            )

    recurse(
        {
            "poses": tuple(frame["poses"]),
            "q": int(frame["q"]),
            "depths": tuple(frame["depths"]),
            "last": frame["last"],
            "budget": int(frame["budget"]),
        },
        int(hq),
        (),
        (),
    )

    return {
        "stats": stats,
        "skeletons": skeletons,
        "endpoints": endpoints,
    }


def _parse_local_args(argv):
    ap = argparse.ArgumentParser(add_help=False)

    ap.add_argument(
        "--shadow-max-find-samples",
        type=int,
        default=4096,
    )
    ap.add_argument(
        "--shadow-seed",
        type=int,
        default=20261042,
    )
    ap.add_argument(
        "--probe-horizons",
        nargs="+",
        type=int,
        default=[6, 7],
    )
    ap.add_argument(
        "--portfolio-max-r4-suffixes",
        type=int,
        default=64,
    )
    ap.add_argument(
        "--portfolio-prefix-node-cap",
        type=int,
        default=100000,
    )
    ap.add_argument(
        "--k2-probe-cap",
        type=int,
        default=10,
    )
    ap.add_argument(
        "--k2-call-cap",
        type=int,
        default=10000,
    )
    ap.add_argument(
        "--offline-time-limit",
        type=float,
        default=30.0,
    )
    ap.add_argument(
        "--report-examples",
        type=int,
        default=12,
    )
    ap.add_argument(
        "--shadow-output",
        default="reports/v37/post_dr_k2_deficit_landscape_v37_128.json",
    )

    ns, remaining = ap.parse_known_args(argv)

    ns.shadow_max_find_samples = max(
        1,
        int(ns.shadow_max_find_samples),
    )
    ns.probe_horizons = tuple(
        sorted(
            {
                int(x)
                for x in ns.probe_horizons
                if int(x) >= 4
            }
        )
    )
    ns.portfolio_max_r4_suffixes = max(
        1,
        int(ns.portfolio_max_r4_suffixes),
    )
    ns.portfolio_prefix_node_cap = max(
        1,
        int(ns.portfolio_prefix_node_cap),
    )
    ns.k2_probe_cap = max(
        0,
        int(ns.k2_probe_cap),
    )
    ns.k2_call_cap = max(
        1,
        int(ns.k2_call_cap),
    )
    ns.offline_time_limit = max(
        0.1,
        float(ns.offline_time_limit),
    )
    ns.report_examples = max(
        1,
        int(ns.report_examples),
    )

    return ns, remaining


def main():
    global _SHADOW_MAX_SAMPLES
    global _SHADOW_SEED
    global _HORIZONS
    global _R4_SUFFIX_CAP
    global _PREFIX_NODE_CAP
    global _K2_PROBE_CAP
    global _K2_CALL_CAP
    global _OFFLINE_TIME_LIMIT
    global _REPORT_EXAMPLES
    global _OUTPUT
    global _RNG

    local, remaining = _parse_local_args(
        sys.argv[1:]
    )

    _SHADOW_MAX_SAMPLES = int(local.shadow_max_find_samples)
    _SHADOW_SEED = int(local.shadow_seed)
    _HORIZONS = tuple(local.probe_horizons)
    _R4_SUFFIX_CAP = int(local.portfolio_max_r4_suffixes)
    _PREFIX_NODE_CAP = int(local.portfolio_prefix_node_cap)
    _K2_PROBE_CAP = int(local.k2_probe_cap)
    _K2_CALL_CAP = int(local.k2_call_cap)
    _OFFLINE_TIME_LIMIT = float(local.offline_time_limit)
    _REPORT_EXAMPLES = int(local.report_examples)
    _OUTPUT = local.shadow_output
    _RNG = random.Random(_SHADOW_SEED)

    v114._combined_installer = _v37128_combined_installer
    sys.argv = [sys.argv[0], *remaining]

    print("# CubeLab v37.128 - POST-DR K2 DEFICIT LANDSCAPE AUDIT")
    print("stable search        : v37.24.1 + v37.114 UNCHANGED")
    print("H horizons           :", list(_HORIZONS))
    print("portfolio            : dynamic H-prefix + exact R4 -> DR")
    print("K2 probe             :", f"budgets 0..{_K2_PROBE_CAP}")
    print("K2 call cap          :", _K2_CALL_CAP)
    print("prefix node cap      :", _PREFIX_NODE_CAP)
    print("offline time cap     :", f"{_OFFLINE_TIME_LIMIT:.1f}s")
    print("production changes   : NONE")
    print()

    stable_t0 = time.perf_counter()
    rc = v114.main()
    stable_outer_wall = time.perf_counter() - stable_t0

    print()
    print("# OFFLINE POST-DR K2 LANDSCAPE")

    oracle = v114._ORACLE
    if oracle is None:
        raise RuntimeError("v37.114 oracle unavailable")
    if _BASE_FIND_NO_SHADOW is None:
        raise RuntimeError("base find capture missing")

    unique = {}

    for frame in _SAMPLES:
        key = _frame_key(frame)
        previous = unique.get(key)

        if previous is None:
            unique[key] = frame
        elif (
            not previous["stable_result"]["found"]
            and frame["stable_result"]["found"]
        ):
            unique[key] = frame

    t0 = time.perf_counter()
    deadline = t0 + float(_OFFLINE_TIME_LIMIT)

    prefix_budget = {
        "nodes": 0,
        "cap": int(_PREFIX_NODE_CAP),
    }

    k2_calls = {
        "calls": 0,
        "cap": int(_K2_CALL_CAP),
    }

    endpoint_probe_cache = {}

    rows = []
    analysis_truncated = False

    aggregate_status = Counter()
    aggregate_min_k2 = Counter()
    aggregate_deficit = Counter()

    for frame in unique.values():
        if (
            time.perf_counter() >= deadline
            or int(prefix_budget["nodes"]) >= int(prefix_budget["cap"])
            or int(k2_calls["calls"]) >= int(k2_calls["cap"])
        ):
            analysis_truncated = True
            break

        solver = frame["solver"]

        hq = int(
            v114.v24.grc.ts.phase1_lb(
                int(frame["q"]),
                solver.cos,
                solver.eos,
            )
        )

        positive_depths = {
            int(d)
            for d in frame["depths"]
            if int(d) > 0
        }

        if (
            hq not in _HORIZONS
            or hq not in positive_depths
            or int(frame["budget"]) < hq
        ):
            continue

        portfolio = _enumerate_context_endpoints(
            frame=frame,
            oracle=oracle,
            hq=int(hq),
            r4_suffix_cap=int(_R4_SUFFIX_CAP),
            prefix_budget=prefix_budget,
            deadline=deadline,
        )

        remaining_budget = int(frame["budget"]) - int(hq)

        endpoint_rows = []

        for key, endpoint in portfolio["endpoints"].items():
            if (
                time.perf_counter() >= deadline
                or int(k2_calls["calls"]) >= int(k2_calls["cap"])
            ):
                analysis_truncated = True
                break

            cache_key = (
                key,
                id(solver),
            )

            probe = endpoint_probe_cache.get(cache_key)

            if probe is None:
                probe = _probe_min_k2(
                    solver=solver,
                    end_poses=endpoint["end_poses"],
                    end_q=int(endpoint["end_q"]),
                    final_face=endpoint["final_face"],
                    k2_probe_cap=int(_K2_PROBE_CAP),
                    global_calls=k2_calls,
                    deadline=deadline,
                )
                endpoint_probe_cache[cache_key] = probe

            status = str(probe["status"])
            aggregate_status[status] += 1

            deficit = None

            if status == "FOUND":
                min_k2 = int(probe["min_k2"])
                aggregate_min_k2[min_k2] += 1
                deficit = min_k2 - int(remaining_budget)

                if deficit <= 0:
                    bucket = "<=0"
                elif deficit == 1:
                    bucket = "+1"
                elif deficit == 2:
                    bucket = "+2"
                elif deficit == 3:
                    bucket = "+3"
                else:
                    bucket = ">=+4"

                aggregate_deficit[bucket] += 1

            endpoint_rows.append(
                {
                    "status": status,
                    "min_k2": (
                        None
                        if probe.get("min_k2") is None
                        else int(probe["min_k2"])
                    ),
                    "remaining_budget": int(remaining_budget),
                    "deficit": (
                        None
                        if deficit is None
                        else int(deficit)
                    ),
                }
            )

        found_deficits = [
            int(row["deficit"])
            for row in endpoint_rows
            if row["deficit"] is not None
        ]

        best_deficit = (
            min(found_deficits)
            if found_deficits
            else None
        )

        best_min_k2 = None

        found_min_k2 = [
            int(row["min_k2"])
            for row in endpoint_rows
            if row["min_k2"] is not None
        ]

        if found_min_k2:
            best_min_k2 = min(found_min_k2)

        rows.append(
            {
                "hq": int(hq),
                "budget": int(frame["budget"]),
                "remaining_k2_budget": int(remaining_budget),
                "stable_result": dict(frame["stable_result"]),
                "prefix_nodes": int(portfolio["stats"]["prefix_nodes"]),
                "r4_leaf_states": int(portfolio["stats"]["r4_leaf_states"]),
                "r4_oracle_nonempty": int(portfolio["stats"]["r4_oracle_nonempty"]),
                "r4_word_sum": int(portfolio["stats"]["r4_word_sum"]),
                "r4_words_considered": int(portfolio["stats"]["r4_words_considered"]),
                "unique_dr_endpoints": len(portfolio["endpoints"]),
                "probe_status_histogram": dict(
                    Counter(
                        row["status"]
                        for row in endpoint_rows
                    )
                ),
                "best_min_k2": best_min_k2,
                "best_deficit": best_deficit,
                "fit_endpoint_count": sum(
                    1
                    for row in endpoint_rows
                    if (
                        row["deficit"] is not None
                        and int(row["deficit"]) <= 0
                    )
                ),
                "deficit1_endpoint_count": sum(
                    1
                    for row in endpoint_rows
                    if row["deficit"] == 1
                ),
                "deficit2_endpoint_count": sum(
                    1
                    for row in endpoint_rows
                    if row["deficit"] == 2
                ),
            }
        )

        if analysis_truncated:
            break

    offline_wall = time.perf_counter() - t0

    stable_no_rows = [
        row
        for row in rows
        if not row["stable_result"]["found"]
    ]

    stable_found_rows = [
        row
        for row in rows
        if row["stable_result"]["found"]
    ]

    stable_no_best_deficits = Counter()

    for row in stable_no_rows:
        d = row["best_deficit"]

        if d is None:
            stable_no_best_deficits["UNKNOWN"] += 1
        elif int(d) <= 0:
            stable_no_best_deficits["<=0"] += 1
        elif int(d) == 1:
            stable_no_best_deficits["+1"] += 1
        elif int(d) == 2:
            stable_no_best_deficits["+2"] += 1
        elif int(d) == 3:
            stable_no_best_deficits["+3"] += 1
        else:
            stable_no_best_deficits[">=+4"] += 1

    near_miss_rows = [
        row
        for row in stable_no_rows
        if row["best_deficit"] in (1, 2)
    ]

    fit_no_rows = [
        row
        for row in stable_no_rows
        if (
            row["best_deficit"] is not None
            and int(row["best_deficit"]) <= 0
        )
    ]

    print("find calls observed  :", f"{_STATS['find_calls']:,}")
    print("positive samples     :", len(_SAMPLES))
    print("unique contexts      :", len(unique))
    print("analyzed H contexts  :", len(rows))
    print("stable-found ctx     :", len(stable_found_rows))
    print("stable-NO ctx        :", len(stable_no_rows))
    print("prefix nodes         :", prefix_budget["nodes"])
    print("unique endpoint probes:", len(endpoint_probe_cache))
    print("K2 terminal calls    :", k2_calls["calls"])
    print("probe status hist    :", dict(sorted(aggregate_status.items())))
    print("min K2 histogram     :", dict(sorted(aggregate_min_k2.items())))
    print("endpoint deficit hist:", dict(aggregate_deficit))
    print("stable-NO best deficit:", dict(stable_no_best_deficits))
    print("stable-NO fit endpoints:", len(fit_no_rows))
    print("stable-NO +1/+2 near misses:", len(near_miss_rows))
    print("analysis truncated   :", analysis_truncated)
    print("offline wall         :", f"{offline_wall:.3f}s")

    if near_miss_rows:
        print()
        print("NEAR-MISS EXAMPLES:")

        for row in near_miss_rows[: int(_REPORT_EXAMPLES)]:
            print(
                "   ",
                f"Hq={row['hq']} "
                f"budget={row['budget']} "
                f"K2budget={row['remaining_k2_budget']} "
                f"bestMinK2={row['best_min_k2']} "
                f"deficit={row['best_deficit']} "
                f"endpoints={row['unique_dr_endpoints']} "
                f"def1={row['deficit1_endpoint_count']} "
                f"def2={row['deficit2_endpoint_count']}",
            )

    if fit_no_rows:
        decision = "K2_DEFICIT_AUDIT_INCONSISTENT_FIT_STABLE_NO"
        note = (
            "At least one stable-NO H frontier has an exact DR endpoint whose "
            "measured minimal K2 fits the remaining budget. This should have "
            "produced a portfolio rescue; inspect ownership/cut semantics before "
            "using the deficit model."
        )
    elif near_miss_rows:
        decision = "POST_DR_K2_NEAR_MISS_TARGET_FOUND"
        note = (
            "Stable-NO H6/H7 frontiers contain exact DR endpoints whose measured "
            "K2 requirement misses the remaining budget by only one or two moves. "
            "This gives a concrete directed-repair objective: synthesize H "
            "frontiers toward DR endpoints with lower K2 residual, rather than "
            "ranking H moves by DR reachability alone."
        )
    elif analysis_truncated:
        decision = "POST_DR_K2_DEFICIT_AUDIT_TRUNCATED"
        note = (
            "The K2 landscape audit reached an explicit offline cap. Increase "
            "only that cap before drawing a structural conclusion."
        )
    else:
        decision = "POST_DR_K2_NEAR_MISS_NOT_OBSERVED"
        note = (
            "Stable-NO H6/H7 frontiers do not expose one- or two-move K2 near "
            "misses within the probe cap. The current short corpus offers no "
            "clear local K2-residual repair target; move to a harder/fresh global "
            "H-skeleton corpus rather than extending local horizons."
        )

    print()
    print("# DECISION")
    print(decision)
    print(note)
    print()
    print("NEXT:")

    if near_miss_rows:
        print("  v37.129 K2-residual-directed H-skeleton synthesis")
        print("  target: reduce measured/P2 lower-bound K2 deficit by 1-2")
        print("  validate on the concrete near-miss frontiers first")
    elif analysis_truncated:
        print("  rerun only with larger offline K2 cap")
    else:
        print("  stop extending this short stable corpus")
        print("  move directed-repair evaluation to a harder fresh/global H-skeleton corpus")

    payload = {
        "version": "v37.128",
        "mode": "POST_DR_K2_DEFICIT_LANDSCAPE",
        "stable_return_code": int(rc),
        "stable_outer_wall": float(stable_outer_wall),
        "config": {
            "shadow_max_find_samples": int(_SHADOW_MAX_SAMPLES),
            "shadow_seed": int(_SHADOW_SEED),
            "probe_horizons": list(_HORIZONS),
            "portfolio_max_r4_suffixes": int(_R4_SUFFIX_CAP),
            "portfolio_prefix_node_cap": int(_PREFIX_NODE_CAP),
            "k2_probe_cap": int(_K2_PROBE_CAP),
            "k2_call_cap": int(_K2_CALL_CAP),
            "offline_time_limit": float(_OFFLINE_TIME_LIMIT),
        },
        "aggregate": {
            "positive_samples": len(_SAMPLES),
            "unique_contexts": len(unique),
            "analyzed_h_contexts": len(rows),
            "stable_found_contexts": len(stable_found_rows),
            "stable_no_contexts": len(stable_no_rows),
            "prefix_nodes": int(prefix_budget["nodes"]),
            "unique_endpoint_probes": len(endpoint_probe_cache),
            "k2_terminal_calls": int(k2_calls["calls"]),
            "probe_status_histogram": dict(aggregate_status),
            "min_k2_histogram": {
                str(k): int(v)
                for k, v in sorted(aggregate_min_k2.items())
            },
            "endpoint_deficit_histogram": dict(aggregate_deficit),
            "stable_no_best_deficit_histogram": dict(stable_no_best_deficits),
            "stable_no_fit_endpoint_contexts": len(fit_no_rows),
            "stable_no_near_miss_contexts": len(near_miss_rows),
            "analysis_truncated": bool(analysis_truncated),
            "offline_wall": float(offline_wall),
        },
        "rows": rows,
        "decision": decision,
        "note": note,
        "production_changes": "NONE",
    }

    out = Path(_OUTPUT)
    out.parent.mkdir(parents=True, exist_ok=True)

    out.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=repr,
        ),
        encoding="utf-8",
    )

    print("shadow JSON          :", out)

    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
