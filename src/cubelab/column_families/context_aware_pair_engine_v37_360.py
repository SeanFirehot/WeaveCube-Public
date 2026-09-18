"""Context-aware branch-local exact pair-family completion for v37.360.

This is an additive research adapter over the immutable v37.359 assets.  The
only language change is selecting the registry root for the supplied canonical
previous-face context; all subsequent nodes already encode their context.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from functools import lru_cache
import resource
from time import perf_counter, process_time
from typing import Callable

import numpy as np

from cubelab.ato.column_grounding import apply_word
from cubelab.ato.regular_support import (
    MOVE_NAMES, canonical_allow, canonical_word_arrays,
)
from cubelab.column_families.residual_factor_view import CONTEXTS

from .r4_pair_suffix_closure_v37_359 import (
    R4SparseSupportAsset,
    decode_names,
    explicit_four_layer_codes,
    recursive_product_codes,
)
from .resident_pair_atlas_v37_359 import ResidentPairAtlas


SOLVED = (0,) * 20
MODES = ("ENUMERATE_ALL_SOLUTIONS", "FIRST_REPLAY_VALID_WITNESS")
BACKENDS = (
    "R4_RECURSIVE_PRODUCT",
    "R4_SPARSE_WORD_SUPPORT",
    "R4_RECURSIVE_PRODUCT_WITH_EXACT_SIGNATURE",
    "R4_EXPLICIT_FOUR_LAYER_CONTROL_WITH_EXACT_SIGNATURE",
)
SIGNATURE_BACKENDS = (
    "R4_SPARSE_WORD_SUPPORT",
    "R4_RECURSIVE_PRODUCT_WITH_EXACT_SIGNATURE",
    "R4_EXPLICIT_FOUR_LAYER_CONTROL_WITH_EXACT_SIGNATURE",
)


def _rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / 1024.0 if value < 10**9 else value / (1024.0 * 1024.0))


def _solution_hash(solutions: tuple[tuple[str, ...], ...]) -> str:
    payload = [list(word) for word in solutions]
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=32)
def _continuation_words(length: int, previous_face: str) -> np.ndarray:
    rows = canonical_word_arrays(int(length), previous_face)
    rows.flags.writeable = False
    return rows


def warm_streaming_templates() -> dict[str, int]:
    """Materialize the immutable exact continuation schedules used at d8--d10."""
    rows = 0
    live_bytes = 0
    for length in (2, 3, 4):
        for face in "URFDLB":
            table = _continuation_words(length, face)
            rows += len(table)
            live_bytes += table.nbytes
    return {"template_rows": rows, "template_live_bytes": live_bytes}


@dataclass(slots=True)
class PairFamilyStreamResult:
    status: str
    horizon: int
    mode: str
    backend: str
    solutions: tuple[tuple[str, ...], ...]
    metrics: dict[str, object]
    resource_cap_reason: str | None = None

    @property
    def first_witness(self) -> tuple[str, ...] | None:
        return self.solutions[0] if self.solutions else None

    @property
    def solution_hash(self) -> str:
        return _solution_hash(self.solutions)


def _common_masks(registries, keys: np.ndarray) -> np.ndarray:
    result = np.full(len(keys), (1 << len(MOVE_NAMES)) - 1, dtype=np.uint32)
    for column, registry in enumerate(registries):
        result &= registry.outgoing_mask[keys[:, column]]
    return result


def _advance_mixed(registries, keys: np.ndarray, moves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    result = np.empty_like(keys)
    valid = np.ones(len(keys), dtype=np.bool_)
    move_ids = np.asarray(moves, dtype=np.int64)
    for column, registry in enumerate(registries):
        target = registry.child[keys[:, column], move_ids]
        result[:, column] = target
        valid &= target != 0
    return result, valid


def _closure_codes(
    backend: str,
    registries,
    support: R4SparseSupportAsset,
    key: tuple[int, ...],
    *,
    first: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    if backend == "R4_SPARSE_WORD_SUPPORT":
        return support.close_codes(key, first=first), {"calls": 1, "memo_hits": 0, "memo_entries": 1}
    if backend in ("R4_RECURSIVE_PRODUCT", "R4_RECURSIVE_PRODUCT_WITH_EXACT_SIGNATURE"):
        return recursive_product_codes(registries, key, first=first)
    if backend == "R4_EXPLICIT_FOUR_LAYER_CONTROL_WITH_EXACT_SIGNATURE":
        values = explicit_four_layer_codes(registries, key)
        return values[:1] if first else values, {"calls": 1, "memo_hits": 0, "memo_entries": 0}
    raise ValueError(f"unsupported closure backend: {backend}")


def stream_packed_pair_families(
    atlas: ResidentPairAtlas,
    support: R4SparseSupportAsset,
    state: tuple[int, ...],
    horizon: int,
    *,
    mode: str = "ENUMERATE_ALL_SOLUTIONS",
    closure_backend: str = "R4_SPARSE_WORD_SUPPORT",
    batch_size: int = 4096,
    closure_memo_limit: int = 65_536,
    closure_audit_callback: Callable[[tuple[int, ...], np.ndarray], None] | None = None,
    previous_face: str | None = None,
) -> PairFamilyStreamResult:
    """Packed bounded-batch engine; it never constructs a cross-root frontier."""
    started = perf_counter()
    cpu_started = process_time()
    baseline_rss = _rss_mib()
    horizon = int(horizon)
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    if closure_backend not in BACKENDS:
        raise ValueError(f"unsupported closure backend: {closure_backend}")
    if horizon not in (8, 9, 10):
        return PairFamilyStreamResult(
            status="UNKNOWN_RESOURCE_CAP",
            horizon=horizon,
            mode=mode,
            backend=closure_backend,
            solutions=(),
            metrics={
                "query_wall": perf_counter() - started,
                "CPU_wall": process_time() - cpu_started,
                "teacher_reads": 0,
                "full_global_frontiers_materialized": 0,
                "remaining4_global_frontiers_materialized": 0,
            },
            resource_cap_reason="UNSUPPORTED_HORIZON",
        )
    if batch_size <= 0:
        return PairFamilyStreamResult(
            status="UNKNOWN_RESOURCE_CAP",
            horizon=horizon,
            mode=mode,
            backend=closure_backend,
            solutions=(),
            metrics={
                "query_wall": perf_counter() - started,
                "CPU_wall": process_time() - cpu_started,
                "teacher_reads": 0,
                "full_global_frontiers_materialized": 0,
                "remaining4_global_frontiers_materialized": 0,
            },
            resource_cap_reason="INVALID_BATCH_CAP",
        )

    if previous_face not in CONTEXTS:
        return PairFamilyStreamResult(
            status="ERROR", horizon=horizon, mode=mode, backend=closure_backend,
            solutions=(), metrics={
                "query_wall": perf_counter() - started,
                "CPU_wall": process_time() - cpu_started,
                "teacher_reads": 0,
                "full_global_frontiers_materialized": 0,
                "remaining4_global_frontiers_materialized": 0,
                "resource_cap_interpreted_as_UNSAT": 0,
            }, resource_cap_reason="MALFORMED_PREVIOUS_FACE",
        )
    # Do not mutate or duplicate the atlas.  Its physical root table already
    # contains all seven context dimensions.
    factors = tuple(
        registry.view(
            state[left], state[right], horizon,
            previous_face=previous_face,
            scope_hash=sha256(
                repr((tuple(state), horizon, previous_face, left, right)).encode()
            ).hexdigest(),
        )
        for registry, (left, right) in zip(atlas.registries, atlas.partition)
    )
    registries = atlas.registries
    roots = np.asarray([factor.root_node for factor in factors], dtype=np.result_type(*[registry.child.dtype for registry in registries]))
    common_root = (1 << len(MOVE_NAMES)) - 1
    for factor in factors:
        common_root &= int(factor.outgoing_mask[factor.root_node])
    root_moves = [move_id for move_id in range(len(MOVE_NAMES)) if common_root & (1 << move_id)]
    family_status = [
        {"move": MOVE_NAMES[move_id], "status": "UNVISITED"} for move_id in root_moves
    ]
    solutions: list[tuple[str, ...]] = []
    solution_seen: set[tuple[str, ...]] = set()
    closure_memo: dict[tuple[int, ...], np.ndarray] = {}
    metrics: dict[str, object] = {
        "root_families_available": len(root_moves),
        "root_families_opened": 0,
        "root_families_completed": 0,
        "root_families_unvisited": len(root_moves),
        "prefix_states_expanded": 0,
        "child_transitions_evaluated": 0,
        "closure_queries": 0,
        "closure_unique_tuples": 0,
        "closure_memo_hits": 0,
        "closure_signature_rejects": 0,
        "closure_full_support_checks": 0,
        "closure_empty": 0,
        "closure_nonempty": 0,
        "closure_suffix_count": 0,
        "closure_C1_recursive_calls": 0,
        "closure_C1_internal_memo_hits": 0,
        "peak_live_prefix_rows": 0,
        "peak_live_tuple_bytes": 0,
        "full_cross_root_frontiers_materialized": 0,
        "full_global_frontiers_materialized": 0,
        "remaining4_global_frontiers_materialized": 0,
        "all_move_global_concatenations": 0,
        "teacher_reads": 0,
        "resource_cap_interpreted_as_UNSAT": 0,
        "batch_size": int(batch_size),
        "closure_memo_limit": int(closure_memo_limit),
        "prefix_schedule": "ROOT_FAMILY_THEN_CANONICAL_BATCH_THEN_BOUNDARY_MOVE",
        "previous_face": previous_face,
        "context_in_scope_hash": True,
    }

    additional = horizon - 6
    stopped = False
    for family_index, root_move in enumerate(root_moves):
        metrics["root_families_opened"] = int(metrics["root_families_opened"]) + 1
        metrics["root_families_unvisited"] = len(root_moves) - int(metrics["root_families_opened"])
        family_status[family_index]["status"] = "OPEN"
        root_key = np.empty(10, dtype=roots.dtype)
        valid_root = True
        for column, registry in enumerate(registries):
            target = int(registry.child[int(roots[column]), root_move])
            root_key[column] = target
            valid_root &= target != 0
        metrics["child_transitions_evaluated"] = int(metrics["child_transitions_evaluated"]) + 1
        if not valid_root:
            family_status[family_index]["status"] = "COMPLETE_EMPTY"
            metrics["root_families_completed"] = int(metrics["root_families_completed"]) + 1
            continue
        continuations = _continuation_words(additional, MOVE_NAMES[root_move][0])
        for batch_start in range(0, len(continuations), int(batch_size)):
            words = continuations[batch_start : batch_start + int(batch_size)]
            keys = np.broadcast_to(root_key, (len(words), 10)).copy()
            live_words = words
            metrics["peak_live_prefix_rows"] = max(int(metrics["peak_live_prefix_rows"]), len(keys))
            metrics["peak_live_tuple_bytes"] = max(int(metrics["peak_live_tuple_bytes"]), int(keys.nbytes + live_words.nbytes))
            for column in range(additional):
                before = len(keys)
                keys, valid = _advance_mixed(registries, keys, live_words[:, column])
                metrics["prefix_states_expanded"] = int(metrics["prefix_states_expanded"]) + before
                metrics["child_transitions_evaluated"] = int(metrics["child_transitions_evaluated"]) + before
                keys = keys[valid]
                live_words = live_words[valid]
                if not len(keys):
                    break
            if not len(keys):
                continue
            common = _common_masks(registries, keys)
            if horizon == 8 and closure_backend == "R4_SPARSE_WORD_SUPPORT":
                # At d8 each root family has only 198 continuation rows.  A
                # single row-major flatten is cheaper than 18 tiny move
                # passes, while remaining comfortably below the batch cap.
                allowed = (
                    common[:, None]
                    & (np.uint32(1) << np.arange(len(MOVE_NAMES), dtype=np.uint32))
                ) != 0
                prefix_indices, boundary_moves = np.nonzero(allowed)
                conceptual_queries = len(prefix_indices)
                if conceptual_queries:
                    query_keys = np.empty((conceptual_queries, 10), dtype=keys.dtype)
                    parents = keys[prefix_indices]
                    for factor_index, registry in enumerate(registries):
                        query_keys[:, factor_index] = registry.child[
                            parents[:, factor_index], boundary_moves
                        ]
                    metrics["closure_queries"] = int(metrics["closure_queries"]) + conceptual_queries
                    metrics["child_transitions_evaluated"] = int(metrics["child_transitions_evaluated"]) + conceptual_queries
                    needs_full = support.signature_nonempty_mask(query_keys)
                    rejected = int(np.count_nonzero(~needs_full))
                    metrics["closure_signature_rejects"] = int(metrics["closure_signature_rejects"]) + rejected
                    metrics["closure_empty"] = int(metrics["closure_empty"]) + rejected
                    query_keys = query_keys[needs_full]
                    prefix_indices = prefix_indices[needs_full]
                    boundary_moves = boundary_moves[needs_full]
                    metrics["peak_live_tuple_bytes"] = max(
                        int(metrics["peak_live_tuple_bytes"]),
                        int(keys.nbytes + live_words.nbytes + query_keys.nbytes),
                    )
                    for query_index in range(len(query_keys)):
                        key = tuple(int(value) for value in query_keys[query_index])
                        cached = closure_memo.get(key)
                        if cached is not None:
                            codes = cached[:1] if mode == "FIRST_REPLAY_VALID_WITNESS" else cached
                            metrics["closure_memo_hits"] = int(metrics["closure_memo_hits"]) + 1
                        else:
                            codes, closure_metrics = _closure_codes(
                                closure_backend,
                                registries,
                                support,
                                key,
                                first=mode == "FIRST_REPLAY_VALID_WITNESS",
                            )
                            metrics["closure_unique_tuples"] = int(metrics["closure_unique_tuples"]) + 1
                            metrics["closure_C1_recursive_calls"] = int(metrics["closure_C1_recursive_calls"]) + int(closure_metrics["calls"])
                            metrics["closure_C1_internal_memo_hits"] = int(metrics["closure_C1_internal_memo_hits"]) + int(closure_metrics["memo_hits"])
                            if len(closure_memo) < int(closure_memo_limit):
                                closure_memo[key] = codes
                        if closure_audit_callback is not None:
                            closure_audit_callback(key, codes)
                        if not len(codes):
                            metrics["closure_empty"] = int(metrics["closure_empty"]) + 1
                            continue
                        metrics["closure_nonempty"] = int(metrics["closure_nonempty"]) + 1
                        metrics["closure_suffix_count"] = int(metrics["closure_suffix_count"]) + len(codes)
                        prefix_index = int(prefix_indices[query_index])
                        prefix = (MOVE_NAMES[root_move],) + tuple(
                            MOVE_NAMES[int(value)] for value in live_words[prefix_index]
                        ) + (MOVE_NAMES[int(boundary_moves[query_index])],)
                        for code in codes:
                            word = prefix + decode_names(int(code))
                            if not canonical_allow(previous_face, word[0]):
                                raise AssertionError(f"cross-boundary canonical mismatch: {word}")
                            if apply_word(state, word) != SOLVED:
                                raise AssertionError(f"pair streaming replay mismatch: {word}")
                            if word not in solution_seen:
                                solution_seen.add(word)
                                solutions.append(word)
                            if mode == "FIRST_REPLAY_VALID_WITNESS":
                                family_status[family_index]["status"] = "STOPPED_ON_WITNESS"
                                stopped = True
                                break
                        if stopped:
                            break
                if stopped:
                    break
                continue
            # Boundary moves are processed separately.  For C2 the exact
            # necessary signature is intersected factor-by-factor and dead
            # rows are released immediately, so ten-node remaining-4 tuples
            # are materialized only for the very small exact-candidate set.
            for boundary_move in range(len(MOVE_NAMES)):
                prefix_indices = np.flatnonzero(common & (1 << boundary_move))
                if not len(prefix_indices):
                    continue
                conceptual_queries = len(prefix_indices)
                metrics["closure_queries"] = int(metrics["closure_queries"]) + conceptual_queries
                metrics["child_transitions_evaluated"] = int(metrics["child_transitions_evaluated"]) + conceptual_queries
                if closure_backend in SIGNATURE_BACKENDS:
                    active = np.arange(conceptual_queries, dtype=np.int64)
                    signature = np.empty((conceptual_queries, 4), dtype=np.uint64)
                    for factor_index, (registry, factor_support) in enumerate(
                        zip(registries, support.factors)
                    ):
                        parent_rows = prefix_indices[active]
                        nodes = registry.child[
                            keys[parent_rows, factor_index], boundary_move
                        ].astype(np.int64, copy=False)
                        rows = factor_support.node_to_row[nodes]
                        if np.any(rows < 0):
                            raise RuntimeError("R4 closure-cache miss outside supported scope")
                        if factor_index == 0:
                            signature = factor_support.signatures[rows].copy()
                        else:
                            signature &= factor_support.signatures[rows]
                        keep = np.any(signature != 0, axis=1)
                        active = active[keep]
                        signature = signature[keep]
                        if not len(active):
                            break
                    rejected = conceptual_queries - len(active)
                    metrics["closure_signature_rejects"] = int(metrics["closure_signature_rejects"]) + rejected
                    metrics["closure_empty"] = int(metrics["closure_empty"]) + rejected
                    prefix_indices = prefix_indices[active]
                if not len(prefix_indices):
                    continue
                query_keys = np.empty((len(prefix_indices), 10), dtype=keys.dtype)
                parents = keys[prefix_indices]
                for factor_index, registry in enumerate(registries):
                    query_keys[:, factor_index] = registry.child[
                        parents[:, factor_index], boundary_move
                    ]
                metrics["peak_live_tuple_bytes"] = max(
                    int(metrics["peak_live_tuple_bytes"]),
                    int(keys.nbytes + live_words.nbytes + query_keys.nbytes),
                )
                for query_index in range(len(query_keys)):
                    key = tuple(int(value) for value in query_keys[query_index])
                    cached = closure_memo.get(key)
                    if cached is not None:
                        codes = cached[:1] if mode == "FIRST_REPLAY_VALID_WITNESS" else cached
                        metrics["closure_memo_hits"] = int(metrics["closure_memo_hits"]) + 1
                    else:
                        codes, closure_metrics = _closure_codes(
                            closure_backend,
                            registries,
                            support,
                            key,
                            first=mode == "FIRST_REPLAY_VALID_WITNESS",
                        )
                        metrics["closure_unique_tuples"] = int(metrics["closure_unique_tuples"]) + 1
                        metrics["closure_C1_recursive_calls"] = int(metrics["closure_C1_recursive_calls"]) + int(closure_metrics["calls"])
                        metrics["closure_C1_internal_memo_hits"] = int(metrics["closure_C1_internal_memo_hits"]) + int(closure_metrics["memo_hits"])
                        if len(closure_memo) < int(closure_memo_limit):
                            closure_memo[key] = codes
                    if closure_audit_callback is not None:
                        closure_audit_callback(key, codes)
                    if not len(codes):
                        metrics["closure_empty"] = int(metrics["closure_empty"]) + 1
                        continue
                    metrics["closure_nonempty"] = int(metrics["closure_nonempty"]) + 1
                    metrics["closure_suffix_count"] = int(metrics["closure_suffix_count"]) + len(codes)
                    prefix_index = int(prefix_indices[query_index])
                    prefix = (MOVE_NAMES[root_move],) + tuple(
                        MOVE_NAMES[int(value)] for value in live_words[prefix_index]
                    ) + (MOVE_NAMES[boundary_move],)
                    for code in codes:
                        word = prefix + decode_names(int(code))
                        if not canonical_allow(previous_face, word[0]):
                            raise AssertionError(f"cross-boundary canonical mismatch: {word}")
                        if apply_word(state, word) != SOLVED:
                            raise AssertionError(f"pair streaming replay mismatch: {word}")
                        if word not in solution_seen:
                            solution_seen.add(word)
                            solutions.append(word)
                        if mode == "FIRST_REPLAY_VALID_WITNESS":
                            family_status[family_index]["status"] = "STOPPED_ON_WITNESS"
                            stopped = True
                            break
                    if stopped:
                        break
                if stopped:
                    break
            if stopped:
                break
        if stopped:
            for later in range(family_index + 1, len(family_status)):
                family_status[later]["status"] = "UNVISITED_AFTER_WITNESS"
            break
        family_status[family_index]["status"] = "COMPLETE"
        metrics["root_families_completed"] = int(metrics["root_families_completed"]) + 1

    if mode == "ENUMERATE_ALL_SOLUTIONS":
        solutions = sorted(solution_seen)
    metrics["root_families_unvisited"] = sum(
        row["status"] in ("UNVISITED", "UNVISITED_AFTER_WITNESS") for row in family_status
    )
    metrics["family_status"] = family_status
    metrics["closure_full_support_checks"] = int(metrics["closure_unique_tuples"])
    metrics["peak_RSS_mib"] = _rss_mib()
    metrics["baseline_RSS_mib"] = baseline_rss
    metrics["RSS_growth_mib"] = max(0.0, _rss_mib() - baseline_rss)
    metrics["query_wall"] = perf_counter() - started
    metrics["CPU_wall"] = process_time() - cpu_started
    metrics["solution_count"] = len(solutions)
    metrics["solution_hash"] = _solution_hash(tuple(solutions))
    metrics["first_witness_replay"] = not solutions or apply_word(state, solutions[0]) == SOLVED
    return PairFamilyStreamResult(
        status="SAT" if solutions else "EXACT_EMPTY",
        horizon=horizon,
        mode=mode,
        backend=closure_backend,
        solutions=tuple(solutions),
        metrics=metrics,
    )


def stream_reference_pair_families(
    atlas: ResidentPairAtlas,
    support: R4SparseSupportAsset,
    state: tuple[int, ...],
    horizon: int,
    *,
    mode: str = "ENUMERATE_ALL_SOLUTIONS",
    closure_backend: str = "R4_SPARSE_WORD_SUPPORT",
    previous_face: str | None = None,
) -> PairFamilyStreamResult:
    """Transparent path-stack DFS reference, one root family at a time."""
    started = perf_counter()
    cpu_started = process_time()
    baseline_rss = _rss_mib()
    if horizon not in (8, 9, 10):
        return PairFamilyStreamResult(
            "UNKNOWN_RESOURCE_CAP", int(horizon), mode, closure_backend, (),
            {"teacher_reads": 0, "full_global_frontiers_materialized": 0, "remaining4_global_frontiers_materialized": 0},
            "UNSUPPORTED_HORIZON",
        )
    if previous_face not in CONTEXTS:
        return PairFamilyStreamResult(
            "ERROR", int(horizon), mode, closure_backend, (),
            {"teacher_reads": 0, "resource_cap_interpreted_as_UNSAT": 0},
            "MALFORMED_PREVIOUS_FACE",
        )
    factors = tuple(
        registry.view(
            state[left], state[right], int(horizon), previous_face=previous_face,
            scope_hash=sha256(
                repr((tuple(state), int(horizon), previous_face, left, right)).encode()
            ).hexdigest(),
        )
        for registry, (left, right) in zip(atlas.registries, atlas.partition)
    )
    registries = atlas.registries
    root = tuple(factor.root_node for factor in factors)
    solutions: list[tuple[str, ...]] = []
    metrics: dict[str, object] = {
        "root_families_opened": 0,
        "root_families_completed": 0,
        "root_families_unvisited": 0,
        "prefix_states_expanded": 0,
        "child_transitions_evaluated": 0,
        "closure_queries": 0,
        "closure_nonempty": 0,
        "closure_empty": 0,
        "closure_suffix_count": 0,
        "peak_live_prefix_rows": 1,
        "peak_live_tuple_bytes": 10 * np.dtype(np.uint32).itemsize,
        "full_cross_root_frontiers_materialized": 0,
        "full_global_frontiers_materialized": 0,
        "remaining4_global_frontiers_materialized": 0,
        "teacher_reads": 0,
        "previous_face": previous_face,
        "resource_cap_interpreted_as_UNSAT": 0,
    }
    root_common = (1 << len(MOVE_NAMES)) - 1
    for registry, node in zip(registries, root):
        root_common &= int(registry.outgoing_mask[node])
    root_moves = [move for move in range(len(MOVE_NAMES)) if root_common & (1 << move)]
    stopped = False

    def visit(key: tuple[int, ...], remaining: int, prefix: tuple[str, ...]) -> bool:
        metrics["prefix_states_expanded"] = int(metrics["prefix_states_expanded"]) + 1
        common = (1 << len(MOVE_NAMES)) - 1
        for registry, node in zip(registries, key):
            common &= int(registry.outgoing_mask[node])
        for move_id in range(len(MOVE_NAMES)):
            if not common & (1 << move_id):
                continue
            metrics["child_transitions_evaluated"] = int(metrics["child_transitions_evaluated"]) + 1
            child = tuple(
                int(registry.child[node, move_id])
                for registry, node in zip(registries, key)
            )
            word_prefix = prefix + (MOVE_NAMES[move_id],)
            if remaining == 5:
                metrics["closure_queries"] = int(metrics["closure_queries"]) + 1
                codes, _ = _closure_codes(
                    closure_backend,
                    registries,
                    support,
                    child,
                    first=mode == "FIRST_REPLAY_VALID_WITNESS",
                )
                if not len(codes):
                    metrics["closure_empty"] = int(metrics["closure_empty"]) + 1
                    continue
                metrics["closure_nonempty"] = int(metrics["closure_nonempty"]) + 1
                metrics["closure_suffix_count"] = int(metrics["closure_suffix_count"]) + len(codes)
                for code in codes:
                    word = word_prefix + decode_names(int(code))
                    if not canonical_allow(previous_face, word[0]):
                        raise AssertionError("reference cross-boundary canonical mismatch")
                    if apply_word(state, word) != SOLVED:
                        raise AssertionError("reference streaming replay mismatch")
                    solutions.append(word)
                    if mode == "FIRST_REPLAY_VALID_WITNESS":
                        return True
            elif visit(child, remaining - 1, word_prefix):
                return True
        return False

    for family_index, root_move in enumerate(root_moves):
        metrics["root_families_opened"] = int(metrics["root_families_opened"]) + 1
        child = tuple(
            int(registry.child[node, root_move])
            for registry, node in zip(registries, root)
        )
        metrics["child_transitions_evaluated"] = int(metrics["child_transitions_evaluated"]) + 1
        if visit(child, horizon - 1, (MOVE_NAMES[root_move],)):
            metrics["root_families_unvisited"] = len(root_moves) - family_index - 1
            stopped = True
            break
        metrics["root_families_completed"] = int(metrics["root_families_completed"]) + 1
    if not stopped:
        metrics["root_families_unvisited"] = 0
    if mode == "ENUMERATE_ALL_SOLUTIONS":
        solutions = sorted(set(solutions))
    metrics.update(
        {
            "query_wall": perf_counter() - started,
            "CPU_wall": process_time() - cpu_started,
            "baseline_RSS_mib": baseline_rss,
            "peak_RSS_mib": _rss_mib(),
            "RSS_growth_mib": max(0.0, _rss_mib() - baseline_rss),
            "solution_count": len(solutions),
            "solution_hash": _solution_hash(tuple(solutions)),
            "first_witness_replay": not solutions or apply_word(state, solutions[0]) == SOLVED,
        }
    )
    return PairFamilyStreamResult(
        "SAT" if solutions else "EXACT_EMPTY",
        int(horizon),
        mode,
        closure_backend,
        tuple(solutions),
        metrics,
    )


__all__ = [
    "BACKENDS",
    "MODES",
    "PairFamilyStreamResult",
    "stream_packed_pair_families",
    "stream_reference_pair_families",
    "warm_streaming_templates",
]
