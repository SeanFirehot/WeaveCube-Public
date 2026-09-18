"""Exact per-piece and grouped Q-language construction."""

from __future__ import annotations

from hashlib import sha256
from time import perf_counter

from cubelab.ato.column_grounding import Q_NEXT
from cubelab.ato.pose24 import Q_TO_OMEGA
from cubelab.pdcc.model import PIECE_NAMES
from cubelab.pdcc.state import slot_for_pose

from .symbolic_mdd_v2 import ResourceCapUnknown, SymbolicMDD


def piece_pose_language(
    piece: int,
    initial_q: int,
    horizon: int,
    *,
    previous_face: str | None = None,
    scope_hash: str = "",
    node_cap: int = 5_000_000,
) -> SymbolicMDD:
    piece = int(piece)
    return SymbolicMDD.from_automaton(
        int(initial_q),
        int(horizon),
        lambda q, move_id: int(Q_NEXT[piece, int(q), int(move_id)]),
        lambda q: int(q) == 0,
        previous_face=previous_face,
        scope_hash=str(scope_hash),
        node_cap=node_cap,
        construction=f"PIECE_POSE_{piece}_{PIECE_NAMES[piece]}",
    )


def piece_orientation_language(
    piece: int,
    initial_q: int,
    horizon: int,
    *,
    previous_face: str | None = None,
    scope_hash: str = "",
    node_cap: int = 5_000_000,
) -> SymbolicMDD:
    piece = int(piece)
    return SymbolicMDD.from_automaton(
        int(initial_q),
        int(horizon),
        lambda q, move_id: int(Q_NEXT[piece, int(q), int(move_id)]),
        lambda q: int(Q_TO_OMEGA[int(q)]) == 0,
        previous_face=previous_face,
        scope_hash=str(scope_hash),
        node_cap=node_cap,
        construction=f"PIECE_ORIENTATION_{piece}_{PIECE_NAMES[piece]}",
    )


def piece_permutation_language(
    piece: int,
    initial_q: int,
    horizon: int,
    *,
    previous_face: str | None = None,
    scope_hash: str = "",
    node_cap: int = 5_000_000,
) -> SymbolicMDD:
    piece = int(piece)
    piece_name = PIECE_NAMES[piece]
    return SymbolicMDD.from_automaton(
        int(initial_q),
        int(horizon),
        lambda q, move_id: int(Q_NEXT[piece, int(q), int(move_id)]),
        lambda q: slot_for_pose(piece_name, int(q)) == piece_name,
        previous_face=previous_face,
        scope_hash=str(scope_hash),
        node_cap=node_cap,
        construction=f"PIECE_PERMUTATION_{piece}_{piece_name}",
    )


def intersect_languages(
    languages: list[tuple[str, SymbolicMDD]],
    *,
    node_cap: int = 5_000_000,
    schedule: str = "DECLARED",
) -> tuple[SymbolicMDD, list[dict[str, object]]]:
    if not languages:
        raise ValueError("at least one language is required")
    rows = list(languages)
    if schedule == "SMALLEST_FIRST":
        rows.sort(key=lambda item: (item[1].node_count, item[1].path_count(), item[0]))
    current_name, current = rows[0]
    anatomy = [
        {
            "factor_added": current_name,
            "paths_before": None,
            "paths_after": current.path_count(),
            "node_count": current.node_count,
            "intersection_wall_seconds": 0.0,
            "peak_nodes": current.node_count,
        }
    ]
    peak = current.node_count
    for name, language in rows[1:]:
        started = perf_counter()
        before = current.path_count()
        current = current.intersect(language, node_cap=node_cap)
        peak = max(peak, current.node_count)
        anatomy.append(
            {
                "factor_added": name,
                "paths_before": before,
                "paths_after": current.path_count(),
                "node_count": current.node_count,
                "intersection_wall_seconds": perf_counter() - started,
                "peak_nodes": peak,
            }
        )
    return current, anatomy


__all__ = [
    "intersect_languages",
    "piece_orientation_language",
    "piece_permutation_language",
    "piece_pose_language",
]
