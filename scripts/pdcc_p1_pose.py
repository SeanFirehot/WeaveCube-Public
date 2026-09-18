#!/usr/bin/env python3
"""Exact full-solution corner-star pose projections for P1/global pruning.

Each retained projection tracks one native corner star:

    corner + its three solved-adjacent edges

using the exact PDCC pose IDs W_i of those four pieces.  A projection state is
therefore a 4-tuple of pose IDs in [0,23].  The solved projection is (I,I,I,I).
The table distance is the exact HTM distance in the projected 18-move graph to
solve those four tracked pieces simultaneously.

For every full cube state s and star j,

    d_star_j(proj_j(s)) <= d_HTM(s, solved)

because every full solution also solves the projected four pieces.  Therefore

    h_star(s) = max_j d_star_j(proj_j(s))

is an admissible lower bound on the *full remaining solution distance*.

This is deliberately NOT a DR-distance bound: full piece identity/pose is not
needed to define the DR quotient.  The bound is applied during P1 search only
as a global incumbent prune: if g + h_star >= incumbent, the branch cannot
contain a strictly shorter full solution.
"""
from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path

import search_twist_skeleton as ts
import search_pdcc_guided as pg
from cubelab.pdcc import CORNER_NAMES, NATIVE_ADJACENCY, PIECE_NAMES
from cubelab.pdcc.model import PIECE_INDEX

VERSION = 1
DEFAULT_CACHE = Path("reports/pdcc_cache/p1_corner_star_tables_v1.pkl")
_CACHE_PATH = DEFAULT_CACHE
_TABLES: tuple[bytearray, ...] | None = None

# Build the eight native corner stars directly from the solved adjacency graph.
_adj = {p: [] for p in PIECE_NAMES}
for _a, _b in NATIVE_ADJACENCY:
    _adj[_a].append(_b)
    _adj[_b].append(_a)
CORNER_STARS = tuple((c, *sorted(_adj[c])) for c in CORNER_NAMES)
STAR_INDICES = tuple(tuple(PIECE_INDEX[p] for p in star) for star in CORNER_STARS)


def configure(cache_path: str | Path | None = None) -> None:
    global _CACHE_PATH, _TABLES
    if cache_path is None:
        return
    p = Path(cache_path)
    if _TABLES is not None and p != _CACHE_PATH:
        raise RuntimeError("P1 corner-star tables already loaded from another path")
    _CACHE_PATH = p


def _build_one(indices: tuple[int, ...]) -> bytearray:
    """BFS the exact 4-piece projection graph from the solved projection."""
    if len(indices) != 4:
        raise ValueError("corner-star projection must contain four pieces")
    size = 24 ** 4
    dist = bytearray([255]) * size
    dist[0] = 0
    queue = deque([0])

    while queue:
        x = queue.popleft()
        nd = dist[x] + 1
        z = x
        p0 = z % 24; z //= 24
        p1 = z % 24; z //= 24
        p2 = z % 24; z //= 24
        p3 = z % 24
        i0, i1, i2, i3 = indices
        for mi in range(len(ts.MOVE_ORDER)):
            trans = pg.POSE_NEXT[mi]
            y0 = trans[i0][p0]
            y1 = trans[i1][p1]
            y2 = trans[i2][p2]
            y3 = trans[i3][p3]
            y = y0 + 24 * (y1 + 24 * (y2 + 24 * y3))
            if dist[y] == 255:
                dist[y] = nd
                queue.append(y)
    return dist


def _build_all() -> tuple[bytearray, ...]:
    tables = []
    for i, indices in enumerate(STAR_INDICES, 1):
        print(f"building P1 corner-star pose PDB {i}/8...", flush=True)
        tables.append(_build_one(indices))
    return tuple(tables)


def _ensure_loaded() -> None:
    global _TABLES
    if _TABLES is not None:
        return

    if _CACHE_PATH.exists():
        try:
            with _CACHE_PATH.open("rb") as fh:
                payload = pickle.load(fh)
            tables = tuple(payload.get("tables", ()))
            clusters = tuple(tuple(x) for x in payload.get("clusters", ()))
            if payload.get("version") == VERSION and len(tables) == 8 and clusters == CORNER_STARS:
                _TABLES = tables
                return
        except Exception:
            pass

    tables = _build_all()
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_PATH.open("wb") as fh:
        pickle.dump(
            {"version": VERSION, "clusters": CORNER_STARS, "tables": tables},
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    _TABLES = tables


def preload() -> None:
    _ensure_loaded()


def _index4(poses: tuple[int, ...], inds: tuple[int, int, int, int]) -> int:
    a, b, c, d = inds
    return poses[a] + 24 * (poses[b] + 24 * (poses[c] + 24 * poses[d]))


def profile(poses: tuple[int, ...]) -> tuple[int, ...]:
    """Eight exact corner-star projected distances, sorted descending."""
    _ensure_loaded()
    assert _TABLES is not None
    vals = [tab[_index4(poses, inds)] for tab, inds in zip(_TABLES, STAR_INDICES)]
    return tuple(sorted(vals, reverse=True))


def lower_bound(poses: tuple[int, ...]) -> int:
    """Admissible lower bound on remaining full HTM solution distance."""
    _ensure_loaded()
    assert _TABLES is not None
    best = 0
    for tab, inds in zip(_TABLES, STAR_INDICES):
        d = tab[_index4(poses, inds)]
        if d > best:
            best = d
    return best
