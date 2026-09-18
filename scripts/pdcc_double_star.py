#!/usr/bin/env python3
"""Fast exact double-star projection queries on the phase-2 coordinate.

The retained 12 PDBs cover adjacent corner-star unions (2 corners + 5 edges).
A second retained index cache maps the full P2 permutation coordinates directly
onto each PDB's partial coordinate, avoiding per-node permutation inversion.
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import search_twist_skeleton as ts

DEFAULT_CACHE = Path("reports/pdcc_cache/double_star_k_tables_v1.pkl")
DEFAULT_INDEX = Path("reports/pdcc_cache/double_star_p2_index_v1.pkl")
_CACHE_PATH = DEFAULT_CACHE
_INDEX_PATH = DEFAULT_INDEX
_TABLES = None
_CMAP = _UMAP = _SMAP = None


def configure(cache_path: str | Path | None = None, index_path: str | Path | None = None):
    global _CACHE_PATH, _INDEX_PATH, _TABLES
    if cache_path is not None:
        p = Path(cache_path)
        if _TABLES is not None and p != _CACHE_PATH:
            raise RuntimeError("double-star tables already loaded from another path")
        _CACHE_PATH = p
    if index_path is not None:
        p = Path(index_path)
        if _TABLES is not None and p != _INDEX_PATH:
            raise RuntimeError("double-star index already loaded from another path")
        _INDEX_PATH = p


def _ensure_loaded():
    global _TABLES, _CMAP, _UMAP, _SMAP
    if _TABLES is not None:
        return
    with _CACHE_PATH.open("rb") as f:
        payload = pickle.load(f)
    tabs = tuple(payload.get("tables", ()))
    if payload.get("version") != 1 or len(tabs) != 12:
        raise RuntimeError(f"invalid double-star cache: {_CACHE_PATH}")
    with _INDEX_PATH.open("rb") as f:
        idx = pickle.load(f)
    cm = tuple(idx.get("corner_maps", ()))
    um = tuple(idx.get("ud_maps", ()))
    sm = tuple(idx.get("slice_maps", ()))
    if idx.get("version") != 1 or not (len(cm) == len(um) == len(sm) == 12):
        raise RuntimeError(f"invalid double-star P2 index: {_INDEX_PATH}")
    _TABLES, _CMAP, _UMAP, _SMAP = tabs, cm, um, sm


@lru_cache(maxsize=500_000)
def profile(key: int) -> tuple[int, ...]:
    """Sorted descending exact distances of the 12 double-star projections."""
    if key == ts.GOAL_P:
        return (0,) * 12
    _ensure_loaded()
    c, u, s = ts.unpack_p(key)
    ds = []
    for j, tab in enumerate(_TABLES):
        ci = _CMAP[j][c]
        ui = _UMAP[j][u]
        si = _SMAP[j][s]
        idx = (ci * tab["nu"] + ui) * tab["ns"] + si
        ds.append(tab["dist"][idx])
    return tuple(sorted(ds, reverse=True))


def lower_bound(key: int) -> int:
    """Admissible K-distance lower bound from the strongest projection."""
    return profile(key)[0] if key != ts.GOAL_P else 0
