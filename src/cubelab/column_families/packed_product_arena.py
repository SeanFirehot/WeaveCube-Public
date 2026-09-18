"""Packed exact-key synchronized residual-product backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import resource
from time import perf_counter
from typing import Protocol

import numpy as np

from cubelab.ato.regular_support import MOVE_NAMES

from .q_product_reference import ExactQViability
from .residual_factor_view import CONTEXT_INDEX, ResidualFactorView
from .symbolic_mdd_v2 import ResourceCapUnknown


def _rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / 1024.0 if value < 10**9 else value / (1024.0 * 1024.0))


class PackedAdapter(Protocol):
    representation: str
    pieces: tuple[int, ...]
    root_key: np.ndarray

    def common_mask(self, keys: np.ndarray, remaining: int) -> np.ndarray: ...
    def child_keys(self, keys: np.ndarray, move_id: int) -> np.ndarray: ...
    def accepting(self, keys: np.ndarray) -> np.ndarray: ...


@dataclass(slots=True)
class ResidualPackedAdapter:
    factors: tuple[ResidualFactorView, ...]
    grouped_masks: bool = False
    representation: str = field(init=False)
    pieces: tuple[int, ...] = field(init=False)
    root_key: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if not self.factors:
            raise ValueError("at least one residual factor is required")
        horizons = {factor.horizon for factor in self.factors}
        scopes = {factor.scope_hash for factor in self.factors if factor.scope_hash}
        if len(horizons) != 1 or len(scopes) > 1:
            raise ValueError("residual factors do not share horizon/scope")
        dtype = np.uint16
        if any(factor.node_id_dtype == "uint32" for factor in self.factors):
            dtype = np.uint32
        self.root_key = np.asarray(
            [factor.root_node for factor in self.factors], dtype=dtype
        )
        self.pieces = tuple(factor.piece for factor in self.factors)
        self.representation = (
            "RESIDUAL_HIERARCHICAL_8_12" if self.grouped_masks else "RESIDUAL_NODE_PRODUCT"
        )

    def _mask_range(self, keys: np.ndarray, columns: range) -> np.ndarray:
        mask = np.full(len(keys), (1 << len(MOVE_NAMES)) - 1, dtype=np.uint32)
        for column in columns:
            factor = self.factors[column]
            mask &= factor.outgoing_mask[keys[:, column]]
        return mask

    def common_mask(self, keys: np.ndarray, remaining: int) -> np.ndarray:
        if self.grouped_masks and len(self.factors) == 20:
            corners = self._mask_range(keys, range(8))
            edges = self._mask_range(keys, range(8, 20))
            return corners & edges
        return self._mask_range(keys, range(len(self.factors)))

    def child_keys(self, keys: np.ndarray, move_id: int) -> np.ndarray:
        result = np.empty_like(keys)
        for column, factor in enumerate(self.factors):
            result[:, column] = factor.child[keys[:, column], int(move_id)]
        return result

    def accepting(self, keys: np.ndarray) -> np.ndarray:
        result = np.ones(len(keys), dtype=np.bool_)
        for column, factor in enumerate(self.factors):
            result &= factor.accepting[keys[:, column]]
        return result


@dataclass(slots=True)
class QPackedAdapter:
    start_q: tuple[int, ...]
    table: ExactQViability
    horizon: int
    previous_face: str | None = None
    pieces: tuple[int, ...] = tuple(range(20))
    representation: str = "EXACT_Q_PRODUCT"
    root_key: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if len(self.start_q) != len(self.pieces):
            raise ValueError("Q state/piece shape mismatch")
        self.root_key = np.asarray(
            [CONTEXT_INDEX[self.previous_face], *self.start_q], dtype=np.uint8
        )

    def common_mask(self, keys: np.ndarray, remaining: int) -> np.ndarray:
        return self.table.common_mask(keys[:, 1:], remaining, keys[:, 0], self.pieces)

    def child_keys(self, keys: np.ndarray, move_id: int) -> np.ndarray:
        child_q = self.table.child_states(keys[:, 1:], move_id, self.pieces)
        result = np.empty_like(keys)
        result[:, 0] = CONTEXT_INDEX[MOVE_NAMES[int(move_id)][0]]
        result[:, 1:] = child_q
        return result

    def accepting(self, keys: np.ndarray) -> np.ndarray:
        return np.all(keys[:, 1:] == 0, axis=1)


@dataclass(slots=True)
class ProductEdgeLayer:
    source: np.ndarray
    move: np.ndarray
    child: np.ndarray


@dataclass(slots=True)
class PackedProductResult:
    representation: str
    mode: str
    horizon: int
    pieces: tuple[int, ...]
    layers: list[np.ndarray]
    prefix_counts: list[np.ndarray]
    edges: list[ProductEdgeLayer]
    productive: list[np.ndarray]
    path_counts: list[np.ndarray]
    solutions: tuple[tuple[str, ...], ...]
    metrics: dict[str, object]
    status: str = "COMPLETE"

    def structural_hash(self) -> str:
        if not self.productive or not bool(self.productive[0][0]):
            return sha256(b"EMPTY_PRODUCT").hexdigest()
        hashes: list[dict[int, str]] = [dict() for _ in self.layers]
        for node in np.flatnonzero(self.productive[-1]):
            hashes[-1][int(node)] = sha256(b"TRUE").hexdigest()
        for depth in range(self.horizon - 1, -1, -1):
            edge_layer = self.edges[depth]
            good = self.productive[depth + 1][edge_layer.child]
            for source in np.flatnonzero(self.productive[depth]):
                selected = np.flatnonzero(good & (edge_layer.source == source))
                payload = f"{self.horizon-depth}|" + ";".join(
                    f"{int(edge_layer.move[row])}:{hashes[depth+1][int(edge_layer.child[row])]}"
                    for row in selected[np.argsort(edge_layer.move[selected])]
                )
                hashes[depth][int(source)] = sha256(payload.encode()).hexdigest()
        return hashes[0][0]

    def to_bytes(self) -> bytes:
        payload = {
            "representation": self.representation,
            "mode": self.mode,
            "horizon": self.horizon,
            "pieces": list(self.pieces),
            "solutions": [list(word) for word in self.solutions],
            "metrics": self.metrics,
            "structural_hash": self.structural_hash(),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _exact_unique(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    packed = np.ascontiguousarray(rows)
    byte_rows = packed.view(
        np.dtype((np.void, packed.dtype.itemsize * packed.shape[1]))
    ).ravel()
    _values, first, inverse = np.unique(
        byte_rows, return_index=True, return_inverse=True
    )
    return packed[first], first.astype(np.int64), inverse.astype(np.int64)


def _enumerate_productive(
    edges: list[ProductEdgeLayer],
    productive: list[np.ndarray],
    horizon: int,
) -> tuple[tuple[str, ...], ...]:
    result: list[tuple[str, ...]] = []

    def visit(depth: int, node: int, prefix: tuple[str, ...]) -> None:
        if depth == horizon:
            result.append(prefix)
            return
        layer = edges[depth]
        selected = np.flatnonzero(
            (layer.source == int(node)) & productive[depth + 1][layer.child]
        )
        for row in selected[np.argsort(layer.move[selected])]:
            visit(
                depth + 1,
                int(layer.child[row]),
                prefix + (MOVE_NAMES[int(layer.move[row])],),
            )

    if productive and bool(productive[0][0]):
        visit(0, 0, ())
    return tuple(result)


def build_packed_product(
    adapter: PackedAdapter,
    horizon: int,
    *,
    mode: str = "BUILD_ALL_FAMILY_DAG",
    node_cap: int = 5_000_000,
    time_cap_seconds: float | None = None,
    rss_growth_cap_mib: float = 1536.0,
) -> PackedProductResult:
    """Construct an exact synchronized product with exact packed key dedup."""

    started = perf_counter()
    baseline_rss = _rss_mib()
    horizon = int(horizon)
    layers = [np.asarray(adapter.root_key[None, :])]
    prefix_counts = [np.asarray([1], dtype=np.int64)]
    edges: list[ProductEdgeLayer] = []
    frontier_counts = [1]
    cardinality_histograms: list[dict[str, int]] = []
    common_all18 = []
    cache_hits = 0
    total_edges = 0
    peak_live = 1
    peak_key_bytes = int(layers[0].nbytes)

    for depth in range(horizon):
        if time_cap_seconds is not None and perf_counter() - started > time_cap_seconds:
            raise ResourceCapUnknown("lazy product time cap reached")
        if _rss_mib() - baseline_rss > float(rss_growth_cap_mib):
            raise ResourceCapUnknown("lazy product RSS growth cap reached")
        remaining = horizon - depth
        current = layers[-1]
        masks = adapter.common_mask(current, remaining)
        cardinalities = np.bit_count(masks) if hasattr(np, "bit_count") else np.asarray(
            [int(value).bit_count() for value in masks], dtype=np.uint8
        )
        unique_card, card_counts = np.unique(cardinalities, return_counts=True)
        cardinality_histograms.append(
            {str(int(card)): int(count) for card, count in zip(unique_card, card_counts)}
        )
        common_all18.append(int(np.count_nonzero(cardinalities == len(MOVE_NAMES))))
        candidate_keys: list[np.ndarray] = []
        candidate_source: list[np.ndarray] = []
        candidate_move: list[np.ndarray] = []
        for move_id in range(len(MOVE_NAMES)):
            selected = np.flatnonzero((masks & np.uint32(1 << move_id)) != 0)
            if not len(selected):
                continue
            candidate_keys.append(adapter.child_keys(current[selected], move_id))
            candidate_source.append(selected.astype(np.uint32))
            candidate_move.append(np.full(len(selected), move_id, dtype=np.uint8))
        if candidate_keys:
            joined_keys = np.concatenate(candidate_keys, axis=0)
            joined_source = np.concatenate(candidate_source)
            joined_move = np.concatenate(candidate_move)
            next_keys, _first, inverse = _exact_unique(joined_keys)
            next_counts = np.zeros(len(next_keys), dtype=np.int64)
            np.add.at(next_counts, inverse, prefix_counts[-1][joined_source])
            edge_layer = ProductEdgeLayer(
                source=joined_source,
                move=joined_move,
                child=inverse.astype(np.uint32),
            )
        else:
            next_keys = np.empty((0, current.shape[1]), dtype=current.dtype)
            next_counts = np.empty(0, dtype=np.int64)
            edge_layer = ProductEdgeLayer(
                source=np.empty(0, dtype=np.uint32),
                move=np.empty(0, dtype=np.uint8),
                child=np.empty(0, dtype=np.uint32),
            )
        total_edges += len(edge_layer.source)
        cache_hits += len(edge_layer.source) - len(next_keys)
        layers.append(next_keys)
        prefix_counts.append(next_counts)
        edges.append(edge_layer)
        frontier_counts.append(len(next_keys))
        peak_live = max(peak_live, len(current) + len(next_keys) + len(edge_layer.source))
        peak_key_bytes = max(
            peak_key_bytes,
            int(current.nbytes + next_keys.nbytes + sum(value.nbytes for value in candidate_keys)),
        )
        if sum(frontier_counts) > int(node_cap):
            raise ResourceCapUnknown(f"lazy product node cap {node_cap} reached")

    productive = [np.zeros(len(layer), dtype=np.bool_) for layer in layers]
    path_counts = [np.zeros(len(layer), dtype=np.int64) for layer in layers]
    if len(layers[-1]):
        productive[-1] = adapter.accepting(layers[-1])
        path_counts[-1][productive[-1]] = 1
    productive_edges = 0
    for depth in range(horizon - 1, -1, -1):
        layer = edges[depth]
        good = productive[depth + 1][layer.child]
        productive_edges += int(np.count_nonzero(good))
        if np.any(good):
            np.logical_or.at(productive[depth], layer.source[good], True)
            np.add.at(
                path_counts[depth],
                layer.source[good],
                path_counts[depth + 1][layer.child[good]],
            )
    solutions = _enumerate_productive(edges, productive, horizon)
    solution_count = int(path_counts[0][0]) if len(path_counts[0]) else 0
    if len(solutions) != solution_count:
        raise AssertionError("packed product solution enumeration/count mismatch")
    merge_hist = {
        str(horizon - depth): int(np.count_nonzero(prefix_counts[depth] > 1))
        for depth in range(1, horizon)
    }
    nonterminal_merges = sum(merge_hist.values())
    total_states = sum(frontier_counts)
    productive_states = sum(int(np.count_nonzero(row)) for row in productive)
    wall = perf_counter() - started
    metrics: dict[str, object] = {
        "product_states_created": total_states,
        "product_state_cache_hits": cache_hits,
        "product_state_cache_misses": total_states,
        "product_edges_created": total_edges,
        "productive_product_edges": productive_edges,
        "dead_product_states": total_states - productive_states,
        "accepting_product_states": int(np.count_nonzero(productive[-1])),
        "frontier_states_by_depth": frontier_counts,
        "outgoing_mask_cardinality_histogram_by_depth": cardinality_histograms,
        "common_mask_all18_states_by_depth": common_all18,
        "prefixes_represented_by_product_state": [
            {
                "depth": depth,
                "prefixes": int(row.sum()),
                "states": len(row),
                "maximum": int(row.max(initial=0)),
            }
            for depth, row in enumerate(prefix_counts)
        ],
        "real_nonterminal_merges_by_depth": merge_hist,
        "real_nonterminal_merge_count": nonterminal_merges,
        "peak_live_product_states": peak_live,
        "peak_product_key_bytes": peak_key_bytes,
        "peak_RSS_mib": _rss_mib(),
        "RSS_growth_mib": max(0.0, _rss_mib() - baseline_rss),
        "solution_count": solution_count,
        "wall_product_build": wall,
        "pairwise_intermediate_MDDs_materialized": 0,
        "pairwise_intermediate_nodes_materialized": 0,
        "full_canonical_words_retained": 0,
        "exact_key_comparison": True,
        "lossy_hashing": False,
        "node_id_dtype": str(layers[0].dtype),
    }
    return PackedProductResult(
        representation=adapter.representation,
        mode=mode,
        horizon=horizon,
        pieces=adapter.pieces,
        layers=layers,
        prefix_counts=prefix_counts,
        edges=edges,
        productive=productive,
        path_counts=path_counts,
        solutions=solutions,
        metrics=metrics,
    )


def first_packed_witness(
    adapter: PackedAdapter,
    horizon: int,
    *,
    node_cap: int = 5_000_000,
    time_cap_seconds: float | None = None,
    rss_growth_cap_mib: float = 1536.0,
) -> dict[str, object]:
    """Find one deterministic exact witness without claiming enumeration."""

    started = perf_counter()
    baseline_rss = _rss_mib()
    current = np.asarray(adapter.root_key[None, :])
    prefix_codes = np.asarray([0], dtype=np.int64)
    prefix_counts = np.asarray([1], dtype=np.int64)
    states_created = 1
    edges_created = 0
    frontier_counts = [1]
    cache_hits = 0
    found_code: int | None = None
    final_candidate_count = 0
    for depth in range(int(horizon)):
        if time_cap_seconds is not None and perf_counter() - started > time_cap_seconds:
            raise ResourceCapUnknown("lazy first-witness time cap reached")
        if _rss_mib() - baseline_rss > float(rss_growth_cap_mib):
            raise ResourceCapUnknown("lazy first-witness RSS growth cap reached")
        remaining = int(horizon) - depth
        masks = adapter.common_mask(current, remaining)
        key_parts = []
        code_parts = []
        count_parts = []
        for move_id in range(len(MOVE_NAMES)):
            selected = np.flatnonzero((masks & np.uint32(1 << move_id)) != 0)
            if not len(selected):
                continue
            key_parts.append(adapter.child_keys(current[selected], move_id))
            code_parts.append(prefix_codes[selected] * len(MOVE_NAMES) + move_id)
            count_parts.append(prefix_counts[selected])
        if not key_parts:
            current = np.empty((0, current.shape[1]), dtype=current.dtype)
            frontier_counts.append(0)
            break
        joined = np.concatenate(key_parts, axis=0)
        joined_codes = np.concatenate(code_parts)
        joined_counts = np.concatenate(count_parts)
        edges_created += len(joined)
        unique, _first, inverse = _exact_unique(joined)
        minimum_codes = np.full(len(unique), np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(minimum_codes, inverse, joined_codes)
        next_counts = np.zeros(len(unique), dtype=np.int64)
        np.add.at(next_counts, inverse, joined_counts)
        order = np.argsort(minimum_codes, kind="stable")
        current = unique[order]
        prefix_codes = minimum_codes[order]
        prefix_counts = next_counts[order]
        cache_hits += len(joined) - len(unique)
        states_created += len(unique)
        frontier_counts.append(len(unique))
        if states_created > int(node_cap):
            raise ResourceCapUnknown(f"lazy first-witness node cap {node_cap} reached")
        if depth + 1 == int(horizon):
            accepting = adapter.accepting(current)
            final_candidate_count = int(prefix_counts[accepting].sum())
            if np.any(accepting):
                found_code = int(prefix_codes[np.flatnonzero(accepting)[0]])
            break
    witness: tuple[str, ...] | None = None
    if found_code is not None:
        values = [0] * int(horizon)
        code = found_code
        for column in range(int(horizon) - 1, -1, -1):
            values[column] = code % len(MOVE_NAMES)
            code //= len(MOVE_NAMES)
        witness = tuple(MOVE_NAMES[value] for value in values)
    return {
        "status": "SAT" if witness is not None else "UNSAT",
        "witness": None if witness is None else list(witness),
        "all_solutions_enumerated": False,
        "unvisited_exact_families": max(0, final_candidate_count - 1),
        "product_states_created": states_created,
        "product_state_cache_hits": cache_hits,
        "product_edges_created": edges_created,
        "frontier_states_by_depth": frontier_counts,
        "pairwise_intermediate_MDDs_materialized": 0,
        "pairwise_intermediate_nodes_materialized": 0,
        "wall_first_witness": perf_counter() - started,
        "peak_RSS_mib": _rss_mib(),
        "RSS_growth_mib": max(0.0, _rss_mib() - baseline_rss),
    }


__all__ = [
    "PackedProductResult",
    "ProductEdgeLayer",
    "QPackedAdapter",
    "ResidualPackedAdapter",
    "build_packed_product",
    "first_packed_witness",
]
