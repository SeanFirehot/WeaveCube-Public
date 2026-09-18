"""Exact remaining-four closure kernels for the selected v37.359 pair atlas.

The sparse asset stores one sorted uint32 support list for every productive
remaining-four node.  A 256-bit necessary signature is stored beside each
list; a zero ten-way signature intersection is therefore an exact empty
certificate, while non-zero intersections are always checked against the
full sorted supports.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from cubelab.ato.regular_support import MOVE_NAMES
from cubelab.pdcc.model import PIECE_NAMES

from .pair_residual_factor_v37_358 import (
    CANONICAL_SEMANTICS_HASH,
    Q_TRANSITION_HASH,
    PairResidualFactorRegistry,
)


SCHEMA = "cubelab.exact-pair-r4-suffix-support.v37.359.1"
ENCODING = "BASE18_BIG_ENDIAN_R4_V1"
CLOSURE_VERSION = "R4_C1_C2_V37_359_1"
WORD_BASE = len(MOVE_NAMES)
SUFFIX_LENGTH = 4
SIGNATURE_WORDS = 4
SIGNATURE_BITS = SIGNATURE_WORDS * 64


class R4ClosureAssetMismatch(ValueError):
    """Raised when an R4 cache is incomplete or bound to different semantics."""


def encode_moves(move_ids: tuple[int, ...] | list[int]) -> int:
    if len(move_ids) != SUFFIX_LENGTH:
        raise ValueError("R4 word code requires exactly four moves")
    code = 0
    for move_id in move_ids:
        value = int(move_id)
        if not 0 <= value < WORD_BASE:
            raise ValueError("move id outside encoding alphabet")
        code = code * WORD_BASE + value
    return code


def decode_code(code: int) -> tuple[int, int, int, int]:
    value = int(code)
    if not 0 <= value < WORD_BASE ** SUFFIX_LENGTH:
        raise ValueError("invalid R4 word code")
    out = [0] * SUFFIX_LENGTH
    for index in range(SUFFIX_LENGTH - 1, -1, -1):
        out[index] = value % WORD_BASE
        value //= WORD_BASE
    return tuple(out)  # type: ignore[return-value]


def decode_names(code: int) -> tuple[str, str, str, str]:
    return tuple(MOVE_NAMES[index] for index in decode_code(code))  # type: ignore[return-value]


def direct_node_codes(registry: PairResidualFactorRegistry, node_id: int) -> np.ndarray:
    """Directly enumerate the exact accepted four-move paths of one MDD node."""
    start = int(node_id)
    if int(registry.remaining[start]) != SUFFIX_LENGTH:
        raise ValueError("direct R4 enumeration requires a remaining-four node")
    codes: list[int] = []

    def visit(node: int, depth: int, code: int) -> None:
        if depth == SUFFIX_LENGTH:
            if bool(registry.accepting[node]):
                codes.append(code)
            return
        mask = int(registry.outgoing_mask[node])
        for move_id in range(WORD_BASE):
            if mask & (1 << move_id):
                visit(
                    int(registry.child[node, move_id]),
                    depth + 1,
                    code * WORD_BASE + move_id,
                )

    visit(start, 0, 0)
    result = np.asarray(codes, dtype=np.uint32)
    if len(result) and (not np.all(result[1:] > result[:-1])):
        raise AssertionError("direct node support is not strict sorted unique")
    if len(result) != int(registry.path_counts[start]):
        raise AssertionError("direct node support/path-count mismatch")
    return result


def _signature(codes: np.ndarray) -> np.ndarray:
    result = np.zeros(SIGNATURE_WORDS, dtype=np.uint64)
    for code in np.asarray(codes, dtype=np.uint32):
        bit = int(code) % SIGNATURE_BITS
        result[bit // 64] |= np.uint64(1) << np.uint64(bit % 64)
    return result


@dataclass(frozen=True, slots=True)
class FactorR4Support:
    node_to_row: np.ndarray
    node_ids: np.ndarray
    offsets: np.ndarray
    codes: np.ndarray
    signatures: np.ndarray

    @property
    def resident_bytes(self) -> int:
        return sum(
            array.nbytes
            for array in (
                self.node_to_row,
                self.node_ids,
                self.offsets,
                self.codes,
                self.signatures,
            )
        )

    def support(self, node_id: int) -> np.ndarray:
        node = int(node_id)
        if not 0 <= node < len(self.node_to_row):
            raise R4ClosureAssetMismatch(f"node outside R4 asset: {node}")
        row = int(self.node_to_row[node])
        if row < 0:
            raise R4ClosureAssetMismatch(f"node missing from R4 asset: {node}")
        return self.codes[int(self.offsets[row]) : int(self.offsets[row + 1])]


@dataclass(slots=True)
class R4SparseSupportAsset:
    factors: tuple[FactorR4Support, ...]
    partition: tuple[tuple[int, int], ...]
    partition_hash: str
    registry_hashes: tuple[str, ...]
    metadata: dict[str, object]
    cold_load_wall: float = 0.0
    query_count: int = 0
    signature_rejects: int = 0
    full_support_checks: int = 0

    @property
    def resident_bytes(self) -> int:
        return sum(factor.resident_bytes for factor in self.factors)

    def necessary_signatures(self, node_keys: np.ndarray) -> np.ndarray:
        keys = np.asarray(node_keys)
        if keys.ndim != 2 or keys.shape[1] != len(self.factors):
            raise ValueError("R4 node keys must have ten columns")
        common = np.full((len(keys), SIGNATURE_WORDS), np.uint64(~np.uint64(0)), dtype=np.uint64)
        for column, factor in enumerate(self.factors):
            nodes = keys[:, column].astype(np.int64, copy=False)
            valid = (nodes >= 0) & (nodes < len(factor.node_to_row))
            rows = np.full(len(keys), -1, dtype=np.int64)
            rows[valid] = factor.node_to_row[nodes[valid]]
            present = rows >= 0
            common[~present] = 0
            if np.any(present):
                common[present] &= factor.signatures[rows[present]]
        return common

    def signature_nonempty_mask(self, node_keys: np.ndarray) -> np.ndarray:
        signatures = self.necessary_signatures(node_keys)
        return np.any(signatures != 0, axis=1)

    def close_codes(self, nodes: tuple[int, ...] | np.ndarray, *, first: bool = False) -> np.ndarray:
        key = np.asarray(nodes, dtype=np.int64)
        if key.shape != (len(self.factors),):
            raise ValueError("R4 closure key must contain ten nodes")
        self.query_count += 1
        signatures = self.necessary_signatures(key.reshape(1, -1))[0]
        if not np.any(signatures):
            self.signature_rejects += 1
            return np.empty(0, dtype=np.uint32)
        supports = [factor.support(int(node)) for factor, node in zip(self.factors, key)]
        order = sorted(range(len(supports)), key=lambda index: len(supports[index]))
        candidates = supports[order[0]]
        self.full_support_checks += 1
        for index in order[1:]:
            if not len(candidates):
                break
            other = supports[index]
            positions = np.searchsorted(other, candidates)
            in_bounds = positions < len(other)
            keep = np.zeros(len(candidates), dtype=np.bool_)
            keep[in_bounds] = other[positions[in_bounds]] == candidates[in_bounds]
            candidates = candidates[keep]
            # Do not truncate in first mode until every factor has been
            # checked: the smallest current candidate may fail a later factor
            # while a larger candidate remains in the exact intersection.
        return np.asarray(candidates[:1] if first else candidates, dtype=np.uint32)


def build_factor_support(registry: PairResidualFactorRegistry) -> FactorR4Support:
    node_ids = np.flatnonzero(registry.remaining == SUFFIX_LENGTH).astype(
        registry.child.dtype, copy=False
    )
    offsets = np.zeros(len(node_ids) + 1, dtype=np.uint32)
    rows: list[np.ndarray] = []
    signatures = np.zeros((len(node_ids), SIGNATURE_WORDS), dtype=np.uint64)
    cursor = 0
    for row, node_id in enumerate(node_ids):
        codes = direct_node_codes(registry, int(node_id))
        rows.append(codes)
        signatures[row] = _signature(codes)
        cursor += len(codes)
        offsets[row + 1] = cursor
    codes = np.concatenate(rows).astype(np.uint32, copy=False) if rows else np.empty(0, dtype=np.uint32)
    node_to_row = np.full(len(registry.remaining), -1, dtype=np.int32)
    node_to_row[node_ids.astype(np.int64)] = np.arange(len(node_ids), dtype=np.int32)
    for array in (node_to_row, node_ids, offsets, codes, signatures):
        array.flags.writeable = False
    return FactorR4Support(node_to_row, node_ids, offsets, codes, signatures)


def save_r4_support_asset(
    path: Path,
    registries: tuple[PairResidualFactorRegistry, ...] | list[PairResidualFactorRegistry],
    *,
    partition: tuple[tuple[int, int], ...],
    partition_hash: str,
    source_hashes: dict[str, str],
) -> dict[str, object]:
    started = perf_counter()
    ordered = tuple(registries)
    if len(ordered) != 10 or tuple(registry.pair for registry in ordered) != tuple(partition):
        raise ValueError("R4 support asset requires the frozen ten-registry order")
    factors = tuple(build_factor_support(registry) for registry in ordered)
    metadata = {
        "schema": SCHEMA,
        "PIECE_NAMES": list(PIECE_NAMES),
        "MOVE_NAMES": list(MOVE_NAMES),
        "canonical_rule_hash": CANONICAL_SEMANTICS_HASH,
        "Q_transition_hash": Q_TRANSITION_HASH,
        "selected_partition": [list(pair) for pair in partition],
        "selected_partition_hash": partition_hash,
        "maximum_horizon": 10,
        "remaining_scope": SUFFIX_LENGTH,
        "registry_hashes": [registry.registry_hash for registry in ordered],
        "source_code_hashes": dict(sorted(source_hashes.items())),
        "word_code_encoding_version": ENCODING,
        "closure_kernel_version": CLOSURE_VERSION,
        "signature": "EXACT_NECESSARY_CODE_MOD_256_V1",
    }
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")), dtype="U"
        )
    }
    for index, factor in enumerate(factors):
        arrays.update(
            {
                f"f{index}_node_to_row": factor.node_to_row,
                f"f{index}_node_ids": factor.node_ids,
                f"f{index}_offsets": factor.offsets,
                f"f{index}_codes": factor.codes,
                f"f{index}_signatures": factor.signatures,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {
        "build_and_serialization_wall_seconds": perf_counter() - started,
        "compressed_bytes": path.stat().st_size,
        "live_bytes": sum(factor.resident_bytes for factor in factors),
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "pair_node_counts": [len(factor.node_ids) for factor in factors],
        "support_code_counts": [len(factor.codes) for factor in factors],
        "metadata": metadata,
    }


def load_r4_support_asset(
    path: Path,
    *,
    expected_partition_hash: str,
    expected_registry_hashes: tuple[str, ...] | list[str] | None = None,
    expected_source_hashes: dict[str, str] | None = None,
) -> R4SparseSupportAsset:
    started = perf_counter()
    with np.load(path, allow_pickle=False) as loaded:
        metadata = json.loads(str(loaded["metadata_json"].item()))
        checks = {
            "schema": metadata.get("schema") == SCHEMA,
            "PIECE_NAMES": metadata.get("PIECE_NAMES") == list(PIECE_NAMES),
            "MOVE_NAMES": metadata.get("MOVE_NAMES") == list(MOVE_NAMES),
            "canonical_rule_hash": metadata.get("canonical_rule_hash") == CANONICAL_SEMANTICS_HASH,
            "Q_transition_hash": metadata.get("Q_transition_hash") == Q_TRANSITION_HASH,
            "partition_hash": metadata.get("selected_partition_hash") == expected_partition_hash,
            "horizon": metadata.get("maximum_horizon") == 10,
            "remaining": metadata.get("remaining_scope") == SUFFIX_LENGTH,
            "encoding": metadata.get("word_code_encoding_version") == ENCODING,
            "closure": metadata.get("closure_kernel_version") == CLOSURE_VERSION,
            "registry_hashes": expected_registry_hashes is None
            or metadata.get("registry_hashes") == list(expected_registry_hashes),
            "source_hashes": expected_source_hashes is None
            or metadata.get("source_code_hashes") == dict(sorted(expected_source_hashes.items())),
        }
        failures = [name for name, passed in checks.items() if not passed]
        if failures:
            raise R4ClosureAssetMismatch("R4 support metadata mismatch: " + ",".join(failures))
        factors = []
        for index in range(10):
            arrays = [
                loaded[f"f{index}_node_to_row"].copy(),
                loaded[f"f{index}_node_ids"].copy(),
                loaded[f"f{index}_offsets"].copy(),
                loaded[f"f{index}_codes"].copy(),
                loaded[f"f{index}_signatures"].copy(),
            ]
            node_to_row, node_ids, offsets, codes, signatures = arrays
            if node_to_row.dtype != np.int32 or offsets.dtype != np.uint32 or codes.dtype != np.uint32:
                raise R4ClosureAssetMismatch(f"R4 support dtype mismatch: factor {index}")
            if signatures.dtype != np.uint64 or signatures.shape != (len(node_ids), SIGNATURE_WORDS):
                raise R4ClosureAssetMismatch(f"R4 signature shape mismatch: factor {index}")
            if len(offsets) != len(node_ids) + 1 or int(offsets[-1]) != len(codes):
                raise R4ClosureAssetMismatch(f"R4 offset mismatch: factor {index}")
            for array in arrays:
                array.flags.writeable = False
            factors.append(FactorR4Support(node_to_row, node_ids, offsets, codes, signatures))
    return R4SparseSupportAsset(
        factors=tuple(factors),
        partition=tuple(tuple(map(int, pair)) for pair in metadata["selected_partition"]),
        partition_hash=str(metadata["selected_partition_hash"]),
        registry_hashes=tuple(str(value) for value in metadata["registry_hashes"]),
        metadata=metadata,
        cold_load_wall=perf_counter() - started,
    )


def recursive_product_codes(
    registries: tuple[PairResidualFactorRegistry, ...],
    nodes: tuple[int, ...] | np.ndarray,
    *,
    first: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    """C1: exact recursive ten-factor product restricted to four remaining moves."""
    root = tuple(int(value) for value in nodes)
    if len(registries) != 10 or len(root) != 10:
        raise ValueError("C1 requires ten pair nodes")
    if any(int(registry.remaining[node]) != SUFFIX_LENGTH for registry, node in zip(registries, root)):
        raise ValueError("C1 root is outside remaining-four scope")
    memo: dict[tuple[int, tuple[int, ...]], tuple[int, ...]] = {}
    calls = 0
    hits = 0

    def visit(depth: int, key: tuple[int, ...]) -> tuple[int, ...]:
        nonlocal calls, hits
        calls += 1
        memo_key = (depth, key)
        cached = memo.get(memo_key)
        if cached is not None:
            hits += 1
            return cached
        if depth == SUFFIX_LENGTH:
            result = (0,) if all(
                bool(registry.accepting[node]) for registry, node in zip(registries, key)
            ) else ()
            memo[memo_key] = result
            return result
        common = (1 << WORD_BASE) - 1
        for registry, node in zip(registries, key):
            common &= int(registry.outgoing_mask[node])
        scale = WORD_BASE ** (SUFFIX_LENGTH - depth - 1)
        values: list[int] = []
        for move_id in range(WORD_BASE):
            if not common & (1 << move_id):
                continue
            child = tuple(
                int(registry.child[node, move_id])
                for registry, node in zip(registries, key)
            )
            for suffix in visit(depth + 1, child):
                values.append(move_id * scale + suffix)
                if first:
                    result = tuple(values[:1])
                    memo[memo_key] = result
                    return result
        result = tuple(values)
        memo[memo_key] = result
        return result

    values = np.asarray(visit(0, root), dtype=np.uint32)
    return values, {"calls": calls, "memo_hits": hits, "memo_entries": len(memo)}


def explicit_four_layer_codes(
    registries: tuple[PairResidualFactorRegistry, ...],
    nodes: tuple[int, ...] | np.ndarray,
) -> np.ndarray:
    """Independent iterative four-layer product used by the verifier."""
    frontier: list[tuple[tuple[int, ...], int]] = [(tuple(int(value) for value in nodes), 0)]
    for _depth in range(SUFFIX_LENGTH):
        nxt: list[tuple[tuple[int, ...], int]] = []
        for key, code in frontier:
            common = (1 << WORD_BASE) - 1
            for registry, node in zip(registries, key):
                common &= int(registry.outgoing_mask[node])
            for move_id in range(WORD_BASE):
                if common & (1 << move_id):
                    nxt.append(
                        (
                            tuple(
                                int(registry.child[node, move_id])
                                for registry, node in zip(registries, key)
                            ),
                            code * WORD_BASE + move_id,
                        )
                    )
        frontier = nxt
    return np.asarray(
        [
            code
            for key, code in frontier
            if all(bool(registry.accepting[node]) for registry, node in zip(registries, key))
        ],
        dtype=np.uint32,
    )


__all__ = [
    "CLOSURE_VERSION",
    "ENCODING",
    "R4ClosureAssetMismatch",
    "R4SparseSupportAsset",
    "SCHEMA",
    "build_factor_support",
    "decode_code",
    "decode_names",
    "direct_node_codes",
    "encode_moves",
    "explicit_four_layer_codes",
    "load_r4_support_asset",
    "recursive_product_codes",
    "save_r4_support_asset",
]
