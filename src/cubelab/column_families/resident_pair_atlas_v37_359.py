"""Strict immutable selected-pair H10 resident asset for v37.359."""

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
    CANONICAL_SEMANTICS_HASH, Q_TRANSITION_HASH, PairResidualFactorRegistry,
)


SCHEMA = "cubelab.resident-selected-pair-atlas.v37.359.1"


class ResidentPairAssetMismatch(ValueError):
    pass


def _key(pair: tuple[int, int]) -> str:
    return f"p{pair[0]:02d}_{pair[1]:02d}"


@dataclass(slots=True)
class ResidentPairAtlas:
    registries: tuple[PairResidualFactorRegistry, ...]
    partition: tuple[tuple[int, int], ...]
    partition_hash: str
    metadata: dict[str, object]
    cold_load_wall: float = 0.0
    query_count: int = 0

    @property
    def resident_bytes(self) -> int:
        return sum(registry.uncompressed_bytes() for registry in self.registries)

    def factors(
        self, start_q: tuple[int, ...], horizon: int, scope_hash: str = "",
    ):
        self.query_count += 1
        return tuple(
            registry.view(start_q[left], start_q[right], horizon, scope_hash=scope_hash)
            for registry, (left, right) in zip(self.registries, self.partition)
        )


def save_resident_pair_atlas(
    path: Path,
    registries: list[PairResidualFactorRegistry] | tuple[PairResidualFactorRegistry, ...],
    *, partition: tuple[tuple[int, int], ...], partition_hash: str,
    source_hashes: dict[str, str],
) -> dict[str, object]:
    started = perf_counter()
    ordered = tuple(registries)
    if tuple(registry.pair for registry in ordered) != tuple(partition):
        raise ValueError("registry order must equal sealed partition order")
    if len(ordered) != 10 or any(registry.maximum_horizon != 10 for registry in ordered):
        raise ValueError("resident selected asset requires ten H10 registries")
    metadata = {
        "schema": SCHEMA, "PIECE_NAMES": list(PIECE_NAMES), "MOVE_NAMES": list(MOVE_NAMES),
        "canonical_rule_hash": CANONICAL_SEMANTICS_HASH, "Q_transition_hash": Q_TRANSITION_HASH,
        "selected_partition": [list(pair) for pair in partition],
        "selected_partition_hash": partition_hash, "maximum_horizon": 10,
        "registry_hashes": [registry.registry_hash for registry in ordered],
        "node_id_dtypes": [registry.node_id_dtype.name for registry in ordered],
        "source_code_hashes": dict(sorted(source_hashes.items())),
        "word_code_encoding_version": "BASE18_BIG_ENDIAN_R4_V1",
        "closure_kernel_version": "R4_C1_C2_V37_359_1",
    }
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")), dtype="U")
    }
    for registry in ordered:
        arrays.update(registry.cache_arrays(_key(registry.pair)))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {
        "serialization_wall_seconds": perf_counter() - started,
        "compressed_bytes": path.stat().st_size,
        "uncompressed_bytes": sum(array.nbytes for array in arrays.values()),
        "sha256": sha256(path.read_bytes()).hexdigest(), "metadata": metadata,
    }


def load_resident_pair_atlas(
    path: Path, *, expected_partition_hash: str,
    expected_source_hashes: dict[str, str] | None = None,
) -> ResidentPairAtlas:
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
            "encoding": metadata.get("word_code_encoding_version") == "BASE18_BIG_ENDIAN_R4_V1",
            "closure": metadata.get("closure_kernel_version") == "R4_C1_C2_V37_359_1",
            "source_hashes": expected_source_hashes is None or metadata.get("source_code_hashes") == dict(sorted(expected_source_hashes.items())),
        }
        failures = [name for name, value in checks.items() if not value]
        if failures:
            raise ResidentPairAssetMismatch("resident pair metadata mismatch: " + ",".join(failures))
        partition = tuple(tuple(map(int, pair)) for pair in metadata["selected_partition"])
        registries = []
        for index, pair in enumerate(partition):
            prefix = _key(pair)
            arrays = [
                loaded[f"{prefix}_roots"].copy(), loaded[f"{prefix}_remaining"].copy(),
                loaded[f"{prefix}_outgoing_mask"].copy(), loaded[f"{prefix}_child"].copy(),
                loaded[f"{prefix}_accepting"].copy(), loaded[f"{prefix}_path_counts"].copy(),
            ]
            roots, remaining, outgoing, child, accepting, counts = arrays
            if child.dtype.name != metadata["node_id_dtypes"][index]:
                raise ResidentPairAssetMismatch(f"node dtype mismatch: {pair}")
            for array in arrays:
                array.flags.writeable = False
            registries.append(PairResidualFactorRegistry(
                pair=pair, maximum_horizon=10, arena=None,
                node_for_pair_q_remaining_context=roots, remaining=remaining,
                outgoing_mask=outgoing, child=child, accepting=accepting,
                path_counts=counts, registry_hash=metadata["registry_hashes"][index],
                canonical_semantics_hash=CANONICAL_SEMANTICS_HASH,
                q_transition_hash=Q_TRANSITION_HASH, build_wall_seconds=0.0,
            ))
    return ResidentPairAtlas(
        registries=tuple(registries), partition=partition,
        partition_hash=str(metadata["selected_partition_hash"]), metadata=metadata,
        cold_load_wall=perf_counter() - started,
    )


__all__ = [
    "ResidentPairAssetMismatch", "ResidentPairAtlas", "SCHEMA",
    "load_resident_pair_atlas", "save_resident_pair_atlas",
]
