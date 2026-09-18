"""Exact universal two-piece SAME-word residual registries for v37.358."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from time import perf_counter

import numpy as np

from cubelab.ato.column_grounding import Q_NEXT
from cubelab.ato.regular_support import FACE_ORDER, MOVE_NAMES, canonical_allow
from cubelab.pdcc.model import PIECE_NAMES

from .residual_factor_view import CONTEXTS, CONTEXT_INDEX
from .symbolic_mdd_v2 import NodeArena, ResourceCapUnknown, SymbolicMDD


def _canonical_hash() -> str:
    payload = [
        (previous, move, canonical_allow(previous, move))
        for previous in CONTEXTS
        for move in MOVE_NAMES
    ]
    return sha256(repr(payload).encode()).hexdigest()


CANONICAL_SEMANTICS_HASH = _canonical_hash()
Q_TRANSITION_HASH = sha256(np.ascontiguousarray(Q_NEXT).tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class PairResidualFactorView:
    pair: tuple[int, int]
    horizon: int
    root_node: int
    remaining: np.ndarray
    outgoing_mask: np.ndarray
    child: np.ndarray
    accepting: np.ndarray
    path_counts: np.ndarray
    scope_hash: str
    structural_hash: str
    node_id_dtype: str
    full_words_materialized: int = 0
    python_word_set_materialized: int = 0

    def validate(self) -> dict[str, object]:
        failures: list[str] = []
        for node_id in range(2, len(self.remaining)):
            mask = 0
            for move_id in range(len(MOVE_NAMES)):
                target = int(self.child[node_id, move_id])
                if target:
                    mask |= 1 << move_id
                    if int(self.remaining[target]) != int(self.remaining[node_id]) - 1:
                        failures.append(f"remaining:{node_id}:{move_id}:{target}")
            if mask != int(self.outgoing_mask[node_id]):
                failures.append(f"mask:{node_id}")
        if bool(self.accepting[0]) or not bool(self.accepting[1]):
            failures.append("terminal-acceptance")
        return {
            "pair": list(self.pair),
            "nodes": len(self.remaining),
            "root": self.root_node,
            "root_paths": int(self.path_counts[self.root_node]),
            "failures": failures,
            "passed": not failures,
            "full_words_materialized": self.full_words_materialized,
        }


@dataclass(slots=True)
class PairResidualFactorRegistry:
    pair: tuple[int, int]
    maximum_horizon: int
    arena: NodeArena | None
    node_for_pair_q_remaining_context: np.ndarray
    remaining: np.ndarray
    outgoing_mask: np.ndarray
    child: np.ndarray
    accepting: np.ndarray
    path_counts: np.ndarray
    registry_hash: str
    canonical_semantics_hash: str
    q_transition_hash: str
    build_wall_seconds: float
    structural_hash_memo: dict[int, str] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        pair: tuple[int, int],
        *,
        maximum_horizon: int = 8,
        node_cap: int = 5_000_000,
    ) -> "PairResidualFactorRegistry":
        started = perf_counter()
        left, right = (int(pair[0]), int(pair[1]))
        if left == right or not (0 <= left < 20 and 0 <= right < 20):
            raise ValueError("invalid concrete piece pair")
        maximum_horizon = int(maximum_horizon)
        arena = NodeArena(node_cap=node_cap)
        memo: dict[tuple[int, int, int, int], int] = {}

        def build_node(q_left: int, q_right: int, remaining: int, context_id: int) -> int:
            key = (int(q_left), int(q_right), int(remaining), int(context_id))
            cached = memo.get(key)
            if cached is not None:
                return cached
            if remaining == 0:
                result = 1 if q_left == 0 and q_right == 0 else 0
                memo[key] = result
                return result
            previous = CONTEXTS[context_id]
            edges = []
            for move_id, move in enumerate(MOVE_NAMES):
                if not canonical_allow(previous, move):
                    continue
                child = build_node(
                    int(Q_NEXT[left, q_left, move_id]),
                    int(Q_NEXT[right, q_right, move_id]),
                    remaining - 1,
                    CONTEXT_INDEX[move[0]],
                )
                if child:
                    edges.append((move_id, child))
            result = arena.intern(remaining, edges)
            memo[key] = result
            return result

        roots = np.zeros(
            (24, 24, maximum_horizon + 1, len(CONTEXTS)), dtype=np.uint32
        )
        for remaining in range(maximum_horizon + 1):
            for q_left in range(24):
                for q_right in range(24):
                    for context_id in range(len(CONTEXTS)):
                        roots[q_left, q_right, remaining, context_id] = build_node(
                            q_left, q_right, remaining, context_id
                        )

        node_count = len(arena.nodes)
        remaining_table = np.zeros(node_count, dtype=np.uint8)
        outgoing = np.zeros(node_count, dtype=np.uint32)
        child32 = np.zeros((node_count, len(MOVE_NAMES)), dtype=np.uint32)
        accepting = np.zeros(node_count, dtype=np.bool_)
        accepting[1] = True
        for node_id, node in enumerate(arena.nodes[2:], start=2):
            remaining_table[node_id] = node.remaining
            mask = 0
            for move_id, child in node.edges:
                mask |= 1 << move_id
                child32[node_id, move_id] = child
            outgoing[node_id] = mask
        counts = np.zeros(node_count, dtype=np.int64)
        counts[1] = 1
        for remaining in range(1, maximum_horizon + 1):
            for node_id in np.flatnonzero(remaining_table == remaining):
                counts[node_id] = sum(
                    int(counts[child]) for child in child32[node_id] if child
                )
        dtype = np.uint16 if node_count <= np.iinfo(np.uint16).max else np.uint32
        child = child32.astype(dtype, copy=False)
        roots = roots.astype(dtype, copy=False)
        payload = b"|".join((
            f"{left},{right},{maximum_horizon}".encode(),
            CANONICAL_SEMANTICS_HASH.encode(), Q_TRANSITION_HASH.encode(),
            roots.tobytes(), remaining_table.tobytes(), outgoing.tobytes(),
            child.tobytes(), accepting.tobytes(), counts.tobytes(),
        ))
        for array in (roots, remaining_table, outgoing, child, accepting, counts):
            array.flags.writeable = False
        return cls(
            pair=(left, right), maximum_horizon=maximum_horizon, arena=arena,
            node_for_pair_q_remaining_context=roots, remaining=remaining_table,
            outgoing_mask=outgoing, child=child, accepting=accepting,
            path_counts=counts, registry_hash=sha256(payload).hexdigest(),
            canonical_semantics_hash=CANONICAL_SEMANTICS_HASH,
            q_transition_hash=Q_TRANSITION_HASH,
            build_wall_seconds=perf_counter() - started,
        )

    @property
    def node_id_dtype(self) -> np.dtype:
        return self.child.dtype

    @property
    def pair_names(self) -> tuple[str, str]:
        return PIECE_NAMES[self.pair[0]], PIECE_NAMES[self.pair[1]]

    def root(
        self, q_left: int, q_right: int, horizon: int,
        previous_face: str | None = None,
    ) -> int:
        return int(self.node_for_pair_q_remaining_context[
            int(q_left), int(q_right), int(horizon), CONTEXT_INDEX[previous_face]
        ])

    def view(
        self, q_left: int, q_right: int, horizon: int, *,
        previous_face: str | None = None, scope_hash: str = "",
    ) -> PairResidualFactorView:
        root = self.root(q_left, q_right, horizon, previous_face)
        return PairResidualFactorView(
            pair=self.pair, horizon=int(horizon), root_node=root,
            remaining=self.remaining, outgoing_mask=self.outgoing_mask,
            child=self.child, accepting=self.accepting, path_counts=self.path_counts,
            scope_hash=str(scope_hash), structural_hash=self.root_language_hash(root),
            node_id_dtype=self.child.dtype.name,
        )

    def root_language_hash(self, node_id: int) -> str:
        false_hash = sha256(b"FALSE").hexdigest()
        true_hash = sha256(b"TRUE").hexdigest()
        if not self.structural_hash_memo:
            self.structural_hash_memo.update({0: false_hash, 1: true_hash})

        def visit(node: int) -> str:
            cached = self.structural_hash_memo.get(int(node))
            if cached is not None:
                return cached
            payload = f"{int(self.remaining[node])}|" + ";".join(
                f"{move_id}:{visit(int(target))}"
                for move_id, target in enumerate(self.child[node]) if int(target)
            )
            value = sha256(payload.encode()).hexdigest()
            self.structural_hash_memo[int(node)] = value
            return value

        return visit(int(node_id))

    def drop_arena(self) -> None:
        """Release Python node objects after dense exact tables are frozen."""
        self.arena = None

    def cache_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        return {
            f"{prefix}_roots": self.node_for_pair_q_remaining_context,
            f"{prefix}_remaining": self.remaining,
            f"{prefix}_outgoing_mask": self.outgoing_mask,
            f"{prefix}_child": self.child,
            f"{prefix}_accepting": self.accepting,
            f"{prefix}_path_counts": self.path_counts,
        }

    def uncompressed_bytes(self) -> int:
        return sum(array.nbytes for array in (
            self.node_for_pair_q_remaining_context, self.remaining,
            self.outgoing_mask, self.child, self.accepting, self.path_counts,
        ))


def build_pair_registries(
    partition: tuple[tuple[int, int], ...], *, maximum_horizon: int = 8,
    node_cap: int = 5_000_000,
) -> list[PairResidualFactorRegistry]:
    return [
        PairResidualFactorRegistry.build(
            pair, maximum_horizon=maximum_horizon, node_cap=node_cap
        )
        for pair in partition
    ]


__all__ = [
    "CANONICAL_SEMANTICS_HASH", "Q_TRANSITION_HASH", "PairResidualFactorRegistry",
    "PairResidualFactorView", "ResourceCapUnknown", "build_pair_registries",
]
