"""Read-only exact residual-language factors for v37.357.

Each piece registry minimizes every ``(Q, remaining, canonical context)``
future language in one canonical node arena.  A factor view is only a root
handle plus dense read-only transition tables; it never retains full words.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from cubelab.ato.column_grounding import Q_NEXT
from cubelab.ato.regular_support import FACE_ORDER, MOVE_NAMES, canonical_allow

from .symbolic_mdd_v2 import NodeArena, ResourceCapUnknown, SymbolicMDD


CONTEXTS: tuple[str | None, ...] = (None, *FACE_ORDER)
CONTEXT_INDEX = {value: index for index, value in enumerate(CONTEXTS)}


@dataclass(frozen=True, slots=True)
class ResidualFactorView:
    piece: int
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
            "piece": self.piece,
            "nodes": len(self.remaining),
            "root": self.root_node,
            "root_paths": int(self.path_counts[self.root_node]),
            "failures": failures,
            "passed": not failures,
            "full_words_materialized": self.full_words_materialized,
            "python_word_set_materialized": self.python_word_set_materialized,
        }


@dataclass(slots=True)
class ResidualFactorRegistry:
    piece: int
    maximum_horizon: int
    arena: NodeArena
    node_for_q_remaining_context: np.ndarray
    remaining: np.ndarray
    outgoing_mask: np.ndarray
    child: np.ndarray
    accepting: np.ndarray
    path_counts: np.ndarray
    registry_hash: str

    @classmethod
    def build(
        cls,
        piece: int,
        *,
        maximum_horizon: int = 8,
        node_cap: int = 5_000_000,
    ) -> "ResidualFactorRegistry":
        piece = int(piece)
        maximum_horizon = int(maximum_horizon)
        arena = NodeArena(node_cap=node_cap)
        memo: dict[tuple[int, int, int], int] = {}

        def build_node(q: int, remaining: int, context_id: int) -> int:
            key = (int(q), int(remaining), int(context_id))
            cached = memo.get(key)
            if cached is not None:
                return cached
            if remaining == 0:
                result = 1 if q == 0 else 0
                memo[key] = result
                return result
            previous = CONTEXTS[context_id]
            edges = []
            for move_id, move in enumerate(MOVE_NAMES):
                if not canonical_allow(previous, move):
                    continue
                next_q = int(Q_NEXT[piece, q, move_id])
                child = build_node(
                    next_q,
                    remaining - 1,
                    CONTEXT_INDEX[move[0]],
                )
                if child:
                    edges.append((move_id, child))
            result = arena.intern(remaining, edges)
            memo[key] = result
            return result

        table = np.zeros((24, maximum_horizon + 1, len(CONTEXTS)), dtype=np.uint32)
        for remaining_depth in range(maximum_horizon + 1):
            for q in range(24):
                for context_id in range(len(CONTEXTS)):
                    table[q, remaining_depth, context_id] = build_node(
                        q, remaining_depth, context_id
                    )

        node_count = len(arena.nodes)
        remaining_table = np.zeros(node_count, dtype=np.uint8)
        outgoing = np.zeros(node_count, dtype=np.uint32)
        child_table = np.zeros((node_count, len(MOVE_NAMES)), dtype=np.uint32)
        accepting = np.zeros(node_count, dtype=np.bool_)
        accepting[1] = True
        for node_id, node in enumerate(arena.nodes[2:], start=2):
            remaining_table[node_id] = node.remaining
            mask = 0
            for move_id, child in node.edges:
                mask |= 1 << move_id
                child_table[node_id, move_id] = child
            outgoing[node_id] = mask
        counts = np.zeros(node_count, dtype=np.int64)
        counts[1] = 1
        for remaining_depth in range(1, maximum_horizon + 1):
            for node_id in np.flatnonzero(remaining_table == remaining_depth):
                counts[node_id] = sum(
                    int(counts[child]) for child in child_table[node_id] if child
                )
        payload = b"|".join(
            (
                str(piece).encode(),
                table.tobytes(),
                remaining_table.tobytes(),
                outgoing.tobytes(),
                child_table.tobytes(),
                counts.tobytes(),
            )
        )
        return cls(
            piece=piece,
            maximum_horizon=maximum_horizon,
            arena=arena,
            node_for_q_remaining_context=table,
            remaining=remaining_table,
            outgoing_mask=outgoing,
            child=child_table,
            accepting=accepting,
            path_counts=counts,
            registry_hash=sha256(payload).hexdigest(),
        )

    @property
    def node_id_dtype(self) -> np.dtype:
        return np.dtype(np.uint16 if len(self.remaining) <= np.iinfo(np.uint16).max else np.uint32)

    def root(self, q: int, horizon: int, previous_face: str | None = None) -> int:
        return int(
            self.node_for_q_remaining_context[
                int(q), int(horizon), CONTEXT_INDEX[previous_face]
            ]
        )

    def view(
        self,
        q: int,
        horizon: int,
        *,
        previous_face: str | None = None,
        scope_hash: str = "",
    ) -> ResidualFactorView:
        root_node = self.root(q, horizon, previous_face)
        mdd = SymbolicMDD(
            arena=self.arena,
            root=root_node,
            horizon=int(horizon),
            previous_face=previous_face,
            scope_hash=scope_hash,
            construction=f"RESIDUAL_FACTOR_VIEW_P{self.piece}",
        )
        dtype = self.node_id_dtype
        child = self.child.astype(dtype, copy=False)
        for array in (
            self.remaining,
            self.outgoing_mask,
            child,
            self.accepting,
            self.path_counts,
        ):
            array.flags.writeable = False
        return ResidualFactorView(
            piece=self.piece,
            horizon=int(horizon),
            root_node=root_node,
            remaining=self.remaining,
            outgoing_mask=self.outgoing_mask,
            child=child,
            accepting=self.accepting,
            path_counts=self.path_counts,
            scope_hash=str(scope_hash),
            structural_hash=mdd.structural_hash(),
            node_id_dtype=dtype.name,
        )

    def cache_arrays(self) -> dict[str, np.ndarray]:
        return {
            f"p{self.piece:02d}_node_for_q_remaining_context": self.node_for_q_remaining_context,
            f"p{self.piece:02d}_remaining": self.remaining,
            f"p{self.piece:02d}_outgoing_mask": self.outgoing_mask,
            f"p{self.piece:02d}_child": self.child,
            f"p{self.piece:02d}_accepting": self.accepting,
            f"p{self.piece:02d}_path_counts": self.path_counts,
        }


def build_registries(
    *, maximum_horizon: int = 8, node_cap: int = 5_000_000
) -> list[ResidualFactorRegistry]:
    return [
        ResidualFactorRegistry.build(
            piece, maximum_horizon=maximum_horizon, node_cap=node_cap
        )
        for piece in range(20)
    ]


__all__ = [
    "CONTEXTS",
    "CONTEXT_INDEX",
    "ResidualFactorRegistry",
    "ResidualFactorView",
    "ResourceCapUnknown",
    "build_registries",
]
