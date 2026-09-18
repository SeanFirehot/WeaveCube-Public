"""True node-level symbolic move-labelled MDD kernel for v37.356.

The kernel stores a reduced layered DAG only.  It never retains an accepted
word portfolio and all Boolean operations recurse over node pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Callable, Hashable, Iterable

import numpy as np

from cubelab.ato.regular_support import MOVE_INDEX, MOVE_NAMES, canonical_allow


State = Hashable
Transition = Callable[[State, int], State]
Accept = Callable[[State], bool]


class ResourceCapUnknown(RuntimeError):
    status = "UNKNOWN_RESOURCE_CAP"


@dataclass(frozen=True, slots=True)
class SymbolicNode:
    remaining: int
    edges: tuple[tuple[int, int], ...]
    accepting: bool = False


class NodeArena:
    """Canonical shared node arena.

    Node 0 is FALSE and node 1 is TRUE.  All nonterminal nodes are reduced by
    ``(remaining, labelled outgoing edges)`` in one arena.
    """

    def __init__(self, *, node_cap: int = 5_000_000) -> None:
        self.node_cap = int(node_cap)
        self.nodes: list[SymbolicNode] = [
            SymbolicNode(0, (), False),
            SymbolicNode(0, (), True),
        ]
        self.unique: dict[tuple[int, tuple[tuple[int, int], ...]], int] = {}

    def intern(self, remaining: int, edges: Iterable[tuple[int, int]]) -> int:
        remaining = int(remaining)
        normalized = tuple(
            (int(move_id), int(child))
            for move_id, child in sorted(edges)
            if int(child) != 0
        )
        if remaining <= 0:
            raise ValueError("Nonterminal node remaining must be positive")
        if not normalized:
            return 0
        key = (remaining, normalized)
        existing = self.unique.get(key)
        if existing is not None:
            return existing
        if len(self.nodes) >= self.node_cap:
            raise ResourceCapUnknown(
                f"symbolic node cap {self.node_cap} reached"
            )
        node_id = len(self.nodes)
        self.nodes.append(SymbolicNode(remaining, normalized, False))
        self.unique[key] = node_id
        return node_id


@dataclass(slots=True)
class SymbolicMDD:
    arena: NodeArena
    root: int
    horizon: int
    previous_face: str | None
    scope_hash: str
    construction: str
    full_words_materialized: int = 0
    python_word_set_materialized: int = 0

    @classmethod
    def from_automaton(
        cls,
        state: State,
        horizon: int,
        transition: Transition,
        accept: Accept,
        *,
        previous_face: str | None = None,
        scope_hash: str = "",
        node_cap: int = 5_000_000,
        construction: str = "DETERMINISTIC_AUTOMATON",
    ) -> "SymbolicMDD":
        arena = NodeArena(node_cap=node_cap)
        memo: dict[tuple[State, int, str | None], int] = {}

        def build(current: State, remaining: int, context: str | None) -> int:
            key = (current, remaining, context)
            cached = memo.get(key)
            if cached is not None:
                return cached
            if remaining == 0:
                result = 1 if accept(current) else 0
                memo[key] = result
                return result
            children = []
            for move_id, move in enumerate(MOVE_NAMES):
                if not canonical_allow(context, move):
                    continue
                child = build(
                    transition(current, move_id), remaining - 1, move[0]
                )
                if child:
                    children.append((move_id, child))
            result = arena.intern(remaining, children)
            memo[key] = result
            return result

        root = build(state, int(horizon), previous_face)
        return cls(
            arena=arena,
            root=root,
            horizon=int(horizon),
            previous_face=previous_face,
            scope_hash=str(scope_hash),
            construction=str(construction),
        )

    @classmethod
    def from_packed(
        cls,
        moves: np.ndarray,
        accepted: np.ndarray,
        *,
        previous_face: str | None = None,
        scope_hash: str = "",
        node_cap: int = 5_000_000,
    ) -> "SymbolicMDD":
        """Ground-truth constructor over packed rows, without Python words."""

        rows = np.asarray(moves, dtype=np.uint8)
        keep = np.asarray(accepted, dtype=np.bool_)
        if rows.ndim != 2 or keep.shape != (len(rows),):
            raise ValueError("packed moves/accepted shape mismatch")
        selected = rows[keep]
        horizon = rows.shape[1]
        arena = NodeArena(node_cap=node_cap)
        if horizon == 0:
            root = 1 if len(selected) else 0
        elif not len(selected):
            root = 0
        else:
            # Canonical arrays are lexicographic.  A stable lexsort keeps this
            # constructor correct for any packed input order.
            order = np.lexsort(tuple(selected[:, col] for col in range(horizon - 1, -1, -1)))
            selected = selected[order]
            child_ids = np.ones(len(selected), dtype=np.int64)
            for column in range(horizon - 1, -1, -1):
                prefix_width = column
                if prefix_width == 0:
                    starts = np.asarray([0], dtype=np.int64)
                    ends = np.asarray([len(selected)], dtype=np.int64)
                else:
                    changed = np.any(selected[1:, :prefix_width] != selected[:-1, :prefix_width], axis=1)
                    starts = np.concatenate(([0], np.flatnonzero(changed) + 1))
                    ends = np.concatenate((starts[1:], [len(selected)]))
                next_ids = np.empty(len(selected), dtype=np.int64)
                for start, end in zip(starts, ends):
                    edges: dict[int, int] = {}
                    for row_id in range(int(start), int(end)):
                        move_id = int(selected[row_id, column])
                        child = int(child_ids[row_id])
                        prior = edges.get(move_id)
                        if prior is not None and prior != child:
                            raise AssertionError("packed rows are not deterministic by prefix")
                        edges[move_id] = child
                    node = arena.intern(horizon - column, edges.items())
                    next_ids[int(start) : int(end)] = node
                child_ids = next_ids
            root = int(child_ids[0])
        return cls(
            arena=arena,
            root=root,
            horizon=horizon,
            previous_face=previous_face,
            scope_hash=str(scope_hash),
            construction="PACKED_GROUND_TRUTH",
            full_words_materialized=0,
            python_word_set_materialized=0,
        )

    def _compatible(self, other: "SymbolicMDD") -> None:
        if self.horizon != other.horizon or self.previous_face != other.previous_face:
            raise ValueError("symbolic MDD horizon/context mismatch")
        if self.scope_hash and other.scope_hash and self.scope_hash != other.scope_hash:
            raise ValueError("symbolic MDD scope mismatch")

    def apply(
        self,
        other: "SymbolicMDD",
        operation: str,
        *,
        node_cap: int = 5_000_000,
    ) -> "SymbolicMDD":
        self._compatible(other)
        op = operation.upper()
        if op not in {"UNION", "INTERSECTION", "DIFFERENCE"}:
            raise ValueError(f"unsupported symbolic operation: {operation}")
        arena = NodeArena(node_cap=node_cap)
        memo: dict[tuple[int, int, int], int] = {}

        def terminal(left: bool, right: bool) -> bool:
            if op == "UNION":
                return left or right
            if op == "INTERSECTION":
                return left and right
            return left and not right

        def edges_for(mdd: "SymbolicMDD", node_id: int) -> dict[int, int]:
            if node_id in (0, 1):
                return {}
            return dict(mdd.arena.nodes[node_id].edges)

        def visit(left: int, right: int, remaining: int) -> int:
            key = (left, right, remaining)
            cached = memo.get(key)
            if cached is not None:
                return cached
            if remaining == 0:
                result = 1 if terminal(left == 1, right == 1) else 0
                memo[key] = result
                return result
            left_edges = edges_for(self, left)
            right_edges = edges_for(other, right)
            if op == "UNION":
                move_ids = left_edges.keys() | right_edges.keys()
            elif op == "INTERSECTION":
                move_ids = left_edges.keys() & right_edges.keys()
            else:
                move_ids = left_edges.keys()
            children = []
            for move_id in sorted(move_ids):
                child = visit(
                    left_edges.get(move_id, 0),
                    right_edges.get(move_id, 0),
                    remaining - 1,
                )
                if child:
                    children.append((move_id, child))
            result = arena.intern(remaining, children)
            memo[key] = result
            return result

        root = visit(self.root, other.root, self.horizon)
        return SymbolicMDD(
            arena=arena,
            root=root,
            horizon=self.horizon,
            previous_face=self.previous_face,
            scope_hash=self.scope_hash or other.scope_hash,
            construction=f"NODE_APPLY_{op}",
        )

    def union(self, other: "SymbolicMDD", *, node_cap: int = 5_000_000) -> "SymbolicMDD":
        return self.apply(other, "UNION", node_cap=node_cap)

    def intersect(self, other: "SymbolicMDD", *, node_cap: int = 5_000_000) -> "SymbolicMDD":
        return self.apply(other, "INTERSECTION", node_cap=node_cap)

    def difference(self, other: "SymbolicMDD", *, node_cap: int = 5_000_000) -> "SymbolicMDD":
        return self.apply(other, "DIFFERENCE", node_cap=node_cap)

    def complement(self, universe: "SymbolicMDD", *, node_cap: int = 5_000_000) -> "SymbolicMDD":
        return universe.difference(self, node_cap=node_cap)

    def reachable_nodes(self) -> set[int]:
        if self.root == 0:
            return set()
        seen: set[int] = set()
        pending = [self.root]
        while pending:
            node_id = pending.pop()
            if node_id in seen or node_id == 0:
                continue
            seen.add(node_id)
            if node_id != 1:
                pending.extend(child for _, child in self.arena.nodes[node_id].edges)
        return seen

    @property
    def node_count(self) -> int:
        return len(self.reachable_nodes())

    @property
    def edge_count(self) -> int:
        return sum(
            len(self.arena.nodes[node].edges)
            for node in self.reachable_nodes()
            if node != 1
        )

    def path_count(self) -> int:
        memo = {0: 0, 1: 1}

        def count(node_id: int) -> int:
            cached = memo.get(node_id)
            if cached is not None:
                return cached
            value = sum(count(child) for _, child in self.arena.nodes[node_id].edges)
            memo[node_id] = value
            return value

        return count(self.root)

    def contains(self, word: Iterable[str]) -> bool:
        values = tuple(word)
        if len(values) != self.horizon or self.root == 0:
            return False
        node = self.root
        for move in values:
            if node in (0, 1):
                return False
            node = dict(self.arena.nodes[node].edges).get(MOVE_INDEX[move], 0)
            if node == 0:
                return False
        return node == 1

    def sample(self, limit: int = 1) -> tuple[tuple[str, ...], ...]:
        rows: list[tuple[str, ...]] = []

        def visit(node: int, prefix: tuple[str, ...]) -> None:
            if len(rows) >= int(limit) or node == 0:
                return
            if node == 1:
                rows.append(prefix)
                return
            for move_id, child in self.arena.nodes[node].edges:
                visit(child, prefix + (MOVE_NAMES[move_id],))

        visit(self.root, ())
        return tuple(rows)

    def accepts_packed(self, moves: np.ndarray) -> np.ndarray:
        rows = np.asarray(moves, dtype=np.uint8)
        if rows.ndim != 2 or rows.shape[1] != self.horizon:
            raise ValueError("packed word horizon mismatch")
        transition = np.zeros((len(self.arena.nodes), len(MOVE_NAMES)), dtype=np.int64)
        for node_id, node in enumerate(self.arena.nodes[2:], start=2):
            for move_id, child in node.edges:
                transition[node_id, move_id] = child
        current = np.full(len(rows), self.root, dtype=np.int64)
        for column in range(self.horizon):
            active = current > 1
            if not np.any(active):
                current = np.zeros(len(rows), dtype=np.int64)
                break
            nxt = np.zeros(len(rows), dtype=np.int64)
            nxt[active] = transition[current[active], rows[active, column]]
            current = nxt
        return current == 1

    def supported_move_masks(self) -> tuple[int, ...]:
        masks = []
        layer = {self.root} if self.root else set()
        for _ in range(self.horizon):
            mask = 0
            nxt: set[int] = set()
            for node_id in layer:
                if node_id in (0, 1):
                    continue
                for move_id, child in self.arena.nodes[node_id].edges:
                    mask |= 1 << move_id
                    nxt.add(child)
            masks.append(mask)
            layer = nxt
        return tuple(masks)

    def structural_hash(self) -> str:
        memo = {0: sha256(b"FALSE").hexdigest(), 1: sha256(b"TRUE").hexdigest()}

        def visit(node_id: int) -> str:
            cached = memo.get(node_id)
            if cached is not None:
                return cached
            node = self.arena.nodes[node_id]
            payload = f"{node.remaining}|" + ";".join(
                f"{move_id}:{visit(child)}" for move_id, child in node.edges
            )
            value = sha256(payload.encode()).hexdigest()
            memo[node_id] = value
            return value

        return visit(self.root)

    def metrics(self) -> dict[str, object]:
        paths = self.path_count()
        nodes = self.node_count
        return {
            "paths": paths,
            "nodes": nodes,
            "edges": self.edge_count,
            "path_to_node_compression": paths / nodes if nodes else 0.0,
            "structural_hash": self.structural_hash(),
            "full_words_materialized": self.full_words_materialized,
            "python_word_set_materialized": self.python_word_set_materialized,
            "construction": self.construction,
        }

    def to_bytes(self) -> bytes:
        payload = {
            "root": self.root,
            "horizon": self.horizon,
            "previous_face": self.previous_face,
            "scope_hash": self.scope_hash,
            "construction": self.construction,
            "node_cap": self.arena.node_cap,
            "nodes": [
                {
                    "remaining": node.remaining,
                    "accepting": node.accepting,
                    "edges": [list(edge) for edge in node.edges],
                }
                for node in self.arena.nodes
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "SymbolicMDD":
        payload = json.loads(data.decode())
        arena = NodeArena(node_cap=max(int(payload["node_cap"]), len(payload["nodes"]) + 1))
        arena.nodes = []
        arena.unique = {}
        for node_id, row in enumerate(payload["nodes"]):
            node = SymbolicNode(
                int(row["remaining"]),
                tuple((int(move), int(child)) for move, child in row["edges"]),
                bool(row["accepting"]),
            )
            arena.nodes.append(node)
            if node_id > 1:
                arena.unique[(node.remaining, node.edges)] = node_id
        return cls(
            arena=arena,
            root=int(payload["root"]),
            horizon=int(payload["horizon"]),
            previous_face=payload["previous_face"],
            scope_hash=str(payload["scope_hash"]),
            construction=str(payload["construction"]),
        )


__all__ = [
    "NodeArena",
    "ResourceCapUnknown",
    "SymbolicMDD",
    "SymbolicNode",
]
