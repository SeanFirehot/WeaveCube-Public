from __future__ import annotations

"""Requirement/removal provenance graph for auditable propagation."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json


@dataclass(frozen=True, slots=True)
class ProvenanceNode:
    node_id: str
    node_type: str
    label: str
    payload: dict[str, object]

    def row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProvenanceEdge:
    source_id: str
    target_id: str
    relation: str

    def row(self) -> dict[str, object]:
        return asdict(self)


class ProvenanceGraph:
    def __init__(self) -> None:
        self.nodes: list[ProvenanceNode] = []
        self.edges: list[ProvenanceEdge] = []
        self._node_ids: set[str] = set()
        self._edge_keys: set[tuple[str, str, str]] = set()

    def snapshot(self) -> tuple[int, int]:
        return len(self.nodes), len(self.edges)

    def restore(self, snapshot: tuple[int, int]) -> None:
        node_count, edge_count = snapshot
        del self.nodes[node_count:]
        del self.edges[edge_count:]
        self._node_ids = {node.node_id for node in self.nodes}
        self._edge_keys = {
            (edge.source_id, edge.target_id, edge.relation)
            for edge in self.edges
        }

    def add_node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        payload: dict[str, object] | None = None,
    ) -> str:
        if node_id not in self._node_ids:
            self.nodes.append(
                ProvenanceNode(
                    node_id=node_id,
                    node_type=node_type,
                    label=label,
                    payload={} if payload is None else dict(payload),
                )
            )
            self._node_ids.add(node_id)
        return node_id

    def add_edge(self, source_id: str, target_id: str, relation: str) -> None:
        edge = ProvenanceEdge(source_id, target_id, relation)
        key = (source_id, target_id, relation)
        if key not in self._edge_keys:
            self.edges.append(edge)
            self._edge_keys.add(key)

    def record_requirement(
        self,
        requirement_id: str,
        constraint_name: str,
        requirement_type: str,
    ) -> None:
        source = self.add_node(
            f"CONSTRAINT:{constraint_name}",
            "constraint",
            constraint_name,
        )
        target = self.add_node(
            requirement_id,
            "requirement",
            requirement_type,
        )
        self.add_edge(source, target, "generated")

    def record_removal(
        self,
        piece_id: str,
        candidate_id: str,
        constraint_name: str,
        reason: str,
        requirement_ids: tuple[str, ...],
    ) -> None:
        removal_id = (
            "REMOVE-"
            + sha256(
                f"{piece_id}|{candidate_id}|{constraint_name}|{reason}".encode()
            ).hexdigest()[:20]
        )
        self.add_node(
            removal_id,
            "candidate_removal",
            candidate_id,
            {
                "piece_id": piece_id,
                "constraint": constraint_name,
                "reason": reason,
            },
        )
        source = self.add_node(
            f"CONSTRAINT:{constraint_name}",
            "constraint",
            constraint_name,
        )
        self.add_edge(source, removal_id, "removed")
        for requirement_id in requirement_ids:
            self.add_edge(requirement_id, removal_id, "supports")

    def record_assignment(
        self,
        piece_id: str,
        candidate_id: str,
        reason: str,
    ) -> None:
        assignment_id = f"ASSIGN:{piece_id}:{candidate_id}"
        self.add_node(
            assignment_id,
            "forced_assignment",
            candidate_id,
            {"piece_id": piece_id, "reason": reason},
        )

    def state_hash_payload(self) -> object:
        return {
            "nodes": [node.node_id for node in self.nodes],
            "edges": [
                (edge.source_id, edge.target_id, edge.relation)
                for edge in self.edges
            ],
        }

    def row(self) -> dict[str, object]:
        return {
            "nodes": [node.row() for node in self.nodes],
            "edges": [edge.row() for edge in self.edges],
        }


__all__ = [
    "ProvenanceNode",
    "ProvenanceEdge",
    "ProvenanceGraph",
]
