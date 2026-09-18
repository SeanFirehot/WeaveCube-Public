"""High-level facade for the PDCC pose-difference engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .analysis import (
    StructureSignature,
    current_adjacency,
    native_boundary,
    relative_difference,
    rigid_bundles,
    structure_signature,
    task_bundles,
    twist_clusters,
)
from .state import PDCCState
from .trace import ColumnTrace, build_trace
from .validation import ValidationReport, run_structural_validation


@dataclass(frozen=True, slots=True)
class PDCCEngine:
    """Stateless API object; all core tables are deterministic module singletons."""

    version: str = "PDCC-v1.0.0-alpha.1"

    def solved(self) -> PDCCState:
        return PDCCState.solved()

    def from_word(
        self, word: str | Iterable[str], *, initial: PDCCState | None = None
    ) -> PDCCState:
        return (initial or self.solved()).apply_word(word)

    def from_coordinates(
        self,
        *,
        permutation: Mapping[str, str] | None = None,
        orientation: Mapping[str, int] | None = None,
        require_legal: bool = True,
    ) -> PDCCState:
        return PDCCState.from_coordinates(
            permutation=permutation,
            orientation=orientation,
            require_legal=require_legal,
        )

    def superflip(self) -> PDCCState:
        return PDCCState.superflip()

    def trace(
        self, word: str | Iterable[str], *, initial: PDCCState | None = None
    ) -> ColumnTrace:
        return build_trace(initial or self.solved(), word)

    def signature(self, state: PDCCState) -> StructureSignature:
        return structure_signature(state)

    def relative_difference(self, state: PDCCState, left: str, right: str) -> int:
        return relative_difference(state, left, right)

    def rigid_bundles(self, state: PDCCState) -> tuple[tuple[str, ...], ...]:
        return rigid_bundles(state)

    def twist_clusters(self, state: PDCCState) -> tuple[tuple[str, ...], ...]:
        return twist_clusters(state)

    def native_boundary(self, state: PDCCState) -> tuple[tuple[str, str, int], ...]:
        return native_boundary(state)

    def current_adjacency(self, state: PDCCState) -> tuple[tuple[str, str], ...]:
        return current_adjacency(state)

    def task_bundles(
        self, before: PDCCState, after: PDCCState
    ) -> tuple[tuple[int, tuple[str, ...]], ...]:
        return task_bundles(before, after)

    def validate(self, *, random_steps: int = 5000) -> ValidationReport:
        return run_structural_validation(random_steps=random_steps)
