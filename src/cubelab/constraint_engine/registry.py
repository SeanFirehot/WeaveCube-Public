from __future__ import annotations

"""Ordered constraint registration with unique names."""

from dataclasses import dataclass
from typing import Iterable

from .types import IncrementalConstraint


@dataclass(frozen=True, slots=True)
class ConstraintRegistryEntry:
    order: int
    name: str
    tier: str
    estimated_cost_class: int

    def row(self) -> dict[str, object]:
        return {
            "order": self.order,
            "name": self.name,
            "tier": self.tier,
            "estimated_cost_class": self.estimated_cost_class,
        }


class ConstraintRegistry:
    def __init__(
        self, constraints: Iterable[IncrementalConstraint] = ()
    ) -> None:
        self._constraints: list[IncrementalConstraint] = []
        for constraint in constraints:
            self.register(constraint)

    def register(self, constraint: IncrementalConstraint) -> None:
        if any(value.name == constraint.name for value in self._constraints):
            raise ValueError(f"duplicate constraint name: {constraint.name}")
        self._constraints.append(constraint)

    @property
    def constraints(self) -> tuple[IncrementalConstraint, ...]:
        return tuple(self._constraints)

    def rows(self) -> tuple[dict[str, object], ...]:
        return tuple(
            ConstraintRegistryEntry(
                order=index + 1,
                name=constraint.name,
                tier=constraint.tier.value,
                estimated_cost_class=constraint.estimated_cost_class,
            ).row()
            for index, constraint in enumerate(self._constraints)
        )


__all__ = ["ConstraintRegistryEntry", "ConstraintRegistry"]
