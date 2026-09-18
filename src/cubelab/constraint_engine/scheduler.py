from __future__ import annotations

"""Fixed v101.5 scheduler; adaptive scheduling is intentionally deferred."""

from dataclasses import dataclass
from typing import Sequence

from .types import IncrementalConstraint


@dataclass(frozen=True, slots=True)
class FixedConstraintScheduler:
    constraints: tuple[IncrementalConstraint, ...]
    policy_name: str = "V101.5_FIXED_CHEAP_TO_EXPENSIVE"

    @classmethod
    def from_sequence(
        cls, constraints: Sequence[IncrementalConstraint]
    ) -> "FixedConstraintScheduler":
        return cls(tuple(constraints))

    def ordered_constraints(self) -> tuple[IncrementalConstraint, ...]:
        return self.constraints

    def row(self) -> dict[str, object]:
        return {
            "policy_name": self.policy_name,
            "dynamic": False,
            "constraints": [
                {
                    "order": index + 1,
                    "name": value.name,
                    "estimated_cost_class": value.estimated_cost_class,
                }
                for index, value in enumerate(self.constraints)
            ],
        }


__all__ = ["FixedConstraintScheduler"]
