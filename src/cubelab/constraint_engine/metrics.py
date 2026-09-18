from __future__ import annotations

"""Direct and conditional-marginal utility metrics."""

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable

from .types import ConstraintResult, ConstraintStatus


@dataclass(slots=True)
class ConstraintMetric:
    constraint_name: str
    calls: int = 0
    total_runtime_ns: int = 0
    maximum_runtime_ns: int = 0
    cumulative_domain_before: int = 0
    cumulative_domain_after: int = 0
    removed_candidates: int = 0
    forced_candidates: int = 0
    generated_requirements: int = 0
    consumed_requirements: int = 0
    unknown_count: int = 0
    proven_unsat_count: int = 0
    invocation_round_max: int = 0

    def row(self) -> dict[str, object]:
        payload = asdict(self)
        payload["average_runtime_ns"] = (
            0 if self.calls == 0 else self.total_runtime_ns / self.calls
        )
        payload["marginal_domain_reduction"] = (
            self.cumulative_domain_before - self.cumulative_domain_after
        )
        return payload


class ConstraintMetrics:
    def __init__(self) -> None:
        self._metrics: dict[str, ConstraintMetric] = {}
        self.results: list[dict[str, object]] = []
        self.requirement_consumers: dict[str, set[str]] = defaultdict(set)

    def record(self, result: ConstraintResult, *, round_index: int) -> None:
        metric = self._metrics.setdefault(
            result.constraint_name,
            ConstraintMetric(result.constraint_name),
        )
        metric.calls += 1
        metric.total_runtime_ns += result.runtime_ns
        metric.maximum_runtime_ns = max(
            metric.maximum_runtime_ns, result.runtime_ns
        )
        metric.cumulative_domain_before += result.domain_size_before
        metric.cumulative_domain_after += result.domain_size_after
        metric.removed_candidates += len(result.removals)
        metric.forced_candidates += len(result.forced_assignments)
        metric.generated_requirements += len(result.requirements)
        metric.consumed_requirements += len(
            result.consumed_requirement_ids
        )
        metric.unknown_count += int(
            result.status is ConstraintStatus.UNKNOWN_CAP
        )
        metric.proven_unsat_count += int(
            result.status is ConstraintStatus.UNSAT_PROVEN
        )
        metric.invocation_round_max = max(
            metric.invocation_round_max, round_index
        )
        for requirement_id in result.consumed_requirement_ids:
            self.requirement_consumers[requirement_id].add(
                result.constraint_name
            )
        self.results.append(
            {
                "round_index": round_index,
                **result.row(),
            }
        )

    def rows(self) -> tuple[dict[str, object], ...]:
        return tuple(
            self._metrics[name].row() for name in sorted(self._metrics)
        )

    def requirement_reuse_row(
        self, generated_requirement_ids: Iterable[str]
    ) -> dict[str, object]:
        identifiers = tuple(generated_requirement_ids)
        reused = {
            requirement_id: sorted(
                self.requirement_consumers.get(requirement_id, set())
            )
            for requirement_id in identifiers
            if self.requirement_consumers.get(requirement_id)
        }
        return {
            "generated_requirement_count": len(identifiers),
            "consumed_requirement_count": len(reused),
            "requirement_consumers": reused,
        }


__all__ = ["ConstraintMetric", "ConstraintMetrics"]
