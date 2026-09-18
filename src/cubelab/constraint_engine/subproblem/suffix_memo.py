from __future__ import annotations

"""Branch-shared immutable exact suffix memo."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

from .subproblem_key import JointSupportSubproblemKey


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class MemoLookup(Generic[T]):
    value: T | None
    hit: bool
    unknown_reused: bool


class ExactSuffixMemoStore(Generic[T]):
    """LRU whose eviction is logically neutral.

    Proven results are cap independent.  UNKNOWN results are reused only when
    the new remaining budget is no larger than the budget that already failed.
    A larger budget always recomputes.
    """

    def __init__(self, maximum_entries: int = 100_000) -> None:
        if maximum_entries < 1:
            raise ValueError("maximum_entries must be positive")
        self.maximum_entries = maximum_entries
        self._entries: OrderedDict[
            JointSupportSubproblemKey, T
        ] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.proven_hits = 0
        self.unknown_hits = 0
        self.puts = 0
        self.evictions = 0

    def get(
        self,
        key: JointSupportSubproblemKey,
        *,
        remaining_node_budget: int,
    ) -> MemoLookup[T]:
        value = self._entries.get(key)
        if value is None:
            self.misses += 1
            return MemoLookup(None, False, False)
        status = str(getattr(value, "status", ""))
        if status == "UNKNOWN_CAP":
            recorded = int(getattr(value, "unknown_budget", 0))
            if remaining_node_budget > recorded:
                self.misses += 1
                return MemoLookup(None, False, False)
            self.unknown_hits += 1
            unknown = True
        else:
            self.proven_hits += 1
            unknown = False
        self.hits += 1
        self._entries.move_to_end(key)
        return MemoLookup(value, True, unknown)

    def put(self, key: JointSupportSubproblemKey, value: T) -> None:
        self._entries[key] = value
        self._entries.move_to_end(key)
        self.puts += 1
        while len(self._entries) > self.maximum_entries:
            self._entries.popitem(last=False)
            self.evictions += 1

    def metrics_row(self) -> dict[str, object]:
        lookups = self.hits + self.misses
        return {
            "entries": len(self._entries),
            "maximum_entries": self.maximum_entries,
            "hits": self.hits,
            "misses": self.misses,
            "proven_hits": self.proven_hits,
            "unknown_hits": self.unknown_hits,
            "puts": self.puts,
            "evictions": self.evictions,
            "hit_rate": self.hits / lookups if lookups else 0.0,
        }


__all__ = ["MemoLookup", "ExactSuffixMemoStore"]
