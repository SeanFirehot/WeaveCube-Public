from __future__ import annotations

"""Bounded logical-result memo for exact state-slot joint support."""

from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Generic, TypeVar


@dataclass(frozen=True, slots=True)
class JointSupportKey:
    inventory_signature: str
    active_domain_signature: str
    selected_signature: str
    node_cap: int
    compute_candidate_support: bool

    def row(self) -> dict[str, object]:
        return asdict(self)


T = TypeVar("T")


class JointSupportCache(Generic[T]):
    """LRU memo.

    Entries contain immutable solver results.  This store is deliberately
    non-logical: eviction or growth cannot affect satisfiability, and branch
    restoration therefore restores the branch-local hot entry while allowing
    this shared memo to survive across sibling branches.
    """

    def __init__(self, *, maximum_entries: int = 256) -> None:
        if maximum_entries < 1:
            raise ValueError("maximum_entries must be positive")
        self.maximum_entries = maximum_entries
        self._entries: OrderedDict[JointSupportKey, T] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: JointSupportKey) -> T | None:
        value = self._entries.get(key)
        if value is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: JointSupportKey, value: T) -> None:
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.maximum_entries:
            self._entries.popitem(last=False)
            self.evictions += 1

    def clear(self) -> None:
        self._entries.clear()

    def metrics_row(self) -> dict[str, object]:
        total = self.hits + self.misses
        return {
            "maximum_entries": self.maximum_entries,
            "entry_count": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": (self.hits / total if total else 0.0),
        }


__all__ = ["JointSupportKey", "JointSupportCache"]
