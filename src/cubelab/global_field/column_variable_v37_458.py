#!/usr/bin/env python3
"""
CubeLab v37.458-A2
ELASTIC UNASSIGNED GLOBAL_COLUMN VARIABLES

This module contains representation mechanics only.

A GLOBAL_COLUMN is a CSP variable whose domain is an 18-bit move mask.
Columns:
- are not required to be assigned chronologically,
- have stable UIDs independent of list position,
- may be inserted at arbitrary positions,
- may be restricted, assigned, unassigned, or removed,
- do not carry global-H / remaining-length semantics.

No search, propagation, ATO terminal, ORI-prefix, or K2/PER-suffix lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from cubelab.global_field import ato_bidirectional_v37_458 as bi


ALL_MOVES_MASK = (1 << 18) - 1


def _normalize_domain(mask: int) -> int:
    value = int(mask)
    if value < 0 or value & ~ALL_MOVES_MASK:
        raise ValueError(f"invalid 18-bit GLOBAL_COLUMN domain: {value:#x}")
    return value


@dataclass
class ColumnVariable:
    uid: int
    domain_mask: int = ALL_MOVES_MASK
    assignment: int | None = None
    reason: str | None = None
    _pre_assignment_domain_mask: int | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.uid = int(self.uid)
        self.domain_mask = _normalize_domain(self.domain_mask)

        if self.assignment is not None:
            mid = bi.move_id(self.assignment)
            bit = 1 << mid
            if not (self.domain_mask & bit):
                raise ValueError(
                    f"assignment {mid} not present in domain {self.domain_mask:#x}"
                )
            self.assignment = mid
            self._pre_assignment_domain_mask = int(self.domain_mask)
            self.domain_mask = bit

    @property
    def domain_size(self) -> int:
        return int(self.domain_mask.bit_count())

    @property
    def is_assigned(self) -> bool:
        return self.assignment is not None

    @property
    def is_contradiction(self) -> bool:
        return self.domain_mask == 0

    @property
    def move_tokens(self) -> tuple[str, ...]:
        return tuple(
            bi.MOVE_NAMES[mid]
            for mid in range(18)
            if self.domain_mask & (1 << mid)
        )

    def restrict(self, allowed_mask: int) -> bool:
        """
        Intersect this column with an allowed 18-bit mask.

        Returns True iff the domain changed.
        """
        allowed = _normalize_domain(allowed_mask)
        new_mask = int(self.domain_mask) & allowed
        changed = new_mask != int(self.domain_mask)
        self.domain_mask = int(new_mask)

        if self.assignment is not None:
            bit = 1 << int(self.assignment)
            if not (self.domain_mask & bit):
                # Keep the contradictory state explicit; do not silently
                # unassign an implication/decision.
                self.domain_mask = 0

        return changed

    def assign(self, move: int | str, *, reason: str = "decision") -> None:
        if self.assignment is not None:
            raise RuntimeError(
                f"column {self.uid} already assigned to "
                f"{bi.MOVE_NAMES[int(self.assignment)]}"
            )

        mid = bi.move_id(move)
        bit = 1 << mid

        if not (self.domain_mask & bit):
            raise ValueError(
                f"move {bi.MOVE_NAMES[mid]} absent from column {self.uid} "
                f"domain={self.move_tokens}"
            )

        self._pre_assignment_domain_mask = int(self.domain_mask)
        self.assignment = int(mid)
        self.reason = str(reason)
        self.domain_mask = int(bit)

    def unassign(self) -> None:
        if self.assignment is None:
            return

        if self._pre_assignment_domain_mask is None:
            raise RuntimeError(
                f"column {self.uid} has no pre-assignment domain snapshot"
            )

        self.domain_mask = int(self._pre_assignment_domain_mask)
        self.assignment = None
        self.reason = None
        self._pre_assignment_domain_mask = None

    def to_json(self) -> dict:
        return {
            "uid": int(self.uid),
            "domain_mask": int(self.domain_mask),
            "domain_size": int(self.domain_size),
            "domain_moves": list(self.move_tokens),
            "assignment": (
                None
                if self.assignment is None
                else bi.MOVE_NAMES[int(self.assignment)]
            ),
            "assignment_move_id": (
                None
                if self.assignment is None
                else int(self.assignment)
            ),
            "reason": self.reason,
        }


class ElasticColumnField:
    """
    Elastic ordered container of GLOBAL_COLUMN variables.

    List position is mutable; UID is stable.
    The exact endpoint anchors are stored, but A2 deliberately does not claim
    any path-consistency between the unresolved columns and those anchors.
    That consistency belongs to A3 bidirectional propagation.
    """

    def __init__(
        self,
        *,
        left_anchor_q: Iterable[int],
        right_anchor_q: Iterable[int],
        initial_columns: int = 1,
    ) -> None:
        if int(initial_columns) < 0:
            raise ValueError("initial_columns must be >= 0")

        self.left_anchor_q = bi.normalize_q(left_anchor_q)
        self.right_anchor_q = bi.normalize_q(right_anchor_q)

        self._next_uid = 0
        self.columns: list[ColumnVariable] = []

        for _ in range(int(initial_columns)):
            self.insert_column(len(self.columns))

    @property
    def materialized_length(self) -> int:
        return len(self.columns)

    @property
    def has_contradiction(self) -> bool:
        return any(column.is_contradiction for column in self.columns)

    @property
    def all_assigned(self) -> bool:
        return all(column.is_assigned for column in self.columns)

    def _new_uid(self) -> int:
        uid = int(self._next_uid)
        self._next_uid += 1
        return uid

    def index_of(self, uid: int) -> int:
        target = int(uid)
        for index, column in enumerate(self.columns):
            if int(column.uid) == target:
                return index
        raise KeyError(f"unknown GLOBAL_COLUMN uid={uid}")

    def get(self, uid: int) -> ColumnVariable:
        return self.columns[self.index_of(uid)]

    def insert_column(
        self,
        position: int,
        *,
        domain_mask: int = ALL_MOVES_MASK,
    ) -> int:
        pos = int(position)
        if not 0 <= pos <= len(self.columns):
            raise IndexError(
                f"insert position {pos} outside [0,{len(self.columns)}]"
            )

        uid = self._new_uid()
        column = ColumnVariable(
            uid=uid,
            domain_mask=_normalize_domain(domain_mask),
        )
        self.columns.insert(pos, column)
        return uid

    def remove_column(self, uid: int) -> ColumnVariable:
        index = self.index_of(uid)
        column = self.columns[index]

        if column.is_assigned:
            raise RuntimeError(
                f"cannot remove assigned column uid={uid}; unassign first"
            )

        return self.columns.pop(index)

    def restrict_domain(self, uid: int, allowed_mask: int) -> bool:
        return self.get(uid).restrict(allowed_mask)

    def assign(
        self,
        uid: int,
        move: int | str,
        *,
        reason: str = "decision",
    ) -> None:
        self.get(uid).assign(move, reason=reason)

    def unassign(self, uid: int) -> None:
        self.get(uid).unassign()

    def snapshot(self) -> dict:
        return {
            "left_anchor_q": [int(x) for x in self.left_anchor_q],
            "right_anchor_q": [int(x) for x in self.right_anchor_q],
            "materialized_length": int(self.materialized_length),
            "next_uid": int(self._next_uid),
            "columns": [column.to_json() for column in self.columns],
        }
