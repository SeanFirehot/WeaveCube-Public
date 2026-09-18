#!/usr/bin/env python3
"""CubeLab v37.0 - Orientation-led symbolic COLUMN domains.

Research-only. No production solver changes.

Provides:
- ColumnDomain: 18-bit HTM move domain.
- D6MoveDomainOracle: lifts the proven d6 no-suffix refuter into a supported
  next-move domain query.
- RestrictedOrientationMDD: keeps several future COLUMN domains unresolved
  simultaneously and removes move values with no globally supported
  true-last-DR-entry path.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import search_twist_skeleton as ts

from p1_tail_nosuffix_v36_6i13 import (
    CTX_ID,
    NoSuffixMembershipTable,
    NoSuffixRefuter,
)

MOVE_ORDER = tuple(ts.MOVE_ORDER)
MOVE_INDEX = {m: i for i, m in enumerate(MOVE_ORDER)}
FULL_MASK = (1 << len(MOVE_ORDER)) - 1


@dataclass(frozen=True)
class ColumnDomain:
    mask: int

    def __post_init__(self):
        object.__setattr__(self, "mask", int(self.mask) & FULL_MASK)

    @classmethod
    def full(cls):
        return cls(FULL_MASK)

    @classmethod
    def empty(cls):
        return cls(0)

    @classmethod
    def from_moves(cls, moves: Iterable[str]):
        mask = 0
        for move in moves:
            if move not in MOVE_INDEX:
                raise ValueError(f"unknown HTM move: {move!r}")
            mask |= 1 << MOVE_INDEX[move]
        return cls(mask)

    def moves(self):
        return tuple(
            m for i, m in enumerate(MOVE_ORDER)
            if self.mask & (1 << i)
        )

    def size(self):
        return self.mask.bit_count()

    def contains(self, move):
        return bool(self.mask & (1 << MOVE_INDEX[move]))

    def is_singleton(self):
        return self.size() == 1

    def singleton_move(self):
        xs = self.moves()
        return xs[0] if len(xs) == 1 else None


@dataclass(frozen=True)
class OrientationState:
    q: int
    last_face: str | None


@dataclass(frozen=True)
class OrientationArc:
    move: str
    target: OrientationState


@dataclass
class OrientationMDDResult:
    initial_domains: tuple[ColumnDomain, ...]
    reduced_domains: tuple[ColumnDomain, ...]
    layers: tuple[frozenset[OrientationState], ...]
    supported_states: tuple[frozenset[OrientationState], ...]
    supported_arcs: tuple[dict[OrientationState, tuple[OrientationArc, ...]], ...]
    all_arc_count: int
    supported_arc_count: int
    accepted_word_count: int

    @property
    def horizon(self):
        return len(self.initial_domains)

    @property
    def initial_domain_sizes(self):
        return tuple(d.size() for d in self.initial_domains)

    @property
    def reduced_domain_sizes(self):
        return tuple(d.size() for d in self.reduced_domains)

    @property
    def singleton_columns(self):
        return tuple(i for i,d in enumerate(self.reduced_domains) if d.is_singleton())

    @property
    def impossible(self):
        return self.accepted_word_count == 0


class D6MoveDomainOracle:
    def __init__(self, table, co, eo, sl, cos, eos):
        self.table = table
        self.refuter = NoSuffixRefuter(table)
        self.co, self.eo, self.sl = co, eo, sl
        self.cos, self.eos = cos, eos

    @classmethod
    def load(cls, path, co, eo, sl, cos, eos):
        return cls(
            NoSuffixMembershipTable.load(path),
            co, eo, sl, cos, eos,
        )

    @lru_cache(maxsize=500_000)
    def allowed_mask(self, q: int, remaining: int, previous_face: str | None):
        remaining = int(remaining)
        if remaining <= 0:
            return 0
        if remaining > int(self.table.max_depth):
            raise ValueError(
                f"remaining={remaining} > table.max_depth={self.table.max_depth}"
            )
        child_remaining = remaining - 1
        mask = 0
        for mi, move in enumerate(MOVE_ORDER):
            if not ts.allow(previous_face, move):
                continue
            nq = ts.q_move(q, mi, self.co, self.eo, self.sl)
            if ts.phase1_lb(nq, self.cos, self.eos) > child_remaining:
                continue
            ans = self.refuter.query(
                nq, child_remaining, CTX_ID[move[0]], q
            )
            if ans is not False:
                mask |= 1 << mi
        return mask

    def allowed_domain(self, q, remaining, previous_face):
        return ColumnDomain(
            self.allowed_mask(q, remaining, previous_face)
        )


def build_restricted_orientation_mdd(q0: int, domains: Sequence[ColumnDomain], co, eo, sl):
    initial_domains = tuple(domains)
    horizon = len(initial_domains)

    layers = [{OrientationState(int(q0), None)}]
    all_arcs = []
    all_arc_count = 0

    for domain in initial_domains:
        current = layers[-1]
        next_layer = set()
        layer_arcs = {}
        for state in current:
            arcs = []
            for move in domain.moves():
                if not ts.allow(state.last_face, move):
                    continue
                nq = ts.q_move(state.q, MOVE_INDEX[move], co, eo, sl)
                target = OrientationState(nq, move[0])
                arcs.append(OrientationArc(move, target))
                next_layer.add(target)
                all_arc_count += 1
            if arcs:
                layer_arcs[state] = arcs
        all_arcs.append(layer_arcs)
        layers.append(next_layer)

    supported_states = [set() for _ in range(horizon + 1)]
    supported_arcs = [dict() for _ in range(horizon)]

    if horizon:
        i = horizon - 1
        for state, arcs in all_arcs[i].items():
            keep = tuple(
                arc for arc in arcs
                if arc.target.q == ts.GOAL_Q and state.q != ts.GOAL_Q
            )
            if keep:
                supported_arcs[i][state] = keep
                supported_states[i].add(state)
                supported_states[i+1].update(a.target for a in keep)

        for i in range(horizon-2, -1, -1):
            nxt = supported_states[i+1]
            for state, arcs in all_arcs[i].items():
                keep = tuple(a for a in arcs if a.target in nxt)
                if keep:
                    supported_arcs[i][state] = keep
                    supported_states[i].add(state)

    reduced = []
    supported_arc_count = 0
    for i in range(horizon):
        mask = 0
        for arcs in supported_arcs[i].values():
            supported_arc_count += len(arcs)
            for arc in arcs:
                mask |= 1 << MOVE_INDEX[arc.move]
        reduced.append(ColumnDomain(mask))

    counts = {state: 1 for state in supported_states[horizon]}
    for i in range(horizon-1, -1, -1):
        cur = {}
        for state, arcs in supported_arcs[i].items():
            total = sum(counts.get(a.target, 0) for a in arcs)
            if total:
                cur[state] = total
        counts = cur

    start = OrientationState(int(q0), None)
    return OrientationMDDResult(
        initial_domains=initial_domains,
        reduced_domains=tuple(reduced),
        layers=tuple(frozenset(x) for x in layers),
        supported_states=tuple(frozenset(x) for x in supported_states),
        supported_arcs=tuple(supported_arcs),
        all_arc_count=all_arc_count,
        supported_arc_count=supported_arc_count,
        accepted_word_count=counts.get(start, 0),
    )


def enumerate_supported_words(result: OrientationMDDResult, q0: int, limit=None):
    start = OrientationState(int(q0), None)
    out = []
    path = []

    def rec(layer, state):
        if limit is not None and len(out) >= limit:
            return
        if layer == result.horizon:
            out.append(tuple(path))
            return
        for arc in result.supported_arcs[layer].get(state, ()):
            path.append(arc.move)
            rec(layer+1, arc.target)
            path.pop()

    if result.accepted_word_count:
        rec(0, start)
    return tuple(out)
