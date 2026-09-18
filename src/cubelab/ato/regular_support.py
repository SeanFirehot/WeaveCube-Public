"""Generic finite-state GLOBAL COLUMN support propagation for v37.352.

Every factor consumes the same explicit move label at each column.  The kernel
performs regular-language generalized arc consistency only; it never branches
and never combines independent per-factor witness words into a positive.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np

from cubelab.pdcc.moves import MOVE_ORDER


MOVE_NAMES: tuple[str, ...] = tuple(MOVE_ORDER)
MOVE_INDEX = {move: index for index, move in enumerate(MOVE_NAMES)}
FACE_ORDER: tuple[str, ...] = tuple("URFDLB")
FACE_INDEX = {face: index for index, face in enumerate(FACE_ORDER)}
OPPOSITE = {"U": "D", "D": "U", "R": "L", "L": "R", "F": "B", "B": "F"}


def canonical_allow(previous_face: str | None, move: str) -> bool:
    """Current production ``search_twist_skeleton.allow`` semantics."""

    face = move[0]
    if previous_face == face:
        return False
    if (
        previous_face
        and OPPOSITE[face] == previous_face
        and FACE_INDEX[face] < FACE_INDEX[previous_face]
    ):
        return False
    return True


def is_canonical_word(word: Sequence[str], previous_face: str | None = None) -> bool:
    last = previous_face
    for move in word:
        if not canonical_allow(last, move):
            return False
        last = move[0]
    return True


def iter_canonical_words(
    exact_depth: int,
    previous_face: str | None = None,
) -> Iterable[tuple[str, ...]]:
    """Yield the complete production-canonical exact-depth language."""

    depth = int(exact_depth)
    if depth < 0:
        raise ValueError("exact_depth must be non-negative")

    def visit(prefix: tuple[str, ...], last: str | None):
        if len(prefix) == depth:
            yield prefix
            return
        for move in MOVE_NAMES:
            if canonical_allow(last, move):
                yield from visit(prefix + (move,), move[0])

    yield from visit((), previous_face)


def canonical_word_arrays(
    exact_depth: int,
    previous_face: str | None = None,
) -> np.ndarray:
    words = tuple(iter_canonical_words(exact_depth, previous_face))
    if exact_depth == 0:
        return np.empty((1, 0), dtype=np.uint8)
    return np.asarray(
        [[MOVE_INDEX[move] for move in word] for word in words],
        dtype=np.uint8,
    )


def canonical_counts(maximum_depth: int) -> tuple[int, ...]:
    counts = [1]
    vector = np.zeros(7, dtype=np.int64)
    vector[6] = 1
    for _ in range(int(maximum_depth)):
        nxt = np.zeros(7, dtype=np.int64)
        for state, count in enumerate(vector):
            if not count:
                continue
            previous = None if state == 6 else FACE_ORDER[state]
            for move in MOVE_NAMES:
                if canonical_allow(previous, move):
                    nxt[FACE_INDEX[move[0]]] += count
        vector = nxt
        counts.append(int(vector.sum()))
    return tuple(counts)


@dataclass(frozen=True, slots=True)
class RegularFactor:
    """One nondeterministic finite-state factor over GLOBAL move labels."""

    name: str
    state_count: int
    initial_mask: int
    target_mask: int
    # ``relation[state, move]`` is a bit mask of possible next states.
    relation: np.ndarray

    def __post_init__(self) -> None:
        if self.relation.shape != (self.state_count, len(MOVE_NAMES)):
            raise ValueError(
                f"{self.name}: relation shape {self.relation.shape}, expected "
                f"({self.state_count}, {len(MOVE_NAMES)})"
            )
        if self.state_count > 63:
            raise ValueError("Bit-mask kernel supports at most 63 states per factor")


def canonical_adjacency_factor(previous_face: str | None = None) -> RegularFactor:
    # States 0..5 mean last face; 6 is the interval-start sentinel.
    relation = np.zeros((7, len(MOVE_NAMES)), dtype=np.uint64)
    for state in range(7):
        previous = previous_face if state == 6 else FACE_ORDER[state]
        for move_id, move in enumerate(MOVE_NAMES):
            if canonical_allow(previous, move):
                relation[state, move_id] = np.uint64(1 << FACE_INDEX[move[0]])
    initial_state = 6 if previous_face is None else FACE_INDEX[previous_face]
    return RegularFactor(
        name="CANONICAL_ADJACENCY",
        state_count=7,
        initial_mask=1 << initial_state,
        target_mask=(1 << 7) - 1,
        relation=relation,
    )


def _step_mask(mask: int, move_id: int, factor: RegularFactor) -> int:
    result = 0
    remaining = int(mask)
    while remaining:
        low = remaining & -remaining
        state = low.bit_length() - 1
        result |= int(factor.relation[state, move_id])
        remaining ^= low
    return result


def _preimage_mask(target_mask: int, move_id: int, factor: RegularFactor) -> int:
    result = 0
    for state in range(factor.state_count):
        if int(factor.relation[state, move_id]) & int(target_mask):
            result |= 1 << state
    return result


def _forward_backward(
    factor: RegularFactor,
    domains: Sequence[int],
) -> tuple[list[int], list[int], list[int]]:
    length = len(domains)
    forward = [0] * (length + 1)
    backward = [0] * (length + 1)
    forward[0] = int(factor.initial_mask)
    for column in range(length):
        reachable = 0
        domain = int(domains[column])
        for move_id in range(len(MOVE_NAMES)):
            if domain & (1 << move_id):
                reachable |= _step_mask(forward[column], move_id, factor)
        forward[column + 1] = reachable

    backward[length] = int(factor.target_mask)
    for column in range(length - 1, -1, -1):
        coreachable = 0
        domain = int(domains[column])
        for move_id in range(len(MOVE_NAMES)):
            if domain & (1 << move_id):
                coreachable |= _preimage_mask(backward[column + 1], move_id, factor)
        backward[column] = coreachable

    supports = [0] * length
    for column in range(length):
        domain = int(domains[column])
        support = 0
        for move_id in range(len(MOVE_NAMES)):
            if not domain & (1 << move_id):
                continue
            if _step_mask(forward[column], move_id, factor) & backward[column + 1]:
                support |= 1 << move_id
        supports[column] = support
    return forward, backward, supports


def propagate_regular_support(
    factors: Sequence[RegularFactor],
    length: int,
    *,
    previous_face: str | None = None,
    initial_domains: Sequence[int] | None = None,
) -> dict[str, object]:
    """Intersect all factor supports to a deterministic fixpoint."""

    started = perf_counter()
    full_domain = (1 << len(MOVE_NAMES)) - 1
    domains = (
        [full_domain] * int(length)
        if initial_domains is None
        else [int(value) for value in initial_domains]
    )
    if len(domains) != int(length):
        raise ValueError("initial_domains length mismatch")
    all_factors = (canonical_adjacency_factor(previous_face), *tuple(factors))
    before = tuple(domains)
    iterations = 0
    state_visits = 0
    factor_nonempty: dict[str, bool] = {}

    while True:
        iterations += 1
        intersected = list(domains)
        for factor in all_factors:
            forward, _backward, support = _forward_backward(factor, domains)
            factor_nonempty[factor.name] = bool(
                forward[-1] & int(factor.target_mask)
            )
            state_visits += sum(mask.bit_count() for mask in forward)
            for column in range(int(length)):
                intersected[column] &= int(support[column])
        if intersected == domains or any(value == 0 for value in intersected):
            domains = intersected
            break
        domains = intersected

    return {
        "domains_before": before,
        "domains_after": tuple(domains),
        "domain_width_before": tuple(value.bit_count() for value in before),
        "domain_width_after": tuple(value.bit_count() for value in domains),
        "domain_sum_before": sum(value.bit_count() for value in before),
        "domain_sum_after": sum(value.bit_count() for value in domains),
        "iterations": iterations,
        "factor_nonempty": factor_nonempty,
        "all_factors_nonempty": all(factor_nonempty.values()),
        "gac_nonempty": all(value != 0 for value in domains),
        "factor_state_visits": state_visits,
        "wall_seconds": perf_counter() - started,
    }


def exact_supported_domains(words: np.ndarray, accepted: np.ndarray) -> tuple[int, ...]:
    """Return per-column move masks grounded in accepted explicit word IDs."""

    rows = np.asarray(words, dtype=np.uint8)
    keep = np.asarray(accepted, dtype=np.bool_)
    if rows.ndim != 2 or keep.shape != (len(rows),):
        raise ValueError("word/accepted shape mismatch")
    if rows.shape[1] == 0:
        return ()
    selected = rows[keep]
    if not len(selected):
        return tuple(0 for _ in range(rows.shape[1]))
    masks: list[int] = []
    for column in range(rows.shape[1]):
        mask = 0
        for move_id in np.unique(selected[:, column]):
            mask |= 1 << int(move_id)
        masks.append(mask)
    return tuple(masks)


def relation_from_deterministic(next_state: np.ndarray) -> np.ndarray:
    values = np.asarray(next_state, dtype=np.int64)
    relation = np.zeros(values.shape, dtype=np.uint64)
    for state in range(values.shape[0]):
        for move_id in range(values.shape[1]):
            relation[state, move_id] = np.uint64(1 << int(values[state, move_id]))
    return relation


__all__ = [
    "FACE_INDEX",
    "FACE_ORDER",
    "MOVE_INDEX",
    "MOVE_NAMES",
    "RegularFactor",
    "canonical_adjacency_factor",
    "canonical_allow",
    "canonical_counts",
    "canonical_word_arrays",
    "exact_supported_domains",
    "is_canonical_word",
    "iter_canonical_words",
    "propagate_regular_support",
    "relation_from_deterministic",
]
