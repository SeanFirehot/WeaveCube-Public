#!/usr/bin/env python3
"""
CubeLab v37.458-A1
EXACT BIDIRECTIONAL ATO KERNEL

Semantics
---------
Q_NEXT[piece, q, move] is the authoritative production-ordered exact ATO
transition imported from cubelab.ato.column_grounding.

This module constructs the exact inverse transition axis Q_PREV such that:

    Q_PREV[p, Q_NEXT[p, q, m], m] == q
    Q_NEXT[p, Q_PREV[p, q, m], m] == q

for every piece p, pose q, and move m.

`backward_q(q_after, m)` means:
    return the exact predecessor state q_before satisfying
        forward_q(q_before, m) == q_after

It is therefore equivalent to applying the inverse move token, but keeping the
same column label m is useful for future bidirectional GLOBAL_COLUMN support.

No search, pruning, terminal phase, or solution-length semantics live here.
"""

from __future__ import annotations

import sys
from typing import Iterable, Sequence

import numpy as np


# Historical CubeLab imports may require Unix `resource` even on Windows.
if sys.platform.startswith("win"):
    try:
        import resource as _resource  # pragma: no cover
    except ModuleNotFoundError:
        import types as _types

        _resource = _types.ModuleType("resource")
        _resource.RUSAGE_SELF = 0
        _resource.RUSAGE_CHILDREN = -1
        _resource.RLIM_INFINITY = -1

        for _name, _value in {
            "RLIMIT_CPU": 0,
            "RLIMIT_FSIZE": 1,
            "RLIMIT_DATA": 2,
            "RLIMIT_STACK": 3,
            "RLIMIT_CORE": 4,
            "RLIMIT_RSS": 5,
            "RLIMIT_NPROC": 6,
            "RLIMIT_NOFILE": 7,
            "RLIMIT_MEMLOCK": 8,
            "RLIMIT_AS": 9,
        }.items():
            setattr(_resource, _name, _value)

        class _RUsage:
            ru_utime = 0.0
            ru_stime = 0.0
            ru_maxrss = 0.0

        _resource.getrusage = lambda _who: _RUsage()
        _resource.getrlimit = lambda _which: (-1, -1)
        _resource.setrlimit = lambda _which, _limits: None
        sys.modules["resource"] = _resource


from cubelab.ato import column_grounding as _cg
from cubelab.ato import regular_support as _rs


MOVE_NAMES: tuple[str, ...] = tuple(str(x) for x in _rs.MOVE_NAMES)
Q_NEXT: np.ndarray = np.asarray(_cg.Q_NEXT, dtype=np.uint8)

if Q_NEXT.shape != (20, 24, 18):
    raise RuntimeError(
        "V37_458_A1_Q_NEXT_SHAPE_DRIFT "
        f"observed={Q_NEXT.shape} expected=(20,24,18)"
    )

if len(MOVE_NAMES) != 18 or len(set(MOVE_NAMES)) != 18:
    raise RuntimeError(
        "V37_458_A1_MOVE_NAMES_INVALID "
        f"names={MOVE_NAMES}"
    )

MOVE_INDEX = {name: i for i, name in enumerate(MOVE_NAMES)}
PIECES = np.arange(20, dtype=np.intp)
POSES = np.arange(24, dtype=np.uint8)


def _inverse_token(token: str) -> str:
    if token.endswith("2"):
        return token
    if token.endswith("'"):
        return token[:-1]
    return token + "'"


INVERSE_MOVE_ID: tuple[int, ...] = tuple(
    MOVE_INDEX[_inverse_token(token)]
    for token in MOVE_NAMES
)


def _build_q_prev() -> np.ndarray:
    prev = np.empty_like(Q_NEXT)

    expected = np.arange(24, dtype=np.uint8)

    for piece in range(20):
        for move_id in range(18):
            row = np.asarray(
                Q_NEXT[piece, :, move_id],
                dtype=np.uint8,
            )

            if not np.array_equal(np.sort(row), expected):
                raise RuntimeError(
                    "V37_458_A1_NON_BIJECTIVE_PIECE_TRANSITION "
                    f"piece={piece} move_id={move_id} "
                    f"move={MOVE_NAMES[move_id]} row={row.tolist()}"
                )

            inv = np.empty(24, dtype=np.uint8)
            inv[row] = expected
            prev[piece, :, move_id] = inv

    return prev


Q_PREV: np.ndarray = _build_q_prev()


def move_id(move: int | str) -> int:
    if isinstance(move, str):
        try:
            return int(MOVE_INDEX[move])
        except KeyError as exc:
            raise ValueError(f"unknown move token: {move!r}") from exc

    value = int(move)
    if not 0 <= value < 18:
        raise ValueError(f"move id out of range: {value}")
    return value


def normalize_q(q: Iterable[int]) -> np.ndarray:
    arr = np.asarray(tuple(int(x) for x in q), dtype=np.uint8)
    if arr.shape != (20,):
        raise ValueError(f"exact Q must have shape (20,), got {arr.shape}")
    if np.any(arr >= 24):
        raise ValueError("exact Q contains pose outside [0,23]")
    return arr


def forward_q(q: Iterable[int], move: int | str) -> np.ndarray:
    state = normalize_q(q)
    mid = move_id(move)
    return np.asarray(
        Q_NEXT[PIECES, state.astype(np.intp), mid],
        dtype=np.uint8,
    )


def backward_q(q_after: Iterable[int], move: int | str) -> np.ndarray:
    state = normalize_q(q_after)
    mid = move_id(move)
    return np.asarray(
        Q_PREV[PIECES, state.astype(np.intp), mid],
        dtype=np.uint8,
    )


def inverse_forward_q(q: Iterable[int], move: int | str) -> np.ndarray:
    """Apply the inverse token through the authoritative forward table."""
    mid = move_id(move)
    return forward_q(q, INVERSE_MOVE_ID[mid])


def forward_all(q: Iterable[int]) -> np.ndarray:
    """
    Return all 18 exact children.

    Shape: (18, 20), row m = forward_q(q, m).
    """
    state = normalize_q(q)
    out = np.empty((18, 20), dtype=np.uint8)

    for mid in range(18):
        out[mid] = Q_NEXT[
            PIECES,
            state.astype(np.intp),
            mid,
        ]

    return out


def backward_all(q_after: Iterable[int]) -> np.ndarray:
    """
    Return all 18 exact predecessors.

    Shape: (18, 20), row m = backward_q(q_after, m).
    """
    state = normalize_q(q_after)
    out = np.empty((18, 20), dtype=np.uint8)

    for mid in range(18):
        out[mid] = Q_PREV[
            PIECES,
            state.astype(np.intp),
            mid,
        ]

    return out


def apply_word_forward(
    q: Iterable[int],
    word: Sequence[int | str],
) -> np.ndarray:
    state = normalize_q(q)
    for move in word:
        state = forward_q(state, move)
    return state


def rewind_word(
    q_after: Iterable[int],
    word: Sequence[int | str],
) -> np.ndarray:
    """
    Rewind a forward word exactly.

    If:
        q_after = apply_word_forward(q_before, word)

    then:
        rewind_word(q_after, word) == q_before
    """
    state = normalize_q(q_after)
    for move in reversed(tuple(word)):
        state = backward_q(state, move)
    return state


def exhaustive_transition_parity() -> dict[str, int | bool]:
    """
    Exhaustive 20*18*24 table-level proof of both inverse identities and
    inverse-token equivalence.
    """
    forward_then_back_mismatch = 0
    back_then_forward_mismatch = 0
    inverse_token_mismatch = 0

    for piece in range(20):
        for mid in range(18):
            inv_mid = INVERSE_MOVE_ID[mid]

            for q in range(24):
                q2 = int(Q_NEXT[piece, q, mid])
                q0 = int(Q_PREV[piece, q2, mid])

                if q0 != q:
                    forward_then_back_mismatch += 1

                q1 = int(Q_PREV[piece, q, mid])
                q2_again = int(Q_NEXT[piece, q1, mid])

                if q2_again != q:
                    back_then_forward_mismatch += 1

                via_inverse_token = int(
                    Q_NEXT[piece, q, inv_mid]
                )

                if q1 != via_inverse_token:
                    inverse_token_mismatch += 1

    return {
        "piece_move_pose_rows": int(20 * 18 * 24),
        "forward_then_backward_mismatch": int(
            forward_then_back_mismatch
        ),
        "backward_then_forward_mismatch": int(
            back_then_forward_mismatch
        ),
        "backward_vs_inverse_token_mismatch": int(
            inverse_token_mismatch
        ),
        "pass": bool(
            forward_then_back_mismatch == 0
            and back_then_forward_mismatch == 0
            and inverse_token_mismatch == 0
        ),
    }
