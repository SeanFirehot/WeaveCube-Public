"""Q-conditioned transition tables and canonical short-word compiler."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from cubelab.pdcc.model import PIECE_INDEX, PIECE_NAMES
from cubelab.pdcc.tables import local_transition

from .pose24 import MOVE_INDEX, MOVE_NAMES, Q_TO_OMEGA


NEXT_Q = np.empty((len(PIECE_NAMES), 24, len(MOVE_NAMES)), dtype=np.uint8)
ACTIVE = np.empty_like(NEXT_Q, dtype=np.bool_)
NEXT_OMEGA = np.empty_like(NEXT_Q, dtype=np.uint8)

for piece_index, piece in enumerate(PIECE_NAMES):
    for q in range(24):
        for move_index, move in enumerate(MOVE_NAMES):
            entry = local_transition(piece, q, move)
            NEXT_Q[piece_index, q, move_index] = entry.pose_after
            ACTIVE[piece_index, q, move_index] = entry.active
            NEXT_OMEGA[piece_index, q, move_index] = Q_TO_OMEGA[entry.pose_after]


OMEGA_RELATION_MASK = np.zeros(
    (len(PIECE_NAMES), 6, len(MOVE_NAMES)), dtype=np.uint8
)
for piece_index in range(len(PIECE_NAMES)):
    for q in range(24):
        omega = Q_TO_OMEGA[q]
        for move_index in range(len(MOVE_NAMES)):
            target = int(NEXT_OMEGA[piece_index, q, move_index])
            OMEGA_RELATION_MASK[piece_index, omega, move_index] |= np.uint8(1 << target)


def apply_q_vector(q_vector: Iterable[int], move: str | int) -> tuple[int, ...]:
    poses = np.asarray(tuple(q_vector), dtype=np.uint8)
    if poses.shape != (len(PIECE_NAMES),):
        raise ValueError(f"Expected Q vector shape ({len(PIECE_NAMES)},)")
    move_index = MOVE_INDEX[move] if isinstance(move, str) else int(move)
    result = NEXT_Q[np.arange(len(PIECE_NAMES)), poses, move_index]
    return tuple(int(value) for value in result)


def apply_q_batch(q_batch: np.ndarray, move: str | int) -> np.ndarray:
    values = np.asarray(q_batch, dtype=np.uint8)
    if values.ndim == 1:
        return np.asarray(apply_q_vector(values, move), dtype=np.uint8)
    if values.shape[-1] != len(PIECE_NAMES):
        raise ValueError(f"Last batch dimension must be {len(PIECE_NAMES)}")
    move_index = MOVE_INDEX[move] if isinstance(move, str) else int(move)
    flattened = values.reshape((-1, len(PIECE_NAMES)))
    updated = NEXT_Q[
        np.arange(len(PIECE_NAMES))[None, :],
        flattened,
        move_index,
    ]
    return updated.reshape(values.shape)


def iter_canonical_words(maximum_depth: int = 3) -> Iterable[tuple[str, ...]]:
    """Yield exact-depth 1..N words with no adjacent same-face turns."""

    def visit(prefix: tuple[str, ...], depth: int) -> Iterable[tuple[str, ...]]:
        if len(prefix) == depth:
            yield prefix
            return
        previous_face = prefix[-1][0] if prefix else None
        for move in MOVE_NAMES:
            if move[0] == previous_face:
                continue
            yield from visit(prefix + (move,), depth)

    for depth in range(1, maximum_depth + 1):
        yield from visit((), depth)


def compile_word_support(maximum_depth: int = 3) -> dict[str, np.ndarray]:
    words = tuple(iter_canonical_words(maximum_depth))
    count = len(words)
    padded = np.full((count, maximum_depth), 255, dtype=np.uint8)
    lengths = np.empty(count, dtype=np.uint8)
    first_face = np.empty(count, dtype=np.uint8)
    last_face = np.empty(count, dtype=np.uint8)
    face_order = {face: index for index, face in enumerate("UDLRFB")}
    previous_face_legal = np.ones((count, 7), dtype=np.bool_)
    q_transition = np.empty((count, len(PIECE_NAMES), 24), dtype=np.uint8)
    affected_any = np.empty_like(q_transition, dtype=np.bool_)
    omega_relation = np.zeros(
        (count, len(PIECE_NAMES), 6), dtype=np.uint8
    )
    base = np.broadcast_to(
        np.arange(24, dtype=np.uint8), (len(PIECE_NAMES), 24)
    ).copy()
    piece_index = np.arange(len(PIECE_NAMES))[:, None]

    for word_id, word in enumerate(words):
        move_ids = [MOVE_INDEX[move] for move in word]
        padded[word_id, : len(move_ids)] = move_ids
        lengths[word_id] = len(word)
        first_face[word_id] = face_order[word[0][0]]
        last_face[word_id] = face_order[word[-1][0]]
        previous_face_legal[word_id, 1 + face_order[word[0][0]]] = False
        current = base.copy()
        touched = np.zeros_like(base, dtype=np.bool_)
        for move_id in move_ids:
            before = current
            touched |= ACTIVE[piece_index, before, move_id]
            current = NEXT_Q[piece_index, before, move_id]
        q_transition[word_id] = current
        affected_any[word_id] = touched
        for piece in range(len(PIECE_NAMES)):
            for q in range(24):
                start_omega = Q_TO_OMEGA[q]
                end_omega = Q_TO_OMEGA[int(current[piece, q])]
                omega_relation[word_id, piece, start_omega] |= np.uint8(
                    1 << end_omega
                )

    return {
        "word_move_ids": padded,
        "word_lengths": lengths,
        "word_first_face": first_face,
        "word_last_face": last_face,
        "word_previous_face_legal": previous_face_legal,
        "word_q_transition": q_transition,
        "word_affected_any": affected_any,
        "word_omega_relation": omega_relation,
    }


__all__ = [
    "ACTIVE",
    "NEXT_OMEGA",
    "NEXT_Q",
    "OMEGA_RELATION_MASK",
    "apply_q_batch",
    "apply_q_vector",
    "compile_word_support",
    "iter_canonical_words",
]
