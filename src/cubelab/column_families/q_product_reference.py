"""Independent exact-Q fixed-bound viability tables and product adapter."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from cubelab.ato.column_grounding import Q_NEXT
from cubelab.ato.regular_support import MOVE_NAMES, canonical_allow

from .residual_factor_view import CONTEXTS, CONTEXT_INDEX


@dataclass(slots=True)
class ExactQViability:
    maximum_horizon: int
    viable: np.ndarray
    path_counts: np.ndarray
    outgoing_mask: np.ndarray
    next_q: np.ndarray
    table_hash: str

    @classmethod
    def build(cls, *, maximum_horizon: int = 8) -> "ExactQViability":
        maximum_horizon = int(maximum_horizon)
        viable = np.zeros(
            (20, 24, maximum_horizon + 1, len(CONTEXTS)), dtype=np.bool_
        )
        counts = np.zeros(
            (20, 24, maximum_horizon + 1, len(CONTEXTS)), dtype=np.int64
        )
        masks = np.zeros_like(counts, dtype=np.uint32)
        viable[:, 0, 0, :] = True
        counts[:, 0, 0, :] = 1
        for remaining in range(1, maximum_horizon + 1):
            for piece in range(20):
                for q in range(24):
                    for context_id, previous in enumerate(CONTEXTS):
                        total = 0
                        mask = 0
                        for move_id, move in enumerate(MOVE_NAMES):
                            if not canonical_allow(previous, move):
                                continue
                            child_q = int(Q_NEXT[piece, q, move_id])
                            child_count = int(
                                counts[
                                    piece,
                                    child_q,
                                    remaining - 1,
                                    CONTEXT_INDEX[move[0]],
                                ]
                            )
                            if child_count:
                                mask |= 1 << move_id
                                total += child_count
                        counts[piece, q, remaining, context_id] = total
                        masks[piece, q, remaining, context_id] = mask
                        viable[piece, q, remaining, context_id] = total > 0
        payload = b"|".join((viable.tobytes(), counts.tobytes(), masks.tobytes()))
        for array in (viable, counts, masks):
            array.flags.writeable = False
        return cls(
            maximum_horizon=maximum_horizon,
            viable=viable,
            path_counts=counts,
            outgoing_mask=masks,
            next_q=Q_NEXT,
            table_hash=sha256(payload).hexdigest(),
        )

    def common_mask(
        self,
        q_states: np.ndarray,
        remaining: int,
        context_ids: np.ndarray,
        pieces: tuple[int, ...] = tuple(range(20)),
    ) -> np.ndarray:
        states = np.asarray(q_states, dtype=np.uint8)
        contexts = np.asarray(context_ids, dtype=np.uint8)
        result = np.full(len(states), (1 << len(MOVE_NAMES)) - 1, dtype=np.uint32)
        for local_column, piece in enumerate(pieces):
            result &= self.outgoing_mask[
                piece, states[:, local_column], int(remaining), contexts
            ]
        return result

    def child_states(
        self,
        q_states: np.ndarray,
        move_id: int,
        pieces: tuple[int, ...] = tuple(range(20)),
    ) -> np.ndarray:
        states = np.asarray(q_states, dtype=np.uint8)
        result = np.empty_like(states)
        for local_column, piece in enumerate(pieces):
            result[:, local_column] = self.next_q[
                piece, states[:, local_column], int(move_id)
            ]
        return result

    def cache_arrays(self) -> dict[str, np.ndarray]:
        return {
            "q_viable": self.viable,
            "q_path_counts": self.path_counts,
            "q_outgoing_mask": self.outgoing_mask,
            "q_next": self.next_q,
        }


__all__ = ["ExactQViability"]
