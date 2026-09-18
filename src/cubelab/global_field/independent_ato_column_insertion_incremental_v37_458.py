#!/usr/bin/env python3
"""
CubeLab v37.458-A14-A2
INCREMENTAL BOUNDARY-REUSE INDEPENDENT ATO

Semantics are IDENTICAL to A14-A1.

A1 already vectorized all 20 independent piece automata.  Its remaining hot
cost is rebuilding the complete forward/backward DP after every move-domain
restriction.

A11 already passes the parent's propagator `final_state` into each child
move-CSP node.  A2 uses that exact parent->child restriction relation.

Restriction recurrence
----------------------
Suppose old domain masks D become stricter masks D' of the SAME shape.

Let:
    first = first changed column
    last  = last changed column

Then exactly:
- forward boundaries 0..first are unchanged,
- backward boundaries last+1..n are unchanged.

A2 copies those proven-identical boundaries and recomputes only:
- forward steps first..n-1,
- backward steps last..0.

This is exact because no changed domain occurs before `first` or after `last`.

Within a fixpoint call, the same rule is reused from one restriction round to
the next.  Across search nodes, the A11-supplied parent state is reused when
the child's masks are a restriction of the parent.

No heuristic, no pair coupling, no P5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cubelab.ato.regular_support import canonical_allow
from cubelab.global_field import ato_bidirectional_v37_458 as bi
from cubelab.global_field.column_variable_v37_458 import ElasticColumnField
from cubelab.global_field.independent_ato_column_insertion_v37_458 import (
    ATOPropagationOutcome,
    ATOPropagationReceipt,
    PieceSupport,
    ResidentPerPieceAutomata,
)


CONTEXTS = (None, "U", "R", "F", "D", "L", "B")
CONTEXT_INDEX = {
    value: index
    for index, value in enumerate(CONTEXTS)
}

MOVE_COUNT = 18
PIECE_COUNT = 20
POSE_COUNT = 24

PIECES = np.arange(
    PIECE_COUNT,
    dtype=np.intp,
)
ALL_POSES = np.arange(
    POSE_COUNT,
    dtype=np.intp,
)

MOVE_CONTEXT = np.asarray(
    [
        CONTEXT_INDEX[move[0]]
        for move in bi.MOVE_NAMES
    ],
    dtype=np.uint8,
)

CANONICAL_ALLOW = np.zeros(
    (7, MOVE_COUNT),
    dtype=np.bool_,
)

for ctx, previous_face in enumerate(CONTEXTS):
    for mid, move in enumerate(bi.MOVE_NAMES):
        CANONICAL_ALLOW[ctx, mid] = bool(
            canonical_allow(
                previous_face,
                move,
            )
        )

ALLOWED_CONTEXTS = tuple(
    tuple(
        int(x)
        for x in np.flatnonzero(
            CANONICAL_ALLOW[:, mid]
        )
    )
    for mid in range(MOVE_COUNT)
)


@dataclass(frozen=True)
class IncrementalATOState:
    left_anchor_q: np.ndarray
    right_anchor_q: np.ndarray
    previous_face: str | None
    next_move: str | None

    domain_masks: tuple[int, ...]
    materialized_columns: int

    forward: np.ndarray   # [n+1,20,7,24] bool
    backward: np.ndarray  # [n+1,20,7,24] bool

    piece_path_exists: np.ndarray   # [20] bool
    piece_support_masks: np.ndarray # [20,n] uint32

    forward_state_counts: np.ndarray   # [20,n+1] uint16
    backward_state_counts: np.ndarray  # [20,n+1] uint16

    def same_anchor_context(
        self,
        field: ElasticColumnField,
        previous_face: str | None,
        next_move: str | None,
    ) -> bool:
        return bool(
            self.previous_face == previous_face
            and self.next_move == next_move
            and np.array_equal(
                self.left_anchor_q,
                np.asarray(
                    field.left_anchor_q,
                    dtype=np.uint8,
                ),
            )
            and np.array_equal(
                self.right_anchor_q,
                np.asarray(
                    field.right_anchor_q,
                    dtype=np.uint8,
                ),
            )
        )


def _domain_move_ids(
    mask: int,
) -> tuple[int, ...]:
    value = int(mask)
    return tuple(
        mid
        for mid in range(MOVE_COUNT)
        if value & (1 << mid)
    )


def _is_restriction(
    old_masks: tuple[int, ...],
    new_masks: tuple[int, ...],
) -> bool:
    if len(old_masks) != len(new_masks):
        return False

    return all(
        not (
            int(new)
            & ~int(old)
        )
        for old, new in zip(
            old_masks,
            new_masks,
        )
    )


def _transition_surface(
    automata: ResidentPerPieceAutomata,
    q_target: np.ndarray,
) -> np.ndarray:
    """
    [20,24,18] exact resident transition for the fixed targets.
    """
    return automata.next_q[
        PIECES,
        q_target.astype(
            np.intp,
            copy=False,
        ),
    ].astype(
        np.intp,
        copy=False,
    )


def _forward_step(
    *,
    forward: np.ndarray,
    position: int,
    domain_mask: int,
    transition: np.ndarray,
) -> None:
    """
    Recompute forward boundary position+1 exactly from boundary position.
    Caller must clear destination first.
    """
    mids = _domain_move_ids(
        domain_mask
    )

    for mid in mids:
        child_ctx = int(
            MOVE_CONTEXT[mid]
        )
        child_map = transition[
            :,
            :,
            mid,
        ]  # [20,24]

        p_idx = np.broadcast_to(
            PIECES[:, None],
            child_map.shape,
        )

        for ctx in ALLOWED_CONTEXTS[mid]:
            source = forward[
                position,
                :,
                int(ctx),
                :,
            ]

            if not np.any(
                source
            ):
                continue

            dest = forward[
                position + 1,
                :,
                child_ctx,
                :,
            ]

            np.logical_or.at(
                dest,
                (
                    p_idx.reshape(-1),
                    child_map.reshape(-1),
                ),
                source.reshape(-1),
            )


def _backward_step(
    *,
    backward: np.ndarray,
    position: int,
    domain_mask: int,
    transition: np.ndarray,
) -> None:
    """
    Recompute backward boundary position exactly from position+1.
    Caller must clear destination first.
    """
    mids = _domain_move_ids(
        domain_mask
    )

    for mid in mids:
        child_ctx = int(
            MOVE_CONTEXT[mid]
        )

        child_accept = backward[
            position + 1,
            :,
            child_ctx,
            :,
        ]

        if not np.any(
            child_accept
        ):
            continue

        child_map = transition[
            :,
            :,
            mid,
        ]

        mapped_good = np.take_along_axis(
            child_accept,
            child_map,
            axis=1,
        )

        for ctx in ALLOWED_CONTEXTS[mid]:
            backward[
                position,
                :,
                int(ctx),
                :,
            ] |= mapped_good


def _extract_support(
    *,
    forward: np.ndarray,
    backward: np.ndarray,
    domain_masks: tuple[int, ...],
    transition: np.ndarray,
    q_current: np.ndarray,
    previous_face: str | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    n = len(
        domain_masks
    )

    start_ctx = int(
        CONTEXT_INDEX[
            previous_face
        ]
    )

    path_exists = backward[
        0,
        PIECES,
        start_ctx,
        q_current.astype(
            np.intp,
            copy=False,
        ),
    ].copy()

    support = np.zeros(
        (
            PIECE_COUNT,
            n,
        ),
        dtype=np.uint32,
    )

    for position in range(
        n
    ):
        mids = _domain_move_ids(
            domain_masks[
                position
            ]
        )

        for mid in mids:
            child_ctx = int(
                MOVE_CONTEXT[mid]
            )

            child_accept = backward[
                position + 1,
                :,
                child_ctx,
                :,
            ]

            if not np.any(
                child_accept
            ):
                continue

            child_map = transition[
                :,
                :,
                mid,
            ]

            mapped_good = np.take_along_axis(
                child_accept,
                child_map,
                axis=1,
            )

            supported_piece = np.zeros(
                PIECE_COUNT,
                dtype=np.bool_,
            )

            for ctx in ALLOWED_CONTEXTS[mid]:
                supported_piece |= np.any(
                    forward[
                        position,
                        :,
                        int(ctx),
                        :,
                    ]
                    & mapped_good,
                    axis=1,
                )

            if np.any(
                supported_piece
            ):
                support[
                    supported_piece,
                    position,
                ] |= np.uint32(
                    1 << mid
                )

    return (
        path_exists.astype(
            np.bool_,
            copy=False,
        ),
        support,
    )


def _finish_state(
    *,
    field: ElasticColumnField,
    previous_face: str | None,
    next_move: str | None,
    domain_masks: tuple[int, ...],
    forward: np.ndarray,
    backward: np.ndarray,
    transition: np.ndarray,
    q_current: np.ndarray,
) -> IncrementalATOState:
    path_exists, support = _extract_support(
        forward=forward,
        backward=backward,
        domain_masks=domain_masks,
        transition=transition,
        q_current=q_current,
        previous_face=previous_face,
    )

    forward_counts = np.count_nonzero(
        forward,
        axis=(2, 3),
    ).T.astype(
        np.uint16,
        copy=False,
    )

    backward_counts = np.count_nonzero(
        backward,
        axis=(2, 3),
    ).T.astype(
        np.uint16,
        copy=False,
    )

    return IncrementalATOState(
        left_anchor_q=np.asarray(
            field.left_anchor_q,
            dtype=np.uint8,
        ).copy(),
        right_anchor_q=np.asarray(
            field.right_anchor_q,
            dtype=np.uint8,
        ).copy(),
        previous_face=previous_face,
        next_move=next_move,
        domain_masks=tuple(
            int(x)
            for x in domain_masks
        ),
        materialized_columns=int(
            field.materialized_length
        ),
        forward=forward,
        backward=backward,
        piece_path_exists=path_exists,
        piece_support_masks=support,
        forward_state_counts=forward_counts,
        backward_state_counts=backward_counts,
    )


def build_state(
    *,
    automata: ResidentPerPieceAutomata,
    field: ElasticColumnField,
    domain_masks: tuple[int, ...],
    previous_face: str | None,
    next_move: str | None,
) -> IncrementalATOState:
    n = len(
        domain_masks
    )

    q_current = bi.normalize_q(
        field.left_anchor_q
    )
    q_target = bi.normalize_q(
        field.right_anchor_q
    )

    transition = _transition_surface(
        automata,
        q_target,
    )

    forward = np.zeros(
        (
            n + 1,
            PIECE_COUNT,
            7,
            POSE_COUNT,
        ),
        dtype=np.bool_,
    )

    start_ctx = int(
        CONTEXT_INDEX[
            previous_face
        ]
    )

    forward[
        0,
        PIECES,
        start_ctx,
        q_current.astype(
            np.intp,
            copy=False,
        ),
    ] = True

    for position in range(
        n
    ):
        _forward_step(
            forward=forward,
            position=position,
            domain_mask=domain_masks[
                position
            ],
            transition=transition,
        )

    backward = np.zeros_like(
        forward
    )

    for ctx, end_previous_face in enumerate(
        CONTEXTS
    ):
        if (
            next_move is None
            or canonical_allow(
                end_previous_face,
                next_move,
            )
        ):
            backward[
                n,
                PIECES,
                int(ctx),
                q_target.astype(
                    np.intp,
                    copy=False,
                ),
            ] = True

    for position in range(
        n - 1,
        -1,
        -1,
    ):
        _backward_step(
            backward=backward,
            position=position,
            domain_mask=domain_masks[
                position
            ],
            transition=transition,
        )

    return _finish_state(
        field=field,
        previous_face=previous_face,
        next_move=next_move,
        domain_masks=domain_masks,
        forward=forward,
        backward=backward,
        transition=transition,
        q_current=q_current,
    )


def update_restrict_only(
    *,
    automata: ResidentPerPieceAutomata,
    parent: IncrementalATOState,
    field: ElasticColumnField,
    domain_masks: tuple[int, ...],
    previous_face: str | None,
    next_move: str | None,
) -> tuple[
    IncrementalATOState,
    int,
    int,
    int,
    int,
]:
    """
    Returns:
      state,
      reused_forward_boundaries,
      reused_backward_boundaries,
      recomputed_forward_steps,
      recomputed_backward_steps
    """
    if not parent.same_anchor_context(
        field,
        previous_face,
        next_move,
    ):
        raise ValueError(
            "A14_A2_PARENT_CONTEXT_MISMATCH"
        )

    if not _is_restriction(
        parent.domain_masks,
        domain_masks,
    ):
        raise ValueError(
            "A14_A2_UPDATE_REQUIRES_RESTRICTION"
        )

    n = len(
        domain_masks
    )

    if parent.materialized_columns != n:
        raise ValueError(
            "A14_A2_UPDATE_LENGTH_MISMATCH"
        )

    changed = [
        position
        for position, (
            old,
            new,
        ) in enumerate(
            zip(
                parent.domain_masks,
                domain_masks,
            )
        )
        if int(old)
        != int(new)
    ]

    if not changed:
        return (
            parent,
            n + 1,
            n + 1,
            0,
            0,
        )

    first = int(
        changed[0]
    )
    last = int(
        changed[-1]
    )

    q_current = bi.normalize_q(
        field.left_anchor_q
    )
    q_target = bi.normalize_q(
        field.right_anchor_q
    )

    transition = _transition_surface(
        automata,
        q_target,
    )

    # Copy is cheap at current small boundary tensors and guarantees immutable
    # parent semantics for sibling branches / rollback.
    forward = parent.forward.copy()
    backward = parent.backward.copy()

    # Boundaries 0..first are proven unchanged.
    for position in range(
        first,
        n,
    ):
        forward[
            position + 1
        ].fill(
            False
        )

        _forward_step(
            forward=forward,
            position=position,
            domain_mask=domain_masks[
                position
            ],
            transition=transition,
        )

    # Boundaries last+1..n are proven unchanged.
    for position in range(
        last,
        -1,
        -1,
    ):
        backward[
            position
        ].fill(
            False
        )

        _backward_step(
            backward=backward,
            position=position,
            domain_mask=domain_masks[
                position
            ],
            transition=transition,
        )

    state = _finish_state(
        field=field,
        previous_face=previous_face,
        next_move=next_move,
        domain_masks=domain_masks,
        forward=forward,
        backward=backward,
        transition=transition,
        q_current=q_current,
    )

    return (
        state,
        first + 1,
        n - last,
        n - first,
        last + 1,
    )


class IncrementalBoundaryReusePerPieceATOPropagator:
    def __init__(
        self,
        *,
        automata: ResidentPerPieceAutomata,
    ) -> None:
        self.automata = automata
        self.partition = tuple(
            (piece,)
            for piece in range(
                PIECE_COUNT
            )
        )
        self.move_names = tuple(
            bi.MOVE_NAMES
        )

        self.calls = 0
        self.fixpoint_rounds = 0
        self.contradiction_calls = 0
        self.changed_move_bits = 0
        self.wall_s = 0.0

        self.scratch_builds = 0
        self.incremental_updates = 0
        self.exact_state_reuses = 0

        self.reused_forward_boundaries = 0
        self.reused_backward_boundaries = 0
        self.recomputed_forward_steps = 0
        self.recomputed_backward_steps = 0

    @staticmethod
    def _masks(
        field: ElasticColumnField,
    ) -> tuple[int, ...]:
        return tuple(
            int(
                column.domain_mask
            )
            for column in field.columns
        )

    def _state_for_masks(
        self,
        field: ElasticColumnField,
        *,
        masks: tuple[int, ...],
        previous_face: str | None,
        next_move: str | None,
        seed_state,
    ):
        n = int(
            field.materialized_length
        )

        if (
            isinstance(
                seed_state,
                IncrementalATOState,
            )
            and seed_state.same_anchor_context(
                field,
                previous_face,
                next_move,
            )
            and _is_restriction(
                seed_state.domain_masks,
                masks,
            )
        ):
            if (
                seed_state.domain_masks
                == masks
            ):
                self.exact_state_reuses += 1
                self.reused_forward_boundaries += (
                    n + 1
                )
                self.reused_backward_boundaries += (
                    n + 1
                )

                return (
                    seed_state,
                    0,
                    0,
                    n + 1,
                    n + 1,
                    0,
                    0,
                )

            (
                state,
                rf,
                rb,
                cf,
                cb,
            ) = update_restrict_only(
                automata=self.automata,
                parent=seed_state,
                field=field,
                domain_masks=masks,
                previous_face=previous_face,
                next_move=next_move,
            )

            self.incremental_updates += 1
            self.reused_forward_boundaries += int(
                rf
            )
            self.reused_backward_boundaries += int(
                rb
            )
            self.recomputed_forward_steps += int(
                cf
            )
            self.recomputed_backward_steps += int(
                cb
            )

            return (
                state,
                0,
                1,
                int(rf),
                int(rb),
                int(cf),
                int(cb),
            )

        state = build_state(
            automata=self.automata,
            field=field,
            domain_masks=masks,
            previous_face=previous_face,
            next_move=next_move,
        )

        self.scratch_builds += 1
        self.recomputed_forward_steps += n
        self.recomputed_backward_steps += n

        return (
            state,
            1,
            0,
            0,
            0,
            n,
            n,
        )

    def propagate_cached(
        self,
        field: ElasticColumnField,
        *,
        previous_face: str | None,
        next_move: str | None,
        seed_state=None,
        max_rounds: int = 32,
    ) -> ATOPropagationOutcome:
        import time

        t0 = time.perf_counter()
        self.calls += 1

        n = int(
            field.materialized_length
        )

        if n < 1:
            raise ValueError(
                "incremental per-piece ATO requires >=1 column"
            )

        before = self._masks(
            field
        )

        contradiction = False
        converged = False
        rounds_done = 0
        contradiction_pieces = tuple()
        empty_columns = tuple()
        last_piece_rows = tuple()

        state = seed_state

        call_scratch = 0
        call_incremental = 0
        call_rf = 0
        call_rb = 0
        call_cf = 0
        call_cb = 0

        for round_index in range(
            int(max_rounds)
        ):
            rounds_done = (
                round_index + 1
            )
            self.fixpoint_rounds += 1

            current = self._masks(
                field
            )

            (
                state,
                scratch,
                incremental,
                rf,
                rb,
                cf,
                cb,
            ) = self._state_for_masks(
                field,
                masks=current,
                previous_face=previous_face,
                next_move=next_move,
                seed_state=state,
            )

            call_scratch += int(
                scratch
            )
            call_incremental += int(
                incremental
            )
            call_rf += int(
                rf
            )
            call_rb += int(
                rb
            )
            call_cf += int(
                cf
            )
            call_cb += int(
                cb
            )

            contradiction_pieces = tuple(
                int(piece)
                for piece in np.flatnonzero(
                    ~state.piece_path_exists
                )
            )

            last_piece_rows = tuple(
                PieceSupport(
                    piece=int(piece),
                    path_exists=bool(
                        state.piece_path_exists[
                            piece
                        ]
                    ),
                    support_masks=tuple(
                        int(x)
                        for x in state.piece_support_masks[
                            piece
                        ]
                    ),
                    forward_state_counts=tuple(
                        int(x)
                        for x in state.forward_state_counts[
                            piece
                        ]
                    ),
                    backward_state_counts=tuple(
                        int(x)
                        for x in state.backward_state_counts[
                            piece
                        ]
                    ),
                )
                for piece in range(
                    PIECE_COUNT
                )
            )

            if contradiction_pieces:
                contradiction = True

                for column in field.columns:
                    column.restrict(
                        0
                    )

                empty_columns = tuple(
                    range(
                        n
                    )
                )
                state = None
                break

            aggregate = np.bitwise_and.reduce(
                state.piece_support_masks,
                axis=0,
            ).astype(
                np.uint32,
                copy=False,
            )

            aggregate &= np.asarray(
                current,
                dtype=np.uint32,
            )

            empty_columns = tuple(
                int(position)
                for position in np.flatnonzero(
                    aggregate == 0
                )
            )

            for position, mask in enumerate(
                aggregate
            ):
                field.columns[
                    position
                ].restrict(
                    int(mask)
                )

            after = self._masks(
                field
            )

            if (
                empty_columns
                or any(
                    int(mask) == 0
                    for mask in after
                )
            ):
                contradiction = True
                state = None
                break

            if after == current:
                converged = True
                break

            # State for `current` is now the exact parent of stricter `after`.
            # Next loop update_restrict_only reuses it.

        final = self._masks(
            field
        )

        changed_columns = tuple(
            int(position)
            for position, (
                old,
                new,
            ) in enumerate(
                zip(
                    before,
                    final,
                )
            )
            if int(old) != int(new)
        )

        forced_columns = tuple(
            int(position)
            for position, mask
            in enumerate(
                final
            )
            if int(mask).bit_count()
            == 1
        )

        removed = sum(
            int(old).bit_count()
            - int(new).bit_count()
            for old, new in zip(
                before,
                final,
            )
        )
        self.changed_move_bits += int(
            removed
        )

        if contradiction:
            self.contradiction_calls += 1

        wall = float(
            time.perf_counter()
            - t0
        )
        self.wall_s += wall

        receipt = ATOPropagationReceipt(
            rounds=int(
                rounds_done
            ),
            contradiction=bool(
                contradiction
            ),
            converged=bool(
                converged
                or contradiction
            ),
            domain_masks_before=before,
            domain_masks_after=final,
            changed_columns=changed_columns,
            forced_columns=forced_columns,
            contradiction_pieces=contradiction_pieces,
            empty_common_columns=empty_columns,
            piece_support_last_round=last_piece_rows,
        )

        final_state = (
            None
            if contradiction
            else state
        )

        if (
            not contradiction
            and final_state is not None
            and final_state.domain_masks
            != final
        ):
            # Normally convergence means state is already for `final`.
            # Defensive exact synchronization if max_rounds or future changes
            # alter that control flow.
            (
                final_state,
                scratch,
                incremental,
                rf,
                rb,
                cf,
                cb,
            ) = self._state_for_masks(
                field,
                masks=final,
                previous_face=previous_face,
                next_move=next_move,
                seed_state=final_state,
            )

            call_scratch += int(
                scratch
            )
            call_incremental += int(
                incremental
            )
            call_rf += int(
                rf
            )
            call_rb += int(
                rb
            )
            call_cf += int(
                cf
            )
            call_cb += int(
                cb
            )

        return ATOPropagationOutcome(
            receipt=receipt,
            final_state=final_state,
            scratch_builds=int(
                call_scratch
            ),
            incremental_updates=int(
                call_incremental
            ),
            reused_forward_boundaries=int(
                call_rf
            ),
            reused_backward_boundaries=int(
                call_rb
            ),
            recomputed_forward_steps=int(
                call_cf
            ),
            recomputed_backward_steps=int(
                call_cb
            ),
        )

    def telemetry(
        self,
    ) -> dict[str, Any]:
        return {
            "calls": int(
                self.calls
            ),
            "fixpoint_rounds": int(
                self.fixpoint_rounds
            ),
            "contradiction_calls": int(
                self.contradiction_calls
            ),
            "changed_move_bits": int(
                self.changed_move_bits
            ),
            "wall_s": float(
                self.wall_s
            ),
            "scratch_builds": int(
                self.scratch_builds
            ),
            "incremental_updates": int(
                self.incremental_updates
            ),
            "exact_state_reuses": int(
                self.exact_state_reuses
            ),
            "reused_forward_boundaries": int(
                self.reused_forward_boundaries
            ),
            "reused_backward_boundaries": int(
                self.reused_backward_boundaries
            ),
            "recomputed_forward_steps": int(
                self.recomputed_forward_steps
            ),
            "recomputed_backward_steps": int(
                self.recomputed_backward_steps
            ),
            "transition_source": (
                "A14-P0 RESIDENT ROUTE RELATION TABLE"
            ),
            "twenty_piece_vectorization": True,
            "parent_child_exact_restriction_reuse": True,
            "pair_coupling": "NONE",
            "P5_inside_search": "ZERO",
        }
