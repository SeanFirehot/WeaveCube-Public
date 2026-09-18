#!/usr/bin/env python3
"""
CubeLab v37.458-A14-A0
INDEPENDENT ATO + COLUMN_INSERTION PROPAGATOR

This is Branch A.

It intentionally does NOT use:
- P5 / ten-pair propagation,
- pair coupling,
- fixed prefix,
- configured solution length,
- global H,
- materialized-column ceiling,
- next-length stage,
- left-to-right solution generation.

Resident exact source
---------------------
A14-P0 per-piece current->target route atlas.

Before search, that asset is compiled once into fixed-target 24-state
per-piece automata:
    NEXT_Q[piece, target_q, current_q, move]

The compilation itself uses only:
    route_class[piece,current_q,target_q]
    left_next_class[relation_class,move]

Runtime propagation never calls Q_NEXT to reconstruct these per-piece
transitions.

ATO propagation
---------------
For the CURRENT elastic GLOBAL_COLUMN shape:
- each of 20 pieces is solved independently through all current column domains,
- canonical previous-face / next-move boundary contexts are exact,
- exact per-column support masks are derived for each piece,
- all 20 supports are intersected,
- domains are restricted to a fixpoint.

Authority:
- no accepting word for one piece => full-cube UNSAT (exact necessary)
- unsupported move for one piece  => safe hard deletion
- survival for all pieces         => NOT full-cube SAT authority

COLUMN_INSERTION
----------------
Structural growth is supplied by the frozen A11 length-free search:
- initial columns = 0
- every legal insertion gap is generated
- insertion occurs only at decision-free shape roots
- move variables use nonchronological MRV
- no column-count ceiling
- observed solution length is output only

Full-Q exact interval/leaf checking remains final authority.

This module only implements the independent ATO propagator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cubelab.ato.regular_support import canonical_allow
from cubelab.global_field import ato_bidirectional_v37_458 as bi
from cubelab.global_field.column_variable_v37_458 import (
    ElasticColumnField,
)


CONTEXTS = (None, "U", "R", "F", "D", "L", "B")
CONTEXT_INDEX = {
    value: index
    for index, value in enumerate(CONTEXTS)
}

MOVE_COUNT = 18
PIECE_COUNT = 20
POSE_COUNT = 24
ALL_MOVES_MASK = (1 << MOVE_COUNT) - 1

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


@dataclass(frozen=True)
class ResidentPerPieceAutomata:
    """
    Fixed-target 24-state automata compiled from the P0 relation atlas.

    next_q[piece,target_q,current_q,move] is exact.
    """

    route_class: np.ndarray          # [20,576] uint16
    class_distance: np.ndarray       # [480] uint8
    next_q: np.ndarray               # [20,24,24,18] uint8

    @classmethod
    def load_and_compile(
        cls,
        route_asset: Path,
    ) -> "ResidentPerPieceAutomata":
        with np.load(
            Path(route_asset),
            allow_pickle=False,
        ) as data:
            required = {
                "route_class",
                "class_distance",
                "left_next_class",
            }

            missing = sorted(
                required - set(data.files)
            )

            if missing:
                raise RuntimeError(
                    "A14_A0_ROUTE_ASSET_KEYS_MISSING "
                    f"{missing}"
                )

            route_class = np.asarray(
                data["route_class"],
                dtype=np.uint16,
            )
            class_distance = np.asarray(
                data["class_distance"],
                dtype=np.uint8,
            )
            left_next_class = np.asarray(
                data["left_next_class"],
                dtype=np.uint16,
            )

        if route_class.shape != (
            PIECE_COUNT,
            POSE_COUNT * POSE_COUNT,
        ):
            raise RuntimeError(
                "A14_A0_ROUTE_CLASS_SHAPE_DRIFT "
                f"{route_class.shape}"
            )

        if left_next_class.shape != (
            480,
            MOVE_COUNT,
        ):
            raise RuntimeError(
                "A14_A0_LEFT_NEXT_SHAPE_DRIFT "
                f"{left_next_class.shape}"
            )

        next_q = np.empty(
            (
                PIECE_COUNT,
                POSE_COUNT,
                POSE_COUNT,
                MOVE_COUNT,
            ),
            dtype=np.uint8,
        )

        q_values = np.arange(
            POSE_COUNT,
            dtype=np.intp,
        )

        for piece in range(
            PIECE_COUNT
        ):
            for target_q in range(
                POSE_COUNT
            ):
                pair_ids = (
                    q_values * POSE_COUNT
                    + int(target_q)
                )

                classes = route_class[
                    piece,
                    pair_ids,
                ].astype(
                    np.int32,
                    copy=False,
                )

                if np.unique(
                    classes
                ).size != POSE_COUNT:
                    raise RuntimeError(
                        "A14_A0_FIXED_TARGET_RELATION_NOT_BIJECTIVE "
                        f"piece={piece} target_q={target_q}"
                    )

                class_to_q = {
                    int(cid): int(q)
                    for q, cid
                    in enumerate(classes)
                }

                for current_q in range(
                    POSE_COUNT
                ):
                    cid = int(
                        classes[
                            current_q
                        ]
                    )

                    for mid in range(
                        MOVE_COUNT
                    ):
                        child_cid = int(
                            left_next_class[
                                cid,
                                mid,
                            ]
                        )

                        if child_cid not in class_to_q:
                            raise RuntimeError(
                                "A14_A0_RELATION_TRANSITION_ESCAPES_FIXED_TARGET "
                                f"piece={piece} target={target_q} "
                                f"current={current_q} move={mid} "
                                f"child_class={child_cid}"
                            )

                        next_q[
                            piece,
                            target_q,
                            current_q,
                            mid,
                        ] = np.uint8(
                            class_to_q[
                                child_cid
                            ]
                        )

        return cls(
            route_class=route_class,
            class_distance=class_distance,
            next_q=next_q,
        )

    def relation_class(
        self,
        piece: int,
        current_q: int,
        target_q: int,
    ) -> int:
        return int(
            self.route_class[
                int(piece),
                int(current_q) * POSE_COUNT
                + int(target_q),
            ]
        )

    def local_distance(
        self,
        piece: int,
        current_q: int,
        target_q: int,
    ) -> int:
        cid = self.relation_class(
            piece,
            current_q,
            target_q,
        )

        return int(
            self.class_distance[
                cid
            ]
        )


@dataclass(frozen=True)
class PieceSupport:
    piece: int
    path_exists: bool
    support_masks: tuple[int, ...]
    forward_state_counts: tuple[int, ...]
    backward_state_counts: tuple[int, ...]


@dataclass(frozen=True)
class ATOFixpointState:
    domain_masks: tuple[int, ...]
    materialized_columns: int


@dataclass(frozen=True)
class ATOPropagationReceipt:
    rounds: int
    contradiction: bool
    converged: bool
    domain_masks_before: tuple[int, ...]
    domain_masks_after: tuple[int, ...]
    changed_columns: tuple[int, ...]
    forced_columns: tuple[int, ...]

    contradiction_pieces: tuple[int, ...]
    empty_common_columns: tuple[int, ...]

    piece_support_last_round: tuple[PieceSupport, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "rounds": int(
                self.rounds
            ),
            "contradiction": bool(
                self.contradiction
            ),
            "converged": bool(
                self.converged
            ),
            "domain_masks_before": [
                int(x)
                for x in self.domain_masks_before
            ],
            "domain_masks_after": [
                int(x)
                for x in self.domain_masks_after
            ],
            "domain_sizes_before": [
                int(x).bit_count()
                for x in self.domain_masks_before
            ],
            "domain_sizes_after": [
                int(x).bit_count()
                for x in self.domain_masks_after
            ],
            "changed_columns": [
                int(x)
                for x in self.changed_columns
            ],
            "forced_columns": [
                int(x)
                for x in self.forced_columns
            ],
            "contradiction_pieces": [
                int(x)
                for x in self.contradiction_pieces
            ],
            "empty_common_columns": [
                int(x)
                for x in self.empty_common_columns
            ],
            "authority": {
                "hard_delete": (
                    "EXACT_SINGLE_PIECE_CURRENT_TARGET_NECESSITY"
                ),
                "survival_full_cube_SAT_authority": False,
                "pair_coupling": "NONE",
            },
        }


@dataclass(frozen=True)
class ATOPropagationOutcome:
    receipt: ATOPropagationReceipt
    final_state: ATOFixpointState | None

    # Compatibility fields consumed by frozen A11 search telemetry.
    scratch_builds: int
    incremental_updates: int

    reused_forward_boundaries: int = 0
    reused_backward_boundaries: int = 0
    recomputed_forward_steps: int = 0
    recomputed_backward_steps: int = 0


def _domain_move_ids(
    mask: int,
) -> tuple[int, ...]:
    value = int(mask)

    return tuple(
        mid
        for mid in range(
            MOVE_COUNT
        )
        if value
        & (
            1 << mid
        )
    )


def _piece_support_from_resident(
    *,
    automata: ResidentPerPieceAutomata,
    piece: int,
    q_current: int,
    q_target: int,
    domain_masks: tuple[int, ...],
    previous_face: str | None,
    next_move: str | None,
) -> PieceSupport:
    """
    Exact fixed-target single-piece support using resident compiled transitions.
    """
    piece = int(piece)
    q_current = int(q_current)
    q_target = int(q_target)
    n = len(
        domain_masks
    )

    transition = automata.next_q[
        piece,
        q_target,
    ]  # [24,18]

    forward = np.zeros(
        (
            n + 1,
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
        start_ctx,
        q_current,
    ] = True

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
                MOVE_CONTEXT[
                    mid
                ]
            )

            for ctx in np.flatnonzero(
                CANONICAL_ALLOW[
                    :,
                    mid,
                ]
            ):
                source_q = np.flatnonzero(
                    forward[
                        position,
                        int(ctx),
                    ]
                )

                if not source_q.size:
                    continue

                child_q = transition[
                    source_q.astype(
                        np.intp,
                        copy=False,
                    ),
                    int(mid),
                ].astype(
                    np.intp,
                    copy=False,
                )

                forward[
                    position + 1,
                    child_ctx,
                    child_q,
                ] = True

    backward = np.zeros(
        (
            n + 1,
            7,
            POSE_COUNT,
        ),
        dtype=np.bool_,
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
                ctx,
                q_target,
            ] = True

    all_q = np.arange(
        POSE_COUNT,
        dtype=np.intp,
    )

    for position in range(
        n - 1,
        -1,
        -1,
    ):
        mids = _domain_move_ids(
            domain_masks[
                position
            ]
        )

        for ctx in range(
            7
        ):
            for mid in mids:
                if not bool(
                    CANONICAL_ALLOW[
                        ctx,
                        mid,
                    ]
                ):
                    continue

                child_ctx = int(
                    MOVE_CONTEXT[
                        mid
                    ]
                )
                child_accept = backward[
                    position + 1,
                    child_ctx,
                ]

                if not np.any(
                    child_accept
                ):
                    continue

                child_q = transition[
                    all_q,
                    int(mid),
                ].astype(
                    np.intp,
                    copy=False,
                )

                good = all_q[
                    child_accept[
                        child_q
                    ]
                ]

                if good.size:
                    backward[
                        position,
                        ctx,
                        good,
                    ] = True

    path_exists = bool(
        backward[
            0,
            start_ctx,
            q_current,
        ]
    )

    support_masks = []

    if not path_exists:
        support_masks = [
            0
            for _ in range(
                n
            )
        ]
    else:
        for position in range(
            n
        ):
            support = 0
            mids = _domain_move_ids(
                domain_masks[
                    position
                ]
            )

            for mid in mids:
                child_ctx = int(
                    MOVE_CONTEXT[
                        mid
                    ]
                )
                supported = False

                for ctx in range(
                    7
                ):
                    if not bool(
                        CANONICAL_ALLOW[
                            ctx,
                            mid,
                        ]
                    ):
                        continue

                    source_q = np.flatnonzero(
                        forward[
                            position,
                            ctx,
                        ]
                    )

                    if not source_q.size:
                        continue

                    child_q = transition[
                        source_q.astype(
                            np.intp,
                            copy=False,
                        ),
                        int(mid),
                    ].astype(
                        np.intp,
                        copy=False,
                    )

                    if np.any(
                        backward[
                            position + 1,
                            child_ctx,
                            child_q,
                        ]
                    ):
                        supported = True
                        break

                if supported:
                    support |= (
                        1 << mid
                    )

            support_masks.append(
                int(
                    support
                )
            )

    return PieceSupport(
        piece=piece,
        path_exists=bool(
            path_exists
        ),
        support_masks=tuple(
            int(x)
            for x in support_masks
        ),
        forward_state_counts=tuple(
            int(
                np.count_nonzero(
                    forward[
                        position
                    ]
                )
            )
            for position in range(
                n + 1
            )
        ),
        backward_state_counts=tuple(
            int(
                np.count_nonzero(
                    backward[
                        position
                    ]
                )
            )
            for position in range(
                n + 1
            )
        ),
    )


class IndependentPerPieceATOPropagator:
    """
    Exact necessary ATO propagator for Branch A.

    The optional seed_state is deliberately ignored in A0.  All runtime
    transition work is still resident-table driven; incremental state reuse can
    be studied later without changing semantics.
    """

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

    def propagate_cached(
        self,
        field: ElasticColumnField,
        *,
        previous_face: str | None,
        next_move: str | None,
        seed_state: ATOFixpointState | None = None,
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
                "independent per-piece ATO propagation requires >=1 column"
            )

        before = self._masks(
            field
        )

        q_current = bi.normalize_q(
            field.left_anchor_q
        )
        q_target = bi.normalize_q(
            field.right_anchor_q
        )

        contradiction = False
        converged = False
        rounds_done = 0
        contradiction_pieces: tuple[int, ...] = tuple()
        empty_columns: tuple[int, ...] = tuple()
        last_piece_rows: tuple[PieceSupport, ...] = tuple()

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

            piece_rows = tuple(
                _piece_support_from_resident(
                    automata=self.automata,
                    piece=piece,
                    q_current=int(
                        q_current[
                            piece
                        ]
                    ),
                    q_target=int(
                        q_target[
                            piece
                        ]
                    ),
                    domain_masks=current,
                    previous_face=previous_face,
                    next_move=next_move,
                )
                for piece in range(
                    PIECE_COUNT
                )
            )

            last_piece_rows = piece_rows

            contradiction_pieces = tuple(
                row.piece
                for row in piece_rows
                if not row.path_exists
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
                break

            aggregate = list(
                current
            )

            for row in piece_rows:
                for position in range(
                    n
                ):
                    aggregate[
                        position
                    ] &= int(
                        row.support_masks[
                            position
                        ]
                    )

            empty_columns = tuple(
                position
                for position, mask
                in enumerate(
                    aggregate
                )
                if int(
                    mask
                )
                == 0
            )

            for position, mask in enumerate(
                aggregate
            ):
                field.columns[
                    position
                ].restrict(
                    int(
                        mask
                    )
                )

            after = self._masks(
                field
            )

            if empty_columns or any(
                int(mask) == 0
                for mask in after
            ):
                contradiction = True
                break

            if after == current:
                converged = True
                break

        final = self._masks(
            field
        )

        changed_columns = tuple(
            position
            for position, (
                old,
                new,
            ) in enumerate(
                zip(
                    before,
                    final,
                )
            )
            if int(old)
            != int(new)
        )

        forced_columns = tuple(
            position
            for position, mask
            in enumerate(
                final
            )
            if int(
                mask
            ).bit_count()
            == 1
        )

        removed = sum(
            int(old).bit_count()
            - int(new).bit_count()
            for old, new
            in zip(
                before,
                final,
            )
        )

        self.changed_move_bits += int(
            removed
        )

        if contradiction:
            self.contradiction_calls += 1

        wall = (
            time.perf_counter()
            - t0
        )
        self.wall_s += float(
            wall
        )

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
            contradiction_pieces=tuple(
                int(x)
                for x in contradiction_pieces
            ),
            empty_common_columns=tuple(
                int(x)
                for x in empty_columns
            ),
            piece_support_last_round=last_piece_rows,
        )

        final_state = (
            None
            if contradiction
            else ATOFixpointState(
                domain_masks=final,
                materialized_columns=n,
            )
        )

        return ATOPropagationOutcome(
            receipt=receipt,
            final_state=final_state,
            scratch_builds=int(
                rounds_done
            ),
            incremental_updates=0,
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
            "transition_source": (
                "A14-P0 RESIDENT ROUTE RELATION TABLE"
            ),
            "runtime_Q_NEXT_reconstruction": False,
            "pair_coupling": "NONE",
        }
