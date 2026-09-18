from __future__ import annotations

from time import perf_counter
from typing import Any, Sequence

import numpy as np

from cubelab.ato.regular_support import MOVE_NAMES

from .c4_packed_first_witness import (
    C4SeedTargetPackedAdapter,
    _build_preimage_tables,
    _checked_q_next,
    _validate_seed_language,
)
from .legacy_compat import ensure_legacy_compat


def _batched_suffix_q_masks_vectorized(
    *,
    nodes: tuple[Any, ...],
    seed_child: np.ndarray,
    node_remaining: np.ndarray,
    transitions: np.ndarray,
    target_masks: Sequence[int],
    horizon: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Exact batched co-reachability for all pending pieces.

    Recurrence is identical to the scalar C4-P5-R1 implementation:

        S_p(TRUE) = target_mask[p]

        S_p(node) =
            OR over labelled edges (move -> child)
                preimage_p,move(S_p(child))

    Difference: nodes with the same `remaining` value are evaluated in one
    NumPy batch, and all pending pieces are evaluated together.

    No joint condition is introduced here. This computes the exact same
    per-piece marginal masks as the scalar implementation, only faster.
    """

    started = perf_counter()

    transitions = np.asarray(
        transitions,
        dtype=np.uint8,
    )

    if (
        transitions.ndim != 3
        or transitions.shape[1:] != (
            24,
            len(MOVE_NAMES),
        )
    ):
        raise ValueError(
            "pending transitions must have shape (P,24,18)"
        )

    piece_count = int(
        transitions.shape[0]
    )

    targets = np.asarray(
        [
            int(mask)
            for mask in target_masks
        ],
        dtype=np.uint32,
    )

    if targets.shape != (
        piece_count,
    ):
        raise ValueError(
            "target mask count does not match pending piece count"
        )

    # Reuse the exact scalar LUT builder for each piece. LUT construction is
    # tiny; the expensive node×edge Python traversal is what is removed.
    lut_started = perf_counter()

    tables = np.stack(
        [
            _build_preimage_tables(
                transitions[
                    piece_index
                ]
            )
            for piece_index in range(
                piece_count
            )
        ],
        axis=0,
    ).astype(
        np.uint32,
        copy=False,
    )

    lut_wall = (
        perf_counter()
        - lut_started
    )

    suffix = np.zeros(
        (
            piece_count,
            len(
                nodes
            ),
        ),
        dtype=np.uint32,
    )

    # SymbolicMDD terminal convention: node 1 is TRUE.
    suffix[
        :,
        1
    ] = targets

    piece_index = np.arange(
        piece_count,
        dtype=np.int64,
    )[
        :,
        None
    ]

    layer_receipts = []

    recurrence_started = perf_counter()

    edge_visits_per_piece = 0

    for remaining in range(
        1,
        int(
            horizon
        )
        + 1,
    ):
        node_ids = np.flatnonzero(
            node_remaining
            == int(
                remaining
            )
        ).astype(
            np.int64,
            copy=False,
        )

        if not len(
            node_ids
        ):
            layer_receipts.append({
                "remaining":
                    int(
                        remaining
                    ),

                "nodes":
                    0,

                "labelled_edges":
                    0,
            })
            continue

        answer = np.zeros(
            (
                piece_count,
                len(
                    node_ids
                ),
            ),
            dtype=np.uint32,
        )

        layer_edges = 0

        for move_id in range(
            len(
                MOVE_NAMES
            )
        ):
            children_all = seed_child[
                node_ids,
                move_id,
            ].astype(
                np.int64,
                copy=False,
            )

            valid_positions = np.flatnonzero(
                children_all
                != 0
            ).astype(
                np.int64,
                copy=False,
            )

            if not len(
                valid_positions
            ):
                continue

            children = children_all[
                valid_positions
            ]

            child_masks = suffix[
                :,
                children,
            ]

            byte0 = (
                child_masks
                & np.uint32(
                    255
                )
            ).astype(
                np.int64,
                copy=False,
            )

            byte1 = (
                (
                    child_masks
                    >> np.uint32(
                        8
                    )
                )
                & np.uint32(
                    255
                )
            ).astype(
                np.int64,
                copy=False,
            )

            byte2 = (
                (
                    child_masks
                    >> np.uint32(
                        16
                    )
                )
                & np.uint32(
                    255
                )
            ).astype(
                np.int64,
                copy=False,
            )

            preimage = (
                tables[
                    piece_index,
                    move_id,
                    0,
                    byte0,
                ]
                |
                tables[
                    piece_index,
                    move_id,
                    1,
                    byte1,
                ]
                |
                tables[
                    piece_index,
                    move_id,
                    2,
                    byte2,
                ]
            )

            answer[
                :,
                valid_positions
            ] |= preimage

            layer_edges += int(
                len(
                    valid_positions
                )
            )

        suffix[
            :,
            node_ids
        ] = answer

        edge_visits_per_piece += (
            layer_edges
        )

        layer_receipts.append({
            "remaining":
                int(
                    remaining
                ),

            "nodes":
                int(
                    len(
                        node_ids
                    )
                ),

            "labelled_edges":
                int(
                    layer_edges
                ),
        })

    recurrence_wall = (
        perf_counter()
        - recurrence_started
    )

    return (
        suffix,
        {
            "pending_piece_count":
                piece_count,

            "node_count":
                int(
                    len(
                        nodes
                    )
                ),

            "edge_visits_per_piece":
                int(
                    edge_visits_per_piece
                ),

            "logical_piece_edge_visits":
                int(
                    edge_visits_per_piece
                    * piece_count
                ),

            "lut_bytes":
                int(
                    tables.nbytes
                ),

            "suffix_bytes":
                int(
                    suffix.nbytes
                ),

            "lut_wall_s":
                float(
                    lut_wall
                ),

            "recurrence_wall_s":
                float(
                    recurrence_wall
                ),

            "wall_s":
                float(
                    perf_counter()
                    - started
                ),

            "layers":
                layer_receipts,
        },
    )


class VectorizedC4SeedTargetPackedAdapter(
    C4SeedTargetPackedAdapter
):
    """
    C4-P5-R2 adapter.

    Packed search semantics are inherited unchanged from C4-P5-R1.
    Only adapter preparation is replaced by an exact batched/vectorized
    implementation of the SAME suffix-Q co-reachability recurrence.
    """

    def __post_init__(
        self,
    ) -> None:
        started = perf_counter()

        q_next = _checked_q_next(
            self.q_next
        )

        piece_count = int(
            q_next.shape[
                0
            ]
        )

        q_before = tuple(
            int(v)
            for v in self.q_before
        )

        target_masks = tuple(
            int(v)
            for v in self.target_masks
        )

        pending = tuple(
            int(v)
            for v in self.pending_pieces
        )

        if len(
            q_before
        ) != piece_count:
            raise ValueError(
                "q_before piece count mismatch"
            )

        if len(
            target_masks
        ) != piece_count:
            raise ValueError(
                "target_masks piece count mismatch"
            )

        if (
            len(
                set(
                    pending
                )
            )
            != len(
                pending
            )
            or any(
                piece < 0
                or piece >= piece_count
                for piece in pending
            )
        ):
            raise ValueError(
                "invalid pending piece set"
            )

        if not pending:
            raise ValueError(
                "C4 packed adapter requires pending pieces"
            )

        (
            nodes,
            seed_child,
            seed_outgoing,
            node_remaining,
        ) = _validate_seed_language(
            self.language
        )

        pending_transitions = q_next[
            np.asarray(
                pending,
                dtype=np.int64,
            )
        ]

        pending_targets = tuple(
            target_masks[
                piece
            ]
            for piece in pending
        )

        suffix_masks, vector_receipt = (
            _batched_suffix_q_masks_vectorized(
                nodes=
                    nodes,

                seed_child=
                    seed_child,

                node_remaining=
                    node_remaining,

                transitions=
                    pending_transitions,

                target_masks=
                    pending_targets,

                horizon=
                    int(
                        self.language.horizon
                    ),
            )
        )

        root = int(
            self.language.root
        )

        root_key = np.empty(
            1
            + len(
                pending
            ),
            dtype=np.uint32,
        )

        root_key[
            0
        ] = np.uint32(
            root
        )

        root_key[
            1:
        ] = np.asarray(
            [
                q_before[
                    piece
                ]
                for piece in pending
            ],
            dtype=np.uint32,
        )

        piece_receipts = []

        root_joint_marginal_feasible = True

        for column, piece in enumerate(
            pending
        ):
            initial_q = int(
                q_before[
                    piece
                ]
            )

            root_mask = int(
                suffix_masks[
                    column,
                    root,
                ]
            )

            root_live = bool(
                root_mask
                &
                (
                    1
                    << initial_q
                )
            )

            root_joint_marginal_feasible &= (
                root_live
            )

            piece_receipts.append({
                "piece":
                    int(
                        piece
                    ),

                "target_mask":
                    int(
                        target_masks[
                            piece
                        ]
                    ),

                "initial_q":
                    initial_q,

                "root_popcount":
                    int(
                        root_mask
                    ).bit_count(),

                "root_initial_q_live":
                    root_live,
            })

        self.q_next = q_next
        self.q_before = q_before
        self.target_masks = (
            target_masks
        )
        self.pieces = pending

        self.seed_child = (
            seed_child
        )

        self.seed_outgoing = (
            seed_outgoing
        )

        self.node_remaining = (
            node_remaining
        )

        self.suffix_masks = (
            suffix_masks
        )

        self.root_key = (
            root_key
        )

        self.prep_metrics = {
            "representation":
                "C4_SEED_MDD_PLUS_PENDING_Q_EXACT_COREACH_VECTORIZED",

            "horizon":
                int(
                    self.language.horizon
                ),

            "seed_nodes":
                len(
                    nodes
                ),

            "seed_edges":
                int(
                    np.count_nonzero(
                        seed_child
                    )
                ),

            "pending_piece_count":
                len(
                    pending
                ),

            "pending_pieces":[
                int(v)
                for v in pending
            ],

            "seed_child_bytes":
                int(
                    seed_child.nbytes
                ),

            "seed_outgoing_bytes":
                int(
                    seed_outgoing.nbytes
                ),

            "suffix_mask_bytes":
                int(
                    suffix_masks.nbytes
                ),

            "suffix_piece_receipts":
                piece_receipts,

            "root_joint_marginal_feasible":
                bool(
                    root_joint_marginal_feasible
                ),

            "vectorized_suffix":
                vector_receipt,

            "prep_wall_s":
                float(
                    perf_counter()
                    - started
                ),
        }


def first_c4_packed_witness_vectorized(
    *,
    language: Any,
    q_next: Any,
    q_before: Sequence[int],
    target_masks: Sequence[int],
    pending_pieces: Sequence[int],
    node_cap: int = 5_000_000,
    time_cap_seconds: float | None = None,
    rss_growth_cap_mib: float = 1536.0,
) -> dict[str, Any]:
    """
    Reuse packed_product_arena.first_packed_witness unchanged.
    """

    ensure_legacy_compat()

    from cubelab.column_families.packed_product_arena import (
        first_packed_witness,
    )

    started = perf_counter()

    adapter = VectorizedC4SeedTargetPackedAdapter(
        language=
            language,

        q_next=
            q_next,

        q_before=
            q_before,

        target_masks=
            target_masks,

        pending_pieces=
            pending_pieces,
    )

    prep_wall = (
        perf_counter()
        - started
    )

    engine_started = perf_counter()

    result = first_packed_witness(
        adapter,
        int(
            language.horizon
        ),
        node_cap=
            int(
                node_cap
            ),

        time_cap_seconds=
            time_cap_seconds,

        rss_growth_cap_mib=
            float(
                rss_growth_cap_mib
            ),
    )

    engine_wall = (
        perf_counter()
        - engine_started
    )

    out = dict(
        result
    )

    out.update({
        "adapter_representation":
            "C4_SEED_MDD_PLUS_PENDING_Q_EXACT_COREACH_VECTORIZED",

        "adapter_pieces":[
            int(v)
            for v in adapter.pieces
        ],

        "adapter_prep":
            dict(
                adapter.prep_metrics
            ),

        "adapter_prep_wall_s":
            float(
                prep_wall
            ),

        "packed_engine_wall_s":
            float(
                engine_wall
            ),

        "wall_total_with_adapter_prep_s":
            float(
                perf_counter()
                - started
            ),

        "full_closure_MDD_materialized":
            False,

        "first_packed_witness_reused":
            True,

        "vectorized_suffix_prep":
            True,
    })

    return out


__all__ = [
    "VectorizedC4SeedTargetPackedAdapter",
    "first_c4_packed_witness_vectorized",
]
