from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from cubelab.ato.regular_support import MOVE_NAMES

from .legacy_compat import ensure_legacy_compat


FULL24 = (1 << 24) - 1
FULL_MOVE_MASK = (1 << len(MOVE_NAMES)) - 1


def _checked_q_next(
    q_next: Any,
) -> np.ndarray:
    table = np.asarray(
        q_next
    )

    if (
        table.ndim != 3
        or table.shape[1:] != (
            24,
            len(MOVE_NAMES),
        )
        or not np.issubdtype(
            table.dtype,
            np.integer,
        )
        or np.any(
            table < 0
        )
        or np.any(
            table >= 24
        )
    ):
        raise ValueError(
            "exact deterministic Q transition table "
            "must have shape (pieces,24,18)"
        )

    return np.asarray(
        table,
        dtype=np.uint8,
    )


def _build_preimage_tables(
    transition: np.ndarray,
) -> np.ndarray:
    """
    Exact 24-bit preimage LUT:
      18 moves * 3 bytes * 256 masks.

    Entry [move, byte, bits] returns the 24-bit set of source Q states whose
    destination under `move` is selected by that byte of the destination mask.
    """

    tr = np.asarray(
        transition,
        dtype=np.uint8,
    )

    if tr.shape != (
        24,
        len(MOVE_NAMES),
    ):
        raise ValueError(
            "piece transition must be 24x18"
        )

    tables = np.zeros(
        (
            len(MOVE_NAMES),
            3,
            256,
        ),
        dtype=np.uint32,
    )

    for move_id in range(
        len(MOVE_NAMES)
    ):
        by_destination = [
            0
        ] * 24

        for q in range(
            24
        ):
            destination = int(
                tr[
                    q,
                    move_id,
                ]
            )

            by_destination[
                destination
            ] |= (
                1 << q
            )

        for byte in range(
            3
        ):
            row = tables[
                move_id,
                byte,
            ]

            for bits in range(
                1,
                256
            ):
                low = (
                    bits
                    & -bits
                )

                destination = (
                    8 * byte
                    + low.bit_length()
                    - 1
                )

                row[
                    bits
                ] = np.uint32(
                    int(
                        row[
                            bits
                            ^ low
                        ]
                    )
                    |
                    int(
                        by_destination[
                            destination
                        ]
                    )
                )

    return tables


def _preimage(
    mask: int,
    move_id: int,
    tables: np.ndarray,
) -> int:
    mask = int(
        mask
    )

    move_id = int(
        move_id
    )

    if not (
        0 <= mask <= FULL24
        and
        0 <= move_id < len(
            MOVE_NAMES
        )
    ):
        raise ValueError(
            "invalid target mask or move"
        )

    return int(
        int(
            tables[
                move_id,
                0,
                mask & 255,
            ]
        )
        |
        int(
            tables[
                move_id,
                1,
                (mask >> 8)
                & 255,
            ]
        )
        |
        int(
            tables[
                move_id,
                2,
                (mask >> 16)
                & 255,
            ]
        )
    )


def _validate_seed_language(
    language: Any,
) -> tuple[
    tuple[Any, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    nodes = tuple(
        language.arena.nodes
    )

    if len(
        nodes
    ) < 2:
        raise ValueError(
            "seed MDD arena missing FALSE/TRUE terminals"
        )

    false_node = nodes[
        0
    ]

    true_node = nodes[
        1
    ]

    if (
        (
            int(
                false_node.remaining
            ),
            bool(
                false_node.accepting
            ),
            tuple(
                false_node.edges
            ),
        )
        != (
            0,
            False,
            (),
        )
        or
        (
            int(
                true_node.remaining
            ),
            bool(
                true_node.accepting
            ),
            tuple(
                true_node.edges
            ),
        )
        != (
            0,
            True,
            (),
        )
    ):
        raise ValueError(
            "seed MDD terminal convention drift"
        )

    root = int(
        language.root
    )

    if not (
        0 <= root < len(
            nodes
        )
    ):
        raise ValueError(
            "seed MDD root out of range"
        )

    horizon = int(
        language.horizon
    )

    if (
        root not in (
            0,
            1,
        )
        and int(
            nodes[
                root
            ].remaining
        )
        != horizon
    ):
        raise ValueError(
            "seed MDD root horizon drift"
        )

    node_count = len(
        nodes
    )

    child = np.zeros(
        (
            node_count,
            len(
                MOVE_NAMES
            ),
        ),
        dtype=np.uint32,
    )

    outgoing = np.zeros(
        node_count,
        dtype=np.uint32,
    )

    remaining = np.zeros(
        node_count,
        dtype=np.int16,
    )

    for node_id, node in enumerate(
        nodes
    ):
        remaining[
            node_id
        ] = int(
            node.remaining
        )

        if node_id < 2:
            continue

        if (
            int(
                node.remaining
            )
            < 1
            or bool(
                node.accepting
            )
            or not node.edges
        ):
            raise ValueError(
                "seed MDD contains nonproductive nonterminal"
            )

        previous_move = -1
        mask = 0

        for move_id, child_id in node.edges:
            move_id = int(
                move_id
            )

            child_id = int(
                child_id
            )

            if not (
                0 <= move_id
                < len(
                    MOVE_NAMES
                )
                and move_id
                > previous_move
                and 0
                < child_id
                < node_id
                and int(
                    nodes[
                        child_id
                    ].remaining
                )
                == int(
                    node.remaining
                )
                - 1
            ):
                raise ValueError(
                    "seed MDD requires ordered topological labelled edges"
                )

            previous_move = (
                move_id
            )

            child[
                node_id,
                move_id,
            ] = np.uint32(
                child_id
            )

            mask |= (
                1
                << move_id
            )

        outgoing[
            node_id
        ] = np.uint32(
            mask
        )

    return (
        nodes,
        child,
        outgoing,
        remaining,
    )


def _suffix_q_masks(
    *,
    nodes: tuple[Any, ...],
    transition: np.ndarray,
    target_mask: int,
) -> tuple[
    np.ndarray,
    dict[str, Any],
]:
    """
    Exact per-node Q co-reachability over THIS seed MDD.

    masks[node] contains exactly the Q poses from which at least one word
    accepted by the seed sublanguage rooted at `node` ends in target_mask.
    """

    target_mask = int(
        target_mask
    )

    if not (
        0 <= target_mask <= FULL24
    ):
        raise ValueError(
            "target mask must be a 24-bit set"
        )

    started = perf_counter()

    tables = _build_preimage_tables(
        transition
    )

    masks = np.zeros(
        len(
            nodes
        ),
        dtype=np.uint32,
    )

    masks[
        1
    ] = np.uint32(
        target_mask
    )

    edge_count = 0

    for node_id in range(
        2,
        len(
            nodes
        )
    ):
        node = nodes[
            node_id
        ]

        answer = 0

        for move_id, child_id in node.edges:
            answer |= _preimage(
                int(
                    masks[
                        int(
                            child_id
                        )
                    ]
                ),
                int(
                    move_id
                ),
                tables,
            )

            edge_count += 1

        masks[
            node_id
        ] = np.uint32(
            answer
        )

    return (
        masks,
        {
            "target_mask":
                target_mask,

            "root_popcount":
                int(
                    masks[
                        -1
                    ]
                ).bit_count()
                if len(
                    masks
                )
                else 0,

            "edge_visits":
                edge_count,

            "lut_bytes":
                int(
                    tables.nbytes
                ),

            "mask_bytes":
                int(
                    masks.nbytes
                ),

            "wall_s":
                perf_counter()
                - started,
        },
    )


@dataclass(slots=True)
class C4SeedTargetPackedAdapter:
    """
    Thin exact adapter for packed_product_arena.first_packed_witness.

    Key columns:
        [seed_mdd_node_id, q(piece_0), ..., q(piece_k)]

    The seed MDD already contains:
        - the exact labelled SAME-word seed language,
        - candidate column domains,
        - canonical boundary context,
        - all seed-piece obligations.

    Pending Q obligations are carried explicitly in the packed key.

    `common_mask` uses exact per-piece co-reachability over the SAME seed MDD.
    A move is deleted only if:
        - the seed node has no such labelled edge, or
        - at least one pending piece has no seed-language suffix from its
          post-move Q pose into its exact target mask.

    Marginal co-reachability may leave joint false positives, but cannot delete
    a joint witness. Exact joint authority comes from packed state propagation
    plus terminal acceptance in `accepting`.
    """

    language: Any
    q_next: Any
    q_before: Sequence[int]
    target_masks: Sequence[int]
    pending_pieces: Sequence[int]

    representation: str = field(
        init=False,
        default=
            "C4_SEED_MDD_PLUS_PENDING_Q_EXACT_COREACH",
    )

    pieces: tuple[int, ...] = field(
        init=False
    )

    root_key: np.ndarray = field(
        init=False
    )

    seed_child: np.ndarray = field(
        init=False
    )

    seed_outgoing: np.ndarray = field(
        init=False
    )

    node_remaining: np.ndarray = field(
        init=False
    )

    suffix_masks: np.ndarray = field(
        init=False
    )

    prep_metrics: dict[
        str,
        Any,
    ] = field(
        init=False
    )

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

        suffix_rows = []

        suffix_receipts = []

        root = int(
            self.language.root
        )

        root_feasible = True

        for piece in pending:
            masks, receipt = _suffix_q_masks(
                nodes=
                    nodes,

                transition=
                    q_next[
                        piece
                    ],

                target_mask=
                    target_masks[
                        piece
                    ],
            )

            receipt[
                "piece"
            ] = int(
                piece
            )

            receipt[
                "root_popcount"
            ] = int(
                masks[
                    root
                ]
            ).bit_count()

            initial_q = int(
                q_before[
                    piece
                ]
            )

            receipt[
                "initial_q"
            ] = initial_q

            receipt[
                "root_initial_q_live"
            ] = bool(
                int(
                    masks[
                        root
                    ]
                )
                &
                (
                    1
                    << initial_q
                )
            )

            root_feasible &= bool(
                receipt[
                    "root_initial_q_live"
                ]
            )

            suffix_rows.append(
                masks
            )

            suffix_receipts.append(
                receipt
            )

        suffix_masks = np.stack(
            suffix_rows,
            axis=0,
        ).astype(
            np.uint32,
            copy=False,
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

        self.q_next = q_next
        self.q_before = q_before
        self.target_masks = target_masks
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
                self.representation,

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
                suffix_receipts,

            "root_joint_marginal_feasible":
                bool(
                    root_feasible
                ),

            "prep_wall_s":
                perf_counter()
                - started,
        }

    def common_mask(
        self,
        keys: np.ndarray,
        remaining: int,
    ) -> np.ndarray:
        keys = np.asarray(
            keys
        )

        if (
            keys.ndim != 2
            or keys.shape[
                1
            ]
            != len(
                self.root_key
            )
        ):
            raise ValueError(
                "packed key shape mismatch"
            )

        remaining = int(
            remaining
        )

        node_ids = keys[
            :,
            0
        ].astype(
            np.int64,
            copy=False,
        )

        if np.any(
            node_ids < 0
        ) or np.any(
            node_ids
            >= len(
                self.seed_outgoing
            )
        ):
            raise ValueError(
                "seed node id out of range"
            )

        if np.any(
            self.node_remaining[
                node_ids
            ]
            != remaining
        ):
            raise ValueError(
                "packed frontier remaining-depth drift"
            )

        masks = self.seed_outgoing[
            node_ids
        ].copy()

        if not len(
            masks
        ):
            return masks

        q_columns = keys[
            :,
            1:
        ].astype(
            np.int64,
            copy=False,
        )

        pending_array = np.asarray(
            self.pieces,
            dtype=np.int64,
        )

        for move_id in range(
            len(
                MOVE_NAMES
            )
        ):
            bit = np.uint32(
                1
                << move_id
            )

            selected = np.flatnonzero(
                (
                    masks
                    & bit
                )
                != 0
            )

            if not len(
                selected
            ):
                continue

            selected_nodes = node_ids[
                selected
            ]

            children = self.seed_child[
                selected_nodes,
                move_id,
            ].astype(
                np.int64,
                copy=False,
            )

            if np.any(
                children <= 0
            ):
                raise RuntimeError(
                    "seed outgoing mask/child table drift"
                )

            alive = np.ones(
                len(
                    selected
                ),
                dtype=np.bool_,
            )

            for column, piece in enumerate(
                self.pieces
            ):
                q = q_columns[
                    selected,
                    column,
                ]

                q2 = self.q_next[
                    int(
                        piece
                    ),
                    q,
                    move_id,
                ].astype(
                    np.uint32,
                    copy=False,
                )

                suffix = self.suffix_masks[
                    column,
                    children,
                ]

                live_bit = np.left_shift(
                    np.uint32(
                        1
                    ),
                    q2,
                )

                alive &= (
                    suffix
                    & live_bit
                ) != 0

                if not np.any(
                    alive
                ):
                    break

            if np.all(
                alive
            ):
                continue

            masks[
                selected[
                    ~alive
                ]
            ] &= np.uint32(
                FULL_MOVE_MASK
                ^ int(
                    bit
                )
            )

        return masks

    def child_keys(
        self,
        keys: np.ndarray,
        move_id: int,
    ) -> np.ndarray:
        keys = np.asarray(
            keys
        )

        move_id = int(
            move_id
        )

        if not (
            0 <= move_id < len(
                MOVE_NAMES
            )
        ):
            raise ValueError(
                "move id out of range"
            )

        node_ids = keys[
            :,
            0
        ].astype(
            np.int64,
            copy=False,
        )

        children = self.seed_child[
            node_ids,
            move_id,
        ]

        if np.any(
            children == 0
        ):
            raise RuntimeError(
                "first_packed_witness requested absent seed edge"
            )

        result = np.empty(
            keys.shape,
            dtype=np.uint32,
        )

        result[
            :,
            0
        ] = children

        q = keys[
            :,
            1:
        ].astype(
            np.int64,
            copy=False,
        )

        pieces = np.asarray(
            self.pieces,
            dtype=np.int64,
        )

        result[
            :,
            1:
        ] = self.q_next[
            pieces[
                None,
                :
            ],
            q,
            move_id,
        ].astype(
            np.uint32,
            copy=False,
        )

        return result

    def accepting(
        self,
        keys: np.ndarray,
    ) -> np.ndarray:
        keys = np.asarray(
            keys
        )

        result = (
            keys[
                :,
                0
            ]
            == 1
        )

        for column, piece in enumerate(
            self.pieces
        ):
            q = keys[
                :,
                column
                + 1
            ].astype(
                np.uint32,
                copy=False,
            )

            target = np.uint32(
                self.target_masks[
                    int(
                        piece
                    )
                ]
            )

            q_bit = np.left_shift(
                np.uint32(
                    1
                ),
                q,
            )

            result &= (
                target
                & q_bit
            ) != 0

        return result


def first_c4_packed_witness(
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
    Reuse the existing packed exact first-witness engine unchanged.
    """

    ensure_legacy_compat()

    from cubelab.column_families.packed_product_arena import (
        first_packed_witness,
    )

    started = perf_counter()

    adapter = C4SeedTargetPackedAdapter(
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
            adapter.representation,

        "adapter_pieces":[
            int(v)
            for v in adapter.pieces
        ],

        "adapter_prep":
            dict(
                adapter.prep_metrics
            ),

        "adapter_prep_wall_s":
            prep_wall,

        "packed_engine_wall_s":
            engine_wall,

        "wall_total_with_adapter_prep_s":
            perf_counter()
            - started,

        "full_closure_MDD_materialized":
            False,

        "first_packed_witness_reused":
            True,
    })

    return out


__all__ = [
    "C4SeedTargetPackedAdapter",
    "first_c4_packed_witness",
]
