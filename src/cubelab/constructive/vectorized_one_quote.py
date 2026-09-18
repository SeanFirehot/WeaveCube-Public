from __future__ import annotations

from array import array
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from cubelab.column_families.symbolic_mdd_v2 import (
    SymbolicMDD,
)

from .direct_q_restrict import (
    DirectRestrictUnknown,
    checked_target_mask,
    checked_transition,
)

from .exact_live_estimator import (
    OneLiveEstimate,
)

from .exact_one_scheduler import (
    ExactOneQuote,
)

from .prepared_q_restrict import (
    PreparedPieceMask,
)


POPCOUNT8 = np.asarray(
    [
        value.bit_count()
        for value in range(256)
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True, slots=True)
class FlatLayer:

    remaining: int

    # Parent-grouped representation for suffix DP.
    parent_unique: np.ndarray
    parent_starts: np.ndarray

    back_child: np.ndarray
    back_mid: np.ndarray

    # Child-grouped representation for forward DP.
    child_unique: np.ndarray
    child_starts: np.ndarray

    forward_parent: np.ndarray
    forward_mid: np.ndarray


@dataclass(frozen=True, slots=True)
class FlatMDD:

    language: SymbolicMDD

    layers: tuple[
        FlatLayer | None,
        ...
    ]

    degree: np.ndarray

    node_count: int
    edge_count: int

    build_wall_s: float


class VectorizedOneQuoteBank:

    def __init__(
        self,
        q_next,
    ):

        started = perf_counter()


        transition = np.asarray(
            q_next
        )


        if (
            transition.ndim != 3
            or
            transition.shape[
                1:
            ] != (
                24,
                18,
            )
            or
            not np.issubdtype(
                transition.dtype,
                np.integer,
            )
        ):

            raise ValueError(
                "q_next must have shape "
                "(piece_count,24,18)"
            )


        if (
            np.any(
                transition < 0
            )
            or
            np.any(
                transition >= 24
            )
        ):

            raise ValueError(
                "q_next contains invalid Q state"
            )


        self.q_next = np.asarray(
            transition,
            dtype=np.uint8,
        )


        self.piece_count = int(
            self.q_next.shape[
                0
            ]
        )


        self.pre_lut = np.zeros(
            (
                self.piece_count,
                18,
                3,
                256,
            ),
            dtype=np.uint32,
        )


        self.image_lut = np.zeros(
            (
                self.piece_count,
                18,
                3,
                256,
            ),
            dtype=np.uint32,
        )


        for piece in range(
            self.piece_count
        ):

            for mid in range(
                18
            ):

                destination_sources = [
                    0
                ] * 24


                for q in range(
                    24
                ):

                    destination = int(
                        self.q_next[
                            piece,
                            q,
                            mid,
                        ]
                    )


                    destination_sources[
                        destination
                    ] |= (
                        1 << q
                    )


                for byte in range(
                    3
                ):

                    base = (
                        8 * byte
                    )


                    pre = self.pre_lut[
                        piece,
                        mid,
                        byte,
                    ]


                    image = self.image_lut[
                        piece,
                        mid,
                        byte,
                    ]


                    for bits in range(
                        1,
                        256,
                    ):

                        low = (
                            bits
                            & -bits
                        )


                        offset = (
                            low.bit_length()
                            - 1
                        )


                        pre[
                            bits
                        ] = (
                            pre[
                                bits
                                ^ low
                            ]
                            |
                            destination_sources[
                                base
                                + offset
                            ]
                        )


                        source = (
                            base
                            + offset
                        )


                        destination = int(
                            self.q_next[
                                piece,
                                source,
                                mid,
                            ]
                        )


                        image[
                            bits
                        ] = (
                            image[
                                bits
                                ^ low
                            ]
                            |
                            (
                                1
                                << destination
                            )
                        )


        self.build_wall_s = (
            perf_counter()
            - started
        )


    def transition_rows(
        self,
        piece: int,
    ):

        return tuple(
            tuple(
                int(v)
                for v in row
            )
            for row in self.q_next[
                int(
                    piece
                )
            ].tolist()
        )


def _popcount_u32(
    values: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=np.uint32,
    )


    return (
        POPCOUNT8[
            values
            & np.uint32(
                255
            )
        ]
        +
        POPCOUNT8[
            (
                values
                >> np.uint32(
                    8
                )
            )
            & np.uint32(
                255
            )
        ]
        +
        POPCOUNT8[
            (
                values
                >> np.uint32(
                    16
                )
            )
            & np.uint32(
                255
            )
        ]
    )


def _transform_masks(
    lut: np.ndarray,
    move_ids: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:

    masks = np.asarray(
        masks,
        dtype=np.uint32,
    )


    move_ids = np.asarray(
        move_ids,
        dtype=np.intp,
    )


    byte0 = (
        masks
        &
        np.uint32(
            255
        )
    ).astype(
        np.intp,
        copy=False,
    )


    byte1 = (
        (
            masks
            >> np.uint32(
                8
            )
        )
        &
        np.uint32(
            255
        )
    ).astype(
        np.intp,
        copy=False,
    )


    byte2 = (
        (
            masks
            >> np.uint32(
                16
            )
        )
        &
        np.uint32(
            255
        )
    ).astype(
        np.intp,
        copy=False,
    )


    return (
        lut[
            move_ids,
            0,
            byte0,
        ]
        |
        lut[
            move_ids,
            1,
            byte1,
        ]
        |
        lut[
            move_ids,
            2,
            byte2,
        ]
    )


def compile_flat_mdd(
    language: SymbolicMDD,
) -> FlatMDD:

    started = perf_counter()


    nodes = language.arena.nodes

    node_count = len(
        nodes
    )


    horizon = int(
        language.horizon
    )


    degree = np.zeros(
        node_count,
        dtype=np.uint8,
    )


    edge_counts = np.zeros(
        horizon + 1,
        dtype=np.int64,
    )


    for nid in range(
        2,
        node_count,
    ):

        node = nodes[
            nid
        ]


        remaining = int(
            node.remaining
        )


        if not (
            1
            <= remaining
            <= horizon
        ):

            raise RuntimeError(
                "nonterminal layer drift"
            )


        deg = len(
            node.edges
        )


        if deg > 18:

            raise RuntimeError(
                "node degree > 18"
            )


        degree[
            nid
        ] = deg


        edge_counts[
            remaining
        ] += deg


    raw_parent = [
        None
    ] * (
        horizon + 1
    )


    raw_child = [
        None
    ] * (
        horizon + 1
    )


    raw_mid = [
        None
    ] * (
        horizon + 1
    )


    for remaining in range(
        1,
        horizon + 1,
    ):

        count = int(
            edge_counts[
                remaining
            ]
        )


        raw_parent[
            remaining
        ] = np.empty(
            count,
            dtype=np.int32,
        )


        raw_child[
            remaining
        ] = np.empty(
            count,
            dtype=np.int32,
        )


        raw_mid[
            remaining
        ] = np.empty(
            count,
            dtype=np.uint8,
        )


    positions = np.zeros(
        horizon + 1,
        dtype=np.int64,
    )


    edge_count = 0


    for nid in range(
        2,
        node_count,
    ):

        node = nodes[
            nid
        ]


        remaining = int(
            node.remaining
        )


        previous_mid = -1


        for mid, child in node.edges:

            mid = int(
                mid
            )

            child = int(
                child
            )


            if (
                mid <= previous_mid
                or
                not 0
                <= mid
                < 18
                or
                not 0
                <= child
                < nid
            ):

                raise RuntimeError(
                    "SymbolicMDD edge contract drift"
                )


            if (
                child >= 2
                and
                int(
                    nodes[
                        child
                    ].remaining
                )
                !=
                remaining - 1
            ):

                raise RuntimeError(
                    "SymbolicMDD layer contract drift"
                )


            previous_mid = mid


            position = int(
                positions[
                    remaining
                ]
            )


            raw_parent[
                remaining
            ][
                position
            ] = nid


            raw_child[
                remaining
            ][
                position
            ] = child


            raw_mid[
                remaining
            ][
                position
            ] = mid


            positions[
                remaining
            ] += 1


            edge_count += 1


    layers = [
        None
    ] * (
        horizon + 1
    )


    for remaining in range(
        1,
        horizon + 1,
    ):

        parents = raw_parent[
            remaining
        ]

        children = raw_child[
            remaining
        ]

        mids = raw_mid[
            remaining
        ]


        if len(
            parents
        ) == 0:

            layers[
                remaining
            ] = FlatLayer(
                remaining=
                    remaining,

                parent_unique=
                    np.empty(
                        0,
                        dtype=np.int32,
                    ),

                parent_starts=
                    np.empty(
                        0,
                        dtype=np.int64,
                    ),

                back_child=
                    children,

                back_mid=
                    mids,

                child_unique=
                    np.empty(
                        0,
                        dtype=np.int32,
                    ),

                child_starts=
                    np.empty(
                        0,
                        dtype=np.int64,
                    ),

                forward_parent=
                    parents,

                forward_mid=
                    mids,
            )

            continue


        parent_starts = np.concatenate(
            (
                np.asarray(
                    [
                        0
                    ],
                    dtype=np.int64,
                ),

                np.flatnonzero(
                    parents[
                        1:
                    ]
                    !=
                    parents[
                        :-1
                    ]
                ).astype(
                    np.int64
                )
                + 1,
            )
        )


        parent_unique = parents[
            parent_starts
        ].copy()


        child_order = np.argsort(
            children,
            kind="stable",
        )


        forward_child = children[
            child_order
        ]


        forward_parent = parents[
            child_order
        ]


        forward_mid = mids[
            child_order
        ]


        child_starts = np.concatenate(
            (
                np.asarray(
                    [
                        0
                    ],
                    dtype=np.int64,
                ),

                np.flatnonzero(
                    forward_child[
                        1:
                    ]
                    !=
                    forward_child[
                        :-1
                    ]
                ).astype(
                    np.int64
                )
                + 1,
            )
        )


        child_unique = forward_child[
            child_starts
        ].copy()


        layers[
            remaining
        ] = FlatLayer(
            remaining=
                remaining,

            parent_unique=
                parent_unique,

            parent_starts=
                parent_starts,

            back_child=
                children,

            back_mid=
                mids,

            child_unique=
                child_unique,

            child_starts=
                child_starts,

            forward_parent=
                forward_parent,

            forward_mid=
                forward_mid,
        )


    return FlatMDD(
        language=
            language,

        layers=
            tuple(
                layers
            ),

        degree=
            degree,

        node_count=
            node_count,

        edge_count=
            edge_count,

        build_wall_s=(
            perf_counter()
            - started
        ),
    )


def _suffix_masks(
    flat: FlatMDD,
    bank: VectorizedOneQuoteBank,

    *,
    piece: int,
    target_mask: int,
):

    started = perf_counter()


    target = checked_target_mask(
        int(
            target_mask
        )
    )


    masks = np.zeros(
        flat.node_count,
        dtype=np.uint32,
    )


    masks[
        1
    ] = np.uint32(
        target
    )


    lut = bank.pre_lut[
        int(
            piece
        )
    ]


    for remaining in range(
        1,
        int(
            flat.language.horizon
        )
        + 1,
    ):

        layer = flat.layers[
            remaining
        ]


        if (
            layer is None
            or
            len(
                layer.back_child
            ) == 0
        ):

            continue


        child_masks = masks[
            layer.back_child
        ]


        transformed = _transform_masks(
            lut,
            layer.back_mid,
            child_masks,
        )


        reduced = np.bitwise_or.reduceat(
            transformed,
            layer.parent_starts,
        )


        masks[
            layer.parent_unique
        ] = reduced


    return (
        masks,
        perf_counter()
        - started,
    )


def _forward_and_live(
    flat: FlatMDD,
    bank: VectorizedOneQuoteBank,

    *,
    piece: int,
    initial_q: int,
    suffix_masks: np.ndarray,
):

    started = perf_counter()


    q0 = int(
        initial_q
    )


    if not 0 <= q0 < 24:

        raise ValueError(
            "initial_q must be 0..23"
        )


    forward = np.zeros(
        flat.node_count,
        dtype=np.uint32,
    )


    forward[
        int(
            flat.language.root
        )
    ] = np.uint32(
        1 << q0
    )


    lut = bank.image_lut[
        int(
            piece
        )
    ]


    for remaining in range(
        int(
            flat.language.horizon
        ),
        0,
        -1,
    ):

        layer = flat.layers[
            remaining
        ]


        if (
            layer is None
            or
            len(
                layer.forward_parent
            ) == 0
        ):

            continue


        source_masks = forward[
            layer.forward_parent
        ]


        transformed = _transform_masks(
            lut,
            layer.forward_mid,
            source_masks,
        )


        reduced = np.bitwise_or.reduceat(
            transformed,
            layer.child_starts,
        )


        forward[
            layer.child_unique
        ] |= reduced


    live = (
        forward
        &
        suffix_masks
    )


    counts = _popcount_u32(
        live[
            2:
        ]
    )


    exact_states = int(
        counts.astype(
            np.uint64
        ).sum(
            dtype=np.uint64
        )
    )


    exact_edges = int(
        (
            counts.astype(
                np.uint64
            )
            *
            flat.degree[
                2:
            ].astype(
                np.uint64
            )
        ).sum(
            dtype=np.uint64
        )
    )


    forward_counts = _popcount_u32(
        forward[
            2:
        ]
    )


    forward_pairs = int(
        forward_counts.astype(
            np.uint64
        ).sum(
            dtype=np.uint64
        )
    )


    root = int(
        flat.language.root
    )


    root_live = bool(
        int(
            live[
                root
            ]
        )
        &
        (
            1 << q0
        )
    )


    return (
        forward,
        live,
        exact_states,
        exact_edges,
        forward_pairs,
        root_live,
        perf_counter()
        - started,
    )


def _u32_array(
    values: np.ndarray,
) -> array:

    out = array(
        "I"
    )


    if out.itemsize != 4:

        raise RuntimeError(
            "array('I') is not uint32"
        )


    contiguous = np.ascontiguousarray(
        values,
        dtype=np.uint32,
    )


    out.frombytes(
        contiguous.tobytes()
    )


    return out


def quote_vectorized_one_candidates(
    language: SymbolicMDD,

    *,
    pieces: Iterable[int],

    bank: VectorizedOneQuoteBank,

    q_before,
    targets,

) -> tuple[
    int,
    dict[str, Any],
]:
    """
    Exact workload quote for every candidate.

    Returns the exact-min-work piece id.

    No output MDD is materialized.
    """

    started = perf_counter()


    flat = compile_flat_mdd(
        language
    )


    rows = []


    normalized = tuple(
        sorted(
            {
                int(v)
                for v in pieces
            }
        )
    )


    for piece in normalized:

        suffix, suffix_wall = _suffix_masks(
            flat,
            bank,

            piece=
                piece,

            target_mask=
                int(
                    targets[
                        piece
                    ]
                ),
        )


        (
            _forward,
            _live,
            exact_states,
            exact_edges,
            forward_pairs,
            root_live,
            forward_wall,
        ) = _forward_and_live(
            flat,
            bank,

            piece=
                piece,

            initial_q=
                int(
                    q_before[
                        piece
                    ]
                ),

            suffix_masks=
                suffix,
        )


        coreach_counts = _popcount_u32(
            suffix[
                2:
            ]
        )


        coreach_pairs = (
            int(
                int(
                    targets[
                        piece
                    ]
                ).bit_count()
            )
            +
            int(
                coreach_counts.astype(
                    np.uint64
                ).sum(
                    dtype=np.uint64
                )
            )
        )


        rows.append({
            "piece":
                piece,

            "root_live":
                root_live,

            "exact_states":
                exact_states,

            "exact_edges":
                exact_edges,

            "coreach_pairs":
                coreach_pairs,

            "forward_pairs":
                forward_pairs,

            "suffix_wall_s":
                suffix_wall,

            "forward_wall_s":
                forward_wall,

            "candidate_wall_s":(
                suffix_wall
                + forward_wall
            ),
        })


    dead = [
        row[
            "piece"
        ]
        for row in rows
        if not row[
            "root_live"
        ]
    ]


    if dead:

        selected_piece = int(
            min(
                dead
            )
        )


    else:

        selected_piece = int(
            min(
                rows,
                key=lambda row:(
                    int(
                        row[
                            "exact_edges"
                        ]
                    ),

                    int(
                        row[
                            "exact_states"
                        ]
                    ),

                    int(
                        row[
                            "piece"
                        ]
                    ),
                ),
            )[
                "piece"
            ]
        )


    return (
        selected_piece,

        {
            "schema":
                "cubelab.vectorized-one-quote.v1",

            "bank_build_wall_s":
                float(
                    bank.build_wall_s
                ),

            "flat_build_wall_s":
                float(
                    flat.build_wall_s
                ),

            "candidate_count":
                len(
                    rows
                ),

            "selected_piece":
                selected_piece,

            "dead_pieces":
                dead,

            "rows":
                rows,

            "wall_s":(
                perf_counter()
                - started
            ),
        },
    )


def materialize_vectorized_one_quote(
    language: SymbolicMDD,

    *,
    piece: int,

    bank: VectorizedOneQuoteBank,

    q_before,
    targets,

) -> tuple[
    ExactOneQuote,
    dict[str, Any],
]:
    """
    Recompute ONLY the already-selected candidate and convert
    its exact masks into the existing PreparedPieceMask /
    OneLiveEstimate contracts.
    """

    started = perf_counter()


    flat = compile_flat_mdd(
        language
    )


    piece = int(
        piece
    )


    suffix, suffix_wall = _suffix_masks(
        flat,
        bank,

        piece=
            piece,

        target_mask=
            int(
                targets[
                    piece
                ]
            ),
    )


    (
        forward,
        live,
        exact_states,
        exact_edges,
        forward_pairs,
        root_live,
        forward_wall,
    ) = _forward_and_live(
        flat,
        bank,

        piece=
            piece,

        initial_q=
            int(
                q_before[
                    piece
                ]
            ),

        suffix_masks=
            suffix,
    )


    coreach_counts = _popcount_u32(
        suffix[
            2:
        ]
    )


    coreach_pairs = (
        int(
            int(
                targets[
                    piece
                ]
            ).bit_count()
        )
        +
        int(
            coreach_counts.astype(
                np.uint64
            ).sum(
                dtype=np.uint64
            )
        )
    )


    transition = checked_transition(
        bank.q_next[
            piece
        ]
    )


    prepared = PreparedPieceMask(
        language=
            language,

        transition=
            transition,

        transition_rows=
            bank.transition_rows(
                piece
            ),

        initial_q=
            int(
                q_before[
                    piece
                ]
            ),

        target_mask=
            int(
                targets[
                    piece
                ]
            ),

        masks=
            _u32_array(
                suffix
            ),

        coreach_live_state_pairs=
            int(
                coreach_pairs
            ),

        root_live=
            bool(
                root_live
            ),

        prepare_wall_s=
            float(
                suffix_wall
            ),

        metrics={
            "schema":
                "cubelab.vectorized-selected-mask.v1",

            "flat_build_wall_s":
                float(
                    flat.build_wall_s
                ),

            "suffix_wall_s":
                float(
                    suffix_wall
                ),
        },
    )


    estimate = OneLiveEstimate(
        prepared=
            prepared,

        forward_masks=
            _u32_array(
                forward
            ),

        live_masks=
            _u32_array(
                live
            ),

        forward_state_pairs=
            int(
                forward_pairs
            ),

        exact_product_states=
            int(
                exact_states
            ),

        exact_edge_examinations=
            int(
                exact_edges
            ),

        root_live=
            bool(
                root_live
            ),

        estimator_wall_s=
            float(
                forward_wall
            ),

        metrics={
            "schema":
                "cubelab.vectorized-selected-live.v1",

            "exact_product_states":
                int(
                    exact_states
                ),

            "exact_edge_examinations":
                int(
                    exact_edges
                ),

            "forward_state_pairs":
                int(
                    forward_pairs
                ),

            "wall_s":
                float(
                    forward_wall
                ),
        },
    )


    quote = ExactOneQuote(
        piece=
            piece,

        prepared=
            prepared,

        estimate=
            estimate,
    )


    return (
        quote,

        {
            "piece":
                piece,

            "exact_states":
                exact_states,

            "exact_edges":
                exact_edges,

            "flat_build_wall_s":
                float(
                    flat.build_wall_s
                ),

            "suffix_wall_s":
                float(
                    suffix_wall
                ),

            "forward_wall_s":
                float(
                    forward_wall
                ),

            "wall_s":(
                perf_counter()
                - started
            ),
        },
    )


__all__ = [
    "FlatLayer",
    "FlatMDD",
    "VectorizedOneQuoteBank",
    "compile_flat_mdd",
    "materialize_vectorized_one_quote",
    "quote_vectorized_one_candidates",
]


# ==============================================================
# Production helpers
# ==============================================================

def quote_vectorized_one_candidates_prepared(
    language: SymbolicMDD,

    *,
    pieces: Iterable[int],

    bank: VectorizedOneQuoteBank,

    q_before,
    targets,

) -> tuple[
    ExactOneQuote,
    dict[str, Any],
]:
    """
    Exact-min-work ONE quote.

    Unlike quote_vectorized_one_candidates(), the already-computed
    masks for the selected candidate are RETAINED and converted
    directly into PreparedPieceMask / OneLiveEstimate.

    Therefore the selected candidate does not require a second
    flat-MDD/suffix/forward materialization pass.
    """

    started = perf_counter()


    flat = compile_flat_mdd(
        language
    )


    normalized = tuple(
        sorted(
            {
                int(v)
                for v in pieces
            }
        )
    )


    if not normalized:

        raise ValueError(
            "no candidate pieces"
        )


    rows = []


    selected_data = None
    selected_key = None


    for piece in normalized:

        suffix, suffix_wall = _suffix_masks(
            flat,
            bank,

            piece=
                piece,

            target_mask=
                int(
                    targets[
                        piece
                    ]
                ),
        )


        (
            forward,
            live,
            exact_states,
            exact_edges,
            forward_pairs,
            root_live,
            forward_wall,
        ) = _forward_and_live(
            flat,
            bank,

            piece=
                piece,

            initial_q=
                int(
                    q_before[
                        piece
                    ]
                ),

            suffix_masks=
                suffix,
        )


        coreach_counts = _popcount_u32(
            suffix[
                2:
            ]
        )


        coreach_pairs = (
            int(
                int(
                    targets[
                        piece
                    ]
                ).bit_count()
            )
            +
            int(
                coreach_counts.astype(
                    np.uint64
                ).sum(
                    dtype=np.uint64
                )
            )
        )


        row = {
            "piece":
                piece,

            "root_live":
                bool(
                    root_live
                ),

            "exact_states":
                int(
                    exact_states
                ),

            "exact_edges":
                int(
                    exact_edges
                ),

            "coreach_pairs":
                int(
                    coreach_pairs
                ),

            "forward_pairs":
                int(
                    forward_pairs
                ),

            "suffix_wall_s":
                float(
                    suffix_wall
                ),

            "forward_wall_s":
                float(
                    forward_wall
                ),
        }


        rows.append(
            row
        )


        # Any exact-dead mandatory condition has priority.
        # Otherwise choose minimum exact edge-work.
        if not root_live:

            key = (
                0,
                piece,
            )


        else:

            key = (
                1,
                int(
                    exact_edges
                ),
                int(
                    exact_states
                ),
                piece,
            )


        if (
            selected_key is None
            or key < selected_key
        ):

            selected_key = key


            selected_data = (
                piece,
                suffix,
                forward,
                live,
                exact_states,
                exact_edges,
                forward_pairs,
                root_live,
                coreach_pairs,
                suffix_wall,
                forward_wall,
            )


    if selected_data is None:

        raise RuntimeError(
            "failed to select ONE candidate"
        )


    (
        piece,
        suffix,
        forward,
        live,
        exact_states,
        exact_edges,
        forward_pairs,
        root_live,
        coreach_pairs,
        suffix_wall,
        forward_wall,
    ) = selected_data


    transition = checked_transition(
        bank.q_next[
            piece
        ]
    )


    prepared = PreparedPieceMask(
        language=
            language,

        transition=
            transition,

        transition_rows=
            bank.transition_rows(
                piece
            ),

        initial_q=
            int(
                q_before[
                    piece
                ]
            ),

        target_mask=
            int(
                targets[
                    piece
                ]
            ),

        masks=
            _u32_array(
                suffix
            ),

        coreach_live_state_pairs=
            int(
                coreach_pairs
            ),

        root_live=
            bool(
                root_live
            ),

        prepare_wall_s=
            float(
                suffix_wall
            ),

        metrics={
            "schema":
                "cubelab.vector-selected-prepared.v1",

            "suffix_wall_s":
                float(
                    suffix_wall
                ),

            "flat_build_wall_s":
                float(
                    flat.build_wall_s
                ),
        },
    )


    estimate = OneLiveEstimate(
        prepared=
            prepared,

        forward_masks=
            _u32_array(
                forward
            ),

        live_masks=
            _u32_array(
                live
            ),

        forward_state_pairs=
            int(
                forward_pairs
            ),

        exact_product_states=
            int(
                exact_states
            ),

        exact_edge_examinations=
            int(
                exact_edges
            ),

        root_live=
            bool(
                root_live
            ),

        estimator_wall_s=
            float(
                forward_wall
            ),

        metrics={
            "schema":
                "cubelab.vector-selected-estimate.v1",

            "exact_product_states":
                int(
                    exact_states
                ),

            "exact_edge_examinations":
                int(
                    exact_edges
                ),

            "wall_s":
                float(
                    forward_wall
                ),
        },
    )


    quote = ExactOneQuote(
        piece=
            int(
                piece
            ),

        prepared=
            prepared,

        estimate=
            estimate,
    )


    rows_sorted = sorted(
        rows,

        key=lambda row:(
            0
            if not row[
                "root_live"
            ]
            else 1,

            int(
                row[
                    "exact_edges"
                ]
            ),

            int(
                row[
                    "exact_states"
                ]
            ),

            int(
                row[
                    "piece"
                ]
            ),
        )
    )


    return (
        quote,

        {
            "schema":
                "cubelab.vectorized-one-quote-prepared.v1",

            "flat_build_wall_s":
                float(
                    flat.build_wall_s
                ),

            "candidate_count":
                len(
                    rows
                ),

            "selected_piece":
                int(
                    piece
                ),

            "rows":
                rows,

            "top5":
                rows_sorted[
                    :5
                ],

            "wall_s":(
                perf_counter()
                - started
            ),
        },
    )


def prepare_vectorized_piece_masks(
    language: SymbolicMDD,

    *,
    pieces: Iterable[int],

    bank: VectorizedOneQuoteBank,

    q_before,
    targets,

) -> tuple[
    dict[int, PreparedPieceMask],
    dict[str, Any],
]:
    """
    Suffix-only vectorized preparation for a FIXED direct execution
    plan such as the frozen hybrid.

    No forward estimator is computed because execution order is
    already fixed.
    """

    started = perf_counter()


    flat = compile_flat_mdd(
        language
    )


    result = {}

    rows = []


    for piece in tuple(
        sorted(
            {
                int(v)
                for v in pieces
            }
        )
    ):

        suffix, suffix_wall = _suffix_masks(
            flat,
            bank,

            piece=
                piece,

            target_mask=
                int(
                    targets[
                        piece
                    ]
                ),
        )


        root = int(
            language.root
        )


        q0 = int(
            q_before[
                piece
            ]
        )


        root_live = bool(
            int(
                suffix[
                    root
                ]
            )
            &
            (
                1 << q0
            )
        )


        coreach_counts = _popcount_u32(
            suffix[
                2:
            ]
        )


        coreach_pairs = (
            int(
                int(
                    targets[
                        piece
                    ]
                ).bit_count()
            )
            +
            int(
                coreach_counts.astype(
                    np.uint64
                ).sum(
                    dtype=np.uint64
                )
            )
        )


        transition = checked_transition(
            bank.q_next[
                piece
            ]
        )


        prepared = PreparedPieceMask(
            language=
                language,

            transition=
                transition,

            transition_rows=
                bank.transition_rows(
                    piece
                ),

            initial_q=
                q0,

            target_mask=
                int(
                    targets[
                        piece
                    ]
                ),

            masks=
                _u32_array(
                    suffix
                ),

            coreach_live_state_pairs=
                int(
                    coreach_pairs
                ),

            root_live=
                root_live,

            prepare_wall_s=
                float(
                    suffix_wall
                ),

            metrics={
                "schema":
                    "cubelab.vectorized-suffix-only.v1",

                "suffix_wall_s":
                    float(
                        suffix_wall
                    ),

                "flat_build_wall_s":
                    float(
                        flat.build_wall_s
                    ),
            },
        )


        result[
            piece
        ] = prepared


        rows.append({
            "piece":
                piece,

            "root_live":
                root_live,

            "suffix_wall_s":
                float(
                    suffix_wall
                ),

            "coreach_pairs":
                int(
                    coreach_pairs
                ),
        })


    return (
        result,

        {
            "schema":
                "cubelab.vectorized-piece-mask-batch.v1",

            "flat_build_wall_s":
                float(
                    flat.build_wall_s
                ),

            "piece_count":
                len(
                    result
                ),

            "rows":
                rows,

            "wall_s":(
                perf_counter()
                - started
            ),
        },
    )
