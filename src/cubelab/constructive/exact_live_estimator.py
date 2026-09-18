from __future__ import annotations

from array import array
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from cubelab.column_families.symbolic_mdd_v2 import (
    SymbolicMDD,
)

from .direct_q_restrict import (
    DirectRestrictUnknown,
)

from .prepared_q_restrict import (
    PreparedPairMask,
    PreparedPieceMask,
)


# ==============================================================
# Exact forward-image lookup
# ==============================================================

def image_tables(
    prepared: PreparedPieceMask,
    *,
    deadline: float | None = None,
) -> tuple[array, ...]:

    started = perf_counter()

    dl = (
        float("inf")
        if deadline is None
        else float(deadline)
    )

    transition = prepared.transition

    tables = []


    for mid in range(18):

        if perf_counter() >= dl:
            raise DirectRestrictUnknown(
                "UNKNOWN_TIME: image_tables",
                {
                    "phase":
                        "IMAGE_TABLES",

                    "wall_s":
                        perf_counter()
                        - started,
                },
            )


        for byte in range(3):

            base = (
                8 * byte
            )


            tab = array(
                "I",
                [0],
            ) * 256


            for bits in range(
                1,
                256,
            ):

                low = (
                    bits
                    & -bits
                )


                source = (
                    base
                    + low.bit_length()
                    - 1
                )


                destination = int(
                    transition[
                        source,
                        mid,
                    ]
                )


                tab[
                    bits
                ] = (
                    tab[
                        bits
                        ^ low
                    ]
                    |
                    (
                        1
                        << destination
                    )
                )


            tables.append(
                tab
            )


    return tuple(
        tables
    )


def image_mask(
    mask: int,
    move_id: int,
    tables,
) -> int:

    index = (
        3
        * int(
            move_id
        )
    )


    return int(
        tables[
            index
        ][
            mask
            & 255
        ]
        |
        tables[
            index + 1
        ][
            (
                mask
                >> 8
            )
            & 255
        ]
        |
        tables[
            index + 2
        ][
            mask
            >> 16
        ]
    )


# ==============================================================
# ONE exact estimate
# ==============================================================

@dataclass(
    frozen=True,
    slots=True,
)
class OneLiveEstimate:

    prepared: PreparedPieceMask

    forward_masks: array
    live_masks: array

    forward_state_pairs: int

    exact_product_states: int
    exact_edge_examinations: int

    root_live: bool

    estimator_wall_s: float

    metrics: dict[str, Any]


def estimate_prepared_one(
    prepared: PreparedPieceMask,
    *,
    deadline: float | None = None,
) -> OneLiveEstimate:

    started = perf_counter()

    dl = (
        float("inf")
        if deadline is None
        else float(deadline)
    )


    language = prepared.language

    nodes = language.arena.nodes


    tables = image_tables(
        prepared,
        deadline=
            dl,
    )


    forward = array(
        "I",
        [0],
    ) * len(
        nodes
    )


    live = array(
        "I",
        [0],
    ) * len(
        nodes
    )


    root = int(
        language.root
    )


    forward[
        root
    ] = (
        1
        << int(
            prepared.initial_q
        )
    )


    propagated_edges = 0


    # NodeArena contract:
    # every child id is smaller than its parent id.
    #
    # Therefore descending nid is exact topological
    # forward propagation even when unused arena nodes exist.
    for nid in range(
        len(nodes) - 1,
        1,
        -1,
    ):

        if (
            nid
            & 4095
        ) == 0:

            if perf_counter() >= dl:

                raise DirectRestrictUnknown(
                    "UNKNOWN_TIME: forward live estimator",
                    {
                        "phase":
                            "FORWARD_MASK",

                        "nid":
                            nid,

                        "wall_s":
                            perf_counter()
                            - started,
                    },
                )


        source_mask = int(
            forward[
                nid
            ]
        )


        if source_mask == 0:
            continue


        node = nodes[
            nid
        ]


        for mid, child in node.edges:

            child = int(
                child
            )


            if child >= nid:

                raise RuntimeError(
                    "SymbolicMDD topological "
                    "child-id contract drift"
                )


            forward[
                child
            ] |= image_mask(
                source_mask,
                int(
                    mid
                ),
                tables,
            )


            propagated_edges += 1


    forward_pairs = 0

    exact_states = 0
    exact_edge_work = 0


    for nid in range(
        2,
        len(
            nodes
        ),
    ):

        fmask = int(
            forward[
                nid
            ]
        )


        if fmask == 0:
            continue


        forward_count = (
            fmask.bit_count()
        )


        forward_pairs += (
            forward_count
        )


        lmask = (
            fmask
            &
            int(
                prepared.masks[
                    nid
                ]
            )
        )


        live[
            nid
        ] = lmask


        count = int(
            lmask.bit_count()
        )


        if count == 0:
            continue


        exact_states += count


        exact_edge_work += (
            count
            * len(
                nodes[
                    nid
                ].edges
            )
        )


    root_live = bool(
        int(
            live[
                root
            ]
        )
        &
        (
            1
            << int(
                prepared.initial_q
            )
        )
    )


    wall = (
        perf_counter()
        - started
    )


    metrics = {
        "schema":
            "cubelab.forward-live-estimator.v1",

        "forward_state_pairs":
            int(
                forward_pairs
            ),

        "exact_product_states":
            int(
                exact_states
            ),

        "exact_edge_examinations":
            int(
                exact_edge_work
            ),

        "suffix_coreach_pairs":
            int(
                prepared
                .coreach_live_state_pairs
            ),

        "propagated_DAG_edges":
            int(
                propagated_edges
            ),

        "root_live":
            root_live,

        "wall_s":
            wall,
    }


    return OneLiveEstimate(
        prepared=
            prepared,

        forward_masks=
            forward,

        live_masks=
            live,

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
                exact_edge_work
            ),

        root_live=
            root_live,

        estimator_wall_s=
            wall,

        metrics=
            metrics,
    )


# ==============================================================
# TWO tight marginal upper
# ==============================================================

@dataclass(
    frozen=True,
    slots=True,
)
class PairLiveEstimate:

    left: OneLiveEstimate
    right: OneLiveEstimate

    marginal_reachable_state_upper: int
    marginal_reachable_edge_upper: int

    suffix_only_state_upper: int

    tightening_ratio: float

    estimator_wall_s: float

    metrics: dict[str, Any]


def estimate_prepared_pair(
    prepared_pair: PreparedPairMask,

    left: OneLiveEstimate,
    right: OneLiveEstimate,

    *,
    deadline: float | None = None,
) -> PairLiveEstimate:

    if (
        prepared_pair.left
        is not left.prepared
        or
        prepared_pair.right
        is not right.prepared
    ):

        raise ValueError(
            "prepared-pair/live-estimate mismatch"
        )


    language = (
        prepared_pair
        .left
        .language
    )


    if (
        language
        is not
        prepared_pair
        .right
        .language
    ):

        raise ValueError(
            "pair language mismatch"
        )


    started = perf_counter()


    dl = (
        float("inf")
        if deadline is None
        else float(deadline)
    )


    nodes = language.arena.nodes


    state_upper = 0
    edge_upper = 0


    for nid in range(
        2,
        len(
            nodes
        ),
    ):

        if (
            nid
            & 8191
        ) == 0:

            if perf_counter() >= dl:

                raise DirectRestrictUnknown(
                    "UNKNOWN_TIME: pair forward-live upper",
                    {
                        "phase":
                            "PAIR_FORWARD_LIVE",

                        "nid":
                            nid,

                        "wall_s":
                            perf_counter()
                            - started,
                    },
                )


        left_count = int(
            left.live_masks[
                nid
            ]
        ).bit_count()


        if left_count == 0:
            continue


        right_count = int(
            right.live_masks[
                nid
            ]
        ).bit_count()


        if right_count == 0:
            continue


        product = (
            left_count
            * right_count
        )


        state_upper += (
            product
        )


        edge_upper += (
            product
            * len(
                nodes[
                    nid
                ].edges
            )
        )


    suffix_only = int(
        prepared_pair
        .marginal_joint_live_upper
    )


    tightening = (
        0.0
        if suffix_only <= 0
        else (
            float(
                state_upper
            )
            /
            float(
                suffix_only
            )
        )
    )


    wall = (
        perf_counter()
        - started
    )


    metrics = {
        "schema":
            "cubelab.forward-live-pair-estimator.v1",

        "marginal_reachable_state_upper":
            int(
                state_upper
            ),

        "marginal_reachable_edge_upper":
            int(
                edge_upper
            ),

        "suffix_only_state_upper":
            suffix_only,

        "tightening_ratio":
            tightening,

        "wall_s":
            wall,
    }


    return PairLiveEstimate(
        left=
            left,

        right=
            right,

        marginal_reachable_state_upper=
            int(
                state_upper
            ),

        marginal_reachable_edge_upper=
            int(
                edge_upper
            ),

        suffix_only_state_upper=
            suffix_only,

        tightening_ratio=
            tightening,

        estimator_wall_s=
            wall,

        metrics=
            metrics,
    )


__all__ = [
    "OneLiveEstimate",
    "PairLiveEstimate",
    "estimate_prepared_one",
    "estimate_prepared_pair",
]
