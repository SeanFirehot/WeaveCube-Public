from __future__ import annotations

from array import array
from time import perf_counter
from typing import Any

import numpy as np

from cubelab.column_families.symbolic_mdd_v2 import (
    NodeArena,
    ResourceCapUnknown,
    SymbolicMDD,
)


FULL24 = (1 << 24) - 1

ONE_TAG = "V374_DIRECT_TARGETMASK_Q24_ONE"
TWO_TAG = "V374_DIRECT_TARGETMASK_Q24_TWO"


class DirectRestrictUnknown(ResourceCapUnknown):
    """
    Exact operation stopped on a resource limit.

    No partial language is returned.
    """

    def __init__(
        self,
        message: str,
        metrics: dict[str, Any],
    ):
        super().__init__(message)
        self.metrics = dict(metrics)


def _deadline_value(
    deadline: float | None,
) -> float:

    return (
        float("inf")
        if deadline is None
        else float(deadline)
    )


def _check_deadline(
    deadline: float,
    message: str,
    metrics: dict[str, Any],
) -> None:

    if perf_counter() >= deadline:

        raise DirectRestrictUnknown(
            message,
            metrics,
        )


def checked_transition(
    transition: np.ndarray,
) -> np.ndarray:

    tr = np.asarray(
        transition
    )

    if (
        tr.shape != (24, 18)
        or
        not np.issubdtype(
            tr.dtype,
            np.integer,
        )
        or
        np.any(
            tr < 0
        )
        or
        np.any(
            tr >= 24
        )
    ):

        raise ValueError(
            "exact deterministic 24x18 "
            "piece transition required"
        )

    return tr


def checked_target_mask(
    target_mask: int,
) -> int:

    if (
        type(target_mask) is not int
        or
        not 0
        <= target_mask
        <= FULL24
    ):

        raise ValueError(
            "target_mask must be a "
            "24-bit integer set"
        )

    return int(
        target_mask
    )


def checked_initial_q(
    initial_q: int,
) -> int:

    if (
        type(initial_q) is not int
        or
        not 0
        <= initial_q
        < 24
    ):

        raise ValueError(
            "initial_q must be an integer "
            "in 0..23"
        )

    return int(
        initial_q
    )


# ==============================================================
# Exact deterministic preimage
# ==============================================================

def preimage_tables(
    transition: np.ndarray,
    *,
    deadline: float | None = None,
    metrics: dict[str, Any] | None = None,
) -> tuple[array, ...]:
    """
    18 moves × 3 bytes × 256 uint32 lookup rows.

    Works for any deterministic 24-state transition,
    including non-injective transitions.

    For one destination-Q mask T:

        preimage(T, move)

    returns every source Q whose move-successor is in T.
    """

    tr = checked_transition(
        transition
    )

    dl = _deadline_value(
        deadline
    )

    out_metrics = (
        {}
        if metrics is None
        else metrics
    )


    if array(
        "I"
    ).itemsize != 4:

        raise RuntimeError(
            "direct restriction requires "
            "32-bit array('I')"
        )


    t0 = perf_counter()

    tables: list[array] = []


    for mid in range(
        18
    ):

        _check_deadline(
            dl,
            "UNKNOWN_TIME: "
            "preimage table construction",
            out_metrics,
        )


        by_destination = [
            0
        ] * 24


        for q in range(
            24
        ):

            by_destination[
                int(
                    tr[
                        q,
                        mid,
                    ]
                )
            ] |= (
                1 << q
            )


        for byte in range(
            3
        ):

            tab = array(
                "I",
                [0],
            ) * 256

            base = (
                8 * byte
            )


            for bits in range(
                1,
                256,
            ):

                low = (
                    bits
                    & -bits
                )

                destination = (
                    base
                    + low.bit_length()
                    - 1
                )

                tab[
                    bits
                ] = (
                    tab[
                        bits
                        ^ low
                    ]
                    |
                    by_destination[
                        destination
                    ]
                )


            tables.append(
                tab
            )


    out_metrics[
        "preimage_table_wall_s"
    ] = (
        perf_counter()
        - t0
    )

    out_metrics[
        "preimage_table_payload_bytes"
    ] = (
        18
        * 3
        * 256
        * 4
    )


    return tuple(
        tables
    )


def preimage(
    mask: int,
    move_id: int,
    tables: tuple[
        array,
        ...
    ],
) -> int:

    mask = checked_target_mask(
        mask
    )


    if (
        type(move_id) is not int
        or
        not 0
        <= move_id
        < 18
    ):

        raise ValueError(
            "move_id must be an integer "
            "in 0..17"
        )


    index = (
        3
        * move_id
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
# Input DAG contract
# ==============================================================

def _validate_language(
    language: SymbolicMDD,
    *,
    input_node_cap: int,
):

    if (
        type(input_node_cap)
        is not int
        or
        input_node_cap < 2
    ):

        raise ValueError(
            "input_node_cap must be >= 2"
        )


    nodes = (
        language
        .arena
        .nodes
    )


    if len(
        nodes
    ) > input_node_cap:

        raise DirectRestrictUnknown(
            "UNKNOWN_INPUT_NODE_CAP",
            {
                "input_arena_nodes":
                    len(
                        nodes
                    ),

                "input_node_cap":
                    input_node_cap,
            },
        )


    if len(
        nodes
    ) < 2:

        raise RuntimeError(
            "SymbolicMDD arena is missing "
            "FALSE/TRUE terminals"
        )


    false_node = nodes[
        0
    ]

    true_node = nodes[
        1
    ]


    if (
        false_node.remaining,
        false_node.accepting,
        false_node.edges,
    ) != (
        0,
        False,
        (),
    ):

        raise RuntimeError(
            "unexpected SymbolicMDD "
            "FALSE terminal contract"
        )


    if (
        true_node.remaining,
        true_node.accepting,
        true_node.edges,
    ) != (
        0,
        True,
        (),
    ):

        raise RuntimeError(
            "unexpected SymbolicMDD "
            "TRUE terminal contract"
        )


    if (
        type(
            language.root
        ) is not int
        or
        not 0
        <= language.root
        < len(
            nodes
        )
    ):

        raise RuntimeError(
            "invalid SymbolicMDD root"
        )


    return nodes


# ==============================================================
# Exact arbitrary-target suffix coreach mask
# ==============================================================

def suffix_masks_target(
    language: SymbolicMDD,
    tables: tuple[
        array,
        ...
    ],
    target_mask: int,
    *,
    input_node_cap: int = 1_500_000,
    deadline: float | None = None,
    metrics: dict[str, Any] | None = None,
) -> array:
    """
    masks[nid] is the exact set of start-Q states
    from which SOME labelled path in this residual MDD
    reaches target_mask.

    This is an exact negative gate.
    It never chooses a word.
    """

    target = checked_target_mask(
        target_mask
    )

    dl = _deadline_value(
        deadline
    )

    out_metrics = (
        {}
        if metrics is None
        else metrics
    )


    nodes = _validate_language(
        language,
        input_node_cap=
            input_node_cap,
    )


    t0 = perf_counter()


    _check_deadline(
        dl,
        "UNKNOWN_TIME: "
        "before suffix target-mask DP",
        out_metrics,
    )


    masks = array(
        "I",
        [0],
    ) * len(
        nodes
    )


    # Generalization of the old Q=0 terminal:
    # TRUE terminal now admits every requested target Q.
    masks[
        1
    ] = target


    edge_count = 0

    live_state_pairs = int(
        target.bit_count()
    )


    for nid in range(
        2,
        len(
            nodes
        ),
    ):

        if (
            nid
            & 1023
        ) == 0:

            out_metrics.update(
                mask_nonterminal_rows=
                    nid - 2,

                mask_edges_examined=
                    edge_count,

                mask_live_state_pairs=
                    live_state_pairs,

                mask_dp_wall_s=(
                    perf_counter()
                    - t0
                ),
            )


            _check_deadline(
                dl,
                "UNKNOWN_TIME: "
                "inside suffix target-mask DP",
                out_metrics,
            )


        node = nodes[
            nid
        ]


        if (
            node.remaining < 1
            or
            node.accepting
            or
            not node.edges
        ):

            raise RuntimeError(
                "unproductive nonterminal "
                "in SymbolicMDD arena"
            )


        answer = 0

        previous_mid = -1


        for (
            mid,
            child,
        ) in node.edges:

            if (
                type(mid) is not int
                or
                type(child) is not int
                or
                not 0
                <= mid
                < 18
                or
                mid
                <= previous_mid
                or
                not 0
                < child
                < nid
                or
                nodes[
                    child
                ].remaining
                !=
                node.remaining
                - 1
            ):

                raise RuntimeError(
                    "direct restriction requires "
                    "ordered topological labelled DAG"
                )


            previous_mid = mid


            mask = int(
                masks[
                    child
                ]
            )


            index = (
                3
                * mid
            )


            answer |= int(
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


            edge_count += 1


        masks[
            nid
        ] = answer


        live_state_pairs += int(
            answer.bit_count()
        )


    if (
        language.root != 0
        and
        nodes[
            language.root
        ].remaining
        != language.horizon
    ):

        raise RuntimeError(
            "input root/horizon drift"
        )


    out_metrics.update(
        mask_input_arena_nodes=
            len(
                nodes
            ),

        mask_storage_bytes=
            len(
                masks
            )
            * 4,

        mask_nonterminal_rows=
            max(
                0,
                len(
                    nodes
                )
                - 2,
            ),

        mask_edges_examined=
            edge_count,

        mask_root_popcount=
            int(
                masks[
                    language.root
                ]
            ).bit_count(),

        mask_live_state_pairs=
            live_state_pairs,

        mask_target_popcount=
            target.bit_count(),

        mask_dp_wall_s=(
            perf_counter()
            - t0
        ),

        mask_semantics=(
            "EXACT_START_Q_SET_FOR_CURRENT_"
            "LABELLED_LANGUAGE_TO_ARBITRARY_TARGET_MASK"
        ),
    )


    _check_deadline(
        dl,
        "UNKNOWN_TIME: "
        "after suffix target-mask DP",
        out_metrics,
    )


    return masks


def _new_language(
    *,
    language: SymbolicMDD,
    arena: NodeArena,
    root: int,
    construction: str,
) -> SymbolicMDD:

    return SymbolicMDD(
        arena=
            arena,

        root=
            int(
                root
            ),

        horizon=
            language.horizon,

        previous_face=
            language.previous_face,

        scope_hash=
            language.scope_hash,

        construction=
            construction,
    )


# ==============================================================
# ONE raw-Q direct restriction
# ==============================================================

def restrict_one_targetmask(
    language: SymbolicMDD,
    transition: np.ndarray,
    initial_q: int,
    target_mask: int,
    *,
    node_cap: int = 1_500_000,
    input_node_cap: int | None = None,
    product_cap: int = 4_000_000,
    deadline: float | None = None,
) -> tuple[
    SymbolicMDD,
    dict[str, Any],
]:
    """
    Exact:

        C ∩ Q-language

    without materializing the standalone Q-language MDD.

    Product state:
        (current_MDD_node, Q)
    """

    q0 = checked_initial_q(
        initial_q
    )

    target = checked_target_mask(
        target_mask
    )


    if (
        type(node_cap)
        is not int
        or
        node_cap < 2
    ):

        raise ValueError(
            "node_cap must be >= 2"
        )


    if (
        type(product_cap)
        is not int
        or
        product_cap < 1
    ):

        raise ValueError(
            "product_cap must be >= 1"
        )


    input_cap = (
        node_cap
        if input_node_cap is None
        else int(
            input_node_cap
        )
    )


    dl = _deadline_value(
        deadline
    )


    tr = checked_transition(
        transition
    )


    started = perf_counter()


    metrics: dict[
        str,
        Any,
    ] = {
        "schema":
            "cubelab.direct-targetmask-restrict.v1",

        "arity":
            1,

        "target_mask":
            target,

        "target_popcount":
            target.bit_count(),

        "initial_q":
            q0,

        "product_cap":
            product_cap,

        "node_cap":
            node_cap,

        "partial_language_returned":
            False,

        "accepted_words_materialized":
            0,

        "intermediate_path_count_calls":
            0,

        "phase":
            "PREIMAGE_TABLES",
    }


    try:

        _check_deadline(
            dl,
            "UNKNOWN_TIME: "
            "before one-piece direct restriction",
            metrics,
        )


        tables = preimage_tables(
            tr,
            deadline=
                dl,
            metrics=
                metrics,
        )


        metrics[
            "phase"
        ] = (
            "SUFFIX_MASK_DP"
        )


        masks = suffix_masks_target(
            language,
            tables,
            target,

            input_node_cap=
                input_cap,

            deadline=
                dl,

            metrics=
                metrics,
        )


        nodes = (
            language
            .arena
            .nodes
        )


        tr_rows = tr.tolist()


        arena = NodeArena(
            node_cap=
                node_cap
        )


        memo: dict[
            int,
            int,
        ] = {}


        entered = 0
        calls = 0
        edges_examined = 0
        memo_hits = 0
        rejected = 0


        metrics[
            "phase"
        ] = (
            "DIRECT_LIVE_PRODUCT"
        )


        product_started = (
            perf_counter()
        )


        def update_product_metrics():

            metrics.update(
                product_states=
                    entered,

                memo_entries=
                    len(
                        memo
                    ),

                memo_hits=
                    memo_hits,

                recursive_calls=
                    calls,

                labelled_edges_examined=
                    edges_examined,

                dead_live_mask_rejects=
                    rejected,

                output_arena_nodes=
                    len(
                        arena.nodes
                    ),

                product_wall_s=(
                    perf_counter()
                    - product_started
                ),
            )


        def visit(
            nid: int,
            q: int,
        ) -> int:

            nonlocal entered
            nonlocal calls
            nonlocal edges_examined
            nonlocal memo_hits
            nonlocal rejected


            calls += 1


            if (
                calls
                & 1023
            ) == 0:

                update_product_metrics()


                _check_deadline(
                    dl,
                    "UNKNOWN_TIME: "
                    "inside one-piece direct product",
                    metrics,
                )


            if not (
                int(
                    masks[
                        nid
                    ]
                )
                & (
                    1 << q
                )
            ):

                rejected += 1

                return 0


            if nid < 2:

                return nid


            key = (
                nid
                * 24
                + q
            )


            cached = memo.get(
                key
            )


            if cached is not None:

                memo_hits += 1

                return cached


            if entered >= product_cap:

                update_product_metrics()


                raise DirectRestrictUnknown(
                    "UNKNOWN_PRODUCT_CAP: "
                    "one-piece direct restriction",
                    metrics,
                )


            entered += 1


            node = nodes[
                nid
            ]


            nxt = tr_rows[
                q
            ]


            edges: list[
                tuple[
                    int,
                    int,
                ]
            ] = []


            for (
                mid,
                child,
            ) in node.edges:

                edges_examined += 1


                q2 = int(
                    nxt[
                        mid
                    ]
                )


                if not (
                    int(
                        masks[
                            child
                        ]
                    )
                    & (
                        1 << q2
                    )
                ):

                    rejected += 1

                    continue


                target_child = visit(
                    int(
                        child
                    ),
                    q2,
                )


                if target_child:

                    edges.append(
                        (
                            int(
                                mid
                            ),
                            int(
                                target_child
                            ),
                        )
                    )


            answer = arena.intern(
                node.remaining,
                edges,
            )


            # For ONE factor, live-mask positivity is exact.
            # Therefore a live (nid,q) must have at least one
            # accepted labelled continuation.
            if answer == 0:

                raise RuntimeError(
                    "positive exact live mask produced "
                    "FALSE one-piece product"
                )


            memo[
                key
            ] = answer


            return answer


        root = visit(
            int(
                language.root
            ),
            q0,
        )


        update_product_metrics()


        _check_deadline(
            dl,
            "UNKNOWN_TIME: "
            "after one-piece direct product",
            metrics,
        )


        out = _new_language(
            language=
                language,

            arena=
                arena,

            root=
                root,

            construction=
                ONE_TAG,
        )


        metrics.update(
            phase=
                "COMPLETE",

            root=
                int(
                    root
                ),

            root_live=
                bool(
                    int(
                        masks[
                            language.root
                        ]
                    )
                    & (
                        1 << q0
                    )
                ),

            output_arena_nodes=
                len(
                    arena.nodes
                ),

            wall_s=(
                perf_counter()
                - started
            ),
        )


        return (
            out,
            metrics,
        )


    except DirectRestrictUnknown:

        raise


    except ResourceCapUnknown as exc:

        metrics.update(
            phase=
                "UNKNOWN_RESOURCE_CAP",

            wall_s=(
                perf_counter()
                - started
            ),

            reason=
                str(
                    exc
                ),
        )


        raise DirectRestrictUnknown(
            "UNKNOWN_NODE_CAP: "
            "one-piece direct restriction",
            metrics,
        ) from exc


# ==============================================================
# TWO raw-Q joint direct restriction
# ==============================================================

def restrict_two_targetmask(
    language: SymbolicMDD,

    transition_a: np.ndarray,
    initial_q_a: int,
    target_mask_a: int,

    transition_b: np.ndarray,
    initial_q_b: int,
    target_mask_b: int,

    *,
    node_cap: int = 1_500_000,
    input_node_cap: int | None = None,
    product_cap: int = 6_000_000,
    deadline: float | None = None,
) -> tuple[
    SymbolicMDD,
    dict[str, Any],
]:
    """
    Exact:

        C ∩ A ∩ B

    using ONE joint product:

        (current_MDD_node, Qa, Qb)

    There is no materialized C∩A intermediate.

    The two marginal suffix masks are negative gates only.
    Joint SAME-word authority remains the recursive product.
    """

    qa0 = checked_initial_q(
        initial_q_a
    )

    qb0 = checked_initial_q(
        initial_q_b
    )


    target_a = checked_target_mask(
        target_mask_a
    )

    target_b = checked_target_mask(
        target_mask_b
    )


    if (
        type(node_cap)
        is not int
        or
        node_cap < 2
    ):

        raise ValueError(
            "node_cap must be >= 2"
        )


    if (
        type(product_cap)
        is not int
        or
        product_cap < 1
    ):

        raise ValueError(
            "product_cap must be >= 1"
        )


    input_cap = (
        node_cap
        if input_node_cap is None
        else int(
            input_node_cap
        )
    )


    dl = _deadline_value(
        deadline
    )


    tr_a = checked_transition(
        transition_a
    )

    tr_b = checked_transition(
        transition_b
    )


    started = perf_counter()


    metrics: dict[
        str,
        Any,
    ] = {
        "schema":
            "cubelab.direct-targetmask-restrict.v1",

        "arity":
            2,

        "target_mask_a":
            target_a,

        "target_mask_b":
            target_b,

        "target_popcount_a":
            target_a.bit_count(),

        "target_popcount_b":
            target_b.bit_count(),

        "initial_q_a":
            qa0,

        "initial_q_b":
            qb0,

        "product_cap":
            product_cap,

        "node_cap":
            node_cap,

        "partial_language_returned":
            False,

        "accepted_words_materialized":
            0,

        "intermediate_path_count_calls":
            0,

        "joint_false_products":
            0,

        "phase":
            "PREIMAGE_TABLES",
    }


    try:

        _check_deadline(
            dl,
            "UNKNOWN_TIME: "
            "before two-piece direct restriction",
            metrics,
        )


        metrics_a = {}
        metrics_b = {}


        tables_a = preimage_tables(
            tr_a,
            deadline=
                dl,
            metrics=
                metrics_a,
        )


        tables_b = preimage_tables(
            tr_b,
            deadline=
                dl,
            metrics=
                metrics_b,
        )


        metrics[
            "preimage_a"
        ] = metrics_a

        metrics[
            "preimage_b"
        ] = metrics_b


        metrics[
            "phase"
        ] = (
            "MARGINAL_SUFFIX_MASK_DP"
        )


        mask_metrics_a = {}
        mask_metrics_b = {}


        masks_a = suffix_masks_target(
            language,
            tables_a,
            target_a,

            input_node_cap=
                input_cap,

            deadline=
                dl,

            metrics=
                mask_metrics_a,
        )


        masks_b = suffix_masks_target(
            language,
            tables_b,
            target_b,

            input_node_cap=
                input_cap,

            deadline=
                dl,

            metrics=
                mask_metrics_b,
        )


        metrics[
            "mask_a"
        ] = mask_metrics_a

        metrics[
            "mask_b"
        ] = mask_metrics_b


        nodes = (
            language
            .arena
            .nodes
        )


        # Exact marginal-product upper bound.
        # This is telemetry only, not deletion authority.
        marginal_joint_upper = 0


        for nid in range(
            len(
                nodes
            )
        ):

            marginal_joint_upper += (
                int(
                    masks_a[
                        nid
                    ]
                ).bit_count()
                *
                int(
                    masks_b[
                        nid
                    ]
                ).bit_count()
            )


        metrics[
            "marginal_joint_live_state_upper_bound"
        ] = marginal_joint_upper


        arena = NodeArena(
            node_cap=
                node_cap
        )


        rows_a = tr_a.tolist()
        rows_b = tr_b.tolist()


        memo: dict[
            int,
            int,
        ] = {}


        entered = 0
        calls = 0
        edges_examined = 0
        memo_hits = 0
        rejected = 0


        metrics[
            "phase"
        ] = (
            "DIRECT_JOINT_LIVE_PRODUCT"
        )


        product_started = (
            perf_counter()
        )


        def update_product_metrics():

            metrics.update(
                product_states=
                    entered,

                memo_entries=
                    len(
                        memo
                    ),

                memo_hits=
                    memo_hits,

                recursive_calls=
                    calls,

                labelled_edges_examined=
                    edges_examined,

                marginal_dead_rejects=
                    rejected,

                output_arena_nodes=
                    len(
                        arena.nodes
                    ),

                product_wall_s=(
                    perf_counter()
                    - product_started
                ),
            )


        def live(
            nid: int,
            qa: int,
            qb: int,
        ) -> bool:

            return bool(
                (
                    int(
                        masks_a[
                            nid
                        ]
                    )
                    & (
                        1 << qa
                    )
                )
                and
                (
                    int(
                        masks_b[
                            nid
                        ]
                    )
                    & (
                        1 << qb
                    )
                )
            )


        def visit(
            nid: int,
            qa: int,
            qb: int,
        ) -> int:

            nonlocal entered
            nonlocal calls
            nonlocal edges_examined
            nonlocal memo_hits
            nonlocal rejected


            calls += 1


            if (
                calls
                & 1023
            ) == 0:

                update_product_metrics()


                _check_deadline(
                    dl,
                    "UNKNOWN_TIME: "
                    "inside two-piece direct product",
                    metrics,
                )


            if not live(
                nid,
                qa,
                qb,
            ):

                rejected += 1

                return 0


            if nid < 2:

                return nid


            key = (
                576
                * nid
                + 24
                * qa
                + qb
            )


            cached = memo.get(
                key
            )


            if cached is not None:

                memo_hits += 1

                return cached


            if entered >= product_cap:

                update_product_metrics()


                raise DirectRestrictUnknown(
                    "UNKNOWN_PRODUCT_CAP: "
                    "two-piece direct restriction",
                    metrics,
                )


            entered += 1


            node = nodes[
                nid
            ]


            nxt_a = rows_a[
                qa
            ]

            nxt_b = rows_b[
                qb
            ]


            edges: list[
                tuple[
                    int,
                    int,
                ]
            ] = []


            for (
                mid,
                child,
            ) in node.edges:

                edges_examined += 1


                qa2 = int(
                    nxt_a[
                        mid
                    ]
                )


                qb2 = int(
                    nxt_b[
                        mid
                    ]
                )


                if not live(
                    int(
                        child
                    ),
                    qa2,
                    qb2,
                ):

                    rejected += 1

                    continue


                target_child = visit(
                    int(
                        child
                    ),
                    qa2,
                    qb2,
                )


                if target_child:

                    edges.append(
                        (
                            int(
                                mid
                            ),
                            int(
                                target_child
                            ),
                        )
                    )


            answer = arena.intern(
                node.remaining,
                edges,
            )


            # IMPORTANT:
            #
            # marginal A-live AND marginal B-live
            # does NOT imply one common SAME-word continuation.
            #
            # Therefore answer==0 is valid here.
            if answer == 0:

                metrics[
                    "joint_false_products"
                ] = (
                    int(
                        metrics.get(
                            "joint_false_products",
                            0,
                        )
                    )
                    + 1
                )


            memo[
                key
            ] = answer


            return answer


        root = visit(
            int(
                language.root
            ),
            qa0,
            qb0,
        )


        update_product_metrics()


        _check_deadline(
            dl,
            "UNKNOWN_TIME: "
            "after two-piece direct product",
            metrics,
        )


        out = _new_language(
            language=
                language,

            arena=
                arena,

            root=
                root,

            construction=
                TWO_TAG,
        )


        metrics.update(
            phase=
                "COMPLETE",

            root=
                int(
                    root
                ),

            root_marginal_live=
                live(
                    int(
                        language.root
                    ),
                    qa0,
                    qb0,
                ),

            output_arena_nodes=
                len(
                    arena.nodes
                ),

            wall_s=(
                perf_counter()
                - started
            ),
        )


        return (
            out,
            metrics,
        )


    except DirectRestrictUnknown:

        raise


    except ResourceCapUnknown as exc:

        metrics.update(
            phase=
                "UNKNOWN_RESOURCE_CAP",

            wall_s=(
                perf_counter()
                - started
            ),

            reason=
                str(
                    exc
                ),
        )


        raise DirectRestrictUnknown(
            "UNKNOWN_NODE_CAP: "
            "two-piece direct restriction",
            metrics,
        ) from exc


__all__ = [
    "DirectRestrictUnknown",
    "FULL24",
    "ONE_TAG",
    "TWO_TAG",
    "checked_transition",
    "preimage",
    "preimage_tables",
    "restrict_one_targetmask",
    "restrict_two_targetmask",
    "suffix_masks_target",
]
