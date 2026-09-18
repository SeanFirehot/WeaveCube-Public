from __future__ import annotations

from array import array
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from cubelab.column_families.symbolic_mdd_v2 import (
    NodeArena,
    ResourceCapUnknown,
    SymbolicMDD,
)

from .direct_q_restrict import (
    DirectRestrictUnknown,
    checked_initial_q,
    checked_target_mask,
    checked_transition,
    preimage_tables,
    suffix_masks_target,
)


# ==============================================================
# Prepared exact masks
# ==============================================================

@dataclass(
    frozen=True,
    slots=True,
)
class PreparedPieceMask:

    language: SymbolicMDD

    transition: np.ndarray
    transition_rows: tuple[
        tuple[int, ...],
        ...
    ]

    initial_q: int
    target_mask: int

    masks: array

    coreach_live_state_pairs: int
    root_live: bool

    prepare_wall_s: float

    metrics: dict[str, Any]


@dataclass(
    frozen=True,
    slots=True,
)
class PreparedPairMask:

    left: PreparedPieceMask
    right: PreparedPieceMask

    marginal_joint_live_upper: int

    prepare_wall_s: float

    metrics: dict[str, Any]


def _deadline(
    value: float | None,
) -> float:

    return (
        float("inf")
        if value is None
        else float(value)
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


def _validate_prepared_language(
    language: SymbolicMDD,
    prepared: PreparedPieceMask,
) -> None:

    # Masks are indexed by the exact arena node ids.
    # Semantic equality alone is NOT sufficient for reuse.
    if language is not prepared.language:

        raise ValueError(
            "prepared live masks belong to a different "
            "SymbolicMDD object"
        )


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
            int(root),

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
# PREPARE ONE PIECE
# ==============================================================

def prepare_piece_mask(
    language: SymbolicMDD,

    transition: np.ndarray,
    initial_q: int,
    target_mask: int,

    *,
    input_node_cap: int = 1_500_000,
    deadline: float | None = None,
) -> PreparedPieceMask:

    started = perf_counter()

    dl = _deadline(
        deadline
    )

    tr = checked_transition(
        transition
    )

    q0 = checked_initial_q(
        initial_q
    )

    target = checked_target_mask(
        target_mask
    )

    metrics: dict[
        str,
        Any,
    ] = {
        "schema":
            "cubelab.prepared-q-mask.v1",

        "phase":
            "PREIMAGE",

        "initial_q":
            q0,

        "target_mask":
            target,

        "target_popcount":
            int(
                target.bit_count()
            ),
    }


    _check_deadline(
        dl,
        "UNKNOWN_TIME: prepare_piece_mask preimage",
        metrics,
    )


    preimage_metrics = {}


    tables = preimage_tables(
        tr,

        deadline=
            dl,

        metrics=
            preimage_metrics,
    )


    metrics[
        "preimage"
    ] = preimage_metrics


    metrics[
        "phase"
    ] = "SUFFIX_MASK"


    mask_metrics = {}


    masks = suffix_masks_target(
        language,
        tables,
        target,

        input_node_cap=
            int(
                input_node_cap
            ),

        deadline=
            dl,

        metrics=
            mask_metrics,
    )


    metrics[
        "mask"
    ] = mask_metrics


    # This is exact coreach pressure across the input arena.
    # It is not yet root-reachable product-state count.
    live_pairs = int(
        mask_metrics.get(
            "mask_live_state_pairs",
            0,
        )
    )


    root_live = bool(
        int(
            masks[
                int(
                    language.root
                )
            ]
        )
        &
        (
            1 << q0
        )
    )


    wall = (
        perf_counter()
        - started
    )


    metrics.update(
        phase=
            "COMPLETE",

        coreach_live_state_pairs=
            live_pairs,

        root_live=
            root_live,

        wall_s=
            wall,
    )


    rows = tuple(
        tuple(
            int(v)
            for v in row
        )
        for row in tr.tolist()
    )


    return PreparedPieceMask(
        language=
            language,

        transition=
            tr,

        transition_rows=
            rows,

        initial_q=
            q0,

        target_mask=
            target,

        masks=
            masks,

        coreach_live_state_pairs=
            live_pairs,

        root_live=
            root_live,

        prepare_wall_s=
            wall,

        metrics=
            metrics,
    )


# ==============================================================
# PREPARE TWO-PIECE ESTIMATE
# ==============================================================

def prepare_pair_mask(
    left: PreparedPieceMask,
    right: PreparedPieceMask,

    *,
    deadline: float | None = None,
) -> PreparedPairMask:

    if left.language is not right.language:

        raise ValueError(
            "pair masks must belong to the exact same "
            "input SymbolicMDD"
        )


    if len(
        left.masks
    ) != len(
        right.masks
    ):

        raise RuntimeError(
            "prepared mask length drift"
        )


    started = perf_counter()

    dl = _deadline(
        deadline
    )


    upper = 0


    for nid, (
        left_mask,
        right_mask,
    ) in enumerate(
        zip(
            left.masks,
            right.masks,
        )
    ):

        if (
            nid
            & 8191
        ) == 0:

            _check_deadline(
                dl,
                "UNKNOWN_TIME: prepare_pair_mask",
                {
                    "rows_examined":
                        nid,
                },
            )


        upper += (
            int(
                left_mask
            ).bit_count()
            *
            int(
                right_mask
            ).bit_count()
        )


    wall = (
        perf_counter()
        - started
    )


    metrics = {
        "schema":
            "cubelab.prepared-q-pair.v1",

        "marginal_joint_live_upper":
            int(
                upper
            ),

        "left_coreach_live_pairs":
            int(
                left.coreach_live_state_pairs
            ),

        "right_coreach_live_pairs":
            int(
                right.coreach_live_state_pairs
            ),

        "root_marginal_live":
            bool(
                left.root_live
                and right.root_live
            ),

        "wall_s":
            wall,
    }


    return PreparedPairMask(
        left=
            left,

        right=
            right,

        marginal_joint_live_upper=
            int(
                upper
            ),

        prepare_wall_s=
            wall,

        metrics=
            metrics,
    )


# ==============================================================
# EXECUTE PREPARED ONE
# ==============================================================

def execute_prepared_one(
    prepared: PreparedPieceMask,

    *,
    node_cap: int = 1_500_000,
    product_cap: int = 3_000_000,
    deadline: float | None = None,
) -> tuple[
    SymbolicMDD,
    dict[str, Any],
]:

    language = prepared.language

    _validate_prepared_language(
        language,
        prepared,
    )


    dl = _deadline(
        deadline
    )


    if (
        type(node_cap) is not int
        or node_cap < 2
    ):

        raise ValueError(
            "node_cap must be >= 2"
        )


    if (
        type(product_cap) is not int
        or product_cap < 1
    ):

        raise ValueError(
            "product_cap must be >= 1"
        )


    started = perf_counter()


    nodes = language.arena.nodes

    masks = prepared.masks

    transition_rows = (
        prepared.transition_rows
    )


    arena = NodeArena(
        node_cap=
            node_cap
    )


    memo: dict[
        int,
        int,
    ] = {}


    calls = 0
    entered = 0
    memo_hits = 0

    edges_examined = 0
    dead_rejects = 0


    metrics: dict[
        str,
        Any,
    ] = {
        "schema":
            "cubelab.execute-prepared-q.v1",

        "arity":
            1,

        "prepared_coreach_live_pairs":
            int(
                prepared.coreach_live_state_pairs
            ),

        "prepared_wall_s":
            float(
                prepared.prepare_wall_s
            ),

        "product_cap":
            int(
                product_cap
            ),

        "node_cap":
            int(
                node_cap
            ),

        "phase":
            "PRODUCT",
    }


    def update_metrics():

        metrics.update(
            product_states=
                entered,

            recursive_calls=
                calls,

            memo_entries=
                len(
                    memo
                ),

            memo_hits=
                memo_hits,

            labelled_edges_examined=
                edges_examined,

            dead_rejects=
                dead_rejects,

            output_arena_nodes=
                len(
                    arena.nodes
                ),

            execute_wall_s=(
                perf_counter()
                - started
            ),
        )


    def visit(
        nid: int,
        q: int,
    ) -> int:

        nonlocal calls
        nonlocal entered
        nonlocal memo_hits

        nonlocal edges_examined
        nonlocal dead_rejects


        calls += 1


        if (
            calls
            & 1023
        ) == 0:

            update_metrics()


            _check_deadline(
                dl,
                "UNKNOWN_TIME: execute_prepared_one",
                metrics,
            )


        if not (
            int(
                masks[
                    nid
                ]
            )
            &
            (
                1 << q
            )
        ):

            dead_rejects += 1

            return 0


        if nid < 2:

            return nid


        key = (
            24
            * nid
            + q
        )


        cached = memo.get(
            key
        )


        if cached is not None:

            memo_hits += 1

            return cached


        if entered >= product_cap:

            update_metrics()


            raise DirectRestrictUnknown(
                "UNKNOWN_PRODUCT_CAP: "
                "execute_prepared_one",
                metrics,
            )


        entered += 1


        node = nodes[
            nid
        ]


        row = transition_rows[
            q
        ]


        edges = []


        for mid, child in node.edges:

            edges_examined += 1


            q2 = int(
                row[
                    int(
                        mid
                    )
                ]
            )


            if not (
                int(
                    masks[
                        int(
                            child
                        )
                    ]
                )
                &
                (
                    1 << q2
                )
            ):

                dead_rejects += 1

                continue


            child_out = visit(
                int(
                    child
                ),
                q2,
            )


            if child_out:

                edges.append(
                    (
                        int(
                            mid
                        ),
                        int(
                            child_out
                        ),
                    )
                )


        answer = arena.intern(
            node.remaining,
            edges,
        )


        # For one exact factor, a positive coreach state
        # must possess at least one accepted continuation.
        if answer == 0:

            raise RuntimeError(
                "prepared ONE live-mask inconsistency"
            )


        memo[
            key
        ] = answer


        return answer


    try:

        root = visit(
            int(
                language.root
            ),
            int(
                prepared.initial_q
            ),
        )


    except ResourceCapUnknown as exc:

        update_metrics()


        raise DirectRestrictUnknown(
            "UNKNOWN_NODE_CAP: "
            "execute_prepared_one",
            metrics,
        ) from exc


    update_metrics()


    out = _new_language(
        language=
            language,

        arena=
            arena,

        root=
            root,

        construction=
            "V375_PREPARED_ONE",
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
                prepared.root_live
            ),

        execute_wall_s=(
            perf_counter()
            - started
        ),

        total_prepare_plus_execute_wall_s=(
            float(
                prepared.prepare_wall_s
            )
            +
            (
                perf_counter()
                - started
            )
        ),
    )


    return (
        out,
        metrics,
    )


# ==============================================================
# EXECUTE PREPARED TWO
# ==============================================================

def execute_prepared_two(
    prepared: PreparedPairMask,

    *,
    node_cap: int = 1_500_000,
    product_cap: int = 4_000_000,
    deadline: float | None = None,
) -> tuple[
    SymbolicMDD,
    dict[str, Any],
]:

    left = prepared.left
    right = prepared.right


    if left.language is not right.language:

        raise ValueError(
            "prepared pair language drift"
        )


    language = left.language


    dl = _deadline(
        deadline
    )


    started = perf_counter()


    nodes = language.arena.nodes


    masks_a = left.masks
    masks_b = right.masks


    rows_a = left.transition_rows
    rows_b = right.transition_rows


    arena = NodeArena(
        node_cap=
            node_cap
    )


    memo: dict[
        int,
        int,
    ] = {}


    calls = 0
    entered = 0

    memo_hits = 0

    edges_examined = 0
    dead_rejects = 0

    joint_false = 0


    metrics: dict[
        str,
        Any,
    ] = {
        "schema":
            "cubelab.execute-prepared-q.v1",

        "arity":
            2,

        "left_prepare_wall_s":
            float(
                left.prepare_wall_s
            ),

        "right_prepare_wall_s":
            float(
                right.prepare_wall_s
            ),

        "pair_estimator_wall_s":
            float(
                prepared.prepare_wall_s
            ),

        "marginal_joint_live_upper":
            int(
                prepared.marginal_joint_live_upper
            ),

        "product_cap":
            int(
                product_cap
            ),

        "node_cap":
            int(
                node_cap
            ),

        "phase":
            "JOINT_PRODUCT",
    }


    def update_metrics():

        metrics.update(
            product_states=
                entered,

            recursive_calls=
                calls,

            memo_entries=
                len(
                    memo
                ),

            memo_hits=
                memo_hits,

            labelled_edges_examined=
                edges_examined,

            marginal_dead_rejects=
                dead_rejects,

            joint_false_products=
                joint_false,

            output_arena_nodes=
                len(
                    arena.nodes
                ),

            execute_wall_s=(
                perf_counter()
                - started
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
                &
                (
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
                &
                (
                    1 << qb
                )
            )
        )


    def visit(
        nid: int,
        qa: int,
        qb: int,
    ) -> int:

        nonlocal calls
        nonlocal entered

        nonlocal memo_hits

        nonlocal edges_examined
        nonlocal dead_rejects

        nonlocal joint_false


        calls += 1


        if (
            calls
            & 1023
        ) == 0:

            update_metrics()


            _check_deadline(
                dl,
                "UNKNOWN_TIME: execute_prepared_two",
                metrics,
            )


        if not live(
            nid,
            qa,
            qb,
        ):

            dead_rejects += 1

            return 0


        if nid < 2:

            return nid


        key = (
            576
            * nid
            +
            24
            * qa
            +
            qb
        )


        cached = memo.get(
            key
        )


        if cached is not None:

            memo_hits += 1

            return cached


        if entered >= product_cap:

            update_metrics()


            raise DirectRestrictUnknown(
                "UNKNOWN_PRODUCT_CAP: "
                "execute_prepared_two",
                metrics,
            )


        entered += 1


        node = nodes[
            nid
        ]


        row_a = rows_a[
            qa
        ]

        row_b = rows_b[
            qb
        ]


        edges = []


        for mid, child in node.edges:

            edges_examined += 1


            mid = int(
                mid
            )

            child = int(
                child
            )


            qa2 = int(
                row_a[
                    mid
                ]
            )


            qb2 = int(
                row_b[
                    mid
                ]
            )


            if not live(
                child,
                qa2,
                qb2,
            ):

                dead_rejects += 1

                continue


            child_out = visit(
                child,
                qa2,
                qb2,
            )


            if child_out:

                edges.append(
                    (
                        mid,
                        int(
                            child_out
                        ),
                    )
                )


        answer = arena.intern(
            node.remaining,
            edges,
        )


        # Marginal A/B positivity does not imply one common word.
        if answer == 0:

            joint_false += 1


        memo[
            key
        ] = answer


        return answer


    try:

        root = visit(
            int(
                language.root
            ),

            int(
                left.initial_q
            ),

            int(
                right.initial_q
            ),
        )


    except ResourceCapUnknown as exc:

        update_metrics()


        raise DirectRestrictUnknown(
            "UNKNOWN_NODE_CAP: "
            "execute_prepared_two",
            metrics,
        ) from exc


    update_metrics()


    out = _new_language(
        language=
            language,

        arena=
            arena,

        root=
            root,

        construction=
            "V375_PREPARED_TWO",
    )


    execute_wall = (
        perf_counter()
        - started
    )


    metrics.update(
        phase=
            "COMPLETE",

        root=
            int(
                root
            ),

        root_marginal_live=
            bool(
                left.root_live
                and right.root_live
            ),

        execute_wall_s=
            execute_wall,

        total_piece_prepare_wall_s=(
            float(
                left.prepare_wall_s
            )
            +
            float(
                right.prepare_wall_s
            )
        ),

        total_prepare_estimate_execute_wall_s=(
            float(
                left.prepare_wall_s
            )
            +
            float(
                right.prepare_wall_s
            )
            +
            float(
                prepared.prepare_wall_s
            )
            +
            execute_wall
        ),
    )


    return (
        out,
        metrics,
    )


__all__ = [
    "PreparedPairMask",
    "PreparedPieceMask",
    "execute_prepared_one",
    "execute_prepared_two",
    "prepare_pair_mask",
    "prepare_piece_mask",
]
