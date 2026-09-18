from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Any

import numpy as np

from .legacy_compat import ensure_legacy_compat

ensure_legacy_compat()

from cubelab.column_families.canonical_language import (
    CANONICAL_V356_HASH,
)

from cubelab.column_families.piece_language import (
    piece_orientation_language,
)

from .mdd_support import SymbolicMDDSupport
from .problem_scope import ProblemScope


@dataclass(
    frozen=True,
    slots=True,
)
class ATOFactorInfo:
    piece: int
    start_q: int

    paths: int
    nodes: int
    edges: int

    build_wall_s: float


@dataclass(
    frozen=True,
    slots=True,
)
class ATOIntersectionInfo:
    ordinal: int
    piece: int

    paths_after: int
    nodes_after: int
    edges_after: int

    intersection_wall_s: float


@dataclass(
    frozen=True,
    slots=True,
)
class ExactATOLanguage:
    problem_scope: ProblemScope
    support: SymbolicMDDSupport

    factor_order: tuple[int, ...]

    factors: tuple[
        ATOFactorInfo,
        ...
    ]

    intersections: tuple[
        ATOIntersectionInfo,
        ...
    ]


def _state_hash(
    q: np.ndarray,
) -> str:

    q = np.asarray(
        q,
        dtype=np.uint8,
    )

    if q.shape != (
        20,
    ):
        raise ValueError(
            "root Q must have shape (20,)"
        )

    return sha256(
        np.ascontiguousarray(
            q
        ).tobytes()
    ).hexdigest()


def make_ato_problem_scope(
    *,
    root_q: np.ndarray,
    horizon: int,
    previous_face: str | None,
    q_transition_hash: str,
) -> ProblemScope:

    return ProblemScope(
        schema=(
            "cubelab.constructive."
            "problem-scope.v1"
        ),

        state_sha256=
            _state_hash(
                root_q
            ),

        remaining=
            int(
                horizon
            ),

        previous_face=
            previous_face,

        q_transition_hash=
            str(
                q_transition_hash
            ),

        canonical_rule_hash=
            str(
                CANONICAL_V356_HASH
            ),

        goal=
            "ALL20_OMEGA_ZERO_AT_END",
    )


def build_exact_ato_language(
    *,
    root_q: np.ndarray,
    horizon: int,
    previous_face: str | None,
    q_transition_hash: str,

    node_cap: int = 1_000_000,

    progress:
        Callable[
            [
                dict[
                    str,
                    Any,
                ]
            ],
            None,
        ]
        | None
        = None,

) -> ExactATOLanguage:

    from time import perf_counter

    q = np.asarray(
        root_q,
        dtype=np.uint8,
    )

    if q.shape != (
        20,
    ):
        raise ValueError(
            "root Q must have shape (20,)"
        )

    scope = make_ato_problem_scope(
        root_q=
            q,

        horizon=
            horizon,

        previous_face=
            previous_face,

        q_transition_hash=
            q_transition_hash,
    )

    built = []

    # ----------------------------------------------------------
    # Build all 20 exact piece languages independently.
    # ----------------------------------------------------------

    for piece in range(
        20
    ):
        t0 = perf_counter()

        mdd = piece_orientation_language(
            piece,
            int(
                q[
                    piece
                ]
            ),
            int(
                horizon
            ),

            previous_face=
                previous_face,

            scope_hash=
                scope.digest,

            node_cap=
                node_cap,
        )

        info = ATOFactorInfo(
            piece=
                piece,

            start_q=
                int(
                    q[
                        piece
                    ]
                ),

            paths=
                int(
                    mdd.path_count()
                ),

            nodes=
                int(
                    mdd.node_count
                ),

            edges=
                int(
                    mdd.edge_count
                ),

            build_wall_s=
                float(
                    perf_counter()
                    - t0
                ),
        )

        built.append(
            (
                info,
                mdd,
            )
        )

        if progress is not None:
            progress({
                "phase":
                    "factor",

                "piece":
                    piece,

                "start_q":
                    info.start_q,

                "paths":
                    info.paths,

                "nodes":
                    info.nodes,

                "edges":
                    info.edges,

                "wall_s":
                    info.build_wall_s,
            })

    # ----------------------------------------------------------
    # Economics only:
    # restrictive exact language first.
    #
    # This changes intersection order only, never semantics.
    # ----------------------------------------------------------

    built.sort(
        key=lambda row:(
            row[
                0
            ].paths,

            row[
                0
            ].nodes,

            row[
                0
            ].piece,
        )
    )

    factor_order = tuple(
        info.piece
        for info, _
        in built
    )

    if progress is not None:
        progress({
            "phase":
                "order",

            "piece_order":
                factor_order,
        })

    # ----------------------------------------------------------
    # Exact SAME-word intersection.
    # ----------------------------------------------------------

    current = None

    intersections = []

    for ordinal, (
        info,
        mdd,
    ) in enumerate(
        built,
        1,
    ):

        factor = SymbolicMDDSupport(
            problem_scope=
                scope,

            mdd=
                mdd,
        )

        if current is None:
            current = factor
            wall = 0.0

        else:
            t0 = perf_counter()

            current = current.intersect(
                factor,
                node_cap=
                    node_cap,
            )

            wall = float(
                perf_counter()
                - t0
            )

        row = ATOIntersectionInfo(
            ordinal=
                ordinal,

            piece=
                info.piece,

            paths_after=
                int(
                    current.count()
                ),

            nodes_after=
                int(
                    current.mdd.node_count
                ),

            edges_after=
                int(
                    current.mdd.edge_count
                ),

            intersection_wall_s=
                wall,
        )

        intersections.append(
            row
        )

        if progress is not None:
            progress({
                "phase":
                    "intersection",

                "ordinal":
                    ordinal,

                "piece":
                    info.piece,

                "paths_after":
                    row.paths_after,

                "nodes_after":
                    row.nodes_after,

                "edges_after":
                    row.edges_after,

                "wall_s":
                    wall,
            })

        if current.empty_p():
            break

    if current is None:
        raise RuntimeError(
            "no ATO factors built"
        )

    return ExactATOLanguage(
        problem_scope=
            scope,

        support=
            current,

        factor_order=
            factor_order,

        factors=
            tuple(
                info
                for info, _
                in built
            ),

        intersections=
            tuple(
                intersections
            ),
    )
