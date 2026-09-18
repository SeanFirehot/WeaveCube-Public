from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .legacy_compat import ensure_legacy_compat

ensure_legacy_compat()

from cubelab.column_families.symbolic_mdd_v2 import (
    ResourceCapUnknown,
)

from .ato_language import (
    build_exact_ato_language,
)


FULL_ATO_MASK = (
    1 << 20
) - 1


@dataclass(
    frozen=True,
    slots=True,
)
class ExactClosureResult:

    status: str

    board: object | None
    word: tuple[str, ...]

    horizon: int | None

    path_count: int | None
    nodes: int | None
    edges: int | None

    attempts: tuple[
        dict[str, object],
        ...
    ]

    wall_s: float


class ExactResidualClosure:
    """
    On-demand exact SAME-word ATO closure.

    This is NOT a lock on currently satisfied pieces.

    Each attempted residual word may disturb any piece arbitrarily.
    The only hard piece condition is ALL20 Omega-zero at the END.
    """

    def __init__(
        self,
        *,
        rewrite_bank,

        min_horizon: int = 5,
        max_horizon: int = 6,

        node_cap: int = 1_500_000,
    ) -> None:

        self.rewrite = rewrite_bank

        self.min_horizon = int(
            min_horizon
        )

        self.max_horizon = int(
            max_horizon
        )

        self.node_cap = int(
            node_cap
        )

        self.move_index = {
            name:
                i
            for i, name
            in enumerate(
                self.rewrite.move_names
            )
        }

        self._cache = {}

    def _key(
        self,
        board,
    ):

        return (
            board.end_q().tobytes(),
            board.residual_previous_face(),
            self.min_horizon,
            self.max_horizon,
        )

    def try_close(
        self,
        board,
    ) -> ExactClosureResult:

        key = self._key(
            board
        )

        cached = self._cache.get(
            key
        )

        if cached is not None:
            return cached

        started = perf_counter()

        attempts = []

        for horizon in range(
            self.min_horizon,
            self.max_horizon + 1,
        ):

            t0 = perf_counter()

            try:

                language = build_exact_ato_language(
                    root_q=
                        board.end_q(),

                    horizon=
                        horizon,

                    previous_face=
                        board.residual_previous_face(),

                    q_transition_hash=
                        self.rewrite.q_transition_hash,

                    node_cap=
                        self.node_cap,

                    progress=
                        None,
                )

                resource_unknown = False

            except ResourceCapUnknown:

                language = None
                resource_unknown = True

            wall = (
                perf_counter()
                - t0
            )

            if resource_unknown:

                attempts.append({
                    "horizon":
                        horizon,

                    "status":
                        "UNKNOWN_RESOURCE_CAP",

                    "wall_s":
                        wall,
                })

                result = ExactClosureResult(
                    status=
                        "UNKNOWN_RESOURCE_CAP",

                    board=
                        None,

                    word=
                        tuple(),

                    horizon=
                        horizon,

                    path_count=
                        None,

                    nodes=
                        None,

                    edges=
                        None,

                    attempts=
                        tuple(
                            attempts
                        ),

                    wall_s=
                        perf_counter()
                        - started,
                )

                self._cache[
                    key
                ] = result

                return result

            assert language is not None

            support = language.support

            paths = int(
                support.count()
            )

            attempts.append({
                "horizon":
                    horizon,

                "status":(
                    "SAT"
                    if paths
                    else "EXACT_EMPTY"
                ),

                "paths":
                    paths,

                "nodes":
                    int(
                        support.mdd.node_count
                    ),

                "edges":
                    int(
                        support.mdd.edge_count
                    ),

                "wall_s":
                    wall,
            })

            if paths == 0:
                continue

            word = support.first_word()

            if word is None:
                raise RuntimeError(
                    "positive exact closure language has no witness"
                )

            ids = tuple(
                int(
                    self.move_index[
                        token
                    ]
                )
                for token in word
            )

            child = board.recompose(
                start=
                    len(
                        board
                    ),

                delete_count=
                    0,

                insert_ids=
                    ids,
            )

            signature = 0

            q = child.end_q()

            for piece in range(
                20
            ):
                if int(
                    self.rewrite.omega[
                        int(
                            q[
                                piece
                            ]
                        )
                    ]
                ) == 0:

                    signature |= (
                        1
                        << piece
                    )

            if signature != FULL_ATO_MASK:
                raise RuntimeError(
                    "exact closure witness failed ALL20 ATO"
                )

            result = ExactClosureResult(
                status=
                    "SAT_ATO",

                board=
                    child,

                word=
                    tuple(
                        word
                    ),

                horizon=
                    horizon,

                path_count=
                    paths,

                nodes=
                    int(
                        support.mdd.node_count
                    ),

                edges=
                    int(
                        support.mdd.edge_count
                    ),

                attempts=
                    tuple(
                        attempts
                    ),

                wall_s=
                    perf_counter()
                    - started,
            )

            self._cache[
                key
            ] = result

            return result

        result = ExactClosureResult(
            status=
                "EXACT_NO_CLOSURE_IN_LOCAL_WORKSPACE",

            board=
                None,

            word=
                tuple(),

            horizon=
                self.max_horizon,

            path_count=
                0,

            nodes=
                None,

            edges=
                None,

            attempts=
                tuple(
                    attempts
                ),

            wall_s=
                perf_counter()
                - started,
        )

        self._cache[
            key
        ] = result

        return result
