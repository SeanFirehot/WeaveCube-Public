from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np

from cubelab.column_families.symbolic_mdd_v2 import (
    ResourceCapUnknown,
    SymbolicMDD,
)

from .c4_packed_first_witness_v2 import (
    first_c4_packed_witness_vectorized,
)

from .exact_closure_dispatcher import (
    COMPLETE,
    DYNAMIC,
    ClosureRequest,
    ClosureResult,
    ExactClosureDispatcher,
    exact_closure_signature,
)


FIRST_WITNESS = "FIRST_WITNESS"


@dataclass(frozen=True, slots=True)
class FirstWitnessPlan:
    """
    Exact-signature registration for SAT-only first-witness execution.

    `signature` is the normal factor-less C4 exact closure signature.
    `fallback_backend` is an existing exact full-closure backend.

    A first-witness miss or resource cap NEVER becomes EMPTY.
    """

    name: str
    signature: str
    fallback_backend: str = DYNAMIC


def factorless_request(
    request: ClosureRequest,
) -> ClosureRequest:
    return ClosureRequest(
        language=
            request.language,

        pending_pieces=
            request.pending(),

        q_before=
            request.q_before,

        target_masks=
            request.target_masks,

        rewrite=
            request.rewrite,

        factor_mdds={},
    )


def factorless_signature(
    request: ClosureRequest,
) -> str:
    return exact_closure_signature(
        factorless_request(
            request
        )
    )


def _move_id_map(
    rewrite: Any,
) -> dict[str, int]:
    names = tuple(
        str(v)
        for v in getattr(
            rewrite,
            "move_names",
        )
    )

    if len(names) != 18 or len(set(names)) != 18:
        raise ValueError(
            "rewrite.move_names must contain 18 unique moves"
        )

    return {
        name:
            index
        for index, name
        in enumerate(
            names
        )
    }


def _validate_first_witness(
    request: ClosureRequest,
    witness: Sequence[str],
) -> tuple[int, ...]:
    word = tuple(
        str(v)
        for v in witness
    )

    if len(word) != int(
        request.language.horizon
    ):
        raise RuntimeError(
            "first-witness length/horizon mismatch"
        )

    if not bool(
        request.language.contains(
            word
        )
    ):
        raise RuntimeError(
            "first-witness is not in seed SymbolicMDD"
        )

    move_map = _move_id_map(
        request.rewrite
    )

    try:
        move_ids = tuple(
            int(
                move_map[
                    token
                ]
            )
            for token in word
        )

    except KeyError as exc:
        raise RuntimeError(
            "first-witness contains move outside rewrite vocabulary"
        ) from exc

    q = np.asarray(
        request.q_before,
        dtype=np.uint8,
    ).copy()

    q_next = np.asarray(
        request.rewrite.q_next,
        dtype=np.uint8,
    )

    if (
        q.ndim != 1
        or q_next.ndim != 3
        or q_next.shape[
            0
        ]
        != len(q)
        or q_next.shape[
            1:
        ]
        != (
            24,
            18,
        )
    ):
        raise ValueError(
            "C4 first-witness Q shape drift"
        )

    all_pieces = np.arange(
        len(q),
        dtype=np.int64,
    )

    for move_id in move_ids:
        q = q_next[
            all_pieces,
            q.astype(
                np.int64,
                copy=False,
            ),
            int(
                move_id
            ),
        ].astype(
            np.uint8,
            copy=False,
        )

    targets = tuple(
        int(v)
        for v in request.target_masks
    )

    for piece in request.pending():
        end_q = int(
            q[
                piece
            ]
        )

        if not (
            targets[
                piece
            ]
            &
            (
                1
                << end_q
            )
        ):
            raise RuntimeError(
                f"first-witness failed pending target piece {piece}"
            )

    return move_ids


def _singleton_language(
    *,
    request: ClosureRequest,
    witness: Sequence[str],
    move_ids: Sequence[int],
) -> SymbolicMDD:
    """
    Materialize only the accepted witness as a tiny exact MDD.

    This keeps the existing ResidualGraphClosureBridge contract intact
    without materializing the full closure language.
    """

    word = tuple(
        str(v)
        for v in witness
    )

    ids = tuple(
        int(v)
        for v in move_ids
    )

    h = int(
        request.language.horizon
    )

    if len(ids) != h:
        raise RuntimeError(
            "singleton witness length drift"
        )

    def transition(
        column,
        move_id,
        *,
        ids=ids,
        h=h,
    ):
        column = int(
            column
        )

        if (
            column < 0
            or column >= h
        ):
            return -1

        if int(
            move_id
        ) != ids[
            column
        ]:
            return -1

        return (
            column
            + 1
        )

    def accept(
        column,
        *,
        h=h,
    ):
        return bool(
            int(
                column
            )
            == h
        )

    singleton = SymbolicMDD.from_automaton(
        0,
        h,
        transition,
        accept,
        previous_face=
            request.language.previous_face,

        scope_hash=
            request.language.scope_hash,

        node_cap=
            max(
                64,
                4 * h
                + 16,
            ),

        construction=
            "C4_FIRST_WITNESS_SINGLETON",
    )

    if int(
        singleton.path_count()
    ) != 1:
        raise RuntimeError(
            "first-witness singleton language is not singleton"
        )

    if not bool(
        singleton.contains(
            word
        )
    ):
        raise RuntimeError(
            "first-witness singleton does not contain witness"
        )

    return singleton


class FirstWitnessExactClosureRouter:
    """
    C4 execution router.

    Exact registered factor-less signature:
        FIRST_WITNESS SAT
            -> COMPLETE with a tiny singleton MDD
            -> existing bridge/graph contract unchanged

        FIRST_WITNESS UNSAT/no-witness/resource-cap
            -> fallback to existing exact full-closure backend
            -> NEVER interpreted as EMPTY by this backend

    Unregistered signatures:
        delegate entirely to ExactClosureDispatcher.

    This object is intentionally duck-compatible with ExactClosureDispatcher
    for ExactLocalContextPublisher and ResidualGraphClosureBridge.
    """

    def __init__(
        self,
        *,
        base_dispatcher: ExactClosureDispatcher,
        first_witness_plans: Sequence[FirstWitnessPlan] = (),
        node_cap: int = 5_000_000,
        time_cap_seconds: float | None = 90.0,
        rss_growth_cap_mib: float = 1536.0,
        runner: Callable[..., dict[str, Any]] | None = None,
    ):
        self.base = base_dispatcher

        self.node_cap = int(
            node_cap
        )

        self.time_cap_seconds = (
            None
            if time_cap_seconds is None
            else float(
                time_cap_seconds
            )
        )

        self.rss_growth_cap_mib = float(
            rss_growth_cap_mib
        )

        self.runner = (
            first_c4_packed_witness_vectorized
            if runner is None
            else runner
        )

        self.first_witness: dict[
            str,
            FirstWitnessPlan,
        ] = {}

        for plan in first_witness_plans:
            signature = str(
                plan.signature
            )

            if signature in self.first_witness:
                raise ValueError(
                    "duplicate first-witness signature"
                )

            if signature in getattr(
                self.base,
                "frozen",
                {},
            ):
                raise ValueError(
                    "first-witness signature conflicts with frozen signature"
                )

            if signature in getattr(
                self.base,
                "rescue",
                {},
            ):
                raise ValueError(
                    "first-witness signature conflicts with rescue signature"
                )

            if str(
                plan.fallback_backend
            ) == FIRST_WITNESS:
                raise ValueError(
                    "FIRST_WITNESS cannot fall back to itself"
                )

            self.first_witness[
                signature
            ] = plan

        self.first_attempts = 0
        self.first_sat = 0
        self.first_fallbacks = 0

    @property
    def frozen(
        self,
    ):
        return self.base.frozen

    @property
    def rescue(
        self,
    ):
        return self.base.rescue

    @property
    def quote_bank(
        self,
    ):
        return self.base.quote_bank

    def select_backend(
        self,
        request: ClosureRequest,
    ) -> tuple[
        str,
        str,
        str | None,
    ]:
        signature = exact_closure_signature(
            request
        )

        plan = self.first_witness.get(
            signature
        )

        if plan is not None:
            return (
                FIRST_WITNESS,
                signature,
                plan.name,
            )

        return self.base.select_backend(
            request
        )

    def _fallback(
        self,
        *,
        request: ClosureRequest,
        plan: FirstWitnessPlan,
        signature: str,
        first_status: str,
        first_reason: str | None,
        first_wall_s: float,
        first_metrics: dict[str, Any],
    ) -> ClosureResult:
        self.first_fallbacks += 1

        fallback_started = perf_counter()

        result = self.base.execute(
            request,
            force_backend=
                str(
                    plan.fallback_backend
                ),
        )

        fallback_wall = (
            perf_counter()
            - fallback_started
        )

        result.trace.insert(
            0,
            {
                "op":
                    "FIRST_WITNESS_FALLBACK",

                "first_status":
                    first_status,

                "first_reason":
                    first_reason,

                "first_wall_s":
                    float(
                        first_wall_s
                    ),

                "fallback_backend":
                    str(
                        plan.fallback_backend
                    ),

                "first_metrics":
                    dict(
                        first_metrics
                    ),
            },
        )

        result.timings = {
            **dict(
                result.timings
            ),

            "first_witness_attempt_s":
                float(
                    first_wall_s
                ),

            "fallback_execute_s":
                float(
                    fallback_wall
                ),

            "wall_with_first_fallback_s":
                float(
                    first_wall_s
                    + fallback_wall
                ),
        }

        return result

    def _first(
        self,
        *,
        request: ClosureRequest,
        signature: str,
        plan: FirstWitnessPlan,
    ) -> ClosureResult:
        self.first_attempts += 1

        started = perf_counter()

        first_metrics: dict[
            str,
            Any,
        ] = {}

        try:
            result = self.runner(
                language=
                    request.language,

                q_next=
                    request.rewrite.q_next,

                q_before=
                    request.q_before,

                target_masks=
                    request.target_masks,

                pending_pieces=
                    request.pending(),

                node_cap=
                    self.node_cap,

                time_cap_seconds=
                    self.time_cap_seconds,

                rss_growth_cap_mib=
                    self.rss_growth_cap_mib,
            )

        except ResourceCapUnknown as exc:
            first_wall = (
                perf_counter()
                - started
            )

            return self._fallback(
                request=
                    request,

                plan=
                    plan,

                signature=
                    signature,

                first_status=
                    "UNKNOWN_RESOURCE",

                first_reason=
                    str(
                        exc
                    ),

                first_wall_s=
                    first_wall,

                first_metrics={},
            )

        first_wall = (
            perf_counter()
            - started
        )

        first_metrics = dict(
            result
        )

        status = str(
            result.get(
                "status",
                "",
            )
        )

        witness_raw = result.get(
            "witness"
        )

        if (
            status != "SAT"
            or witness_raw is None
        ):
            return self._fallback(
                request=
                    request,

                plan=
                    plan,

                signature=
                    signature,

                first_status=
                    status
                    or "NO_WITNESS",

                first_reason=
                    None,

                first_wall_s=
                    first_wall,

                first_metrics=
                    first_metrics,
            )

        witness = tuple(
            str(v)
            for v in witness_raw
        )

        move_ids = _validate_first_witness(
            request,
            witness,
        )

        singleton_started = (
            perf_counter()
        )

        singleton = _singleton_language(
            request=
                request,

            witness=
                witness,

            move_ids=
                move_ids,
        )

        singleton_wall = (
            perf_counter()
            - singleton_started
        )

        self.first_sat += 1

        total_wall = (
            perf_counter()
            - started
        )

        return ClosureResult(
            status=
                COMPLETE,

            backend=
                FIRST_WITNESS,

            signature=
                signature,

            language=
                singleton,

            trace=[
                {
                    "op":
                        "FIRST_WITNESS",

                    "plan":
                        plan.name,

                    "witness":
                        list(
                            witness
                        ),

                    "product_states_created":
                        int(
                            result.get(
                                "product_states_created",
                                0,
                            )
                        ),

                    "product_edges_created":
                        int(
                            result.get(
                                "product_edges_created",
                                0,
                            )
                        ),

                    "adapter_prep_wall_s":
                        float(
                            result.get(
                                "adapter_prep_wall_s",
                                0.0,
                            )
                        ),

                    "packed_engine_wall_s":
                        float(
                            result.get(
                                "packed_engine_wall_s",
                                0.0,
                            )
                        ),

                    "singleton_wall_s":
                        float(
                            singleton_wall
                        ),
                },
            ],

            timings={
                "adapter_prep_s":
                    float(
                        result.get(
                            "adapter_prep_wall_s",
                            0.0,
                        )
                    ),

                "packed_engine_s":
                    float(
                        result.get(
                            "packed_engine_wall_s",
                            0.0,
                        )
                    ),

                "singleton_s":
                    float(
                        singleton_wall
                    ),

                "wall_s":
                    float(
                        total_wall
                    ),
            },

            pending_after=(),

            reason=
                None,
        )

    def execute(
        self,
        request: ClosureRequest,
        *,
        force_backend: str | None = None,
    ) -> ClosureResult:
        selected, signature, _name = (
            self.select_backend(
                request
            )
        )

        backend = (
            selected
            if force_backend is None
            else str(
                force_backend
            )
        )

        if backend != FIRST_WITNESS:
            return self.base.execute(
                request,
                force_backend=
                    force_backend,
            )

        plan = self.first_witness.get(
            signature
        )

        if plan is None:
            raise ValueError(
                "no exact first-witness plan for forced backend"
            )

        return self._first(
            request=
                request,

            signature=
                signature,

            plan=
                plan,
        )


__all__ = [
    "FIRST_WITNESS",
    "FirstWitnessPlan",
    "FirstWitnessExactClosureRouter",
    "factorless_request",
    "factorless_signature",
]
