from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable

from .exact_live_estimator import (
    OneLiveEstimate,
    estimate_prepared_one,
)

from .prepared_q_restrict import (
    PreparedPieceMask,
    prepare_piece_mask,
)


@dataclass(frozen=True, slots=True)
class ExactOneQuote:

    piece: int

    prepared: PreparedPieceMask
    estimate: OneLiveEstimate


    @property
    def exact_states(self) -> int:

        return int(
            self.estimate.exact_product_states
        )


    @property
    def exact_edges(self) -> int:

        return int(
            self.estimate.exact_edge_examinations
        )


    @property
    def root_live(self) -> bool:

        return bool(
            self.prepared.root_live
        )


    @property
    def quote_wall_s(self) -> float:

        return float(
            self.prepared.prepare_wall_s
            +
            self.estimate.estimator_wall_s
        )


def quote_exact_one_candidates(
    language,

    *,
    pieces: Iterable[int],

    rewrite,
    q_before,
    targets,

    input_node_cap: int = 1_500_000,

    per_candidate_deadline_s: float = 60.0,

) -> tuple[
    tuple[ExactOneQuote, ...],
    dict[str, Any],
]:
    """
    Quote every pending mandatory ONE on the SAME exact residual.

    No output MDD is materialized.

    exact_states / exact_edges are exact for direct ONE execution.
    """

    started = perf_counter()


    normalized = tuple(
        sorted(
            {
                int(v)
                for v in pieces
            }
        )
    )


    quotes = []

    rows = []


    for piece in normalized:

        prepared = prepare_piece_mask(
            language,

            rewrite.q_next[
                piece
            ],

            int(
                q_before[
                    piece
                ]
            ),

            int(
                targets[
                    piece
                ]
            ),

            input_node_cap=
                int(
                    input_node_cap
                ),

            deadline=(
                perf_counter()
                +
                float(
                    per_candidate_deadline_s
                )
            ),
        )


        estimate = estimate_prepared_one(
            prepared,

            deadline=(
                perf_counter()
                +
                float(
                    per_candidate_deadline_s
                )
            ),
        )


        quote = ExactOneQuote(
            piece=
                piece,

            prepared=
                prepared,

            estimate=
                estimate,
        )


        quotes.append(
            quote
        )


        rows.append({
            "piece":
                piece,

            "root_live":
                quote.root_live,

            "exact_states":
                quote.exact_states,

            "exact_edges":
                quote.exact_edges,

            "prepare_wall_s":
                float(
                    prepared.prepare_wall_s
                ),

            "estimator_wall_s":
                float(
                    estimate.estimator_wall_s
                ),

            "quote_wall_s":
                quote.quote_wall_s,
        })


    quotes.sort(
        key=lambda quote:(
            quote.exact_edges,
            quote.exact_states,
            quote.piece,
        )
    )


    return (
        tuple(
            quotes
        ),

        {
            "schema":
                "cubelab.exact-one-quote-portfolio.v1",

            "candidate_count":
                len(
                    quotes
                ),

            "wall_s":(
                perf_counter()
                - started
            ),

            "rows":
                rows,
        },
    )


def choose_exact_min_work(
    quotes: Iterable[ExactOneQuote],
) -> ExactOneQuote:
    """
    Deterministic scheduling authority only.

    No semantic-deletion authority.
    No solution-branch authority.
    """

    quotes = tuple(
        quotes
    )


    if not quotes:

        raise ValueError(
            "no ONE candidates"
        )


    dead = [
        quote.piece
        for quote in quotes
        if not quote.root_live
    ]


    if dead:

        raise ValueError(
            "mandatory candidate has exact-empty intersection: "
            +
            repr(
                tuple(
                    dead
                )
            )
        )


    return min(
        quotes,

        key=lambda quote:(
            quote.exact_edges,
            quote.exact_states,
            quote.piece,
        ),
    )


__all__ = [
    "ExactOneQuote",
    "choose_exact_min_work",
    "quote_exact_one_candidates",
]
