#!/usr/bin/env python3
"""
CubeLab v37.458-A14-A9-P0
EXACT SHARED-WORD BITMAP KERNEL

This module compiles CubeLab's SAME-word constraint itself into bitmaps.

A bit does NOT represent one move.
A bit represents one complete internally-canonical word of local length k.

Supported local windows:
    k = 2, 3, 4

No overall solution length is configured or implied by k.

For each k:
- WORDS[k]                 exact canonical k-word universe
- POS[k][position][move]   words using move at that position
- START[k][context]        words canonical after previous-face context
- NEXT[k][next_move]       words canonical before next boundary move
- REL[k][relation_class]   words that take one exact piece relation class
                           to exact current==target in exactly k moves

REL is compiled from the already-proven A14-P0 480-class exact relation
transition algebra.  Runtime exact-boundary full-Q query is:

    G = START & NEXT
    for 20 pieces:
        G &= REL[class(piece,current_q,target_q)]

For a wider per-piece boundary:
    left pose-set L
    right pose-set R

the exact-necessary piece word set is:

    P = OR REL[class(piece, ql, qr)]
        for ql in L, qr in R

and twenty such P masks are ANDed.

This is the intended CubeLab bit architecture:
    OR  = alternatives inside one exact constraint
    AND = SAME-word synchronization across exact constraints
    XOR = delta/change telemetry
    AND-NOT = exact removed candidates
    bit_count = exact candidate count

Python `int` is deliberately used for the hot bitmap algebra.  The underlying
multi-limb AND/OR/XOR/popcount executes in C.

Authority
---------
Exact boundary:
    GLOBAL bitmap is the exact set of k-move full-Q words.

Pose-set boundary:
    per-piece bitmap is exact for that piece's allowed boundary sets.
    AND across pieces is exact-necessary, as in independent ATO; positive is
    not full-Q SAT authority because independently chosen boundary poses may
    not form one common cube state.

No P5, pair coupling, heuristic, fixed prefix, global H, or solution-length
assumption lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cubelab.global_field import ato_bidirectional_v37_458 as bi
from cubelab.global_field.independent_ato_column_insertion_incremental_v37_458 import (
    CANONICAL_ALLOW,
    CONTEXTS,
    CONTEXT_INDEX,
    MOVE_CONTEXT,
    MOVE_COUNT,
    PIECE_COUNT,
    PIECES,
)
from cubelab.global_field.per_piece_route_readout_v37_458 import (
    LoadedRouteAtlas,
)


CLASS_COUNT = 480
POSE_COUNT = 24
SUPPORTED_K = (2, 3, 4)


def _bool_to_int(
    values: np.ndarray,
) -> int:
    arr = np.asarray(
        values,
        dtype=np.bool_,
    ).reshape(
        -1
    )

    if arr.size == 0:
        return 0

    packed = np.packbits(
        arr,
        bitorder="little",
    )

    return int.from_bytes(
        packed.tobytes(),
        byteorder="little",
        signed=False,
    )


def _packed_row_to_int(
    row: np.ndarray,
) -> int:
    return int.from_bytes(
        np.ascontiguousarray(
            row,
            dtype=np.uint8,
        ).tobytes(),
        byteorder="little",
        signed=False,
    )


def _set_bit_indices(
    value: int,
) -> tuple[int, ...]:
    x = int(value)
    out = []

    while x:
        lsb = (
            x
            & -x
        )

        out.append(
            int(
                lsb.bit_length()
                - 1
            )
        )

        x ^= lsb

    return tuple(
        out
    )


def generate_internal_canonical_words(
    k: int,
) -> np.ndarray:
    """
    All internally canonical k-move words.

    Previous/next external boundary contexts are NOT fixed here.  They are
    separate START/NEXT bitmaps so one universe serves all contexts.
    """
    depth = int(
        k
    )

    if depth < 1:
        raise ValueError(
            "word length k must be >=1"
        )

    words = np.arange(
        MOVE_COUNT,
        dtype=np.uint8,
    ).reshape(
        MOVE_COUNT,
        1,
    )

    move_ids = np.arange(
        MOVE_COUNT,
        dtype=np.intp,
    )

    for _position in range(
        1,
        depth,
    ):
        last = words[
            :,
            -1,
        ].astype(
            np.intp,
            copy=False,
        )

        last_context = MOVE_CONTEXT[
            last
        ].astype(
            np.intp,
            copy=False,
        )

        valid = CANONICAL_ALLOW[
            last_context[
                :,
                None,
            ],
            move_ids[
                None,
                :,
            ],
        ]

        parent_index, next_mid = np.nonzero(
            valid
        )

        words = np.concatenate(
            (
                words[
                    parent_index
                ],
                next_mid.astype(
                    np.uint8,
                    copy=False,
                )[
                    :,
                    None,
                ],
            ),
            axis=1,
        )

    return np.ascontiguousarray(
        words,
        dtype=np.uint8,
    )


def _context_valid_bool(
    words: np.ndarray,
    *,
    previous_face: str | None,
    next_move: str | None,
) -> np.ndarray:
    w = np.asarray(
        words,
        dtype=np.uint8,
    )

    n = int(
        w.shape[
            0
        ]
    )

    valid = np.ones(
        n,
        dtype=np.bool_,
    )

    start_ctx = int(
        CONTEXT_INDEX[
            previous_face
        ]
    )

    valid &= CANONICAL_ALLOW[
        start_ctx,
        w[
            :,
            0,
        ].astype(
            np.intp,
            copy=False,
        ),
    ]

    if next_move is not None:
        next_mid = int(
            bi.move_id(
                next_move
            )
        )

        last_ctx = MOVE_CONTEXT[
            w[
                :,
                -1,
            ].astype(
                np.intp,
                copy=False,
            )
        ].astype(
            np.intp,
            copy=False,
        )

        valid &= CANONICAL_ALLOW[
            last_ctx,
            next_mid,
        ]

    return valid


@dataclass(frozen=True)
class WordBitmapLevel:
    k: int
    words: np.ndarray              # [W,k] uint8

    relation_words: tuple[int, ...]  # len 480 Python ints
    position_move: tuple[
        tuple[int, ...],
        ...,
    ]  # [k][18]

    start_context: tuple[int, ...]   # [7]
    next_move: tuple[int, ...]       # [19], index0=None, index mid+1

    all_mask: int
    packed_bytes_per_bitmap: int

    @property
    def word_count(
        self,
    ) -> int:
        return int(
            self.words.shape[
                0
            ]
        )

    def context_mask(
        self,
        *,
        previous_face: str | None,
        next_move: str | None,
    ) -> int:
        start = int(
            self.start_context[
                int(
                    CONTEXT_INDEX[
                        previous_face
                    ]
                )
            ]
        )

        if next_move is None:
            end = int(
                self.next_move[
                    0
                ]
            )
        else:
            end = int(
                self.next_move[
                    int(
                        bi.move_id(
                            next_move
                        )
                    )
                    + 1
                ]
            )

        return int(
            start
            & end
        )

    def project_support(
        self,
        candidate_bitmap: int,
    ) -> tuple[int, ...]:
        """
        Exact per-position move support of the candidate word set.
        """
        candidates = int(
            candidate_bitmap
        )

        supports = []

        for position in range(
            int(
                self.k
            )
        ):
            mask = 0

            for mid in range(
                MOVE_COUNT
            ):
                if candidates & int(
                    self.position_move[
                        position
                    ][
                        mid
                    ]
                ):
                    mask |= (
                        1
                        << mid
                    )

            supports.append(
                int(
                    mask
                )
            )

        return tuple(
            supports
        )

    def first_word(
        self,
        candidate_bitmap: int,
    ) -> tuple[str, ...] | None:
        value = int(
            candidate_bitmap
        )

        if value == 0:
            return None

        bit = (
            value
            & -value
        )

        index = int(
            bit.bit_length()
            - 1
        )

        return tuple(
            str(
                bi.MOVE_NAMES[
                    int(
                        mid
                    )
                ]
            )
            for mid in self.words[
                index
            ]
        )


@dataclass(frozen=True)
class SharedWordBitmapAtlas:
    route_class: np.ndarray   # [20,576] uint16
    levels: dict[
        int,
        WordBitmapLevel,
    ]
    asset_path: Path | None = None

    def level(
        self,
        k: int,
    ) -> WordBitmapLevel:
        depth = int(
            k
        )

        if depth not in self.levels:
            raise KeyError(
                f"word bitmap k={depth} not loaded"
            )

        return self.levels[
            depth
        ]

    def exact_boundary_classes(
        self,
        q_left,
        q_right,
    ) -> np.ndarray:
        ql = bi.normalize_q(
            q_left
        )
        qr = bi.normalize_q(
            q_right
        )

        pair_index = (
            ql.astype(
                np.intp,
                copy=False,
            )
            * POSE_COUNT
            + qr.astype(
                np.intp,
                copy=False,
            )
        )

        return self.route_class[
            PIECES,
            pair_index,
        ].astype(
            np.uint16,
            copy=False,
        )

    def exact_boundary_bitmap(
        self,
        *,
        k: int,
        q_left,
        q_right,
        previous_face: str | None,
        next_move: str | None,
    ) -> int:
        level = self.level(
            k
        )

        value = level.context_mask(
            previous_face=previous_face,
            next_move=next_move,
        )

        if value == 0:
            return 0

        classes = self.exact_boundary_classes(
            q_left,
            q_right,
        )

        for cid in classes:
            value &= int(
                level.relation_words[
                    int(
                        cid
                    )
                ]
            )

            if value == 0:
                break

        return int(
            value
        )

    def piece_pose_set_bitmap(
        self,
        *,
        k: int,
        piece: int,
        left_pose_bits: int,
        right_pose_bits: int,
        previous_face: str | None,
        next_move: str | None,
    ) -> int:
        """
        Exact k-word set for one piece with allowed pose sets at both
        boundaries.
        """
        level = self.level(
            k
        )

        left = _set_bit_indices(
            int(
                left_pose_bits
            )
        )
        right = _set_bit_indices(
            int(
                right_pose_bits
            )
        )

        if (
            not left
            or not right
        ):
            return 0

        relation_matrix = self.route_class[
            int(
                piece
            )
        ].reshape(
            POSE_COUNT,
            POSE_COUNT,
        )

        classes = np.unique(
            relation_matrix[
                np.ix_(
                    np.asarray(
                        left,
                        dtype=np.intp,
                    ),
                    np.asarray(
                        right,
                        dtype=np.intp,
                    ),
                )
            ]
        )

        value = 0

        for cid in classes:
            value |= int(
                level.relation_words[
                    int(
                        cid
                    )
                ]
            )

        value &= level.context_mask(
            previous_face=previous_face,
            next_move=next_move,
        )

        return int(
            value
        )

    def independent_pose_set_global_bitmap(
        self,
        *,
        k: int,
        left_pose_bits,
        right_pose_bits,
        previous_face: str | None,
        next_move: str | None,
    ) -> int:
        """
        Exact-necessary SAME-word bitmap for 20 independent piece boundary
        pose sets.
        """
        left = tuple(
            int(x)
            for x in left_pose_bits
        )
        right = tuple(
            int(x)
            for x in right_pose_bits
        )

        if (
            len(
                left
            )
            != PIECE_COUNT
            or len(
                right
            )
            != PIECE_COUNT
        ):
            raise ValueError(
                "pose-set boundary must have 20 piece bitsets"
            )

        level = self.level(
            k
        )

        value = level.context_mask(
            previous_face=previous_face,
            next_move=next_move,
        )

        if value == 0:
            return 0

        for piece in range(
            PIECE_COUNT
        ):
            piece_value = self.piece_pose_set_bitmap(
                k=k,
                piece=piece,
                left_pose_bits=left[
                    piece
                ],
                right_pose_bits=right[
                    piece
                ],
                previous_face=previous_face,
                next_move=next_move,
            )

            value &= int(
                piece_value
            )

            if value == 0:
                break

        return int(
            value
        )


def compile_level_arrays(
    *,
    k: int,
    route_atlas: LoadedRouteAtlas,
) -> dict[str, np.ndarray]:
    words = generate_internal_canonical_words(
        int(
            k
        )
    )

    W = int(
        words.shape[
            0
        ]
    )

    packed_bytes = int(
        (
            W
            + 7
        )
        // 8
    )

    # Exact relation-class transition for every start class and every word.
    states = np.broadcast_to(
        np.arange(
            CLASS_COUNT,
            dtype=np.uint16,
        )[
            :,
            None,
        ],
        (
            CLASS_COUNT,
            W,
        ),
    ).copy()

    for position in range(
        int(
            k
        )
    ):
        mids = words[
            :,
            position,
        ].astype(
            np.intp,
            copy=False,
        )

        states = route_atlas.left_next_class[
            states.astype(
                np.intp,
                copy=False,
            ),
            mids[
                None,
                :,
            ],
        ].astype(
            np.uint16,
            copy=False,
        )

    success = (
        route_atlas.class_distance[
            states.astype(
                np.intp,
                copy=False,
            )
        ]
        == 0
    )

    relation_packed = np.packbits(
        success,
        axis=1,
        bitorder="little",
    ).astype(
        np.uint8,
        copy=False,
    )

    if relation_packed.shape != (
        CLASS_COUNT,
        packed_bytes,
    ):
        raise RuntimeError(
            "A14_A9_P0_RELATION_PACKED_SHAPE_DRIFT "
            f"k={k} got={relation_packed.shape} "
            f"expected={(CLASS_COUNT,packed_bytes)}"
        )

    position_packed = np.empty(
        (
            int(
                k
            ),
            MOVE_COUNT,
            packed_bytes,
        ),
        dtype=np.uint8,
    )

    for position in range(
        int(
            k
        )
    ):
        for mid in range(
            MOVE_COUNT
        ):
            position_packed[
                position,
                mid,
            ] = np.packbits(
                words[
                    :,
                    position,
                ]
                == int(
                    mid
                ),
                bitorder="little",
            )

    start_packed = np.empty(
        (
            len(
                CONTEXTS
            ),
            packed_bytes,
        ),
        dtype=np.uint8,
    )

    for ctx in range(
        len(
            CONTEXTS
        )
    ):
        start_packed[
            ctx
        ] = np.packbits(
            CANONICAL_ALLOW[
                int(
                    ctx
                ),
                words[
                    :,
                    0,
                ].astype(
                    np.intp,
                    copy=False,
                ),
            ],
            bitorder="little",
        )

    next_packed = np.empty(
        (
            MOVE_COUNT
            + 1,
            packed_bytes,
        ),
        dtype=np.uint8,
    )

    next_packed[
        0
    ] = np.packbits(
        np.ones(
            W,
            dtype=np.bool_,
        ),
        bitorder="little",
    )

    last_context = MOVE_CONTEXT[
        words[
            :,
            -1,
        ].astype(
            np.intp,
            copy=False,
        )
    ].astype(
        np.intp,
        copy=False,
    )

    for next_mid in range(
        MOVE_COUNT
    ):
        next_packed[
            next_mid
            + 1
        ] = np.packbits(
            CANONICAL_ALLOW[
                last_context,
                int(
                    next_mid
                ),
            ],
            bitorder="little",
        )

    return {
        "words": words,
        "relation_packed": relation_packed,
        "position_packed": position_packed,
        "start_packed": start_packed,
        "next_packed": next_packed,
        "word_count": np.asarray(
            [
                W
            ],
            dtype=np.uint32,
        ),
        "packed_bytes": np.asarray(
            [
                packed_bytes
            ],
            dtype=np.uint32,
        ),
    }


def build_word_bitmap_asset(
    *,
    route_atlas_path: Path,
    output_path: Path,
    ks=SUPPORTED_K,
) -> dict[str, Any]:
    import time

    atlas = LoadedRouteAtlas.load(
        Path(
            route_atlas_path
        )
    )

    payload: dict[
        str,
        np.ndarray,
    ] = {
        "schema_version": np.asarray(
            [
                1
            ],
            dtype=np.uint16,
        ),
        "route_class": np.asarray(
            atlas.route_class,
            dtype=np.uint16,
        ),
        "move_names": np.asarray(
            bi.MOVE_NAMES,
            dtype="<U3",
        ),
        "supported_k": np.asarray(
            [
                int(x)
                for x in ks
            ],
            dtype=np.uint8,
        ),
    }

    per_level = {}
    total_t0 = time.perf_counter()

    for k in ks:
        t0 = time.perf_counter()

        row = compile_level_arrays(
            k=int(
                k
            ),
            route_atlas=atlas,
        )

        wall = (
            time.perf_counter()
            - t0
        )

        payload[
            f"words_k{k}"
        ] = row[
            "words"
        ]

        payload[
            f"relation_packed_k{k}"
        ] = row[
            "relation_packed"
        ]

        payload[
            f"position_packed_k{k}"
        ] = row[
            "position_packed"
        ]

        payload[
            f"start_packed_k{k}"
        ] = row[
            "start_packed"
        ]

        payload[
            f"next_packed_k{k}"
        ] = row[
            "next_packed"
        ]

        payload[
            f"word_count_k{k}"
        ] = row[
            "word_count"
        ]

        payload[
            f"packed_bytes_k{k}"
        ] = row[
            "packed_bytes"
        ]

        payload[
            f"build_wall_s_k{k}"
        ] = np.asarray(
            [
                float(
                    wall
                )
            ],
            dtype=np.float64,
        )

        per_level[
            str(
                int(
                    k
                )
            )
        ] = {
            "word_count": int(
                row[
                    "words"
                ].shape[
                    0
                ]
            ),
            "packed_bytes_per_bitmap": int(
                row[
                    "relation_packed"
                ].shape[
                    1
                ]
            ),
            "relation_payload_MiB": float(
                row[
                    "relation_packed"
                ].nbytes
                / (
                    1024
                    ** 2
                )
            ),
            "build_wall_s": float(
                wall
            ),
        }

    output = Path(
        output_path
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        output,
        **payload,
    )

    return {
        "asset": str(
            output
        ),
        "asset_bytes": int(
            output.stat().st_size
        ),
        "asset_MiB": float(
            output.stat().st_size
            / (
                1024
                ** 2
            )
        ),
        "build_wall_s": float(
            time.perf_counter()
            - total_t0
        ),
        "levels": per_level,
    }


def load_word_bitmap_asset(
    path: Path,
) -> SharedWordBitmapAtlas:
    source = Path(
        path
    )

    with np.load(
        source,
        allow_pickle=False,
    ) as data:
        if int(
            np.asarray(
                data[
                    "schema_version"
                ]
            ).reshape(
                -1
            )[
                0
            ]
        ) != 1:
            raise RuntimeError(
                "A14_A9_P0_WORD_BITMAP_SCHEMA_DRIFT"
            )

        move_names = tuple(
            str(x)
            for x in np.asarray(
                data[
                    "move_names"
                ]
            )
        )

        if move_names != tuple(
            bi.MOVE_NAMES
        ):
            raise RuntimeError(
                "A14_A9_P0_MOVE_ORDER_DRIFT "
                f"asset={move_names} runtime={bi.MOVE_NAMES}"
            )

        route_class = np.asarray(
            data[
                "route_class"
            ],
            dtype=np.uint16,
        ).copy()

        if route_class.shape != (
            PIECE_COUNT,
            POSE_COUNT
            * POSE_COUNT,
        ):
            raise RuntimeError(
                "A14_A9_P0_ROUTE_CLASS_SHAPE_DRIFT "
                f"{route_class.shape}"
            )

        levels = {}

        for k in np.asarray(
            data[
                "supported_k"
            ],
            dtype=np.uint8,
        ):
            depth = int(
                k
            )

            words = np.asarray(
                data[
                    f"words_k{depth}"
                ],
                dtype=np.uint8,
            ).copy()

            relation_packed = np.asarray(
                data[
                    f"relation_packed_k{depth}"
                ],
                dtype=np.uint8,
            )

            position_packed = np.asarray(
                data[
                    f"position_packed_k{depth}"
                ],
                dtype=np.uint8,
            )

            start_packed = np.asarray(
                data[
                    f"start_packed_k{depth}"
                ],
                dtype=np.uint8,
            )

            next_packed = np.asarray(
                data[
                    f"next_packed_k{depth}"
                ],
                dtype=np.uint8,
            )

            W = int(
                words.shape[
                    0
                ]
            )

            packed_bytes = int(
                (
                    W
                    + 7
                )
                // 8
            )

            if relation_packed.shape != (
                CLASS_COUNT,
                packed_bytes,
            ):
                raise RuntimeError(
                    "A14_A9_P0_RELATION_LOAD_SHAPE_DRIFT "
                    f"k={depth} got={relation_packed.shape}"
                )

            relation_words = tuple(
                _packed_row_to_int(
                    relation_packed[
                        cid
                    ]
                )
                for cid in range(
                    CLASS_COUNT
                )
            )

            position_move = tuple(
                tuple(
                    _packed_row_to_int(
                        position_packed[
                            position,
                            mid,
                        ]
                    )
                    for mid in range(
                        MOVE_COUNT
                    )
                )
                for position in range(
                    depth
                )
            )

            start_context = tuple(
                _packed_row_to_int(
                    start_packed[
                        ctx
                    ]
                )
                for ctx in range(
                    len(
                        CONTEXTS
                    )
                )
            )

            next_move = tuple(
                _packed_row_to_int(
                    next_packed[
                        index
                    ]
                )
                for index in range(
                    MOVE_COUNT
                    + 1
                )
            )

            all_mask = (
                (
                    1
                    << W
                )
                - 1
            )

            levels[
                depth
            ] = WordBitmapLevel(
                k=int(
                    depth
                ),
                words=words,
                relation_words=relation_words,
                position_move=position_move,
                start_context=start_context,
                next_move=next_move,
                all_mask=int(
                    all_mask
                ),
                packed_bytes_per_bitmap=int(
                    packed_bytes
                ),
            )

    return SharedWordBitmapAtlas(
        route_class=route_class,
        levels=levels,
        asset_path=source,
    )


def direct_full_q_bitmap(
    *,
    level: WordBitmapLevel,
    q_left,
    q_right,
    previous_face: str | None,
    next_move: str | None,
) -> int:
    """
    Independent parity reference using raw Q_NEXT replay for every word.
    """
    ql = bi.normalize_q(
        q_left
    )
    qr = bi.normalize_q(
        q_right
    )

    words = level.words
    W = int(
        words.shape[
            0
        ]
    )

    states = np.broadcast_to(
        ql[
            None,
            :,
        ],
        (
            W,
            PIECE_COUNT,
        ),
    ).copy()

    piece_axis = np.arange(
        PIECE_COUNT,
        dtype=np.intp,
    )[
        None,
        :,
    ]

    for position in range(
        int(
            level.k
        )
    ):
        mids = words[
            :,
            position,
        ].astype(
            np.intp,
            copy=False,
        )[
            :,
            None,
        ]

        states = bi.Q_NEXT[
            piece_axis,
            states.astype(
                np.intp,
                copy=False,
            ),
            mids,
        ].astype(
            np.uint8,
            copy=False,
        )

    success = np.all(
        states
        == qr[
            None,
            :,
        ],
        axis=1,
    )

    success &= _context_valid_bool(
        words,
        previous_face=previous_face,
        next_move=next_move,
    )

    return _bool_to_int(
        success
    )


def direct_piece_pose_set_bitmap(
    *,
    level: WordBitmapLevel,
    piece: int,
    left_pose_bits: int,
    right_pose_bits: int,
    previous_face: str | None,
    next_move: str | None,
) -> int:
    """
    Independent raw Q_NEXT reference for one piece and two allowed pose sets.
    """
    left = _set_bit_indices(
        int(
            left_pose_bits
        )
    )

    right = _set_bit_indices(
        int(
            right_pose_bits
        )
    )

    if (
        not left
        or not right
    ):
        return 0

    words = level.words
    W = int(
        words.shape[
            0
        ]
    )

    starts = np.asarray(
        left,
        dtype=np.uint8,
    )

    states = np.broadcast_to(
        starts[
            None,
            :,
        ],
        (
            W,
            starts.shape[
                0
            ],
        ),
    ).copy()

    for position in range(
        int(
            level.k
        )
    ):
        mids = words[
            :,
            position,
        ].astype(
            np.intp,
            copy=False,
        )[
            :,
            None,
        ]

        states = bi.Q_NEXT[
            int(
                piece
            ),
            states.astype(
                np.intp,
                copy=False,
            ),
            mids,
        ].astype(
            np.uint8,
            copy=False,
        )

    right_allowed = np.zeros(
        POSE_COUNT,
        dtype=np.bool_,
    )

    right_allowed[
        np.asarray(
            right,
            dtype=np.intp,
        )
    ] = True

    success = np.any(
        right_allowed[
            states.astype(
                np.intp,
                copy=False,
            )
        ],
        axis=1,
    )

    success &= _context_valid_bool(
        words,
        previous_face=previous_face,
        next_move=next_move,
    )

    return _bool_to_int(
        success
    )
