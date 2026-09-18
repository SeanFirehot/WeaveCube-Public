#!/usr/bin/env python3
"""
CubeLab v37.458-A14-P1
PER-PIECE ROUTE READOUT

Loads the frozen A14-P0 resident route atlas and turns one exact full-Q
current/target boundary into a 20 x 18 response matrix by pure lookup.

No pair coupling.

For one exact cube boundary:
    class_id[piece]
    local_distance[piece]
    response[piece, move] in I/A/N/R
    next_relation[piece, move]

are available without replaying 18 moves per piece.

The readout is an exact information surface only.  No move-count aggregation
is a hard search authority in P1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cubelab.global_field import ato_bidirectional_v37_458 as bi
from cubelab.global_field.per_piece_route_atlas_v37_458 import (
    ALL_MOVE_MASK,
    MOVE_COUNT,
    ROLE_A,
    ROLE_I,
    ROLE_N,
    ROLE_R,
    ROLE_NAMES,
)


PIECES = np.arange(20, dtype=np.intp)


@dataclass(frozen=True)
class LoadedRouteAtlas:
    route_class: np.ndarray
    raw_distance: np.ndarray
    class_kind: np.ndarray
    class_current_position: np.ndarray
    class_target_position: np.ndarray
    class_orientation_slot: np.ndarray
    class_distance: np.ndarray
    left_next_class: np.ndarray
    left_response_code: np.ndarray
    left_role_mask: np.ndarray
    right_next_class: np.ndarray
    right_response_code: np.ndarray
    right_role_mask: np.ndarray

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> "LoadedRouteAtlas":
        with np.load(
            Path(path),
            allow_pickle=False,
        ) as data:
            required = {
                "route_class",
                "raw_distance",
                "class_kind",
                "class_current_position",
                "class_target_position",
                "class_orientation_slot",
                "class_distance",
                "left_next_class",
                "left_response_code",
                "left_role_mask",
                "right_next_class",
                "right_response_code",
                "right_role_mask",
            }

            missing = sorted(
                required - set(data.files)
            )

            if missing:
                raise RuntimeError(
                    "A14_P1_ROUTE_ASSET_KEYS_MISSING "
                    f"{missing}"
                )

            arrays = {
                key: np.asarray(data[key])
                for key in required
            }

        return cls(
            route_class=np.asarray(
                arrays["route_class"],
                dtype=np.uint16,
            ),
            raw_distance=np.asarray(
                arrays["raw_distance"],
                dtype=np.uint8,
            ),
            class_kind=np.asarray(
                arrays["class_kind"],
                dtype=np.uint8,
            ),
            class_current_position=np.asarray(
                arrays["class_current_position"],
                dtype=np.uint8,
            ),
            class_target_position=np.asarray(
                arrays["class_target_position"],
                dtype=np.uint8,
            ),
            class_orientation_slot=np.asarray(
                arrays["class_orientation_slot"],
                dtype=np.uint8,
            ),
            class_distance=np.asarray(
                arrays["class_distance"],
                dtype=np.uint8,
            ),
            left_next_class=np.asarray(
                arrays["left_next_class"],
                dtype=np.uint16,
            ),
            left_response_code=np.asarray(
                arrays["left_response_code"],
                dtype=np.uint8,
            ),
            left_role_mask=np.asarray(
                arrays["left_role_mask"],
                dtype=np.uint32,
            ),
            right_next_class=np.asarray(
                arrays["right_next_class"],
                dtype=np.uint16,
            ),
            right_response_code=np.asarray(
                arrays["right_response_code"],
                dtype=np.uint8,
            ),
            right_role_mask=np.asarray(
                arrays["right_role_mask"],
                dtype=np.uint32,
            ),
        )


@dataclass(frozen=True)
class RouteReadout:
    q_current: np.ndarray
    q_target: np.ndarray

    relation_class: np.ndarray       # [20] uint16
    local_distance: np.ndarray       # [20] uint8
    current_position: np.ndarray     # [20] uint8
    target_position: np.ndarray      # [20] uint8
    orientation_slot: np.ndarray     # [20] uint8

    response_code: np.ndarray        # [20,18] uint8
    role_mask: np.ndarray            # [20,4] uint32
    next_relation_class: np.ndarray  # [20,18] uint16

    from_target_side: bool

    def move_role_counts(
        self,
    ) -> np.ndarray:
        """
        [18,4] counts of pieces with I/A/N/R response to each move.
        Diagnostic only.
        """
        out = np.zeros(
            (MOVE_COUNT, 4),
            dtype=np.uint8,
        )

        for role in range(4):
            out[:, role] = np.count_nonzero(
                self.response_code == role,
                axis=0,
            ).astype(np.uint8)

        return out

    def unresolved_piece_count(
        self,
    ) -> int:
        return int(
            np.count_nonzero(
                self.local_distance > 0
            )
        )

    def class_rows_json(
        self,
    ) -> list[dict[str, Any]]:
        rows = []

        for piece in range(20):
            rows.append({
                "piece": int(piece),
                "relation_class": int(
                    self.relation_class[piece]
                ),
                "kind": (
                    "CORNER"
                    if int(piece) < 8
                    else "EDGE"
                ),
                "current_position": int(
                    self.current_position[piece]
                ),
                "target_position": int(
                    self.target_position[piece]
                ),
                "relative_orientation_slot": int(
                    self.orientation_slot[piece]
                ),
                "exact_local_distance": int(
                    self.local_distance[piece]
                ),
                "I_mask": int(
                    self.role_mask[piece, ROLE_I]
                ),
                "A_mask": int(
                    self.role_mask[piece, ROLE_A]
                ),
                "N_mask": int(
                    self.role_mask[piece, ROLE_N]
                ),
                "R_mask": int(
                    self.role_mask[piece, ROLE_R]
                ),
                "A_mask_is_global_hard_authority": False,
            })

        return rows


def route_readout(
    atlas: LoadedRouteAtlas,
    q_current: Iterable[int],
    q_target: Iterable[int],
    *,
    from_target_side: bool = False,
) -> RouteReadout:
    current = bi.normalize_q(
        q_current
    )
    target = bi.normalize_q(
        q_target
    )

    pair_ids = (
        current.astype(
            np.intp,
            copy=False,
        )
        * 24
        + target.astype(
            np.intp,
            copy=False,
        )
    )

    classes = atlas.route_class[
        PIECES,
        pair_ids,
    ].astype(
        np.uint16,
        copy=False,
    )

    class_idx = classes.astype(
        np.intp,
        copy=False,
    )

    response_table = (
        atlas.right_response_code
        if from_target_side
        else atlas.left_response_code
    )
    masks_table = (
        atlas.right_role_mask
        if from_target_side
        else atlas.left_role_mask
    )
    next_table = (
        atlas.right_next_class
        if from_target_side
        else atlas.left_next_class
    )

    return RouteReadout(
        q_current=current.copy(),
        q_target=target.copy(),
        relation_class=classes.copy(),
        local_distance=atlas.class_distance[
            class_idx
        ].copy(),
        current_position=atlas.class_current_position[
            class_idx
        ].copy(),
        target_position=atlas.class_target_position[
            class_idx
        ].copy(),
        orientation_slot=atlas.class_orientation_slot[
            class_idx
        ].copy(),
        response_code=response_table[
            class_idx
        ].copy(),
        role_mask=masks_table[
            class_idx
        ].copy(),
        next_relation_class=next_table[
            class_idx
        ].copy(),
        from_target_side=bool(
            from_target_side
        ),
    )


def direct_route_readout(
    atlas: LoadedRouteAtlas,
    q_current: Iterable[int],
    q_target: Iterable[int],
    *,
    from_target_side: bool = False,
) -> RouteReadout:
    """
    Reconstruct the same 20x18 information directly from Q_NEXT/Q_PREV.
    Used only as an audit/economics baseline.
    """
    current = bi.normalize_q(
        q_current
    )
    target = bi.normalize_q(
        q_target
    )

    classes = np.empty(
        20,
        dtype=np.uint16,
    )
    distance = np.empty(
        20,
        dtype=np.uint8,
    )
    response = np.empty(
        (20, MOVE_COUNT),
        dtype=np.uint8,
    )
    masks = np.zeros(
        (20, 4),
        dtype=np.uint32,
    )
    next_classes = np.empty(
        (20, MOVE_COUNT),
        dtype=np.uint16,
    )

    current_position = np.empty(
        20,
        dtype=np.uint8,
    )
    target_position = np.empty_like(
        current_position
    )
    orientation_slot = np.empty_like(
        current_position
    )

    for piece in range(20):
        qc = int(
            current[piece]
        )
        qt = int(
            target[piece]
        )
        pid = (
            qc * 24
            + qt
        )

        cid = int(
            atlas.route_class[
                piece,
                pid,
            ]
        )
        classes[
            piece
        ] = np.uint16(
            cid
        )

        d0 = int(
            atlas.raw_distance[
                piece,
                pid,
            ]
        )
        distance[
            piece
        ] = np.uint8(
            d0
        )

        current_position[
            piece
        ] = atlas.class_current_position[
            cid
        ]
        target_position[
            piece
        ] = atlas.class_target_position[
            cid
        ]
        orientation_slot[
            piece
        ] = atlas.class_orientation_slot[
            cid
        ]

        for mid in range(
            MOVE_COUNT
        ):
            if from_target_side:
                qt2 = int(
                    bi.Q_PREV[
                        piece,
                        qt,
                        mid,
                    ]
                )
                next_pid = (
                    qc * 24
                    + qt2
                )
                unchanged = (
                    qt2 == qt
                )
            else:
                qc2 = int(
                    bi.Q_NEXT[
                        piece,
                        qc,
                        mid,
                    ]
                )
                next_pid = (
                    qc2 * 24
                    + qt
                )
                unchanged = (
                    qc2 == qc
                )

            d1 = int(
                atlas.raw_distance[
                    piece,
                    next_pid,
                ]
            )

            if unchanged:
                role = ROLE_I
            else:
                delta = (
                    d1 - d0
                )

                if delta == -1:
                    role = ROLE_A
                elif delta == 0:
                    role = ROLE_N
                elif delta == 1:
                    role = ROLE_R
                else:
                    raise RuntimeError(
                        "A14_P1_DIRECT_DELTA_DRIFT "
                        f"piece={piece} "
                        f"q={qc}->{qt} move={mid} "
                        f"d={d0}->{d1}"
                    )

            response[
                piece,
                mid,
            ] = np.uint8(
                role
            )

            masks[
                piece,
                role,
            ] |= np.uint32(
                1 << mid
            )

            next_classes[
                piece,
                mid,
            ] = atlas.route_class[
                piece,
                next_pid,
            ]

    return RouteReadout(
        q_current=current.copy(),
        q_target=target.copy(),
        relation_class=classes,
        local_distance=distance,
        current_position=current_position,
        target_position=target_position,
        orientation_slot=orientation_slot,
        response_code=response,
        role_mask=masks,
        next_relation_class=next_classes,
        from_target_side=bool(
            from_target_side
        ),
    )


def readout_equal(
    a: RouteReadout,
    b: RouteReadout,
) -> bool:
    return bool(
        a.from_target_side
        == b.from_target_side
        and np.array_equal(
            a.relation_class,
            b.relation_class,
        )
        and np.array_equal(
            a.local_distance,
            b.local_distance,
        )
        and np.array_equal(
            a.current_position,
            b.current_position,
        )
        and np.array_equal(
            a.target_position,
            b.target_position,
        )
        and np.array_equal(
            a.orientation_slot,
            b.orientation_slot,
        )
        and np.array_equal(
            a.response_code,
            b.response_code,
        )
        and np.array_equal(
            a.role_mask,
            b.role_mask,
        )
        and np.array_equal(
            a.next_relation_class,
            b.next_relation_class,
        )
    )


def apply_move_q(
    q_state: Iterable[int],
    move_id: int,
) -> np.ndarray:
    q = bi.normalize_q(
        q_state
    )

    return np.asarray(
        bi.Q_NEXT[
            PIECES,
            q.astype(
                np.intp,
                copy=False,
            ),
            int(move_id),
        ],
        dtype=np.uint8,
    )
