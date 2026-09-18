#!/usr/bin/env python3
"""
CubeLab v37.458-A14-P0
PER-PIECE CURRENT->TARGET ROUTE ATLAS

This is the direct implementation of the research idea:

    "For a specific piece, current position/orientation -> target
     position/orientation has a finite set of exact move alternatives.
     Compile them before search and read them at runtime."

No pair coupling is used in this atlas.

State key
---------
Raw exact key:
    (piece, q_current, q_target)
    20 * 24 * 24 = 11,520 rows.

C0/C1-R1 already proved that these rows map to 480 exact relative relation
classes:
    corner: 8 current positions x 8 target positions x 3 relative orientations
    edge  : 12 current positions x 12 target positions x 2 relative orientations

Runtime can therefore do:
    relation_class = route_class[piece, q_current*24 + q_target]
    row = CLASS_ROUTE_TABLE[relation_class]

Compiled per relation class
---------------------------
- current physical position
- target physical position
- deterministic relative-orientation slot (3 corner / 2 edge)
- exact local distance current->target
- LEFT/current-side:
      next relation class for all 18 moves
      I/A/N/R response code for all 18 moves
      I/A/N/R masks
      local-shortest first-move mask (= A mask)
- RIGHT/target-side:
      next relation class when the target boundary is replayed backward
      I/A/N/R response code/masks
      local-shortest first-move mask

Response semantics are TARGET-RELATIVE:
    I = the moved boundary pose is physically unchanged
    A = exact local distance decreases by 1
    N = boundary pose changes but exact local distance is unchanged
    R = exact local distance increases by 1

Because one move is one graph edge, target-distance delta is exhaustively
required to be in {-1,0,+1}.

Important authority boundary
----------------------------
A_MASK is the exact set of first moves on a shortest route for THIS PIECE.
It is NOT a hard global move filter. A full-cube solution may temporarily keep
or increase one piece's local distance.

The hard/exact asset is the full 18-way transition/response table itself.
Search policy decides later how to combine the twenty piece rows.

No solution length, global H, materialized-column ceiling, heuristic, or
production change is contained here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cubelab.global_field import ato_bidirectional_v37_458 as bi


PAIR_COUNT = 24 * 24
CLASS_COUNT = 480
MOVE_COUNT = 18
ALL_MOVE_MASK = (1 << MOVE_COUNT) - 1

ROLE_I = 0
ROLE_A = 1
ROLE_N = 2
ROLE_R = 3
ROLE_NAMES = ("I", "A", "N", "R")


@dataclass(frozen=True)
class PerPieceRouteAtlas:
    # raw exact current/target -> relation class
    route_class: np.ndarray              # [20,576] uint16
    raw_distance: np.ndarray             # [20,576] uint8

    # class coordinates
    class_kind: np.ndarray               # [480] uint8 0 corner / 1 edge
    class_current_position: np.ndarray   # [480] uint8
    class_target_position: np.ndarray    # [480] uint8
    class_orientation_slot: np.ndarray   # [480] uint8
    class_distance: np.ndarray           # [480] uint8

    # current boundary move
    left_next_class: np.ndarray          # [480,18] uint16
    left_response_code: np.ndarray       # [480,18] uint8
    left_role_mask: np.ndarray           # [480,4] uint32

    # target boundary replayed backward
    right_next_class: np.ndarray         # [480,18] uint16
    right_response_code: np.ndarray      # [480,18] uint8
    right_role_mask: np.ndarray          # [480,4] uint32

    class_raw_member_count: np.ndarray   # [480] uint16

    def relation_class(
        self,
        piece: int,
        q_current: int,
        q_target: int,
    ) -> int:
        return int(
            self.route_class[
                int(piece),
                int(q_current) * 24 + int(q_target),
            ]
        )

    def distance(
        self,
        piece: int,
        q_current: int,
        q_target: int,
    ) -> int:
        cid = self.relation_class(
            piece,
            q_current,
            q_target,
        )
        return int(
            self.class_distance[
                cid
            ]
        )

    def local_shortest_move_mask(
        self,
        piece: int,
        q_current: int,
        q_target: int,
        *,
        from_target_side: bool = False,
    ) -> int:
        cid = self.relation_class(
            piece,
            q_current,
            q_target,
        )

        masks = (
            self.right_role_mask
            if from_target_side
            else self.left_role_mask
        )

        return int(
            masks[
                cid,
                ROLE_A,
            ]
        )

    def response_masks(
        self,
        piece: int,
        q_current: int,
        q_target: int,
        *,
        from_target_side: bool = False,
    ) -> tuple[int, int, int, int]:
        cid = self.relation_class(
            piece,
            q_current,
            q_target,
        )

        masks = (
            self.right_role_mask
            if from_target_side
            else self.left_role_mask
        )

        return tuple(
            int(
                masks[
                    cid,
                    role,
                ]
            )
            for role in range(4)
        )

    def next_relation(
        self,
        relation_class: int,
        move_id: int,
        *,
        from_target_side: bool = False,
    ) -> int:
        table = (
            self.right_next_class
            if from_target_side
            else self.left_next_class
        )

        return int(
            table[
                int(relation_class),
                int(move_id),
            ]
        )

    def class_row_json(
        self,
        relation_class: int,
    ) -> dict[str, Any]:
        cid = int(relation_class)

        return {
            "relation_class": cid,
            "kind": (
                "CORNER"
                if int(self.class_kind[cid]) == 0
                else "EDGE"
            ),
            "current_position": int(
                self.class_current_position[cid]
            ),
            "target_position": int(
                self.class_target_position[cid]
            ),
            "relative_orientation_slot": int(
                self.class_orientation_slot[cid]
            ),
            "exact_local_distance": int(
                self.class_distance[cid]
            ),
            "left": {
                "next_class": [
                    int(x)
                    for x in self.left_next_class[cid]
                ],
                "response_code": [
                    ROLE_NAMES[int(x)]
                    for x in self.left_response_code[cid]
                ],
                "I_mask": int(
                    self.left_role_mask[cid, ROLE_I]
                ),
                "A_mask": int(
                    self.left_role_mask[cid, ROLE_A]
                ),
                "N_mask": int(
                    self.left_role_mask[cid, ROLE_N]
                ),
                "R_mask": int(
                    self.left_role_mask[cid, ROLE_R]
                ),
                "local_shortest_first_move_mask": int(
                    self.left_role_mask[cid, ROLE_A]
                ),
            },
            "right": {
                "next_class": [
                    int(x)
                    for x in self.right_next_class[cid]
                ],
                "response_code": [
                    ROLE_NAMES[int(x)]
                    for x in self.right_response_code[cid]
                ],
                "I_mask": int(
                    self.right_role_mask[cid, ROLE_I]
                ),
                "A_mask": int(
                    self.right_role_mask[cid, ROLE_A]
                ),
                "N_mask": int(
                    self.right_role_mask[cid, ROLE_N]
                ),
                "R_mask": int(
                    self.right_role_mask[cid, ROLE_R]
                ),
                "local_shortest_first_move_mask": int(
                    self.right_role_mask[cid, ROLE_A]
                ),
            },
            "raw_member_count": int(
                self.class_raw_member_count[cid]
            ),
            "geodesic_mask_is_global_hard_authority": False,
        }


def _load_assets(
    *,
    relative_asset: Path,
    groupoid_asset: Path,
) -> dict[str, np.ndarray]:
    with np.load(
        Path(relative_asset),
        allow_pickle=False,
    ) as data:
        required_c0 = {
            "relative_class",
            "relative_distance",
            "class_kind",
            "class_left_next",
            "class_right_next",
        }

        missing = sorted(
            required_c0
            - set(data.files)
        )

        if missing:
            raise RuntimeError(
                "A14_P0_C0_KEYS_MISSING "
                f"{missing}"
            )

        c0 = {
            key: np.asarray(data[key])
            for key in required_c0
        }

    with np.load(
        Path(groupoid_asset),
        allow_pickle=False,
    ) as data:
        required_g = {
            "corner_global_classes",
            "corner_global_to_local",
            "corner_source_position",
            "corner_target_position",
            "edge_global_classes",
            "edge_global_to_local",
            "edge_source_position",
            "edge_target_position",
        }

        missing = sorted(
            required_g
            - set(data.files)
        )

        if missing:
            raise RuntimeError(
                "A14_P0_GROUPOID_KEYS_MISSING "
                f"{missing}"
            )

        groupoid = {
            key: np.asarray(data[key])
            for key in required_g
        }

    return {
        **c0,
        **groupoid,
    }


def _class_coordinates(
    arrays: dict[str, np.ndarray],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    current_position = np.full(
        CLASS_COUNT,
        255,
        dtype=np.uint8,
    )
    target_position = np.full_like(
        current_position,
        255,
    )
    orientation_slot = np.full_like(
        current_position,
        255,
    )

    for prefix, expected_parallel in (
        ("corner", 3),
        ("edge", 2),
    ):
        global_classes = np.asarray(
            arrays[
                f"{prefix}_global_classes"
            ],
            dtype=np.uint16,
        )
        source = np.asarray(
            arrays[
                f"{prefix}_source_position"
            ],
            dtype=np.uint8,
        )
        target = np.asarray(
            arrays[
                f"{prefix}_target_position"
            ],
            dtype=np.uint8,
        )

        if not (
            global_classes.size
            == source.size
            == target.size
        ):
            raise RuntimeError(
                "A14_P0_CLASS_COORDINATE_SHAPE_DRIFT "
                f"prefix={prefix}"
            )

        # Deterministic local orientation gauge:
        # within each ordered physical-position pair, sort global class IDs.
        pair_to_classes: dict[
            tuple[int, int],
            list[int],
        ] = {}

        for local, cid_raw in enumerate(
            global_classes
        ):
            cid = int(cid_raw)
            src = int(source[local])
            dst = int(target[local])

            current_position[cid] = np.uint8(src)
            target_position[cid] = np.uint8(dst)

            pair_to_classes.setdefault(
                (src, dst),
                [],
            ).append(cid)

        for pair, classes in pair_to_classes.items():
            classes = sorted(classes)

            if len(classes) != expected_parallel:
                raise RuntimeError(
                    "A14_P0_PARALLEL_RELATION_COUNT_DRIFT "
                    f"prefix={prefix} pair={pair} "
                    f"count={len(classes)} "
                    f"expected={expected_parallel}"
                )

            for slot, cid in enumerate(classes):
                orientation_slot[
                    cid
                ] = np.uint8(slot)

    if np.any(
        current_position == 255
    ):
        raise RuntimeError(
            "A14_P0_CURRENT_POSITION_INCOMPLETE"
        )

    if np.any(
        target_position == 255
    ):
        raise RuntimeError(
            "A14_P0_TARGET_POSITION_INCOMPLETE"
        )

    if np.any(
        orientation_slot == 255
    ):
        raise RuntimeError(
            "A14_P0_ORIENTATION_SLOT_INCOMPLETE"
        )

    return (
        current_position,
        target_position,
        orientation_slot,
    )


def _response_code(
    *,
    physically_unchanged: bool,
    before_distance: int,
    after_distance: int,
) -> int:
    if physically_unchanged:
        if int(after_distance) != int(before_distance):
            raise RuntimeError(
                "A14_P0_INACTIVE_DISTANCE_CHANGED "
                f"before={before_distance} after={after_distance}"
            )
        return ROLE_I

    delta = (
        int(after_distance)
        - int(before_distance)
    )

    if delta == -1:
        return ROLE_A
    if delta == 0:
        return ROLE_N
    if delta == 1:
        return ROLE_R

    raise RuntimeError(
        "A14_P0_DISTANCE_DELTA_OUT_OF_RANGE "
        f"before={before_distance} after={after_distance} "
        f"delta={delta}"
    )


def build_per_piece_route_atlas(
    *,
    relative_asset: Path,
    groupoid_asset: Path,
) -> PerPieceRouteAtlas:
    arrays = _load_assets(
        relative_asset=relative_asset,
        groupoid_asset=groupoid_asset,
    )

    route_class = np.asarray(
        arrays[
            "relative_class"
        ],
        dtype=np.uint16,
    )
    raw_distance = np.asarray(
        arrays[
            "relative_distance"
        ],
        dtype=np.uint8,
    )
    class_kind = np.asarray(
        arrays[
            "class_kind"
        ],
        dtype=np.uint8,
    )
    left_next_class = np.asarray(
        arrays[
            "class_left_next"
        ],
        dtype=np.uint16,
    )
    right_next_class = np.asarray(
        arrays[
            "class_right_next"
        ],
        dtype=np.uint16,
    )

    if route_class.shape != (
        20,
        PAIR_COUNT,
    ):
        raise RuntimeError(
            "A14_P0_ROUTE_CLASS_SHAPE_DRIFT "
            f"{route_class.shape}"
        )

    if raw_distance.shape != (
        20,
        PAIR_COUNT,
    ):
        raise RuntimeError(
            "A14_P0_DISTANCE_SHAPE_DRIFT "
            f"{raw_distance.shape}"
        )

    if class_kind.shape != (
        CLASS_COUNT,
    ):
        raise RuntimeError(
            "A14_P0_CLASS_KIND_SHAPE_DRIFT "
            f"{class_kind.shape}"
        )

    if left_next_class.shape != (
        CLASS_COUNT,
        MOVE_COUNT,
    ):
        raise RuntimeError(
            "A14_P0_LEFT_NEXT_SHAPE_DRIFT "
            f"{left_next_class.shape}"
        )

    if right_next_class.shape != left_next_class.shape:
        raise RuntimeError(
            "A14_P0_RIGHT_NEXT_SHAPE_DRIFT "
            f"{right_next_class.shape}"
        )

    (
        current_position,
        target_position,
        orientation_slot,
    ) = _class_coordinates(
        arrays
    )

    class_distance = np.full(
        CLASS_COUNT,
        255,
        dtype=np.uint8,
    )

    class_member_count = np.zeros(
        CLASS_COUNT,
        dtype=np.uint16,
    )

    left_response = np.full(
        (
            CLASS_COUNT,
            MOVE_COUNT,
        ),
        255,
        dtype=np.uint8,
    )
    right_response = np.full_like(
        left_response,
        255,
    )

    # Exhaustive raw member grounding.
    for piece in range(20):
        expected_kind = (
            0
            if piece < 8
            else 1
        )

        for q_current in range(24):
            for q_target in range(24):
                pid = (
                    q_current * 24
                    + q_target
                )
                cid = int(
                    route_class[
                        piece,
                        pid,
                    ]
                )
                distance = int(
                    raw_distance[
                        piece,
                        pid,
                    ]
                )

                if int(
                    class_kind[
                        cid
                    ]
                ) != expected_kind:
                    raise RuntimeError(
                        "A14_P0_CLASS_KIND_MEMBER_DRIFT "
                        f"piece={piece} class={cid}"
                    )

                class_member_count[
                    cid
                ] += 1

                if int(
                    class_distance[
                        cid
                    ]
                ) == 255:
                    class_distance[
                        cid
                    ] = np.uint8(
                        distance
                    )
                elif int(
                    class_distance[
                        cid
                    ]
                ) != distance:
                    raise RuntimeError(
                        "A14_P0_DISTANCE_NOT_CLASS_INVARIANT "
                        f"class={cid} "
                        f"old={int(class_distance[cid])} "
                        f"new={distance} "
                        f"piece={piece} "
                        f"q={q_current}->{q_target}"
                    )

                for mid in range(
                    MOVE_COUNT
                ):
                    q_current_next = int(
                        bi.Q_NEXT[
                            piece,
                            q_current,
                            mid,
                        ]
                    )
                    next_pid = (
                        q_current_next
                        * 24
                        + q_target
                    )
                    next_cid = int(
                        route_class[
                            piece,
                            next_pid,
                        ]
                    )
                    next_distance = int(
                        raw_distance[
                            piece,
                            next_pid,
                        ]
                    )

                    observed_left_next = int(
                        left_next_class[
                            cid,
                            mid,
                        ]
                    )

                    if observed_left_next != next_cid:
                        raise RuntimeError(
                            "A14_P0_LEFT_NEXT_CLASS_PARITY_FAIL "
                            f"class={cid} move={mid} "
                            f"observed={observed_left_next} "
                            f"expected={next_cid}"
                        )

                    code = _response_code(
                        physically_unchanged=(
                            q_current_next
                            == q_current
                        ),
                        before_distance=distance,
                        after_distance=next_distance,
                    )

                    stored = int(
                        left_response[
                            cid,
                            mid,
                        ]
                    )

                    if stored == 255:
                        left_response[
                            cid,
                            mid,
                        ] = np.uint8(
                            code
                        )
                    elif stored != code:
                        raise RuntimeError(
                            "A14_P0_LEFT_RESPONSE_NOT_CLASS_INVARIANT "
                            f"class={cid} move={mid} "
                            f"old={stored} new={code}"
                        )

                    # Target-side exact reverse replay.
                    q_target_prev = int(
                        bi.Q_PREV[
                            piece,
                            q_target,
                            mid,
                        ]
                    )
                    right_pid = (
                        q_current
                        * 24
                        + q_target_prev
                    )
                    right_cid = int(
                        route_class[
                            piece,
                            right_pid,
                        ]
                    )
                    right_distance = int(
                        raw_distance[
                            piece,
                            right_pid,
                        ]
                    )

                    observed_right_next = int(
                        right_next_class[
                            cid,
                            mid,
                        ]
                    )

                    if observed_right_next != right_cid:
                        raise RuntimeError(
                            "A14_P0_RIGHT_NEXT_CLASS_PARITY_FAIL "
                            f"class={cid} move={mid} "
                            f"observed={observed_right_next} "
                            f"expected={right_cid}"
                        )

                    rcode = _response_code(
                        physically_unchanged=(
                            q_target_prev
                            == q_target
                        ),
                        before_distance=distance,
                        after_distance=right_distance,
                    )

                    stored = int(
                        right_response[
                            cid,
                            mid,
                        ]
                    )

                    if stored == 255:
                        right_response[
                            cid,
                            mid,
                        ] = np.uint8(
                            rcode
                        )
                    elif stored != rcode:
                        raise RuntimeError(
                            "A14_P0_RIGHT_RESPONSE_NOT_CLASS_INVARIANT "
                            f"class={cid} move={mid} "
                            f"old={stored} new={rcode}"
                        )

    if np.any(
        class_distance == 255
    ):
        raise RuntimeError(
            "A14_P0_CLASS_DISTANCE_INCOMPLETE"
        )

    if np.any(
        left_response == 255
    ):
        raise RuntimeError(
            "A14_P0_LEFT_RESPONSE_INCOMPLETE"
        )

    if np.any(
        right_response == 255
    ):
        raise RuntimeError(
            "A14_P0_RIGHT_RESPONSE_INCOMPLETE"
        )

    left_masks = np.zeros(
        (
            CLASS_COUNT,
            4,
        ),
        dtype=np.uint32,
    )
    right_masks = np.zeros_like(
        left_masks
    )

    for cid in range(
        CLASS_COUNT
    ):
        for mid in range(
            MOVE_COUNT
        ):
            left_masks[
                cid,
                int(
                    left_response[
                        cid,
                        mid,
                    ]
                ),
            ] |= np.uint32(
                1
                << mid
            )

            right_masks[
                cid,
                int(
                    right_response[
                        cid,
                        mid,
                    ]
                ),
            ] |= np.uint32(
                1
                << mid
            )

    # Response partition must cover every move exactly once.
    for masks, side in (
        (
            left_masks,
            "LEFT",
        ),
        (
            right_masks,
            "RIGHT",
        ),
    ):
        for cid in range(
            CLASS_COUNT
        ):
            union = 0
            overlap = 0

            for role in range(
                4
            ):
                mask = int(
                    masks[
                        cid,
                        role,
                    ]
                )
                overlap |= (
                    union
                    & mask
                )
                union |= mask

            if overlap != 0:
                raise RuntimeError(
                    "A14_P0_ROLE_MASK_OVERLAP "
                    f"side={side} class={cid} "
                    f"overlap={overlap:#x}"
                )

            if union != ALL_MOVE_MASK:
                raise RuntimeError(
                    "A14_P0_ROLE_MASK_COVERAGE "
                    f"side={side} class={cid} "
                    f"union={union:#x}"
                )

    # Exact local geodesic property:
    # d>0 has at least one A move; d=0 has no A move.
    for cid in range(
        CLASS_COUNT
    ):
        distance = int(
            class_distance[
                cid
            ]
        )

        for masks, side in (
            (
                left_masks,
                "LEFT",
            ),
            (
                right_masks,
                "RIGHT",
            ),
        ):
            a_mask = int(
                masks[
                    cid,
                    ROLE_A,
                ]
            )

            if (
                distance == 0
                and a_mask != 0
            ):
                raise RuntimeError(
                    "A14_P0_IDENTITY_HAS_DECREASING_MOVE "
                    f"side={side} class={cid}"
                )

            if (
                distance > 0
                and a_mask == 0
            ):
                raise RuntimeError(
                    "A14_P0_NONIDENTITY_WITHOUT_GEODESIC_MOVE "
                    f"side={side} class={cid} "
                    f"distance={distance}"
                )

    return PerPieceRouteAtlas(
        route_class=route_class,
        raw_distance=raw_distance,
        class_kind=class_kind,
        class_current_position=current_position,
        class_target_position=target_position,
        class_orientation_slot=orientation_slot,
        class_distance=class_distance,
        left_next_class=left_next_class,
        left_response_code=left_response,
        left_role_mask=left_masks,
        right_next_class=right_next_class,
        right_response_code=right_response,
        right_role_mask=right_masks,
        class_raw_member_count=class_member_count,
    )


def route_atlas_arrays(
    atlas: PerPieceRouteAtlas,
) -> dict[str, np.ndarray]:
    return {
        "route_class": atlas.route_class,
        "raw_distance": atlas.raw_distance,
        "class_kind": atlas.class_kind,
        "class_current_position": atlas.class_current_position,
        "class_target_position": atlas.class_target_position,
        "class_orientation_slot": atlas.class_orientation_slot,
        "class_distance": atlas.class_distance,
        "left_next_class": atlas.left_next_class,
        "left_response_code": atlas.left_response_code,
        "left_role_mask": atlas.left_role_mask,
        "right_next_class": atlas.right_next_class,
        "right_response_code": atlas.right_response_code,
        "right_role_mask": atlas.right_role_mask,
        "class_raw_member_count": atlas.class_raw_member_count,
    }
