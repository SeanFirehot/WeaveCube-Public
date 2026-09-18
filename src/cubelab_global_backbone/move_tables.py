"""CubeLab-owned fixed pose-transition table generated from frozen PDCC."""

from __future__ import annotations

from functools import lru_cache

from cubelab.pdcc import MOVE_ORDER, PIECE_NAMES, PDCCState, transition_piece_pose


MOVE_INDEX = {move: index for index, move in enumerate(MOVE_ORDER)}
PIECE_COUNT = len(PIECE_NAMES)
POSE_COUNT = 24


@lru_cache(maxsize=1)
def pose_move_table() -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Return `[piece][pose][move] -> pose` using only the PDCC contract."""

    return tuple(
        tuple(
            tuple(
                transition_piece_pose(piece, pose, move)[0]
                for move in MOVE_ORDER
            )
            for pose in range(POSE_COUNT)
        )
        for piece in PIECE_NAMES
    )


def apply_pose_move(poses: tuple[int, ...], move: str) -> tuple[int, ...]:
    move_index = MOVE_INDEX[move]
    table = pose_move_table()
    return tuple(
        table[piece_index][pose][move_index]
        for piece_index, pose in enumerate(poses)
    )


def apply_pose_word(poses: tuple[int, ...], word: tuple[str, ...]) -> tuple[int, ...]:
    result = poses
    for move in word:
        result = apply_pose_move(result, move)
    return result


def solved_poses() -> tuple[int, ...]:
    return PDCCState.solved().poses
