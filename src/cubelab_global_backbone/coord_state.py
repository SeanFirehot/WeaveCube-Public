"""Eleven-field CubeLab coordinate state with a full-pose audit shadow."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

from cubelab.ato.projections import legacy_orientation
from cubelab.pdcc import (
    CORNER_NAMES,
    EDGE_NAMES,
    PDCCState,
    inverse_word,
    parse_word,
)
from cubelab.pdcc.model import CORNER_INDEX, EDGE_INDEX

from .move_tables import MOVE_ORDER, apply_pose_move, solved_poses


AXES = ("UD", "FB", "LR")
EDGE_BANDS = (
    ("UR", "UF", "UL", "UB"),
    ("FR", "FL", "BL", "BR"),
    ("DR", "DF", "DL", "DB"),
)
U_CORNER_SLOTS = frozenset(("UFR", "UFL", "UBL", "UBR"))


def rank_permutation(values: tuple[int, ...]) -> int:
    rank = 0
    for left, value in enumerate(values):
        smaller = sum(other < value for other in values[left + 1 :])
        factorial = 1
        for item in range(2, len(values) - left):
            factorial *= item
        rank += smaller * factorial
    return rank


def rank_k_permutation(values: tuple[int, ...], population: int) -> int:
    available = list(range(population))
    rank = 0
    width = len(values)
    for position, value in enumerate(values):
        offset = available.index(value)
        remaining = width - position - 1
        block = 1
        for factor in range(population - position - remaining, population - position):
            block *= factor
        rank += offset * block
        available.pop(offset)
    return rank


def rank_combination(values: tuple[int, ...], population: int) -> int:
    rank = 0
    previous = -1
    remaining = len(values)
    for value in values:
        for candidate in range(previous + 1, value):
            rank += comb(population - candidate - 1, remaining - 1)
        previous = value
        remaining -= 1
    return rank


def rank_orientation(values: tuple[int, ...], base: int) -> int:
    rank = 0
    for value in values[:-1]:
        rank = rank * base + value
    return rank


@dataclass(frozen=True, slots=True)
class CoordinateState:
    poses: tuple[int, ...]
    edge_band_pos: tuple[int, int, int]
    edge_flip: tuple[int, int, int]
    corner_twist: tuple[int, int, int]
    corner_perm: int
    corner_ud_split: int
    origin_word: tuple[str, ...] | None = None

    @classmethod
    def from_poses(
        cls, poses: tuple[int, ...], *, origin_word: tuple[str, ...] | None = None
    ) -> "CoordinateState":
        state = PDCCState(poses)
        edge_positions = state.edge_permutation_tuple()
        edge_band_pos = tuple(
            rank_k_permutation(
                tuple(edge_positions[EDGE_INDEX[piece]] for piece in band), 12
            )
            for band in EDGE_BANDS
        )
        edge_flip = tuple(
            rank_orientation(
                tuple(legacy_orientation(piece, poses[8 + index], axis) for index, piece in enumerate(EDGE_NAMES)),
                2,
            )
            for axis in AXES
        )
        corner_twist = tuple(
            rank_orientation(
                tuple(legacy_orientation(piece, poses[index], axis) for index, piece in enumerate(CORNER_NAMES)),
                3,
            )
            for axis in AXES
        )
        corner_permutation = rank_permutation(state.corner_permutation_tuple())
        upper_pieces = tuple(
            index
            for index, piece in enumerate(CORNER_NAMES)
            if state.slot(piece) in U_CORNER_SLOTS
        )
        return cls(
            poses=poses,
            edge_band_pos=edge_band_pos,  # type: ignore[arg-type]
            edge_flip=edge_flip,  # type: ignore[arg-type]
            corner_twist=corner_twist,  # type: ignore[arg-type]
            corner_perm=corner_permutation,
            corner_ud_split=rank_combination(upper_pieces, 8),
            origin_word=origin_word,
        )

    @classmethod
    def solved(cls) -> "CoordinateState":
        return cls.from_poses(solved_poses(), origin_word=())

    @classmethod
    def from_word(cls, word: str | tuple[str, ...]) -> "CoordinateState":
        parsed = parse_word(word)
        state = cls.solved()
        for move in parsed:
            state = state.move(move)
        return state

    def move(self, move: str) -> "CoordinateState":
        if move not in MOVE_ORDER:
            raise ValueError(f"unsupported HTM move: {move}")
        word = self.origin_word + (move,) if self.origin_word is not None else None
        return self.from_poses(apply_pose_move(self.poses, move), origin_word=word)

    def inverse(self) -> "CoordinateState":
        if self.origin_word is None:
            raise ValueError("inverse requires an auditable origin word")
        return self.from_word(inverse_word(self.origin_word))

    def is_solved(self) -> bool:
        return self.poses == solved_poses()

    def eleven_fields(self) -> tuple[int, ...]:
        return (
            *self.edge_band_pos,
            *self.edge_flip,
            *self.corner_twist,
            self.corner_perm,
            self.corner_ud_split,
        )

    def axis_signature(self, axis: str) -> tuple[int, int]:
        try:
            index = AXES.index(axis)
        except ValueError as exc:
            raise ValueError(f"unknown physical axis: {axis}") from exc
        return self.edge_flip[index], self.corner_twist[index]
