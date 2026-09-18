"""Immutable 20-piece pose state and exact HTM transition engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .group import CUBE_ROTATIONS, Vector3
from .model import (
    CORNER_INDEX,
    CORNER_NAMES,
    CORNER_SLOT_BY_POSITION,
    EDGE_INDEX,
    EDGE_NAMES,
    EDGE_SLOT_BY_POSITION,
    PIECE_INDEX,
    PIECE_NAMES,
    PIECE_SPECS,
    PieceKind,
)
from .moves import MOVES, MoveSpec, parse_word
from .orientation import ORIENTATION_SYSTEM
from .tables import local_transition


def _permutation_parity(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return inversions % 2


def slot_for_pose(piece: str, pose_id: int) -> str:
    spec = PIECE_SPECS[piece]
    position = CUBE_ROTATIONS.apply(pose_id, spec.solved_position)
    if spec.kind is PieceKind.CORNER:
        return CORNER_SLOT_BY_POSITION[position]
    return EDGE_SLOT_BY_POSITION[position]


def transition_piece_pose(piece: str, pose_id: int, move: str | MoveSpec) -> tuple[int, bool]:
    move_name = move if isinstance(move, str) else move.name
    entry = local_transition(piece, pose_id, move_name)
    return entry.pose_after, entry.active


@dataclass(frozen=True, slots=True)
class LegalityReport:
    legal: bool
    unique_corner_slots: bool
    unique_edge_slots: bool
    corner_orientation_sum_mod3: int
    edge_orientation_sum_mod2: int
    corner_parity: int
    edge_parity: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PDCCState:
    """Exact state represented by one cube-rotation pose ID per movable piece."""

    poses: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.poses) != len(PIECE_NAMES):
            raise ValueError(
                f"Expected {len(PIECE_NAMES)} poses, got {len(self.poses)}"
            )
        if any(pose < 0 or pose >= len(CUBE_ROTATIONS.elements) for pose in self.poses):
            raise ValueError("Pose IDs must be in [0, 23]")

    @classmethod
    def solved(cls) -> "PDCCState":
        return cls((CUBE_ROTATIONS.identity,) * len(PIECE_NAMES))

    @classmethod
    def from_coordinates(
        cls,
        *,
        permutation: Mapping[str, str] | None = None,
        orientation: Mapping[str, int] | None = None,
        require_legal: bool = True,
    ) -> "PDCCState":
        """Build a state from piece-to-slot and per-piece orientation coordinates.

        Unspecified pieces remain in their solved slot with orientation zero.
        ``permutation`` maps piece identity to destination slot identity.
        """

        permutation = permutation or {}
        orientation = orientation or {}
        unknown = (set(permutation) | set(orientation)) - set(PIECE_NAMES)
        if unknown:
            raise ValueError(f"Unknown pieces: {sorted(unknown)}")

        slots: dict[str, str] = {}
        poses: list[int] = []
        for piece in PIECE_NAMES:
            slot = permutation.get(piece, piece)
            if slot not in PIECE_SPECS:
                raise ValueError(f"Unknown slot {slot!r} for piece {piece}")
            if PIECE_SPECS[slot].kind is not PIECE_SPECS[piece].kind:
                raise ValueError(f"Piece {piece} cannot occupy slot {slot}")
            if slot in slots:
                raise ValueError(
                    f"Slot {slot} assigned to both {slots[slot]} and {piece}"
                )
            slots[slot] = piece
            poses.append(
                ORIENTATION_SYSTEM.pose(piece, slot, orientation.get(piece, 0))
            )

        state = cls(tuple(poses))
        if require_legal:
            report = state.legality()
            if not report.legal:
                raise ValueError(
                    "Coordinates do not describe a legal cube state: "
                    + "; ".join(report.reasons)
                )
        return state

    @classmethod
    def superflip(cls) -> "PDCCState":
        return cls.from_coordinates(
            orientation={edge: 1 for edge in EDGE_NAMES}, require_legal=True
        )

    def pose_id(self, piece: str) -> int:
        try:
            return self.poses[PIECE_INDEX[piece]]
        except KeyError as exc:
            raise ValueError(f"Unknown piece: {piece}") from exc

    def position(self, piece: str) -> Vector3:
        return CUBE_ROTATIONS.apply(
            self.pose_id(piece), PIECE_SPECS[piece].solved_position
        )

    def slot(self, piece: str) -> str:
        return slot_for_pose(piece, self.pose_id(piece))

    def orientation(self, piece: str) -> int:
        return ORIENTATION_SYSTEM.orientation(piece, self.pose_id(piece))

    def display_vector(self, piece: str) -> tuple[int, int, int, int]:
        solved = PIECE_SPECS[piece].solved_position
        current = self.position(piece)
        delta = tuple(current[axis] - solved[axis] for axis in range(3))
        return (delta[0], delta[1], delta[2], self.orientation(piece))

    def permutation(self) -> dict[str, str]:
        return {piece: self.slot(piece) for piece in PIECE_NAMES}

    def orientation_map(self) -> dict[str, int]:
        return {piece: self.orientation(piece) for piece in PIECE_NAMES}

    def transition(
        self, move: str | MoveSpec
    ) -> tuple["PDCCState", tuple[str, ...], tuple[str, ...]]:
        move_spec = MOVES[move] if isinstance(move, str) else move
        updated = list(self.poses)
        active_corners: list[str] = []
        active_edges: list[str] = []
        for piece in PIECE_NAMES:
            index = PIECE_INDEX[piece]
            pose_id, active = transition_piece_pose(piece, self.poses[index], move_spec)
            if active:
                updated[index] = pose_id
                if PIECE_SPECS[piece].kind is PieceKind.CORNER:
                    active_corners.append(piece)
                else:
                    active_edges.append(piece)
        return PDCCState(tuple(updated)), tuple(active_corners), tuple(active_edges)

    def apply(self, move: str | MoveSpec) -> "PDCCState":
        return self.transition(move)[0]

    def apply_word(self, word: str | Iterable[str]) -> "PDCCState":
        state = self
        for move in parse_word(word):
            state = state.apply(move)
        return state

    def is_solved(self) -> bool:
        return all(pose == CUBE_ROTATIONS.identity for pose in self.poses)

    def corner_permutation_tuple(self) -> tuple[int, ...]:
        return tuple(CORNER_INDEX[self.slot(piece)] for piece in CORNER_NAMES)

    def edge_permutation_tuple(self) -> tuple[int, ...]:
        return tuple(EDGE_INDEX[self.slot(piece)] for piece in EDGE_NAMES)

    def legality(self) -> LegalityReport:
        corner_slots = tuple(self.slot(piece) for piece in CORNER_NAMES)
        edge_slots = tuple(self.slot(piece) for piece in EDGE_NAMES)
        unique_corners = len(set(corner_slots)) == len(CORNER_NAMES)
        unique_edges = len(set(edge_slots)) == len(EDGE_NAMES)
        corner_orientation_sum = sum(
            self.orientation(piece) for piece in CORNER_NAMES
        ) % 3
        edge_orientation_sum = sum(self.orientation(piece) for piece in EDGE_NAMES) % 2
        corner_parity = _permutation_parity(self.corner_permutation_tuple())
        edge_parity = _permutation_parity(self.edge_permutation_tuple())

        reasons: list[str] = []
        if not unique_corners:
            reasons.append("corner slots are not a permutation")
        if not unique_edges:
            reasons.append("edge slots are not a permutation")
        if corner_orientation_sum != 0:
            reasons.append(
                f"corner orientation sum is {corner_orientation_sum} mod 3"
            )
        if edge_orientation_sum != 0:
            reasons.append(f"edge orientation sum is {edge_orientation_sum} mod 2")
        if corner_parity != edge_parity:
            reasons.append(
                f"corner/edge permutation parity mismatch: {corner_parity}/{edge_parity}"
            )

        return LegalityReport(
            legal=not reasons,
            unique_corner_slots=unique_corners,
            unique_edge_slots=unique_edges,
            corner_orientation_sum_mod3=corner_orientation_sum,
            edge_orientation_sum_mod2=edge_orientation_sum,
            corner_parity=corner_parity,
            edge_parity=edge_parity,
            reasons=tuple(reasons),
        )

    def to_coordinate_dict(self) -> dict[str, dict[str, int | str]]:
        return {
            piece: {
                "slot": self.slot(piece),
                "orientation": self.orientation(piece),
                "pose_id": self.pose_id(piece),
            }
            for piece in PIECE_NAMES
        }
