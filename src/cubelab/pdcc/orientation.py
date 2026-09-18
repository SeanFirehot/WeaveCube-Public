"""Exact base-fiber decomposition and orientation cocycle tables."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .group import CUBE_ROTATIONS, Matrix3, Vector3, matrix_vec
from .model import PIECE_SPECS, PieceKind, slots_for_kind


def _mapped(matrix: Matrix3, vectors: tuple[Vector3, ...]) -> tuple[Vector3, ...]:
    return tuple(matrix_vec(matrix, vector) for vector in vectors)


def _unique_rotation_mapping(
    source: tuple[Vector3, ...], target: tuple[Vector3, ...]
) -> int:
    matches = [
        rotation_id
        for rotation_id, matrix in enumerate(CUBE_ROTATIONS.elements)
        if _mapped(matrix, source) == target
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one exact cube rotation for ordered sticker mapping, "
            f"got {len(matches)} for {source} -> {target}"
        )
    return matches[0]


@dataclass(frozen=True, slots=True)
class OrientationSystem:
    """Deterministic cubie orientation gauge backed by exact pose matrices.

    For each piece/slot pair, ``section_pose[(piece, slot)]`` is the unique pose
    that maps the piece's ordered solved sticker normals to that slot's ordered
    sticker normals.  The residual stabilizer power is the orientation value.
    """

    section_pose: Mapping[tuple[str, str], int]
    generator: Mapping[str, int]
    generator_powers: Mapping[str, tuple[int, ...]]
    orientation_by_pose: Mapping[tuple[str, int], int]
    pose_by_coordinate: Mapping[tuple[str, str, int], int]

    @classmethod
    def build(cls) -> "OrientationSystem":
        section_pose: dict[tuple[str, str], int] = {}
        generator: dict[str, int] = {}
        generator_powers: dict[str, tuple[int, ...]] = {}
        orientation_by_pose: dict[tuple[str, int], int] = {}
        pose_by_coordinate: dict[tuple[str, str, int], int] = {}

        for piece_name, piece in PIECE_SPECS.items():
            slot_names = slots_for_kind(piece.kind)
            for slot_name in slot_names:
                slot = PIECE_SPECS[slot_name]
                section_pose[(piece_name, slot_name)] = _unique_rotation_mapping(
                    piece.ordered_stickers, slot.ordered_stickers
                )

            # Cyclic shift of ordered sticker normals.  For edges this is the
            # unique flip; for corners this is a 120-degree body-diagonal twist.
            shifted = piece.ordered_stickers[1:] + piece.ordered_stickers[:1]
            generator_id = _unique_rotation_mapping(piece.ordered_stickers, shifted)
            generator[piece_name] = generator_id

            powers = [CUBE_ROTATIONS.identity]
            for _ in range(1, piece.orientation_order):
                powers.append(CUBE_ROTATIONS.compose(powers[-1], generator_id))
            if CUBE_ROTATIONS.compose(powers[-1], generator_id) != CUBE_ROTATIONS.identity:
                raise RuntimeError(f"Bad stabilizer generator order for {piece_name}")
            generator_powers[piece_name] = tuple(powers)
            power_to_orientation = {pose_id: ori for ori, pose_id in enumerate(powers)}

            for pose_id in range(len(CUBE_ROTATIONS.elements)):
                position = CUBE_ROTATIONS.apply(pose_id, piece.solved_position)
                slot_name = next(
                    name
                    for name in slot_names
                    if PIECE_SPECS[name].solved_position == position
                )
                section_id = section_pose[(piece_name, slot_name)]
                residual = CUBE_ROTATIONS.compose(
                    CUBE_ROTATIONS.inverse(section_id), pose_id
                )
                try:
                    orientation = power_to_orientation[residual]
                except KeyError as exc:
                    raise RuntimeError(
                        f"Pose {pose_id} of {piece_name} does not decompose in slot {slot_name}"
                    ) from exc
                orientation_by_pose[(piece_name, pose_id)] = orientation
                pose_by_coordinate[(piece_name, slot_name, orientation)] = pose_id

        expected_coordinate_count = sum(
            len(slots_for_kind(spec.kind)) * spec.orientation_order
            for spec in PIECE_SPECS.values()
        )
        if len(pose_by_coordinate) != expected_coordinate_count:
            raise RuntimeError("Incomplete pose-coordinate table")

        return cls(
            section_pose=MappingProxyType(section_pose),
            generator=MappingProxyType(generator),
            generator_powers=MappingProxyType(generator_powers),
            orientation_by_pose=MappingProxyType(orientation_by_pose),
            pose_by_coordinate=MappingProxyType(pose_by_coordinate),
        )

    def orientation(self, piece: str, pose_id: int) -> int:
        try:
            return self.orientation_by_pose[(piece, pose_id)]
        except KeyError as exc:
            raise ValueError(f"Unknown piece/pose: {piece}/{pose_id}") from exc

    def pose(self, piece: str, slot: str, orientation: int) -> int:
        spec = PIECE_SPECS[piece]
        if PIECE_SPECS[slot].kind is not spec.kind:
            raise ValueError(f"Piece {piece} cannot occupy slot {slot}")
        normalized = orientation % spec.orientation_order
        return self.pose_by_coordinate[(piece, slot, normalized)]

    def cocycle(self, piece: str, rotation_id: int, slot: str) -> int:
        """Return orientation delta for a global rotation at a given slot."""

        spec = PIECE_SPECS[piece]
        if PIECE_SPECS[slot].kind is not spec.kind:
            raise ValueError(f"Piece {piece} and slot {slot} have different kinds")
        target_position = CUBE_ROTATIONS.apply(
            rotation_id, PIECE_SPECS[slot].solved_position
        )
        target_slot = next(
            name
            for name in slots_for_kind(spec.kind)
            if PIECE_SPECS[name].solved_position == target_position
        )
        source_section = self.section_pose[(piece, slot)]
        target_section = self.section_pose[(piece, target_slot)]
        residual = CUBE_ROTATIONS.compose(
            CUBE_ROTATIONS.inverse(target_section),
            CUBE_ROTATIONS.compose(rotation_id, source_section),
        )
        try:
            return self.generator_powers[piece].index(residual)
        except ValueError as exc:
            raise RuntimeError(
                f"Cocycle residual is outside stabilizer for {piece} at {slot}"
            ) from exc

    def target_slot(self, piece: str, rotation_id: int, slot: str) -> str:
        spec = PIECE_SPECS[piece]
        target_position = CUBE_ROTATIONS.apply(
            rotation_id, PIECE_SPECS[slot].solved_position
        )
        for candidate in slots_for_kind(spec.kind):
            if PIECE_SPECS[candidate].solved_position == target_position:
                return candidate
        raise RuntimeError("Rotation did not map slot to a same-kind slot")


ORIENTATION_SYSTEM = OrientationSystem.build()
