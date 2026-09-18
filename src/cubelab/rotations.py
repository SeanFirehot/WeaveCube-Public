"""Whole-cube rotation equivalence for CubeLab.

A whole-cube rotation does not alter a move's turn amount; it relabels the
face on which the move is performed.  CubeLab therefore materializes rotated
representatives by relabelling a transformation's move sequence and then
reusing the existing permutation and orientation engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from typing import Iterable, Mapping, Tuple

from .orientation import OrientationAnalyzer
from .transformations import PIECES, Transformation, from_sequence

Face = str
Vector = tuple[int, int, int]
Matrix = tuple[Vector, Vector, Vector]

FACE_VECTORS: dict[Face, Vector] = {
    "R": (1, 0, 0),
    "L": (-1, 0, 0),
    "U": (0, 1, 0),
    "D": (0, -1, 0),
    "F": (0, 0, 1),
    "B": (0, 0, -1),
}
VECTOR_FACES = {vector: face for face, vector in FACE_VECTORS.items()}
FACE_ORDER: tuple[Face, ...] = ("U", "D", "L", "R", "F", "B")
OPPOSITE_FACE: dict[Face, Face] = {
    "U": "D", "D": "U", "L": "R", "R": "L", "F": "B", "B": "F"
}


def _determinant(matrix: Matrix) -> int:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _apply(matrix: Matrix, vector: Vector) -> Vector:
    return tuple(sum(matrix[row][column] * vector[column] for column in range(3)) for row in range(3))  # type: ignore[return-value]


def _rotation_matrices() -> tuple[Matrix, ...]:
    matrices: list[Matrix] = []
    for axis_order in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            rows = []
            for row, axis in enumerate(axis_order):
                vector = [0, 0, 0]
                vector[axis] = signs[row]
                rows.append(tuple(vector))
            matrix = tuple(rows)  # type: ignore[assignment]
            if _determinant(matrix) == 1:
                matrices.append(matrix)
    return tuple(sorted(set(matrices)))


@dataclass(frozen=True)
class CubeRotation:
    key: str
    face_map: Mapping[Face, Face]

    def map_face(self, face: Face) -> Face:
        try:
            return self.face_map[face]
        except KeyError as exc:
            raise ValueError(f"Unknown face: {face}") from exc

    def map_move(self, move: str) -> str:
        if not move:
            raise ValueError("Move cannot be empty")
        return self.map_face(move[0]) + move[1:]

    def map_sequence(self, sequence: Iterable[str]) -> tuple[str, ...]:
        return tuple(self.map_move(move) for move in sequence)

    def apply(self, transformation: Transformation) -> Transformation:
        return from_sequence(self.map_sequence(transformation.sequence))

    def map_face_set(self, faces: Iterable[Face]) -> frozenset[Face]:
        return frozenset(self.map_face(face) for face in faces)


def _make_rotations() -> tuple[CubeRotation, ...]:
    rotations = []
    for matrix in _rotation_matrices():
        face_map = {
            face: VECTOR_FACES[_apply(matrix, vector)]
            for face, vector in FACE_VECTORS.items()
        }
        key = "".join(face_map[face] for face in FACE_ORDER)
        rotations.append(CubeRotation(key=key, face_map=face_map))
    return tuple(sorted(rotations, key=lambda item: item.key))


CUBE_ROTATIONS: tuple[CubeRotation, ...] = _make_rotations()
IDENTITY_ROTATION = next(
    rotation for rotation in CUBE_ROTATIONS
    if all(rotation.face_map[face] == face for face in FACE_ORDER)
)


@dataclass(frozen=True)
class FaceSetRotationClass:
    key: str
    representative: frozenset[Face]
    size: int
    description: str


FACE_SET_CLASSES: dict[str, FaceSetRotationClass] = {
    "faces-0": FaceSetRotationClass("faces-0", frozenset(), 0, "empty face set"),
    "faces-1": FaceSetRotationClass("faces-1", frozenset({"U"}), 1, "one face"),
    "faces-2-adjacent": FaceSetRotationClass("faces-2-adjacent", frozenset({"U", "R"}), 2, "two adjacent faces"),
    "faces-2-opposite": FaceSetRotationClass("faces-2-opposite", frozenset({"U", "D"}), 2, "two opposite faces"),
    "faces-3-corner": FaceSetRotationClass("faces-3-corner", frozenset({"U", "R", "F"}), 3, "three faces meeting at one corner"),
    "faces-3-opposite-pair": FaceSetRotationClass("faces-3-opposite-pair", frozenset({"L", "R", "F"}), 3, "one opposite pair plus one face"),
    "faces-4-missing-opposite": FaceSetRotationClass("faces-4-missing-opposite", frozenset({"U", "D", "L", "R"}), 4, "four faces whose omitted pair is opposite"),
    "faces-4-missing-adjacent": FaceSetRotationClass("faces-4-missing-adjacent", frozenset({"U", "F", "R", "D"}), 4, "four faces whose omitted pair is adjacent"),
    "faces-5": FaceSetRotationClass("faces-5", frozenset({"U", "L", "F", "R", "D"}), 5, "all but one face"),
    "faces-6": FaceSetRotationClass("faces-6", frozenset(FACE_ORDER), 6, "all six faces"),
}


class FaceSetRotationClassifier:
    @staticmethod
    def classify(faces: Iterable[Face]) -> FaceSetRotationClass:
        face_set = frozenset(faces)
        unknown = face_set - set(FACE_ORDER)
        if unknown:
            raise ValueError(f"Unknown faces: {sorted(unknown)}")
        size = len(face_set)
        if size in (0, 1, 5, 6):
            return FACE_SET_CLASSES[f"faces-{size}"]
        if size == 2:
            first, second = tuple(face_set)
            return FACE_SET_CLASSES[
                "faces-2-opposite" if OPPOSITE_FACE[first] == second else "faces-2-adjacent"
            ]
        if size == 3:
            has_opposite_pair = any(OPPOSITE_FACE[face] in face_set for face in face_set)
            return FACE_SET_CLASSES[
                "faces-3-opposite-pair" if has_opposite_pair else "faces-3-corner"
            ]
        if size == 4:
            omitted = frozenset(FACE_ORDER) - face_set
            first, second = tuple(omitted)
            return FACE_SET_CLASSES[
                "faces-4-missing-opposite" if OPPOSITE_FACE[first] == second else "faces-4-missing-adjacent"
            ]
        raise AssertionError("A cube face set cannot have any other size")

    @staticmethod
    def canonicalize(faces: Iterable[Face]) -> tuple[FaceSetRotationClass, CubeRotation]:
        face_set = frozenset(faces)
        rotation_class = FaceSetRotationClassifier.classify(face_set)
        matches = [
            rotation for rotation in CUBE_ROTATIONS
            if rotation.map_face_set(face_set) == rotation_class.representative
        ]
        if not matches:
            raise ValueError(f"No cube rotation maps {sorted(face_set)} to the representative")
        return rotation_class, min(matches, key=lambda item: item.key)


FACE_CLASS_DISPLAY_ORDER: dict[str, tuple[Face, ...]] = {
    "faces-0": (),
    "faces-1": ("U",),
    "faces-2-adjacent": ("U", "R"),
    "faces-2-opposite": ("U", "D"),
    "faces-3-corner": ("U", "R", "F", "B"),
    "faces-3-opposite-pair": ("U", "D", "R", "L", "F", "B"),
    "faces-4-missing-opposite": ("U", "R", "D", "L"),
    "faces-4-missing-adjacent": ("U", "R", "F", "D", "L", "B"),
    "faces-5": ("U", "R", "F", "D", "L", "B"),
    "faces-6": ("U", "R", "F", "D", "L", "B"),
}

def distinct_face_order(sequence: Iterable[str]) -> tuple[Face, ...]:
    seen: set[Face] = set()
    order: list[Face] = []
    for move in sequence:
        face = move[0]
        if face not in seen:
            seen.add(face)
            order.append(face)
    return tuple(order)

def _display_order_rank(sequence: tuple[str, ...]) -> tuple:
    faces = distinct_face_order(sequence)
    rotation_class = FaceSetRotationClassifier.classify(faces)
    preferred = FACE_CLASS_DISPLAY_ORDER[rotation_class.key]
    index = {face: position for position, face in enumerate(preferred)}
    return (
        tuple(index.get(face, len(preferred) + FACE_ORDER.index(face)) for face in faces),
        sequence,
    )


def full_transformation_signature(transformation: Transformation) -> tuple:
    orientation = OrientationAnalyzer.analyze(transformation)
    return (
        tuple(transformation.mapping[piece] for piece in PIECES),
        tuple((change.piece, change.position, change.twist) for change in orientation.corner_changes),
        tuple((change.piece, change.position, change.orientation) for change in orientation.edge_changes),
    )


@dataclass(frozen=True)
class RotationEquivalentTransformation:
    rotation: CubeRotation
    transformation: Transformation
    signature: tuple


@dataclass(frozen=True)
class RotationEquivalenceResult:
    original: Transformation
    canonical: RotationEquivalentTransformation
    display_canonical: RotationEquivalentTransformation
    members: tuple[RotationEquivalentTransformation, ...]

    @property
    def class_size(self) -> int:
        return len({member.signature for member in self.members})


class RotationCanonicalizer:
    @staticmethod
    def analyze(transformation: Transformation) -> RotationEquivalenceResult:
        members = tuple(
            RotationEquivalentTransformation(
                rotation=rotation,
                transformation=rotated,
                signature=full_transformation_signature(rotated),
            )
            for rotation in CUBE_ROTATIONS
            for rotated in (rotation.apply(transformation),)
        )
        canonical = min(members, key=lambda member: (member.signature, member.transformation.sequence, member.rotation.key))
        display_canonical = min(
            members,
            key=lambda member: (_display_order_rank(member.transformation.sequence), member.rotation.key),
        )
        return RotationEquivalenceResult(transformation, canonical, display_canonical, members)

    @staticmethod
    def equivalent(left: Transformation, right: Transformation) -> bool:
        return (
            RotationCanonicalizer.analyze(left).canonical.signature
            == RotationCanonicalizer.analyze(right).canonical.signature
        )


__all__ = [
    "CUBE_ROTATIONS",
    "CubeRotation",
    "FACE_SET_CLASSES",
    "FACE_CLASS_DISPLAY_ORDER",
    "FaceSetRotationClass",
    "FaceSetRotationClassifier",
    "RotationCanonicalizer",
    "RotationEquivalenceResult",
    "RotationEquivalentTransformation",
    "distinct_face_order",
    "full_transformation_signature",
]
