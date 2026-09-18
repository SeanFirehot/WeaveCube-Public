"""Finite rotation group used by the PDCC pose engine.

The engine deliberately uses exact integer matrices.  Every pose and move is one
of the 24 orientation-preserving signed permutation matrices of the cube.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from types import MappingProxyType
from typing import Iterable, Mapping, TypeAlias

Vector3: TypeAlias = tuple[int, int, int]
Matrix3: TypeAlias = tuple[Vector3, Vector3, Vector3]

IDENTITY_MATRIX: Matrix3 = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def matrix_mul(left: Matrix3, right: Matrix3) -> Matrix3:
    """Return ``left @ right`` using exact integer arithmetic."""

    return tuple(
        tuple(
            sum(left[row][k] * right[k][column] for k in range(3))
            for column in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]


def matrix_vec(matrix: Matrix3, vector: Vector3) -> Vector3:
    """Apply a rotation matrix to a 3-vector."""

    return tuple(
        sum(matrix[row][k] * vector[k] for k in range(3)) for row in range(3)
    )  # type: ignore[return-value]


def transpose(matrix: Matrix3) -> Matrix3:
    return tuple(
        tuple(matrix[column][row] for column in range(3)) for row in range(3)
    )  # type: ignore[return-value]


def determinant(matrix: Matrix3) -> int:
    return (
        matrix[0][0]
        * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1]
        * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2]
        * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def rotation_x(quarter_turns: int) -> Matrix3:
    """Right-hand rotation about +x by multiples of 90 degrees."""

    matrices: tuple[Matrix3, ...] = (
        IDENTITY_MATRIX,
        ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
        ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
        ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    )
    return matrices[quarter_turns % 4]


def rotation_y(quarter_turns: int) -> Matrix3:
    """Right-hand rotation about +y by multiples of 90 degrees."""

    matrices: tuple[Matrix3, ...] = (
        IDENTITY_MATRIX,
        ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
        ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
        ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
    )
    return matrices[quarter_turns % 4]


def rotation_z(quarter_turns: int) -> Matrix3:
    """Right-hand rotation about +z by multiples of 90 degrees."""

    matrices: tuple[Matrix3, ...] = (
        IDENTITY_MATRIX,
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    )
    return matrices[quarter_turns % 4]


def _generate_cube_rotations() -> tuple[Matrix3, ...]:
    matrices: set[Matrix3] = set()
    for permutation in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            rows: list[Vector3] = []
            for row_index in range(3):
                row = [0, 0, 0]
                row[permutation[row_index]] = signs[row_index]
                rows.append(tuple(row))
            matrix: Matrix3 = tuple(rows)  # type: ignore[assignment]
            if determinant(matrix) == 1:
                matrices.add(matrix)

    ordered = sorted(matrices)
    ordered.remove(IDENTITY_MATRIX)
    return (IDENTITY_MATRIX, *ordered)


@dataclass(frozen=True, slots=True)
class RotationGroup:
    """Exact table representation of the 24-element cube rotation group."""

    elements: tuple[Matrix3, ...]
    index: Mapping[Matrix3, int]
    multiplication: tuple[tuple[int, ...], ...]
    inverses: tuple[int, ...]
    identity: int = 0

    @classmethod
    def build(cls) -> "RotationGroup":
        elements = _generate_cube_rotations()
        if len(elements) != 24:
            raise RuntimeError(f"Expected 24 cube rotations, got {len(elements)}")

        index_dict = {matrix: position for position, matrix in enumerate(elements)}
        multiplication = tuple(
            tuple(index_dict[matrix_mul(left, right)] for right in elements)
            for left in elements
        )
        inverses = tuple(index_dict[transpose(matrix)] for matrix in elements)
        return cls(
            elements=elements,
            index=MappingProxyType(index_dict),
            multiplication=multiplication,
            inverses=inverses,
        )

    def id_of(self, matrix: Matrix3) -> int:
        try:
            return self.index[matrix]
        except KeyError as exc:
            raise ValueError(f"Matrix is not a cube rotation: {matrix}") from exc

    def matrix(self, rotation_id: int) -> Matrix3:
        return self.elements[rotation_id]

    def compose(self, left: int, right: int) -> int:
        """Return the ID of ``left @ right``."""

        return self.multiplication[left][right]

    def inverse(self, rotation_id: int) -> int:
        return self.inverses[rotation_id]

    def apply(self, rotation_id: int, vector: Vector3) -> Vector3:
        return matrix_vec(self.elements[rotation_id], vector)

    def product(self, rotation_ids: Iterable[int]) -> int:
        result = self.identity
        for rotation_id in rotation_ids:
            result = self.compose(rotation_id, result)
        return result


CUBE_ROTATIONS = RotationGroup.build()
