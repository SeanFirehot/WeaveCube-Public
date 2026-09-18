"""Exact signed-permutation algebra for ATO Q and Omega states."""

from __future__ import annotations

from types import MappingProxyType
from typing import TypeAlias

from cubelab.pdcc.group import (
    CUBE_ROTATIONS,
    IDENTITY_MATRIX,
    Matrix3,
    determinant,
    matrix_mul,
    transpose,
)

OmegaMatrix: TypeAlias = Matrix3


def absolute_matrix(matrix: Matrix3) -> Matrix3:
    return tuple(tuple(abs(value) for value in row) for row in matrix)  # type: ignore[return-value]


def is_signed_permutation(matrix: Matrix3, *, proper: bool = True) -> bool:
    values_ok = all(value in (-1, 0, 1) for row in matrix for value in row)
    row_ok = all(sum(value != 0 for value in row) == 1 for row in matrix)
    column_ok = all(
        sum(matrix[row][column] != 0 for row in range(3)) == 1
        for column in range(3)
    )
    return values_ok and row_ok and column_ok and (
        not proper or determinant(matrix) == 1
    )


def is_permutation_matrix(matrix: Matrix3) -> bool:
    return is_signed_permutation(matrix, proper=False) and all(
        value >= 0 for row in matrix for value in row
    )


Q_MATRICES: tuple[Matrix3, ...] = CUBE_ROTATIONS.elements
Q_INDEX = CUBE_ROTATIONS.index
Q_MUL: tuple[tuple[int, ...], ...] = CUBE_ROTATIONS.multiplication
Q_INV: tuple[int, ...] = CUBE_ROTATIONS.inverses

_omega_set = {absolute_matrix(matrix) for matrix in Q_MATRICES}
_omega_set.remove(IDENTITY_MATRIX)
OMEGA_MATRICES: tuple[OmegaMatrix, ...] = (
    IDENTITY_MATRIX,
    *sorted(_omega_set),
)
OMEGA_INDEX = MappingProxyType(
    {matrix: index for index, matrix in enumerate(OMEGA_MATRICES)}
)
OMEGA_MUL: tuple[tuple[int, ...], ...] = tuple(
    tuple(OMEGA_INDEX[matrix_mul(left, right)] for right in OMEGA_MATRICES)
    for left in OMEGA_MATRICES
)
OMEGA_INV: tuple[int, ...] = tuple(
    OMEGA_INDEX[transpose(matrix)] for matrix in OMEGA_MATRICES
)


def omega_id(matrix: Matrix3) -> int:
    return OMEGA_INDEX[absolute_matrix(matrix)]


__all__ = [
    "IDENTITY_MATRIX",
    "OMEGA_INDEX",
    "OMEGA_INV",
    "OMEGA_MATRICES",
    "OMEGA_MUL",
    "OmegaMatrix",
    "Q_INDEX",
    "Q_INV",
    "Q_MATRICES",
    "Q_MUL",
    "absolute_matrix",
    "determinant",
    "is_permutation_matrix",
    "is_signed_permutation",
    "matrix_mul",
    "omega_id",
    "transpose",
]
