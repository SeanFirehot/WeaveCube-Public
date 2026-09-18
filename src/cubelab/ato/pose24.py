"""Deterministic Q, Omega, and kappa coordinate tables."""

from __future__ import annotations

from cubelab.pdcc.moves import MOVES

from .matrices import OMEGA_MATRICES, OMEGA_MUL, Q_MATRICES, Q_MUL, omega_id


# Frozen CubeLab COLUMN order from ``move_demand_balance.MOVE_ORDER``.  It is
# repeated here because that historical module has optional build-time imports
# that are intentionally unavailable in the minimal foundation environment.
MOVE_NAMES: tuple[str, ...] = tuple(
    f"{face}{suffix}"
    for face in "UDLRFB"
    for suffix in ("", "'", "2")
)
MOVE_INDEX = {name: index for index, name in enumerate(MOVE_NAMES)}
MOVE_ROTATION_IDS: tuple[int, ...] = tuple(MOVES[name].rotation_id for name in MOVE_NAMES)
MOVE_ROTATION = tuple(Q_MATRICES[rotation_id] for rotation_id in MOVE_ROTATION_IDS)

Q_TO_OMEGA: tuple[int, ...] = tuple(omega_id(matrix) for matrix in Q_MATRICES)
OMEGA_FIBERS: tuple[tuple[int, ...], ...] = tuple(
    tuple(q for q, omega in enumerate(Q_TO_OMEGA) if omega == omega_id_value)
    for omega_id_value in range(len(OMEGA_MATRICES))
)
Q_TO_KAPPA_LIST = [0] * len(Q_MATRICES)
for omega, fiber in enumerate(OMEGA_FIBERS):
    for kappa, q in enumerate(fiber):
        Q_TO_KAPPA_LIST[q] = kappa
Q_TO_KAPPA: tuple[int, ...] = tuple(Q_TO_KAPPA_LIST)
OMEGA_KAPPA_TO_Q: tuple[tuple[int, ...], ...] = OMEGA_FIBERS
KERNEL_Q_IDS: tuple[int, ...] = OMEGA_FIBERS[0]


def q_to_omega_kappa(q: int) -> tuple[int, int]:
    return Q_TO_OMEGA[q], Q_TO_KAPPA[q]


def q_from_omega_kappa(omega: int, kappa: int) -> int:
    return OMEGA_KAPPA_TO_Q[omega][kappa]


def compose_q(left: int, right: int) -> int:
    return Q_MUL[left][right]


def affected_omega_transition(omega: int, move: str | int) -> int:
    move_index = MOVE_INDEX[move] if isinstance(move, str) else move
    move_omega = Q_TO_OMEGA[MOVE_ROTATION_IDS[move_index]]
    return OMEGA_MUL[move_omega][omega]


def coordinate_product(
    left_omega: int,
    left_kappa: int,
    right_omega: int,
    right_kappa: int,
) -> tuple[int, int]:
    q = Q_MUL[
        q_from_omega_kappa(left_omega, left_kappa)
    ][q_from_omega_kappa(right_omega, right_kappa)]
    return q_to_omega_kappa(q)


__all__ = [
    "KERNEL_Q_IDS",
    "MOVE_INDEX",
    "MOVE_NAMES",
    "MOVE_ROTATION",
    "MOVE_ROTATION_IDS",
    "OMEGA_FIBERS",
    "OMEGA_KAPPA_TO_Q",
    "Q_TO_KAPPA",
    "Q_TO_OMEGA",
    "affected_omega_transition",
    "compose_q",
    "coordinate_product",
    "q_from_omega_kappa",
    "q_to_omega_kappa",
]
