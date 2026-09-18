"""Admissible heuristic contracts and CubeLab-owned small exact PDB."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .coord_state import CoordinateState
from .move_tables import MOVE_ORDER, apply_pose_move, solved_poses


class HeuristicProvider(Protocol):
    def estimate(self, state: CoordinateState) -> int: ...

    def forbidden_moves(self, state: CoordinateState, remaining: int) -> frozenset[str]: ...


@dataclass(frozen=True, slots=True)
class ZeroHeuristic:
    def estimate(self, state: CoordinateState) -> int:
        return 0

    def forbidden_moves(self, state: CoordinateState, remaining: int) -> frozenset[str]:
        return frozenset()


@dataclass(frozen=True, slots=True)
class SmallExactPDB:
    distances: dict[tuple[int, ...], int]
    maximum_depth: int

    @classmethod
    def build(cls, maximum_depth: int = 4) -> "SmallExactPDB":
        solved = solved_poses()
        distances = {solved: 0}
        queue: deque[tuple[int, ...]] = deque((solved,))
        while queue:
            state = queue.popleft()
            depth = distances[state]
            if depth >= maximum_depth:
                continue
            for move in MOVE_ORDER:
                child = apply_pose_move(state, move)
                if child not in distances:
                    distances[child] = depth + 1
                    queue.append(child)
        return cls(distances=distances, maximum_depth=maximum_depth)

    def estimate(self, state: CoordinateState) -> int:
        return self.distances.get(state.poses, 0)

    def forbidden_moves(self, state: CoordinateState, remaining: int) -> frozenset[str]:
        current = self.estimate(state)
        if current != remaining:
            return frozenset()
        return frozenset(
            move for move in MOVE_ORDER if self.estimate(state.move(move)) >= remaining
        )


@dataclass(frozen=True, slots=True)
class OwnedGX31Metadata:
    path: Path
    domain_size: int
    solved_index: int
    payload_sha256: str
    complete: bool


class OwnedGX31Provider:
    """Fail-closed placeholder until a CubeLab-owned complete GX31 asset exists."""

    def __init__(self, metadata: OwnedGX31Metadata) -> None:
        if not metadata.complete or metadata.domain_size != 9_863_588_700:
            raise ValueError("GX31 asset is absent or incomplete")
        if metadata.solved_index != 0:
            raise ValueError("GX31 solved index is not the owned-format zero")
        raise NotImplementedError("GX31 reader opens only after the shadow parity gate")
