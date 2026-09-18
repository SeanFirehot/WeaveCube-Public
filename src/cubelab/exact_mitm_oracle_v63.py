from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

from .cubie_effect import CubieEffect
from .piece_route_graph import FACE_MOVES, inverse_sequence
from .transformations import from_sequence


@dataclass(frozen=True, slots=True)
class ExactMitmOracleResult:
    """Exact bounded result with deterministic shortest-solution witnesses.

    ``solutions`` is deliberately a witness sample, not an enumeration of all
    shortest paths.  The oracle keeps one shortest path per encountered cube
    state and collects distinct witnesses through every minimum-depth MITM
    intersection.  That is enough to avoid auditing route recall against only
    the literal inverse while keeping the table compact.
    """

    solved: bool
    bound: int
    shortest_depth: int | None
    solutions: tuple[tuple[str, ...], ...]
    forward_expanded: int
    forward_states: int
    backward_states: int
    minimum_intersections: int

    @property
    def sequence(self) -> tuple[str, ...]:
        """Return the first deterministic witness for single-solution callers."""
        return self.solutions[0] if self.solutions else ()


@dataclass(frozen=True, slots=True)
class _StateWitness:
    effect: CubieEffect
    path: tuple[str, ...]


class ExactMitmOracleV63:
    """Full-state bounded meet-in-the-middle oracle.

    Unlike the earlier first-hit MITM helper, this implementation evaluates
    every reachable intersection in the selected split and returns the true
    minimum depth within ``bound``.  Full-cube exact state is used only for
    identity checking, duplicate-state detection, and the MITM join.  It is
    never converted into a partial-state proximity score.
    """

    def __init__(self, backward_depth: int = 4) -> None:
        if backward_depth < 0:
            raise ValueError("backward_depth must be non-negative")
        self.backward_depth = backward_depth
        self.identity = CubieEffect.identity()
        self.move_effects = {
            move: CubieEffect.from_transformation(from_sequence((move,)))
            for move in FACE_MOVES
        }
        self._backward_tables: dict[int, dict[tuple, _StateWitness]] = {}

    def _build_backward_table(self, depth: int) -> dict[tuple, _StateWitness]:
        cached = self._backward_tables.get(depth)
        if cached is not None:
            return cached

        root = _StateWitness(self.identity, ())
        table: dict[tuple, _StateWitness] = {self.identity.signature: root}
        queue = deque([(root, "")])
        while queue:
            witness, last_face = queue.popleft()
            if len(witness.path) == depth:
                continue
            for move in FACE_MOVES:
                if last_face and move[0] == last_face:
                    continue
                effect = self.move_effects[move].compose_after(witness.effect)
                signature = effect.signature
                if signature in table:
                    continue
                child = _StateWitness(effect, witness.path + (move,))
                table[signature] = child
                queue.append((child, move[0]))

        self._backward_tables[depth] = table
        return table

    def _verify_solution(self, start: CubieEffect, sequence: Sequence[str]) -> bool:
        action = CubieEffect.from_transformation(from_sequence(tuple(sequence)))
        return action.compose_after(start) == self.identity

    def solve(
        self,
        scramble: Sequence[str],
        *,
        bound: int,
        max_solutions: int = 64,
    ) -> ExactMitmOracleResult:
        if bound < 0:
            raise ValueError("bound must be non-negative")
        if max_solutions < 1:
            raise ValueError("max_solutions must be positive")

        start = CubieEffect.from_transformation(from_sequence(tuple(scramble)))
        if start == self.identity:
            return ExactMitmOracleResult(
                solved=True,
                bound=bound,
                shortest_depth=0,
                solutions=((),),
                forward_expanded=0,
                forward_states=1,
                backward_states=1,
                minimum_intersections=1,
            )

        backward_depth = min(self.backward_depth, bound)
        forward_depth = bound - backward_depth
        backward = self._build_backward_table(backward_depth)

        root = _StateWitness(start, ())
        forward: dict[tuple, _StateWitness] = {start.signature: root}
        queue = deque([(root, "")])
        expanded = 0
        intersections: list[tuple[int, tuple[str, ...]]] = []

        while queue:
            witness, last_face = queue.popleft()
            expanded += 1
            other = backward.get(witness.effect.signature)
            if other is not None:
                candidate = witness.path + inverse_sequence(other.path)
                if len(candidate) <= bound and self._verify_solution(start, candidate):
                    intersections.append((len(candidate), candidate))

            if len(witness.path) == forward_depth:
                continue
            for move in FACE_MOVES:
                if last_face and move[0] == last_face:
                    continue
                effect = self.move_effects[move].compose_after(witness.effect)
                signature = effect.signature
                if signature in forward:
                    continue
                child = _StateWitness(effect, witness.path + (move,))
                forward[signature] = child
                queue.append((child, move[0]))

        if not intersections:
            return ExactMitmOracleResult(
                solved=False,
                bound=bound,
                shortest_depth=None,
                solutions=(),
                forward_expanded=expanded,
                forward_states=len(forward),
                backward_states=len(backward),
                minimum_intersections=0,
            )

        shortest_depth = min(depth for depth, _ in intersections)
        shortest = sorted(
            {
                sequence
                for depth, sequence in intersections
                if depth == shortest_depth
            },
            key=lambda sequence: tuple(FACE_MOVES.index(move) for move in sequence),
        )
        return ExactMitmOracleResult(
            solved=True,
            bound=bound,
            shortest_depth=shortest_depth,
            solutions=tuple(shortest[:max_solutions]),
            forward_expanded=expanded,
            forward_states=len(forward),
            backward_states=len(backward),
            minimum_intersections=len(shortest),
        )


__all__ = ["ExactMitmOracleResult", "ExactMitmOracleV63"]
