from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

from .cubie_effect import CubieEffect
from .pieces import CORNER_ORDER, EDGE_ORDER
from .transformations import PIECES, from_sequence

if TYPE_CHECKING:
    from .piecewise_effects import TransformationEffectVector

FACE_MOVES: tuple[str, ...] = tuple(
    f"{face}{suffix}"
    for face in ("R", "U", "F", "L", "D", "B")
    for suffix in ("", "'", "2")
)
_FACE_INDEX = {face: index for index, face in enumerate(("R", "U", "F", "L", "D", "B"))}
_PIECE_INDEX = {piece: index for index, piece in enumerate(PIECES)}
_CORNER_INDEX = {piece: index for index, piece in enumerate(CORNER_ORDER)}
_EDGE_INDEX = {piece: index for index, piece in enumerate(EDGE_ORDER)}


@dataclass(frozen=True, slots=True, order=True)
class PieceState:
    position: str
    orientation: int


@dataclass(frozen=True, slots=True)
class RouteWitness:
    sequence: tuple[str, ...]
    face_vector: tuple[int, int, int, int, int, int]

    @classmethod
    def from_sequence(cls, sequence: Sequence[str]) -> "RouteWitness":
        seq = tuple(sequence)
        return cls(seq, face_vector(seq))


@dataclass(frozen=True, slots=True)
class MinimalRouteSet:
    piece_type: str
    source: PieceState
    target: PieceState
    minimum_depth: int
    witnesses: tuple[RouteWitness, ...]


@dataclass(frozen=True, slots=True)
class NeutralResidual:
    first: RouteWitness
    second: RouteWitness
    sequence: tuple[str, ...]
    effect: "TransformationEffectVector"


def face_vector(sequence: Sequence[str]) -> tuple[int, int, int, int, int, int]:
    values = [0] * 6
    for token in sequence:
        if not token or token[0] not in _FACE_INDEX:
            raise ValueError(f"Unsupported face move: {token}")
        amount = 1 if len(token) == 1 else 2 if token[1:] == "2" else 3 if token[1:] == "'" else None
        if amount is None:
            raise ValueError(f"Unsupported face move: {token}")
        values[_FACE_INDEX[token[0]]] = (values[_FACE_INDEX[token[0]]] + amount) % 4
    return tuple(values)  # type: ignore[return-value]


def inverse_sequence(sequence: Sequence[str]) -> tuple[str, ...]:
    def inverse_token(token: str) -> str:
        if token.endswith("2"):
            return token
        if token.endswith("'"):
            return token[:-1]
        return token + "'"
    return tuple(inverse_token(token) for token in reversed(sequence))


class SinglePieceRouteGraph:
    """Exact 24-state graph for one corner or edge cubie.

    The graph tracks only the chosen cubie's position and orientation, while
    each edge records a legal full-cube face turn. All shortest move-token
    sequences are retained, not just one BFS parent.
    """

    def __init__(self, piece_type: str, moves: Sequence[str] = FACE_MOVES) -> None:
        if piece_type not in {"corner", "edge"}:
            raise ValueError("piece_type must be 'corner' or 'edge'")
        self.piece_type = piece_type
        self.positions = CORNER_ORDER if piece_type == "corner" else EDGE_ORDER
        self.modulus = 3 if piece_type == "corner" else 2
        self.moves = tuple(moves)
        self._move_effects = {move: CubieEffect.from_transformation(from_sequence((move,))) for move in self.moves}
        self.states = tuple(PieceState(position, orientation) for position in self.positions for orientation in range(self.modulus))
        self.transitions: Mapping[PieceState, Mapping[str, PieceState]] = {
            state: {move: self.apply_move(state, move) for move in self.moves}
            for state in self.states
        }
        self._distance_table: dict[PieceState, dict[PieceState, int]] = {
            source: self._bfs_distances(source) for source in self.states
        }

    def _bfs_distances(self, source: PieceState) -> dict[PieceState, int]:
        distance = {source: 0}
        queue = deque([source])
        while queue:
            state = queue.popleft()
            next_depth = distance[state] + 1
            for next_state in self.transitions[state].values():
                if next_state not in distance:
                    distance[next_state] = next_depth
                    queue.append(next_state)
        return distance

    def distance(self, source: PieceState, target: PieceState) -> int:
        """Return exact HTM distance without reconstructing route witnesses."""
        self._validate_state(source)
        self._validate_state(target)
        try:
            return self._distance_table[source][target]
        except KeyError as exc:
            raise ValueError(f"Target {target} is unreachable from {source}") from exc

    def apply_move(self, state: PieceState, move: str) -> PieceState:
        effect = self._move_effects[move]
        destination = PIECES[effect.destinations[_PIECE_INDEX[state.position]]]
        if self.piece_type == "corner":
            delta = effect.corner_twists[_CORNER_INDEX[state.position]]
        else:
            delta = effect.edge_flips[_EDGE_INDEX[state.position]]
        return PieceState(destination, (state.orientation + delta) % self.modulus)

    def apply_sequence(self, source: PieceState, sequence: Sequence[str]) -> PieceState:
        """Apply a move sequence in the exact one-piece 24-state projection."""
        self._validate_state(source)
        state = source
        for move in sequence:
            state = self.transitions[state][move]
        return state

    def all_shortest_routes(self, source: PieceState, target: PieceState) -> MinimalRouteSet:
        self._validate_state(source)
        self._validate_state(target)
        distance: dict[PieceState, int] = {source: 0}
        parents: dict[PieceState, list[tuple[PieceState, str]]] = defaultdict(list)
        queue = deque([source])

        while queue:
            state = queue.popleft()
            depth = distance[state]
            if target in distance and depth >= distance[target]:
                continue
            for move, next_state in self.transitions[state].items():
                next_depth = depth + 1
                if next_state not in distance:
                    distance[next_state] = next_depth
                    parents[next_state].append((state, move))
                    queue.append(next_state)
                elif distance[next_state] == next_depth:
                    parents[next_state].append((state, move))

        if target not in distance:
            raise ValueError(f"Target {target} is unreachable from {source}")

        memo: dict[PieceState, tuple[tuple[str, ...], ...]] = {}

        def recover(state: PieceState) -> tuple[tuple[str, ...], ...]:
            if state == source:
                return ((),)
            if state in memo:
                return memo[state]
            paths = []
            for previous, move in parents[state]:
                for prefix in recover(previous):
                    paths.append(prefix + (move,))
            memo[state] = tuple(sorted(set(paths)))
            return memo[state]

        sequences = recover(target)
        return MinimalRouteSet(
            piece_type=self.piece_type,
            source=source,
            target=target,
            minimum_depth=distance[target],
            witnesses=tuple(RouteWitness.from_sequence(sequence) for sequence in sequences),
        )


    def routes_with_slack(
        self,
        source: PieceState,
        target: PieceState,
        *,
        slack: int = 0,
        max_witnesses: int | None = None,
    ) -> tuple[RouteWitness, ...]:
        """Return reduced routes whose length is at most shortest+slack.

        Adjacent turns of the same face are excluded because they reduce to a
        single HTM move (or cancel), so retaining them would only create
        syntactic duplicates.  The exact shortest witnesses are always first.
        """
        if slack < 0:
            raise ValueError("slack must be non-negative")
        shortest = self.all_shortest_routes(source, target)
        if slack == 0:
            witnesses = shortest.witnesses
            return witnesses if max_witnesses is None else witnesses[:max_witnesses]

        # Reverse BFS gives an exact remaining-distance bound in the 24-state
        # projection, allowing exhaustive bounded DFS without blind expansion.
        reverse_distance: dict[PieceState, int] = {target: 0}
        queue = deque([target])
        reverse_edges: dict[PieceState, list[PieceState]] = defaultdict(list)
        for state in self.states:
            for next_state in self.transitions[state].values():
                reverse_edges[next_state].append(state)
        while queue:
            state = queue.popleft()
            for previous in reverse_edges[state]:
                if previous not in reverse_distance:
                    reverse_distance[previous] = reverse_distance[state] + 1
                    queue.append(previous)

        maximum_depth = shortest.minimum_depth + slack
        sequences: set[tuple[str, ...]] = set()

        def dfs(state: PieceState, sequence: tuple[str, ...]) -> None:
            if len(sequence) + reverse_distance[state] > maximum_depth:
                return
            if state == target:
                sequences.add(sequence)
                return
            last_face = sequence[-1][0] if sequence else None
            for move in self.moves:
                if move[0] == last_face:
                    continue
                dfs(self.transitions[state][move], sequence + (move,))

        dfs(source, ())
        ordered = sorted(sequences, key=lambda seq: (len(seq), seq))
        if max_witnesses is not None:
            ordered = ordered[:max_witnesses]
        return tuple(RouteWitness.from_sequence(sequence) for sequence in ordered)

    def routes_to_target(self, target: PieceState) -> tuple[MinimalRouteSet, ...]:
        return tuple(self.all_shortest_routes(source, target) for source in self.states)

    def _validate_state(self, state: PieceState) -> None:
        if state.position not in self.positions:
            raise ValueError(f"Invalid {self.piece_type} position: {state.position}")
        if not 0 <= state.orientation < self.modulus:
            raise ValueError(f"Invalid {self.piece_type} orientation: {state.orientation}")


def neutral_residuals(route_set: MinimalRouteSet) -> tuple[NeutralResidual, ...]:
    """Build B^-1 A residuals for every unordered pair of minimal witnesses."""
    from .piecewise_effects import TransformationEffectVector
    residuals = []
    for first, second in combinations(route_set.witnesses, 2):
        sequence = first.sequence + inverse_sequence(second.sequence)
        effect = TransformationEffectVector.from_sequence(
            sequence,
            name=f"{' '.join(first.sequence)} · ({' '.join(second.sequence)})^-1",
        )
        residuals.append(NeutralResidual(first, second, sequence, effect))
    return tuple(residuals)


__all__ = [
    "FACE_MOVES",
    "PieceState",
    "RouteWitness",
    "MinimalRouteSet",
    "NeutralResidual",
    "SinglePieceRouteGraph",
    "face_vector",
    "inverse_sequence",
    "neutral_residuals",
]
