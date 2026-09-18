from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations
from typing import Literal, Sequence

from .cubie_effect import CubieEffect
from .ordered_piece_route_solver import OrderedPieceRouteSolver
from .piece_route_graph import FACE_MOVES, PieceState, inverse_sequence
from .transformations import PIECES, from_sequence

SearchMode = Literal["plain", "pair_prune", "pair_order"]
Pair = tuple[str, str]
_MOVE_INDEX = {move: index for index, move in enumerate(FACE_MOVES)}
_PIECE_INDEX = {piece: index for index, piece in enumerate(PIECES)}


@dataclass(frozen=True, slots=True)
class DensePairPolicy:
    pieces: Pair
    piece_indices: tuple[int, int]
    distances: tuple[int, ...]
    shortest_masks: tuple[int, ...]
    slack_one_masks: tuple[int, ...]

    def state_index(self, piece_codes: Sequence[int]) -> int:
        first, second = self.piece_indices
        return piece_codes[first] * 24 + piece_codes[second]

    def distance(self, piece_codes: Sequence[int]) -> int:
        distance = self.distances[self.state_index(piece_codes)]
        if distance < 0:
            raise ValueError(f"Unreachable physical state for pair {self.pieces}")
        return distance


@dataclass(frozen=True, slots=True)
class DenseMoveEvaluation:
    move: str
    move_index: int
    next_piece_codes: tuple[int, ...]
    lower_bound_after: int
    shortest_votes: int
    slack_one_votes: int

    @property
    def rank_key(self) -> tuple[int, int, int, int]:
        return (
            self.lower_bound_after,
            -self.shortest_votes,
            -self.slack_one_votes,
            self.move_index,
        )


class DenseProjectionRepository:
    """Reusable dense exact policies for named two-cubie projections."""

    def __init__(self) -> None:
        self.route_solver = OrderedPieceRouteSolver()
        self.corner_states = self.route_solver.corner_graph.states
        self.edge_states = self.route_solver.edge_graph.states
        self._corner_index = {
            state: index for index, state in enumerate(self.corner_states)
        }
        self._edge_index = {
            state: index for index, state in enumerate(self.edge_states)
        }
        self.corner_transitions = self._dense_transitions("corner")
        self.edge_transitions = self._dense_transitions("edge")
        self._policy_cache: dict[frozenset[str], DensePairPolicy] = {}
        self._portfolio_cache: dict[
            tuple[tuple[int, ...], int], tuple[DensePairPolicy, ...]
        ] = {}

    def _dense_transitions(self, piece_type: str) -> tuple[tuple[int, ...], ...]:
        graph = (
            self.route_solver.corner_graph
            if piece_type == "corner"
            else self.route_solver.edge_graph
        )
        index = self._corner_index if piece_type == "corner" else self._edge_index
        return tuple(
            tuple(index[graph.transitions[state][move]] for move in FACE_MOVES)
            for state in graph.states
        )

    def _transitions_for_piece(self, piece_index: int) -> tuple[tuple[int, ...], ...]:
        return self.corner_transitions if piece_index < 8 else self.edge_transitions

    def encode_effect(self, effect: CubieEffect) -> tuple[int, ...]:
        codes: list[int] = []
        for index, piece in enumerate(PIECES):
            state = self.route_solver.piece_state(effect, piece)
            state_index = self._corner_index if index < 8 else self._edge_index
            codes.append(state_index[state])
        return tuple(codes)

    def advance_codes(
        self,
        piece_codes: Sequence[int],
        move_index: int,
    ) -> tuple[int, ...]:
        return tuple(
            self._transitions_for_piece(index)[code][move_index]
            for index, code in enumerate(piece_codes)
        )

    def _target_code(self, piece: str) -> int:
        piece_index = _PIECE_INDEX[piece]
        state = PieceState(piece, 0)
        return (
            self._corner_index[state]
            if piece_index < 8
            else self._edge_index[state]
        )

    def policy(self, first: str, second: str) -> DensePairPolicy:
        if first == second:
            raise ValueError("A pair projection requires distinct cubies")
        if first not in _PIECE_INDEX or second not in _PIECE_INDEX:
            unknown = first if first not in _PIECE_INDEX else second
            raise ValueError(f"Unknown cubie: {unknown}")
        if _PIECE_INDEX[first] > _PIECE_INDEX[second]:
            first, second = second, first
        identity = frozenset((first, second))
        cached = self._policy_cache.get(identity)
        if cached is not None:
            return cached

        first_index = _PIECE_INDEX[first]
        second_index = _PIECE_INDEX[second]
        transitions_a = self._transitions_for_piece(first_index)
        transitions_b = self._transitions_for_piece(second_index)
        target = self._target_code(first) * 24 + self._target_code(second)
        distances = [-1] * (24 * 24)
        distances[target] = 0
        queue = deque([target])
        while queue:
            state_index = queue.popleft()
            state_a, state_b = divmod(state_index, 24)
            next_depth = distances[state_index] + 1
            for move_index in range(len(FACE_MOVES)):
                next_a = transitions_a[state_a][move_index]
                next_b = transitions_b[state_b][move_index]
                next_index = next_a * 24 + next_b
                if distances[next_index] < 0:
                    distances[next_index] = next_depth
                    queue.append(next_index)

        shortest_masks = [0] * (24 * 24)
        slack_one_masks = [0] * (24 * 24)
        for state_index, distance in enumerate(distances):
            if distance < 0:
                continue
            state_a, state_b = divmod(state_index, 24)
            shortest_mask = 0
            slack_mask = 0
            for move_index in range(len(FACE_MOVES)):
                next_a = transitions_a[state_a][move_index]
                next_b = transitions_b[state_b][move_index]
                next_distance = distances[next_a * 24 + next_b]
                if distance > 0 and next_distance == distance - 1:
                    shortest_mask |= 1 << move_index
                if 1 + next_distance <= distance + 1:
                    slack_mask |= 1 << move_index
            shortest_masks[state_index] = shortest_mask
            slack_one_masks[state_index] = slack_mask

        policy = DensePairPolicy(
            pieces=(first, second),
            piece_indices=(first_index, second_index),
            distances=tuple(distances),
            shortest_masks=tuple(shortest_masks),
            slack_one_masks=tuple(slack_one_masks),
        )
        self._policy_cache[identity] = policy
        return policy

    def all_policies(self) -> tuple[DensePairPolicy, ...]:
        return tuple(self.policy(first, second) for first, second in combinations(PIECES, 2))

    def select_portfolio(
        self,
        piece_codes: tuple[int, ...],
        *,
        limit: int = 12,
    ) -> tuple[DensePairPolicy, ...]:
        if not 1 <= limit <= 190:
            raise ValueError("limit must be between 1 and 190")
        cache_key = (piece_codes, limit)
        cached = self._portfolio_cache.get(cache_key)
        if cached is not None:
            return cached

        candidates = list(self.all_policies())
        usage = [0] * len(PIECES)
        selected: list[DensePairPolicy] = []
        while candidates and len(selected) < limit:
            best = max(
                candidates,
                key=lambda policy: self._portfolio_key(policy, piece_codes, usage),
            )
            selected.append(best)
            candidates.remove(best)
            for piece_index in best.piece_indices:
                usage[piece_index] += 1
        result = tuple(selected)
        self._portfolio_cache[cache_key] = result
        return result

    @staticmethod
    def _portfolio_key(
        policy: DensePairPolicy,
        piece_codes: Sequence[int],
        usage: Sequence[int],
    ) -> tuple[int, int, int, int, int, int]:
        first, second = policy.piece_indices
        new_pieces = int(usage[first] == 0) + int(usage[second] == 0)
        mixed_type = int((first < 8) != (second < 8))
        return (
            policy.distance(piece_codes) + 2 * new_pieces,
            policy.distance(piece_codes),
            new_pieces,
            -usage[first] - usage[second],
            mixed_type,
            -first * len(PIECES) - second,
        )

    @staticmethod
    def lower_bound(
        piece_codes: Sequence[int],
        portfolio: Sequence[DensePairPolicy],
    ) -> int:
        return max((policy.distance(piece_codes) for policy in portfolio), default=0)

    def rank_moves(
        self,
        piece_codes: tuple[int, ...],
        portfolio: Sequence[DensePairPolicy],
        *,
        last_face: str = "",
    ) -> tuple[DenseMoveEvaluation, ...]:
        rows: list[DenseMoveEvaluation] = []
        for move_index, move in enumerate(FACE_MOVES):
            if last_face and move[0] == last_face:
                continue
            next_codes = self.advance_codes(piece_codes, move_index)
            lower_bound_after = 0
            shortest_votes = 0
            slack_votes = 0
            bit = 1 << move_index
            for policy in portfolio:
                state_index = policy.state_index(piece_codes)
                lower_bound_after = max(
                    lower_bound_after,
                    policy.distance(next_codes),
                )
                shortest_votes += int(bool(policy.shortest_masks[state_index] & bit))
                slack_votes += int(bool(policy.slack_one_masks[state_index] & bit))
            rows.append(
                DenseMoveEvaluation(
                    move=move,
                    move_index=move_index,
                    next_piece_codes=next_codes,
                    lower_bound_after=lower_bound_after,
                    shortest_votes=shortest_votes,
                    slack_one_votes=slack_votes,
                )
            )
        return tuple(sorted(rows, key=lambda row: row.rank_key))


@dataclass(frozen=True, slots=True)
class DenseProjectionSearchResult:
    solved: bool
    sequence: tuple[str, ...]
    shortest_depth: int | None
    mode: SearchMode
    pair_count: int
    portfolio: tuple[Pair, ...]
    expanded: int
    generated: int
    pruned_pair: int
    thresholds: tuple[int, ...]
    suffix_states: int
    truncated: bool


class DenseProjectionCompleteSolverV64:
    """Complete bounded prefix search with a reusable exact suffix table.

    Pair policies may prove a prefix impossible and may order children, but
    ordering never removes a legal branch.  Exact full-cube state is used only
    for the suffix-table join, duplicate detection, and final verification.
    """

    def __init__(
        self,
        repository: DenseProjectionRepository | None = None,
        *,
        suffix_depth: int = 3,
    ) -> None:
        if suffix_depth < 0:
            raise ValueError("suffix_depth must be non-negative")
        self.repository = repository or DenseProjectionRepository()
        self.suffix_depth = suffix_depth
        self.identity = CubieEffect.identity()
        self.move_effects = {
            move: CubieEffect.from_transformation(from_sequence((move,)))
            for move in FACE_MOVES
        }
        self.suffix_table = self._build_suffix_table()

    def _build_suffix_table(self) -> dict[tuple, tuple[str, ...]]:
        table = {self.identity.signature: ()}
        queue = deque([(self.identity, (), "")])
        while queue:
            effect, path, last_face = queue.popleft()
            if len(path) == self.suffix_depth:
                continue
            for move in FACE_MOVES:
                if last_face and move[0] == last_face:
                    continue
                nxt = self.move_effects[move].compose_after(effect)
                if nxt.signature in table:
                    continue
                next_path = path + (move,)
                table[nxt.signature] = next_path
                queue.append((nxt, next_path, move[0]))
        return table

    def _verify(self, start: CubieEffect, sequence: Sequence[str]) -> bool:
        action = CubieEffect.from_transformation(from_sequence(tuple(sequence)))
        return action.compose_after(start) == self.identity

    def solve(
        self,
        scramble: Sequence[str],
        *,
        bound: int,
        mode: SearchMode = "pair_order",
        pair_limit: int = 8,
        max_nodes: int = 2_000_000,
    ) -> DenseProjectionSearchResult:
        if bound < 0:
            raise ValueError("bound must be non-negative")
        if max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        if mode not in ("plain", "pair_prune", "pair_order"):
            raise ValueError(f"Unsupported search mode: {mode}")

        start = CubieEffect.from_transformation(from_sequence(tuple(scramble)))
        start_codes = self.repository.encode_effect(start)
        portfolio = (
            ()
            if mode == "plain"
            else self.repository.select_portfolio(start_codes, limit=pair_limit)
        )
        root_lower_bound = self.repository.lower_bound(start_codes, portfolio)
        first_threshold = root_lower_bound if portfolio else 0
        expanded = 0
        generated = 0
        pruned_pair = 0
        truncated = False
        solved = False
        answer: tuple[str, ...] = ()
        attempted: list[int] = []

        for threshold in range(first_threshold, bound + 1):
            attempted.append(threshold)
            forward_limit = max(0, threshold - self.suffix_depth)
            seen: dict[tuple[tuple, str], int] = {}
            path: list[str] = []

            def dfs(
                effect: CubieEffect,
                piece_codes: tuple[int, ...],
                last_face: str,
            ) -> bool:
                nonlocal expanded, generated, pruned_pair, truncated, solved, answer
                if expanded >= max_nodes:
                    truncated = True
                    return False

                suffix = self.suffix_table.get(effect.signature)
                if suffix is not None and len(path) + len(suffix) <= threshold:
                    candidate = tuple(path) + inverse_sequence(suffix)
                    if self._verify(start, candidate):
                        solved = True
                        answer = candidate
                        return True

                if len(path) == forward_limit:
                    return False

                remaining = threshold - len(path)
                if portfolio:
                    lower_bound = self.repository.lower_bound(piece_codes, portfolio)
                    if lower_bound > remaining:
                        pruned_pair += 1
                        return False

                transposition_key = (effect.signature, last_face)
                previous = seen.get(transposition_key)
                forward_remaining = forward_limit - len(path)
                if previous is not None and previous >= forward_remaining:
                    return False
                seen[transposition_key] = forward_remaining
                expanded += 1

                if mode == "pair_order":
                    ordered = self.repository.rank_moves(
                        piece_codes,
                        portfolio,
                        last_face=last_face,
                    )
                else:
                    ordered = tuple(
                        DenseMoveEvaluation(
                            move=move,
                            move_index=move_index,
                            next_piece_codes=self.repository.advance_codes(
                                piece_codes,
                                move_index,
                            ),
                            lower_bound_after=0,
                            shortest_votes=0,
                            slack_one_votes=0,
                        )
                        for move_index, move in enumerate(FACE_MOVES)
                        if not last_face or move[0] != last_face
                    )

                for row in ordered:
                    generated += 1
                    nxt = self.move_effects[row.move].compose_after(effect)
                    path.append(row.move)
                    if dfs(nxt, row.next_piece_codes, row.move[0]):
                        return True
                    path.pop()
                    if truncated:
                        return False
                return False

            if dfs(start, start_codes, ""):
                break
            if truncated:
                break

        return DenseProjectionSearchResult(
            solved=solved,
            sequence=answer,
            shortest_depth=len(answer) if solved else None,
            mode=mode,
            pair_count=len(portfolio),
            portfolio=tuple(policy.pieces for policy in portfolio),
            expanded=expanded,
            generated=generated,
            pruned_pair=pruned_pair,
            thresholds=tuple(attempted),
            suffix_states=len(self.suffix_table),
            truncated=truncated,
        )
__all__ = [
    "DenseMoveEvaluation",
    "DensePairPolicy",
    "DenseProjectionCompleteSolverV64",
    "DenseProjectionRepository",
    "DenseProjectionSearchResult",
    "Pair",
    "SearchMode",
]
