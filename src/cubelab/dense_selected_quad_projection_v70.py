from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from itertools import combinations
import time
from typing import Sequence

from .byte_suffix_exact_key_v69 import (
    ByteSuffixExactKeyResult,
    ByteSuffixExactKeySolverV69,
    shared_byte_suffix_index_v69,
)
from .completed_proof_lookup_gate_v68 import (
    shared_completed_proof_lookup_solver_v68,
)
from .cubie_effect import CubieEffect
from .dense_projection_search_v64 import (
    DenseMoveEvaluation,
    DensePairPolicy,
    DenseProjectionRepository,
)
from .piece_route_graph import FACE_MOVES
from .transformations import PIECES, from_sequence


Quad = tuple[str, str, str, str]
_PIECE_INDEX = {piece: index for index, piece in enumerate(PIECES)}
_UNKNOWN = 255


@dataclass(frozen=True, slots=True)
class QuadCandidateFeatures:
    pieces: Quad
    max_pair_distance: int
    sum_pair_distance: int
    min_pair_distance: int
    common_shortest_moves: int
    union_shortest_moves: int
    mixed_piece_types: bool


@dataclass(frozen=True, slots=True)
class DenseQuadPolicyV70:
    pieces: Quad
    piece_indices: tuple[int, int, int, int]
    distances: bytes
    reachable_states: int
    maximum_distance: int
    cutoff_depth: int
    build_seconds: float

    def state_index(self, piece_codes: Sequence[int]) -> int:
        first, second, third, fourth = self.piece_indices
        return (
            ((piece_codes[first] * 24 + piece_codes[second]) * 24
             + piece_codes[third])
            * 24
            + piece_codes[fourth]
        )

    def distance(self, piece_codes: Sequence[int]) -> int:
        distance = self.distances[self.state_index(piece_codes)]
        if distance == _UNKNOWN:
            return self.cutoff_depth + 1
        return distance


@dataclass(frozen=True, slots=True)
class SelectedQuadMetricsV70:
    candidate_limit: int
    selected_limit: int
    candidates: tuple[QuadCandidateFeatures, ...]
    candidate_distances: tuple[tuple[Quad, int, int], ...]
    selected: tuple[Quad, ...]
    selected_distances: tuple[int, ...]
    selected_pair_gains: tuple[int, ...]
    new_tables_built: int
    table_cache_hits: int
    preparation_seconds: float
    table_build_seconds: float


@dataclass(frozen=True, slots=True)
class DenseSelectedQuadResultV70:
    search: ByteSuffixExactKeyResult
    quad_metrics: SelectedQuadMetricsV70
    precheck: ByteSuffixExactKeyResult | None = None

    @property
    def solved(self) -> bool:
        return self.search.solved

    @property
    def sequence(self) -> tuple[str, ...]:
        return self.search.sequence

    @property
    def shortest_depth(self) -> int | None:
        return self.search.shortest_depth

    @property
    def expanded(self) -> int:
        return self.search.expanded + (
            self.precheck.expanded if self.precheck is not None else 0
        )

    @property
    def generated(self) -> int:
        return self.search.generated + (
            self.precheck.generated if self.precheck is not None else 0
        )

    @property
    def truncated(self) -> bool:
        return self.search.truncated or (
            self.precheck.truncated if self.precheck is not None else False
        )

    @property
    def pruned_projection(self) -> int:
        return self.search.pruned_pair + (
            self.precheck.pruned_pair if self.precheck is not None else 0
        )

    @property
    def pruned_completed_cross_lane(self) -> int:
        return self.search.pruned_completed_cross_lane + (
            self.precheck.pruned_completed_cross_lane
            if self.precheck is not None
            else 0
        )

    @property
    def completed_proofs_available(self) -> int:
        return self.search.completed_proofs_available + (
            self.precheck.completed_proofs_available
            if self.precheck is not None
            else 0
        )

    @property
    def thresholds(self) -> tuple[int, ...]:
        return (
            self.precheck.thresholds + self.search.thresholds
            if self.precheck is not None
            else self.search.thresholds
        )


class DenseQuadPolicyBankV70:
    """Process-local dense exact four-cubie PDB cache and selector.

    A cheap pair-distance/move-mask filter chooses a small candidate set.  Only
    those candidates receive a dense exact four-cubie table.  Table selection
    can weaken the lower bound but never removes a legal move.
    """

    def __init__(
        self,
        repository: DenseProjectionRepository,
        *,
        cutoff_depth: int = 6,
        max_cached_tables: int = 64,
    ) -> None:
        if cutoff_depth < 0 or cutoff_depth >= _UNKNOWN - 1:
            raise ValueError("quad cutoff depth must be in 0..253")
        if max_cached_tables < 1:
            raise ValueError("maximum cached quad tables must be positive")
        self.repository = repository
        self.cutoff_depth = cutoff_depth
        self.max_cached_tables = max_cached_tables
        self.identity_codes = repository.encode_effect(CubieEffect.identity())
        self._policies: OrderedDict[Quad, DenseQuadPolicyV70] = OrderedDict()

    @staticmethod
    def _ordered_quad(raw: Sequence[str]) -> Quad:
        ordered = tuple(sorted(raw, key=_PIECE_INDEX.__getitem__))
        if len(ordered) != 4 or len(set(ordered)) != 4:
            raise ValueError("quad must contain four distinct cubies")
        return ordered  # type: ignore[return-value]

    def policy(self, pieces: Sequence[str]) -> tuple[DenseQuadPolicyV70, bool]:
        quad = self._ordered_quad(pieces)
        cached = self._policies.get(quad)
        if cached is not None:
            self._policies.move_to_end(quad)
            return cached, True

        piece_indices = tuple(_PIECE_INDEX[piece] for piece in quad)
        transitions = tuple(
            self.repository._transitions_for_piece(piece_index)
            for piece_index in piece_indices
        )
        size = 24**4
        distances = bytearray([_UNKNOWN]) * size
        target = 0
        for piece_index in piece_indices:
            target = target * 24 + self.identity_codes[piece_index]
        distances[target] = 0
        queue = deque([target])
        reachable = 1
        maximum_distance = 0
        started = time.perf_counter()

        while queue:
            state_index = queue.popleft()
            depth = distances[state_index]
            if depth >= self.cutoff_depth:
                continue
            first, remainder = divmod(state_index, 24**3)
            second, remainder = divmod(remainder, 24**2)
            third, fourth = divmod(remainder, 24)
            codes = (first, second, third, fourth)
            next_depth = depth + 1
            for move_index in range(len(FACE_MOVES)):
                next_index = 0
                for transition, code in zip(transitions, codes):
                    next_index = (
                        next_index * 24 + transition[code][move_index]
                    )
                if distances[next_index] != _UNKNOWN:
                    continue
                distances[next_index] = next_depth
                queue.append(next_index)
                reachable += 1
                maximum_distance = max(maximum_distance, next_depth)

        policy = DenseQuadPolicyV70(
            pieces=quad,
            piece_indices=piece_indices,  # type: ignore[arg-type]
            distances=bytes(distances),
            reachable_states=reachable,
            maximum_distance=maximum_distance,
            cutoff_depth=self.cutoff_depth,
            build_seconds=time.perf_counter() - started,
        )
        self._policies[quad] = policy
        while len(self._policies) > self.max_cached_tables:
            self._policies.popitem(last=False)
        return policy, False

    def candidate_features(
        self,
        piece_codes: Sequence[int],
        quad: Sequence[str],
    ) -> QuadCandidateFeatures:
        pieces = self._ordered_quad(quad)
        pair_distances: list[int] = []
        active_masks: list[int] = []
        union_mask = 0
        for first, second in combinations(pieces, 2):
            policy = self.repository.policy(first, second)
            state_index = policy.state_index(piece_codes)
            distance = policy.distances[state_index]
            if distance < 0:
                raise ValueError(f"Unreachable pair state for {policy.pieces}")
            pair_distances.append(distance)
            if distance:
                mask = policy.shortest_masks[state_index]
                active_masks.append(mask)
                union_mask |= mask
        common_mask = (1 << len(FACE_MOVES)) - 1
        for mask in active_masks:
            common_mask &= mask
        if not active_masks:
            common_mask = 0
        indices = tuple(_PIECE_INDEX[piece] for piece in pieces)
        return QuadCandidateFeatures(
            pieces=pieces,
            max_pair_distance=max(pair_distances),
            sum_pair_distance=sum(pair_distances),
            min_pair_distance=min(pair_distances),
            common_shortest_moves=common_mask.bit_count(),
            union_shortest_moves=union_mask.bit_count(),
            mixed_piece_types=any(index < 8 for index in indices)
            and any(index >= 8 for index in indices),
        )

    @staticmethod
    def _candidate_key(features: QuadCandidateFeatures) -> tuple:
        return (
            features.max_pair_distance,
            features.sum_pair_distance,
            -features.common_shortest_moves,
            features.min_pair_distance,
            features.union_shortest_moves,
            int(features.mixed_piece_types),
            tuple(-_PIECE_INDEX[piece] for piece in features.pieces),
        )

    def ranked_candidates(
        self,
        piece_codes: Sequence[int],
        *,
        limit: int,
    ) -> tuple[QuadCandidateFeatures, ...]:
        if limit < 1:
            raise ValueError("candidate limit must be positive")
        unsolved = [
            piece
            for index, piece in enumerate(PIECES)
            if piece_codes[index] != self.identity_codes[index]
        ]
        if len(unsolved) < 4:
            return ()
        rows = [
            self.candidate_features(piece_codes, quad)
            for quad in combinations(unsolved, 4)
        ]
        rows.sort(key=self._candidate_key, reverse=True)
        return tuple(rows[:limit])

    def select(
        self,
        piece_codes: Sequence[int],
        *,
        candidate_limit: int,
        selected_limit: int,
    ) -> tuple[tuple[DenseQuadPolicyV70, ...], SelectedQuadMetricsV70]:
        if selected_limit < 1:
            raise ValueError("selected limit must be positive")
        if selected_limit > candidate_limit:
            raise ValueError("selected limit cannot exceed candidate limit")
        started = time.perf_counter()
        candidates = self.ranked_candidates(piece_codes, limit=candidate_limit)
        evaluated: list[
            tuple[DenseQuadPolicyV70, QuadCandidateFeatures, int, int]
        ] = []
        built = hits = 0
        build_seconds = 0.0
        for features in candidates:
            policy, cache_hit = self.policy(features.pieces)
            hits += int(cache_hit)
            built += int(not cache_hit)
            if not cache_hit:
                build_seconds += policy.build_seconds
            distance = policy.distance(piece_codes)
            gain = distance - features.max_pair_distance
            evaluated.append((policy, features, distance, gain))

        evaluated.sort(
            key=lambda row: (
                row[2],
                row[3],
                self._candidate_key(row[1]),
            ),
            reverse=True,
        )
        selected = tuple(row[0] for row in evaluated[:selected_limit])
        metrics = SelectedQuadMetricsV70(
            candidate_limit=candidate_limit,
            selected_limit=selected_limit,
            candidates=candidates,
            candidate_distances=tuple(
                (row[0].pieces, row[2], row[3]) for row in evaluated
            ),
            selected=tuple(policy.pieces for policy in selected),
            selected_distances=tuple(
                policy.distance(piece_codes) for policy in selected
            ),
            selected_pair_gains=tuple(
                policy.distance(piece_codes)
                - self.candidate_features(piece_codes, policy.pieces).max_pair_distance
                for policy in selected
            ),
            new_tables_built=built,
            table_cache_hits=hits,
            preparation_seconds=time.perf_counter() - started,
            table_build_seconds=build_seconds,
        )
        return selected, metrics


class _QuadAugmentedRepositoryV70:
    def __init__(
        self,
        base: DenseProjectionRepository,
        bank: DenseQuadPolicyBankV70,
        *,
        candidate_limit: int,
        selected_limit: int,
    ) -> None:
        self.base = base
        self.bank = bank
        self.candidate_limit = candidate_limit
        self.selected_limit = selected_limit
        self.active_key: bytes | None = None
        self.active_quads: tuple[DenseQuadPolicyV70, ...] = ()
        self.metrics: SelectedQuadMetricsV70 | None = None
        self.enabled = True
        self.proved_root_codes: tuple[int, ...] | None = None
        self.proved_root_lower_bound = 0

    def disable(self) -> None:
        self.enabled = False
        self.active_key = None
        self.active_quads = ()
        self.metrics = None
        self.proved_root_codes = None
        self.proved_root_lower_bound = 0

    def set_proved_root_lower_bound(
        self,
        piece_codes: Sequence[int],
        lower_bound: int,
    ) -> None:
        if lower_bound < 0:
            raise ValueError("proved root lower bound must be non-negative")
        self.proved_root_codes = tuple(piece_codes)
        self.proved_root_lower_bound = lower_bound

    def prepare(self, piece_codes: Sequence[int]) -> SelectedQuadMetricsV70:
        self.enabled = True
        key = bytes(piece_codes)
        if key != self.active_key:
            self.active_quads, self.metrics = self.bank.select(
                piece_codes,
                candidate_limit=self.candidate_limit,
                selected_limit=self.selected_limit,
            )
            self.active_key = key
        if self.metrics is None:
            raise AssertionError("quad selection was not prepared")
        return self.metrics

    def encode_effect(self, effect: CubieEffect) -> tuple[int, ...]:
        return self.base.encode_effect(effect)

    def advance_codes(
        self,
        piece_codes: Sequence[int],
        move_index: int,
    ) -> tuple[int, ...]:
        return self.base.advance_codes(piece_codes, move_index)

    def select_portfolio(
        self,
        piece_codes: tuple[int, ...],
        *,
        limit: int,
    ) -> tuple[DensePairPolicy, ...]:
        if self.enabled:
            self.prepare(piece_codes)
        return self.base.select_portfolio(piece_codes, limit=limit)

    def lower_bound(
        self,
        piece_codes: Sequence[int],
        portfolio: Sequence[DensePairPolicy],
    ) -> int:
        lower_bound = self.base.lower_bound(piece_codes, portfolio)
        for policy in self.active_quads:
            lower_bound = max(lower_bound, policy.distance(piece_codes))
        if (
            self.proved_root_codes is not None
            and piece_codes == self.proved_root_codes
        ):
            lower_bound = max(lower_bound, self.proved_root_lower_bound)
        return lower_bound

    def rank_moves(
        self,
        piece_codes: tuple[int, ...],
        portfolio: Sequence[DensePairPolicy],
        *,
        last_face: str = "",
    ) -> tuple[DenseMoveEvaluation, ...]:
        return self.base.rank_moves(
            piece_codes,
            portfolio,
            last_face=last_face,
        )


class DenseSelectedQuadSolverV70:
    def __init__(
        self,
        *,
        suffix_depth: int = 4,
        candidate_limit: int = 2,
        selected_limit: int = 2,
        quad_min_bound: int = 10,
        quad_pdb_depth: int = 6,
        max_cached_quad_tables: int = 64,
    ) -> None:
        if quad_min_bound < 1:
            raise ValueError("quad minimum bound must be positive")
        runtime = shared_completed_proof_lookup_solver_v68(suffix_depth)
        self.quad_min_bound = quad_min_bound
        self.bank = DenseQuadPolicyBankV70(
            runtime.repository,
            cutoff_depth=quad_pdb_depth,
            max_cached_tables=max_cached_quad_tables,
        )
        self.repository = _QuadAugmentedRepositoryV70(
            runtime.repository,
            self.bank,
            candidate_limit=candidate_limit,
            selected_limit=selected_limit,
        )
        self.core = ByteSuffixExactKeySolverV69(
            runtime=runtime,
            byte_suffix_index=shared_byte_suffix_index_v69(suffix_depth),
        )
        self.core.repository = self.repository  # type: ignore[assignment]

    def prepare(self, scramble: Sequence[str]) -> SelectedQuadMetricsV70:
        start = CubieEffect.from_transformation(from_sequence(tuple(scramble)))
        codes = self.repository.encode_effect(start)
        return self.repository.prepare(codes)

    def _empty_metrics(self) -> SelectedQuadMetricsV70:
        return SelectedQuadMetricsV70(
            candidate_limit=self.repository.candidate_limit,
            selected_limit=self.repository.selected_limit,
            candidates=(),
            candidate_distances=(),
            selected=(),
            selected_distances=(),
            selected_pair_gains=(),
            new_tables_built=0,
            table_cache_hits=0,
            preparation_seconds=0.0,
            table_build_seconds=0.0,
        )

    def solve(
        self,
        scramble: Sequence[str],
        *,
        bound: int,
        **kwargs: object,
    ) -> DenseSelectedQuadResultV70:
        search_kwargs = dict(kwargs)
        max_nodes = int(search_kwargs.pop("max_nodes", 2_000_000))
        if max_nodes < 1:
            raise ValueError("max_nodes must be positive")

        if bound < self.quad_min_bound:
            self.repository.disable()
            search = self.core.solve(  # type: ignore[arg-type]
                scramble,
                bound=bound,
                max_nodes=max_nodes,
                **search_kwargs,
            )
            return DenseSelectedQuadResultV70(
                search=search,
                quad_metrics=self._empty_metrics(),
            )

        self.repository.disable()
        precheck = self.core.solve(  # type: ignore[arg-type]
            scramble,
            bound=self.quad_min_bound - 1,
            max_nodes=max_nodes,
            **search_kwargs,
        )
        if precheck.solved or precheck.truncated:
            return DenseSelectedQuadResultV70(
                search=precheck,
                quad_metrics=self._empty_metrics(),
            )

        remaining_nodes = max_nodes - precheck.expanded
        if remaining_nodes < 1:
            return DenseSelectedQuadResultV70(
                search=precheck,
                quad_metrics=self._empty_metrics(),
            )

        metrics = self.prepare(scramble)
        start = CubieEffect.from_transformation(from_sequence(tuple(scramble)))
        start_codes = self.repository.encode_effect(start)
        self.repository.set_proved_root_lower_bound(
            start_codes,
            self.quad_min_bound,
        )
        search = self.core.solve(  # type: ignore[arg-type]
            scramble,
            bound=bound,
            max_nodes=remaining_nodes,
            **search_kwargs,
        )
        return DenseSelectedQuadResultV70(
            search=search,
            quad_metrics=metrics,
            precheck=precheck,
        )


__all__ = [
    "DenseQuadPolicyBankV70",
    "DenseQuadPolicyV70",
    "DenseSelectedQuadResultV70",
    "DenseSelectedQuadSolverV70",
    "Quad",
    "QuadCandidateFeatures",
    "SelectedQuadMetricsV70",
]
