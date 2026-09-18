from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable, Sequence

from .cubie_effect import CubieEffect
from .dense_projection_search_v64 import DenseProjectionRepository
from .exact_mitm_oracle_v63 import ExactMitmOracleResult, ExactMitmOracleV63
from .piece_route_graph import FACE_MOVES, inverse_sequence
from .sequences import reduce_sequence
from .transformations import from_sequence


ATLAS_SCRAMBLE = ("B", "R'", "B2", "D'", "L'", "U'", "F")
ATLAS_SHORTEST_WITNESS = ("F'", "U", "L", "D", "B2", "R", "B'")

_MOVE_INDEX = {move: index for index, move in enumerate(FACE_MOVES)}
_INVERSE_INDEX = {
    index: _MOVE_INDEX[inverse_sequence((move,))[0]]
    for index, move in enumerate(FACE_MOVES)
}


PathBucket = int | list[int]


@dataclass(slots=True)
class AllWitnessEffectTable:
    """Every reduced exact-length word, grouped by complete cubie effect.

    Paths are base-18 integers.  Keeping the first path directly and allocating
    a list only on an effect collision makes the depth-five table materially
    smaller than a ``dict[bytes, list[tuple[str, ...]]]`` representation.
    """

    depth: int
    buckets: dict[bytes, PathBucket]
    word_count: int
    build_seconds: float

    @property
    def state_count(self) -> int:
        return len(self.buckets)


@dataclass(frozen=True, slots=True)
class LengthEnumeration:
    length: int
    left_depth: int
    right_depth: int
    raw_mitm_joins: int
    reduced_solutions: tuple[tuple[str, ...], ...]
    normalized_join_cores: tuple[tuple[str, ...], ...]
    unique_state_paths: int
    prefix_words: int
    suffix_words: int
    suffix_states: int
    join_seconds: float

    @property
    def reduced_solution_count(self) -> int:
        return len(self.reduced_solutions)

    @property
    def normalized_join_core_count(self) -> int:
        return len(self.normalized_join_cores)


@dataclass(frozen=True, slots=True)
class SolutionEnumerationBatch:
    scramble: tuple[str, ...]
    certification: ExactMitmOracleResult
    table_builds: tuple[tuple[int, int, int, float], ...]
    lengths: tuple[LengthEnumeration, ...]
    total_seconds: float


def _decode_path(code: int, depth: int) -> tuple[int, ...]:
    values = [0] * depth
    for index in range(depth - 1, -1, -1):
        code, values[index] = divmod(code, len(FACE_MOVES))
    return tuple(values)


def _path_moves(code: int, depth: int) -> tuple[str, ...]:
    return tuple(FACE_MOVES[index] for index in _decode_path(code, depth))


def _inverse_path_moves(code: int, depth: int) -> tuple[str, ...]:
    indices = _decode_path(code, depth)
    return tuple(FACE_MOVES[_INVERSE_INDEX[index]] for index in reversed(indices))


def _bucket_values(bucket: PathBucket) -> Iterable[int]:
    return (bucket,) if isinstance(bucket, int) else bucket


def _append_bucket(
    buckets: dict[bytes, PathBucket],
    key: bytes,
    path_code: int,
) -> None:
    current = buckets.get(key)
    if current is None:
        buckets[key] = path_code
    elif isinstance(current, int):
        buckets[key] = [current, path_code]
    else:
        current.append(path_code)


class AllWitnessSolutionEnumeratorV74:
    """Complete reduced-word MITM enumerator for one shallow cube state.

    The solved-side table stores *all* reduced words rather than one BFS parent
    per state.  A solved-side word ``r`` joins a forward prefix at the same
    complete piece-code key; the required suffix is exactly ``inverse(r)``.
    This is a bijection, so no exact word is lost through state deduplication.
    """

    def __init__(self, repository: DenseProjectionRepository | None = None) -> None:
        self.repository = repository or DenseProjectionRepository()
        self.identity_codes = self.repository.encode_effect(CubieEffect.identity())
        self._tables: dict[int, AllWitnessEffectTable] = {}

    def build_effect_table(self, depth: int) -> AllWitnessEffectTable:
        if depth < 0:
            raise ValueError("depth must be non-negative")
        cached = self._tables.get(depth)
        if cached is not None:
            return cached

        started = time.perf_counter()
        buckets: dict[bytes, PathBucket] = {}
        word_count = 0

        def visit(
            piece_codes: tuple[int, ...],
            path_depth: int,
            last_face: str,
            path_code: int,
        ) -> None:
            nonlocal word_count
            if path_depth == depth:
                _append_bucket(buckets, bytes(piece_codes), path_code)
                word_count += 1
                return
            for move_index, move in enumerate(FACE_MOVES):
                if last_face and move[0] == last_face:
                    continue
                visit(
                    self.repository.advance_codes(piece_codes, move_index),
                    path_depth + 1,
                    move[0],
                    path_code * len(FACE_MOVES) + move_index,
                )

        visit(self.identity_codes, 0, "", 0)
        result = AllWitnessEffectTable(
            depth=depth,
            buckets=buckets,
            word_count=word_count,
            build_seconds=time.perf_counter() - started,
        )
        self._tables[depth] = result
        return result

    def enumerate_length(
        self,
        scramble: Sequence[str],
        length: int,
    ) -> LengthEnumeration:
        if length < 0:
            raise ValueError("length must be non-negative")
        left_depth = length // 2
        right_depth = length - left_depth
        suffix_table = self.build_effect_table(right_depth)
        start_effect = CubieEffect.from_transformation(from_sequence(tuple(scramble)))
        start_codes = self.repository.encode_effect(start_effect)
        started = time.perf_counter()
        prefix_words = 0
        raw_joins = 0
        reduced: set[tuple[str, ...]] = set()
        normalized_cores: set[tuple[str, ...]] = set()

        def visit(
            piece_codes: tuple[int, ...],
            path_depth: int,
            last_face: str,
            path_code: int,
        ) -> None:
            nonlocal prefix_words, raw_joins
            if path_depth == left_depth:
                prefix_words += 1
                bucket = suffix_table.buckets.get(bytes(piece_codes))
                if bucket is None:
                    return
                prefix = _path_moves(path_code, left_depth)
                for backward_code in _bucket_values(bucket):
                    raw_joins += 1
                    suffix = _inverse_path_moves(backward_code, right_depth)
                    candidate = prefix + suffix
                    normalized = reduce_sequence(candidate)
                    normalized_cores.add(normalized)
                    if len(normalized) == length:
                        reduced.add(normalized)
                return
            for move_index, move in enumerate(FACE_MOVES):
                if last_face and move[0] == last_face:
                    continue
                visit(
                    self.repository.advance_codes(piece_codes, move_index),
                    path_depth + 1,
                    move[0],
                    path_code * len(FACE_MOVES) + move_index,
                )

        visit(start_codes, 0, "", 0)
        ordered = tuple(
            sorted(
                reduced,
                key=lambda word: tuple(_MOVE_INDEX[move] for move in word),
            )
        )
        cores = tuple(
            sorted(
                normalized_cores,
                key=lambda word: (
                    len(word),
                    tuple(_MOVE_INDEX[move] for move in word),
                ),
            )
        )
        state_paths = {self.state_path_key(scramble, word) for word in ordered}
        return LengthEnumeration(
            length=length,
            left_depth=left_depth,
            right_depth=right_depth,
            raw_mitm_joins=raw_joins,
            reduced_solutions=ordered,
            normalized_join_cores=cores,
            unique_state_paths=len(state_paths),
            prefix_words=prefix_words,
            suffix_words=suffix_table.word_count,
            suffix_states=suffix_table.state_count,
            join_seconds=time.perf_counter() - started,
        )

    def state_path_key(
        self,
        scramble: Sequence[str],
        solution: Sequence[str],
    ) -> tuple[bytes, ...]:
        effect = CubieEffect.from_transformation(from_sequence(tuple(scramble)))
        codes = self.repository.encode_effect(effect)
        path = [bytes(codes)]
        for move in solution:
            codes = self.repository.advance_codes(codes, _MOVE_INDEX[move])
            path.append(bytes(codes))
        return tuple(path)

    def verify_solution(
        self,
        scramble: Sequence[str],
        solution: Sequence[str],
    ) -> bool:
        effect = CubieEffect.from_transformation(
            from_sequence(tuple(scramble) + tuple(solution))
        )
        return effect == CubieEffect.identity()

    def enumerate(
        self,
        scramble: Sequence[str],
        *,
        lengths: Sequence[int],
        certification_bound: int = 7,
    ) -> SolutionEnumerationBatch:
        started = time.perf_counter()
        requested = tuple(sorted(set(lengths)))
        if not requested:
            raise ValueError("at least one length is required")
        oracle = ExactMitmOracleV63(backward_depth=min(4, certification_bound))
        certification = oracle.solve(tuple(scramble), bound=certification_bound)
        results = tuple(self.enumerate_length(scramble, length) for length in requested)
        for result in results:
            if not all(self.verify_solution(scramble, word) for word in result.reduced_solutions):
                raise AssertionError(f"invalid solution emitted at length {result.length}")
        table_builds = tuple(
            (
                depth,
                table.word_count,
                table.state_count,
                table.build_seconds,
            )
            for depth, table in sorted(self._tables.items())
        )
        return SolutionEnumerationBatch(
            scramble=tuple(scramble),
            certification=certification,
            table_builds=table_builds,
            lengths=results,
            total_seconds=time.perf_counter() - started,
        )


__all__ = [
    "ATLAS_SCRAMBLE",
    "ATLAS_SHORTEST_WITNESS",
    "AllWitnessEffectTable",
    "AllWitnessSolutionEnumeratorV74",
    "LengthEnumeration",
    "SolutionEnumerationBatch",
]
