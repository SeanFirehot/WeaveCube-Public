from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations, product
from typing import Iterable, Literal, Sequence

from .cubie_effect import CubieEffect
from .dense_projection_search_v64 import DensePairPolicy, DenseProjectionRepository
from .dense_selected_quad_projection_v70 import DenseQuadPolicyV70
from .ordered_piece_route_solver import OrderedPieceRouteSolver
from .piece_route_graph import FACE_MOVES, PieceState
from .pieces import CORNER_ORDER, EDGE_ORDER
from .rotations import CUBE_ROTATIONS
from .sequences import reduce_sequence
from .transformations import PIECES, from_sequence


EXACT_TWELVE_SCRAMBLE = tuple("D' F D' F U B R F' D R B' U".split())
EXACT_TWELVE_WITNESS = tuple("U' B R' D' F R' B' U' F' D F' D".split())
TARGET_PIECES = ("UFL", "DL")
GAP_FACES = ("B", "R")

_PIECE_INDEX = {piece: index for index, piece in enumerate(PIECES)}
_CORNER_INDEX = {piece: index for index, piece in enumerate(CORNER_ORDER)}
_EDGE_INDEX = {piece: index for index, piece in enumerate(EDGE_ORDER)}
_FACES = ("U", "D", "F", "B", "L", "R")
_SLICE_EDGES = frozenset(("FR", "BR", "BL", "FL"))


PreservationMode = Literal["action_projection", "terminal_solved"]


@dataclass(frozen=True, slots=True)
class GrammarSkeleton:
    name: str
    active_blocks: tuple[tuple[str, ...], ...]
    family: str
    source_asset: str
    preservation_mode: PreservationMode

    @property
    def sequence(self) -> tuple[str, ...]:
        return tuple(move for block in self.active_blocks for move in block)


@dataclass(frozen=True, slots=True)
class GenerationStats:
    skeleton: str
    raw_gap_assignments: int
    preservation_passed: int
    length_pruned: int
    effect_duplicates: int
    retained_unique: int


@dataclass(frozen=True, slots=True)
class SpatialSignature:
    raw_co: tuple[int, ...]
    raw_eo: tuple[int, ...]
    combined_co_eo: tuple[int, ...]
    corner_face_defects: tuple[int, ...]
    edge_face_defects: tuple[int, ...]
    orientation_face_defects: tuple[int, ...]
    corner_same_face_pairs: int
    corner_disjoint_pairs: int
    edge_same_face_pairs: int
    edge_disjoint_pairs: int
    minimum_corner_defect_faces: int
    minimum_edge_defect_faces: int
    minimum_orientation_defect_faces: int
    maximum_single_move_defect_reduction: int
    corner_cycle_type: tuple[int, ...]
    edge_cycle_type: tuple[int, ...]
    misplaced_piece_support: int
    slice_membership_defect: int
    pair_lower_bound: int
    pair_top_twelve_sum: int
    fixed_quad_lower_bound: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GatewayCandidate:
    skeleton: str
    family: str
    route: tuple[str, ...]
    residual: CubieEffect
    signature: SpatialSignature

    @property
    def rank_key(self) -> tuple:
        """Cheap final-gateway ranking; never used on intermediate moves."""
        return (
            self.signature.fixed_quad_lower_bound,
            self.signature.pair_lower_bound,
            self.signature.pair_top_twelve_sum,
            sum(self.signature.raw_eo),
            sum(value != 0 for value in self.signature.raw_co),
            self.signature.misplaced_piece_support,
            len(self.route),
            self.route,
            self.skeleton,
        )


@dataclass(frozen=True, slots=True)
class CompositionStats:
    top_gateway_count: int
    ordered_pairs_considered: int
    length_pruned: int
    single_effect_duplicates: int
    composed_effect_duplicates: int
    retained_unique: int


def generic_skeletons() -> tuple[GrammarSkeleton, ...]:
    """Three short corner-clean primitives from the existing v4 registry.

    These were selected without exact-10 calls.  Each lowers the exact-12
    state's cheap pair profile or represents a structurally distinct two-block
    setup/action/undo form.
    """
    return (
        GrammarSkeleton(
            "MW039",
            (("F'", "L'", "B'"), ("F", "U", "B")),
            "generic_corner_clean",
            "data/generated/wide_sexy_gateway_v4_hybrid.json:MW039",
            "action_projection",
        ),
        GrammarSkeleton(
            "MW059",
            (("L", "F'", "R"), ("L'", "U", "R'")),
            "generic_corner_clean",
            "data/generated/wide_sexy_gateway_v4_hybrid.json:MW059",
            "action_projection",
        ),
        GrammarSkeleton(
            "MW026",
            (("R", "U", "R'"), ("L", "F'", "L'")),
            "generic_corner_clean",
            "data/generated/wide_sexy_gateway_v4_hybrid.json:MW026",
            "action_projection",
        ),
    )


def state_conditioned_skeletons() -> tuple[GrammarSkeleton, ...]:
    """Top three bounded v62-style UFL/DL preservation seeds for the case."""
    return (
        GrammarSkeleton(
            "route-q3",
            (("L2", "R'", "F"), ("D'", "B", "R", "D2")),
            "state_conditioned_pair_route",
            "preservation_route_seed_v62:replanned-UFL-DL",
            "terminal_solved",
        ),
        GrammarSkeleton(
            "route-q4a",
            (("B", "F", "L2"), ("D'", "B", "R", "D2")),
            "state_conditioned_pair_route",
            "preservation_route_seed_v62:replanned-UFL-DL",
            "terminal_solved",
        ),
        GrammarSkeleton(
            "route-q4b",
            (("L", "F'", "L'", "D'"), ("U'", "F'")),
            "state_conditioned_pair_route",
            "preservation_route_seed_v62:replanned-DL-UFL",
            "terminal_solved",
        ),
    )


def gap_words(
    faces: Sequence[str] = GAP_FACES,
    *,
    maximum_depth: int = 2,
) -> tuple[tuple[str, ...], ...]:
    """Reduced face-turn gap words of depth 0..2 by default."""
    if maximum_depth < 0:
        raise ValueError("maximum gap depth must be non-negative")
    if not faces or any(face not in _FACES for face in faces):
        raise ValueError("gap faces must be non-empty cube face names")
    moves = tuple(
        f"{face}{suffix}"
        for face in faces
        for suffix in ("", "'", "2")
    )
    words: list[tuple[str, ...]] = [()]
    frontier = [()]
    for _ in range(maximum_depth):
        next_frontier = []
        for prefix in frontier:
            for move in moves:
                if prefix and prefix[-1][0] == move[0]:
                    continue
                word = prefix + (move,)
                words.append(word)
                next_frontier.append(word)
        frontier = next_frontier
    return tuple(words)


@lru_cache(maxsize=None)
def _effect(sequence: tuple[str, ...]) -> CubieEffect:
    return CubieEffect.from_transformation(from_sequence(sequence))


def _position_orientations(
    effect: CubieEffect,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    corners = [0] * len(CORNER_ORDER)
    edges = [0] * len(EDGE_ORDER)
    for source_index, source in enumerate(CORNER_ORDER):
        destination = PIECES[effect.destinations[_PIECE_INDEX[source]]]
        twist = effect.corner_twists[source_index]
        corners[_CORNER_INDEX[destination]] = -1 if twist == 2 else twist
    for source_index, source in enumerate(EDGE_ORDER):
        destination = PIECES[effect.destinations[_PIECE_INDEX[source]]]
        edges[_EDGE_INDEX[destination]] = effect.edge_flips[source_index]
    return tuple(corners), tuple(edges)


def _face_counts(positions: Sequence[str]) -> tuple[int, ...]:
    return tuple(sum(face in position for position in positions) for face in _FACES)


def _pair_relations(positions: Sequence[str]) -> tuple[int, int]:
    same_face = sum(
        bool(set(left) & set(right))
        for left, right in combinations(positions, 2)
    )
    total = len(positions) * (len(positions) - 1) // 2
    return same_face, total - same_face


def _minimum_face_cover(positions: Sequence[str]) -> int:
    if not positions:
        return 0
    for size in range(1, len(_FACES) + 1):
        if any(
            all(any(face in position for face in chosen) for position in positions)
            for chosen in combinations(_FACES, size)
        ):
            return size
    raise AssertionError("six faces must cover every cubie position")


def _cycle_type(effect: CubieEffect, order: Sequence[str]) -> tuple[int, ...]:
    index = {piece: i for i, piece in enumerate(order)}
    permutation = tuple(
        index[PIECES[effect.destinations[_PIECE_INDEX[piece]]]]
        for piece in order
    )
    seen: set[int] = set()
    lengths: list[int] = []
    for start in range(len(order)):
        if start in seen:
            continue
        cursor = start
        length = 0
        while cursor not in seen:
            seen.add(cursor)
            length += 1
            cursor = permutation[cursor]
        lengths.append(length)
    return tuple(sorted(lengths, reverse=True))


def _orientation_defect_count(effect: CubieEffect) -> int:
    corners, edges = _position_orientations(effect)
    return sum(value != 0 for value in corners) + sum(edges)


def spatial_signature(
    effect: CubieEffect,
    *,
    repository: DenseProjectionRepository,
    pair_policies: Sequence[DensePairPolicy],
    fixed_quad_policies: Sequence[DenseQuadPolicyV70] = (),
) -> SpatialSignature:
    co, eo = _position_orientations(effect)
    corner_positions = tuple(
        position for position, value in zip(CORNER_ORDER, co) if value
    )
    edge_positions = tuple(
        position for position, value in zip(EDGE_ORDER, eo) if value
    )
    all_positions = corner_positions + edge_positions
    corner_same, corner_disjoint = _pair_relations(corner_positions)
    edge_same, edge_disjoint = _pair_relations(edge_positions)
    baseline_defects = len(all_positions)
    maximum_reduction = max(
        baseline_defects
        - _orientation_defect_count(_effect((move,)).compose_after(effect))
        for move in FACE_MOVES
    )
    codes = repository.encode_effect(effect)
    pair_distances = sorted(
        (policy.distance(codes) for policy in pair_policies),
        reverse=True,
    )
    slice_defect = sum(
        ((source in _SLICE_EDGES) != (
            PIECES[effect.destinations[_PIECE_INDEX[source]]] in _SLICE_EDGES
        ))
        for source in EDGE_ORDER
    )
    return SpatialSignature(
        raw_co=co,
        raw_eo=eo,
        combined_co_eo=co + eo,
        corner_face_defects=_face_counts(corner_positions),
        edge_face_defects=_face_counts(edge_positions),
        orientation_face_defects=_face_counts(all_positions),
        corner_same_face_pairs=corner_same,
        corner_disjoint_pairs=corner_disjoint,
        edge_same_face_pairs=edge_same,
        edge_disjoint_pairs=edge_disjoint,
        minimum_corner_defect_faces=_minimum_face_cover(corner_positions),
        minimum_edge_defect_faces=_minimum_face_cover(edge_positions),
        minimum_orientation_defect_faces=_minimum_face_cover(all_positions),
        maximum_single_move_defect_reduction=maximum_reduction,
        corner_cycle_type=_cycle_type(effect, CORNER_ORDER),
        edge_cycle_type=_cycle_type(effect, EDGE_ORDER),
        misplaced_piece_support=sum(code != 0 for code in codes),
        slice_membership_defect=slice_defect,
        pair_lower_bound=max(pair_distances, default=0),
        pair_top_twelve_sum=sum(pair_distances[:12]),
        fixed_quad_lower_bound=max(
            (policy.distance(codes) for policy in fixed_quad_policies),
            default=0,
        ),
    )


def canonical_orientation_signature(
    sequence: Sequence[str],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Lexicographically canonical spatial CO/EO under 24 cube rotations."""
    return min(
        _position_orientations(_effect(rotation.map_sequence(tuple(sequence))))
        for rotation in CUBE_ROTATIONS
    )


def _interleave(
    skeleton: GrammarSkeleton,
    gaps: Sequence[Sequence[str]],
) -> tuple[str, ...]:
    if len(gaps) != len(skeleton.active_blocks) + 1:
        raise ValueError("a skeleton needs one more gap than active blocks")
    sequence: list[str] = []
    for gap, block in zip(gaps, skeleton.active_blocks):
        sequence.extend(gap)
        sequence.extend(block)
    sequence.extend(gaps[-1])
    return reduce_sequence(sequence)


def generate_candidates(
    scramble: Sequence[str],
    skeletons: Sequence[GrammarSkeleton],
    *,
    repository: DenseProjectionRepository,
    pair_policies: Sequence[DensePairPolicy],
    fixed_quad_policies: Sequence[DenseQuadPolicyV70] = (),
    target_pieces: Sequence[str] = TARGET_PIECES,
    gap_faces: Sequence[str] = GAP_FACES,
    maximum_gap_depth: int = 2,
    maximum_route_length: int = 11,
) -> tuple[tuple[GatewayCandidate, ...], tuple[GenerationStats, ...]]:
    """Generate, normalize, preserve-filter, and effect-deduplicate gateways."""
    if maximum_route_length < 0:
        raise ValueError("maximum route length must be non-negative")
    start = _effect(tuple(scramble))
    route_solver = OrderedPieceRouteSolver()
    gap_inventory = gap_words(gap_faces, maximum_depth=maximum_gap_depth)
    target_indices = tuple(_PIECE_INDEX[piece] for piece in target_pieces)
    global_best: dict[tuple, tuple[GrammarSkeleton, tuple[str, ...], CubieEffect]] = {}
    stats: list[GenerationStats] = []

    for skeleton in skeletons:
        if len(skeleton.active_blocks) != 2:
            raise ValueError("v72 POC supports exactly two active blocks")
        baseline_action = _effect(skeleton.sequence)
        baseline_codes = repository.encode_effect(baseline_action)
        projected_target = tuple(baseline_codes[index] for index in target_indices)
        raw = preserved = length_pruned = duplicates = 0
        local_best: dict[tuple, tuple[tuple[str, ...], CubieEffect]] = {}

        for gaps in product(gap_inventory, repeat=3):
            raw += 1
            route = _interleave(skeleton, gaps)
            action = _effect(route)
            residual = action.compose_after(start)
            if skeleton.preservation_mode == "action_projection":
                action_codes = repository.encode_effect(action)
                valid = tuple(action_codes[index] for index in target_indices) == projected_target
            else:
                valid = all(
                    route_solver.piece_state(residual, piece) == PieceState(piece, 0)
                    for piece in target_pieces
                )
            if not valid:
                continue
            preserved += 1
            if len(route) > maximum_route_length:
                length_pruned += 1
                continue
            old = local_best.get(residual.signature)
            if old is not None:
                duplicates += 1
                if (len(route), route) >= (len(old[0]), old[0]):
                    continue
            local_best[residual.signature] = (route, residual)

        for signature, (route, residual) in local_best.items():
            old = global_best.get(signature)
            if old is None or (len(route), route, skeleton.name) < (
                len(old[1]), old[1], old[0].name
            ):
                global_best[signature] = (skeleton, route, residual)
        stats.append(GenerationStats(
            skeleton=skeleton.name,
            raw_gap_assignments=raw,
            preservation_passed=preserved,
            length_pruned=length_pruned,
            effect_duplicates=duplicates,
            retained_unique=len(local_best),
        ))

    candidates = tuple(
        sorted(
            (
                GatewayCandidate(
                    skeleton=skeleton.name,
                    family=skeleton.family,
                    route=route,
                    residual=residual,
                    signature=spatial_signature(
                        residual,
                        repository=repository,
                        pair_policies=pair_policies,
                        fixed_quad_policies=fixed_quad_policies,
                    ),
                )
                for skeleton, route, residual in global_best.values()
            ),
            key=lambda candidate: candidate.rank_key,
        )
    )
    return candidates, tuple(stats)


def compose_top_candidates(
    scramble: Sequence[str],
    candidate_families: Sequence[Sequence[GatewayCandidate]],
    *,
    repository: DenseProjectionRepository,
    pair_policies: Sequence[DensePairPolicy],
    fixed_quad_policies: Sequence[DenseQuadPolicyV70] = (),
    top_per_family: int = 80,
    maximum_route_length: int = 11,
) -> tuple[tuple[GatewayCandidate, ...], CompositionStats]:
    """Try only pairwise composition after single-skeleton deformation fails."""
    if top_per_family < 1:
        raise ValueError("top-per-family must be positive")
    start = _effect(tuple(scramble))
    all_singles = tuple(
        candidate for family in candidate_families for candidate in family
    )
    top = tuple(
        candidate
        for family in candidate_families
        for candidate in tuple(family)[:top_per_family]
    )
    single_effects = {candidate.residual.signature for candidate in all_singles}
    retained: dict[tuple, tuple[GatewayCandidate, GatewayCandidate, tuple[str, ...], CubieEffect]] = {}
    considered = length_pruned = single_duplicates = composed_duplicates = 0

    for left in top:
        for right in top:
            if left.route == right.route:
                continue
            considered += 1
            route = reduce_sequence(left.route + right.route)
            if len(route) > maximum_route_length:
                length_pruned += 1
                continue
            residual = _effect(route).compose_after(start)
            if residual.signature in single_effects:
                single_duplicates += 1
                continue
            old = retained.get(residual.signature)
            if old is not None:
                composed_duplicates += 1
                if (len(route), route, left.skeleton, right.skeleton) >= (
                    len(old[2]), old[2], old[0].skeleton, old[1].skeleton
                ):
                    continue
            retained[residual.signature] = (left, right, route, residual)

    candidates = tuple(sorted(
        (
            GatewayCandidate(
                skeleton=f"{left.skeleton}+{right.skeleton}",
                family="two_gateway_composition",
                route=route,
                residual=residual,
                signature=spatial_signature(
                    residual,
                    repository=repository,
                    pair_policies=pair_policies,
                    fixed_quad_policies=fixed_quad_policies,
                ),
            )
            for left, right, route, residual in retained.values()
        ),
        key=lambda candidate: candidate.rank_key,
    ))
    return candidates, CompositionStats(
        top_gateway_count=len(top),
        ordered_pairs_considered=considered,
        length_pruned=length_pruned,
        single_effect_duplicates=single_duplicates,
        composed_effect_duplicates=composed_duplicates,
        retained_unique=len(candidates),
    )


def boundary_compress(
    gateway: Iterable[str],
    residual_solution: Iterable[str],
) -> tuple[str, ...]:
    return reduce_sequence(tuple(gateway) + tuple(residual_solution))


def full_state_solved(
    scramble: Sequence[str],
    solution: Sequence[str],
) -> bool:
    return _effect(tuple(scramble) + tuple(solution)) == CubieEffect.identity()


__all__ = [
    "EXACT_TWELVE_SCRAMBLE",
    "EXACT_TWELVE_WITNESS",
    "GAP_FACES",
    "TARGET_PIECES",
    "GatewayCandidate",
    "GenerationStats",
    "GrammarSkeleton",
    "CompositionStats",
    "SpatialSignature",
    "boundary_compress",
    "canonical_orientation_signature",
    "compose_top_candidates",
    "full_state_solved",
    "gap_words",
    "generate_candidates",
    "generic_skeletons",
    "spatial_signature",
    "state_conditioned_skeletons",
]
