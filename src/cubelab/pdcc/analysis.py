"""Structural pose-difference, bundle, boundary, and cluster analysis."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Iterable

from .group import CUBE_ROTATIONS
from .model import NATIVE_ADJACENCY, PIECE_INDEX, PIECE_NAMES
from .state import PDCCState


def relative_difference(state: PDCCState, left_piece: str, right_piece: str) -> int:
    """Return ``W_left^-1 W_right`` as a rotation-group element ID."""

    return CUBE_ROTATIONS.compose(
        CUBE_ROTATIONS.inverse(state.pose_id(left_piece)), state.pose_id(right_piece)
    )


def rigid_bundles(state: PDCCState) -> tuple[tuple[str, ...], ...]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for piece in PIECE_NAMES:
        grouped[state.pose_id(piece)].append(piece)
    bundles = [
        tuple(sorted(members, key=PIECE_INDEX.__getitem__))
        for _, members in sorted(grouped.items())
    ]
    return tuple(sorted(bundles, key=lambda group: (-len(group), group)))


def native_boundary(state: PDCCState) -> tuple[tuple[str, str, int], ...]:
    boundary: list[tuple[str, str, int]] = []
    for left, right in NATIVE_ADJACENCY:
        difference = relative_difference(state, left, right)
        if difference != CUBE_ROTATIONS.identity:
            boundary.append((left, right, difference))
    return tuple(boundary)


def current_adjacency(state: PDCCState) -> tuple[tuple[str, str], ...]:
    occupant_by_slot = {state.slot(piece): piece for piece in PIECE_NAMES}
    links = {
        tuple(
            sorted(
                (occupant_by_slot[left_slot], occupant_by_slot[right_slot]),
                key=PIECE_INDEX.__getitem__,
            )
        )
        for left_slot, right_slot in NATIVE_ADJACENCY
    }
    return tuple(sorted(links, key=lambda pair: (PIECE_INDEX[pair[0]], PIECE_INDEX[pair[1]])))


def _components(nodes: set[str], links: Iterable[tuple[str, str]]) -> tuple[tuple[str, ...], ...]:
    neighbors: dict[str, set[str]] = {node: set() for node in nodes}
    for left, right in links:
        if left in nodes and right in nodes:
            neighbors[left].add(right)
            neighbors[right].add(left)

    components: list[tuple[str, ...]] = []
    remaining = set(nodes)
    while remaining:
        start = min(remaining, key=PIECE_INDEX.__getitem__)
        queue = deque([start])
        found: list[str] = []
        remaining.remove(start)
        while queue:
            node = queue.popleft()
            found.append(node)
            for neighbor in sorted(neighbors[node], key=PIECE_INDEX.__getitem__):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        components.append(tuple(found))
    return tuple(sorted(components, key=lambda group: (-len(group), group)))


def twist_clusters(state: PDCCState) -> tuple[tuple[str, ...], ...]:
    twisted = {piece for piece in PIECE_NAMES if state.orientation(piece) != 0}
    if not twisted:
        return ()
    return _components(twisted, current_adjacency(state))


def task_bundles(before: PDCCState, after: PDCCState) -> tuple[tuple[int, tuple[str, ...]], ...]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for piece in PIECE_NAMES:
        effect = CUBE_ROTATIONS.compose(
            after.pose_id(piece), CUBE_ROTATIONS.inverse(before.pose_id(piece))
        )
        grouped[effect].append(piece)
    result = [
        (effect, tuple(sorted(members, key=PIECE_INDEX.__getitem__)))
        for effect, members in grouped.items()
    ]
    return tuple(sorted(result, key=lambda item: (-len(item[1]), item[0])))


@dataclass(frozen=True, slots=True)
class StructureSignature:
    rigid_bundle_sizes: tuple[int, ...]
    rigid_bundle_count: int
    native_boundary_count: int
    boundary_label_histogram: tuple[tuple[int, int], ...]
    twisted_corner_count: int
    flipped_edge_count: int
    twist_cluster_sizes: tuple[int, ...]
    moved_corner_count: int
    moved_edge_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def structure_signature(state: PDCCState) -> StructureSignature:
    bundles = rigid_bundles(state)
    boundary = native_boundary(state)
    histogram: dict[int, int] = defaultdict(int)
    for _, _, difference in boundary:
        histogram[difference] += 1
    twist = twist_clusters(state)
    return StructureSignature(
        rigid_bundle_sizes=tuple(sorted((len(bundle) for bundle in bundles), reverse=True)),
        rigid_bundle_count=len(bundles),
        native_boundary_count=len(boundary),
        boundary_label_histogram=tuple(sorted(histogram.items())),
        twisted_corner_count=sum(
            state.orientation(piece) != 0 for piece in PIECE_NAMES[:8]
        ),
        flipped_edge_count=sum(
            state.orientation(piece) != 0 for piece in PIECE_NAMES[8:]
        ),
        twist_cluster_sizes=tuple(sorted((len(cluster) for cluster in twist), reverse=True)),
        moved_corner_count=sum(state.slot(piece) != piece for piece in PIECE_NAMES[:8]),
        moved_edge_count=sum(state.slot(piece) != piece for piece in PIECE_NAMES[8:]),
    )
