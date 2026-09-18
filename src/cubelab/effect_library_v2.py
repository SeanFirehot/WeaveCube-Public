from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from math import gcd, lcm
from typing import Iterable

from .cubie_effect import CubieEffect
from .pieces import CORNER_ORDER, EDGE_ORDER
from .rotations import CUBE_ROTATIONS
from .transformations import PIECES, from_sequence

MOVES: tuple[str, ...] = tuple(face + suffix for face in "UDLRFB" for suffix in ("", "'", "2"))
PIECE_INDEX = {piece: index for index, piece in enumerate(PIECES)}


def _cycle_lengths(effect: CubieEffect, pieces: tuple[str, ...]) -> tuple[int, ...]:
    indexes = [PIECE_INDEX[p] for p in pieces]
    local = {global_index: i for i, global_index in enumerate(indexes)}
    visited: set[int] = set()
    lengths: list[int] = []
    for start in indexes:
        if start in visited:
            continue
        current = start
        length = 0
        while current not in visited:
            visited.add(current)
            length += 1
            current = effect.destinations[current]
        if length > 1:
            lengths.append(length)
    return tuple(sorted(lengths))


def _effect_order(effect: CubieEffect) -> int:
    contributions: list[int] = []
    for pieces, orientations, modulus in (
        (CORNER_ORDER, effect.corner_twists, 3),
        (EDGE_ORDER, effect.edge_flips, 2),
    ):
        indexes = [PIECE_INDEX[piece] for piece in pieces]
        orientation_index = {global_index: i for i, global_index in enumerate(indexes)}
        visited: set[int] = set()
        for start in indexes:
            if start in visited:
                continue
            current = start
            cycle_length = 0
            orientation_sum = 0
            while current not in visited:
                visited.add(current)
                cycle_length += 1
                orientation_sum += orientations[orientation_index[current]]
                current = effect.destinations[current]
            multiplier = modulus // gcd(modulus, orientation_sum % modulus)
            contributions.append(cycle_length * multiplier)
    return lcm(*contributions)


def _rotation_key(sequence: tuple[str, ...], rotated_move_effects: tuple[dict[str, CubieEffect], ...]) -> tuple:
    signatures = []
    for move_effects in rotated_move_effects:
        effect = CubieEffect.identity()
        for move in sequence:
            effect = move_effects[move].compose_after(effect)
        signatures.append(effect.signature)
    return min(signatures)


@dataclass(frozen=True, slots=True)
class EffectMetadata:
    moved_pieces: int
    preserved_pieces: int
    moved_corners: int
    moved_edges: int
    corner_cycle_lengths: tuple[int, ...]
    edge_cycle_lengths: tuple[int, ...]
    twisted_corners: int
    flipped_edges: int
    orientation_preserving: bool
    order: int


@dataclass(frozen=True, slots=True)
class EffectRecord:
    effect_id: int
    sequence: tuple[str, ...]
    effect: CubieEffect
    metadata: EffectMetadata
    inverse_effect_id: int | None = None
    rotation_class_id: int | None = None

    @property
    def move_count(self) -> int:
        return len(self.sequence)


@dataclass(frozen=True, slots=True)
class EffectLibraryV2:
    max_depth: int
    generated_sequences: int
    records: tuple[EffectRecord, ...]
    depth_stats: tuple[dict, ...]

    @property
    def unique_effects(self) -> int:
        return len(self.records)

    def by_signature(self) -> dict[tuple, EffectRecord]:
        return {record.effect.signature: record for record in self.records}

    def summary(self) -> dict:
        move_hist = Counter(record.move_count for record in self.records)
        moved_hist = Counter(record.metadata.moved_pieces for record in self.records)
        orientation_hist = Counter(
            "preserving" if record.metadata.orientation_preserving else "changing"
            for record in self.records
        )
        rotation_classes = len({record.rotation_class_id for record in self.records})
        return {
            "schema": "cubelab.effect-library-v2.phase1",
            "max_depth": self.max_depth,
            "generated_sequences": self.generated_sequences,
            "unique_effects": self.unique_effects,
            "rotation_classes": rotation_classes,
            "move_count_histogram": dict(sorted(move_hist.items())),
            "moved_piece_histogram": dict(sorted(moved_hist.items())),
            "orientation_histogram": dict(sorted(orientation_hist.items())),
            "depth_stats": list(self.depth_stats),
        }


def _metadata(effect: CubieEffect, *, include_order: bool = True) -> EffectMetadata:
    moved = [i for i, destination in enumerate(effect.destinations) if i != destination]
    moved_corners = sum(PIECES[i] in CORNER_ORDER for i in moved)
    moved_edges = len(moved) - moved_corners
    twisted = sum(value != 0 for value in effect.corner_twists)
    flipped = sum(value != 0 for value in effect.edge_flips)
    return EffectMetadata(
        moved_pieces=len(moved),
        preserved_pieces=len(PIECES) - len(moved),
        moved_corners=moved_corners,
        moved_edges=moved_edges,
        corner_cycle_lengths=_cycle_lengths(effect, CORNER_ORDER),
        edge_cycle_lengths=_cycle_lengths(effect, EDGE_ORDER),
        twisted_corners=twisted,
        flipped_edges=flipped,
        orientation_preserving=(twisted == 0 and flipped == 0),
        order=_effect_order(effect) if include_order else 0,
    )


def build_effect_library_v2(max_depth: int = 5, *, include_rotation_index: bool = True) -> EffectLibraryV2:
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")

    identity = CubieEffect.identity().signature
    move_effects = {move: CubieEffect.from_transformation(from_sequence((move,))) for move in MOVES}
    seen: dict[tuple, tuple[str, ...]] = {}
    frontier: list[tuple[tuple[str, ...], CubieEffect]] = [((), CubieEffect.identity())]
    generated = 0
    depth_stats: list[dict] = []

    for depth in range(1, max_depth + 1):
        next_frontier: list[tuple[tuple[str, ...], CubieEffect]] = []
        new_unique = 0
        for prefix, prefix_effect in frontier:
            previous_face = prefix[-1][0] if prefix else None
            for move in MOVES:
                if move[0] == previous_face:
                    continue
                sequence = prefix + (move,)
                generated += 1
                effect = move_effects[move].compose_after(prefix_effect)
                signature = effect.signature
                if signature == identity:
                    continue
                incumbent = seen.get(signature)
                if incumbent is not None:
                    if (len(sequence), sequence) < (len(incumbent), incumbent):
                        seen[signature] = sequence
                    continue
                seen[signature] = sequence
                next_frontier.append((sequence, effect))
                new_unique += 1
        depth_stats.append({
            "depth": depth,
            "generated_sequences": generated,
            "new_unique_effects": new_unique,
            "total_unique_effects": len(seen),
            "frontier": len(next_frontier),
        })
        frontier = next_frontier

    ordered = sorted(seen.items(), key=lambda item: (len(item[1]), item[1], item[0]))
    signature_to_id = {signature: index for index, (signature, _) in enumerate(ordered)}

    rotation_key_to_id: dict[tuple, int] = {}
    rotated_move_effects: tuple[dict[str, CubieEffect], ...] = ()
    if include_rotation_index:
        rotated_move_effects = tuple(
            {
                move: CubieEffect.from_transformation(from_sequence((rotation.map_move(move),)))
                for move in MOVES
            }
            for rotation in CUBE_ROTATIONS
        )
    records: list[EffectRecord] = []
    for effect_id, (signature, sequence) in enumerate(ordered):
        effect = CubieEffect(*signature)
        inverse_id = signature_to_id.get(effect.inverse().signature)
        rotation_class_id = None
        if include_rotation_index:
            canonical = _rotation_key(sequence, rotated_move_effects)
            rotation_class_id = rotation_key_to_id.setdefault(canonical, len(rotation_key_to_id))
        records.append(EffectRecord(
            effect_id=effect_id,
            sequence=sequence,
            effect=effect,
            metadata=_metadata(effect),
            inverse_effect_id=inverse_id,
            rotation_class_id=rotation_class_id,
        ))

    return EffectLibraryV2(max_depth, generated, tuple(records), tuple(depth_stats))


def write_effect_library_v2(library: EffectLibraryV2, output_dir: str | Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = library.summary()
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    records = []
    for record in library.records:
        records.append({
            "effect_id": record.effect_id,
            "sequence": list(record.sequence),
            "move_count": record.move_count,
            "signature": {
                "destinations": list(record.effect.destinations),
                "corner_twists": list(record.effect.corner_twists),
                "edge_flips": list(record.effect.edge_flips),
            },
            "metadata": asdict(record.metadata),
            "inverse_effect_id": record.inverse_effect_id,
            "rotation_class_id": record.rotation_class_id,
        })
    (output / "effects.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return summary


__all__ = [
    "MOVES", "EffectMetadata", "EffectRecord", "EffectLibraryV2",
    "build_effect_library_v2", "write_effect_library_v2",
]
