from __future__ import annotations

"""Exact compatibility surface for a missing historical short-census module.

The September 2026 public source snapshot references
``cubelab.domino_reduction_short_census`` from older v100/v101 research code,
but the original module was not present in the frozen source tree and was not
recoverable from repository history.

This file is therefore *not* presented as the historical census implementation.
It restores only APIs whose semantics can be derived exactly from the canonical
``CubieEffect`` / ``transformations`` implementation. Artifact-loading helpers
remain fail-closed when the historical sealed census files are unavailable.

Packed effect layout (40 bytes):
    [20 source-indexed destination ids][20 source-indexed orientation deltas]
with corner orientation deltas modulo 3 and edge deltas modulo 2.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator, Sequence

from .cubie_effect import CubieEffect
from .transformations import from_sequence


COMPATIBILITY_RECONSTRUCTION = True
MOVE_ORDER: tuple[str, ...] = (
    "U", "U'", "U2", "D", "D'", "D2",
    "L", "L'", "L2", "R", "R'", "R2",
    "F", "F'", "F2", "B", "B'", "B2",
)


def _pack(effect: CubieEffect) -> bytes:
    return bytes(
        tuple(int(value) for value in effect.destinations)
        + tuple(int(value) for value in effect.corner_twists)
        + tuple(int(value) for value in effect.edge_flips)
    )


def _unpack(key: bytes | bytearray | memoryview) -> CubieEffect:
    raw = bytes(key)
    if len(raw) != 40:
        raise ValueError(f"packed cubie effect must be 40 bytes, got {len(raw)}")
    destinations = tuple(raw[:20])
    if sorted(destinations) != list(range(20)):
        raise ValueError("packed destination bytes are not a permutation of 0..19")
    return CubieEffect(
        destinations,
        tuple(value % 3 for value in raw[20:28]),
        tuple(value % 2 for value in raw[28:40]),
    )


IDENTITY_KEY: bytes = _pack(CubieEffect.identity())


def effect_from_word(word: Sequence[str]) -> bytes:
    """Return the exact packed full-cube effect of ``word``."""
    transformation = from_sequence(tuple(map(str, word)))
    return _pack(CubieEffect.from_transformation(transformation))


def compose_packed(after: bytes, before: bytes) -> bytes:
    """Return ``after ∘ before`` (apply ``before`` first), in packed form."""
    return _pack(_unpack(after).compose_after(_unpack(before)))


def inverse_packed(effect: bytes) -> bytes:
    """Return the exact inverse of a packed effect."""
    return _pack(_unpack(effect).inverse())


def state_codes(effect: bytes) -> tuple[int, ...]:
    """Return 20 ``destination + 20*orientation`` source-state codes."""
    raw = bytes(effect)
    if len(raw) != 40:
        raise ValueError(f"packed cubie effect must be 40 bytes, got {len(raw)}")
    return tuple(raw[source] + 20 * raw[20 + source] for source in range(20))


@dataclass(frozen=True, slots=True)
class MoveAlphabet:
    moves: tuple[str, ...] = MOVE_ORDER


@dataclass(frozen=True, slots=True)
class ShortCensusConfig:
    max_length: int = 5
    alphabet: MoveAlphabet = MoveAlphabet()


@dataclass(frozen=True, slots=True)
class TargetClass:
    target_class_id: str
    witness: tuple[str, ...]
    effect_key: bytes


def build_target_classes(alphabet: Sequence[str] = MOVE_ORDER) -> tuple[TargetClass, ...]:
    """Return deterministic exact DR-compatible representatives.

    This is a compatibility helper, not a reconstruction of the original
    historical target-class census. Existing callers only require stable exact
    representatives; no publication claim uses these class identities.
    """
    available = set(map(str, alphabet))
    candidates: tuple[tuple[str, ...], ...] = (
        (), ("U",), ("D",), ("R2",), ("L2",), ("F2",), ("B2",), ("U", "R2"),
    )
    rows: list[TargetClass] = []
    for word in candidates:
        if all(move in available for move in word):
            rows.append(
                TargetClass(
                    target_class_id=f"COMPAT-DR-TARGET-{len(rows):02d}",
                    witness=word,
                    effect_key=effect_from_word(word),
                )
            )
    if len(rows) < 4:
        raise ValueError("alphabet does not provide four stable DR representatives")
    return tuple(rows)


def iter_words_with_effects(
    config: ShortCensusConfig | None = None,
) -> Iterator[tuple[tuple[str, ...], bytes]]:
    """Yield deterministic reduced words and their exact packed effects.

    Only same-face adjacency reduction is assumed. This helper exists for
    historical compatibility and is not used as evidence for census counts.
    """
    cfg = config or ShortCensusConfig()
    alphabet = tuple(cfg.alphabet.moves)

    def visit(prefix: tuple[str, ...], depth: int):
        if depth:
            yield prefix, effect_from_word(prefix)
        if depth >= int(cfg.max_length):
            return
        previous_face = prefix[-1][0] if prefix else None
        for move in alphabet:
            if previous_face is not None and move[0] == previous_face:
                continue
            yield from visit(prefix + (move,), depth + 1)

    yield from visit((), 0)


@dataclass(frozen=True, slots=True)
class ReloadedEffectCensus:
    representatives: tuple[tuple[int, ...], ...]


def reload_effect_census_for_analysis(
    census_root: str | Path,
    config: ShortCensusConfig | None = None,
) -> ReloadedEffectCensus:
    """Load preserved representative words if a compatible sealed registry exists.

    This function intentionally fails closed if the historical artifacts are not
    present. It never invents a complete census or turns missing data into a
    negative result.
    """
    root = Path(census_root)
    cfg = config or ShortCensusConfig()
    move_to_id = {move: index for index, move in enumerate(cfg.alphabet.moves)}
    candidate_files = (
        root / "transformation_classes.jsonl",
        root / "effect_equivalence_registry.jsonl",
    )
    rows: list[tuple[int, ...]] = []
    for path in candidate_files:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            word = None
            for key in ("canonical_word", "representative_word", "word"):
                value = obj.get(key)
                if isinstance(value, list) and all(isinstance(item, str) for item in value):
                    word = tuple(value)
                    break
            if word is None:
                samples = obj.get("sample_equivalent_words") or obj.get("witness_words")
                if isinstance(samples, list) and samples:
                    value = samples[0]
                    if isinstance(value, list) and all(isinstance(item, str) for item in value):
                        word = tuple(value)
            if word is None or len(word) > cfg.max_length:
                continue
            try:
                rows.append(tuple(move_to_id[move] for move in word))
            except KeyError:
                continue
        if rows:
            break
    if not rows:
        raise FileNotFoundError(
            "Historical sealed short-census artifacts are unavailable under "
            f"{root}; compatibility mode fails closed."
        )
    seen: set[tuple[int, ...]] = set()
    unique = tuple(row for row in rows if not (row in seen or seen.add(row)))
    return ReloadedEffectCensus(unique)


__all__ = [
    "COMPATIBILITY_RECONSTRUCTION",
    "IDENTITY_KEY",
    "MOVE_ORDER",
    "MoveAlphabet",
    "ReloadedEffectCensus",
    "ShortCensusConfig",
    "TargetClass",
    "build_target_classes",
    "compose_packed",
    "effect_from_word",
    "inverse_packed",
    "iter_words_with_effects",
    "reload_effect_census_for_analysis",
    "state_codes",
]
