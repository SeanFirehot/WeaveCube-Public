# Claims and Evidence

## Core method

**Lee Weave Method (LWM)** is the WeaveCube project's name for exact shared-word construction: maintain multiple compatible shared move words, progressively intersect heterogeneous exact constraints over those same words, delay commitment/materialization, then accept a witness only after native exact replay.

WeaveCube is developed in the lineage of established **two-phase-family Rubik's Cube solvers**. The project does not claim the two-phase architecture, DR/subgroup reduction, Phase-2 permutation decomposition, IDA*, pruning tables, or pattern databases as project inventions. In the current solver lineage, Lee Weave is the proposed replacement for the conventional front-end / Phase-1 search; an established Phase-2-style subgroup-completion architecture remains prior art.

| Claim | Evidence | Status |
|---|---|---|
| Lee Weave shared-word exact construction can synthesize witnesses | A18 exact-signature join | Validated mechanism |
| 333.35B conceptual 5+5 pairs need not be enumerated early | A18 candidate-count ladder | Validated |
| Adaptive marginal ordering is secondary to progressive exact intersection | v37.530 A/B/A: all arms 8/8, threshold 8/10/8 | Validated |
| Full compatible-family materialization is avoidable for SAT construction | v37.507 FIRST_WITNESS | Validated |
| Physical ATO-zero is exactly six disjoint half-turn orbits | v37.470 finite-set proof | Proved |
| Frozen operational solver succeeds on sealed sample | v37.527 FULL-24: 24/24 + replay | Validated sample |

## Prior-art and provenance boundary

- Kociemba two-phase solving and related high-performance cube-solving techniques are prior art and are cited.
- Historical CubeLab research intentionally studied Kociemba/RubikTwoPhase, min2phase, Nissy, and related techniques; some historical scripts directly integrated external solver runtime structures.
- The project is **not** described as a clean-room implementation.
- Algorithmic similarity to established two-phase methods is expected and is not itself presented as project novelty.
- Separate provenance review is required to determine whether the Apache-2.0 public release contains copied/translated protected source expression or incompatible runtime dependencies.
- See `PROVENANCE_HISTORY.md` and `PUBLIC_RELEASE_AUDIT.md` for the current release review.

## Non-claims

- No invention claim for two-phase solving, DR/subgroup reduction, Phase-2 permutation coordinates, IDA*, pattern databases, pruning tables, set intersection, cosets, SAT, decision diagrams, or generic product automata.
- No clean-room implementation claim.
- No uniform-random-state solve-rate claim from FULL-24.
- No general speed-superiority claim over Kociemba/Nissy.
- No theorem that A18 succeeds for all roots.

## Historical naming

`CubeLab` is retained only where it is part of historical/internal source names, paths, experiment labels, hashes, and v37.xxx evidence receipts.
