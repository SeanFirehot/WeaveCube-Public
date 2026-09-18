# WeaveCube provenance and implementation lineage

Date: 2026-09-17
Branch: `publication-v1`

## Purpose

This document records the implementation lineage relevant to the WeaveCube public-release review. It is intentionally conservative: it separates prior-art use, historical external-solver integration, and current project-owned implementation. It is an engineering provenance record, not legal advice.

## Solver lineage

WeaveCube did not arise independently of the established Rubik's Cube solver literature. During the historical CubeLab research phase, the project intentionally studied and used techniques from the Kociemba two-phase family and related high-performance solvers, including Kociemba/RubikTwoPhase, min2phase, and Nissy.

The current solver should therefore be described as a **two-phase-family solver with a project-specific front-end construction mechanism**, not as an independently discovered alternative to the two-phase architecture.

At a high level, the lineage is:

```text
Kociemba / min2phase / Nissy prior art
        |
        |  two-phase / DR subgroup structure,
        |  coordinate pruning, IDA*, PDB and related techniques
        v
historical CubeLab research and benchmarking
        |
        |  Lee Weave exact shared-word construction developed as
        |  a replacement for the conventional front-end / Phase-1 search
        v
WeaveCube
        |
        |  established Phase-2-style subgroup completion remains
        v
exact replay-verified solution
```

The project contribution claim is therefore **not** the invention of two-phase solving, subgroup reduction, Phase-2 permutation coordinates, IDA*, pruning tables, or pattern databases.

## Historical external-solver integration

Historical research code includes direct integration with external solver implementations. In particular, `scripts/kociemba_p1_runtime_v36.py` imports `twophase.*` modules from the RubikTwoPhase package and accesses implementation-specific coordinate/pruning-table data. Related historical scripts were built around that bridge for benchmarking and research.

Those files are provenance-sensitive. They must not be treated as evidence that all WeaveCube code is independently implemented, and they require separate release-scope/licensing treatment before an Apache-2.0 release.

Historical Nissy-facing scripts also contain external-solver asset names and verification logic. Such references are expected in benchmark/cross-certification tooling and are distinct from copying external solver source into WeaveCube production modules.

## Current Phase-2 implementation

The current project tree contains a project-side Phase-2 implementation in `src/cubelab/domino_phase2.py`. It intentionally implements an established Phase-2-style decomposition after a DR/subgroup terminal is reached.

The release audit has so far observed that this module uses project-local data structures and code paths, including project-local cubie transformations, permutation rank/unrank routines, generated move tables, generated pair-distance tables, and an IDA*-style search with native physical replay verification.

An automated comparison against selected reference repositories reported **zero normalized exact six-line source-window matches**. This is useful evidence against direct or lightly reformatted copying, but it is not proof of independent authorship and does not establish a legal conclusion.

Because the development process was exposed to and deliberately learned from existing two-phase implementations, the project should **not** describe the current implementation as "clean-room". The appropriate question is narrower: whether the public release contains protectable source-code expression copied or translated from incompatible third-party implementations, or runtime dependencies that impose incompatible distribution terms.

## 2026-09-17 forensic audit

A read-only provenance scan was run against the rewritten `publication-v1` release candidate at commit:

`de06df621fd987d9568e58d5b708574a6bbd6c92`

Summary:

- tracked files inventoried: 3,435
- normalized exact six-line source-window matches against the selected external references: 0
- long string-literal matches: 1
- numeric lookup-table fingerprint matches: 520
- identifier-overlap candidates: 307

The single long-string match was an external Nissy pruning-table identifier in verification tooling. Numeric and identifier matches require contextual review because Rubik's Cube implementations naturally share piece numbers, move indices, permutation constants, coordinate terminology, and subgroup terminology.

The audit is intentionally **not self-certifying**. Its result remains `MANUAL REVIEW REQUIRED`.

## Release provenance policy

Before an Apache-2.0 `LICENSE` file is added, the public-release review must establish all of the following:

1. **Historical dependency disclosure** — external solver integrations and prior-art influences are recorded rather than described as clean-room development.
2. **Release dependency separation** — the code on the supported public execution path does not silently require GPL or otherwise incompatible external implementation code or generated implementation-specific assets.
3. **Expression-level review** — high-risk current modules are checked for copied or translated source expression, distinctive implementation-only constants/tables, comments, identifier structure, and control-flow correspondence.
4. **Release-scope classification** — historical integration scripts are either excluded from the Apache-2.0 licensed release scope, separately attributed/licensed where appropriate, or retained only after an explicit compatibility determination.
5. **Prior-art attribution** — Kociemba two-phase solving and other material prior art remain cited in the publication materials.

## Terminology policy

Until the provenance audit is complete:

- do **not** call WeaveCube a clean-room reimplementation;
- do **not** imply that the two-phase architecture or Phase-2 decomposition is a WeaveCube invention;
- do describe Lee Weave Method as the project's proposed front-end/shared-word exact-construction contribution;
- do distinguish algorithmic prior art from source-code provenance and licensing;
- do keep the repository private and untagged as a public release.

## Current release status

Apache-2.0 has been selected as the **intended** project license, but the license file is intentionally not yet added. The repository remains in pre-publication review while the release scope and implementation provenance are verified.
