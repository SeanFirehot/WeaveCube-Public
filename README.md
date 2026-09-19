# WeaveCube

**Exact Solution Construction with the Lee Weave Method**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22834879.svg)](https://doi.org/10.5281/zenodo.22834879)

**Archived research release:** [10.5281/zenodo.22834879](https://doi.org/10.5281/zenodo.22834879)

WeaveCube is an experimental Rubik's Cube research project focused on exact solution construction over a shared move-word language.

Its core method is the **Lee Weave Method (LWM)**: maintain multiple compatible shared move words, progressively intersect heterogeneous exact constraints over those same words, delay commitment/materialization, and accept a solution only after exact witness construction and native replay verification.

> **Historical note:** the project was developed internally under the name **CubeLab**. Historical source files, experiment labels, paths, logs, hashes, and `v37.xxx` receipts retain that name where changing it would damage reproducibility.

## Publication v1 evidence

- **A18 5+5 exact signature join:** 577,368 x 577,368 = **333,353,807,424 conceptual pairs** represented implicitly; runtime Cartesian materialization is delayed until the exact candidate count is small.
- **Ordering ablation (v37.530):** Adaptive A / Fixed / Adaptive C all found **8/8 witnesses**. Median exact conditions to the <=64 threshold were **8 / 10 / 8**.
- **FIRST_WITNESS:** representative same-process closure result: **36.8624 s** full-closure geometric mean vs **15.4988 s** first exact witness (**2.378x** hot speedup), with native replay PASS.
- **ATO:** the physical ATO-zero predicate is exactly covered by **six pairwise-disjoint half-turn orbits**, each containing 663,552 states (**3,981,312 total**).
- **FULL-24:** a sealed deterministic 25-move random-move corpus solved **24/24**, replay verified **24/24**; median case wall **1.224 s**.

## Contribution boundary

WeaveCube is deliberately built in the lineage of established **two-phase-family Rubik's Cube solvers**. Historical CubeLab research studied and used Kociemba/RubikTwoPhase, min2phase, Nissy, and related techniques as prior art, benchmarks, and—in some historical experiments—external runtime dependencies.

WeaveCube does **not** claim to have invented two-phase solving, subgroup/DR reduction, Phase-2 permutation decomposition, IDA*, pattern databases, pruning tables, cosets, SAT encodings, decision diagrams, generic set intersection, or product automata. It also does not claim a universal random-state solve rate or general speed superiority over Kociemba/Nissy.

The central contribution candidate is architectural: **Lee Weave Method = shared-word exact construction + progressive exact restriction + late commitment/materialization**. In the current solver lineage, this project-specific front-end construction replaces the conventional front-end / Phase-1 search while an established Phase-2-style subgroup-completion architecture remains prior art. The exact six-orbit characterization of the project-specific ATO-zero terminal set is a separate project result.

The release review does **not** describe the implementation as clean-room. Source-code provenance, historical GPL-facing integration code, and the Apache-2.0 release boundary are documented explicitly in the public provenance records.

## Publication files

- [`docs/WeaveCube_Technical_Report_v1.pdf`](docs/WeaveCube_Technical_Report_v1.pdf) - detailed technical report
- [`presentation/WeaveCube_Core_Deck_v1.pptx`](presentation/WeaveCube_Core_Deck_v1.pptx) - presentation deck
- [`publication/CLAIMS_AND_EVIDENCE.md`](publication/CLAIMS_AND_EVIDENCE.md) - claim/evidence matrix
- [`publication/PROVENANCE_HISTORY.md`](publication/PROVENANCE_HISTORY.md) - implementation lineage and release-provenance record
- [`publication/APACHE_RELEASE_SCOPE.md`](publication/APACHE_RELEASE_SCOPE.md) - sanitized Apache-2.0 snapshot boundary
- [`publication/LICENSE_AND_NOTICE_REVIEW.md`](publication/LICENSE_AND_NOTICE_REVIEW.md) - license packaging and external-solver notice review
- [`publication/SUPPORTED_ENTRY_POINTS.md`](publication/SUPPORTED_ENTRY_POINTS.md) - supported v1.0 execution/verification paths
- [`publication/PHASE2_PROVENANCE_REVIEW.md`](publication/PHASE2_PROVENANCE_REVIEW.md) - focused comparison of project-side Phase-2 code and the FULL-24 dependency path
- [`publication/RELEASE_SCOPE_HOLD.md`](publication/RELEASE_SCOPE_HOLD.md) - historical external-solver integration files held out of automatic Apache-2.0 scope
- [`publication/REPRODUCIBILITY.md`](publication/REPRODUCIBILITY.md) - hashes, authority rules, and reproduction notes
- [`publication/EVIDENCE_SUMMARY.md`](publication/EVIDENCE_SUMMARY.md) - compact experimental receipts
- [`publication/PUBLIC_RELEASE_AUDIT.md`](publication/PUBLIC_RELEASE_AUDIT.md) - current release audit and remaining gates
- [`publication/REFERENCES.bib`](publication/REFERENCES.bib) - prior-art references

## Source status

The authoritative September 2026 development tree, whose historical local root remains `C:\python\CubeLab`, was synchronized into `publication-v1` at commit `28cd1418f0068f3b2be9575bad23aa8fc3571d3a` before the metadata rewrite. The corresponding rewritten source-sync commit is recorded in the public-release audit.

Older v100/v101 branches referenced one historical short-census module that was absent from the frozen source snapshot and repository history. The public branch contains a clearly marked exact compatibility reconstruction for that import surface; it is not used as evidence for the headline WeaveCube / Lee Weave Method claims. See the public release audit for details.

Generated resident caches and large experiment-output directories are not redistributed by default. Frozen hashes are documented where applicable, so the sealed FULL-24 result should currently be read as an experimental receipt rather than a claim that every fresh clone can immediately reproduce the benchmark without regenerating or supplying its required assets.

## Release status

This repository is the **history-trimmed public WeaveCube v1.0 research snapshot**. The Windows / Python 3.14 compile/import smoke, provenance separation, manifest/import/ignore gates, focused Phase-2 regression, license-bearing snapshot rebuild, and historical external-solver exclusion checks passed before publication.

The public provenance root is `d3e4773430c3f93dfc26d49772fe064bdc6109ae`. The corresponding build receipt records private source anchor `580f7979545e2c3acfbf7f9e1bb06ae646e7d2cd`, 160 approved code files, and 186 builder-copied files before the generated receipt itself was committed.

A focused review of the project-side Phase-2 modules and the statically resolved FULL-24 path found no direct/lightly reformatted source match in the tested comparison set and no ordinary Python dependency on the historical RubikTwoPhase bridge in the supported public closure. Historical external-solver integration files remain excluded from this Apache-2.0 snapshot.

The repository is public and the explicit `v1.0-research` Git tag and GitHub Release have been published. The corresponding archival research snapshot is deposited on Zenodo under DOI [10.5281/zenodo.22834879](https://doi.org/10.5281/zenodo.22834879).

## License

**Public snapshot license: Apache License 2.0.**

This public snapshot includes a root `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md`. The Apache-2.0 grant applies to the project-authored material included in this sanitized snapshot; provenance-sensitive historical private research material is outside this distribution.
