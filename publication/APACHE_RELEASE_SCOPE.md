# WeaveCube Apache-2.0 release scope

Date: 2026-09-19
Status: **PUBLIC SNAPSHOT SCOPE FROZEN — APACHE-2.0 SNAPSHOT PUBLISHED**

## Purpose

Apache License 2.0 is the license for the sanitized WeaveCube v1.0 research snapshot published from a fresh history root. The private research repository remains separate because it contains historical integration material whose provenance and external-solver dependencies must not be blurred into the public snapshot licensing claim.

This document records the release-scope rule applied to the published snapshot.

## Scope principle

The Apache-2.0 grant is intended to cover only the **sanitized public-release snapshot and project-authored files explicitly retained in that snapshot after provenance review**.

It must not be described as retroactively relicensing third-party code, external-solver packages, generated third-party assets, or provenance-sensitive historical integration material.

## Intended Apache release content

Subject to the remaining audit gates, the intended Apache-2.0 scope includes:

- current project-side `src/` implementation retained after provenance review;
- publication-critical tests and reproducibility tooling that do not depend on excluded external implementation code;
- selected supported solver/reproduction scripts, including the sealed FULL-24 operational path after final runtime verification;
- examples and project documentation authored for WeaveCube;
- publication metadata, provenance records, claim/evidence records, and reference bibliography;
- project-generated report and presentation artifacts, subject to their embedded citations/attributions.

The reviewed Phase-2 implementation is treated as established two-phase algorithmic prior art implemented through project-side code, not as a WeaveCube invention of the two-phase architecture. See `publication/PHASE2_PROVENANCE_REVIEW.md`.

## Excluded / held historical material

The sanitized public snapshot will not contain the following classes unless a later review explicitly clears a file.

### A. Direct RubikTwoPhase integration

Historical files that import `twophase.*`, access RubikTwoPhase internal coordinate/pruning arrays, or depend directly on those implementation-specific structures. Known examples include:

- `scripts/kociemba_p1_runtime_v36.py`
- `scripts/audit_kociemba_p1_bridge_v36_6c0.py`

### B. Historical scripts depending on the RubikTwoPhase bridge

Research scripts whose execution imports `kociemba_p1_runtime_v36` are outside the Apache v1.0 supported snapshot.

### C. Nissy teacher / source-semantic reconstruction and translation tooling

Historical tooling that reconstructs, translates, validates, or consumes Nissy implementation semantics derived from GPL source is outside the Apache v1.0 supported snapshot. In particular, `scripts/audit_gx31_nissy_index_translation_v37_452_g2b0.py` explicitly reconstructs Nissy 2.0.8 source-semantic index behavior and is not ordinary Apache release code.

### D. Third-party source/binaries/assets

No GPL source archive, external executable, external package source tree, teacher-generated third-party implementation asset, or third-party pruning table will be redistributed under the WeaveCube Apache license.

## Public-history policy

**The primary Apache-2.0 release will be history-trimmed / snapshot-only.**

The complete CubeLab/WeaveCube research Git history remains a private provenance archive unless a later, separate review clears it for publication. Merely deleting H1/H2/H3 files from the tip of the present repository is insufficient because those files remain reachable in earlier commits.

The public repository/release therefore must be constructed from the approved snapshot with a new public history root (or an equivalent fresh repository) so that uncleared historical integration files are not reachable through normal public Git history.

The private research repository and local pre-rewrite bundles remain provenance evidence and are not part of the Apache distribution.

## Current evidence supporting the project-side scope

The 2026-09-17 forensic scan and focused Phase-2 review currently support these engineering observations:

- zero normalized exact six-line source-window matches against the selected Nissy/Kociemba/min2phase references;
- reviewed Phase-2 modules use project-local state representation, generated move tables/PDBs, project-side search, and native replay verification;
- the statically resolved sealed FULL-24 path contains 62 project files and had no ordinary Python import dependency on RubikTwoPhase, min2phase, or Nissy;
- that FULL-24 dependency closure did not intersect the H1/H2/H3 hold sets;
- historical RubikTwoPhase and Nissy-facing research integrations are real and are disclosed rather than described as clean-room development.

These observations are not a legal opinion and do not by themselves clear every project file.

## Gate used to add `LICENSE`

The Apache-2.0 `LICENSE` was added only after all of the following were completed:

1. freeze the exact public snapshot manifest;
2. ensure direct RubikTwoPhase and Nissy teacher/source-semantic integration material is absent from that snapshot;
3. run dependency/runtime checks on every supported entry point;
4. review the technical report and presentation for wording that overstates independent invention or clean-room status;
5. create the history-trimmed public checkout and run the final Windows / Python 3.14 smoke there;
6. review the final snapshot diff, notices, metadata, and hashes before public visibility is enabled.


## Apache-2.0 packaging implementation

The release license is now implemented as a **snapshot-only packaging rule**.

The private research tree intentionally has no root Apache `LICENSE`.
Instead, the private branch stores release templates under
`publication/public_snapshot/`, and
`scripts/release/build_public_snapshot.py` maps them into the root of the
history-free public snapshot.

The public snapshot therefore receives:

- `LICENSE` — Apache License 2.0;
- `NOTICE` — WeaveCube project attribution;
- `THIRD_PARTY_NOTICES.md` — informational provenance/prior-art notices.

The third-party notice explicitly states that Kociemba/RubikTwoPhase,
min2phase, Nissy Classic, and Nissy Core/h48 source code and generated solver
assets are **not bundled** in the sanitized snapshot. Those projects retain
their own GPL-family licenses.

The final license-bearing snapshot rebuild/smoke passed before publication.
The public provenance root is
`d3e4773430c3f93dfc26d49772fe064bdc6109ae`.
