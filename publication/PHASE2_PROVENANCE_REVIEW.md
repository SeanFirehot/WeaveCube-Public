# WeaveCube Phase-2 provenance review

Date: 2026-09-17
Status: **PRELIMINARY EXPRESSION-LEVEL DIFFERENTIATION PASS — NOT A REPOSITORY-WIDE LICENSE CLEARANCE**

## Scope

This review focuses on the project-side two-phase / Phase-2 implementation paths most relevant to the publication claims and the sealed FULL-24 operational result.

Reviewed project files:

- `src/cubelab/domino_phase2.py`
- `scripts/search_twist_skeleton.py`
- `scripts/pdcc_phase2.py`
- static import closure rooted at `scripts/presentation_run_sealed_full24_v37_527.py` and its dynamically loaded v37.153 driver

Reference implementations were pinned by the 2026-09-17 forensic scan, including:

- Rubik's Cube Two-Phase Solver: `0c1ea6233cb2d4f294bc97077b60265ba76f3616`
- min2phase: `4d183b9eff8119cac72bc50ef35a7d8990740e06`

This review distinguishes **algorithmic lineage** from **source-code expression**. WeaveCube deliberately uses established two-phase-family ideas; the question here is whether the reviewed project-side code shows evidence of direct/lightly transformed copying or an incompatible runtime dependency.

## Algorithmic lineage: explicitly prior art

The following similarities are expected and are **not** claimed as WeaveCube inventions:

- transition into a DR / Phase-1 terminal subgroup;
- the Phase-2 move set with U/D quarter/half turns and side-face half turns;
- corner permutation, U/D-edge permutation, and slice-edge permutation as Phase-2 state components;
- permutation move tables;
- admissible pruning/PDB lower bounds;
- iterative-deepening / depth-bounded exact search;
- canonical move-order restrictions.

Public materials must continue to attribute the two-phase lineage to prior art, including Kociemba.

## `src/cubelab/domino_phase2.py`

### Project-side implementation characteristics

The module:

- imports project-local `CubieEffect`, `Transformation`, piece orders, and DR helpers;
- computes Phase-2 coordinates from project-local physical transformations;
- contains project-side permutation rank/unrank routines;
- generates move tables from project-local physical move effects;
- generates pair-distance tables with a project-side BFS;
- uses two pair bounds (`corner x slice` and `UD-edge x slice`) through a `max(...)` admissible heuristic;
- performs a project-side IDA-style exact search and verifies the resulting word by physical replay.

### Comparison with pinned RubikTwoPhase implementation

The pinned RubikTwoPhase implementation uses materially different implementation machinery, including:

- `CoordCube` state with implementation-specific coordinate fields;
- symmetry-reduced `corner_classidx` / `corner_sym` and flip-slice classes;
- precomputed move arrays;
- packed modulo-3 pruning representations;
- conjugation tables and symmetry-class representatives;
- a specialized threaded two-phase search and phase-transition logic.

The project-side module does not import `twophase.*` and does not use those packed/symmetry-reduced runtime structures.

The conceptual Phase-2 decomposition overlaps because it is the intended prior-art algorithmic lineage. The current implementation structure and data representation differ substantially from the pinned reference implementation.

### Automated evidence

The repository-wide forensic scan reported **zero normalized exact six-line source-window matches** against the selected reference repositories. No direct external-solver import was found in this module.

### Preliminary classification

**ALGORITHMIC_PRIOR_ART + PROJECT_SIDE_IMPLEMENTATION**

Current evidence does not show direct or lightly reformatted source copying in this module. This is an engineering provenance conclusion, not a legal determination.

## `scripts/search_twist_skeleton.py`

This historical/current operational helper is explicit about its lineage: its module documentation describes the search as using **"Kociemba-style pruning"**.

At the implementation level it builds project-side state machinery from `cubelab.pdcc`:

- corner orientation and edge orientation are encoded by project-side routines;
- slice occupancy is generated from project piece/slot maps;
- Phase-1 tables are generated from project-local move effects;
- Phase-2 compact coordinates enumerate project-side 8!/8!/4! permutations;
- move tables are generated through project-local slot mappings;
- its simple PDB/BFS and meet/search helpers operate on those generated project representations.

It does not import RubikTwoPhase/min2phase/Nissy runtime modules.

The coordinate choices are expected two-phase prior art. The implementation machinery observed here is not the symmetry-reduced packed-table machinery used by the pinned RubikTwoPhase/min2phase references.

Preliminary classification:

**ALGORITHMIC_PRIOR_ART + PROJECT_SIDE_IMPLEMENTATION**

## `scripts/pdcc_phase2.py`

This module layers exact Phase-2 search over the project-side `search_twist_skeleton` representation and the project-specific double-star PDB machinery.

Observed characteristics include:

- project-side lower-bound composition;
- iterative depth limits;
- project-side canonical-move filtering;
- optional double-star profile ordering;
- bounded legacy probing followed by an exact restart;
- explicit distinction between admissible pruning and ordering-only signals.

It imports project scripts, not external solver packages.

Preliminary classification:

**PROJECT_SIDE_IMPLEMENTATION built on ALGORITHMIC_PRIOR_ART**

## FULL-24 supported-path dependency check

A static import-closure analysis was performed from:

- `scripts/presentation_run_sealed_full24_v37_527.py`
- its dynamically loaded driver `scripts/audit_adaptive_random_coverage_v37_153.py`

The resolved project closure contained:

- **49** script files;
- **13** `src/cubelab/pdcc/*` modules;
- **62** project files total.

Within that resolved closure:

- direct `twophase` / Kociemba / min2phase / Nissy imports: **0**;
- intersection with the H1/H2/H3 provenance HOLD sets in `RELEASE_SCOPE_HOLD.md`: **0**.

Two reachable scripts contain descriptive prior-art wording (`two-phase` / `Kociemba-style`) in comments or docstrings; these are not runtime imports.

This is strong evidence that the sealed FULL-24 publication path is not silently using the historical RubikTwoPhase runtime bridge through ordinary Python imports.

Limitations: static import closure cannot prove the absence of every possible runtime subprocess, data-file, plugin, or dynamic-loading dependency. The release review should therefore retain runtime smoke/trace checks for the final scoped release.

## Historical integration is separate

The favorable findings above do **not** clear the historical integration files listed in `RELEASE_SCOPE_HOLD.md`.

In particular:

- `scripts/kociemba_p1_runtime_v36.py`
- `scripts/audit_kociemba_p1_bridge_v36_6c0.py`

remain H1 HOLD because they directly import/use external RubikTwoPhase implementation modules or implementation-specific data structures.

Likewise, scripts that depend on that bridge and Nissy implementation-semantics/translation tooling remain on HOLD pending separate release treatment.

## Current conclusion

For the reviewed project-side Phase-2 modules and the statically resolved FULL-24 path, the evidence currently supports this engineering description:

> WeaveCube deliberately uses established two-phase-family algorithmic prior art, while the reviewed current project-side implementation uses its own state representation, table generation, pruning composition, and search code. The review has not found direct/lightly reformatted source copying or an ordinary Python runtime dependency on the historical GPL-facing bridge in the FULL-24 path.

This is **not** a repository-wide Apache-2.0 clearance. The remaining work is to classify the historical HOLD set, check any additional supported public execution paths, and decide whether held historical research files are excluded, separately licensed/noticed, or otherwise retained under a defensible release structure.
