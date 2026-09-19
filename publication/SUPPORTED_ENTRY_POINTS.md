# WeaveCube v1.0 supported entry points

Date: 2026-09-17
Status: **RELEASED — v1.0-research**

## Purpose

The historical CubeLab tree contains thousands of experiment scripts. Public availability of those files does not mean every historical experiment is a supported WeaveCube v1.0 interface.

For release/provenance purposes, v1.0 support is intentionally narrow. This allows dependency and licensing review to follow concrete execution paths rather than treating every historical research script as production code.

## Supported release verification entry point

### `scripts/repro/public_release_import_smoke.py`

This is the mandatory fresh-checkout import/compatibility smoke for the v1.0 release candidate.

It verifies these project modules:

- `cubelab.ato.six_coset_terminal`
- `cubelab.constructive.c4_first_witness_backend`
- `cubelab.constructive.exact_closure`
- `cubelab.column_families.context_aware_pair_engine_v37_360`
- `cubelab.global_field.shared_word_bitmap_v37_458`
- `cubelab.pdcc.engine`
- `cubelab_global_backbone.ida_kernel`
- `cubelab.constraint_engine.engine`

It also exercises the compatibility reconstruction in `cubelab.domino_reduction_short_census` through exact packed-effect roundtrips.

## Supported publication benchmark path

### `scripts/presentation_run_sealed_full24_v37_527.py`

This is the supported wrapper for the sealed FULL-24 publication receipt.

The wrapper dynamically loads the frozen v37.153 operational driver and injects the already sealed 24-case corpus. The publication audit has statically resolved the ordinary project import closure rooted at the wrapper/driver and found no direct RubikTwoPhase, min2phase, or Nissy runtime import in that closure and no intersection with the then-defined H1/H2/H3 provenance hold sets.

The FULL-24 path still requires its frozen/generated assets as documented in the reproducibility records. The existing FULL-24 result is an experimental receipt, not a claim that a fresh clone can reproduce the benchmark without supplying/regenerating those assets.

## Project-side Phase-2 path

The following current project-side modules/helpers are part of the reviewed implementation lineage and may be used by supported/project-current workflows:

- `src/cubelab/domino_phase2.py`
- `scripts/search_twist_skeleton.py`
- `scripts/pdcc_phase2.py`

Their use of two-phase/DR decomposition is explicitly prior art. The focused provenance review currently classifies them as project-side implementations rather than direct external-solver runtime integrations. See `publication/PHASE2_PROVENANCE_REVIEW.md`.

## Not supported as v1.0 release interfaces

The following are not supported v1.0 interfaces even if retained for research traceability:

- historical RubikTwoPhase/Kociemba bridge experiments;
- scripts depending on `kociemba_p1_runtime_v36`;
- Nissy GPL-teacher/source-semantic translation/reconstruction experiments;
- one-off benchmark, attribution, reverse-engineering, ablation, archaeology, or mechanism-extraction scripts not explicitly promoted into this document;
- scripts whose required generated assets are not part of the supported release/reproduction contract.

No compatibility promise is made for those historical experiments.

## Promotion rule

A historical script or module may be promoted to a supported v1.x entry point only after:

1. its full project dependency closure is reviewed;
2. external runtime/data dependencies are documented;
3. provenance-sensitive dependencies are either removed, independently implemented, or separately licensed/noticed as appropriate;
4. a fresh-checkout test is added;
5. the release audit is updated.

## Post-release verification

The final scoped public snapshot completed the Windows / Python 3.14 fresh-checkout compile/import smoke, focused Phase-2 regression, dependency/runtime review, publication merge, public visibility change, and `v1.0-research` tagging before release.

For future v1.x releases or independent fresh-checkout verification, repeat the supported-entry-point dependency/runtime audit and Windows / Python 3.14 smoke. A supported path must not silently fall back to an excluded historical external-solver bridge.
