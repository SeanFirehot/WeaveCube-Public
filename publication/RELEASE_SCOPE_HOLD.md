# WeaveCube Apache-2.0 release-scope exclusions and review list

Date: 2026-09-17
Status: **H1/H2/H3 CLASSIFIED FOR v1.0 SCOPE — final snapshot packaging still pending**

This file distinguishes historical research/integration material from the intended Apache-2.0 supported WeaveCube v1.0 release scope.

Classification here is a conservative release-boundary decision. It does **not** assert that every excluded file is a derivative work or that redistribution is legally impossible. It means only that WeaveCube v1.0 will not rely on a blanket Apache-2.0 claim for these historical integration paths.

## Reference snapshot used by the forensic scan

The 2026-09-17 scan compared the release candidate against these reference checkouts:

- Nissy Classic: `486e051ce56f5c028a69217122bcd7d5a5431425`
- Nissy Core: `3cb60bcbf4ab9af4e9452a43681f1e7176b0c88f`
- min2phase: `4d183b9eff8119cac72bc50ef35a7d8990740e06`
- Rubik's Cube Two-Phase Solver: `0c1ea6233cb2d4f294bc97077b60265ba76f3616`

The audited WeaveCube release-candidate anchor was `de06df621fd987d956970db14f8603515a4ddec16e`.

## H1 — direct `twophase.*` integration

**v1.0 release treatment: EXCLUDE FROM SUPPORTED APACHE SCOPE.**

- `scripts/audit_kociemba_p1_bridge_v36_6c0.py`
- `scripts/kociemba_p1_runtime_v36.py`

These files directly import implementation modules from the external RubikTwoPhase package. The runtime bridge accesses implementation-specific coordinate/pruning arrays, symmetry classes, packed modulo-3 tables, and edge-merge data.

The files may be preserved as historical research evidence, but they are not supported WeaveCube v1.0 Apache entry points and must not be represented as independently developed project-side Phase-2 code.

## H2 — scripts that depend on the historical Kociemba runtime bridge

**v1.0 release treatment: EXCLUDE FROM SUPPORTED APACHE SCOPE.**

The forensic import inventory found 36 scripts importing `kociemba_p1_runtime_v36`:

- `scripts/audit_all_pair_obligation_mining_v36_6g1.py`
- `scripts/audit_all_triple_holdout_v36_6g3.py`
- `scripts/audit_allquad_vs_official_exact_v36_6g5.py`
- `scripts/audit_antipodal_pair_holdout_v36_6g0.py`
- `scripts/audit_backward_grammar_propagation_potential_v36_6h6.py`
- `scripts/audit_backward_requirement_propagation_v36_6f1.py`
- `scripts/audit_cached_official_first_runtime_v36_6h5.py`
- `scripts/audit_column_local_selector_v36_6h0.py`
- `scripts/audit_elastic_boundary_bridge_v36_6e0.py`
- `scripts/audit_elastic_boundary_frontier_v36_6e1.py`
- `scripts/audit_exact_minimum_coupling_order_v36_6h8.py`
- `scripts/audit_expanded_signature_third_holdout_v36_6g10.py`
- `scripts/audit_four_way_column_causality_v36_6f0.py`
- `scripts/audit_frozen_hitrate_second_holdout_v36_6h3.py`
- `scripts/audit_frozen_sparse_quad_holdout_v36_6g8.py`
- `scripts/audit_global_sparse_quad_library_v36_6g7.py`
- `scripts/audit_k2_column_table_ab_v36_6i4.py`
- `scripts/audit_k2_radius8_practical_coverage_v36_6i1.py`
- `scripts/audit_k2_table_backward_h_tail_v36_6i7.py`
- `scripts/audit_kociemba_p2_positive_control_v36_6d1.py`
- `scripts/audit_learned_witness_cascade_holdout_v36_6h2.py`
- `scripts/audit_minimal_coupled_pattern_v36_6f3.py`
- `scripts/audit_official_first_grammar_augmentation_v36_6h4.py`
- `scripts/audit_official_free_coupled_requirement_v36_6f4.py`
- `scripts/audit_optimal_dr_decision_fork_v36_6e2.py`
- `scripts/audit_ordered_grammar_cascade_v36_6h1.py`
- `scripts/audit_piece_obligation_requirement_ladder_v36_6f2.py`
- `scripts/audit_residual_coupling_order_v36_6h7.py`
- `scripts/audit_residual_quadruple_obligation_mining_v36_6g4.py`
- `scripts/audit_residual_signature_completion_v36_6g9.py`
- `scripts/benchmark_k2_coord_tail_v36_6i9.py`
- `scripts/benchmark_k2_coord_tail_v36_6i91.py`
- `scripts/benchmark_kociemba_joint_h_v36_6c1.py`
- `scripts/benchmark_p2_prematerialize_v36_6d0.py`
- `scripts/search_kociemba_hard20_jointp1_p2pregate_v36_6d0.py`
- `scripts/search_kociemba_hard20_jointp1_v36_6c2.py`

These scripts may contain substantial project-authored research logic. Their v1.0 exclusion arises from their historical dependency chain, not from an automated finding of copied source text.

## H3 — Nissy teacher / implementation-semantics chain

**v1.0 release treatment: EXCLUDE FROM SUPPORTED APACHE SCOPE.**

This category includes historical tooling that reconstructs, translates, validates, or consumes implementation semantics derived from Nissy GPL source or Nissy-specific teacher assets.

A central example is:

- `scripts/audit_gx31_nissy_index_translation_v37_452_g2b0.py`

which explicitly states that it reconstructs Nissy 2.0.8 source-semantic NX31/GX31 indexing behavior and reproduces implementation-specific ordering/translation semantics.

Known dependent/related source-semantic files include:

- `scripts/audit_gx31_nissy_frame_aware_translation_v37_452_g2b0_r1.py`
- `scripts/audit_gx31_official_frame_aware_recurrence_v37_452_g2b1_r2.py`
- `scripts/audit_gx31_official_symdata_semantic_diff_v37_452_g2b1_d0.py`
- `scripts/audit_gx31_official_transtorep_label_direction_v37_452_g2b1_d1.py`
- `scripts/audit_gx31_transtorep_stabilizer_exception_v37_452_g2b1_d1_r1.py`
- `scripts/audit_nx31_compact_fallback_value_shadow_v37_452_g2c0.py`
- `scripts/audit_corners_pdb_value_shadow_v37_452_g2c1.py`
- `scripts/audit_full_multiview_estimator_v37_452_g2c2.py`
- `scripts/audit_compiled_incremental_multiview_kernel_v37_453_k0.py`

The historical v37.449 teacher/provenance tooling is also outside the supported Apache v1.0 execution scope, including teacher-build, teacher-asset, patch-generation, teacher-case, license-boundary, and delivery-assembly tooling where it handles GPL teacher source/assets.

Examples include:

- `scripts/audit_license_boundary_v37_449.py`
- `scripts/audit_teacher_assets_v37_449.py`
- `scripts/audit_teacher_build_v37_449.py`
- `scripts/generate_teacher_patch_v37_449.py`
- `scripts/run_teacher_case_v37_449.py`
- `scripts/assemble_delivery_v37_449.py`

This category is defined semantically, not only by the names above: any historical script whose operation requires Nissy GPL teacher source/assets or reconstructs Nissy implementation-specific source semantics remains outside the supported Apache v1.0 scope unless a later file-specific review explicitly clears it.

## P1 — reviewed current project-side Phase-2 code

**Release treatment: PRELIMINARY EXPRESSION-LEVEL DIFFERENTIATION PASS; not part of H1/H2/H3 exclusion.**

- `src/cubelab/domino_phase2.py`
- `scripts/search_twist_skeleton.py`
- `scripts/pdcc_phase2.py`

A focused follow-up review is recorded in `PHASE2_PROVENANCE_REVIEW.md`. The reviewed paths deliberately use established two-phase-family algorithmic prior art but use project-local state representation, table generation, pruning composition, and search code rather than the pinned RubikTwoPhase/min2phase packed/symmetry-reduced runtime structures.

The automated comparison found no normalized exact six-line source-window match against the selected external references, and the reviewed files do not directly import external solver packages.

This preliminary PASS is an engineering provenance conclusion, not a repository-wide legal clearance.

## FULL-24 supported-path check

The static import closure rooted at `scripts/presentation_run_sealed_full24_v37_527.py` and its dynamically loaded v37.153 driver resolved to 62 project files (49 scripts and 13 `src/cubelab/pdcc/*` modules).

Observed in that resolved closure:

- direct `twophase` / Kociemba / min2phase / Nissy imports: **0**;
- intersection with the then-defined H1/H2/H3 set: **0**.

This supports separation between the sealed publication benchmark path and the historical external-solver integration branches. Final scoped runtime verification is still required because static import analysis cannot prove the absence of every possible subprocess, data-file, plugin, or dynamic-loading dependency.

## Supported-v1 boundary

The supported release boundary is defined in:

- `publication/APACHE_RELEASE_SCOPE.md`
- `publication/SUPPORTED_ENTRY_POINTS.md`

Historical scripts outside that supported boundary are not automatically production interfaces merely because they remain visible in Git history or a research snapshot.

## Final packaging decision still required

Before adding the root Apache-2.0 license, the release must choose one of these mechanically clear packaging strategies:

1. **preferred:** create a scoped final release snapshot/tree that omits H1/H2/H3 material from the current release contents while preserving local/private research backups; or
2. retain historical files in a public research tree but make the Apache scope/exceptions unmistakable with explicit per-path licensing/provenance notices.

If ambiguity remains, use option 1.

## Release rule

Until final packaging is complete:

- do not add the root Apache-2.0 `LICENSE` file;
- do not merge/publicize/tag the release;
- do not call the development history clean-room;
- do not represent H1/H2/H3 as supported Apache-licensed v1.0 code;
- preserve historical research evidence while release packaging is prepared;
- use `PROVENANCE_HISTORY.md`, `PHASE2_PROVENANCE_REVIEW.md`, `APACHE_RELEASE_SCOPE.md`, `SUPPORTED_ENTRY_POINTS.md`, and `PUBLIC_RELEASE_AUDIT.md` as the controlling records.
