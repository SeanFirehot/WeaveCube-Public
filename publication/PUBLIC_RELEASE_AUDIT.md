# WeaveCube Public Release Audit

Date: 2026-09-17
Branch: `publication-v1`

## Status

**Release preparation: HOLD — repository must remain private until implementation provenance and the Apache-2.0 release scope are resolved.**

Completed gates:

1. **PASS** — reachable `main` and `publication-v1` commit metadata was rewritten to remove the personal email identity; the temporary `publication-v1-lineendings` branch was deleted; pre-rewrite bundles were preserved locally.
2. **PASS** — the required Windows / Python 3.14 compile/import smoke passed after the rewrite.
3. **SELECTED** — Apache License 2.0 is the intended software license.

The license file is **not yet added**. The release review has reopened a substantive provenance gate after confirming that historical CubeLab development intentionally used Kociemba/RubikTwoPhase-family architecture and included direct external-solver integration in some historical research scripts.

Remaining substantive gate:

4. **PENDING** — classify implementation provenance and define the exact code that can be distributed under Apache-2.0.

Do not merge to `main`, change visibility to public, add the final `LICENSE`, or create the public release until this provenance gate passes and the final branch diff is reviewed.

## Source synchronization

- Historical authoritative development root: `C:\python\CubeLab`
- Public repository: `SeanFirehot/WeaveCube`
- Pre-history-scrub September source-sync commit: `28cd1418f0068f3b2be9575bad23aa8fc3571d3a`
- Rewritten September source-sync commit: `cb50d3ed98639aabe61bfa93b51cbf7d04d8f082`
- Source-sync commit message: `Sync September 2026 WeaveCube source snapshot (historical CubeLab tree)`
- Reviewed source snapshot ZIP SHA-256: `9e0e202912a673a86bed2e3f230c02843a261a194a2e61fef08abf5fd78082d5`

Metadata rewrite preserved source tree content; the commit subject and tree content remain the continuity anchors.

### Rewritten branch heads at metadata-rewrite completion

- `main`: `d5d4b3896c5c02589c87aa9d47b51fff6520b83d`
- `publication-v1`: `de06df621fd987d9568e58d5b708574a6bbd6c92`

Subsequent publication-document commits advance `publication-v1`; the SHAs above are retained as rewrite/audit anchors rather than asserted as the current publication branch head.

## Repository hygiene

- `.gitattributes` added to enforce canonical LF for source/document text, CRLF for PowerShell working copies, and binary handling for publication/cache formats.
- `.gitignore` expanded to exclude runtime caches, generated arrays, resident cache files, archives, backups, editor metadata, and local publication/render directories.
- Recursive publication-branch tree review found no stale `CubeLab_Technical_Report` / `CubeLab_Core_Deck` binaries and no committed `.bak`, `.npy`, `.pkl`, or `__pycache__` artifacts.
- The pre-publication source scan found no private-key blocks, common GitHub/OpenAI/AWS token patterns, generic password/API-key assignments, email strings, or `C:\Users\...` personal paths in the reviewed source snapshot.

### Git-history privacy result

**PASS.**

The release preparation workflow created local pre-rewrite Git bundles before changing history. The affected author/committer identity was rewritten to the GitHub noreply identity while preserving names, commit messages, trees, and branch structure.

Post-rewrite checks on local `main` and `publication-v1` showed only noreply identities:

- `84167055+SeanFirehot@users.noreply.github.com`
- `noreply@github.com`

A non-noreply filter over both rewritten local branches returned no results.

The rewritten histories were then pushed with `--force-with-lease`, and `publication-v1-lineendings` was deleted from the remote. A subsequent fetch confirmed local and remote branch heads matched at the rewrite-completion checkpoint.

## Implementation provenance review

### Development lineage

The project is **not** being represented as a clean-room implementation of a Rubik's Cube solver.

Historical CubeLab development intentionally studied and used established Kociemba two-phase-family techniques and related solver work. The current solver lineage retains an established Phase-2-style subgroup-completion architecture while the project-specific Lee Weave mechanism replaces the conventional front-end / Phase-1 search with exact shared-word construction.

This distinction is now part of the release boundary:

- two-phase solving, DR/subgroup reduction, Phase-2 permutation decomposition, IDA*, pruning tables, and PDB techniques are prior art;
- Lee Weave shared-word exact construction is the project's proposed architectural contribution;
- prior-art use does not by itself determine source-code licensing, so implementation provenance must be reviewed separately.

See `publication/PROVENANCE_HISTORY.md`.

### Historical external integration

Historical source contains direct external-solver integration. In particular, `scripts/kociemba_p1_runtime_v36.py` imports `twophase.*` modules and accesses RubikTwoPhase implementation-specific coordinate/pruning data. Related historical research scripts depend on that bridge.

These files are **provenance-sensitive** and must be classified before the Apache-2.0 release scope is finalized. They are not treated as evidence of repository-wide independent implementation.

A historical v37.449 GPL teacher/prototype boundary audit remains useful evidence for the narrow experiment it covered, but it is **not** treated as a repository-wide clean-room certificate.

### 2026-09-17 forensic scan

A read-only source-provenance scan was run against the rewritten release candidate at commit `de06df621fd987d9568e58d5b708574a6bbd6c92` and compared selected source characteristics against Nissy Classic, Nissy Core, min2phase, and Rubik's Cube Two-Phase reference repositories.

Automated results:

- tracked files inventoried: **3,435**
- provenance/license/reference marker hits: **336**
- Python import records: **31,131**
- relevant commit-message hits: **0**
- normalized exact six-line source-window matches: **0**
- long string-literal matches: **1**
- numeric lookup-table fingerprint matches: **520**
- identifier-overlap candidates: **307**

Interpretation:

- zero exact normalized six-line matches is evidence against direct/lightly reformatted copying within the tested comparison set, but is **not proof of independent authorship**;
- the single long-string match is an external Nissy pruning-table identifier in verification tooling and is currently classified as external-reference/attribution context rather than source-copy evidence;
- numeric and identifier matches require contextual review because cube software naturally shares piece numbering, move indices, subgroup terminology, and permutation-related constants;
- high-risk production and historical integration modules still require manual expression-level review.

Current provenance result: **MANUAL REVIEW REQUIRED**.

### Required provenance closure

Before adding the Apache-2.0 license file, the release review must establish:

1. **Historical dependency disclosure** — external solver integrations and prior-art influences are recorded accurately.
2. **Release dependency separation** — the supported public execution path does not silently require GPL or otherwise incompatible external implementation code or implementation-specific generated assets.
3. **Expression-level review** — high-risk current modules are checked for copied/translated protected source expression, distinctive implementation-only constants/tables, comments, identifier structure, and control-flow correspondence.
4. **Release-scope classification** — historical integration scripts are excluded, separately attributed/licensed, or explicitly cleared for the intended release scope.
5. **Prior-art attribution** — Kociemba two-phase solving and other material references remain cited in public materials.

## Publication artifacts

The GitHub blob identities match the generated final artifacts exactly.

| Artifact | Bytes | SHA-256 | Git blob SHA |
|---|---:|---|---|
| `docs/WeaveCube_Technical_Report_v1.pdf` | 453,832 | `f1e4acd09ac7399b5c532da5fd010ce10305870faf2bf5ff4307bcee3b263358` | `b082875d8918129fc03371ca659a0f707f87792e` |
| `presentation/WeaveCube_Core_Deck_v1.pptx` | 53,136 | `3a0990c18177929d8536b3fb4c96752bf44495137a24964ba83f0424f9344bde` | `cc69ccb57c936177b33a78c8e08aab6623cd76c5` |

The blob identities and SHA-256 digests remain valid across the metadata-only history rewrite because the file contents did not change.

**Publication-content note:** the technical report and presentation predate the provenance clarification above. Before final release they must be reviewed for any wording that could imply independent invention of two-phase structure or a clean-room implementation.

## Compile and import checks

**PASS on Windows / Python 3.14.6 after history rewrite.**

Executed from the rewritten `publication-v1` checkout:

```powershell
$env:PYTHONPATH = "$PWD\src"
python --version
python -m compileall -q src scripts tests
python scripts\repro\public_release_import_smoke.py
```

Observed environment:

- Python `3.14.6`
- `compileall`: no compile errors
- one pre-existing `SyntaxWarning` in `scripts/audit_fast_exact_min_r5_cores_v37_96.py` for an invalid escape sequence; non-blocking and already known

Publication-critical imports passed:

- `cubelab.ato.six_coset_terminal`
- `cubelab.constructive.c4_first_witness_backend`
- `cubelab.constructive.exact_closure`
- `cubelab.column_families.context_aware_pair_engine_v37_360`
- `cubelab.global_field.shared_word_bitmap_v37_458`
- `cubelab.pdcc.engine`
- `cubelab_global_backbone.ida_kernel`
- `cubelab.constraint_engine.engine`

The exact packed-effect compatibility roundtrip also passed.

Final smoke result: **`PUBLIC RELEASE IMPORT SMOKE: PASS`**.

Focused related tests from the preliminary release review had also passed **19/19**.

## Historical compatibility restoration

The September public snapshot referenced `cubelab.domino_reduction_short_census` from older v100/v101 research branches, but that historical module was absent from the frozen source snapshot and was not present in the repository history.

`src/cubelab/domino_reduction_short_census.py` was therefore added as a **transparent compatibility reconstruction**, not as a claim to recover the original census implementation.

The compatibility module:

- derives packed 20-cubie effect semantics exactly from the canonical `CubieEffect` / `transformations` implementation;
- restores exact composition, inverse, state-code, and word-effect APIs used by historical callers;
- labels itself with `COMPATIBILITY_RECONSTRUCTION = True`;
- does not use reconstructed historical census identities as publication evidence;
- fails closed when historical sealed census artifacts required for a census reload are unavailable.

The public claims in the WeaveCube technical report do not depend on this reconstructed historical census module.

## Asset policy

Generated resident caches and large research outputs are **not redistributed by default**. They are either regenerated or documented by frozen hashes. In particular, the D6 cache is currently documented by SHA-256 rather than committed to the repository.

Therefore a fresh public checkout cannot yet be described as reproducing the sealed FULL-24 benchmark end-to-end without separately obtaining or regenerating all required assets. The published FULL-24 result remains an experimental receipt, not a claim that every clone can immediately rerun it.

## Claim boundary check

Public materials must retain these boundaries:

- WeaveCube belongs to the established two-phase-family lineage; two-phase/DR/Phase-2 architecture is prior art.
- Lee Weave is presented as the project's front-end/shared-word exact-construction contribution, not as invention of Phase 2.
- FULL-24 is a sealed deterministic 25-move random-move corpus, not a uniform random cubie-state sample.
- No general speed-superiority claim over Kociemba or Nissy is made.
- Adaptive A18 ordering is presented as a secondary optimization; progressive exact intersection is the core mechanism.
- ATO is described publicly as six pairwise-disjoint half-turn orbits, avoiding left/right coset convention ambiguity.
- The final v37.517 C4 economics claim is not asserted unless its end-to-end receipt is added explicitly.
- The implementation is not described as clean-room.

## Remaining release gates

- [x] Rewrite reachable commit metadata to the GitHub noreply identity; preserve a local pre-rewrite bundle.
- [x] Delete `publication-v1-lineendings`.
- [x] Record rewritten source-sync and rewrite-completion branch-head SHAs.
- [x] Run the exact public checkout smoke on Windows / Python 3.14 after rewrite.
- [x] Select intended software license: Apache License 2.0.
- [x] Add an explicit provenance/lineage record and remove repository-wide clean-room language.
- [x] Complete focused expression-level review of current Phase-2 publication paths; historical integration remains excluded/held.
- [x] Define the Apache-2.0 release architecture as a sanitized, history-trimmed snapshot; keep uncleared historical integration code and the private research history outside the public snapshot.
- [x] Review technical report and presentation wording against the clarified two-phase lineage; revised binaries generated and visually verified, pending repository binary replacement.
- [x] Replace report/deck binaries with the verified provenance-revised artifacts and freeze their new hashes.\n- [ ] Build the exact 160-file code snapshot from `publication/PUBLIC_SNAPSHOT_CODE_MANIFEST.txt` using `scripts/release/build_public_snapshot.py`.\n- [ ] Add `LICENSE` (and `NOTICE`/third-party notices if required) only after the sanitized snapshot passes final provenance/runtime checks.
- [ ] Review final branch diff and metadata.
- [ ] Merge to `main` only after explicit approval.
- [ ] Change repository visibility to public only after explicit approval.
- [ ] Tag/release `v1.0-research` after merge/publication review.


## Sanitized public snapshot decision

The release architecture is now **snapshot-only / history-trimmed**.

The private research repository contains real historical RubikTwoPhase/Nissy-facing integration and therefore will not be exposed as the primary Apache-2.0 distribution merely by deleting held files from the branch tip. The approved public-code dependency closure is frozen in `publication/PUBLIC_SNAPSHOT_CODE_MANIFEST.txt`:

- code files: **160**
- scripts: **50**
- `src/` modules: **106**
- held external-solver integration filenames in the code manifest: **0**
- focused Phase-2 regression included separately: `tests/test_domino_phase2.py`

The builder `scripts/release/build_public_snapshot.py` copies only the approved manifest plus selected publication metadata/artifacts into a new directory and explicitly does not copy `.git` history. The builder passed `py_compile` and a synthetic end-to-end build smoke before being committed.

See:
- `publication/APACHE_RELEASE_SCOPE.md`
- `publication/PUBLICATION_ARTIFACT_WORDING_REVIEW.md`


### Snapshot import-closure correction

The first history-free snapshot smoke exposed two project-local manifest omissions:

- `src/cubelab/constructive/exact_live_estimator.py`
- `src/cubelab/global_field/ato_bidirectional_v37_458.py`

Both were reviewed and found to depend only on already approved project-local modules. No RubikTwoPhase, min2phase, or Nissy dependency was introduced. The public code manifest was therefore expanded from 157 to **159** files. This was a packaging/dependency-closure correction, not a provenance exception.


### Second snapshot smoke correction

The next history-free snapshot smoke passed Phase-2 regression (10/10), history exclusion (`.git` absent), and external-solver filename checks, but exposed one further project-local import omission:

- `src/cubelab/constructive/exact_one_scheduler.py`

Review showed that this module depends only on already approved project-local `exact_live_estimator` and `prepared_q_restrict` modules. The manifest was expanded to **160** code files.

To prevent iterative one-by-one packaging failures, `scripts/release/build_public_snapshot.py` now performs a pre-copy AST-based closure check for obvious project-local Python imports that resolve in the private source tree but are absent from the public manifest.

The manual `Select-String` check also reported two strings from the snapshot builder itself because the builder contains those names in its forbidden-marker list. Those self-referential gate strings are not runtime imports or historical integration dependencies.


### Static import-gate namespace-package correction

The first AST closure gate over-reported `src/cubelab/ato/__init__.py` as required by
`from cubelab.ato import column_grounding`. Review showed that the private package
initializer imports a substantially broader historical API surface (`adapter`,
`projections`, `transitions`, etc.) that is intentionally outside the sanitized
snapshot.

The public snapshot already uses namespace-package directories successfully for the
supported imports. The static gate was therefore corrected to require concrete local
module files (for example `src/cubelab/ato/column_grounding.py`) rather than
automatically requiring source-tree package initializers.

Runtime import smoke and focused regression tests remain the final authority for the
supported snapshot execution paths.

The failed build removed/recreated the destination before the gate fired, so subsequent
`compileall` / pytest "file not found" output from that empty destination is not a
separate release failure.


### Windows case-insensitive static-gate correction

A subsequent static import-closure run reported `src/cubelab/pdcc/MOVES.py`
as missing from three files. This was a false positive caused by Windows'
case-insensitive filesystem: the imported symbol `MOVES` was incorrectly
matched to the tracked module `src/cubelab/pdcc/moves.py`.

The builder now resolves candidate local modules against the exact,
case-sensitive path set returned by `git ls-files`, rather than relying on
`Path.is_file()` on Windows. This preserves the static closure gate while
preventing symbol/module case collisions from creating false dependencies.


## 2026-09-18 sanitized snapshot runtime/provenance gate

A history-free sanitized snapshot built from private source head
`0c154a78f9ec730233e1a8685cb652b6745703d7` passed the release-critical
runtime/provenance checks before license injection:

- public code files: **160**
- snapshot files before license/notice injection: **182**
- static project-local import closure: **PASS**
- forbidden historical integration marker gate: **PASS**
- `PUBLIC RELEASE IMPORT SMOKE: PASS`
- focused `tests/test_domino_phase2.py`: **10/10 PASS**
- `.git` present in snapshot: **False**
- manual runtime-source scan excluding the gate implementation itself:
  no `kociemba_p1_runtime_v36`, no Nissy source-semantic translator import,
  and no direct `twophase` Python import.

This closes the principal provenance/runtime release gate for the scoped code.
The subsequent licensing change is packaging-only: the builder now injects
Apache-2.0 `LICENSE`, a minimal project `NOTICE`, and an informational
`THIRD_PARTY_NOTICES.md` into the sanitized snapshot root. A final rebuild and
smoke on that license-bearing snapshot remains required before publication.

## License packaging decision

The private research repository intentionally receives **no root LICENSE**.
Instead, authoritative release-license templates live under
`publication/public_snapshot/` and are copied to the sanitized snapshot root:

- `LICENSE` — Apache License 2.0 full text;
- `NOTICE` — minimal WeaveCube project attribution;
- `THIRD_PARTY_NOTICES.md` — transparent prior-art / historical solver
  references and an explicit statement that those GPL solver sources, binaries,
  pruning tables, and implementation assets are not bundled.

This prevents the Apache grant for the sanitized release snapshot from being
confused with the broader private historical CubeLab/WeaveCube research tree.


### Public Git staging / .gitignore correction

The first local root commit of the sanitized snapshot reported only 182 tracked
files even though the builder had copied the full release set plus its build
receipt. Review identified the cause as the unanchored `cache/` rule in
`.gitignore`, which also matched the project source package:

- `src/cubelab/constraint_engine/cache/__init__.py`
- `src/cubelab/constraint_engine/cache/cache_delta.py`
- `src/cubelab/constraint_engine/cache/joint_support_cache.py`
- `src/cubelab/constraint_engine/cache/slot_contribution_index.py`
- `src/cubelab/constraint_engine/cache/state_slot_cache.py`

No public push had succeeded, so the incomplete root commit was not published.

The private source `.gitignore` now anchors generated-output rules at the
repository root (`/cache/`, `/caches/`, `/reports/`, `/pdcc_cache/`)
so the source cache package remains trackable.

The snapshot builder now also runs `git check-ignore --no-index` against every
approved public code-manifest path and fails closed if any approved source file
would be hidden by the release `.gitignore`.


### Windows stale-snapshot replacement correction

A rebuild of the sanitized snapshot failed while deleting the previous local
public-repository `.git/objects` tree with `WinError 5`. The failure occurred
before any new snapshot files were copied and is a Windows filesystem/readonly
artifact issue, not a provenance/runtime failure.

`scripts/release/build_public_snapshot.py` now uses an `onexc` handler for
`shutil.rmtree` that clears the read-only attribute and retries deletion.
If a file is actively locked by another process, the builder fails with an
explicit instruction to close that process or choose a fresh destination.
