# WeaveCube Reproducibility Notes

## Naming convention

The public project/software name is **WeaveCube** and the core construction method is the **Lee Weave Method (LWM)**.

The project was developed internally under the historical name **CubeLab**. Historical source files, local paths, experiment labels, hashes, and `v37.xxx` receipts retain that name where changing it would break reproducibility. In particular, the authoritative development root used for the final presentation/reproducibility runs was:

```text
C:\python\CubeLab
```

## Frozen identifiers

- FULL-24 corpus SHA-256: `77bd96782994e5ae41e98ebab013de42354178bb9589e043dc12c4e416fee88e`
- D6 cache SHA-256: `5b985def30317e416a03eb8bde4c039bc7ea1c4d7c34683dff4ff8cc24055dd0`
- A18 source SHA-256 used in v37.530: `ba36819f53de97056c348de958f85285e79927e84bac6ad5533fd2f30fbe4875`
- Reviewed public source snapshot ZIP SHA-256: `9e0e202912a673a86bed2e3f230c02843a261a194a2e61fef08abf5fd78082d5`
- September source-sync commit: `28cd1418f0068f3b2be9575bad23aa8fc3571d3a`
- Technical report SHA-256: `f1e4acd09ac7399b5c532da5fd010ce10305870faf2bf5ff4307bcee3b263358`
- Core deck SHA-256: `3a0990c18177929d8536b3fb4c96752bf44495137a24964ba83f0424f9344bde`

## Release identity

- Annotated release tag: `v1.0-research`
- Tagged code snapshot commit: `d72d46c360518b460dd9a608ece771afa160e89d`
- Later DOI/citation metadata head: `99f1a9a6f33a9524a7c9988bbf51921286f3076f`

The tag identifies the released code snapshot. The later `main` commit changes DOI/citation metadata only and should not be confused with a new solver release.

## Target environment

The final presentation/reproducibility runs were performed on Windows with Python 3.14.x.

## Validation policy

Before reporting a SAT result:

1. Native exact full-Q replay is final correctness authority.
2. A resource/time cap is `UNKNOWN`, never `UNSAT`.
3. FIRST_WITNESS is SAT-only authority; no witness/resource falls back.
4. Local exact EMPTY does not prune a whole residual-graph child without explicit exhaustive-child authority.

## Public source snapshot inspection

The September 2026 source snapshot contained 3,423 files, approximately 50 MB uncompressed. The publication review found:

- no private-key blocks;
- no GitHub/OpenAI/AWS token patterns;
- no generic password/API-key assignments;
- no email-address strings;
- no `C:\Users\...` personal paths;
- no `reports/`, `pdcc_cache`, `.npy`, `.npz`, `.pkl`, or `__pycache__` content.

After removing obvious duplicate/backup/temp artifacts, the cleaned source tree passed `python -m compileall -q`. One historical script emitted a `SyntaxWarning` for an invalid escape sequence; there were no compile errors.

## Historical compatibility note

Older v100/v101 code referenced `cubelab.domino_reduction_short_census`, but the original module was absent from the frozen September source snapshot and from repository history.

The public branch therefore contains a transparent compatibility reconstruction that restores only exact packed-effect semantics derived from the canonical `CubieEffect` / `transformations` implementation. It is explicitly marked `COMPATIBILITY_RECONSTRUCTION = True`; it is not treated as the recovered historical census implementation, and its census loader fails closed when the corresponding historical sealed artifacts are unavailable.

No headline WeaveCube / Lee Weave Method publication claim depends on this reconstructed historical census module.

## Asset policy

Generated resident caches and large research-output directories are excluded from source control. Frozen assets are regenerated where practical or documented by hash. The D6 cache used by the sealed FULL-24 run is currently documented by SHA-256 rather than redistributed in this repository.

Accordingly, the published FULL-24 result is an experimental receipt. A fresh clone should not be described as end-to-end FULL-24 reproducible until all required assets have been regenerated or supplied and the public-checkout rerun has passed.

## Public checkout gate

For fresh-checkout verification of the released snapshot, run on Windows / Python 3.14:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m compileall -q src scripts tests
python scripts\repro\public_release_import_smoke.py
```

The final scoped public snapshot passed the Windows / Python 3.14.6 compile/import smoke and exact packed-effect roundtrip before release. The public release audit records the completed target-platform gate.

See `publication/PUBLIC_RELEASE_AUDIT.md` for the current audit record.
