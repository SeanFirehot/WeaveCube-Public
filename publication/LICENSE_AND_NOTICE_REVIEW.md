# WeaveCube license and notice review

Date: 2026-09-18
Status: **PUBLIC SNAPSHOT LICENSE PACKAGING REVIEW PASS**

## Apache-2.0 source

The sanitized public snapshot uses the full Apache License, Version 2.0 text.
The license template is stored privately at:

- `publication/public_snapshot/LICENSE`

The snapshot builder maps that file to the public snapshot root as:

- `LICENSE`

The private historical research repository intentionally has no root Apache
`LICENSE`.

## Project NOTICE

The sanitized public snapshot includes a minimal project `NOTICE`:

- project: WeaveCube
- copyright: 2026 Lee Sihyun
- attribution: software developed by the WeaveCube Project

The NOTICE is intentionally narrow. It does not attempt to relicense or absorb
third-party solver notices.

## Third-party / prior-art notice

`THIRD_PARTY_NOTICES.md` is informational and records material prior art and
historical research references. It explicitly states that the sanitized
snapshot does not bundle source, binaries, pruning tables, generated assets, or
runtime modules from the external solver projects below.

### RubikTwoPhase

Repository reviewed:
- `hkociemba/RubiksCube-TwophaseSolver`

Observed repository license:
- GNU General Public License, Version 3 (GPL-3.0)

Release treatment:
- prior art / historical integration reference only;
- historical `twophase.*` bridge excluded from the sanitized snapshot.

### min2phase

Repository reviewed:
- `cs0x7f/min2phase`

Observed source-license evidence:
- `src/Search.java` states GNU GPL Version 3 or, at the user's option, any
  later version.

Release treatment:
- prior art / historical performance-baseline reference only;
- source not bundled in the sanitized snapshot.

### Nissy Classic

Repository reviewed:
- `sebastianotronto/nissy-classic`

Observed repository license:
- GNU GPL Version 3 or later.

Release treatment:
- prior art / historical cross-certification reference only;
- source, executables, pruning tables and teacher assets excluded.

### Nissy Core / h48

Repository reviewed:
- `sebastianotronto/nissy-core`

Observed repository license:
- GNU GPL Version 3 or later.

Release treatment:
- prior art / related-solver reference only;
- source and generated assets excluded.

## Why Apache-2.0 remains scoped to the sanitized snapshot

The project deliberately learned from and benchmarked against GPL-family cube
solver implementations during private research. The public-release audit does
not describe the work as clean-room.

The licensing conclusion is therefore implemented as a release-boundary rule,
not as a claim that the private historical tree was never exposed to GPL code:

1. keep the private historical research tree private;
2. build a history-free sanitized snapshot from an explicit approved manifest;
3. exclude historical external solver runtime/teacher/translation tooling;
4. verify the supported snapshot execution path independently;
5. apply Apache-2.0 only to project-authored material included in that sanitized
   snapshot;
6. preserve transparent algorithmic prior-art attribution.

## Verification status before license injection

The 160-code-file sanitized snapshot already passed:

- static project-local import closure;
- historical-integration marker gate;
- Python 3.14 compileall;
- publication-critical import smoke;
- focused Phase-2 regression: 10/10 PASS;
- no `.git` history;
- manual source scan for direct historical solver runtime imports.

## Final gate result

The final license-bearing snapshot was rebuilt and re-tested successfully.

Observed final builder result:

- code files: **160**
- builder-copied files: **186**
- generated build receipt committed separately: **1**
- public Git blobs at the provenance root: **187**
- license: **Apache-2.0**
- notice pack: **PASS**
- ignore gate: **PASS**
- import gate: **PASS**
- forbidden-marker gate: **PASS**
- publication import smoke: **PASS**
- focused Phase-2 regression: **10/10 PASS**

The published provenance root is
`d3e4773430c3f93dfc26d49772fe064bdc6109ae`. No release tag is created or
implied by this review.
