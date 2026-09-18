# Third-Party Notices and Prior Art

This file is informational. It records material prior art and historical
research references relevant to WeaveCube's provenance review.

The sanitized WeaveCube public snapshot does **not** bundle source code,
binaries, pruning tables, generated implementation assets, or runtime modules
from the external Rubik's Cube solver projects listed below.

## Kociemba / RubikTwoPhase

Reference project:
- Herbert Kociemba, Rubik's Cube Two-Phase Solver
- GitHub: `hkociemba/RubiksCube-TwophaseSolver`
- License observed in the reference repository: GNU General Public License,
  Version 3 (GPL-3.0)

WeaveCube deliberately uses established two-phase-family algorithmic prior art,
including DR / subgroup reduction and Phase-2-style completion. Those
algorithmic ideas are not claimed as WeaveCube inventions. The sanitized
snapshot does not include the historical `twophase.*` runtime bridge used in
some private CubeLab research experiments.

## min2phase

Reference project:
- Shuang Chen, min2phase
- GitHub: `cs0x7f/min2phase`
- The core `src/Search.java` source header states GNU GPL Version 3 or
  (at the user's option) any later version.

min2phase was studied as prior art / a high-performance baseline during
historical CubeLab research. Its source is not bundled in the sanitized public
snapshot.

## Nissy Classic

Reference project:
- Sebastiano Tronto, Nissy
- GitHub: `sebastianotronto/nissy-classic`
- License observed in the reference repository: GNU GPL Version 3 or later.

Nissy was used historically for comparison, cross-certification, and
implementation-semantics research. Nissy source, executables, pruning tables,
and GPL teacher assets are not bundled in the sanitized public snapshot.

## Nissy Core / h48

Reference project:
- Sebastiano Tronto and contributors, h48 / Nissy Core
- GitHub: `sebastianotronto/nissy-core`
- License observed in the reference repository: GNU GPL Version 3 or later.

This project is cited as related solver prior art. Its source and generated
assets are not bundled in the sanitized public snapshot.

## What this notice does not mean

Listing a project here does not mean its source code is incorporated into
WeaveCube, nor does it relicense that project's code.

The WeaveCube sanitized snapshot is distributed under Apache License 2.0 only
for the project-authored material included in that snapshot. Third-party
projects retain their own licenses.

External Python/runtime/test dependencies installed separately by users are not
vendored by this snapshot and retain their own licenses.

For the engineering provenance record and the exact release boundary, see:

- `publication/PROVENANCE_HISTORY.md`
- `publication/PHASE2_PROVENANCE_REVIEW.md`
- `publication/APACHE_RELEASE_SCOPE.md`
- `publication/PUBLIC_RELEASE_AUDIT.md`
- `publication/REFERENCES.bib`
