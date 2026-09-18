# WeaveCube publication artifact wording review

Date: 2026-09-18
Status: **WORDING REVIEW PASS — revised binaries installed at canonical repository paths**

## Purpose

The technical report and core deck were created before the final provenance clarification. This review checks that the publication artifacts do not imply that WeaveCube independently invented the two-phase architecture or that the implementation was developed in a clean-room process.

The engineering provenance position is:

- WeaveCube belongs to the established Kociemba/min2phase/Nissy two-phase-family lineage.
- Two-phase / DR subgroup structure, Phase-2-style completion, IDA*, pruning tables, and related coordinate/PDB ideas are prior art.
- The proposed WeaveCube contribution is the Lee Weave shared-word exact-construction mechanism used as a project-specific front-end in place of the conventional Phase-1 search.
- Historical direct external-solver integrations exist in the private research archive and are outside the intended sanitized Apache-2.0 snapshot.
- The project must not describe itself as a clean-room reimplementation.

## Technical report review

Original artifact:
- path: `docs/WeaveCube_Technical_Report_v1.pdf`
- prior SHA-256: `32d62eaf9b62ff1042f4848b68bbbbd0496a51e06bc2d0a9d0297e7632279ce3`
- prior bytes: 184,174

The report already:
- cites Kociemba's two-phase algorithm as prior art;
- explains the G1 / Phase-2 decomposition;
- excludes invention claims for subgroup solving, IDA*, pruning tables, SAT, decision diagrams, and generic set operations;
- contains no clean-room claim.

Two passages required revision.

### Release-gate wording

The earlier page-8 text instructed the release process to merge the full current solver tree. That no longer matches the provenance policy.

Revised policy:
- build a sanitized, history-trimmed public snapshot from the approved source manifest;
- do not publish the full private research tree/history;
- keep direct RubikTwoPhase and Nissy teacher/source-semantic integrations outside the snapshot;
- add Apache-2.0 only after provenance/release-scope closure;
- run the final smoke on the exact sanitized checkout.

### Conclusion wording

The broad phrase "different exact construction paradigm" was narrowed.

Revised conclusion states that WeaveCube is best described as a **two-phase-family solver with a project-specific front-end exact construction mechanism over shared move words**, while Phase-2-style subgroup completion remains established prior art.

Revised local artifact:
- filename: `WeaveCube_Technical_Report_v1_provenance_revised.pdf`
- bytes: 453,832
- SHA-256: `f1e4acd09ac7399b5c532da5fd010ce10305870faf2bf5ff4307bcee3b263358`

Verification:
- original PDF rendered before editing;
- revised PDF rendered after editing;
- page 8 visually inspected;
- no clipping, overlap, or broken glyphs observed.

## Core deck review

Original artifact:
- path: `presentation/WeaveCube_Core_Deck_v1.pptx`
- prior SHA-256: `91aee4641dcd20d795079b0a4807674f5b2c14cf5fd33e2a2dabdf7e6b1d4eea`
- prior bytes: 53,117

The deck already lists Kociemba two-phase as prior art and does not contain a clean-room claim.

Three wording edits were made in the revised local deck:

1. Slide 3:
   - `Two-phase-family solver`
   - `Lee Weave replaces conventional Phase-1 search.`

2. Slide 11 non-claim boundary:
   - `× Inventing two-phase / Phase-2 architecture`

3. Slide 12 subtitle:
   - `A different exact front-end construction mechanism within the two-phase family.`

Revised local artifact:
- filename: `WeaveCube_Core_Deck_v1_provenance_revised.pptx`
- bytes: 53,136
- SHA-256: `3a0990c18177929d8536b3fb4c96752bf44495137a24964ba83f0424f9344bde`

Verification:
- revised deck rendered to images;
- slides 3, 11, and 12 visually inspected;
- no clipping or overlap observed;
- PPTX reopened and revised text verified.

## Repository synchronization result

**PASS.** The provenance-revised binaries are now installed at the canonical repository paths on `publication-v1`:

- `docs/WeaveCube_Technical_Report_v1.pdf`
  - bytes: 453,832
  - SHA-256: `f1e4acd09ac7399b5c532da5fd010ce10305870faf2bf5ff4307bcee3b263358`
  - Git blob: `b082875d8918129fc03371ca659a0f707f87792e`

- `presentation/WeaveCube_Core_Deck_v1.pptx`
  - bytes: 53,136
  - SHA-256: `3a0990c18177929d8536b3fb4c96752bf44495137a24964ba83f0424f9344bde`
  - Git blob: `cc69ccb57c936177b33a78c8e08aab6623cd76c5`

The temporary `*_provenance_revised` repository paths were removed after the same blobs were restored to the canonical names.

Next gate: build the sanitized history-free snapshot and run its final Windows/Python 3.14 checks.
