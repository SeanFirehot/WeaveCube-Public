# WeaveCube worked example: progressive exact restriction

> **Scope.** This is a mechanism example from **A18 v37.530, Case 0 (the first of eight planted-positive 5+5 cases)**.  
> It is **not** the FULL-24 operational solver trace shown in `SOLUTION_WALKTHROUGH.md`.  
> The two documents illustrate different experimental configurations and must not be read as stages of one execution.

## What this example is meant to show

A18 asks a narrow question: can one exact 10-move witness be reconstructed from two precomputed depth-5 word banks by joining exact signatures, **without materializing the full Cartesian product first**?

For this case:

- left word bank: **577,368** words
- right word bank: **577,368** words
- conceptual pair relation: **333,353,807,424** pairs
- candidate materialization threshold: **64 pairs**
- available exact piece conditions: **20**
- diagnostic construction: **planted-positive**; the planted IDs are hidden from the constructor and checked only post hoc
- final authority: **full-Q replay**

The hidden planted 10-move word for this diagnostic case was:

`B L2 D L' B' U2 F2 D' F B2`

The constructor did not read the planted word or its table IDs while applying the exact restrictions.

## Adaptive ordering: why piece 14 was tested first

Before building the cumulative join key, the query computes the exact **marginal compatible-pair count** for each of the 20 piece conditions. It then sorts those counts ascending.

For Case 0, the smallest one-condition marginal was:

- piece **14**: **7,805,234,752** compatible pairs

So the adaptive order begins:

`14 -> 8 -> 11 -> 13 -> 16 -> 19 -> 4 -> 5`

This is a query-specific ordering heuristic. **It is not correctness authority.** Correctness comes from exact signature equality and, finally, full-Q replay.

## Adaptive restriction ladder

| Step | Piece condition | Pairs before | Pairs after | Retained | Contraction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 14 | 333,353,807,424 | 7,805,234,752 | 2.3414% | 42.71x |
| 2 | 8 | 7,805,234,752 | 295,074,147 | 3.7805% | 26.45x |
| 3 | 11 | 295,074,147 | 11,530,124 | 3.9075% | 25.59x |
| 4 | 13 | 11,530,124 | 613,381 | 5.3198% | 18.80x |
| 5 | 16 | 613,381 | 55,543 | 9.0552% | 11.04x |
| 6 | 19 | 55,543 | 3,021 | 5.4390% | 18.39x |
| 7 | 4 | 3,021 | 326 | 10.7911% | 9.27x |
| 8 | 5 | 326 | 19 | 5.8282% | 17.16x |

The key event is Step 8:

- compatible relation falls from **326** to **19** pairs;
- only then, because `19 <= 64`, are the pairs explicitly materialized;
- **19** candidate pairs are materialized;
- **9** candidates are replayed in deterministic table order before a valid witness is found;
- final witness:

`B L2 D L' B' U2 F2 D' F B2`

- full-Q replay: **PASS**

The witness is the planted pair in this diagnostic case, but that identity is evaluated **post hoc**, not used to choose it.

## What if the exact same case uses a fixed piece order?

The fixed control uses the exact order:

`0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> ...`

Its ladder for the same Case 0 is:

| Step | Piece condition | Pairs before | Pairs after | Retained | Contraction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 333,353,807,424 | 13,902,179,648 | 4.1704% | 23.98x |
| 2 | 1 | 13,902,179,648 | 580,480,326 | 4.1755% | 23.95x |
| 3 | 2 | 580,480,326 | 35,903,298 | 6.1851% | 16.17x |
| 4 | 3 | 35,903,298 | 3,808,840 | 10.6086% | 9.43x |
| 5 | 4 | 3,808,840 | 379,349 | 9.9597% | 10.04x |
| 6 | 5 | 379,349 | 55,361 | 14.5937% | 6.85x |
| 7 | 6 | 55,361 | 2,284 | 4.1256% | 24.24x |
| 8 | 7 | 2,284 | 2,284 | 100.0000% | 1.00x |
| 9 | 8 | 2,284 | 30 | 1.3135% | 76.13x |

Notably, fixed Step 8 (piece 7) removes **zero** pairs: the relation stays at **2,284**.  
Piece 8 then reduces the relation to **30**, which is below the materialization cap.

Observed Case-0 comparison:

| | Adaptive A | Fixed B |
| --- | ---: | ---: |
| Exact conditions used | **8** | **9** |
| Pairs at materialization | **19** | **30** |
| Candidate replays before witness | **9** | **24** |
| Recorded query wall | **0.975 s** | **1.796 s** |
| Full-Q replay | PASS | PASS |
| Final witness | same | same |

The same witness is returned by both arms:

`B L2 D L' B' U2 F2 D' F B2`

## What should be concluded from this example?

This case makes the mechanism visible:

1. the full **333,353,807,424**-pair Cartesian relation exists conceptually;
2. exact conditions are accumulated over the **same pair identities**;
3. the compatible relation becomes progressively smaller;
4. explicit pair construction is delayed until the count is small;
5. a candidate is accepted only after full-Q replay.

It also shows that **ordering can matter to the amount of restriction work** on a particular query.

It does **not** show that:

- adaptive ordering is always better than fixed ordering;
- the method is complete for arbitrary cube states;
- the 333-billion conceptual relation was individually enumerated or visited;
- the method is computationally scalable in general;
- this A18 diagnostic is the operational FULL-24 solver.

Across the full eight planted-positive v37.530 cases, both ordering policies found 8/8 witnesses. The reported median exact-condition counts were 8 for Adaptive A, 10 for Fixed B, and 8 for the repeated Adaptive C control. That remains a small, constructed mechanism experiment rather than a general coverage result.

## Evidence used

- `reports/presentation/a18_ordering_ablation_v37_530/adaptive_A.json`
- `reports/presentation/a18_ordering_ablation_v37_530/fixed_B.json`
- `scripts/audit_a18_split_exact_signature_join_v37_460_a18_p1_win.py`
- preprint: https://doi.org/10.5281/zenodo.22846520
- software archive: https://doi.org/10.5281/zenodo.22834879
