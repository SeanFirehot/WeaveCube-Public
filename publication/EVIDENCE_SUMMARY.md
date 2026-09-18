# Evidence Summary

## E1 - A18 v37.530
- Split: 5+5
- Conceptual pairs: 333,353,807,424
- Adaptive A / Fixed / Adaptive C: 8/8 / 8/8 / 8/8
- Median conditions: 8 / 10 / 8
- Median query wall: 1.7471 / 1.9714 / 1.7894 s
- Only semantic delta in the ablation: condition order

## E2 - FIRST_WITNESS v37.507
- Full frozen A: 36.9212 s
- FIRST_WITNESS: 15.4988 s
- Full frozen C: 36.8037 s
- Full geometric mean: 36.8624 s
- Hot speedup: 2.378x
- Product states / edges: 1,954,520 / 1,958,916
- Native replay: PASS

## E3 - C4 v37.516 revalidation
- Historical A: 18.638914 s
- Optimized policy: 19.034902 s
- Historical B: 18.288457 s
- Historical GM: 18.462854 s
- Deficit: +0.572048 s
- Correctness/native/authority gates: PASS
- Dominant profile target: `common_mask`

## E4 - ATO theorem v37.470
- Half-turn orbit size: 663,552
- Number of pairwise-disjoint physical ATO-zero orbits: 6
- Total physical ATO-zero states: 3,981,312
- Proof style: constructive lower bound equals independent invariant upper bound

## E5 - FULL-24 v37.527
- Sealed 25-move deterministic random-move corpus
- Corpus SHA-256: `77bd96782994e5ae41e98ebab013de42354178bb9589e043dc12c4e416fee88e`
- Solved: 24/24
- Replay: 24/24 PASS
- Median / p90 / max case wall: 1.224 / 3.017 / 4.914 s
- Solution length: 18-23, median 21
- UNKNOWN / fallback / replay failure: 0 / 0 / 0
