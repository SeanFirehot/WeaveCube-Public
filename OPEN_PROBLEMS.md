# Open Problems and Future Research

WeaveCube is a **feasibility demonstration** of exact shared-word construction for Rubik's Cube solving. The documented experiments show that the construction can be implemented and can produce verified solutions in the tested settings.

They do **not** establish general solver reliability across arbitrary legal cube states, computational scalability, consistently short or optimal solutions, or competitive performance relative to mature solvers such as Kociemba and Nissy. The questions below are research directions, not established capabilities or promised results.

- **Paper:** [WeaveCube: A Feasibility Study of Exact Shared-Word Construction for Rubik's Cube Solving](https://doi.org/10.5281/zenodo.22846520)
- **Archived software:** [WeaveCube v1.0 Research Release](https://doi.org/10.5281/zenodo.22834879)

## Evidence boundaries

Historical observations, recovered scripts/assets/receipts, and newly executed reproduction checks must remain distinguishable. A18 signature joins, FIRST_WITNESS closure experiments, later integration tests, and the frozen operational driver are different configurations; they do not constitute one demonstrated end-to-end benchmark.

The reproduced FULL-24 results and independent replay checks establish validity for those 24 scramble-solution pairs. They do not establish arbitrary-state coverage, and they do not transfer historical timings to the reproduction run. Local closure improvements do not establish an end-to-end speedup when preparation, memory, and other costs are included.

A verified witness establishes success for its declared objective. Exhaustive failure applies only to the declared candidate language and context. A time or memory limit must be reported as unresolved, not as proof that no solution exists.

## 1. Coverage

**Question:** Which legal cube states and candidate-language configurations can the method solve reliably under stated resource limits?

Evaluate fresh, held-out uniform random legal states as well as separately labeled random-move scrambles, structured cases, and difficult cases. Report sample construction, seeds, sample sizes, solved fractions, uncertainty, and every timeout or failure. Investigate coverage gaps as word-bank size, split, candidate length, and terminal conditions vary; distinguish empirical coverage from any completeness proof.

## 2. Scaling

**Question:** How do total time and resource requirements grow with candidate length, word-bank size, constraint count, and instance difficulty?

Measure preparation, loading, restriction, materialization, witness construction, completion, and verification separately and in total. Report cold and warm runs, peak memory, and repeated-run variation. Compare conceptual pair counts with actual stored data and performed work; a large implicitly represented relation alone is not evidence of computational scalability.

## 3. Constraint scheduling

**Question:** When does adaptive ordering of exact restrictions improve total cost over fixed or simpler schedules?

Run controlled ablations on identical held-out instances and candidate languages. Account for the cost of selecting a constraint as well as applying it. Compare selectivity, evaluation cost, surviving candidates, total wall time, and success under equal budgets. Scheduling and stopping rules must preserve the declared correctness scope and distinguish verified success from unresolved computation.

## 4. Representation and memory

**Question:** Which representations preserve shared-word identity and reconstructible witnesses with practical memory use?

Compare explicit word banks, signatures, bitmaps, factorized relations, decision diagrams, and streaming or compressed representations where applicable. Measure peak resident memory, temporary allocations, cache size, construction time, and witness-reconstruction cost. Test whether deduplication and compression preserve boundary context and information required by later restrictions.

## 5. Solution length

**Question:** Can witness selection and completion produce consistently shorter solutions without unacceptable increases in time or memory?

Declare the move metric and report the full distribution of complete solution lengths, including tails. Measure front-end and completion contributions separately, and compare first-witness policies with alternatives under equal budgets. Do not label solutions optimal without an appropriate proof or independent optimality certificate.

## 6. Parallelization

**Question:** Which stages benefit from parallel execution after communication, synchronization, and memory costs are included?

Explore word-bank construction, signature evaluation, independent partitions, and witness queries. Measure speedup, efficiency, aggregate resource use, and memory duplication against a stated serial baseline. Preserve deterministic case identities and replay-verifiable outputs, and examine how parallel ordering changes first-witness latency and solution length.

## 7. Broader evaluation

**Question:** How does the method behave beyond the current small, specialized experimental settings?

Use larger held-out corpora, multiple seeds, varied hardware, and repeated runs. Compare against clearly versioned Kociemba and Nissy configurations with disclosed move metrics, thread counts, resource limits, initialization costs, and optimization objectives. Report all outcomes, including failures and unfavorable results; speed and solution quality must be considered together.

Publish sufficient scripts, asset hashes, environment details, raw results, and complete solution words for independent reproduction. Retain an independent replay implementation, and distinguish implementation verification from claims about coverage or performance.

## Contributing evidence

A useful follow-up should state its hypothesis, exact solver configuration, corpus and sampling method, resource budget, baselines, and acceptance criteria before interpreting the results. Include negative results and limits on generalization. These research directions do not alter the archived `v1.0-research` release or its solver code.
