# WeaveCube worked example: one operational solution

> **Scope.** This is **FULL-24 Case 1** from the frozen v37.153/v37.527 operational configuration.  
> It is **not** the A18 signature-join mechanism trace in `DECISION_TRACE.md`.  
> The two examples come from different configurations and should not be combined into one execution narrative.

## Input

The sealed Case-1 scramble is a deterministic 25-move random-move word:

`F' U2 R D R D' L D L' R' L R' B2 U2 B2 L F2 D2 F U' D U' F' D B`

Historical case metadata:

- root Phase-1 lower bound: **8**
- selected offset: **+3**
- selected front-end length `H`: **11**
- exact completion length `K2`: **11**
- complete returned length: **22**
- historical case wall: **3.096544 s**

The historical timing is retained as an observation from the original frozen run. It is not a cold-start latency and is not replaced by the later solution-capture reproduction.

## Stage 1: build a complete front-end portfolio at H = 11

For this case the first tested offset, `root_lb + 3`, was sufficient.

The recorded portfolio pass is:

| Quantity | Case-1 value |
| --- | ---: |
| Prefix nodes | **82,554** |
| R4 leaf states | **1,654** |
| R4 words considered | **408** |
| Distinct raw endpoints | **240** |
| R4-cap truncation | **0** |
| Node/time cut | **none** |
| Portfolio wall | **1.412619 s** |

So the portfolio is recorded as **complete under the declared H and portfolio contract**, rather than as a resource-cut approximation.

## Stage 2: exact completion certification

The certification stage records:

- terminal calls: **207**
- certification wall: **1.683480 s**
- certification status: **certified**
- exact completion length: **K2 = 11**
- witness replay: **PASS**

The resulting complete solution length is therefore:

`H + K2 = 11 + 11 = 22`

## The actual 22-move solution

The historical JSON kept the replay result but did not serialize the complete word.  
A later frozen-policy reproduction recovered the actual solution and replayed it again natively.

### Front-end segment (H = 11)

`D2 L2 F' L' U D2 L' R' D' R2 L`

### Completion segment (K2 = 11)

`F2 U F2 U' R2 D2 L2 F2 U2 R2 D2`

### Complete solution

`D2 L2 F' L' U D2 L' R' D' R2 L F2 U F2 U' R2 D2 L2 F2 U2 R2 D2`

The reproduction record checks that:

- case index matches;
- scramble matches;
- selected H matches;
- exact K2 matches;
- total length matches;
- second native replay: **PASS**.

## Independent replay

The captured `scramble + solution` pair was also replayed in a separate fixed-frame **54-sticker integer-geometry implementation** that imports no CubeLab/WeaveCube module.

Case 1 result:

- valid move tokens: **PASS**
- solution length contract: **PASS**
- `H + K2 = total length`: **PASS**
- independent sticker-level solved state: **PASS**

The full reproduction set passed this separate replay for **24/24** captured pairs.

## What this example demonstrates

For this declared configuration and this particular sealed input, the operational code:

1. constructed a complete H=11 front-end portfolio under its recorded caps;
2. obtained an exact K2=11 completion;
3. produced a concrete 22-move solution;
4. replayed the witness successfully in the native implementation;
5. reproduced the solution later; and
6. passed an independent sticker-level replay implementation.

That is strong evidence that **this concrete solution is valid** and that the frozen research system was capable of producing real cube solutions.

It does **not** establish:

- reliable coverage of arbitrary legal states;
- optimal or consistently short solutions;
- computational scalability;
- competitiveness with Kociemba, Nissy, or other mature solvers;
- that the A18 mechanism experiment caused or constituted this FULL-24 execution.

## Evidence used

- sealed corpus: `SEALED_FULL24_CORPUS_v37_525.json`
- historical result: `RAW_v37_153_FULL24.json`
- historical wrapper: `presentation_run_sealed_full24_v37_527.py`
- captured solution: `FULL24_SOLUTIONS_CAPTURE_v4.json`
- independent replay: `FULL24_INDEPENDENT_REPLAY_AUDIT_v4.json`
- preprint: https://doi.org/10.5281/zenodo.22846520
- software archive: https://doi.org/10.5281/zenodo.22834879
