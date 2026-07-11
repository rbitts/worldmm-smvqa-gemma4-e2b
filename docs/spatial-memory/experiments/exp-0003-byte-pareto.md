# EXP-0003: QA Utility Versus Actual Bytes Pareto

| Metadata | Value |
| --- | --- |
| Page ID | SM-EXP-0003 |
| Experiment ID | EXP-0003 |
| Confluence parent | Spatial Memory / Experiments |
| Status | Planned |
| Evidence level | Not run |
| Last reviewed | 2026-07-11 |
| Depends on | EXP-0002 typed-memory bridge |

## Hypothesis

Learned typed writer가 같은 실제 persistent-byte 비용에서 source-compact heuristic보다
spatial QA utility를 높이고, 반복 방문 시 memory growth를 frame 수보다 새 geometry와
실제 change event에 가깝게 만든다.

## Linked claims, decisions, and papers

| Type | Link | Relevance |
| --- | --- | --- |
| Claim | [C-002: bounded long-term memory](../traceability.md) | QA utility와 실제 저장량의 trade-off 검증 |
| Claim | [C-004: unknown future questions](../traceability.md) | geometry core를 남긴 writer의 OOD 안전성 검증 |
| Claim | [C-005: actual-byte accounting](../traceability.md) | 실제 artifact byte에 대한 Pareto curve 측정 |
| Claim | [C-008: long-horizon evaluation](../traceability.md) | 반복 방문과 disjoint evidence memory growth 측정 |
| Claim | [C-011: novelty and redundancy](../traceability.md) | geometry novelty와 duplicate-removal baseline 비교 |
| Decision | [ADR-0001: explicit typed memory](../decisions/adr-0001-explicit-typed-memory.md) | record type별 저장 효율 비교 |
| Decision | [ADR-0003: value per actual byte](../decisions/adr-0003-value-per-byte-writer.md) | score-per-byte selection을 직접 검증 |
| Paper context | [LONG3R](../papers/long3r.md) | fixed-capacity retention 비교 근거 |
| Paper context | [MeMix](../papers/memix.md) | sparse update 비교 근거 |
| Paper context | [Point3R](../papers/point3r.md) | generic point-feature memory 비교 근거 |
| Paper context | [Geometry-aware token pruning](../papers/geometry-aware-token-pruning.md) | geometry redundancy 제거 근거; persistent-memory 재현은 아님 |

## Fixed contract

EXP-0002가 통과한 뒤 실행 직전에 아래 계약을 digest와 함께 확정한다.

| Item | Fixed value |
| --- | --- |
| Code revision | TBD |
| Official dataset and split | TBD |
| Dataset and annotation digests | TBD |
| 1 Hz sensor-frame manifest digest | TBD |
| Teacher and student checkpoint digests | TBD |
| Retrieval, QA model, and prompt schema | Identical across all variants; exact values TBD |
| Random seeds | TBD |
| Budget points | At least one lower, one matched, and one higher actual-byte point around source-compact; values fixed after baseline measurement |
| Budget normalization | actual persisted bytes/hour and total artifact bytes; numeric per-window and total caps are not treated as equivalent |
| Revisit sequences | unchanged revisit and known change-event revisit; exact IDs TBD |

## Compared variants

| Variant | Only changed factor | Inputs held constant |
| --- | --- | --- |
| A: No spatial | Spatial persistent bytes equal 0 | frames, non-spatial stores, retrieval, QA model |
| B: Source-compact | Heuristic spatial writer | frames, non-spatial stores, retrieval, QA model |
| C: Learned typed writer | EXP-0002 checkpoint and value-per-byte gate | frames, non-spatial stores, retrieval, QA model |

Raw dense features may be reported as a storage reference only. They are not a QA variant
unless the same retrieval and QA contract can consume them without changing other inputs.

## Metrics and go/no-go

| Metric or invariant | Scale | Go condition |
| --- | --- | --- |
| QA-Acc, QA-MRR, Ans-F1 | 0-100 | Report every budget point on identical labels |
| Spatial and geometry-question slices | 0-100 | Report separately; slice IDs fixed before run |
| Actual storage | bytes/hour and total bytes | Measured from persisted artifacts, not predicted token cost |
| Growth | bytes/new m², bytes/object, bytes/change | Inputs and denominators recorded |
| Revisit growth | bytes per unchanged and changed revisit | Unchanged revisit slope reported separately |
| Leakage | count | 0 causal violations |
| Pareto decision | dominance | Learned writer has at least one point with higher target QA at no more bytes, or fewer bytes at no lower target QA, than source-compact |

Official validity also requires the same split, frames, QA model, retrieval contract, and
checkpoint across matched comparisons. A point violating any invariant is excluded rather
than averaged into the curve.

## Results

Not run.

[EXP-0001](exp-0001-source-compact-baseline.md)의 6,050-byte tiny-fixture artifact는
pipeline sanity 한 점일 뿐 Pareto curve, device encoding, long-term growth, or official
benchmark 결과가 아니다.

## Run provenance

| Item | Value |
| --- | --- |
| Run ID family | Not assigned |
| Code revision | Not pinned |
| Budget manifest | None |
| Slurm job IDs or process references | None |
| Company artifact root | None |
| Metrics and plots | None |
| Copied locally | None |

## Conclusion

Pending. 현재 byte efficiency의 공식 결론은 없다.

## Decision impact

Go이면 ADR-0003을 benchmark-supported로 유지하고 선택된 operating point를 별도
deployment 결정에 기록한다. No-go이면 typed schema 자체를 확장하기 전에 learned
gate를 source-compact selector와 동일 candidate/budget 계약에서 비교한다.
