# EXP-0003: QA utility-actual byte Pareto

| 항목 | 값 |
| --- | --- |
| Page ID | SM-EXP-0003 |
| Experiment ID | EXP-0003 |
| Confluence parent | SM-EXPERIMENTS |
| 상태 | 계획 |
| 근거 수준 | 미실행 |
| 최종 검토 | 2026-07-11 |
| 선행 조건 | EXP-0002 typed-memory bridge |

## 핵심 결론

대기. 현재 byte efficiency에 대한 공식 결론은 없다.

## 다음 결정

Go이면 ADR-0003을 benchmark-supported로 유지하고 선택된 operating point를 별도
deployment 결정에 기록한다. No-go이면 typed schema 자체를 확장하기 전에 learned
gate를 source-compact selector와 동일 candidate/budget 계약에서 비교한다.

## 근거

미실행.

[EXP-0001](exp-0001-source-compact-baseline.md)의 6,050-byte tiny-fixture artifact는
pipeline sanity 한 점일 뿐 Pareto curve, device encoding, long-term growth, official
benchmark 결과가 아니다.

## 의사결정 gate

| Metric 또는 invariant | Scale | Go 조건 |
| --- | --- | --- |
| QA-Acc, QA-MRR, Ans-F1 | 0-100 | 동일 label에서 모든 budget point 보고 |
| Spatial/geometry-question slice | 0-100 | 별도 보고하며 run 전 slice ID 고정 |
| Actual storage | bytes/hour와 total bytes | Predicted token cost가 아닌 persisted artifact에서 측정 |
| Growth | bytes/new m², bytes/object, bytes/change | Input과 denominator 기록 |
| Revisit growth | unchanged/changed revisit당 byte | Unchanged revisit slope 별도 보고 |
| Leakage | count | Causal violation 0 |
| Pareto decision | dominance | Learned writer가 source-compact보다 같은 이하 byte에서 target QA가 높거나 같은 이상 QA에서 byte가 적은 point를 최소 1개 보유 |

Official validity에는 matched comparison 전반의 동일 split, frame, QA model,
retrieval contract, checkpoint가 필요하다. Invariant를 위반한 point는 curve에
평균하지 않고 제외한다.

## 비교안

| Variant | 변경 요소 | 고정 input |
| --- | --- | --- |
| A: No spatial | Spatial persistent byte 0 | Frame, non-spatial store, retrieval, QA model |
| B: Source-compact | Heuristic spatial writer | Frame, non-spatial store, retrieval, QA model |
| C: Learned typed writer | EXP-0002 checkpoint와 value-per-byte gate | Frame, non-spatial store, retrieval, QA model |

Raw dense feature는 storage reference로만 보고할 수 있다. 다른 input 변경 없이
동일 retrieval/QA contract가 consume할 수 있을 때만 QA variant로 인정한다.

## 가설

Learned typed writer가 같은 실제 persistent-byte 비용에서 source-compact heuristic보다
spatial QA utility를 높이고, 반복 방문 시 memory growth를 frame 수보다 새 geometry와
실제 change event에 가깝게 만든다.

## 실행 contract

EXP-0002가 통과한 뒤 실행 직전에 아래 계약을 digest와 함께 확정한다.

| 항목 | 고정값 |
| --- | --- |
| Code revision | TBD |
| Official dataset과 split | TBD |
| Dataset/annotation digest | TBD |
| 1 Hz sensor-frame manifest digest | TBD |
| Teacher/student checkpoint digest | TBD |
| Retrieval, QA model, prompt schema | 모든 variant에서 동일, exact value TBD |
| Random seeds | TBD |
| Budget point | Source-compact 주변의 lower/matched/higher actual-byte point를 최소 1개씩 두고 baseline 측정 후 값 고정 |
| Budget normalization | Actual persisted bytes/hour와 total artifact byte; numeric per-window cap과 total cap을 동등하게 취급하지 않음 |
| Revisit sequence | Unchanged revisit와 known change-event revisit, exact ID TBD |

## 추적성

| 유형 | Link | 관련성 |
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

## 실행 provenance

| 항목 | 값 |
| --- | --- |
| Run ID family | 미할당 |
| Code revision | 미고정 |
| Budget manifest | 없음 |
| Slurm job ID 또는 process reference | 없음 |
| Company artifact root | 없음 |
| Metric/plot | 없음 |
| 로컬 복사 | 없음 |
