# EXP-0002: Typed-Memory Bridge

| Metadata | Value |
| --- | --- |
| Page ID | SM-EXP-0002 |
| Experiment ID | EXP-0002 |
| Confluence parent | Spatial Memory / Experiments |
| Status | Planned |
| Evidence level | Schema, writer, training, retrieval contract checks only |
| Last reviewed | 2026-07-11 |
| Depends on | EXP-0004 teacher artifact |

## Hypothesis

Trained spatial student checkpoint를 type-specific geometry record로 decode하고 같은
artifact를 actual-byte writer, retrieval, deterministic geometry executor, QA까지
내부 연결하면 checkpoint 변화가 evidence와 QA 결과에 추적 가능하게 반영된다.

## Linked claims, decisions, and papers

| Type | Link | Relevance |
| --- | --- | --- |
| Claim | [C-001: sparse geometry](../traceability.md) | 1 Hz 관측을 typed geometry candidate로 변환 |
| Claim | [C-002: bounded long-term memory](../traceability.md) | decoded record를 actual-byte budget 아래 선택 |
| Claim | [C-003: verifiable geometry QA](../traceability.md) | checkpoint 출력부터 proof까지 provenance 유지 |
| Claim | [C-005: actual-byte accounting](../traceability.md) | checkpoint-decoded record를 실제 byte budget으로 선택 |
| Decision | [ADR-0001: explicit typed memory](../decisions/adr-0001-explicit-typed-memory.md) | object, plane, portal, free-space, landmark, event schema |
| Decision | [ADR-0002: G-CUT3R as teacher](../decisions/adr-0002-gcut3r-as-teacher.md) | external teacher supervision으로 student 학습 |
| Decision | [ADR-0003: value per actual byte](../decisions/adr-0003-value-per-byte-writer.md) | decoded candidate를 실제 JSONL 비용으로 선택 |
| Decision | [ADR-0004: deterministic geometry proof](../decisions/adr-0004-deterministic-geometry-proof.md) | geometry answer를 typed record와 proof에 결속 |
| Paper context | [G-CUT3R](../papers/g-cut3r.md) | sparse-view guided teacher 후보; 아직 repo에서 재현되지 않음 |
| Paper context | [Point3R](../papers/point3r.md) | position-indexed explicit memory 근거 |
| Paper context | [ConceptGraphs](../papers/conceptgraphs.md) | object-centric explicit graph 근거 |

## Fixed contract

실행 전 dataset, split, code revision, teacher/checkpoint/config digest를 고정해야 한다.
현재 고정된 interface는 다음뿐이다.

| Item | Fixed value |
| --- | --- |
| Candidate types | object, plane, portal, free_space, landmark, event, no_write |
| Persistent types | object, plane, portal, free_space, landmark, event |
| Required record facts | entity and instance IDs, local frame, covariance, validity, first/last seen, confidence, provenance, evidence refs |
| Training input | materialized teacher rows with explicit train/validation split |
| Student outputs | record type, typed geometry target, association, uncertainty, rate, distillation |
| Writer | stable score-per-actual-UTF-8-JSONL-byte ordering |
| Persistence guard | `no_write` never serialized; duplicate IDs and invalid artifacts fail closed |
| Byte guard | serialized file size and recounted record bytes must equal summary and stay within budget |
| Retrieval guard | only causally eligible records from question video scope |
| Proof guard | answerable geometry choice requires a matching deterministic proof |
| Dataset, split, checkpoint, run ID | TBD before execution |

Current learned-lane boundary:

```text
external teacher cache and supervision
  -> materialized rows
  -> DDP typed candidate head
  -> spatial_student.pt
  -X-> type-specific decode and association
  -X-> typed artifact
  -X-> retrieval evidence and QA
```

## Compared variants

| Variant | Only changed factor | Inputs held constant |
| --- | --- | --- |
| A: Source-compact | Heuristic spatial records from EXP-0001 | split, 1 Hz frame manifest, retrieval, QA backend, byte accounting |
| B: Student typed bridge | Checkpoint-decoded typed records | split, 1 Hz frame manifest, retrieval, QA backend, matched artifact-byte budget |

External prebuilt `WORLDMM_QA_EVIDENCE_INPUT` cannot count as Variant B unless its
manifest binds the student checkpoint digest and the repository generated it from that
checkpoint.

## Metrics and go/no-go

| Metric or invariant | Go condition |
| --- | --- |
| Checkpoint traceability | checkpoint, decoder config, typed artifact, evidence, and QA manifest digests form one chain |
| Decode validity | every selected candidate validates as one typed schema or fails closed |
| Association validity | persistent IDs are unique and causal validity intervals do not conflict |
| Actual bytes | artifact file size does not exceed the matched configured budget |
| Persistence | zero `no_write` records in artifact |
| QA grounding | every answerable geometry prediction cites a matching proof and evidence ID |
| Leakage | 0 causal and off-scope evidence violations |
| Comparison | QA-Acc, QA-MRR, Ans-F1 and target spatial slices reported on identical inputs |

## Results

Not run.

Local checks currently prove only that typed schemas validate, the writer enforces exact
JSONL bytes, flat typed records can enter retrieval, teacher rows can be materialized,
and the DDP head can create a checkpoint. They do not test this hypothesis because no
checkpoint-to-typed-artifact inference bridge exists.

## Run provenance

| Item | Value |
| --- | --- |
| Run ID | Not assigned |
| Code revision | Not pinned |
| Student checkpoint | None |
| Decoder config and digest | None |
| Slurm job ID or process reference | None |
| Company artifact path | None |
| Metrics artifact | None |
| Copied locally | None |

## Conclusion

Pending. 현재 구현은 checkpoint에서 멈추므로 final learned-method reproduction으로
간주할 수 없다.

## Decision impact

Go이면 ADR-0001과 ADR-0003을 learned path에 적용 완료로 올리고 EXP-0003 Pareto
평가를 시작한다. No-go이면 새 abstraction을 추가하기 전에 decoder, association,
checkpoint-evidence digest chain 중 실패한 최소 경로만 수정한다.
