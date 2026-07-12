# EXP-0002: Typed-Memory Bridge

| Metadata | Value |
| --- | --- |
| Page ID | SM-EXP-0002 |
| Experiment ID | EXP-0002 |
| Confluence parent | SM-EXPERIMENTS |
| Status | Local contract implemented; run pending |
| Evidence level | Production bridge and lineage contract checks only |
| Last reviewed | 2026-07-12 |
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
현재 production interface는 다음과 같이 고정되어 있다.

| Item | Fixed value |
| --- | --- |
| Candidate types | object, plane, portal, free_space, landmark, event, no_write |
| Persistent types | object, plane, portal, free_space, landmark, event |
| Required record facts | entity and instance IDs, local frame, covariance, validity, first/last seen, confidence, provenance, evidence refs |
| Training input | materialized teacher rows with explicit train/validation split |
| Student outputs | record type, typed geometry target, association, uncertainty, rate, distillation |
| Inference executable | contract version plus exact `worldmm-spatial-infer-v1:self-test-ok`; self-test checks CLI/schema/canonical writer, not accuracy |
| Inference inputs | checkpoint, sanitized `inference_inputs/sources.jsonl`, copied selected `inference_inputs/frames/` root, sensor-frame manifest; no questions or labels |
| Inference outputs | canonical `typed_memory.jsonl` and `typed_memory.inference.json` |
| Adapter lineage | adapter receives and manifest echoes sanitized-source, frame-content-manifest, and producer-executable SHA-256; repository recomputes all three |
| Selection boundary | external executable owns candidate ranking and selection; repository validates only persisted schema, canonical bytes, and budgets |
| Persistence guard | streaming validation; canonical row at most 1 MiB; `no_write` never serialized; duplicate IDs and invalid artifacts fail closed |
| Grounding guard | source video and record times match source bounds; grounded provenance requires bare same-video evidence whose min/max equal first/last seen and whose unique count equals `observation_count` |
| Window contract | `window_seconds=30.0`; key is `(source_video_id, floor(first_seen_time / 30.0))` to prevent validity backdating |
| Byte guard | default 4,096 bytes per window; every window and total canonical file bytes are recounted and matched to the manifest |
| Retrieval guard | only causally eligible records from question video scope |
| Proof guard | answerable geometry choice requires a matching deterministic proof |
| QA guard | spatial evidence exact-matches canonical typed projection; byte-budgeted records cannot certify count/last-seen completeness or label uniqueness, so production count/last-seen and label-only pair queries abstain; explicit-ID local-frame pair proofs reject cross-video entities; real frame and v4 audit required |
| Memory lineage | student evidence records memory-manifest plus episodic/semantic/visual SHA-256 values; typed memory remains separately checkpoint/inference-bound; QA recomputes every referenced artifact |
| Resume guard | QA v4 resume directly binds memory-manifest and evidence-lineage digests; validated lineage transitively binds individual non-spatial store bytes |
| Result guard | profile-neutral `metrics/metrics.json` and `summary/run_identity.json`; finalization input seal includes QA/lineage, memory manifest, episodic/semantic/visual/typed artifacts, config, sensor, and split inputs; probe is `contract_probe`/`PROBE`, full is `student`/`E1` |
| Dataset, split, checkpoint, run ID | TBD before execution |

Current learned-lane boundary:

```text
external teacher cache and supervision
  -> materialized rows
  -> DDP typed candidate head
  -> spatial_student.pt
  -> WORLDMM_SPATIAL_INFER_EXE
  -> type-specific decode, association, actual-byte selection
  -> validated canonical typed artifact
  -> repository-built retrieval evidence and real-frame QA
  -> profile-bound PROBE or learned E1 remote manifest and final report
```

## Compared variants

| Variant | Only changed factor | Inputs held constant |
| --- | --- | --- |
| A: Source-compact | Heuristic spatial records from EXP-0001 | split, 1 Hz frame manifest, retrieval, QA backend, byte accounting |
| B: Student typed bridge | Contract-v1 checkpoint-decoded typed records | split, 1 Hz frame manifest, retrieval, QA backend, matched per-window byte budget |

Prebuilt QA evidence cannot count as Variant B. The production DAG must build
evidence from the executable's typed JSONL after repository validation.

## Metrics and go/no-go

| Metric or invariant | Go condition |
| --- | --- |
| Checkpoint traceability | checkpoint, decoder config, typed artifact, evidence, and QA manifest digests form one chain |
| Decode validity | every selected candidate validates as one typed schema or fails closed |
| Association validity | persistent IDs are unique and causal validity intervals do not conflict |
| Actual bytes | every 30-second window is at most 4,096 bytes by default; canonical file size and manifest total match exactly |
| Persistence | zero `no_write` records in artifact |
| QA grounding | every answerable geometry prediction cites a matching proof and evidence ID |
| Leakage | 0 causal and off-scope evidence violations |
| Comparison | QA-Acc, QA-MRR, Ans-F1 and target spatial slices reported on identical inputs |

## Results

Not run.

Local checks prove that the generated DAG validates the contract version,
sanitizes inference inputs, checks canonical typed JSONL and per-window bytes,
builds evidence internally, verifies student lineage and real frames, and emits
a profile-bound identity plus remote manifest/report. They do not test the hypothesis:
no production executable, checkpoint, model, frame set, or company benchmark run
has been exercised.

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

Pending. Repository handoff는 checkpoint 이후 external production inference
bridge와 E1 reporting까지 연결되었지만 실행되지 않았다. Contract probe 통과 전에는
learned-method reproduction으로 간주할 수 없고, immutable E2/E3 identity가 없으므로
official E1/E2/E3 결과로도 간주할 수 없다.

Contract probe가 성공해도 result는 `contract_probe` / `PROBE`다. 별도 승인된
`full` run만 `student` / `E1`을 생성한다.

## Decision impact

Go이면 ADR-0001과 ADR-0003을 learned path에 적용 완료로 올리고 EXP-0003 Pareto
평가를 시작한다. No-go이면 새 abstraction을 추가하기 전에 decoder, association,
checkpoint-evidence digest chain 중 실패한 최소 경로만 수정한다.

실행, 승인, artifact 경로의 canonical 절차는 repository root의
`HANDOFF.md`를 따른다. Confluence import 후에는 이 문서가
`SM-OPERATIONS` 하위의 `SM-OPERATIONS-HANDOFF` 페이지다.
