# EXP-0002: Typed-memory bridge

| 항목 | 값 |
| --- | --- |
| Page ID | SM-EXP-0002 |
| Experiment ID | EXP-0002 |
| Confluence parent | SM-EXPERIMENTS |
| 상태 | Local contract 구현, run 대기 |
| 근거 수준 | Production bridge와 lineage contract check만 완료 |
| 최종 검토 | 2026-07-12 |
| 선행 조건 | EXP-0004 teacher artifact |

## 핵심 결론

대기. Repository handoff는 checkpoint 이후 external production inference
bridge와 E1 reporting까지 연결되었지만 실행되지 않았다. Contract probe 통과 전에는
learned-method reproduction으로 간주할 수 없고, immutable E2/E3 identity가 없으므로
official E1/E2/E3 결과로도 간주할 수 없다.

Contract probe가 성공해도 result는 `contract_probe` / `PROBE`다. 별도 승인된
`full` run만 `student` / `E1`을 생성한다.

## 다음 결정

Go이면 ADR-0001과 ADR-0003을 learned path에 적용 완료로 올리고 EXP-0003 Pareto
평가를 시작한다. No-go이면 새 abstraction을 추가하기 전에 decoder, association,
checkpoint-evidence digest chain 중 실패한 최소 경로만 수정한다.

실행, 승인, artifact 경로의 canonical 절차는 repository root의
`HANDOFF.md`를 따른다. Confluence import 후에는 이 문서가
`SM-OPERATIONS` 하위의 `SM-OPERATIONS-HANDOFF` 페이지다.

## 근거

미실행.

Local check는 generated DAG가 contract version validation, inference input
sanitization, canonical typed JSONL/window byte 검사, 내부 evidence 생성, student
lineage·real frame 검증, profile-bound identity와 remote manifest/report 생성을
수행함을 입증한다. Production executable, checkpoint, model, frame set, company
benchmark run을 사용하지 않았으므로 가설 자체는 검증하지 않는다.

## 의사결정 gate

| Metric 또는 invariant | Go 조건 |
| --- | --- |
| Checkpoint traceability | Checkpoint, decoder config, typed artifact, evidence, QA manifest digest가 하나의 chain 구성 |
| Decode validity | 모든 selected candidate가 하나의 typed schema로 validate되거나 fail closed |
| Association validity | Persistent ID가 unique하고 causal validity interval이 충돌하지 않음 |
| Actual bytes | 모든 30초 window가 기본 4,096 byte 이하이며 canonical file size와 manifest total이 정확히 일치 |
| Persistence | Artifact의 `no_write` record 0 |
| QA grounding | 모든 answerable geometry prediction이 matching proof와 evidence ID 인용 |
| Leakage | Causal/off-scope evidence violation 0 |
| Comparison | 동일 input에서 QA-Acc, QA-MRR, Ans-F1, target spatial slice 보고 |

## 비교안

| Variant | 변경 요소 | 고정 input |
| --- | --- | --- |
| A: Source-compact | EXP-0001의 heuristic spatial record | Split, 1 Hz frame manifest, retrieval, QA backend, byte accounting |
| B: Student typed bridge | Contract-v1 checkpoint-decoded typed record | Split, 1 Hz frame manifest, retrieval, QA backend, matched per-window byte budget |

Prebuilt QA evidence는 Variant B로 인정하지 않는다. Production DAG는 repository
validation 이후 executable의 typed JSONL에서 evidence를 생성해야 한다.

## 가설

Trained spatial student checkpoint를 type-specific geometry record로 decode하고 같은
artifact를 actual-byte writer, retrieval, deterministic geometry executor, QA까지
내부 연결하면 checkpoint 변화가 evidence와 QA 결과에 추적 가능하게 반영된다.

## 실행 contract

실행 전 dataset, split, code revision, teacher/checkpoint/config digest를 고정해야 한다.
현재 production interface는 다음과 같이 고정되어 있다.

| 항목 | 고정값 |
| --- | --- |
| Candidate types | object, plane, portal, free_space, landmark, event, no_write |
| Persistent types | object, plane, portal, free_space, landmark, event |
| Required record fact | Entity/instance ID, local frame, covariance, validity, first/last seen, confidence, provenance, evidence ref |
| Training input | Explicit train/validation split이 있는 materialized teacher row |
| Student output | Record type, typed geometry target, association, uncertainty, rate, distillation |
| Inference executable | Contract version과 exact `worldmm-spatial-infer-v1:self-test-ok`; self-test는 accuracy가 아닌 CLI/schema/canonical writer 검사 |
| Inference input | Checkpoint, sanitized `inference_inputs/sources.jsonl`, copied selected `inference_inputs/frames/` root, sensor-frame manifest; question/label 없음 |
| Inference output | Canonical `typed_memory.jsonl`, `typed_memory.inference.json` |
| Adapter lineage | Adapter가 sanitized-source, frame-content-manifest, producer-executable SHA-256을 받고 manifest가 이를 echo하며 repository가 3개 모두 재계산 |
| Selection boundary | External executable이 candidate ranking/selection을 소유하고 repository는 persisted schema, canonical byte, budget만 validate |
| Persistence guard | Streaming validation, canonical row 최대 1 MiB, `no_write` serialize 금지, duplicate ID/invalid artifact fail closed |
| Grounding guard | Source video/record time이 source bound와 일치하고 grounded provenance의 bare same-video evidence min/max가 first/last seen과, unique count가 `observation_count`와 일치 |
| Window contract | `window_seconds=30.0`, validity backdating 방지를 위해 key는 `(source_video_id, floor(first_seen_time / 30.0))` |
| Byte guard | Window당 기본 4,096 byte, 모든 window와 total canonical file byte를 재계산해 manifest와 일치 |
| Retrieval guard | Question video scope의 causally eligible record만 허용 |
| Proof guard | Answerable geometry choice에 matching deterministic proof 필요 |
| QA guard | Spatial evidence가 canonical typed projection과 exact match; byte-budgeted record는 count/last-seen completeness 또는 label uniqueness를 certify할 수 없어 해당 query는 abstain; explicit-ID local-frame pair proof는 cross-video entity 거부; real frame과 v4 audit 필수 |
| Memory lineage | Student evidence에 memory-manifest와 episodic/semantic/visual SHA-256 기록; typed memory는 별도 checkpoint/inference-bound 유지; QA가 모든 referenced artifact 재계산 |
| Resume guard | QA v4 resume이 memory-manifest/evidence-lineage digest를 직접 bind하고 validated lineage가 individual non-spatial store byte를 transitive하게 bind |
| Result guard | Profile-neutral `metrics/metrics.json`, `summary/run_identity.json`; finalization input seal에 QA/lineage, memory manifest, episodic/semantic/visual/typed artifact, config, sensor, split input 포함; probe는 `contract_probe`/`PROBE`, full은 `student`/`E1` |
| Dataset, split, checkpoint, run ID | 실행 전 TBD 해소 |

현재 learned-lane 경계:

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

## 추적성

| 유형 | Link | 관련성 |
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

## 실행 provenance

| 항목 | 값 |
| --- | --- |
| Run ID | 미할당 |
| Code revision | 미고정 |
| Student checkpoint | 없음 |
| Decoder config/digest | 없음 |
| Slurm job ID 또는 process reference | 없음 |
| Company artifact path | 없음 |
| Metrics artifact | 없음 |
| 로컬 복사 | 없음 |
