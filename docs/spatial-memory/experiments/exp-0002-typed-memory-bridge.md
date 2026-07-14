# EXP-0002: Minimal hybrid student typed-memory bridge

| 항목 | 값 |
| --- | --- |
| Page ID | SM-EXP-0002 |
| Experiment ID | EXP-0002 |
| Confluence parent | SM-EXPERIMENTS |
| 상태 | 재설계; EXP-0005 Go 대기 |
| 근거 수준 | Legacy feature-head와 lineage contract local check만 완료 |
| 최종 검토 | 2026-07-14 |
| 선행 조건 | EXP-0005 teacher-oracle object/location utility Go |

## 핵심 결론

**대기.** 기존 supplied-vector candidate head는 raw RGB/IMU student도, open-world
typed decoder도, on-device model도 아니다. EXP-0005가 useful target을 확인하기
전에는 이 scaffold를 production student result로 실행하지 않는다.

## 다음 결정

EXP-0005 Go이면 object/place/event 중 실제 QA utility를 만든 최소 target만 선택하고,
small semantic perception + native causal VIO/depth + deterministic projection으로
student input과 output contract를 다시 고정한다. No-go이면 student architecture를
추가하지 않는다.

## 근거

미실행.

Local check는 feature-level candidate head, DDP loss/checkpoint, external inference
lineage, canonical typed JSONL/window-byte validation이 동작함만 입증한다. Image
encoder, sensor fusion, mask-to-geometry, open-world association, mobile executable,
real checkpoint는 없다.

## 의사결정 gate

| Metric 또는 invariant | Scale | Go 조건 |
| --- | --- | --- |
| Oracle retention | object/location slice | EXP-0005 T1 gain의 사전 정의 비율 유지 |
| QA-Acc, QA-MRR, Ans-F1 | 0-100 | E0/T1과 matched input·byte로 보고 |
| Association | precision/recall | Existing/NEW, false merge, duplicate ID 목표를 run 전 고정 |
| Typed validity | count | Invalid record, persisted no-write, duplicate ID 0 |
| Grounding | count | Missing/off-scope evidence와 low-confidence accepted proof 0 |
| Actual serialized bytes | bytes | 동일 30초 window budget; 기본 4,096 byte 이하 |
| Device profile | latency, memory, energy | Target hardware와 허용값을 run 전 고정; server latency 대체 금지 |
| Causal violations | count | 0 |

## 비교안

| Variant | 변경 요소 | 고정 input |
| --- | --- | --- |
| E0: Source-compact | Existing heuristic spatial memory | Split, frame inventory, non-spatial stores, retrieval, QA, byte budget |
| T1: Teacher oracle | EXP-0005의 object/place teacher record | E0 고정 input |
| S1: Hybrid student | Small semantic encoder + native geometry + minimal typed writer | E0 고정 input |

`spatial_train.py`의 supplied feature vector를 raw RGB처럼 보고하는 variant는 허용하지
않는다. 모든 candidate type을 고정 dimension으로 생성하는 legacy head도 S1 정의가
아니다.

## 가설

Offline teacher 전체를 복제하지 않고 object/place semantics만 작은 perception model로
distill하며 calibrated native VIO/depth로 geometry를 계산하면, 동일 byte budget에서
teacher-oracle QA gain의 유의미한 부분을 target device budget 안에 유지할 수 있다.

## 실행 contract

| 항목 | 고정값 |
| --- | --- |
| Architecture | EXP-0005 Go target만; [ADR-0005](../decisions/adr-0005-hybrid-on-device-compiler.md) 준수 |
| Training input | Raw selected RGB + available causal native sensor signal; question/label 없음 |
| Teacher use | Offline target generation only; inference executable에서 G-CUT3R load 금지 |
| Output | Evidence-bound object/place/event/no-write 중 승인된 최소 subset |
| Association | Existing entity pointer 또는 `NEW`; fixed closed-set ID 분류 금지 |
| Code revision | TBD |
| Dataset, split, input digests | TBD |
| Teacher/student/model/config digests | TBD |
| Sensor/frame/calibration digests | TBD |
| Random seed | TBD |
| Byte-budget scope와 값 | 30초 source window당 기본 4,096 byte; run 전 고정 |
| Target device와 profile method | TBD |
| QA backend와 prompt schema | TBD |
| Run ID | TBD |

Checkpoint, executable, typed memory, evidence, QA, metric, report digest chain은 기존
production handoff guard를 재사용한다. EXP-0005 이전 legacy checkpoint probe는
architecture validation이 아니라 scaffold diagnostic으로만 기록한다.

## 추적성

| 유형 | Link | 관련성 |
| --- | --- | --- |
| Claim | [C-001: sparse geometry](../traceability.md) | Native guidance와 semantic perception 결합 |
| Claim | [C-002: bounded long-term memory](../traceability.md) | Minimal typed output과 hard byte budget |
| Claim | [C-003: verifiable geometry QA](../traceability.md) | Student output에서 deterministic proof까지 결속 |
| Claim | [C-005: actual-byte accounting](../traceability.md) | Oracle/student를 동일 serialized-byte budget으로 비교 |
| Claim | [C-006: transient/persistent separation](../traceability.md) | Teacher와 device runtime 분리 |
| Claim | [C-009: provenance and abstention](../traceability.md) | Inferred evidence/confidence gate 유지 |
| Decision | [ADR-0001: explicit typed memory](../decisions/adr-0001-explicit-typed-memory.md) | Persistent output contract |
| Decision | [ADR-0003: value per actual byte](../decisions/adr-0003-value-per-byte-writer.md) | Hard byte writer |
| Decision | [ADR-0005: hybrid device compiler](../decisions/adr-0005-hybrid-on-device-compiler.md) | Student architecture boundary |
| Decision | [ADR-0006: inferred geometry](../decisions/adr-0006-evidence-bound-inferred-geometry.md) | Proof admission boundary |
| Experiment | [EXP-0005: teacher oracle](exp-0005-teacher-oracle-ceiling.md) | Distillation 전 utility ceiling |

## 실행 provenance

| 항목 | 값 |
| --- | --- |
| Run ID | 미할당 |
| Code revision | 미고정 |
| Student checkpoint | 없음 |
| Slurm job ID 또는 process reference | 없음 |
| Company artifact path | 없음 |
| Metrics artifact | 없음 |
| 로컬 복사 | 없음 |
