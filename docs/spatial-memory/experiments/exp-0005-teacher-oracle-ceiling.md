# EXP-0005: Causal teacher-oracle object/location ceiling

| 항목 | 값 |
| --- | --- |
| Page ID | SM-EXP-0005 |
| Experiment ID | EXP-0005 |
| Confluence parent | SM-EXPERIMENTS |
| 상태 | Local contract 구현; company run 대기 |
| 근거 수준 | Tiny contract tests only; benchmark 미실행 |
| 최종 검토 | 2026-07-14 |
| 선행 조건 | Sensor availability audit와 EXP-0004 provider validation |

## 핵심 결론

**대기.** Raw student를 학습하기 전에 offline teacher가 만든 causal
object/place record가 동일 actual-byte budget에서 object/location QA를 개선하는지
측정한다. 이 결과는 teacher-oracle ceiling이며 on-device 또는 student result가
아니다.

## 다음 결정

Go이면 useful target만 [EXP-0002](exp-0002-typed-memory-bridge.md)의 hybrid device
student로 distill한다. No-go이면 raw encoder나 all-type decoder를 추가하지 않고
object semantics, place assignment, association, source signal 중 실패 원인을 먼저
분리한다.

## 근거

미실행.

Local code는 independent camera calibration, trusted causal pose, selected
teacher-point object compiler, inferred evidence/confidence gate, semantic place projection,
`last_location` proof를 검사한다. Real model, dataset frame, teacher output, QA metric은
사용하지 않았다.

## 의사결정 gate

| Metric 또는 invariant | Scale | Go 조건 |
| --- | --- | --- |
| Sensor audit | coverage | Selected RGB asset 100%; calibration/pose/depth availability를 실제 비율로 보고하고 누락 신호를 합성하지 않음 |
| Object/location QA | QA-Acc, QA-MRR, Ans-F1 | Run 전 최소 개선폭과 confidence interval 고정 |
| Selective risk | error/coverage | Unsupported·low-confidence place는 abstain; accepted proof error 상한 사전 고정 |
| Typed validity | count | Invalid record, duplicate ID, persisted no-write 0 |
| Grounding | count | Missing/off-scope evidence, future validity, unbound inferred proof 0 |
| Actual serialized bytes | bytes | Variant별 동일 30초 window budget; 기본 4,096 byte 이하 |
| Causal violations | count | 0 |

## 비교안

| Variant | 변경 요소 | 고정 input |
| --- | --- | --- |
| E0: Source-compact | 기존 heuristic spatial memory | Split, selected 1 Hz RGB, non-spatial stores, retrieval, QA backend, byte budget |
| T0: Teacher object geometry | Offline teacher object centroid/extent/uncertainty | E0 고정 input과 exact byte budget |
| T1: Teacher object + place | T0에 evidence-bound semantic `place_label`과 `last_location` 추가 | E0 고정 input과 exact byte budget |

Prebuilt answer text, question, choice, label, evidence annotation은 teacher target이나
memory writer 입력으로 사용하지 않는다.

## 가설

동일 causal frame inventory와 serialized-byte budget에서 evidence-bound offline
teacher object/place record는 source-compact baseline보다 object/location slice의
QA utility를 높이며 unsupported case의 abstention을 유지한다.

## 실행 contract

| 항목 | 고정값 |
| --- | --- |
| Execution location | 명시적 승인 후 company GPU/CPU resources |
| Code revision | TBD |
| Dataset과 split | TBD; official object/location slice를 실행 전 고정 |
| Dataset, source, question, label digest | TBD |
| Sensor-frame manifest와 frame-asset digest | TBD |
| Camera intrinsics, pose, depth coverage report | TBD; available signal만 사용 |
| Teacher provider, code, checkpoint, config digest | TBD |
| Semantic mask/place provider와 ontology digest | TBD |
| Random seed | TBD |
| Byte-budget scope와 값 | 30초 source window당 기본 4,096 byte; run 전 고정 |
| QA backend와 prompt schema | TBD |
| Result class | `teacher_oracle`; `student`/`E1` 사용 금지 |
| Run ID | TBD |

Prepared source에 camera calibration, readable RGB asset, 필요한 pose/depth가 없으면
해당 modality를 추정값으로 채우지 않는다. Metric geometry가 측정 불가능하면 결과를
No-Go 또는 not measurable로 기록한다.

## 추적성

| 유형 | Link | 관련성 |
| --- | --- | --- |
| Claim | [C-001: sparse geometry](../traceability.md) | Offline teacher geometry ceiling 측정 |
| Claim | [C-003: verifiable geometry QA](../traceability.md) | Object/place record에서 deterministic proof 생성 |
| Claim | [C-006: transient/persistent separation](../traceability.md) | Dense teacher output은 transient, typed record만 persistent |
| Claim | [C-009: provenance and abstention](../traceability.md) | Inferred evidence/confidence/causality gate 검증 |
| Decision | [ADR-0002: G-CUT3R teacher](../decisions/adr-0002-gcut3r-as-teacher.md) | Large model은 offline oracle로 제한 |
| Decision | [ADR-0005: hybrid device compiler](../decisions/adr-0005-hybrid-on-device-compiler.md) | Student 투자 전 oracle utility gate |
| Decision | [ADR-0006: inferred geometry](../decisions/adr-0006-evidence-bound-inferred-geometry.md) | Evidence-bound inferred proof admission |
| Paper | [SuperMemory-VQA](../papers/supermemory-vqa.md) | Object/location long-horizon QA context |

## 실행 provenance

| 항목 | 값 |
| --- | --- |
| Run ID | 미할당 |
| Code revision | 미기록 |
| Slurm job ID 또는 process reference | 없음 |
| Company artifact path | 없음 |
| Metrics artifact | 없음 |
| Log | 없음 |
| 로컬 복사 | 없음 |
