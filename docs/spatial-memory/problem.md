# 문제 정의와 연구 질문

| 항목 | 값 |
|---|---|
| Page ID | SM-PROBLEM |
| 상태 | 활성 |
| 최종 갱신 | 2026-07-14 |
| 범위 | SuperMemory-VQA와 향후 AI-glass 배포 |

## 핵심 결론

Long-term memory를 explicit, typed, byte-bounded spatial record로 구축한다.
Dense geometry는 observation 처리 중에만 사용하고, metric answer에는
deterministic proof를 요구한다. 동일 actual-byte budget의 explicit baseline을
측정하기 전에는 generic latent quantization을 우선하지 않는다.

현재 Goal은 raw student 구현이 아니다. Offline teacher-oracle object/place record가
같은 byte budget에서 object/location QA에 실제 효용이 있는지 먼저 측정한다.
G-CUT3R는 이 oracle/target 생성에만 사용한다. Go 결과가 나온 target만 native
sensor 기반 hybrid device compiler로 distill한다.

## 필요성

AI glass는 약 1 Hz의 sparse RGB를 관측하고 제한된 저장 공간에서 이후의 위치,
거리, 방향, containment, reachability, change 질문에 답해야 한다. Frame, dense
map, recurrent-state snapshot은 시간에 따라 증가하며 직접 감사 가능한 evidence를
제공하지 않는다.

핵심 가설:

```text
sparse observations -> transient geometry -> typed records
    -> deterministic operation -> answer + proof
```

## 핵심 연구 질문

| ID | 제약 | 필요한 결과 |
|---|---|---|
| RQ-001 | Sparse, low-overlap sensing | Causal RGB와 available IMU/VIO 또는 depth guidance 결합 |
| RQ-002 | Lifelong storage | Frame 수가 아닌 새 장소·object·change에 비례하는 growth |
| RQ-003 | Geometry-grounded QA | Explicit record와 deterministic operator에서 metric answer 생성 |
| RQ-004 | 미지의 future question | Query-agnostic geometry core와 bounded evidence reserve 보존 |
| RQ-005 | Device model cost | Record contract와 server baseline 안정화 후 distillation |
| RQ-006 | Causality와 provenance | Future evidence 차단, observed/inferred geometry 구분 |

## RQ-001: Sparse sensing

Available camera calibration과 causal IMU/VIO/depth만 사용해 sparse RGB 사이 geometry를
구성한다. Missing signal은 합성하지 않는다.

## RQ-002: Lifelong storage

Persistent growth를 frame 수가 아닌 useful object, place, event, revisit에 묶고 actual
serialized byte로 측정한다.

## RQ-003: Explicit geometry-grounded QA

Metric과 last-location answer를 typed fact, deterministic operator, evidence-bound
proof에 연결한다.

## RQ-004: Unknown future question

Known query text에 overfit하지 않는 stable geometry core와 bounded evidence reserve를
평가한다.

## RQ-005: Device model cost

Teacher-oracle utility가 확인된 최소 target만 hybrid compiler로 distill하고 target
hardware에서 latency, memory, energy를 측정한다.

## RQ-006: Causality and provenance

Future input과 자기참조 evidence를 거부하고 observed, model-inferred,
relation-inferred provenance를 구분한다.

## 성공 gate

| 영역 | Gate |
|---|---|
| Correctness | Answerable geometry result마다 entity, frame, validity, uncertainty, provenance, evidence, proof 일치 |
| Safety | Unsupported/incomplete geometry는 abstain하고 causal violation은 0 |
| Compression | Hour·area·object·event·revisit당 actual byte와 QA-versus-bytes Pareto 보고 |
| Oracle | Evidence-bound teacher object/place record가 동일 byte baseline보다 target slice utility 개선 |
| Model | Oracle Go 이후 raw RGB semantics와 native sensor geometry에서 valid record 및 existing/`NEW` association 생성 |
| Evaluation | Matched variant의 split, data, frame, model, checkpoint, config, seed, prompt digest 고정 |

## 비목표

- Lifelong dense point map, Gaussian scene, recurrent-state snapshot 저장.
- Photorealistic reconstruction을 primary objective로 사용.
- Spatial-only language model 구축.
- Server-GPU throughput만으로 on-device 가능성 주장.
- Explicit actual-byte baseline 전에 learned codec 도입.

[프로젝트 홈으로 돌아가기](README.md)
