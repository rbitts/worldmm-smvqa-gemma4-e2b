# 근거-구현 추적성

| 항목 | 값 |
|---|---|
| Page ID | SM-TRACEABILITY |
| 상태 | 갱신형 index |
| 최종 갱신 | 2026-07-15 |
| 목적 | 문제, 근거, 결정, 구현, 실험, 결과 연결 |

## 핵심 요약

| 분류 | Claim | 결정 |
|---|---|---|
| 로컬 contract 근거 있음 | C-002, C-003, C-005, C-008, C-009 | Baseline 유지, benchmark 근거로 사용 금지 |
| 차단 또는 일부 구현 | C-001, C-006 | Sensor audit와 teacher oracle에서 검증 |
| 연구 또는 계획 | C-004, C-007, C-011 | P0 이후 평가 |
| 보류 | C-010, C-012 | Explicit baseline 측정 전 투자 금지 |

현재 architecture 방향은 기존 근거와 일치하지만 benchmark-verified claim은 없다.
Real sensor availability와 teacher-oracle QA utility가 즉시 해결해야 할 blocker다.

## 판단 규칙

각 row는 paper claim이 아닌 project claim이다. Paper page는 저자가 실제 평가한
조건을 기록하고, ADR은 project decision, experiment는 project result를 기록한다.
`차단` 또는 `연구` row를 구현 근거로 설명하지 않는다.

## 연구 질문-claim 연결

| 연구 질문 | Project claim | 현재 해석 |
|---|---|---|
| [RQ-001: Sparse sensing](problem.md#rq-001-sparse-sensing) | C-001, C-006, C-007, C-008 | Guidance, working/persistent state 분리, wide-baseline association, 단일 causal frame inventory |
| [RQ-002: Lifelong storage](problem.md#rq-002-lifelong-storage) | C-002, C-004, C-005, C-006 | Bounded typed record, unknown-question reserve, actual-byte accounting, transient state 분리 |
| [RQ-003: Explicit geometry-grounded QA](problem.md#rq-003-explicit-geometry-grounded-qa) | C-003, C-007, C-009, C-010 | Deterministic proof, association, trusted provenance, explicit-versus-latent sufficiency |
| [RQ-004: 미지 future question](problem.md#rq-004-unknown-future-question) | C-004, C-010, C-011 | Stable geometry core, bounded reservoir, coverage-preserving novelty baseline |
| [RQ-005: Device model cost](problem.md#rq-005-device-model-cost) | C-005, C-006, C-012 | Actual-byte objective, teacher/student state 분리, value quantization 보류 |
| [RQ-006: Causality와 provenance](problem.md#rq-006-causality-and-provenance) | C-003, C-008, C-009 | Proof 일치, causal frame inventory, lineage, uncertainty, abstention |

| ID | 문제 또는 project claim | 논문 근거 | 결정 | 구현 | 실험과 결과 | 상태 |
|---|---|---|---|---|---|---|
| C-001 | Sparse 1 Hz RGB의 stable low-overlap geometry에는 pose/depth guidance가 필요하다. | [G-CUT3R](papers/g-cut3r.md), [Mono-Hydra++](papers/mono-hydra-plus-plus.md), [Depth Anything V2](papers/depth-anything-v2.md), [UniDepth V2](papers/unidepth-v2.md), [CUT3R](papers/cut3r.md), [VGGT](papers/vggt.md) | [ADR-0002](decisions/adr-0002-gcut3r-as-teacher.md), [ADR-0005](decisions/adr-0005-hybrid-on-device-compiler.md) | `spatial_sensor.py`, `gcut3r_teacher.py`, `spatial_teacher_targets.py` local contract; real provider 없음 | [EXP-0002](experiments/exp-0002-typed-memory-bridge.md), [EXP-0004](experiments/exp-0004-gcut3r-provider.md), [EXP-0005](experiments/exp-0005-teacher-oracle-ceiling.md) 미실행 | 차단 |
| C-002 | Lifelong memory는 dense point map/recurrent snapshot 대신 bounded explicit typed record를 저장해야 한다. | [Point3R](papers/point3r.md), [ConceptGraphs](papers/conceptgraphs.md), [HOV-SG](papers/hov-sg.md), [FARM](papers/farm.md), [LONG3R](papers/long3r.md), [Memory-Centric EQA](papers/memory-centric-embodied-question-answering.md), [MeMix](papers/memix.md), [Spann3R](papers/spann3r.md) | [ADR-0001](decisions/adr-0001-explicit-typed-memory.md) | `typed_memory.py`, `spatial.py` | [EXP-0001](experiments/exp-0001-source-compact-baseline.md): 216 record/96,456 byte 대신 15 record/6,050 byte, 15.94x smaller; [EXP-0002](experiments/exp-0002-typed-memory-bridge.md), [EXP-0003](experiments/exp-0003-byte-pareto.md), [EXP-0004](experiments/exp-0004-gcut3r-provider.md) 미실행 | Heuristic 로컬 검증 완료; oracle/student 대기 |
| C-003 | Geometry answer에는 deterministic operation과 proof-to-answer 일치가 필요하다. | [GraphEQA](papers/grapheqa.md), [OpenEQA](papers/openeqa.md), [SQA3D](papers/sqa3d.md), [DAAAM](papers/daaam.md), [ScanQA](papers/scanqa.md), [VSI-Bench](papers/vsi-bench.md) | [ADR-0004](decisions/adr-0004-deterministic-geometry-proof.md), [ADR-0006](decisions/adr-0006-evidence-bound-inferred-geometry.md) | `geometry_executor.py`, `qa.py`: evidence/confidence-gated `last_location` 포함 | [EXP-0001](experiments/exp-0001-source-compact-baseline.md) local sanity; [EXP-0002](experiments/exp-0002-typed-memory-bridge.md), [EXP-0005](experiments/exp-0005-teacher-oracle-ceiling.md) 미실행 | 로컬 contract 검증 완료 |
| C-004 | 미지 future question에는 stable geometry core와 small bounded evidence reservoir가 필요하다. | [GraphEQA](papers/grapheqa.md), [FOUND-IT](papers/found-it.md), [Worth Remembering](papers/surprise-gated-robot-episodic-memory.md), [DAAAM](papers/daaam.md), [Memory-Centric EQA](papers/memory-centric-embodied-question-answering.md) | 채택된 reservoir ADR 없음 | Selector preparation에 surprise feature가 있으나 persistent visual reservoir 없음 | [EXP-0003](experiments/exp-0003-byte-pareto.md) 미실행 | 연구 |
| C-005 | Writer decision과 비교는 token count가 아닌 actual serialized byte를 사용해야 한다. | [LONG3R](papers/long3r.md), [MeMix](papers/memix.md), [rate-distortion](papers/end-to-end-optimized-image-compression.md) | [ADR-0003](decisions/adr-0003-value-per-byte-writer.md) | Compact/typed writer가 canonical JSONL byte 측정 | [EXP-0001](experiments/exp-0001-source-compact-baseline.md) local sanity; [EXP-0002](experiments/exp-0002-typed-memory-bridge.md), [EXP-0003](experiments/exp-0003-byte-pareto.md) 미실행 | 로컬 contract 검증 완료; benchmark 대기 |
| C-006 | Fast pose/tracking state와 persistent map memory를 분리해야 한다. | [Mem3R](papers/mem3r.md), [Hydra](papers/hydra.md), [Mono-Hydra++](papers/mono-hydra-plus-plus.md), [TTSA3R](papers/ttsa3r.md), [CUT3R](papers/cut3r.md), [TTT3R](papers/ttt3r.md) | [ADR-0001](decisions/adr-0001-explicit-typed-memory.md), [ADR-0002](decisions/adr-0002-gcut3r-as-teacher.md), [ADR-0005](decisions/adr-0005-hybrid-on-device-compiler.md) | Provider state는 transient, typed record는 persistent; trusted native pose contract 구현 | [EXP-0002](experiments/exp-0002-typed-memory-bridge.md), [EXP-0004](experiments/exp-0004-gcut3r-provider.md), [EXP-0005](experiments/exp-0005-teacher-oracle-ceiling.md) 미실행 | 일부 명세 |
| C-007 | Wide-baseline association은 position, viewing ray, semantics, time, geometry compatibility를 결합해야 한다. | [Ray-Aware Pointer Memory](papers/ray-aware-pointer-memory.md), [Point3R](papers/point3r.md) | 채택된 learned-association ADR 없음 | Explicit ID 또는 one-to-one heuristic spatial association 사용 | Open-world learned-association experiment 없음 | 연구 |
| C-008 | Long-horizon evaluation은 하나의 causal frame inventory를 사용하고 disjoint evidence moment의 memory를 측정해야 한다. | [SuperMemory-VQA](papers/supermemory-vqa.md), [LongSpace](papers/longspace.md), [MV-ScanQA](papers/mv-scanqa.md), [Memory-Centric EQA](papers/memory-centric-embodied-question-answering.md), [VSI-Bench](papers/vsi-bench.md), [OpenEQA](papers/openeqa.md) | Evaluation contract에 포함 | `sensor_frames.py`, `preflight.py`, causal retrieval | [EXP-0003](experiments/exp-0003-byte-pareto.md) 공식 run 미시작 | 로컬 contract 검증 완료 |
| C-009 | Causal scope, provenance, uncertainty, completeness는 answer correctness 요구사항이다. | [SuperMemory-VQA](papers/supermemory-vqa.md), [OpenEQA](papers/openeqa.md) | [ADR-0004](decisions/adr-0004-deterministic-geometry-proof.md), [ADR-0006](decisions/adr-0006-evidence-bound-inferred-geometry.md) | Inferred evidence/confidence gate, proof hash, complete-index gate, lineage, finalization seal | [EXP-0001](experiments/exp-0001-source-compact-baseline.md) local sanity; [EXP-0002](experiments/exp-0002-typed-memory-bridge.md), [EXP-0005](experiments/exp-0005-teacher-oracle-ceiling.md) 미실행 | 로컬 contract 검증 완료 |
| C-010 | Fixed latent bottleneck은 decoder baseline으로 유용하지만 geometry sufficiency 입증 전 explicit geometry를 대체할 수 없다. | [TokenLearner](papers/tokenlearner.md), [Perceiver](papers/perceiver.md), [BLIP-2](papers/blip-2.md) | 채택된 latent-memory ADR 없음 | `spatial_train.py`는 supplied vector candidate head이며 raw encoder가 아님 | Equal-byte typed-versus-latent experiment 없음 | 보류 |
| C-011 | Geometry-aware novelty와 duplicate removal은 generic attention-only pruning보다 강한 baseline이다. | [Good Token Hunting](papers/good-token-hunting.md), [DART](papers/dart.md), [FEATHER](papers/feather.md), [Geometry-Aware Token Pruning](papers/geometry-aware-token-pruning.md), [VisionZip](papers/visionzip.md), [MeMix](papers/memix.md) | 별도 ADR 없이 writer baseline으로 취급 | Selector preparation에 geometry novelty/redundancy feature 존재 | [EXP-0003](experiments/exp-0003-byte-pareto.md) 미실행 | 비교 계획 |
| C-012 | Model/latent value quantization은 primary persistent-memory 방법이 아닌 후속 deployment optimization이다. | [FSQ](papers/finite-scalar-quantization.md), [VQ-VAE](papers/vq-vae.md), [QVGGT](papers/qvggt.md), [rate-distortion](papers/end-to-end-optimized-image-compression.md) | [ADR-0001](decisions/adr-0001-explicit-typed-memory.md)의 explicit-first 방향 | Learned codec/model PTQ 미배포 | Project experiment 없음 | 보류 |
| C-013 | Local wiring evidence와 remote loadability/training authorization을 분리하고 contract→architecture→consensus→checkpoint→retrieval→QA→terminal/report identity를 fail-closed로 전파해야 한다. | Project safety/claim-integrity requirement | [ADR-0007](decisions/adr-0007-model-boundary-contract-and-load-gate.md) | `model_contract.py`, `mock_dag.py`, `spatial_train.py`, `retrieval_types.py`, `qa_transformers.py`, `report.py`; remote control plane | Tiny/mock contract and byte-regression verification only; provider conformance/load/training 미실행 | Local implementation; remote approval 차단 |
| C-014 | A model-alignment candidate must preserve immutable production rows and compare trusted sealed stores with fixed operands before promotion. | Project compatibility and scientific-integrity requirement | Additive model-neutral v2 envelope; no promotion ADR | Opt-in `memory` binding, exact contract-digest validation, four fixed Recall@6 arms, render-only `submission=false` plan | Tiny local contract/evaluator/renderer tests only; no model or remote run | Local implementation; production unchanged |

## 근거 등급

| 등급 | 의미 |
|---|---|
| 논문 보고 | Primary paper가 명시한 dataset/condition에서 검증 |
| 프로젝트 추론 | Paper condition을 현재 문제로 transfer한 design hypothesis |
| 로컬 검증 완료 | Tiny fixture, contract, unit behavior만 검증 |
| Benchmark 검증 완료 | Pinned official data, model, config, run provenance로 재현 |
| 차단 | 필요한 구현 또는 승인된 compute artifact 없음 |

## 갱신 원칙

- 새 research evidence는 paper page와 관련 row를 갱신한다.
- Design 변경은 새 ADR을 만들거나 기존 ADR을 supersede한다.
- Measured result는 experiment page만 갱신한 뒤 이 표의 상태와 link를 바꾼다.
- Architecture page에는 transient metric을 저장하지 않는다.
- Current blocker는 [현재 상태](status.md)에서 관리한다.

[프로젝트 홈으로 돌아가기](README.md)
